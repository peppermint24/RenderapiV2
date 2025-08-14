# main.py
"""
Property Score API - FastAPI service for property risk assessment
Combines SES, Build Year, Flood Safety, and Humidity scores
"""
import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import asyncio
from functools import lru_cache
import hashlib

from fastapi import FastAPI, Depends, HTTPException, Header, Query, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
import numpy as np
from shapely import wkb
from shapely.geometry import Point
from shapely.strtree import STRtree
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables with defaults
API_KEY = os.getenv("API_KEY")
API_BEARER = os.getenv("API_BEARER")
if not API_KEY:
    raise ValueError("API_KEY environment variable is required")

DATA_DIR = os.getenv("DATA_DIR", "/data")
GEOMS_PATH = os.getenv("GEOMS_PATH", f"{DATA_DIR}/tracts_wkb.parquet")
SCORES_PATH = os.getenv("SCORES_PATH", f"{DATA_DIR}/tract_lookup.json")
HUMIDITY_PATH = os.getenv("HUMIDITY_PATH", f"{DATA_DIR}/zip_humidity_scores.csv")
CLAIMS_PATH = os.getenv("CLAIMS_PATH", f"{DATA_DIR}/claims_proportion.json")

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar")

# Weights for total score calculation (MoldGuide methodology)
W_SES = float(os.getenv("W_SES", "0.20"))          # Socioeconomic status
W_BUILD_YEAR = float(os.getenv("W_BUILD_YEAR", "0.20"))  # Building newness
W_FLOOD = float(os.getenv("W_FLOOD", "0.50"))      # Flood as top predictor
W_HUMIDITY = float(os.getenv("W_HUMIDITY", "0.10"))  # Humidity risk

# Initialize FastAPI app
app = FastAPI(
    title="Property Score API",
    version="1.0.0",
    docs_url=None,  # Disable public docs
    redoc_url=None,
    openapi_url=None
)

# Security
security = HTTPBearer(auto_error=False)

# Global data storage
class DataStore:
    def __init__(self):
        self.ses_tree = None
        self.ses_geoms = []
        self.ses_geoids = []
        self.ses_scores = {}
        self.humidity_data = {}
        self.flood_claims = {}
        self.last_reload = {}
    
    def clear_ses(self):
        self.ses_tree = None
        self.ses_geoms = []
        self.ses_geoids = []
        self.ses_scores = {}
    
    def clear_humidity(self):
        self.humidity_data = {}
    
    def clear_flood_claims(self):
        self.flood_claims = {}

data_store = DataStore()

# ====================
# Authentication
# ====================

def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> bool:
    """Verify API key from either X-API-Key header or Bearer token"""
    # Check X-API-Key header
    if x_api_key and compare_api_keys(x_api_key, API_KEY):
        return True
    if API_BEARER and x_api_key and compare_api_keys(x_api_key, API_BEARER):
        return True
    
    # Check Bearer token
    if authorization and authorization.credentials:
        if compare_api_keys(authorization.credentials, API_KEY):
            return True
        if API_BEARER and compare_api_keys(authorization.credentials, API_BEARER):
            return True
    
    raise HTTPException(status_code=401, detail="Invalid or missing API key")

def compare_api_keys(provided: str, expected: str) -> bool:
    """Constant-time comparison of API keys"""
    if len(provided) != len(expected):
        return False
    result = 0
    for x, y in zip(provided, expected):
        result |= ord(x) ^ ord(y)
    return result == 0

# ====================
# Data Loading Functions
# ====================

def load_ses_data():
    """Load SES parquet and scores JSON"""
    try:
        # Load parquet with tract geometries
        if os.path.exists(GEOMS_PATH):
            df = pd.read_parquet(GEOMS_PATH)
            logger.info(f"Loaded {len(df)} tract geometries from {GEOMS_PATH}")
            
            # Decode WKB to shapely geometries
            geoms = []
            geoids = []
            for _, row in df.iterrows():
                geom = wkb.loads(row['wkb'])
                geoms.append(geom)
                geoids.append(row['GEOID'])
            
            # Build STRtree for spatial queries
            data_store.ses_tree = STRtree(geoms)
            data_store.ses_geoms = geoms
            data_store.ses_geoids = geoids
            
            logger.info(f"Built STRtree with {len(geoms)} polygons")
        else:
            logger.warning(f"Geometry file not found: {GEOMS_PATH}")
        
        # Load SES scores
        if os.path.exists(SCORES_PATH):
            with open(SCORES_PATH, 'r') as f:
                data_store.ses_scores = json.load(f)
            logger.info(f"Loaded {len(data_store.ses_scores)} SES scores from {SCORES_PATH}")
        else:
            logger.warning(f"Scores file not found: {SCORES_PATH}")
        
        data_store.last_reload['ses'] = datetime.now().isoformat()
        return {
            "polygons_loaded": len(data_store.ses_geoms),
            "scores_loaded": len(data_store.ses_scores),
            "paths": {"geoms": GEOMS_PATH, "scores": SCORES_PATH}
        }
    except Exception as e:
        logger.error(f"Error loading SES data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load SES data: {str(e)}")

def load_humidity_data():
    """Load humidity CSV"""
    try:
        if os.path.exists(HUMIDITY_PATH):
            df = pd.read_csv(HUMIDITY_PATH)
            # Normalize ZIP to 5 digits
            df['zip'] = df['zip'].astype(str).str.zfill(5)
            data_store.humidity_data = dict(zip(df['zip'], df['humidity_risk_0_100']))
            logger.info(f"Loaded {len(data_store.humidity_data)} humidity scores from {HUMIDITY_PATH}")
        else:
            logger.warning(f"Humidity file not found: {HUMIDITY_PATH}")
        
        data_store.last_reload['humidity'] = datetime.now().isoformat()
        return {
            "rows_loaded": len(data_store.humidity_data),
            "path": HUMIDITY_PATH
        }
    except Exception as e:
        logger.error(f"Error loading humidity data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load humidity data: {str(e)}")

def load_flood_claims():
    """Load flood claims proportion data"""
    try:
        if os.path.exists(CLAIMS_PATH):
            if CLAIMS_PATH.endswith('.json'):
                with open(CLAIMS_PATH, 'r') as f:
                    data_store.flood_claims = json.load(f)
            else:  # CSV
                df = pd.read_csv(CLAIMS_PATH)
                df['zip'] = df['zip'].astype(str).str.zfill(5)
                data_store.flood_claims = dict(zip(df['zip'], df['proportion']))
            
            logger.info(f"Loaded {len(data_store.flood_claims)} flood claims from {CLAIMS_PATH}")
        else:
            logger.warning(f"Claims file not found: {CLAIMS_PATH}")
        
        data_store.last_reload['flood_claims'] = datetime.now().isoformat()
        return {
            "rows_loaded": len(data_store.flood_claims),
            "path": CLAIMS_PATH
        }
    except Exception as e:
        logger.error(f"Error loading flood claims: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load flood claims: {str(e)}")

# ====================
# SES Functions
# ====================

def point_in_polygon_lookup(lat: float, lon: float) -> Tuple[Optional[str], Optional[float]]:
    """Find census tract GEOID for a point using STRtree"""
    if not data_store.ses_tree:
        return None, None
    
    point = Point(lon, lat)
    
    # Query the STRtree
    result = data_store.ses_tree.query(point, predicate='covers')
    
    # Handle both return types (indices or geometry objects)
    if isinstance(result, np.ndarray):  # Indices
        indices = result
    else:  # Geometry objects
        # Map geometry objects back to indices
        geom_to_idx = {id(g): i for i, g in enumerate(data_store.ses_geoms)}
        indices = [geom_to_idx.get(id(g)) for g in result if id(g) in geom_to_idx]
    
    # Find the first polygon that contains the point
    for idx in indices:
        if idx is not None and data_store.ses_geoms[idx].covers(point):
            geoid = data_store.ses_geoids[idx]
            score = data_store.ses_scores.get(geoid)
            return geoid, score
    
    return None, None

# ====================
# External API Functions
# ====================

async def query_fema_nfhl(lat: float, lon: float) -> Dict[str, Any]:
    """Query FEMA NFHL for flood zone information"""
    url = "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28/query"
    params = {
        "f": "json",
        "returnGeometry": "false",
        "spatialRel": "esriSpatialRelWithin",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,DEPTH"
    }
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        for attempt in range(3):
            try:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("features"):
                        attrs = data["features"][0].get("attributes", {})
                        return {
                            "zone": attrs.get("FLD_ZONE"),
                            "zone_subtype": attrs.get("ZONE_SUBTY"),
                            "sfha": attrs.get("SFHA_TF") == "T",
                            "depth_ft": attrs.get("DEPTH"),
                            "floodway": attrs.get("ZONE_SUBTY") == "FLOODWAY"
                        }
            except Exception as e:
                logger.warning(f"NFHL query attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    break
    
    return {}

def calculate_nfhl_score(nfhl_data: Dict[str, Any]) -> float:
    """Convert NFHL zone to 0-1 score (higher = safer)"""
    if not nfhl_data:
        return 0.5
    
    zone = nfhl_data.get("zone")
    depth = nfhl_data.get("depth_ft", 0) or 0
    floodway = nfhl_data.get("floodway", False)
    
    if floodway:
        return 0.2
    elif zone in ["AE", "A", "VE"]:
        if depth > 3:
            return 0.2
        return 0.4
    elif zone in ["X", "SHADED"]:  # Moderate risk
        return 0.5
    elif zone in ["X", "B", "C"]:  # Low risk
        return 0.8
    else:
        return 0.5

async def query_perplexity(hotel_name: str, address: str) -> Dict[str, Any]:
    """Query Perplexity AI for building year"""
    if not PERPLEXITY_API_KEY:
        return {
            "year": None,
            "source": "perplexity",
            "confidence": "low",
            "error": "No API key configured"
        }
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Extract the exact construction year for {hotel_name} at {address}.
    Return JSON: {{"year": int or null, "source": str, "confidence": "high/medium/low"}}"""
    
    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                
                # Try to parse JSON response
                try:
                    result = json.loads(content)
                    return {
                        "year": result.get("year"),
                        "source": "perplexity",
                        "confidence": result.get("confidence", "medium")
                    }
                except json.JSONDecodeError:
                    # Fallback: regex extraction
                    import re
                    year_match = re.search(r'\b(19|20)\d{2}\b', content)
                    if year_match:
                        return {
                            "year": int(year_match.group()),
                            "source": "perplexity",
                            "confidence": "low"
                        }
    except Exception as e:
        logger.error(f"Perplexity query failed: {e}")
    
    return {
        "year": None,
        "source": "perplexity",
        "confidence": "low"
    }

def year_to_score(year: Optional[int]) -> Optional[float]:
    """Convert build year to 0-100 score"""
    if not year:
        return None
    
    current_year = datetime.now().year
    if year >= current_year - 3:  # 2022+
        return 100.0
    elif year >= current_year - 9:  # 2016-2021
        return 85.0
    elif year >= current_year - 20:  # 2005-2015
        return 65.0
    elif year >= 1986:  # 1986-2004
        return 40.0
    else:
        return 20.0

# ====================
# Request/Response Models
# ====================

class BuildYearRequest(BaseModel):
    hotel_name: str = Field(..., description="Hotel name")
    address: str = Field(..., description="Hotel address")

class ReloadResponse(BaseModel):
    success: bool
    message: str
    details: Dict[str, Any]

# ====================
# API Endpoints
# ====================

@app.get("/healthz")
async def health_check():
    """Public health check endpoint"""
    return {"ok": True}

@app.post("/reload/ses", dependencies=[Depends(verify_api_key)])
async def reload_ses():
    """Reload SES data"""
    data_store.clear_ses()
    details = load_ses_data()
    return ReloadResponse(
        success=True,
        message="SES data reloaded successfully",
        details=details
    )

@app.post("/reload/humidity", dependencies=[Depends(verify_api_key)])
async def reload_humidity():
    """Reload humidity data"""
    data_store.clear_humidity()
    details = load_humidity_data()
    return ReloadResponse(
        success=True,
        message="Humidity data reloaded successfully",
        details=details
    )

@app.post("/reload/flood_claims", dependencies=[Depends(verify_api_key)])
async def reload_flood_claims():
    """Reload flood claims data"""
    data_store.clear_flood_claims()
    details = load_flood_claims()
    return ReloadResponse(
        success=True,
        message="Flood claims data reloaded successfully",
        details=details
    )

@app.get("/ses/score", dependencies=[Depends(verify_api_key)])
async def get_ses_score(lat: float = Query(...), lon: float = Query(...)):
    """Get SES score for a location"""
    geoid, score = point_in_polygon_lookup(lat, lon)
    return {
        "geoid": geoid,
        "ses_score": score,
        "source": "ses:tracts_wkb.parquet + tract_lookup.json"
    }

@app.post("/build_year", dependencies=[Depends(verify_api_key)])
async def get_build_year(request: BuildYearRequest):
    """Get build year from Perplexity"""
    result = await query_perplexity(request.hotel_name, request.address)
    
    year = result.get("year")
    score = year_to_score(year)
    stars = (score / 100 * 5) if score else None
    
    return {
        "year": year,
        "source": result.get("source", "perplexity"),
        "confidence": result.get("confidence", "low"),
        "stars_5": round(stars, 1) if stars else None,
        "score_0_100": score
    }

@app.get("/flood", dependencies=[Depends(verify_api_key)])
async def get_flood_score(
    lat: float = Query(...),
    lon: float = Query(...),
    zip: Optional[str] = Query(None)
):
    """Get combined flood safety score"""
    # Query NFHL
    nfhl_data = await query_fema_nfhl(lat, lon)
    nfhl_score = calculate_nfhl_score(nfhl_data)
    
    # Get claims data
    claims_safety = 0.5  # Default
    claims_proportion = None
    zip5 = None
    
    if zip:
        zip5 = str(zip).zfill(5)
        claims_proportion = data_store.flood_claims.get(zip5)
        if claims_proportion is not None:
            claims_safety = 1 - claims_proportion
    
    # Combine scores
    flood_safety = (nfhl_score + claims_safety) / 2
    flood_safety_100 = round(flood_safety * 100, 1)
    flood_stars = round(flood_safety_100 / 100 * 5, 1)
    
    return {
        "nfhl": {
            "zone": nfhl_data.get("zone"),
            "zone_subtype": nfhl_data.get("zone_subtype"),
            "sfha": nfhl_data.get("sfha", False),
            "depth_ft": nfhl_data.get("depth_ft"),
            "floodway": nfhl_data.get("floodway", False),
            "nfhl_score_0_1": nfhl_score
        },
        "claims": {
            "zip5": zip5,
            "proportion_0_1": claims_proportion,
            "claims_safety_0_1": claims_safety if claims_proportion is not None else None
        },
        "flood_safety_0_100": flood_safety_100,
        "flood_safety_stars_5": flood_stars
    }

@app.get("/humidity", dependencies=[Depends(verify_api_key)])
async def get_humidity_score(zip: str = Query(...)):
    """Get humidity risk score for a ZIP code"""
    zip5 = str(zip).zfill(5)
    risk_score = data_store.humidity_data.get(zip5)
    
    if risk_score is not None:
        stars = round((risk_score / 100) * 5, 1)
        return {
            "zip": zip5,
            "humidity_risk_0_100": risk_score,
            "humidity_risk_stars_5": stars
        }
    else:
        return {
            "zip": zip5,
            "humidity_risk_0_100": None,
            "humidity_risk_stars_5": None
        }

@app.get("/score/total", dependencies=[Depends(verify_api_key)])
async def get_total_score(
    lat: float = Query(...),
    lon: float = Query(...),
    zip: Optional[str] = Query(None),
    hotel_name: Optional[str] = Query(None),
    address: Optional[str] = Query(None)
):
    """Get combined total score from all components"""
    
    # Collect all component scores
    components = {}
    weights_used = {}
    
    # SES Score
    geoid, ses_score = point_in_polygon_lookup(lat, lon)
    if ses_score is not None:
        components["ses"] = {
            "geoid": geoid,
            "score_0_100": ses_score
        }
        weights_used["ses"] = W_SES
    
    # Build Year Score
    if hotel_name and address:
        build_result = await query_perplexity(hotel_name, address)
        year = build_result.get("year")
        build_score = year_to_score(year)
        if build_score is not None:
            components["build_year"] = {
                "year": year,
                "score_0_100": build_score,
                "confidence": build_result.get("confidence", "low"),
                "source": "perplexity"
            }
            weights_used["build_year"] = W_BUILD_YEAR
    
    # Flood Safety Score
    flood_result = await get_flood_score(lat, lon, zip)
    flood_safety_100 = flood_result.get("flood_safety_0_100")
    if flood_safety_100 is not None:
        components["flood_safety"] = {
            "score_0_100": flood_safety_100,
            "details": {
                "nfhl": flood_result["nfhl"],
                "claims": flood_result["claims"]
            }
        }
        weights_used["flood"] = W_FLOOD
    
    # Humidity Safety Score
    if zip:
        zip5 = str(zip).zfill(5)
        humidity_risk = data_store.humidity_data.get(zip5)
        if humidity_risk is not None:
            humidity_safety = 100 - humidity_risk
            components["humidity_safety"] = {
                "score_0_100": humidity_safety,
                "raw_humidity_risk_0_100": humidity_risk
            }
            weights_used["humidity"] = W_HUMIDITY
    
    # Calculate total score with renormalized weights
    if weights_used:
        total_weight = sum(weights_used.values())
        normalized_weights = {k: v/total_weight for k, v in weights_used.items()}
        
        total_score = 0
        for component_key, component_data in components.items():
            weight_key = component_key.replace("_safety", "").replace("_year", "")
            if weight_key in normalized_weights:
                total_score += component_data["score_0_100"] * normalized_weights[weight_key]
        
        total_score_100 = round(total_score, 1)
        total_stars = round(total_score_100 / 100 * 5, 1)
    else:
        total_score_100 = None
        total_stars = None
        normalized_weights = {}
    
    return {
        "inputs": {
            "lat": lat,
            "lon": lon,
            "zip": zip,
            "hotel_name": hotel_name,
            "address": address
        },
        "components": components,
        "weights_applied": normalized_weights,
        "total_score_0_100": total_score_100,
        "total_stars_5": total_stars
    }

# ====================
# Startup
# ====================

@app.on_event("startup")
async def startup_event():
    """Load data on startup if available"""
    logger.info("Starting Property Score API...")
    
    # Try to load data if files exist
    try:
        if os.path.exists(GEOMS_PATH) and os.path.exists(SCORES_PATH):
            load_ses_data()
    except Exception as e:
        logger.warning(f"Could not load SES data on startup: {e}")
    
    try:
        if os.path.exists(HUMIDITY_PATH):
            load_humidity_data()
    except Exception as e:
        logger.warning(f"Could not load humidity data on startup: {e}")
    
    try:
        if os.path.exists(CLAIMS_PATH):
            load_flood_claims()
    except Exception as e:
        logger.warning(f"Could not load flood claims on startup: {e}")
    
    logger.info("Property Score API started successfully")

# ====================
# Main
# ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "development") == "development"
    )