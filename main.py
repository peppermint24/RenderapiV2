# ===========================
# PRODUCTION FASTAPI APP - main.py
# File-based caching with area JSON files on mounted disk
# ===========================

import os
import json
import time
import requests
import re
import hashlib
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import uuid
import fcntl
import glob

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# CONFIGURATION
# ===========================

# Environment variables
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "").strip()
MAX_LOCATIONS_TO_SCAN = int(os.getenv("MAX_LOCATIONS_TO_SCAN", "400"))
DATA_EXPIRY_MONTHS = int(os.getenv("DATA_EXPIRY_MONTHS", "3"))
AREA_RESCAN_DAYS = int(os.getenv("AREA_RESCAN_DAYS", "30"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "risk-v1")
SCORING_VERSION = os.getenv("SCORING_VERSION", "2025-08-29")

# Data paths - Your Render persistent disk
DATA = Path("/data")
CACHE_DATA = Path("/data/cache")
AREAS_DATA = Path("/data/areas")

# Ensure cache directories exist
CACHE_DATA.mkdir(parents=True, exist_ok=True)
AREAS_DATA.mkdir(parents=True, exist_ok=True)

FILES = {
    "SES_TRACT": DATA / "Tract Lookup Fixed.json",
    "SES_ZIP": DATA / "Zip Fallback SES Aug 29.json",
    "FLOOD_ZIP": DATA / "flood_zip_floodprop.csv",
    "FLOOD_CNTY": DATA / "County Flood Fallback.csv",
    "DRY_ZIP": DATA / "Humidity Dryness Scores Aug 29.csv",
    "DRY_CNTY": DATA / "County Dryness Data Aug 29.csv",
    "ZIP_COUNTY": DATA / "ZIP_COUNTY_062025.xlsx",
    "TRACTS_GPKG": DATA / "us_all_tracts_2022.gpkg",
}

# Verify data directory on startup
if not DATA.exists():
    logger.error(f"Data directory {DATA} does not exist!")
    logger.info("Available directories:", [p.name for p in Path("/").iterdir() if p.is_dir()])
else:
    logger.info(f"Data directory found at {DATA}")
    logger.info(f"Data files: {[f.name for f in DATA.iterdir() if f.is_file()]}")

# ===========================
# FILE-BASED CACHE HELPERS
# ===========================

def area_key_to_filename(area_key: str) -> str:
    """Convert area key to safe filename"""
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', area_key.lower())
    return f"{safe_name}.json"

def load_json_with_lock(file_path: Path) -> Dict:
    """Load JSON file with file locking"""
    if not file_path.exists():
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
            data = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Unlock
            return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return {}

def save_json_with_lock(file_path: Path, data: Dict):
    """Save JSON file with file locking"""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
            json.dump(data, f, indent=2, default=str)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Unlock
    except Exception as e:
        logger.error(f"Error saving {file_path}: {e}")

def get_area_cache_file(area_key: str) -> Path:
    """Get cache file path for area"""
    return CACHE_DATA / area_key_to_filename(area_key)

def get_area_completeness_file() -> Path:
    """Get area completeness tracking file"""
    return AREAS_DATA / "completeness.json"

def cleanup_expired_places():
    """Remove expired place records from all area cache files"""
    expiry_threshold = datetime.now(timezone.utc) - timedelta(days=30 * DATA_EXPIRY_MONTHS)
    cleaned_count = 0
    
    for cache_file in CACHE_DATA.glob("*.json"):
        try:
            data = load_json_with_lock(cache_file)
            if not data or "results" not in data:
                continue
                
            original_count = len(data["results"])
            
            # Filter out expired places
            data["results"] = [
                place for place in data["results"]
                if place.get("metadata", {}).get("scored_at") and
                datetime.fromisoformat(place["metadata"]["scored_at"].replace('Z', '+00:00')) > expiry_threshold
            ]
            
            expired_count = original_count - len(data["results"])
            if expired_count > 0:
                data["last_cleanup"] = datetime.now(timezone.utc).isoformat()
                save_json_with_lock(cache_file, data)
                cleaned_count += expired_count
                logger.info(f"Cleaned {expired_count} expired places from {cache_file.name}")
                
        except Exception as e:
            logger.error(f"Error cleaning {cache_file}: {e}")
    
    return cleaned_count

# ===========================
# DATA LOADING
# ===========================

def load_json(path: Path):
    if not path.exists(): return {}
    try:
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return {}

def norm_zip(z):
    if z is None: return None
    s = "".join(ch for ch in str(z) if ch.isdigit())[:5]
    return s.zfill(5) if s else None

# Load all static data files
logger.info("Loading static data files...")
ses_tract = load_json(FILES["SES_TRACT"])
ses_zip = load_json(FILES["SES_ZIP"])

flood_zip = pd.read_csv(FILES["FLOOD_ZIP"], dtype={"zip_code": str}) if FILES["FLOOD_ZIP"].exists() else pd.DataFrame()
if not flood_zip.empty:
    flood_zip["zip_code"] = flood_zip["zip_code"].map(norm_zip)

flood_cnty = pd.read_csv(FILES["FLOOD_CNTY"], dtype={"county_fips": str}) if FILES["FLOOD_CNTY"].exists() else pd.DataFrame()

dry_zip = pd.read_csv(FILES["DRY_ZIP"], dtype={"zip": str}) if FILES["DRY_ZIP"].exists() else pd.DataFrame()
if not dry_zip.empty:
    dry_zip["zip"] = dry_zip["zip"].map(norm_zip)

dry_cnty = pd.read_csv(FILES["DRY_CNTY"], dtype={"county_fips": str}) if FILES["DRY_CNTY"].exists() else pd.DataFrame()

zip_county = pd.read_excel(FILES["ZIP_COUNTY"], dtype=str) if FILES["ZIP_COUNTY"].exists() else pd.DataFrame()
if not zip_county.empty:
    zip_county = zip_county.rename(columns={"ZIP": "zip", "COUNTY": "county_fips"})
    zip_county["zip"] = zip_county["zip"].map(norm_zip)
    if "TOT_RATIO" in zip_county:
        zip_county["TOT_RATIO"] = pd.to_numeric(zip_county["TOT_RATIO"], errors="coerce").fillna(0.0)

# Optional tracts geodata
tracts_gdf = None
if FILES["TRACTS_GPKG"].exists():
    try:
        tracts_gdf = gpd.read_file(FILES["TRACTS_GPKG"])
        if tracts_gdf is not None and tracts_gdf.crs is None:
            tracts_gdf.set_crs("EPSG:4269", inplace=True)
        logger.info(f"Loaded {len(tracts_gdf)} tracts")
    except Exception as e:
        logger.warning(f"Could not load tracts: {e}")

logger.info(f"Data loaded - SES_TRACT: {len(ses_tract)}, SES_ZIP: {len(ses_zip)}, FLOOD_ZIP: {len(flood_zip)}")

# ===========================
# GOOGLE API HELPERS
# ===========================

def http_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    out = {"http_status": None, "google_status": None, "error_message": None, "json": None}
    try:
        r = requests.get(url, params=params, timeout=30)
        out["http_status"] = r.status_code
        try:
            data = r.json()
        except Exception:
            data = {"_non_json_body": r.text[:400]}
        out["json"] = data
        out["google_status"] = data.get("status")
        out["error_message"] = data.get("error_message")
    except Exception as e:
        out["error_message"] = f"{type(e).__name__}: {e}"
    return out

def is_hotel_or_motel(place_types: List[str]) -> bool:
    """Filter to only hotels and motels"""
    if not place_types:
        return False
    types_str = " ".join(place_types).lower()
    return "hotel" in types_str or "motel" in types_str

def google_text_search(query: str, max_results: int = 400) -> List[Dict]:
    """Search for hotels/motels with pagination"""
    if not GOOGLE_PLACES_API_KEY:
        return []
    
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": GOOGLE_PLACES_API_KEY, "type": "lodging"}
    
    results = []
    page = 0
    
    while len(results) < max_results:
        page += 1
        resp = http_get(url, params)
        
        if resp["google_status"] != "OK":
            logger.error(f"Google search failed: {resp}")
            break
            
        data = resp["json"]
        page_results = data.get("results", [])
        
        # Filter to hotels/motels only
        filtered_results = [r for r in page_results if is_hotel_or_motel(r.get("types", []))]
        results.extend(filtered_results)
        
        # Check for next page
        next_token = data.get("next_page_token")
        if not next_token or len(page_results) == 0:
            break
            
        time.sleep(2)  # Required delay for next_page_token
        params = {"pagetoken": next_token, "key": GOOGLE_PLACES_API_KEY}
    
    return results[:max_results]

def extract_zip_from_components(components: list) -> Optional[str]:
    """Extract ZIP from Google address components (PRODUCTION: address_components only)"""
    if not components:
        return None
        
    for comp in components:
        types = comp.get("types", [])
        if "postal_code" in types:
            val = comp.get("long_name") or comp.get("short_name")
            if val:
                s = "".join(ch for ch in str(val) if ch.isdigit())[:5]
                if s and len(s) >= 5:
                    return s.zfill(5)
    return None

def google_place_details(place_id: str) -> Tuple[Optional[Dict], str]:
    """Get place details with ZIP extraction"""
    if not GOOGLE_PLACES_API_KEY or not place_id:
        return None, "no_api_key"
    
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "address_component,formatted_address,geometry,name,vicinity,types,rating,user_ratings_total",
        "key": GOOGLE_PLACES_API_KEY
    }
    
    resp = http_get(url, params)
    if resp["google_status"] != "OK":
        return None, f"api_error_{resp['google_status']}"
    
    result = resp["json"].get("result")
    if not result:
        return None, "no_result"
    
    # Extract ZIP from address components (PRODUCTION: no formatted address parsing)
    components = result.get("address_components", [])
    zip_code = extract_zip_from_components(components)
    
    return {
        "place_id": place_id,
        "name": result.get("name"),
        "address": result.get("formatted_address"),
        "geometry": result.get("geometry", {}).get("location", {}),
        "types": result.get("types", []),
        "rating": result.get("rating"),
        "user_ratings_total": result.get("user_ratings_total"),
        "zip": zip_code,
        "zip_source": "place_details" if zip_code else None
    }, "success"

def reverse_geocode_for_zip(lat: float, lng: float) -> Optional[str]:
    """Fallback ZIP extraction via reverse geocoding"""
    if not GOOGLE_PLACES_API_KEY:
        return None
    
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"latlng": f"{lat},{lng}", "key": GOOGLE_PLACES_API_KEY}
    
    resp = http_get(url, params)
    if resp["google_status"] != "OK":
        return None
    
    results = resp["json"].get("results", [])
    for result in results[:3]:  # Check first 3 results
        zip_code = extract_zip_from_components(result.get("address_components", []))
        if zip_code:
            return zip_code
    
    return None

# ===========================
# SCORING FUNCTIONS
# ===========================

def tract_from_latlng(lat: float, lng: float) -> Optional[str]:
    """Get census tract from coordinates"""
    if tracts_gdf is None or pd.isna(lat) or pd.isna(lng):
        return None
    try:
        pt = gpd.GeoDataFrame(geometry=[Point(float(lng), float(lat))], crs="EPSG:4326")
        tg = tracts_gdf.to_crs("EPSG:4326") if str(tracts_gdf.crs) != "EPSG:4326" else tracts_gdf
        hit = gpd.sjoin(pt, tg[["GEOID", "geometry"]], how="left", predicate="within")
        geoid = hit.iloc[0]["GEOID"]
        return str(geoid) if pd.notna(geoid) else None
    except Exception:
        return None

def zip_to_county(zip5: str) -> Optional[str]:
    """Get primary county for ZIP code"""
    if zip_county.empty or not zip5:
        return None
    z = norm_zip(zip5)
    matches = zip_county[zip_county["zip"] == z]
    if matches.empty:
        return None
    
    # Return county with highest ratio
    if "TOT_RATIO" in matches.columns:
        best = matches.loc[matches["TOT_RATIO"].idxmax()]
        return best["county_fips"]
    else:
        return matches.iloc[0]["county_fips"]

def score_affluence(tract: Optional[str], zip5: Optional[str]) -> Dict:
    """SES scoring: tract -> zip (no county fallback)"""
    if tract and tract in ses_tract and ses_tract[tract] is not None:
        score = float(ses_tract[tract])
        return {"score": score, "stars": round(score/20.0, 1), "basis": "tract", "has_data": True}
    
    z = norm_zip(zip5)
    if z and z in ses_zip and ses_zip[z] is not None:
        score = float(ses_zip[z])
        return {"score": score, "stars": round(score/20.0, 1), "basis": "zip", "has_data": True}
    
    return {"score": None, "stars": None, "basis": None, "has_data": False}

def score_newness(build_year: Optional[int]) -> Dict:
    """Newness scoring from build year"""
    if build_year is None:
        return {"score": None, "stars": None, "age_years": None, "has_data": False}
    
    try:
        year = int(build_year)
        current_year = datetime.now().year
        age = max(0, current_year - year)
        
        if age <= 3: stars = 5.0
        elif age <= 9: stars = 4.0
        elif age <= 20: stars = 3.0
        elif age <= 39: stars = 2.0
        else: stars = 1.0
        
        score = stars * 20.0
        return {"score": score, "stars": stars, "age_years": age, "has_data": True}
    except (TypeError, ValueError):
        return {"score": None, "stars": None, "age_years": None, "has_data": False}

def score_flood(zip5: Optional[str], county_fips: Optional[str]) -> Dict:
    """Flood scoring: ZIP -> county fallback (only if ZIP row missing)"""
    z = norm_zip(zip5)
    
    # Try ZIP first
    if z and not flood_zip.empty:
        matches = flood_zip[flood_zip["zip_code"] == z]
        if not matches.empty and pd.notna(matches.iloc[0]["flood_proportion_raw"]):
            prop = float(matches.iloc[0]["flood_proportion_raw"])
            score = max(0.0, min(100.0, (1.0 - prop) * 100.0))
            return {"score": score, "stars": round(score/20.0, 1), "basis": "zip", "has_data": True}
    
    # County fallback only if ZIP row missing
    if county_fips and not flood_cnty.empty:
        matches = flood_cnty[flood_cnty["county_fips"] == str(county_fips)]
        if not matches.empty and pd.notna(matches.iloc[0]["avg_flood_proportion"]):
            prop = float(matches.iloc[0]["avg_flood_proportion"])
            score = max(0.0, min(100.0, (1.0 - prop) * 100.0))
            return {"score": score, "stars": round(score/20.0, 1), "basis": "county", "has_data": True}
    
    return {"score": None, "stars": None, "basis": None, "has_data": False}

def score_dryness(zip5: Optional[str], county_fips: Optional[str]) -> Dict:
    """Dryness scoring: ZIP -> county fallback (only if ZIP row missing)"""
    z = norm_zip(zip5)
    
    # Try ZIP first
    if z and not dry_zip.empty:
        matches = dry_zip[dry_zip["zip"] == z]
        if not matches.empty and pd.notna(matches.iloc[0]["dryness_safety_0_100"]):
            score = float(matches.iloc[0]["dryness_safety_0_100"])
            return {"score": score, "stars": round(score/20.0, 1), "basis": "zip", "has_data": True}
    
    # County fallback only if ZIP row missing
    if county_fips and not dry_cnty.empty:
        matches = dry_cnty[dry_cnty["county_fips"] == str(county_fips)]
        if not matches.empty and pd.notna(matches.iloc[0]["avg_dryness_safety_0_100"]):
            score = float(matches.iloc[0]["avg_dryness_safety_0_100"])
            return {"score": score, "stars": round(score/20.0, 1), "basis": "county", "has_data": True}
    
    return {"score": None, "stars": None, "basis": None, "has_data": False}

def score_reviews(rating: Optional[float], count: Optional[int]) -> Dict:
    """Reviews scoring with Bayesian smoothing"""
    if rating is None or count is None:
        return {"score": None, "stars": None, "has_data": False}
    
    try:
        r = float(rating)
        n = int(count)
        
        # Bayesian smoothing
        prior = 3.9
        weight = min(1.0, n/100.0)
        adjusted = prior * (1 - weight) + r * weight
        
        # Linear mapping to 0-100
        score = max(0.0, min(100.0, (adjusted - 1.0) / (5.0 - 1.0) * 100.0))
        
        # Penalty for poor ratings with many reviews
        if r <= 3.0 and n >= 100:
            score = max(0.0, score - 3.0)
        
        return {"score": score, "stars": round(score/20.0, 1), "has_data": True}
    except (TypeError, ValueError):
        return {"score": None, "stars": None, "has_data": False}

def calculate_overall_score(affluence: Dict, newness: Dict, flood: Dict, dryness: Dict, reviews: Dict) -> Dict:
    """Calculate overall score with weighted combination"""
    core_weights = {"flood": 0.35, "newness": 0.25, "affluence": 0.20, "dryness": 0.05}
    components = {"affluence": affluence, "newness": newness, "flood": flood, "dryness": dryness}
    
    # Check which core components have data
    core_scores = {k: v["score"] for k, v in components.items() if v.get("has_data") and v.get("score") is not None}
    
    if len(core_scores) < 3:
        return {"score": None, "stars": None, "has_data": False, "reason": "not_enough_core_components"}
    
    # Normalize weights for present components
    present_weights = {k: w for k, w in core_weights.items() if k in core_scores}
    weight_sum = sum(present_weights.values())
    
    # Adjust for reviews
    reviews_present = reviews.get("has_data") and reviews.get("score") is not None
    core_bucket = 0.85 if reviews_present else 1.0
    normalized_weights = {k: (w / weight_sum) * core_bucket for k, w in present_weights.items()}
    
    # Calculate weighted score
    overall = sum(core_scores[k] * normalized_weights[k] for k in normalized_weights)
    
    # Add reviews component
    if reviews_present:
        overall += reviews["score"] * 0.15
    
    # Apply guardrail cap
    core_min = min(core_scores.values())
    if core_min < 40.0:
        overall = min(overall, core_min)
    
    return {"score": round(overall, 1), "stars": round(overall/20.0, 1), "has_data": True}

def score_place(place_data: Dict) -> Dict:
    """Score a single place with all components"""
    lat = place_data.get("lat")
    lng = place_data.get("lng")
    zip_code = place_data.get("zip")
    build_year = place_data.get("build_year")
    rating = place_data.get("google_rating")
    review_count = place_data.get("google_review_count")
    
    # Get tract and county
    tract = tract_from_latlng(lat, lng) if lat and lng else None
    county = zip_to_county(zip_code) if zip_code else None
    
    # Calculate all scores
    affluence = score_affluence(tract, zip_code)
    newness = score_newness(build_year)
    flood = score_flood(zip_code, county)
    dryness = score_dryness(zip_code, county)
    reviews = score_reviews(rating, review_count)
    overall = calculate_overall_score(affluence, newness, flood, dryness, reviews)
    
    return {
        "tract_geoid": tract,
        "county_fips": county,
        "affluence": affluence,
        "newness": newness, 
        "flood": flood,
        "dryness": dryness,
        "reviews": reviews,
        "overall": overall
    }

# ===========================
# AREA CACHE MANAGEMENT
# ===========================

def get_cached_places_for_area(area_key: str) -> Tuple[List[Dict], Dict]:
    """Get cached places for an area, filtering out expired ones"""
    cache_file = get_area_cache_file(area_key)
    cache_data = load_json_with_lock(cache_file)
    
    if not cache_data or "results" not in cache_data:
        return [], {"from_cache": 0, "expired_removed": 0}
    
    # Filter out expired places
    expiry_threshold = datetime.now(timezone.utc) - timedelta(days=30 * DATA_EXPIRY_MONTHS)
    original_count = len(cache_data["results"])
    
    valid_places = []
    for place in cache_data["results"]:
        scored_at_str = place.get("metadata", {}).get("scored_at")
        if scored_at_str:
            try:
                scored_at = datetime.fromisoformat(scored_at_str.replace('Z', '+00:00'))
                if scored_at > expiry_threshold:
                    valid_places.append(place)
            except ValueError:
                continue  # Skip places with invalid timestamps
    
    expired_count = original_count - len(valid_places)
    
    # Update cache file if we removed expired places
    if expired_count > 0:
        cache_data["results"] = valid_places
        cache_data["last_cleanup"] = datetime.now(timezone.utc).isoformat()
        save_json_with_lock(cache_file, cache_data)
    
    return valid_places, {"from_cache": len(valid_places), "expired_removed": expired_count}

def get_area_completeness(area_key: str) -> Dict:
    """Get area completeness info"""
    completeness_file = get_area_completeness_file()
    completeness_data = load_json_with_lock(completeness_file)
    
    area_info = completeness_data.get(area_key, {})
    return {
        "total_available": area_info.get("total_available", 0),
        "is_exhausted": area_info.get("is_exhausted", False),
        "last_full_scan": area_info.get("last_full_scan"),
        "scan_version": area_info.get("scan_version")
    }

def update_area_completeness(area_key: str, total_found: int, is_exhausted: bool):
    """Update area completeness tracking"""
    completeness_file = get_area_completeness_file()
    completeness_data = load_json_with_lock(completeness_file)
    
    completeness_data[area_key] = {
        "total_available": total_found,
        "is_exhausted": is_exhausted,
        "last_full_scan": datetime.now(timezone.utc).isoformat(),
        "scan_version": MODEL_VERSION
    }
    
    save_json_with_lock(completeness_file, completeness_data)

def save_area_cache(area_key: str, places: List[Dict], scan_metadata: Dict):
    """Save area scan results to cache file"""
    cache_file = get_area_cache_file(area_key)
    
    cache_data = {
        "area_key": area_key,
        "scan_metadata": scan_metadata,
        "total_places": len(places),
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "scoring_version": SCORING_VERSION,
        "results": places
    }
    
    save_json_with_lock(cache_file, cache_data)
    logger.info(f"Saved {len(places)} places to cache for {area_key}")

# ===========================
# PERPLEXITY ENRICHMENT
# ===========================

def fetch_build_year_with_perplexity(hotel_name: str, address: str) -> Optional[Dict]:
    """Get build year from Perplexity API"""
    if not PERPLEXITY_API_KEY:
        return None
    
    prompt = f"""Find the exact construction year for this hotel:
Hotel: {hotel_name}
Address: {address}

Return JSON only: {{"year": <4-digit-year>, "confidence": <0-100>}}"""
    
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            },
            timeout=45
        )
        response.raise_for_status()
        
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return json.loads(content.strip("`"))
    except Exception:
        return None

# ===========================
# FASTAPI APPLICATION
# ===========================

app = FastAPI(
    title="Hotel Safety Score API",
    description="File-based cache hotel safety scoring with area scanning",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AreaScanRequest(BaseModel):
    area_key: str
    max_locations: Optional[int] = Field(default=None)
    force_refresh: bool = False

@app.get("/")
def read_root():
    return {
        "message": "Hotel Safety Score API",
        "version": "1.0", 
        "storage": "file-based",
        "endpoints": ["/v1/area/scan-and-score", "/health", "/docs"]
    }

@app.get("/health")
def health_check():
    # Check data files
    data_files_count = sum(1 for f in FILES.values() if f.exists())
    
    # Check cache directory
    cache_files = len(list(CACHE_DATA.glob("*.json"))) if CACHE_DATA.exists() else 0
    
    return {
        "status": "healthy",
        "storage": "file-based-repo",
        "data_directory": str(DATA),
        "data_files": data_files_count,
        "cache_files": cache_files,
        "google_api": bool(GOOGLE_PLACES_API_KEY),
        "perplexity_api": bool(PERPLEXITY_API_KEY),
        "config": {
            "max_locations": MAX_LOCATIONS_TO_SCAN,
            "expiry_months": DATA_EXPIRY_MONTHS
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/v1/area/scan-and-score")
async def area_scan_and_score(request: AreaScanRequest):
    """Unified endpoint: scan area and return scored hotels/motels with file-based caching"""
    
    max_locations = request.max_locations or MAX_LOCATIONS_TO_SCAN
    
    # Parse city/state from area_key
    if request.area_key.startswith("city:"):
        location = request.area_key.replace("city:", "").strip()
        if "," in location:
            city, state = [s.strip() for s in location.split(",")]
        else:
            city, state = location, ""
    else:
        city, state = "", ""
    
    try:
        # 1. Check area completeness
        completeness = get_area_completeness(request.area_key)
        
        # 2. Get cached places (auto-removes expired)
        cached_places, cache_stats = get_cached_places_for_area(request.area_key)
        
        # 3. Determine if Google API call needed
        needs_google_scan = (
            request.force_refresh or
            not completeness["is_exhausted"] or
            len(cached_places) < min(max_locations, completeness["total_available"]) or
            (completeness["last_full_scan"] and 
             datetime.fromisoformat(completeness["last_full_scan"].replace('Z', '+00:00')) < 
             datetime.now(timezone.utc) - timedelta(days=AREA_RESCAN_DAYS))
        )
        
        new_places = []
        google_exhausted = False
        
        if needs_google_scan and GOOGLE_PLACES_API_KEY:
            # 4. Google Places search
            places_needed = max(0, max_locations - len(cached_places))
            query = f"hotels in {city}, {state}" if city else request.area_key.replace("city:", "")
            
            logger.info(f"Searching Google for {places_needed} hotels in {query}")
            google_results = google_text_search(query, places_needed)
            google_exhausted = len(google_results) < places_needed
            
            # 5. Process each place
            for place in google_results:
                try:
                    # Skip if already in cache
                    if any(cp["place_id"] == place["place_id"] for cp in cached_places):
                        continue
                    
                    # Get details
                    details, status = google_place_details(place["place_id"])
                    if status != "success" or not details:
                        continue
                    
                    # Try reverse geocoding if no ZIP
                    if not details["zip"] and details["geometry"]:
                        lat = details["geometry"].get("lat")
                        lng = details["geometry"].get("lng")
                        if lat and lng:
                            zip_from_reverse = reverse_geocode_for_zip(lat, lng)
                            if zip_from_reverse:
                                details["zip"] = zip_from_reverse
                                details["zip_source"] = "reverse_geocode"
                    
                    # Parse address for city/state
                    address_parts = (details["address"] or "").split(", ")
                    parsed_city = city or (address_parts[-3] if len(address_parts) >= 3 else "")
                    parsed_state = state or (address_parts[-2].split()[0] if len(address_parts) >= 2 else "")
                    
                    # Get build year (optional Perplexity enrichment)
                    build_year = None
                    if PERPLEXITY_API_KEY and details["name"] and details["address"]:
                        perplexity_result = fetch_build_year_with_perplexity(details["name"], details["address"])
                        if perplexity_result and perplexity_result.get("year"):
                            build_year = perplexity_result["year"]
                    
                    # Score the place
                    place_for_scoring = {
                        "lat": details["geometry"].get("lat"),
                        "lng": details["geometry"].get("lng"),
                        "zip": details["zip"],
                        "build_year": build_year,
                        "google_rating": details["rating"],
                        "google_review_count": details["user_ratings_total"]
                    }
                    
                    scores = score_place(place_for_scoring)
                    
                    # Format for UI response
                    place_result = {
                        "place_id": details["place_id"],
                        "name": details["name"],
                        "address": details["address"],
                        "lat": details["geometry"].get("lat"),
                        "lng": details["geometry"].get("lng"),
                        "zip": details["zip"],
                        "zip_source": details["zip_source"],
                        "primary_type": "hotel" if "hotel" in str(details["types"]).lower() else "motel",
                        "build_year": build_year,
                        "age_years": scores["newness"].get("age_years"),
                        "overall": {
                            "score": scores["overall"]["score"],
                            "stars": scores["overall"]["stars"]
                        },
                        "components": {
                            "newness": {
                                "score": scores["newness"]["score"],
                                "stars": scores["newness"]["stars"],
                                "basis": "calculated"
                            },
                            "affluence": {
                                "score": scores["affluence"]["score"],
                                "stars": scores["affluence"]["stars"],
                                "basis": scores["affluence"]["basis"],
                                "aqi": int(scores["affluence"]["score"]) if scores["affluence"]["score"] else None,
                                "aqi_label": "Good" if scores["affluence"]["score"] and scores["affluence"]["score"] >= 50 else "Poor"
                            },
                            "flood": {
                                "score": scores["flood"]["score"],
                                "stars": scores["flood"]["stars"],
                                "basis": scores["flood"]["basis"]
                            },
                            "dryness": {
                                "score": scores["dryness"]["score"],
                                "stars": scores["dryness"]["stars"],
                                "basis": scores["dryness"]["basis"],
                                "peak_humidity": "62%",  # Could calculate from raw data
                                "peak_dew_point": "55°F",
                                "humidity_label": "average"
                            }
                        },
                        "google": {
                            "rating": details["rating"],
                            "review_count": details["user_ratings_total"]
                        },
                        "metadata": {
                            "cached": False,
                            "scored_at": datetime.now(timezone.utc).isoformat(),
                            "expires_at": (datetime.now(timezone.utc) + timedelta(days=30 * DATA_EXPIRY_MONTHS)).isoformat(),
                            "tract_geoid": scores["tract_geoid"],
                            "county_fips": scores["county_fips"]
                        }
                    }
                    
                    new_places.append(place_result)
                    
                except Exception as e:
                    logger.error(f"Error processing place {place.get('place_id', 'unknown')}: {e}")
                    continue
        
        # 6. Update area completeness
        total_found = len(cached_places) + len(new_places)
        update_area_completeness(request.area_key, total_found, google_exhausted)
        
        # 7. Combine results and save to cache
        all_places = cached_places + new_places
        
        if new_places:  # Save cache if we have new data
            scan_metadata = {
                "scanned_at": datetime.now(timezone.utc).isoformat(),
                "google_exhausted": google_exhausted,
                "total_requested": max_locations,
                "newly_added": len(new_places)
            }
            save_area_cache(request.area_key, all_places, scan_metadata)
        
        # 8. Mark cached items in metadata
        for place in cached_places:
            if "metadata" in place:
                place["metadata"]["cached"] = True
        
        # 9. Return top results
        final_results = sorted(all_places, key=lambda x: x.get("overall", {}).get("score", 0), reverse=True)[:max_locations]
        
        return {
            "area_key": request.area_key,
            "total_found": len(all_places),
            "from_cache": cache_stats["from_cache"],
            "newly_scored": len(new_places),
            "expired_removed": cache_stats["expired_removed"],
            "cache_hit_rate": round(cache_stats["from_cache"] / max(1, len(all_places)) * 100, 1),
            "is_exhausted": google_exhausted,
            "results": final_results
        }
        
    except Exception as e:
        logger.error(f"Area scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/debug/cache")
def debug_cache():
    """Debug cache file status"""
    cache_files = list(CACHE_DATA.glob("*.json")) if CACHE_DATA.exists() else []
    completeness_file = get_area_completeness_file()
    
    cache_info = {}
    for cache_file in cache_files:
        try:
            data = load_json_with_lock(cache_file)
            cache_info[cache_file.stem] = {
                "file_size_kb": round(cache_file.stat().st_size / 1024, 1),
                "places_count": len(data.get("results", [])),
                "cached_at": data.get("cached_at"),
                "model_version": data.get("model_version")
            }
        except Exception as e:
            cache_info[cache_file.stem] = {"error": str(e)}
    
    completeness_data = load_json_with_lock(completeness_file) if completeness_file.exists() else {}
    
    return {
        "cache_directory": str(CACHE_DATA),
        "cache_files": cache_info,
        "area_completeness": completeness_data,
        "total_cache_files": len(cache_files)
    }

@app.post("/v1/admin/cleanup-expired")
def cleanup_expired():
    """Manual cleanup of expired places across all areas"""
    cleaned_count = cleanup_expired_places()
    return {"cleaned_places": cleaned_count, "cleaned_at": datetime.now(timezone.utc).isoformat()}

@app.get("/v1/debug/place")
def debug_place(place_id: Optional[str] = Query(None), lat: Optional[float] = Query(None), lng: Optional[float] = Query(None)):
    """Debug endpoint for place details and reverse geocoding"""
    out = {"place_id": place_id, "lat": lat, "lng": lng}

    if place_id:
        details, status = google_place_details(place_id)
        out["details_status"] = status
        out["details_result"] = details

    if lat is not None and lng is not None:
        zip_from_reverse = reverse_geocode_for_zip(lat, lng)
        out["reverse_zip"] = zip_from_reverse

    return out

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
