"""
Property Score API - FastAPI service for property risk assessment
Uses SES (by tract), FEMA NFHL (live), historical flood claims (by ZIP),
humidity risk (by ZIP), and hotel build year (Perplexity) to compute scores.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple

from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Header, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

import pandas as pd
import numpy as np
from shapely import wkb as shapely_wkb
from shapely.geometry import Point
from shapely.strtree import STRtree
import httpx

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("property-score-api")

# -----------------------------------------------------------------------------
# Environment / Config
# -----------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY")
API_BEARER = os.getenv("API_BEARER")  # optional second key
if not API_KEY:
    raise ValueError("API_KEY environment variable is required")

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")  # optional
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar")

# Default weights (you can override with env vars)
W_SES = float(os.getenv("W_SES", "0.20"))
W_BUILD_YEAR = float(os.getenv("W_BUILD_YEAR", "0.20"))
W_FLOOD = float(os.getenv("W_FLOOD", "0.50"))
W_HUMIDITY = float(os.getenv("W_HUMIDITY", "0.10"))

# Flexible file picking (handles your new names and the old ones)
def pick_path(env_name: str, candidates: List[str]) -> str:
    p = os.getenv(env_name)
    if p and os.path.exists(p):
        return p
    for c in candidates:
        if os.path.exists(c):
            return c
    # Return the first candidate (even if missing) so errors are obvious
    return candidates[0]

DATA_DIR = os.getenv("DATA_DIR", "/data")

GEOMS_PATH = pick_path("GEOMS_PATH", [
    os.path.join(DATA_DIR, "Tracts WKB.parquet"),
    os.path.join(DATA_DIR, "tracts_wkb.parquet"),
])

SCORES_PATH = pick_path("SCORES_PATH", [
    os.path.join(DATA_DIR, "Tract Lookup Fixed.json"),
    os.path.join(DATA_DIR, "tract_lookup.json"),
])

HUMIDITY_PATH = pick_path("HUMIDITY_PATH", [
    os.path.join(DATA_DIR, "Humidity Medians.parquet"),
    os.path.join(DATA_DIR, "zip_humidity_scores.csv"),
])

CLAIMS_PATH = pick_path("CLAIMS_PATH", [
    os.path.join(DATA_DIR, "Claims with Medians.parquet"),
    os.path.join(DATA_DIR, "claims_proportion.json"),
    os.path.join(DATA_DIR, "flood_zip_floodprop.csv"),
])

# -----------------------------------------------------------------------------
# App + Security
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Property Score API",
    version="1.0.0",
    docs_url=None,      # hide openapi
    redoc_url=None,
    openapi_url=None,
)
security = HTTPBearer(auto_error=False)

def compare_api_keys(provided: str, expected: Optional[str]) -> bool:
    if not expected:
        return False
    if len(provided) != len(expected):
        return False
    result = 0
    for x, y in zip(provided, expected):
        result |= ord(x) ^ ord(y)
    return result == 0

def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> bool:
    if x_api_key and compare_api_keys(x_api_key, API_KEY):
        return True
    if API_BEARER and x_api_key and compare_api_keys(x_api_key, API_BEARER):
        return True

    if authorization and authorization.credentials:
        token = authorization.credentials
        if compare_api_keys(token, API_KEY):
            return True
        if API_BEARER and compare_api_keys(token, API_BEARER):
            return True

    raise HTTPException(status_code=401, detail="Invalid or missing API key")

# -----------------------------------------------------------------------------
# Data Store
# -----------------------------------------------------------------------------
class DataStore:
    # SES
    ses_tree: Optional[STRtree] = None
    ses_geoms: List = []
    ses_geoids: List[str] = []
    ses_scores: Dict[str, float] = {}

    # Humidity by ZIP (+ optional medians embedded per row)
    humidity_df: Optional[pd.DataFrame] = None
    # Claims by ZIP (+ optional medians embedded per row)
    claims_df: Optional[pd.DataFrame] = None

    last_reload: Dict[str, str] = {}

data_store = DataStore()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _norm_zip(z: Optional[str]) -> Optional[str]:
    if z is None:
        return None
    return str(z).strip().zfill(5)

def _safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def _read_humidity() -> Dict[str, Any]:
    """
    Load either Parquet with medians or legacy CSV.
    Expects at least a 'zip' (or 'ZIP') column and a humidity value column:
    - 'humidity_risk_0_100' (preferred)
    Optional (for fallbacks per ZIP row):
    - 'humidity_median_county', 'humidity_median_state'
    """
    if not os.path.exists(HUMIDITY_PATH):
        return {"rows_loaded": 0, "path": HUMIDITY_PATH, "detail": "NOT FOUND"}

    if HUMIDITY_PATH.lower().endswith(".parquet"):
        df = pd.read_parquet(HUMIDITY_PATH)
    else:
        df = pd.read_csv(HUMIDITY_PATH)

    zcol = next((c for c in df.columns if c.lower() in {"zip","zip5","zipcode","zcta"}), None)
    vcol = next((c for c in df.columns if c.lower() in {"humidity_risk_0_100","humidity","humidity_risk"}), None)

    if not zcol or not vcol:
        raise HTTPException(status_code=500, detail=f"Humidity file missing required columns (zip/humidity). Columns found: {list(df.columns)}")

    df = df.copy()
    df["zip5"] = df[zcol].astype(str).str.strip().str.zfill(5)
    df["humidity_risk_0_100"] = pd.to_numeric(df[vcol], errors="coerce")

    # Optional medians per row
    cm = next((c for c in df.columns if c.lower() in {"humidity_median_county","county_median","median_county"}), None)
    sm = next((c for c in df.columns if c.lower() in {"humidity_median_state","state_median","median_state"}), None)
    if cm: df["humidity_median_county"] = pd.to_numeric(df[cm], errors="coerce")
    else:  df["humidity_median_county"] = np.nan
    if sm: df["humidity_median_state"] = pd.to_numeric(df[sm], errors="coerce")
    else:  df["humidity_median_state"] = np.nan

    data_store.humidity_df = df.set_index("zip5")
    data_store.last_reload["humidity"] = datetime.utcnow().isoformat()
    return {"rows_loaded": int(len(df)), "path": HUMIDITY_PATH}

def _read_claims() -> Dict[str, Any]:
    """
    Load claims proportion (0..1) by ZIP, with optional median columns.
    Accepts Parquet (preferred) or JSON/CSV legacy variants.

    Expected columns (any of the candidates):
    - zip column: 'zip','zip5','zipcode'
    - proportion column: 'proportion','flood_proportion_raw','value'
    Optional (per-row fallbacks):
    - county_median / state_median
    """
    if not os.path.exists(CLAIMS_PATH):
        return {"rows_loaded": 0, "path": CLAIMS_PATH, "detail": "NOT FOUND"}

    if CLAIMS_PATH.lower().endswith(".parquet"):
        df = pd.read_parquet(CLAIMS_PATH)
    elif CLAIMS_PATH.lower().endswith(".json"):
        with open(CLAIMS_PATH, "r") as f:
            raw = json.load(f)
        # JSON expected as {zip: proportion}
        keys = list(raw.keys())
        df = pd.DataFrame({"zip5": [str(k).zfill(5) for k in keys],
                           "proportion": [raw[k] for k in keys]})
    else:
        df = pd.read_csv(CLAIMS_PATH)

    zcol = next((c for c in df.columns if c.lower() in {"zip","zip5","zipcode","zcta"}), None)
    vcol = next((c for c in df.columns if c.lower() in {"proportion","flood_proportion_raw","value"}), None)

    if not zcol or not vcol:
        raise HTTPException(status_code=500, detail=f"Claims file missing required columns (zip/proportion). Columns found: {list(df.columns)}")

    df = df.copy()
    df["zip5"] = df[zcol].astype(str).str.strip().str.zfill(5)
    df["proportion"] = pd.to_numeric(df[vcol], errors="coerce")

    cm = next((c for c in df.columns if c.lower() in {"county_median","median_county","claims_median_county"}), None)
    sm = next((c for c in df.columns if c.lower() in {"state_median","median_state","claims_median_state"}), None)
    if cm: df["county_median"] = pd.to_numeric(df[cm], errors="coerce")
    else:  df["county_median"] = np.nan
    if sm: df["state_median"] = pd.to_numeric(df[sm], errors="coerce")
    else:  df["state_median"] = np.nan

    data_store.claims_df = df.set_index("zip5")
    data_store.last_reload["flood_claims"] = datetime.utcnow().isoformat()
    return {"rows_loaded": int(len(df)), "path": CLAIMS_PATH}

def _read_ses() -> Dict[str, Any]:
    """
    Build STRtree from Tracts WKB.parquet and load the tract lookup JSON.
    Parquet is expected to have columns: GEOID, wkb (WKB geometry bytes).
    """
    if not os.path.exists(GEOMS_PATH):
        logger.warning(f"Geometry parquet not found: {GEOMS_PATH}")
        return {"polygons_loaded": 0, "scores_loaded": 0, "paths": {"geoms": GEOMS_PATH, "scores": SCORES_PATH}}

    df = pd.read_parquet(GEOMS_PATH)
    # Normalize column casing
    gcol = next((c for c in df.columns if c.lower() == "geoid"), None)
    wcol = next((c for c in df.columns if c.lower() == "wkb"), None)
    if not gcol or not wcol:
        raise HTTPException(status_code=500, detail=f"Tract parquet missing GEOID/wkb columns. Found: {list(df.columns)}")

    geoms = []
    geoids = []
    for _, row in df[[gcol, wcol]].iterrows():
        try:
            geom = shapely_wkb.loads(row[wcol])
            geoms.append(geom)
            geoids.append(str(row[gcol]))
        except Exception:
            continue

    if not os.path.exists(SCORES_PATH):
        logger.warning(f"Tract score file not found: {SCORES_PATH}")
        scores = {}
    else:
        with open(SCORES_PATH, "r") as f:
            scores = json.load(f)
        # normalize keys to 11-digit strings
        scores = {str(k).zfill(11): v for k, v in scores.items()}

    data_store.ses_tree = STRtree(geoms) if geoms else None
    data_store.ses_geoms = geoms
    data_store.ses_geoids = geoids
    data_store.ses_scores = scores
    data_store.last_reload["ses"] = datetime.utcnow().isoformat()

    logger.info(f"SES geoms: {len(geoms)}; SES scores: {len(scores)}")
    return {
        "polygons_loaded": len(geoms),
        "scores_loaded": len(scores),
        "paths": {"geoms": GEOMS_PATH, "scores": SCORES_PATH}
    }

def point_in_polygon_lookup(lat: float, lon: float) -> Tuple[Optional[str], Optional[float]]:
    """Return (tract GEOID, SES score_0_100) or (None, None)."""
    if not data_store.ses_tree:
        return None, None

    pt = Point(lon, lat)

    # Try query_bulk, then query; as a last resort scan all
    try:
        qb = data_store.ses_tree.query_bulk([pt])
        idxs = qb[1].tolist()
    except Exception:
        try:
            res = data_store.ses_tree.query(pt)
            if isinstance(res, np.ndarray) and res.dtype.kind in {"i","u"}:
                idxs = res.tolist()
            else:
                idxs = list(range(len(data_store.ses_geoms)))
        except Exception:
            idxs = list(range(len(data_store.ses_geoms)))

    for idx in idxs:
        try:
            g = data_store.ses_geoms[idx]
            if g.covers(pt):
                geoid = data_store.ses_geoids[idx]
                score = data_store.ses_scores.get(str(geoid).zfill(11))
                return geoid, _safe_float(score)
        except Exception:
            continue

    return None, None

# -----------------------------------------------------------------------------
# FEMA NFHL live query + scoring
# -----------------------------------------------------------------------------
async def query_fema_nfhl(lat: float, lon: float) -> Dict[str, Any]:
    url = "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28/query"
    params = {
        "f": "json",
        "returnGeometry": "false",
        "spatialRel": "esriSpatialRelIntersects",
        "geometry": json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}),
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,DEPTH",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        for attempt in range(3):
            try:
                r = await client.get(url, params=params)
                if r.status_code == 200:
                    data = r.json()
                    feats = data.get("features") or []
                    if feats:
                        a = feats[0].get("attributes") or {}
                        return {
                            "zone": a.get("FLD_ZONE"),
                            "zone_subtype": a.get("ZONE_SUBTY"),
                            "sfha": (a.get("SFHA_TF") == "T"),
                            "depth_ft": a.get("DEPTH"),
                            "floodway": ((a.get("ZONE_SUBTY") or "").upper() == "FLOODWAY"),
                        }
            except Exception as e:
                logger.warning(f"NFHL attempt {attempt+1} failed: {e}")
    return {}

def calculate_nfhl_score(nfhl: Dict[str, Any]) -> float:
    """Map NFHL attributes to a 0..1 safety score (higher = safer)."""
    if not nfhl:
        return 0.5
    zone = (nfhl.get("zone") or "").upper()
    subtype = (nfhl.get("zone_subtype") or "").upper()
    depth = nfhl.get("depth_ft") or 0
    floodway = bool(nfhl.get("floodway"))

    if floodway:
        return 0.2
    if zone in {"AE","A","VE"}:
        return 0.2 if (depth and depth > 3) else 0.4
    if "SHADED" in subtype:    # shaded X
        return 0.5
    if zone in {"X","B","C"}:
        return 0.8
    return 0.5

# -----------------------------------------------------------------------------
# Perplexity (Build Year)
# -----------------------------------------------------------------------------
class BuildYearRequest(BaseModel):
    hotel_name: str = Field(..., description="Hotel name")
    address: str = Field(..., description="Hotel address")

async def query_perplexity(hotel_name: str, address: str) -> Dict[str, Any]:
    if not PERPLEXITY_API_KEY:
        return {"year": None, "source": "perplexity", "confidence": "low", "error": "No API key configured"}

    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    prompt = (
        f"Extract the exact construction year for {hotel_name} at {address}.\n"
        'Return JSON: {"year": int or null, "source":"perplexity", "confidence":"high/medium/low"}'
    )
    payload = {"model": PERPLEXITY_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code == 200:
                data = r.json()
                content = data.get("choices",[{}])[0].get("message",{}).get("content","{}")
                try:
                    obj = json.loads(content)
                    return {"year": obj.get("year"), "source":"perplexity", "confidence": obj.get("confidence","medium")}
                except json.JSONDecodeError:
                    import re
                    m = re.search(r"\b(19|20)\d{2}\b", content)
                    if m:
                        return {"year": int(m.group()), "source":"perplexity", "confidence":"low"}
    except Exception as e:
        logger.warning(f"Perplexity error: {e}")

    return {"year": None, "source":"perplexity", "confidence":"low"}

def year_to_score(year: Optional[int]) -> Optional[float]:
    if year is None:
        return None
    cur = datetime.now().year
    if year >= cur - 3:   return 100.0
    if year >= cur - 9:   return 85.0
    if year >= cur - 20:  return 65.0
    if year >= 1986:      return 40.0
    return 20.0

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/reload/ses", dependencies=[Depends(verify_api_key)])
async def reload_ses():
    details = _read_ses()
    return {"success": True, "message": "SES data reloaded", "details": details}

@app.post("/reload/humidity", dependencies=[Depends(verify_api_key)])
async def reload_humidity():
    details = _read_humidity()
    return {"success": True, "message": "Humidity data reloaded", "details": details}

@app.post("/reload/flood_claims", dependencies=[Depends(verify_api_key)])
async def reload_claims():
    details = _read_claims()
    return {"success": True, "message": "Flood claims data reloaded", "details": details}

@app.get("/ses/score", dependencies=[Depends(verify_api_key)])
async def ses_score(lat: float = Query(...), lon: float = Query(...)):
    geoid, score = point_in_polygon_lookup(lat, lon)
    return {"geoid": geoid, "ses_score": score, "source": "Tracts WKB.parquet + Tract Lookup Fixed.json"}

@app.post("/build_year", dependencies=[Depends(verify_api_key)])
async def build_year(req: BuildYearRequest):
    res = await query_perplexity(req.hotel_name, req.address)
    score = year_to_score(res.get("year"))
    stars = round((score / 100 * 5), 1) if score is not None else None
    return {
        "year": res.get("year"),
        "source": res.get("source"),
        "confidence": res.get("confidence"),
        "score_0_100": score,
        "stars_5": stars
    }

@app.get("/humidity", dependencies=[Depends(verify_api_key)])
async def humidity(zip: str = Query(...)):
    if data_store.humidity_df is None:
        return {"zip": _norm_zip(zip), "humidity_risk_0_100": None, "humidity_risk_stars_5": None}

    z = _norm_zip(zip)
    if z not in data_store.humidity_df.index:
        return {"zip": z, "humidity_risk_0_100": None, "humidity_risk_stars_5": None}

    row = data_store.humidity_df.loc[z]
    risk = _safe_float(row.get("humidity_risk_0_100"))
    # If risk is missing, try per-row county/state medians (if present)
    if risk is None:
        risk = _safe_float(row.get("humidity_median_county")) or _safe_float(row.get("humidity_median_state"))

    if risk is None:
        return {"zip": z, "humidity_risk_0_100": None, "humidity_risk_stars_5": None}

    stars = round((risk/100.0)*5.0, 1)
    return {"zip": z, "humidity_risk_0_100": risk, "humidity_risk_stars_5": stars}

@app.get("/flood", dependencies=[Depends(verify_api_key)])
async def flood(
    lat: float = Query(...),
    lon: float = Query(...),
    zip: Optional[str] = Query(None)
):
    nfhl = await query_fema_nfhl(lat, lon)
    nfhl_score = calculate_nfhl_score(nfhl)  # 0..1

    claims_prop = None
    claims_safety = None
    z = _norm_zip(zip) if zip else None

    if z and data_store.claims_df is not None and z in data_store.claims_df.index:
        row = data_store.claims_df.loc[z]
        claims_prop = _safe_float(row.get("proportion"))
        if claims_prop is None:
            # fallbacks per row (county/state)
            claims_prop = _safe_float(row.get("county_median")) or _safe_float(row.get("state_median"))
        if claims_prop is not None:
            claims_safety = max(0.0, min(1.0, 1.0 - float(claims_prop)))

    parts = [nfhl_score]
    if claims_safety is not None:
        parts.append(claims_safety)

    flood_safety_0_1 = float(sum(parts) / len(parts)) if parts else None
    flood_safety_0_100 = round(flood_safety_0_1 * 100.0, 1) if flood_safety_0_1 is not None else None
    flood_stars = round((flood_safety_0_100 / 100.0) * 5.0, 1) if flood_safety_0_100 is not None else None

    return {
        "nfhl": {
            "zone": nfhl.get("zone"),
            "zone_subtype": nfhl.get("zone_subtype"),
            "sfha": bool(nfhl.get("sfha", False)),
            "depth_ft": nfhl.get("depth_ft"),
            "floodway": bool(nfhl.get("floodway", False)),
            "nfhl_score_0_1": nfhl_score,
        },
        "claims": {
            "zip5": z,
            "proportion_0_1": claims_prop,
            "claims_safety_0_1": claims_safety
        },
        "flood_safety_0_100": flood_safety_0_100,
        "flood_safety_stars_5": flood_stars
    }

@app.get("/score/total", dependencies=[Depends(verify_api_key)])
async def total_score(
    lat: float = Query(...),
    lon: float = Query(...),
    zip: Optional[str] = Query(None),
    hotel_name: Optional[str] = Query(None),
    address: Optional[str] = Query(None)
):
    components: Dict[str, Dict[str, Any]] = {}
    weights_used: Dict[str, float] = {}

    # SES (by tract)
    geoid, ses_val = point_in_polygon_lookup(lat, lon)
    if ses_val is not None:
        components["ses"] = {"geoid": geoid, "score_0_100": float(ses_val)}
        weights_used["ses"] = W_SES

    # Build year
    if hotel_name and address:
        by = await query_perplexity(hotel_name, address)
        year = by.get("year")
        by_score = year_to_score(year)
        if by_score is not None:
            components["build_year"] = {
                "year": year,
                "score_0_100": float(by_score),
                "confidence": by.get("confidence","low"),
                "source": "perplexity"
            }
            weights_used["build_year"] = W_BUILD_YEAR

    # Flood
    fr = await flood(lat, lon, zip)  # call our own endpoint impl
    fscore = fr.get("flood_safety_0_100")
    if fscore is not None:
        components["flood_safety"] = {"score_0_100": float(fscore), "details": {"nfhl": fr["nfhl"], "claims": fr["claims"]}}
        weights_used["flood"] = W_FLOOD

    # Humidity safety = (100 - risk)
    z = _norm_zip(zip) if zip else None
    if z and data_store.humidity_df is not None and z in data_store.humidity_df.index:
        row = data_store.humidity_df.loc[z]
        risk = _safe_float(row.get("humidity_risk_0_100"))
        if risk is None:
            risk = _safe_float(row.get("humidity_median_county")) or _safe_float(row.get("humidity_median_state"))
        if risk is not None:
            hs = float(100.0 - float(risk))
            components["humidity_safety"] = {"score_0_100": hs, "raw_humidity_risk_0_100": float(risk)}
            weights_used["humidity"] = W_HUMIDITY

    # Aggregate with renormalization across present components
    if not weights_used:
        return {
            "inputs": {"lat": lat, "lon": lon, "zip": zip, "hotel_name": hotel_name, "address": address},
            "components": components,
            "weights_applied": {},
            "total_score_0_100": None,
            "total_stars_5": None
        }

    total_w = sum(weights_used.values())
    norm_w = {k: v/total_w for k, v in weights_used.items()}

    key_to_weight = {
        "ses": "ses",
        "build_year": "build_year",
        "flood_safety": "flood",
        "humidity_safety": "humidity",
    }

    total = 0.0
    for k, comp in components.items():
        wk = key_to_weight.get(k)
        sc = comp.get("score_0_100")
        if wk in norm_w and sc is not None:
            total += float(sc) * norm_w[wk]

    total_100 = round(total, 1)
    stars = round((total_100 / 100.0) * 5.0, 1)
    return {
        "inputs": {"lat": lat, "lon": lon, "zip": zip, "hotel_name": hotel_name, "address": address},
        "components": components,
        "weights_applied": norm_w,
        "total_score_0_100": total_100,
        "total_stars_5": stars
    }

# -----------------------------------------------------------------------------
# Startup loader
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Property Score API...")
    try:
        _read_ses()
    except Exception as e:
        logger.warning(f"SES load failed: {e}")

    try:
        _read_humidity()
    except Exception as e:
        logger.warning(f"Humidity load failed: {e}")

    try:
        _read_claims()
    except Exception as e:
        logger.warning(f"Claims load failed: {e}")

    logger.info("Ready.")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "development") == "development",
    )
