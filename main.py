"""
Property Score API (Render-ready)
- SES from tract polygons + tract score lookup (fallback to ZIP affluence)
- Flood resiliency from NFHL (partial-field mapping + retries) + NFIP claims policy
- Humidity (ZIP risk with county/state medians as fallback)
- Build year via Perplexity
- Optional Google-reviews support inside /score/total

All scores are 0–100 where higher = safer/better (stars are 0–5).
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import math

from fastapi import FastAPI, Depends, HTTPException, Header, Query, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

import pandas as pd
import numpy as np

from shapely import wkb as shapely_wkb
from shapely.geometry import Point
from shapely.strtree import STRtree

import httpx

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("property-score-api")

# ---------------------------------------------------------------------
# ENV & Defaults
# ---------------------------------------------------------------------
API_KEY = os.getenv("API_KEY")  # REQUIRED
if not API_KEY:
    raise ValueError("API_KEY env var is required")
API_BEARER = os.getenv("API_BEARER")  # optional secondary bearer

DATA_DIR = os.getenv("DATA_DIR", "/data")

# You told me your file names were changed to these:
GEOMS_PATH    = os.getenv("GEOMS_PATH", os.path.join(DATA_DIR, "Tracts WKB.parquet"))
SCORES_PATH   = os.getenv("SCORES_PATH", os.path.join(DATA_DIR, "Tract Lookup Fixed.json"))
HUMIDITY_PATH = os.getenv("HUMIDITY_PATH", os.path.join(DATA_DIR, "Humidity Medians.parquet"))
CLAIMS_PATH   = os.getenv("CLAIMS_PATH", os.path.join(DATA_DIR, "Claims with Medians.parquet"))

# Optional (ZIP affluence fallback: either affluence_0_100 or mhi_dollars)
AFFLUENCE_ZIP_PATH = os.getenv("AFFLUENCE_ZIP_PATH", os.path.join(DATA_DIR, "zip_affluence.csv"))

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_MODEL   = os.getenv("PERPLEXITY_MODEL", "sonar")

# Weights (can be tuned via env)
W_SES        = float(os.getenv("W_SES",        "0.15"))  # “Area Affluence” in UI
W_BUILD_YEAR = float(os.getenv("W_BUILD_YEAR", "0.20"))
W_FLOOD      = float(os.getenv("W_FLOOD",      "0.45"))
W_HUMIDITY   = float(os.getenv("W_HUMIDITY",   "0.10"))
# Google’s placeholder (actual effective 0.10 or 0.20 applied dynamically)
W_GOOGLE_PLACEHOLDER = float(os.getenv("W_GOOGLE", "0.10"))

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(
    title="Property Score API",
    version="1.1.0",
    docs_url=None, redoc_url=None, openapi_url=None
)
security = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------
# Model / DataStore
# ---------------------------------------------------------------------
class DataStore:
    # SES / Tracts
    ses_tree: Optional[STRtree] = None
    ses_geoms: List = []
    ses_geoids: List[str] = []
    ses_scores: Dict[str, float] = {}

    # Humidity (ZIP risk + medians)
    humidity: Optional[pd.DataFrame] = None  # index on 'zip'
    # Claims (ZIP proportion + medians) — NOTE: medians are NOT used in flood math per policy
    claims: Optional[pd.DataFrame] = None    # index on 'zip'

    # ZIP affluence: zip -> 0..100
    zip_affluence: Dict[str, float] = {}

    last_reload: Dict[str, str] = {}

data_store = DataStore()

# ---------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------
def compare_api_keys(provided: str, expected: Optional[str]) -> bool:
    if not expected or provided is None:
        return False
    if len(provided) != len(expected):
        return False
    # constant time
    acc = 0
    for a, b in zip(provided, expected):
        acc |= ord(a) ^ ord(b)
    return acc == 0

def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> bool:
    if x_api_key and compare_api_keys(x_api_key, API_KEY):
        return True
    if API_BEARER and x_api_key and compare_api_keys(x_api_key, API_BEARER):
        return True
    if authorization and authorization.credentials:
        if compare_api_keys(authorization.credentials, API_KEY):
            return True
        if API_BEARER and compare_api_keys(authorization.credentials, API_BEARER):
            return True
    raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ---------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------
def _read_parquet_or_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        if path.lower().endswith(".parquet"):
            return pd.read_parquet(path)
        # csv fallbacks
        for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
            try:
                return pd.read_csv(path, encoding=enc, engine="python")
            except Exception:
                continue
        return None
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return None

def _zip_norm(s: Any) -> str:
    if s is None: return ""
    z = str(s)
    # keep 3-5 digits then zfill
    import re
    m = re.search(r"(\d{3,5})", z)
    return m.group(1).zfill(5) if m else ""

# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def load_ses_data():
    # geometries (parquet with columns: GEOID:str, wkb:bytes)
    if not os.path.exists(GEOMS_PATH):
        logger.warning(f"Tract parquet not found: {GEOMS_PATH}")
    else:
        df = pd.read_parquet(GEOMS_PATH)
        if not {"GEOID", "wkb"}.issubset(df.columns):
            raise RuntimeError(f"{GEOMS_PATH} must contain columns: GEOID, wkb")
        geoms = [shapely_wkb.loads(b) for b in df["wkb"]]
        data_store.ses_tree = STRtree(geoms)
        data_store.ses_geoms = geoms
        data_store.ses_geoids = df["GEOID"].astype(str).tolist()
        logger.info(f"Loaded {len(geoms):,} tract geometries from {GEOMS_PATH}")

    # scores (JSON mapping "11-digit GEOID" -> score 0..100)
    if not os.path.exists(SCORES_PATH):
        logger.warning(f"SES scores not found: {SCORES_PATH}")
    else:
        with open(SCORES_PATH, "r") as f:
            data_store.ses_scores = json.load(f)
        logger.info(f"Loaded {len(data_store.ses_scores):,} SES scores from {SCORES_PATH}")

    data_store.last_reload["ses"] = datetime.now().isoformat()
    return {
        "polygons_loaded": len(data_store.ses_geoms),
        "scores_loaded": len(data_store.ses_scores),
        "paths": {"geoms": GEOMS_PATH, "scores": SCORES_PATH}
    }

def load_humidity_data():
    df = _read_parquet_or_csv(HUMIDITY_PATH)
    if df is None:
        logger.warning(f"Humidity file not found/readable: {HUMIDITY_PATH}")
        data_store.humidity = None
    else:
        # expected cols: zip, humidity_risk_0_100, (optional) humidity_county_median, humidity_state_median
        cols = {c.lower(): c for c in df.columns}
        zc = cols.get("zip") or cols.get("zip_code") or cols.get("zcta") or "zip"
        df["zip"] = df[zc].apply(_zip_norm)
        df["humidity_risk_0_100"] = pd.to_numeric(df.get("humidity_risk_0_100"), errors="coerce")
        # medians may exist
        for m in ["humidity_county_median", "humidity_state_median", "humidity_nearest_median"]:
            if m in df.columns:
                df[m] = pd.to_numeric(df[m], errors="coerce")
        data_store.humidity = df.set_index("zip")
        logger.info(f"Loaded humidity rows: {len(data_store.humidity):,} from {HUMIDITY_PATH}")

    data_store.last_reload["humidity"] = datetime.now().isoformat()
    return {"rows_loaded": 0 if data_store.humidity is None else len(data_store.humidity),
            "path": HUMIDITY_PATH}

def load_flood_claims():
    df = _read_parquet_or_csv(CLAIMS_PATH)
    if df is None:
        logger.warning(f"Claims file not found/readable: {CLAIMS_PATH}")
        data_store.claims = None
    else:
        # expected cols: zip, claims_proportion_0_1 (plus medians we WILL NOT USE)
        cols = {c.lower(): c for c in df.columns}
        zc = cols.get("zip") or cols.get("zip_code") or "zip"
        vc = cols.get("claims_proportion_0_1") or cols.get("proportion")
        if not (zc and vc):
            logger.warning("Claims file missing required columns (zip + claims_proportion_0_1/proportion)")
            data_store.claims = None
        else:
            df["zip"] = df[zc].apply(_zip_norm)
            df["claims_proportion_0_1"] = pd.to_numeric(df[vc], errors="coerce")
            data_store.claims = df.set_index("zip")
            logger.info(f"Loaded claims rows: {len(data_store.claims):,} from {CLAIMS_PATH}")

    data_store.last_reload["flood_claims"] = datetime.now().isoformat()
    return {"rows_loaded": 0 if data_store.claims is None else len(data_store.claims),
            "path": CLAIMS_PATH}

def load_zip_affluence():
    d = _read_parquet_or_csv(AFFLUENCE_ZIP_PATH)
    if d is None:
        logger.warning(f"AFFLUENCE_ZIP_PATH not found or unreadable: {AFFLUENCE_ZIP_PATH}")
        data_store.zip_affluence = {}
        return {"rows_loaded": 0, "path": AFFLUENCE_ZIP_PATH}

    cols = {c.lower(): c for c in d.columns}
    zc = cols.get("zip") or cols.get("zip5") or cols.get("zipcode") or cols.get("zcta")
    if not zc:
        logger.warning(f"ZIP affluence file lacks 'zip' column: {list(d.columns)}")
        data_store.zip_affluence = {}
        return {"rows_loaded": 0, "path": AFFLUENCE_ZIP_PATH}

    acc: Dict[str, float] = {}
    if "affluence_0_100" in cols:
        ac = cols["affluence_0_100"]
        tmp = d[[zc, ac]].copy()
        tmp["zip"] = tmp[zc].apply(_zip_norm)
        tmp["affluence_0_100"] = pd.to_numeric(tmp[ac], errors="coerce")
        acc = {z: float(v) for z, v in zip(tmp["zip"], tmp["affluence_0_100"]) if pd.notna(v)}
    else:
        # derive percentile from mhi_dollars if present
        mhic = cols.get("mhi_dollars") or cols.get("median_household_income")
        if mhic:
            tmp = d[[zc, mhic]].copy()
            tmp["zip"] = tmp[zc].apply(_zip_norm)
            tmp[mhic]  = pd.to_numeric(tmp[mhic], errors="coerce")
            series = tmp[mhic].dropna()
            if len(series) > 0:
                pct = series.rank(pct=True) * 100.0
                tmp["affluence_0_100"] = tmp[mhic].map(pct)
                acc = {z: float(v) for z, v in zip(tmp["zip"], tmp["affluence_0_100"]) if pd.notna(v)}
        else:
            logger.warning("ZIP affluence file has neither affluence_0_100 nor mhi_dollars")
            acc = {}

    data_store.zip_affluence = acc
    logger.info(f"Loaded ZIP affluence rows: {len(acc):,} from {AFFLUENCE_ZIP_PATH}")
    return {"rows_loaded": len(acc), "path": AFFLUENCE_ZIP_PATH}

# ---------------------------------------------------------------------
# SES lookup
# ---------------------------------------------------------------------
def point_in_polygon_lookup(lat: float, lon: float) -> Tuple[Optional[str], Optional[float]]:
    if not data_store.ses_tree:
        return None, None
    pt = Point(lon, lat)

    # Shapely 2.x preferred: query_bulk
    try:
        idxs = data_store.ses_tree.query_bulk([pt])[1].tolist()
    except Exception:
        # Fallback: query may return indices or geoms
        cand = data_store.ses_tree.query(pt)
        if isinstance(cand, np.ndarray) and cand.dtype.kind in {"i", "u"}:
            idxs = cand.tolist()
        else:
            idxs = range(len(data_store.ses_geoms))

    for idx in idxs:
        try:
            geom = data_store.ses_geoms[idx]
            if geom.covers(pt):
                geoid = data_store.ses_geoids[idx]
                return geoid, data_store.ses_scores.get(geoid)
        except Exception:
            continue
    return None, None

# ---------------------------------------------------------------------
# NFHL discovery + mapping
# ---------------------------------------------------------------------
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://hazards.fema.gov/",
    "Accept": "application/json,text/plain,*/*",
}
_NFHL_DISCOVERY = {"root": None, "layer": None}
_SERVICE_ROOTS = [
    # Newer ArcGIS Server roots (preferred)
    "https://hazards.fema.gov/server/rest/services/public/NFHL/MapServer",
    "https://hazards.fema.gov/server/rest/services/NFHL/MapServer",
    # Legacy fallbacks (often 404 nowadays)
    "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer",
    "https://gis.fema.gov/arcgis/rest/services/NFHL/MapServer",
]

async def _discover_nfhl_layer() -> bool:
    async with httpx.AsyncClient(timeout=25.0, headers=_HEADERS) as client:
        for root in _SERVICE_ROOTS:
            try:
                r = await client.get(root, params={"f": "pjson"})
                if r.status_code != 200:
                    continue
                data = r.json()
                layers = data.get("layers") or []
                target = None
                # Prefer “S_Fld_Haz_Ar”
                for lyr in layers:
                    if "s_fld_haz_ar" in (lyr.get("name", "").lower()):
                        target = lyr
                        break
                if not target:
                    # fallback: find a layer with FLD_ZONE field
                    for lyr in layers:
                        info = await client.get(f"{root}/{lyr['id']}", params={"f": "pjson"})
                        if info.status_code != 200:
                            continue
                        fields = [f.get("name", "").upper() for f in info.json().get("fields", [])]
                        if "FLD_ZONE" in fields:
                            target = lyr
                            break
                if target:
                    _NFHL_DISCOVERY["root"] = root
                    _NFHL_DISCOVERY["layer"] = target["id"]
                    logger.info(f"NFHL discovered: root={root}, layer={target['id']} ({target['name']})")
                    return True
            except Exception as e:
                logger.warning(f"NFHL discovery error at {root}: {e}")
    return False

async def query_fema_nfhl(lat: float, lon: float) -> Dict[str, Any]:
    if not (_NFHL_DISCOVERY["root"] and _NFHL_DISCOVERY["layer"]):
        ok = await _discover_nfhl_layer()
        if not ok:
            return {}
    root = _NFHL_DISCOVERY["root"]
    layer = _NFHL_DISCOVERY["layer"]

    params = {
        "f": "json",
        "returnGeometry": "false",
        "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,DEPTH",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "geometryType": "esriGeometryPoint",
        "geometry": json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}),
    }
    url = f"{root}/{layer}/query"

    async with httpx.AsyncClient(timeout=25.0, headers=_HEADERS) as client:
        try:
            r = await client.get(url, params=params)
            if r.status_code != 200 or "json" not in (r.headers.get("Content-Type") or "").lower():
                # rediscover once
                if await _discover_nfhl_layer():
                    root = _NFHL_DISCOVERY["root"]; layer = _NFHL_DISCOVERY["layer"]
                    url = f"{root}/{layer}/query"
                    r = await client.get(url, params=params)
                    if r.status_code != 200 or "json" not in (r.headers.get("Content-Type") or "").lower():
                        return {}
                else:
                    return {}
            data = r.json()
            feats = data.get("features") or []
            if not feats:
                return {}
            attrs = feats[0].get("attributes", {}) or {}
            # normalize keys we use later
            return {
                "zone": attrs.get("FLD_ZONE"),
                "zone_subtype": attrs.get("ZONE_SUBTY"),
                "sfha": True if attrs.get("SFHA_TF") == "T" else (False if attrs.get("SFHA_TF") == "F" else None),
                "depth_ft": attrs.get("DEPTH"),
            }
        except Exception as e:
            logger.warning(f"NFHL query error: {e}")
            return {}

def compute_zone_safety(nfhl: Dict[str, Any]) -> int:
    z = (nfhl.get("zone") or "").upper().strip()
    sub = (nfhl.get("zone_subtype") or "").upper()
    sfha = nfhl.get("sfha")
    depth = nfhl.get("depth_ft")

    if "FLOODWAY" in sub:
        return 10
    if z.startswith("VE"):
        return 15
    if sfha is True:
        base = 25
        try:
            d = float(depth) if depth is not None else None
            if d is not None:
                if d >= 3.0: base = 15
                elif d >= 1.0: base = 20
        except Exception:
            pass
        return int(base)
    if z in ("X", "B", "C"):
        return 90
    if "SHADED" in z or "0.2 PCT ANNUAL CHANCE" in sub or z == "D":
        return 55
    if sfha is False:
        return 80
    if depth is not None and (not z) and (sfha is None):
        try:
            d = float(depth)
            if d >= 3.0: return 15
            if d >= 1.0: return 20
            return 25
        except Exception:
            return 25
    return 55

def combine_flood_resilience(nfhl_attrs: Dict[str, Any],
                             claims_safety_0_100: Optional[float]) -> Tuple[Optional[int], Optional[float]]:
    zone = compute_zone_safety(nfhl_attrs) if nfhl_attrs else None
    if zone is None and claims_safety_0_100 is None:
        return None, None

    if zone is not None and claims_safety_0_100 is not None:
        safety = 0.65 * float(zone) + 0.35 * float(claims_safety_0_100)
    elif zone is not None:
        safety = float(zone)
    else:
        safety = float(claims_safety_0_100)

    safety = max(10.0, min(100.0, float(round(safety))))
    stars = round(((safety/100.0)*5.0)*2)/2.0
    return int(safety), stars

async def query_nfhl_with_retries(lat: float, lon: float, max_attempts: int = 3) -> Dict[str, Any]:
    nfhl = {}
    for _ in range(max_attempts):
        nfhl = await query_fema_nfhl(lat, lon)
        if nfhl and any(nfhl.get(k) for k in ("zone", "zone_subtype", "sfha", "depth_ft")):
            break
    return nfhl or {}

# ---------------------------------------------------------------------
# Perplexity build-year
# ---------------------------------------------------------------------
class BuildYearRequest(BaseModel):
    hotel_name: str = Field(..., description="Hotel name")
    address: str = Field(..., description="Hotel address")

async def query_perplexity(hotel_name: str, address: str) -> Dict[str, Any]:
    if not PERPLEXITY_API_KEY:
        return {"year": None, "source": "perplexity", "confidence": "low", "error": "No API key"}
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    prompt = (
        f"Extract the exact construction year for {hotel_name} at {address}.\n"
        'Return JSON: {"year": int or null, "source": "perplexity", "confidence": "high/medium/low"}'
    )
    payload = {"model": PERPLEXITY_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code != 200:
                return {"year": None, "source": "perplexity", "confidence": "low"}
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            try:
                obj = json.loads(content)
                return {"year": obj.get("year"), "source": "perplexity", "confidence": obj.get("confidence", "medium")}
            except Exception:
                import re
                m = re.search(r"\b(19|20)\d{2}\b", content)
                if m:
                    return {"year": int(m.group()), "source": "perplexity", "confidence": "low"}
                return {"year": None, "source": "perplexity", "confidence": "low"}
    except Exception:
        return {"year": None, "source": "perplexity", "confidence": "low"}

def year_to_score(year: Optional[int]) -> Optional[float]:
    if not year:
        return None
    y = int(year)
    now = datetime.now().year
    if y >= now - 3:   return 100.0
    if y >= now - 9:   return 85.0
    if y >= now - 20:  return 65.0
    if y >= 1986:      return 40.0
    return 20.0

# ---------------------------------------------------------------------
# Google reviews (optional component for /score/total)
# ---------------------------------------------------------------------
def compute_google_component(rating: Optional[float], review_count: Optional[int]) -> Dict[str, Any]:
    if rating is None or review_count is None or review_count <= 0:
        return {"score100": None, "effective_weight": 0.0}

    # normalize rating 1..5 to 0..1
    normalized = max(0.0, min(1.0, (float(rating) - 1.0) / 4.0))
    # saturating confidence by review count
    k = 40.0
    rc = min(float(review_count), 1000.0)
    confidence = 1.0 - math.exp(-rc / k)
    score100 = 100.0 * (normalized * confidence)

    # asymmetric weight (bad ratings penalize more)
    wg = 0.20 if rating < 3.0 else 0.10
    return {"score100": score100, "effective_weight": wg}

# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/reload/ses", dependencies=[Depends(verify_api_key)])
async def reload_ses():
    details = load_ses_data()
    return {"success": True, "message": "SES data reloaded", "details": details}

@app.post("/reload/humidity", dependencies=[Depends(verify_api_key)])
async def reload_humidity():
    details = load_humidity_data()
    return {"success": True, "message": "Humidity data reloaded", "details": details}

@app.post("/reload/flood_claims", dependencies=[Depends(verify_api_key)])
async def reload_claims():
    details = load_flood_claims()
    return {"success": True, "message": "Claims data reloaded", "details": details}

@app.post("/reload/affluence", dependencies=[Depends(verify_api_key)])
async def reload_affluence():
    details = load_zip_affluence()
    return {"success": True, "message": "ZIP affluence reloaded", "details": details}

@app.get("/ses/score", dependencies=[Depends(verify_api_key)])
async def ses_score(lat: float = Query(...), lon: float = Query(...), zip: Optional[str] = Query(None)):
    geoid, score = point_in_polygon_lookup(lat, lon)
    fallback = None
    if score is None and zip:
        z = _zip_norm(zip)
        aff = data_store.zip_affluence.get(z)
        if aff is not None:
            score = float(aff)
            fallback = "zip_affluence"
    return {"geoid": geoid, "ses_score_0_100": score, "fallback": fallback}

@app.get("/humidity", dependencies=[Depends(verify_api_key)])
async def humidity_score(zip: str = Query(...)):
    z = _zip_norm(zip)
    df = data_store.humidity
    if df is None or z not in df.index:
        return {"zip": z, "humidity_risk_0_100": None, "fallback": "none"}
    row = df.loc[z]
    # prefer primary risk; fallback to medians if missing
    if pd.notna(row.get("humidity_risk_0_100")):
        return {"zip": z, "humidity_risk_0_100": float(row["humidity_risk_0_100"]), "fallback": "primary"}
    for f in ["humidity_nearest_median", "humidity_county_median", "humidity_state_median"]:
        if f in row and pd.notna(row.get(f)):
            return {"zip": z, "humidity_risk_0_100": float(row[f]), "fallback": f}
    return {"zip": z, "humidity_risk_0_100": None, "fallback": "none"}

@app.get("/flood", dependencies=[Depends(verify_api_key)])
async def flood_score(lat: float = Query(...), lon: float = Query(...), zip: Optional[str] = Query(None)):
    nfhl_status = {"attempts": 3, "available": None, "note": None}
    nfhl_attrs: Dict[str, Any] = {}
    try:
        nfhl_attrs = await query_nfhl_with_retries(lat, lon, 3)
        nfhl_status["available"] = bool(nfhl_attrs)
        if not nfhl_attrs:
            nfhl_status["note"] = "nfhl_unavailable_or_empty"
    except Exception as e:
        nfhl_status["available"] = False
        nfhl_status["note"] = f"nfhl_error:{e}"

    # Claims → safety (0–100) = (1 - proportion)*100; per policy no medians if missing
    z = _zip_norm(zip) if zip else None
    claims_prop = None
    claims_safety = None
    claims_missing = False
    if z and data_store.claims is not None and z in data_store.claims.index:
        val = data_store.claims.loc[z].get("claims_proportion_0_1")
        if pd.notna(val):
            claims_prop = float(val)
            claims_safety = max(0.0, min(100.0, (1.0 - claims_prop) * 100.0))
        else:
            claims_missing = True
    elif z:
        claims_missing = True

    safety_0_100, stars_5 = combine_flood_resilience(nfhl_attrs, claims_safety)

    if safety_0_100 is None:
        return {
            "nfhl": {"attributes": nfhl_attrs or None, "status": nfhl_status},
            "claims": {"zip5": z, "proportion_0_1": claims_prop,
                       "claims_safety_0_100": claims_safety, "claims_missing": claims_missing},
            "flood_safety_0_100": None,
            "flood_safety_stars_5": None,
            "low_confidence": True
        }

    low_conf = (not nfhl_attrs) or (claims_safety is None)
    return {
        "nfhl": {"attributes": nfhl_attrs or None, "status": nfhl_status},
        "claims": {"zip5": z, "proportion_0_1": claims_prop,
                   "claims_safety_0_100": claims_safety, "claims_missing": claims_missing},
        "flood_safety_0_100": safety_0_100,
        "flood_safety_stars_5": stars_5,
        "low_confidence": low_conf
    }

@app.post("/build_year", dependencies=[Depends(verify_api_key)])
async def build_year(req: BuildYearRequest):
    res = await query_perplexity(req.hotel_name, req.address)
    year = res.get("year")
    score = year_to_score(year)
    stars = None if score is None else round((score/100.0)*5.0, 1)
    return {
        "year": year,
        "source": res.get("source", "perplexity"),
        "confidence": res.get("confidence", "low"),
        "score_0_100": score,
        "stars_5": stars
    }

@app.get("/score/total", dependencies=[Depends(verify_api_key)])
async def total_score(
    lat: float = Query(...),
    lon: float = Query(...),
    zip: Optional[str] = Query(None),
    hotel_name: Optional[str] = Query(None),
    address: Optional[str] = Query(None),
    # optional Google reviews
    rating: Optional[float] = Query(None),
    review_count: Optional[int] = Query(None),
):
    components: Dict[str, Any] = {}
    weights_used: Dict[str, float] = {}

    # SES (tract) with ZIP affluence fallback
    geoid, ses = point_in_polygon_lookup(lat, lon)
    ses_fallback = None
    if ses is None and zip:
        z = _zip_norm(zip)
        aff = data_store.zip_affluence.get(z)
        if aff is not None:
            ses = float(aff)
            ses_fallback = "zip_affluence"
    if ses is not None:
        components["ses"] = {"geoid": geoid, "score_0_100": float(ses), "fallback": ses_fallback}
        weights_used["ses"] = W_SES

    # Build year (optional)
    if hotel_name and address:
        by = await query_perplexity(hotel_name, address)
        year = by.get("year")
        by_score = year_to_score(year)
        if by_score is not None:
            components["build_year"] = {"year": year, "score_0_100": float(by_score),
                                        "confidence": by.get("confidence", "low"),
                                        "source": "perplexity"}
            weights_used["build_year"] = W_BUILD_YEAR

    # Flood
    flood = await flood_score(lat=lat, lon=lon, zip=zip)  # reuse endpoint logic
    fs = flood.get("flood_safety_0_100")
    if fs is not None:
        components["flood_safety"] = {"score_0_100": float(fs), "details": flood}
        weights_used["flood"] = W_FLOOD

    # Humidity (ZIP)
    hz = None
    if zip:
        h = await humidity_score(zip=zip)
        hz = h.get("humidity_risk_0_100")
        if hz is not None:
            hum_safe = 100.0 - float(hz)
            components["humidity_safety"] = {"score_0_100": hum_safe, "raw_humidity_risk_0_100": float(hz),
                                             "fallback": h.get("fallback")}
            weights_used["humidity"] = W_HUMIDITY

    # Google reviews (optional)
    google_comp = compute_google_component(rating, review_count)
    if google_comp["score100"] is not None:
        components["google_reviews"] = {
            "score_0_100": float(google_comp["score100"]),
            "rating": rating, "review_count": review_count,
            "effective_weight": google_comp["effective_weight"]
        }

    # Weight renormalization
    if not weights_used and google_comp["score100"] is None:
        return {
            "inputs": {"lat": lat, "lon": lon, "zip": zip, "hotel_name": hotel_name, "address": address,
                       "rating": rating, "review_count": review_count},
            "components": components,
            "weights_applied": {},
            "total_score_0_100": None,
            "total_stars_5": None
        }

    # If Google present → substitute effective weight then renormalize
    wf = weights_used.get("flood", 0.0)
    wn = weights_used.get("build_year", 0.0)
    wa = weights_used.get("ses", 0.0)
    wh = weights_used.get("humidity", 0.0)

    if google_comp["score100"] is None:
        total_w = wf + wn + wa + wh
        wf /= total_w if total_w else 1.0
        wn /= total_w if total_w else 1.0
        wa /= total_w if total_w else 1.0
        wh /= total_w if total_w else 1.0
        wg = 0.0
    else:
        wg = float(google_comp["effective_weight"])
        # keep base placeholders (they already approximate sum=0.90 without Google)
        sumw = wf + wn + wa + wh + wg
        wf /= sumw; wn /= sumw; wa /= sumw; wh /= sumw; wg /= sumw

    total = 0.0
    if fs is not None: total += wf * float(fs)
    if "build_year" in components: total += wn * components["build_year"]["score_0_100"]
    if ses is not None: total += wa * float(ses)
    if hz is not None: total += wh * (100.0 - float(hz))
    if google_comp["score100"] is not None: total += wg * float(google_comp["score100"])

    total_100 = round(total, 1)
    # MVP cap rule: if any core < 40 (flood, newness, affluence, humidity), cap to min
    redflags: List[float] = []
    if fs is not None: redflags.append(float(fs))
    if "build_year" in components: redflags.append(float(components["build_year"]["score_0_100"]))
    if ses is not None: redflags.append(float(ses))
    if hz is not None: redflags.append(100.0 - float(hz))
    if redflags and min(redflags) < 40.0:
        total_100 = min(total_100, round(min(redflags), 1))
    stars = round((total_100/100.0)*5.0, 1)

    return {
        "inputs": {"lat": lat, "lon": lon, "zip": zip, "hotel_name": hotel_name, "address": address,
                   "rating": rating, "review_count": review_count},
        "components": components,
        "weights_applied": {"flood": round(wf, 4), "build_year": round(wn, 4),
                            "ses": round(wa, 4), "humidity": round(wh, 4),
                            "google_reviews": round(wg, 4)},
        "total_score_0_100": total_100,
        "total_stars_5": stars
    }

# ---------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Property Score API...")
    try: load_ses_data()
    except Exception as e: logger.warning(f"SES load failed: {e}")
    try: load_humidity_data()
    except Exception as e: logger.warning(f"Humidity load failed: {e}")
    try: load_flood_claims()
    except Exception as e: logger.warning(f"Claims load failed: {e}")
    try: load_zip_affluence()
    except Exception as e: logger.warning(f"ZIP affluence load failed: {e}")
    logger.info("Property Score API ready.")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "production") == "development"
    )
