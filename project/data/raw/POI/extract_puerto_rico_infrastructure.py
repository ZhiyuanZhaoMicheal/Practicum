#!/usr/bin/env python3
"""
Puerto Rico Infrastructure Retry + HIFLD Merge
==============================================
1) Retry failed OSM grid chunks with higher timeout and subdivision.
2) Mandatory HIFLD supplement for power, hospitals, fire, police.
3) Merge into main dataset with source column.
4) Print post-merge summary tables.
"""

import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime


REQUIRED_PACKAGES = {
    "osmnx": "osmnx",
    "geopandas": "geopandas",
    "shapely": "shapely",
    "requests": "requests",
}


def ensure_dependencies() -> None:
    """Install required packages if missing."""
    for module_name, package_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            print(f"[DEPS] Installing missing package: {package_name}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            except subprocess.CalledProcessError:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--break-system-packages", package_name]
                )


ensure_dependencies()

import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MIN_LAT = 17.8
MAX_LAT = 18.6
MIN_LON = -67.35
MAX_LON = -65.2

# Original base grid geometry
LAT_STEP = 0.2
LON_STEP = 0.3

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

OVERPASS_TIMEOUT_SEC = 120
RETRY_OVERPASS_TIMEOUT_SEC = 220
REQUEST_SLEEP_SEC = 1
BASE_RETRIES = 2
RETRY_RETRIES = 3
RETRY_SUBDIVISIONS = 2  # 2x2 per failed chunk

EXPECTED_MIN_WARN = 1000
FAILOVER_MIN = 100

OUTPUT_DIR = os.path.dirname(__file__)
BASE_FILE = os.path.join(OUTPUT_DIR, "puerto_rico_grid_infra.csv")
FAILED_CHUNKS_FILE = os.path.join(OUTPUT_DIR, "puerto_rico_grid_failed_chunks.csv")
RETRY_FILE = os.path.join(OUTPUT_DIR, "puerto_rico_grid_retry_infra.csv")
HIFLD_FILE = os.path.join(OUTPUT_DIR, "puerto_rico_hifld_supplement.csv")
MERGED_FILE = os.path.join(OUTPUT_DIR, "puerto_rico_grid_infra_merged.csv")
CRITICAL_FILE = os.path.join(OUTPUT_DIR, "puerto_rico_critical_infra_2022.csv")

TAG_COLUMNS_TO_KEEP = [
    "name",
    "amenity",
    "power",
    "man_made",
    "shop",
    "tourism",
    "addr:city",
    "generator:source",
    "landuse",
    "social_facility",
    "service",
    "operator",
    "tower:type",
]

TELECOM_KEYWORDS = [
    "telecom",
    "communication",
    "communications",
    "telecommunications",
    "cellular",
    "wireless",
    "mobile",
    "telefon",
    "phone",
]

# authoritative HIFLD critical infrastructure service
HIFLD_SERVICE = (
    "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/"
    "Critical_Infrastructure_Map_Service/FeatureServer"
)

# layer_id, category, type
HIFLD_LAYERS = [
    (5, "core_energy", "plant"),                  # Electric Power Generation Plants
    (10, "healthcare", "hospital"),              # Hospitals
    (11, "emergency_shelter", "police"),         # Law Enforcement Locations
    (12, "emergency_shelter", "fire_station"),   # Fire Stations
    (8, "water_infrastructure", "wastewater_plant"),  # Wastewater plants
]


# ---------------------------------------------------------------------------
# OVERPASS HELPERS
# ---------------------------------------------------------------------------

def build_overpass_query(chunk: dict, timeout_sec: int) -> str:
    south = chunk["south"]
    west = chunk["west"]
    north = chunk["north"]
    east = chunk["east"]

    return f"""
[out:json][timeout:{timeout_sec}];
(
  nwr["power"~"^(plant|generator|substation)$"]({south},{west},{north},{east});
  nwr["amenity"~"^(hospital|clinic|fire_station|police|social_facility)$"]({south},{west},{north},{east});
  nwr["social_facility"="shelter"]({south},{west},{north},{east});
  nwr["man_made"~"^(water_works|wastewater_plant|pumping_station|mast|tower)$"]({south},{west},{north},{east});
  nwr["landuse"="industrial"]({south},{west},{north},{east});
  nwr["tourism"~"^(hotel|resort)$"]({south},{west},{north},{east});
  nwr["shop"="supermarket"]({south},{west},{north},{east});
);
out tags center qt;
""".strip()


def probe_overpass_json(endpoint: str) -> bool:
    query = (
        "[out:json][timeout:25];"
        "node[amenity=hospital](18.30,-66.20,18.40,-66.00);"
        "out 1;"
    )
    try:
        response = requests.post(endpoint, data={"data": query}, timeout=40)
        ctype = (response.headers.get("content-type") or "").lower()
        if response.status_code == 200 and "json" in ctype:
            obj = response.json()
            return isinstance(obj, dict) and "elements" in obj
    except Exception:
        return False
    return False


def pick_overpass_endpoint() -> str:
    for endpoint in OVERPASS_ENDPOINTS:
        if probe_overpass_json(endpoint):
            print(f"[CONFIG] Using Overpass endpoint: {endpoint}")
            return endpoint
    raise RuntimeError("No usable Overpass endpoint returning JSON.")


def fetch_chunk_overpass(endpoint: str, chunk: dict, timeout_sec: int) -> tuple:
    query = build_overpass_query(chunk, timeout_sec)
    started = time.time()

    try:
        response = requests.post(
            endpoint,
            data={"data": query},
            timeout=(25, timeout_sec + 45),
        )
        elapsed = time.time() - started

        if response.status_code != 200:
            return [], elapsed, f"HTTP {response.status_code}: {response.text[:140]}"

        ctype = (response.headers.get("content-type") or "").lower()
        if "json" not in ctype:
            return [], elapsed, f"Non-JSON response content-type={ctype}"

        payload = response.json()
        elements = payload.get("elements", [])
        rows = []
        for element in elements:
            tags = element.get("tags", {})
            if not tags:
                continue

            lat = element.get("lat")
            lon = element.get("lon")
            center = element.get("center", {})
            if lat is None:
                lat = center.get("lat")
            if lon is None:
                lon = center.get("lon")
            if lat is None or lon is None:
                continue

            rows.append(
                {
                    "osm_type": str(element.get("type", "")),
                    "osm_id": int(element.get("id", -1)),
                    "lat": float(lat),
                    "lon": float(lon),
                    "tags": tags,
                }
            )

        return rows, elapsed, ""

    except Exception as exc:
        elapsed = time.time() - started
        return [], elapsed, str(exc)


def subdivide_chunk(chunk: dict, n: int) -> list:
    """Split one bbox chunk into n x n subchunks."""
    south = float(chunk["south"])
    west = float(chunk["west"])
    north = float(chunk["north"])
    east = float(chunk["east"])

    dy = (north - south) / n
    dx = (east - west) / n

    out = []
    sub_id = 0
    for i in range(n):
        for j in range(n):
            s = south + i * dy
            n_ = south + (i + 1) * dy
            w = west + j * dx
            e = west + (j + 1) * dx
            out.append(
                {
                    "chunk_id": int(chunk.get("chunk_id", -1)),
                    "sub_id": sub_id,
                    "south": round(s, 6),
                    "west": round(w, 6),
                    "north": round(n_, 6),
                    "east": round(e, 6),
                }
            )
            sub_id += 1
    return out


# ---------------------------------------------------------------------------
# TAG CLASSIFICATION
# ---------------------------------------------------------------------------

def _safe_lower(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _is_communications_feature(tags: dict) -> bool:
    man_made = _safe_lower(tags.get("man_made"))
    if man_made not in {"mast", "tower"}:
        return False

    text_blob = " ".join(
        [
            _safe_lower(tags.get("service")),
            _safe_lower(tags.get("operator")),
            _safe_lower(tags.get("name")),
            _safe_lower(tags.get("tower:type")),
        ]
    )
    return any(keyword in text_blob for keyword in TELECOM_KEYWORDS)


def classify_tags(tags: dict) -> tuple:
    name = _safe_lower(tags.get("name"))
    amenity = _safe_lower(tags.get("amenity"))
    power = _safe_lower(tags.get("power"))
    man_made = _safe_lower(tags.get("man_made"))
    social_facility = _safe_lower(tags.get("social_facility"))
    tourism = _safe_lower(tags.get("tourism"))
    shop = _safe_lower(tags.get("shop"))
    landuse = _safe_lower(tags.get("landuse"))

    if power in {"plant", "generator", "substation"}:
        return "core_energy", power

    if amenity in {"hospital", "clinic"}:
        return "healthcare", amenity
    if "cdt" in name or "centro de diagnost" in name:
        return "healthcare", "cdt"

    if man_made in {"water_works", "wastewater_plant", "pumping_station"}:
        return "water_infrastructure", man_made

    if amenity in {"fire_station", "police"}:
        return "emergency_shelter", amenity
    if amenity == "social_facility" and (social_facility == "shelter" or "shelter" in name):
        return "emergency_shelter", "shelter"

    if _is_communications_feature(tags):
        return "communications", man_made

    if landuse == "industrial":
        return "major_industry", "industrial"

    if tourism in {"hotel", "resort"}:
        return "tourism_hospitality", tourism

    if shop == "supermarket":
        return "supply_chain", "supermarket"

    return None, None


def build_tags_json(tags: dict) -> str:
    kept = {}
    for key in TAG_COLUMNS_TO_KEEP:
        value = tags.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            kept[key] = value_str
    return json.dumps(kept, ensure_ascii=False, sort_keys=True)


def build_osm_df(records: list, source_name: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["id", "category", "type", "lat", "lon", "name", "city", "tags", "source"])

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["osm_type", "osm_id"], keep="first")

    classes = df["tags"].apply(classify_tags)
    df["category"] = classes.apply(lambda x: x[0])
    df["type"] = classes.apply(lambda x: x[1])
    df = df[df["category"].notna()].copy()

    if df.empty:
        return pd.DataFrame(columns=["id", "category", "type", "lat", "lon", "name", "city", "tags", "source"])

    df["id"] = df["osm_type"].astype(str) + "/" + df["osm_id"].astype(str)
    df["name"] = df["tags"].apply(lambda t: t.get("name"))
    df["city"] = df["tags"].apply(lambda t: t.get("addr:city") or "Puerto Rico")
    df["tags"] = df["tags"].apply(build_tags_json)
    df = df[df["tags"] != "{}"].copy()
    df["source"] = source_name

    df["lat_r"] = df["lat"].round(6)
    df["lon_r"] = df["lon"].round(6)
    df = df.drop_duplicates(subset=["source", "lat_r", "lon_r", "category", "type", "name"], keep="first")

    final_cols = ["id", "category", "type", "lat", "lon", "name", "city", "tags", "source"]
    return df[final_cols].copy()


# ---------------------------------------------------------------------------
# RETRY MODE
# ---------------------------------------------------------------------------

def retry_failed_chunks(endpoint: str, failed_chunks_df: pd.DataFrame) -> tuple:
    """Retry each failed chunk with higher timeout and 2x2 subdivision."""
    all_rows = []
    still_failed = []
    success_subchunks = 0

    if failed_chunks_df.empty:
        return pd.DataFrame(), {"retried": 0, "recovered": 0, "still_failed": 0}

    total = len(failed_chunks_df)
    print(f"[RETRY] Retrying {total} failed chunks with timeout={RETRY_OVERPASS_TIMEOUT_SEC}s")

    for idx, (_, row) in enumerate(failed_chunks_df.iterrows(), start=1):
        chunk = {
            "chunk_id": int(row["chunk_id"]),
            "south": float(row["south"]),
            "west": float(row["west"]),
            "north": float(row["north"]),
            "east": float(row["east"]),
        }

        recovered = False

        # Attempt full failed chunk first
        for attempt in range(1, RETRY_RETRIES + 1):
            rows, elapsed, error_text = fetch_chunk_overpass(endpoint, chunk, RETRY_OVERPASS_TIMEOUT_SEC)
            if rows:
                all_rows.extend(rows)
                recovered = True
                success_subchunks += 1
                print(
                    f"[RETRY-CHUNK] {idx}/{total} chunk={chunk['chunk_id']} "
                    f"rows={len(rows)} elapsed={elapsed:.1f}s"
                )
                break
            if not error_text:
                error_text = "empty response"
            print(
                f"[WARN] retry chunk={chunk['chunk_id']} full attempt={attempt}/{RETRY_RETRIES} "
                f"error={error_text}"
            )
            time.sleep(REQUEST_SLEEP_SEC)

        if recovered:
            continue

        # Subdivide as fallback
        subchunks = subdivide_chunk(chunk, RETRY_SUBDIVISIONS)
        subchunk_ok = 0
        for sub in subchunks:
            sub_success = False
            for attempt in range(1, RETRY_RETRIES + 1):
                rows, elapsed, error_text = fetch_chunk_overpass(endpoint, sub, RETRY_OVERPASS_TIMEOUT_SEC)
                if rows:
                    all_rows.extend(rows)
                    sub_success += 1
                    success_subchunks += 1
                    sub_success = True
                    print(
                        f"[RETRY-SUB] chunk={chunk['chunk_id']} sub={sub['sub_id']} "
                        f"rows={len(rows)} elapsed={elapsed:.1f}s"
                    )
                    break
                if not error_text:
                    error_text = "empty response"
                print(
                    f"[WARN] retry chunk={chunk['chunk_id']} sub={sub['sub_id']} "
                    f"attempt={attempt}/{RETRY_RETRIES} error={error_text}"
                )
                time.sleep(REQUEST_SLEEP_SEC)
            if not sub_success:
                pass

        if subchunk_ok == 0:
            still_failed.append(chunk)

        time.sleep(REQUEST_SLEEP_SEC)

    retry_df = build_osm_df(all_rows, source_name="osm_retry")
    stats = {
        "retried": total,
        "recovered": success_subchunks,
        "still_failed": len(still_failed),
        "still_failed_rows": still_failed,
    }
    return retry_df, stats


# ---------------------------------------------------------------------------
# HIFLD SUPPLEMENT
# ---------------------------------------------------------------------------

def fetch_hifld_layer(layer_id: int, category: str, infra_type: str) -> list:
    """Fetch one HIFLD layer inside Puerto Rico bbox."""
    query_url = f"{HIFLD_SERVICE}/{layer_id}/query"
    geometry = {
        "xmin": MIN_LON,
        "ymin": MIN_LAT,
        "xmax": MAX_LON,
        "ymax": MAX_LAT,
        "spatialReference": {"wkid": 4326},
    }

    offset = 0
    rows = []

    while True:
        params = {
            "f": "json",
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "geometryType": "esriGeometryEnvelope",
            "geometry": json.dumps(geometry),
            "inSR": 4326,
            "outSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "resultOffset": offset,
            "resultRecordCount": 2000,
        }

        response = requests.get(query_url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()

        if payload.get("error"):
            break

        features = payload.get("features", [])
        if not features:
            break

        for feat in features:
            attrs = feat.get("attributes", {}) or {}
            geom = feat.get("geometry", {}) or {}
            lat = geom.get("y")
            lon = geom.get("x")
            if lat is None or lon is None:
                continue

            name = (
                attrs.get("NAME")
                or attrs.get("Name")
                or attrs.get("FACILITY_NAME")
                or attrs.get("Facility_Name")
                or attrs.get("SITE_NAME")
            )
            city = attrs.get("CITY") or attrs.get("City") or attrs.get("addr_city") or "Puerto Rico"

            oid = attrs.get("OBJECTID") or attrs.get("ObjectId") or attrs.get("FID") or offset

            tags_payload = {
                "name": name,
                "addr:city": city,
                "source": "HIFLD",
                "layer_id": layer_id,
                "category_hint": category,
                "type_hint": infra_type,
            }

            rows.append(
                {
                    "id": f"hifld/{layer_id}/{oid}",
                    "category": category,
                    "type": infra_type,
                    "lat": float(lat),
                    "lon": float(lon),
                    "name": name,
                    "city": city,
                    "tags": json.dumps(tags_payload, ensure_ascii=False, sort_keys=True),
                    "source": "hifld",
                }
            )

        offset += len(features)
        if not payload.get("exceededTransferLimit"):
            break

    return rows


def run_hifld_supplement() -> pd.DataFrame:
    print("[HIFLD] Fetching mandatory supplement layers")
    all_rows = []
    for layer_id, category, infra_type in HIFLD_LAYERS:
        try:
            layer_rows = fetch_hifld_layer(layer_id, category, infra_type)
            all_rows.extend(layer_rows)
            print(
                f"[HIFLD] layer={layer_id} category={category} type={infra_type} rows={len(layer_rows)}"
            )
        except Exception as exc:
            print(f"[HIFLD] layer={layer_id} failed: {exc}")

    if not all_rows:
        return pd.DataFrame(columns=["id", "category", "type", "lat", "lon", "name", "city", "tags", "source"])

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["id"], keep="first")
    df["lat_r"] = df["lat"].round(6)
    df["lon_r"] = df["lon"].round(6)
    df = df.drop_duplicates(subset=["source", "lat_r", "lon_r", "category", "type", "name"], keep="first")

    final_cols = ["id", "category", "type", "lat", "lon", "name", "city", "tags", "source"]
    return df[final_cols].copy()


# ---------------------------------------------------------------------------
# MERGE + QA + REPORT
# ---------------------------------------------------------------------------

def normalize_existing_base(df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = ["id", "category", "type", "lat", "lon", "name", "city", "tags", "source"]
    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    out = df.copy()
    if "source" not in out.columns:
        out["source"] = "osm_grid"
    else:
        out["source"] = out["source"].fillna("osm_grid")

    for col in expected_cols:
        if col not in out.columns:
            out[col] = None

    out = out[expected_cols].copy()
    return out


def merge_datasets(base_df: pd.DataFrame, retry_df: pd.DataFrame, hifld_df: pd.DataFrame) -> pd.DataFrame:
    frames = [normalize_existing_base(base_df), retry_df, hifld_df]
    merged = pd.concat(frames, ignore_index=True)

    merged = merged.drop_duplicates(subset=["source", "id"], keep="first")
    merged["lat_r"] = pd.to_numeric(merged["lat"], errors="coerce").round(6)
    merged["lon_r"] = pd.to_numeric(merged["lon"], errors="coerce").round(6)
    merged = merged.drop_duplicates(
        subset=["source", "lat_r", "lon_r", "category", "type", "name"],
        keep="first",
    )

    merged = merged.drop(columns=["lat_r", "lon_r"], errors="ignore")
    merged = merged.sort_values(by=["source", "category", "type", "city", "name"], na_position="last")
    merged = merged.reset_index(drop=True)
    return merged


def verify_dataset(df: pd.DataFrame) -> tuple:
    checks = {
        "count": len(df),
        "outside_bounds": 0,
        "outside_examples": [],
        "pumping_station_count": 0,
        "water_works_count": 0,
        "warnings": [],
    }

    if df.empty:
        checks["warnings"].append("No records extracted.")
        return False, checks

    in_lat = df["lat"].between(MIN_LAT, MAX_LAT, inclusive="both")
    in_lon = df["lon"].between(MIN_LON, MAX_LON, inclusive="both")
    outside = df[~(in_lat & in_lon)]
    checks["outside_bounds"] = len(outside)

    if len(outside) > 0:
        checks["outside_examples"] = outside[["id", "lat", "lon", "source"]].head(10).to_dict(orient="records")
        checks["warnings"].append(
            f"{len(outside)} records fall outside Puerto Rico bounds (possible source/data issue)."
        )

    checks["pumping_station_count"] = int((df["type"] == "pumping_station").sum())
    checks["water_works_count"] = int((df["type"] == "water_works").sum())

    if checks["pumping_station_count"] == 0:
        checks["warnings"].append("pumping_station count is zero (possible tag/filter issue).")
    if checks["water_works_count"] == 0:
        checks["warnings"].append("water_works count is zero (possible tag/filter issue).")
    if checks["count"] < EXPECTED_MIN_WARN:
        checks["warnings"].append(
            f"Record count is {checks['count']}, below expected {EXPECTED_MIN_WARN}+ for whole-island coverage."
        )

    ok = checks["count"] >= FAILOVER_MIN and checks["outside_bounds"] == 0
    return ok, checks


def summary_tables(df: pd.DataFrame) -> tuple:
    by_category = (
        df.groupby("category", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    focus_types = ["hospital", "clinic", "generator", "substation", "plant", "pumping_station", "water_works", "wastewater_plant", "fire_station", "police"]
    type_counts = df.groupby("type", dropna=False).size().reset_index(name="count")
    type_counts = type_counts.sort_values("count", ascending=False)
    focus = type_counts[type_counts["type"].isin(focus_types)].copy()

    by_source = (
        df.groupby(["source", "category"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["source", "count"], ascending=[True, False])
    )

    return by_category, focus, by_source


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 86)
    print("PUERTO RICO INFRA RETRY + HIFLD MERGE")
    print("=" * 86)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(BASE_FILE):
        raise FileNotFoundError(f"Base dataset not found: {BASE_FILE}")
    if not os.path.exists(FAILED_CHUNKS_FILE):
        raise FileNotFoundError(f"Failed chunk file not found: {FAILED_CHUNKS_FILE}")

    base_df = pd.read_csv(BASE_FILE)
    failed_df = pd.read_csv(FAILED_CHUNKS_FILE)

    endpoint = pick_overpass_endpoint()
    ox.settings.log_console = False
    ox.settings.use_cache = True

    retry_df, retry_stats = retry_failed_chunks(endpoint, failed_df)
    retry_df.to_csv(RETRY_FILE, index=False, encoding="utf-8")
    print(f"[SAVE] Retry dataset -> {RETRY_FILE} rows={len(retry_df)}")

    hifld_df = run_hifld_supplement()
    hifld_df.to_csv(HIFLD_FILE, index=False, encoding="utf-8")
    print(f"[SAVE] HIFLD dataset -> {HIFLD_FILE} rows={len(hifld_df)}")

    merged_df = merge_datasets(base_df, retry_df, hifld_df)

    # Update main dataset and also keep dedicated merged output
    merged_df.to_csv(MERGED_FILE, index=False, encoding="utf-8")
    merged_df.to_csv(BASE_FILE, index=False, encoding="utf-8")
    merged_df.to_csv(CRITICAL_FILE, index=False, encoding="utf-8")
    print(f"[SAVE] Merged dataset -> {MERGED_FILE} rows={len(merged_df)}")
    print(f"[SAVE] Main dataset updated -> {BASE_FILE}")

    ok, checks = verify_dataset(merged_df)

    print("-" * 86)
    print("[RETRY STATS]")
    print(f"retried_chunks={retry_stats['retried']} recovered_subchunks={retry_stats['recovered']} still_failed_chunks={retry_stats['still_failed']}")

    print("-" * 86)
    print("[QA]")
    print(f"rows={checks['count']} outside_bounds={checks['outside_bounds']} pumping_station={checks['pumping_station_count']} water_works={checks['water_works_count']}")
    for warning in checks["warnings"]:
        print(f"warning: {warning}")

    by_category, focus, by_source = summary_tables(merged_df)

    print("-" * 86)
    print("[SUMMARY] By Category")
    print(by_category.to_string(index=False))

    print("-" * 86)
    print("[SUMMARY] Key Types")
    print(focus.to_string(index=False))

    print("-" * 86)
    print("[SUMMARY] By Source and Category")
    print(by_source.to_string(index=False))

    print("=" * 86)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not ok:
        if checks.get("outside_examples"):
            print("[OUTSIDE EXAMPLES]")
            for item in checks["outside_examples"]:
                print(item)
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        traceback.print_exc()
        sys.exit(1)
