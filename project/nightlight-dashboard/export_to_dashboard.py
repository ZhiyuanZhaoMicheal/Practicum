"""
export_to_dashboard.py
======================
Converts Stage 2 notebook outputs into web dashboard data files.

PREREQUISITES (run in your project Python environment):
    pip install pandas geopandas rasterio pyproj joblib scikit-learn xgboost

INPUT — files this script reads from your project:
    data/result/stage2/pixel_panel.parquet     ← full pixel panel
    data/result/stage2/rf_final.pkl            ← trained RF model
    data/result/stage2/xgb_final.pkl           ← trained XGBoost model
    data/result/stage2/loeo_strict.csv         ← per-event LOEO results
    data/result/stage2/poi_cache/<event>_poi.csv  ← facility lat/lon per event
    data/processed/<event>-VNP46A2-pre/        ← pre-disaster daily GeoTIFFs
    data/processed/<event>-VNP46A2-post/       ← post-disaster daily GeoTIFFs

OUTPUT — files written to the dashboard's public/data/ folder:
    public/data/prob_<event_id>.geojson        ← pixel probability map
    public/data/ts_<event_id>.json             ← daily R_buffer / R_nonBuffer
    public/data/facilities_<event_id>.json     ← facility probability per POI
    public/data/loeo_results.json              ← LOEO AUC summary
    public/data/events_config.js               ← paste-ready events.js snippet

USAGE:
    python export_to_dashboard.py
    python export_to_dashboard.py --project /path/to/project --dashboard /path/to/dashboard
"""

import os, glob, json, argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy as tif_xy
from pyproj import Transformer
import joblib

# ──────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Map from notebook event_id → dashboard event_id
EVENT_ID_MAP = {
    "Maria_SanJuan":       "maria",
    "Irma_Miami":          "irma",
    "Ida_NewOrleans":      "ida",
    "Laura_LakeCharles":   "laura",
    "Michael_PanamaCity":  "michael",
    "Earthquake_SanJuan":  "eq-pr",
    "Ian_CharlotteHarbor": "ian-charlotte",
    "Ian_FortMyers":       "ian-fortmyers",
    "Earthquake_Hatay":    "eq-hatay",
}

# Metadata for the dashboard events.js — extend as needed
EVENT_META = {
    "Maria_SanJuan":       {"name": "Hurricane Maria",       "subtitle": "San Juan, Puerto Rico",      "year": 2017, "date": "Sep 20, 2017", "color": "#ff6b6b", "affectedUsers": "1.5M+",  "outageDuration": "11 months",  "description": "Category 5 hurricane causing the longest blackout in U.S. history."},
    "Irma_Miami":          {"name": "Hurricane Irma",        "subtitle": "Miami, Florida",              "year": 2017, "date": "Sep 10, 2017", "color": "#63b3ed", "affectedUsers": "4.4M+",  "outageDuration": "1–2 weeks",  "description": "Category 4 at Florida landfall, one of the strongest Atlantic hurricanes on record."},
    "Ida_NewOrleans":      {"name": "Hurricane Ida",         "subtitle": "New Orleans, Louisiana",      "year": 2021, "date": "Aug 29, 2021", "color": "#ff8c69", "affectedUsers": "1.0M+",  "outageDuration": "2–6 weeks",  "description": "Category 4 hurricane making landfall on the 16th anniversary of Katrina."},
    "Laura_LakeCharles":   {"name": "Hurricane Laura",       "subtitle": "Lake Charles, Louisiana",     "year": 2020, "date": "Aug 27, 2020", "color": "#ffcc5c", "affectedUsers": "300K+",  "outageDuration": "3–8 weeks",  "description": "Category 4 storm with record storm surge along Louisiana/Texas coast."},
    "Michael_PanamaCity":  {"name": "Hurricane Michael",     "subtitle": "Panama City, Florida",        "year": 2018, "date": "Oct 10, 2018", "color": "#ffa07a", "affectedUsers": "375K+",  "outageDuration": "2–4 weeks",  "description": "Category 5 landfall with 160 mph winds in the Florida Panhandle."},
    "Earthquake_SanJuan":  {"name": "Puerto Rico Earthquake","subtitle": "San Juan, Puerto Rico",       "year": 2020, "date": "Jan 7, 2020",  "color": "#c084fc", "affectedUsers": "950K+",  "outageDuration": "2–6 weeks",  "description": "M6.4 earthquake triggering island-wide blackout, the largest since Maria."},
    "Ian_CharlotteHarbor": {"name": "Hurricane Ian",         "subtitle": "Charlotte Harbor, Florida",   "year": 2022, "date": "Sep 28, 2022", "color": "#7ec8e3", "affectedUsers": "600K+",  "outageDuration": "2–3 weeks",  "description": "Category 4 storm causing catastrophic wind and storm surge damage."},
    "Ian_FortMyers":       {"name": "Hurricane Ian",         "subtitle": "Fort Myers, Florida",         "year": 2022, "date": "Sep 28, 2022", "color": "#5ab4d6", "affectedUsers": "2.6M+",  "outageDuration": "2–4 weeks",  "description": "Ian's direct hit on Fort Myers caused one of Florida's costliest disasters."},
    "Earthquake_Hatay":    {"name": "Turkey Earthquake",     "subtitle": "Hatay, Turkey",               "year": 2023, "date": "Feb 6, 2023",  "color": "#e879f9", "affectedUsers": "13.5M+", "outageDuration": "2–8 weeks",  "description": "Mw 7.8 earthquake devastating southern Turkey and northern Syria."},
}

# Disaster type for icon/color classification
DISASTER_TYPES = {
    "Maria_SanJuan": "hurricane", "Irma_Miami": "hurricane",
    "Ida_NewOrleans": "hurricane", "Laura_LakeCharles": "hurricane",
    "Michael_PanamaCity": "hurricane", "Earthquake_SanJuan": "earthquake",
    "Ian_CharlotteHarbor": "hurricane", "Ian_FortMyers": "hurricane",
    "Earthquake_Hatay": "earthquake",
}

# Facility type display mapping (notebook tag → dashboard label)
FAC_TYPE_MAP = {
    "hospital":     "hospital",
    "aerodrome":    "airport",
    "power_plant":  "power",
    "fire_station": "fire",
    "police":       "police",
}

# Buffer radii (metres) - must match notebook
BUFFER_RADIUS = {"aerodrome": 1250, "default": 750}

# Ensemble weights (must match notebook cell 26)
RF_WEIGHT  = 0.7
XGB_WEIGHT = 0.3

# Probability threshold: only export pixels above this to keep GeoJSON compact
MIN_PROB_FOR_GEOJSON = 0.05

# ──────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering — exact copy from notebook cell 13
# ──────────────────────────────────────────────────────────────────────────────

EXCLUDED_TYPES = ["government", "substation", "water_works"]

def engineer_features(df):
    """Mirror of notebook engineer_features() — must stay in sync."""
    df = df.copy()

    df["drop_magnitude"]    = -df["delta_ntl"].clip(upper=0)
    df["log_pre_ntl"]       = np.log1p(df["pre_mean_ntl"])
    df["log_post_ntl"]      = np.log1p(df["post_mean_ntl"])
    df["log_city_pre_mean"] = np.log1p(df["city_pre_mean"])
    df["ntl_relative"]      = df["pre_mean_ntl"] / (df["city_pre_mean"] + 1e-6)
    df["log_dist"]          = np.log1p(df["dist_to_facility"])

    df["near_fire_station"] = (df["nearest_fac_type"] == "fire_station").astype(np.uint8)
    df["near_police"]       = (df["nearest_fac_type"] == "police").astype(np.uint8)
    df["near_excluded"]     = df["nearest_fac_type"].isin(EXCLUDED_TYPES).astype(np.uint8)
    df["fac_group"]         = df["nearest_fac_group"].astype(np.float32)

    df["city_size_code"] = df["city_size"].map(
        {"large": 0, "medium": 1, "small": 2}).fillna(1)
    df["is_hurricane"]   = (df["disaster_type"] == "hurricane").astype(np.uint8)
    df["is_earthquake"]  = (df["disaster_type"] == "earthquake").astype(np.uint8)

    df["ntl_x_group"] = df["log_pre_ntl"] * df["fac_group"]

    city_median = df.groupby("event_id")["pre_mean_ntl"].transform("median")
    df["below_city_median"]   = (df["pre_mean_ntl"] < city_median).astype(np.uint8)
    df["below_median_x_group"] = df["below_city_median"] * df["fac_group"]

    features = [
        "drop_magnitude", "delta_ntl",
        "log_pre_ntl", "log_post_ntl",
        "log_city_pre_mean", "ntl_relative",
        "log_dist",
        "near_fire_station", "near_police",
        "near_excluded", "fac_group",
        "city_size_code", "is_hurricane", "is_earthquake",
        "ntl_x_group",
        "below_city_median", "below_median_x_group",
    ]
    return df, features


# ──────────────────────────────────────────────────────────────────────────────
# 3. Helper: get lat/lon for panel rows using the event's GeoTIFF
# ──────────────────────────────────────────────────────────────────────────────

def get_pixel_lonlat(event_df, tif_path):
    """
    Convert (row, col) pixel indices to WGS84 (lon, lat) using the GeoTIFF's
    affine transform. Returns two arrays: lons, lats.
    """
    rows = event_df["row"].values
    cols = event_df["col"].values

    with rasterio.open(tif_path) as src:
        transform = src.transform
        tif_crs   = src.crs

    xs, ys = tif_xy(transform, rows, cols, offset="center")
    xs, ys = np.array(xs), np.array(ys)

    # Reproject to WGS84 if the TIF is in a projected CRS
    if str(tif_crs).upper() not in ("EPSG:4326", "WGS 84", "WGS84"):
        transformer = Transformer.from_crs(tif_crs, "EPSG:4326", always_xy=True)
        lons, lats  = transformer.transform(xs, ys)
    else:
        lons, lats = xs, ys

    return lons.astype(np.float32), lats.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 4. GeoJSON probability map export
# ──────────────────────────────────────────────────────────────────────────────

def export_prob_geojson(event_nb_id, event_df, prob_arr, lons, lats, out_dir):
    """
    Write a GeoJSON FeatureCollection with one Point per pixel.
    Each feature has properties: probability, inBuffer, facType.
    Only pixels with probability > MIN_PROB_FOR_GEOJSON are included
    to keep file size manageable.
    """
    dash_id = EVENT_ID_MAP.get(event_nb_id, event_nb_id.lower())

    features = []
    in_buf   = event_df["in_buffer_strict"].values
    fac_type = event_df["nearest_fac_type"].values

    mask = prob_arr > MIN_PROB_FOR_GEOJSON
    for lon, lat, prob, buf, ft in zip(
            lons[mask], lats[mask], prob_arr[mask],
            in_buf[mask], fac_type[mask]):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(lon), 5), round(float(lat), 5)],
            },
            "properties": {
                "probability": round(float(prob), 3),
                "inBuffer":    bool(buf),
                "facType":     FAC_TYPE_MAP.get(str(ft), str(ft)),
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}

    out_path = os.path.join(out_dir, f"prob_{dash_id}.geojson")
    with open(out_path, "w") as f:
        json.dump(geojson, f, separators=(",", ":"))  # compact JSON

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  GeoJSON: {out_path}  ({len(features):,} pixels, {size_kb:.0f} KB)")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 5. Time-series export: daily R_buffer / R_nonBuffer
# ──────────────────────────────────────────────────────────────────────────────

def load_daily_tifs(folder, data_dir):
    """
    Load all daily GeoTIFFs in a folder and return a dict:
    {filename_stem → 2D ndarray of NTL values}
    Files are expected to be named with their date so they sort chronologically.
    """
    tif_files = sorted(glob.glob(os.path.join(data_dir, folder, "*.tif")))
    if not tif_files:
        return {}
    daily = {}
    for f in tif_files:
        with rasterio.open(f) as src:
            arr = src.read(1).astype(np.float32)
            arr[arr <= 0] = np.nan  # zero = nodata
        daily[os.path.splitext(os.path.basename(f))[0]] = arr
    return daily


def export_time_series(event_nb_id, cfg, event_df, data_dir, out_dir):
    """
    For each daily pre/post TIF, compute:
      - BAU = mean over all pre-disaster TIFs (per pixel)
      - Daily R = NTL_day / BAU
      - R_buffer    = mean(R) for in_buffer_strict == 1 pixels
      - R_nonBuffer = mean(R) for in_buffer_strict == 0 pixels

    Writes: public/data/ts_<dash_id>.json
    """
    dash_id   = EVENT_ID_MAP.get(event_nb_id, event_nb_id.lower())
    rows      = event_df["row"].values
    cols      = event_df["col"].values
    in_buf    = event_df["in_buffer_strict"].values.astype(bool)
    buf_mask  = in_buf
    non_mask  = ~in_buf

    print(f"  Loading pre-TIFs for {event_nb_id} ...")
    pre_daily  = load_daily_tifs(cfg["pre"],  data_dir)
    post_daily = load_daily_tifs(cfg["post"], data_dir)

    if not pre_daily:
        print(f"  [SKIP] No pre-TIFs found for {event_nb_id}")
        return None

    # BAU = pixel-level mean over all pre-disaster days
    pre_stack = np.stack([arr[rows, cols] for arr in pre_daily.values()], axis=0)
    bau       = np.nanmean(pre_stack, axis=0)               # shape: (n_pixels,)
    bau       = np.where(bau > 0, bau, np.nan)              # avoid div-by-zero

    # Compute daily ratios for pre days (negative relative days)
    n_pre  = len(pre_daily)
    series = []

    for day_idx, (stem, arr) in enumerate(pre_daily.items()):
        ntl     = arr[rows, cols]
        R       = np.where(bau > 0, ntl / bau, np.nan)
        day_rel = day_idx - n_pre                            # e.g. -30, -29, ..., -1

        R_buf = float(np.nanmean(R[buf_mask]))  if buf_mask.any()  else float("nan")
        R_non = float(np.nanmean(R[non_mask]))  if non_mask.any()  else float("nan")

        series.append({
            "day": int(day_rel),
            "R_buffer":    round(R_buf, 4) if not np.isnan(R_buf) else None,
            "R_nonBuffer": round(R_non, 4) if not np.isnan(R_non) else None,
            "isPostDisaster": False,
        })

    # Post-disaster days (positive relative days)
    for day_idx, (stem, arr) in enumerate(post_daily.items()):
        ntl     = arr[rows, cols]
        R       = np.where(bau > 0, ntl / bau, np.nan)
        day_rel = day_idx + 1                                # +1, +2, ...

        R_buf = float(np.nanmean(R[buf_mask]))  if buf_mask.any()  else float("nan")
        R_non = float(np.nanmean(R[non_mask]))  if non_mask.any()  else float("nan")

        series.append({
            "day": int(day_rel),
            "R_buffer":    round(R_buf, 4) if not np.isnan(R_buf) else None,
            "R_nonBuffer": round(R_non, 4) if not np.isnan(R_non) else None,
            "isPostDisaster": True,
        })

    # Sort by day (should already be sorted, but be safe)
    series.sort(key=lambda x: x["day"])

    out_path = os.path.join(out_dir, f"ts_{dash_id}.json")
    with open(out_path, "w") as f:
        json.dump(series, f, separators=(",", ":"))
    print(f"  TimeSeries: {out_path}  ({len(series)} days)")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 6. Facility probability export
# ──────────────────────────────────────────────────────────────────────────────

def export_facility_probs(event_nb_id, event_df, prob_arr, poi_cache_dir, out_dir,
                          lons=None, lats=None):
    """
    For each facility POI, compute the mean predicted probability
    within its buffer zone. Also includes the POI's lat/lon for the
    events.js facilities array.

    Only includes POIs within the study area bounding box (from pixel lons/lats).
    Writes: public/data/facilities_<dash_id>.json
    """
    dash_id   = EVENT_ID_MAP.get(event_nb_id, event_nb_id.lower())
    poi_path  = os.path.join(poi_cache_dir, f"{event_nb_id}_poi.csv")

    if not os.path.exists(poi_path):
        print(f"  [WARN] POI cache not found: {poi_path} — skipping facility export")
        return []

    poi_df = pd.read_csv(poi_path)
    if poi_df.empty:
        return []

    # Required columns: facility_type, lon, lat
    required = {"facility_type", "lon", "lat"}
    if not required.issubset(poi_df.columns):
        print(f"  [WARN] POI CSV missing columns {required - set(poi_df.columns)}")
        return []

    # Filter POIs to heatmap bounding box (prob > threshold pixels only)
    # Use the probability array + coordinates to get the visible heatmap bounds
    if lons is not None and lats is not None:
        vis_mask = prob_arr > MIN_PROB_FOR_GEOJSON
        if vis_mask.any():
            vis_lons = lons[vis_mask]
            vis_lats = lats[vis_mask]
            pad = 0.005  # ~500m padding
            lon_min, lon_max = float(vis_lons.min()) - pad, float(vis_lons.max()) + pad
            lat_min, lat_max = float(vis_lats.min()) - pad, float(vis_lats.max()) + pad
        else:
            pad = 0.01
            lon_min, lon_max = float(lons.min()) - pad, float(lons.max()) + pad
            lat_min, lat_max = float(lats.min()) - pad, float(lats.max()) + pad
        before = len(poi_df)
        poi_df = poi_df[
            (poi_df["lon"] >= lon_min) & (poi_df["lon"] <= lon_max) &
            (poi_df["lat"] >= lat_min) & (poi_df["lat"] <= lat_max)
        ]
        print(f"  Filtered POIs to heatmap bounds: {before} → {len(poi_df)}")

    # Fix "nan" names
    if "name" in poi_df.columns:
        poi_df["name"] = poi_df["name"].fillna("").astype(str)
        poi_df.loc[poi_df["name"].isin(["nan", ""]), "name"] = \
            poi_df.loc[poi_df["name"].isin(["nan", ""]), "facility_type"] + " facility"

    # Attach OOF/model probability to each pixel
    event_df = event_df.copy()
    event_df["_prob"] = prob_arr

    facilities = []
    for _, fac in poi_df.iterrows():
        fac_type  = str(fac["facility_type"])
        radius_m  = BUFFER_RADIUS.get(fac_type, BUFFER_RADIUS["default"])

        # Mean probability for pixels whose nearest facility is this type
        # AND are in the buffer.
        in_type_buf = (
            (event_df["nearest_fac_type"] == fac_type) &
            (event_df["in_buffer_strict"] == 1)
        )
        mean_prob = (
            float(event_df.loc[in_type_buf, "_prob"].mean())
            if in_type_buf.any() else float("nan")
        )

        name = str(fac.get("name", f"{fac_type} facility"))
        if name in ("nan", ""):
            name = f"{fac_type} facility"

        facilities.append({
            "name":        name,
            "type":        FAC_TYPE_MAP.get(fac_type, fac_type),
            "coords":      [round(float(fac["lon"]), 5), round(float(fac["lat"]), 5)],
            "probability": round(mean_prob, 3) if not np.isnan(mean_prob) else 0.5,
            "radiusM":     radius_m,
        })

    out_path = os.path.join(out_dir, f"facilities_{dash_id}.json")
    with open(out_path, "w") as f:
        json.dump(facilities, f, indent=2)
    print(f"  Facilities: {out_path}  ({len(facilities)} POIs)")
    return facilities


# ──────────────────────────────────────────────────────────────────────────────
# 7. LOEO results export
# ──────────────────────────────────────────────────────────────────────────────

def export_loeo_results(stage2_dir, out_dir):
    loeo_path = os.path.join(stage2_dir, "loeo_strict.csv")
    if not os.path.exists(loeo_path):
        print(f"  [WARN] loeo_strict.csv not found — skipping")
        return

    loeo = pd.read_csv(loeo_path)
    # Rename held_out column to use dashboard IDs
    if "held_out" in loeo.columns:
        loeo["event_id"] = loeo["held_out"].map(
            lambda x: EVENT_ID_MAP.get(x, x.lower()))

    result = loeo.to_dict(orient="records")
    out_path = os.path.join(out_dir, "loeo_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  LOEO: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Generate updated events_config.js snippet
# ──────────────────────────────────────────────────────────────────────────────

def generate_events_config(all_facilities_by_event, out_dir):
    """
    Writes a paste-ready events_config.js that you can drop into
    src/data/events.js, replacing the EVENTS array's facilities and
    adding real coordinates.
    """
    lines = [
        "// =====================================================",
        "// AUTO-GENERATED by export_to_dashboard.py",
        "// Paste this EVENTS array into src/data/events.js",
        "// =====================================================",
        "",
        "export const EVENTS = [",
    ]

    for nb_id, facilities in all_facilities_by_event.items():
        dash_id  = EVENT_ID_MAP.get(nb_id, nb_id.lower())
        meta     = EVENT_META.get(nb_id, {})
        dis_type = DISASTER_TYPES.get(nb_id, "hurricane")

        # Compute center as mean of facility coords
        if facilities:
            center_lon = round(float(np.mean([f["coords"][0] for f in facilities])), 4)
            center_lat = round(float(np.mean([f["coords"][1] for f in facilities])), 4)
        else:
            center_lon, center_lat = 0.0, 0.0

        lines.append(f"  {{")
        lines.append(f"    id:              '{dash_id}',")
        lines.append(f"    name:            '{meta.get('name', nb_id)}',")
        lines.append(f"    subtitle:        '{meta.get('subtitle', '')}',")
        lines.append(f"    year:            {meta.get('year', 2020)},")
        lines.append(f"    date:            '{meta.get('date', '')}',")
        lines.append(f"    type:            '{dis_type}',")
        lines.append(f"    center:          [{center_lon}, {center_lat}],")
        lines.append(f"    zoom:            12,")
        lines.append(f"    color:           '{meta.get('color', '#00d4ff')}',")
        lines.append(f"    affectedUsers:   '{meta.get('affectedUsers', 'N/A')}',")
        lines.append(f"    outageDuration:  '{meta.get('outageDuration', 'N/A')}',")
        lines.append(f"    description:     '{meta.get('description', '')}',")
        lines.append(f"    // Data files in public/data/:")
        lines.append(f"    probGeoJSON:     '/data/prob_{dash_id}.geojson',")
        lines.append(f"    timeSeries:      '/data/ts_{dash_id}.json',")
        lines.append(f"    facilities: [")

        for fac in facilities:
            fac_str = (
                f"      {{ name: '{fac['name']}', type: '{fac['type']}', "
                f"coords: {fac['coords']}, probability: {fac['probability']} }},"
            )
            lines.append(fac_str)

        lines.append(f"    ],")
        lines.append(f"  }},")

    lines.append("]")
    lines.append("")

    out_path = os.path.join(out_dir, "events_config.js")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  events_config.js → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Main
# ──────────────────────────────────────────────────────────────────────────────

def main(project_root, dashboard_root):
    stage2_dir     = os.path.join(project_root, "data", "result", "stage2")
    data_dir       = os.path.join(project_root, "data", "processed")
    poi_cache_dir  = os.path.join(stage2_dir, "poi_cache")
    out_dir        = os.path.join(dashboard_root, "public", "data")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load panel ──────────────────────────────────────────────────────────
    panel_path = os.path.join(stage2_dir, "pixel_panel.parquet")
    print(f"\n[1/6] Loading panel: {panel_path}")
    df = pd.read_parquet(panel_path)
    print(f"      {len(df):,} rows, events: {df['event_id'].unique().tolist()}")

    # ── Engineer features ───────────────────────────────────────────────────
    print("[2/6] Engineering features ...")
    df, features = engineer_features(df)

    # ── Load models ─────────────────────────────────────────────────────────
    print("[3/6] Loading trained models ...")
    rf_path  = os.path.join(stage2_dir, "rf_final.pkl")
    xgb_path = os.path.join(stage2_dir, "xgb_final.pkl")

    if not os.path.exists(rf_path):
        raise FileNotFoundError(
            f"rf_final.pkl not found at {rf_path}\n"
            "Run the notebook through Cell 23 (Section 6) first."
        )
    rf_model  = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)
    print(f"      RF  loaded: {rf_path}")
    print(f"      XGB loaded: {xgb_path}")

    # ── Get notebook EVENTS config for pre/post folder names ────────────────
    # We reconstruct the minimal EVENTS dict needed for TIF folder access.
    # This mirrors notebook cell 5.
    EVENTS_CFG = {}
    for nb_id in df["event_id"].unique():
        # Try to infer pre/post folder names from data_dir
        # Convention: <Event>-VNP46A2-pre / <Event>-VNP46A2-post
        candidates = [
            (f"{nb_id}-VNP46A2-pre",  f"{nb_id}-VNP46A2-post"),
            (f"{nb_id}_pre",           f"{nb_id}_post"),
        ]
        for pre_dir, post_dir in candidates:
            if os.path.isdir(os.path.join(data_dir, pre_dir)):
                EVENTS_CFG[nb_id] = {"pre": pre_dir, "post": post_dir}
                break
        if nb_id not in EVENTS_CFG:
            # Last resort: try to find any folder with the event name
            matches = [d for d in os.listdir(data_dir)
                       if nb_id.split("_")[0].lower() in d.lower() and "pre" in d.lower()]
            if matches:
                pre_dir  = matches[0]
                post_dir = pre_dir.replace("pre", "post")
                EVENTS_CFG[nb_id] = {"pre": pre_dir, "post": post_dir}
            else:
                print(f"  [WARN] Could not find TIF folders for {nb_id}")
                EVENTS_CFG[nb_id] = {"pre": None, "post": None}

    # ── Export LOEO results ─────────────────────────────────────────────────
    print("[4/6] Exporting LOEO results ...")
    export_loeo_results(stage2_dir, out_dir)

    # ── Per-event export ────────────────────────────────────────────────────
    print("[5/6] Exporting per-event data ...")
    all_facilities = {}

    for nb_id in df["event_id"].unique():
        print(f"\n  ── {nb_id} ──")
        event_df  = df[df["event_id"] == nb_id].copy().reset_index(drop=True)
        X         = event_df[features].fillna(0).values

        # Ensemble prediction (RF×0.7 + XGB×0.3, matching notebook)
        rf_prob   = rf_model.predict_proba(X)[:, 1]
        xgb_prob  = xgb_model.predict_proba(X)[:, 1]
        ens_prob  = rf_prob * RF_WEIGHT + xgb_prob * XGB_WEIGHT

        # Get lat/lon from GeoTIFF
        cfg      = EVENTS_CFG.get(nb_id, {})
        pre_dir  = cfg.get("pre")
        tif_path = None
        if pre_dir:
            tifs = sorted(glob.glob(os.path.join(data_dir, pre_dir, "*.tif")))
            if tifs:
                tif_path = tifs[0]

        if tif_path:
            lons, lats = get_pixel_lonlat(event_df, tif_path)
        else:
            print(f"  [WARN] No TIF for coord conversion — skipping GeoJSON for {nb_id}")
            lons = lats = None

        # GeoJSON probability map
        if lons is not None:
            export_prob_geojson(nb_id, event_df, ens_prob, lons, lats, out_dir)

        # Time-series
        if pre_dir and os.path.isdir(os.path.join(data_dir, pre_dir)):
            export_time_series(nb_id, cfg, event_df, data_dir, out_dir)
        else:
            print(f"  [SKIP] TIF folders not found for time-series")

        # Facility probabilities
        facilities = export_facility_probs(
            nb_id, event_df, ens_prob, poi_cache_dir, out_dir,
            lons=lons, lats=lats)
        all_facilities[nb_id] = facilities

    # ── Generate events_config.js ───────────────────────────────────────────
    print("\n[6/6] Generating events_config.js ...")
    generate_events_config(all_facilities, out_dir)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Export complete. Files written to:")
    for f in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:<45s} {size:8.1f} KB")
    print(f"{'='*60}")
    print("""
NEXT STEPS:
  1. Copy public/data/ into the dashboard's public/ folder
  2. Open public/data/events_config.js and paste its EVENTS array
     into src/data/events.js (replace the existing EVENTS export)
  3. In MapView.vue, replace generateMockProbabilityGeoJSON() with:
       fetch(`/data/prob_${ev.id}.geojson`).then(r => r.json())
  4. In timeseries.js, replace generateRecoverySeries() with:
       fetch(`/data/ts_${ev.id}.json`).then(r => r.json())
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Stage 2 outputs to dashboard")
    parser.add_argument(
        "--project",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        help="Path to project root (contains data/result/stage2/)",
    )
    parser.add_argument(
        "--dashboard",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                              "nightlight-dashboard")),
        help="Path to Vite dashboard root (contains public/)",
    )
    args = parser.parse_args()

    print(f"Project root : {args.project}")
    print(f"Dashboard    : {args.dashboard}")
    main(args.project, args.dashboard)
