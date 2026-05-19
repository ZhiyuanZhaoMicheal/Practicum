from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pyproj import Transformer
from rasterio.enums import MergeAlg, Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy.spatial import cKDTree

import statsmodels.formula.api as smf
from sklearn.metrics import auc, roc_curve
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT / "project"
MODELING_DIR = PROJECT_DIR / "modeling"
PIXEL_DIR = MODELING_DIR / "pixel_data"
OUTPUT_DIR = MODELING_DIR / "output"
REPORT_DIR = PROJECT_DIR / "modeling_report"
FIG_DIR = REPORT_DIR / "figures"
TRACKING_DIR = PROJECT_DIR / "modeling_tracking"

CONFIG_EVENTS = MODELING_DIR / "config" / "events_6.json"
CONFIG_DEFAULTS = MODELING_DIR / "config" / "model_defaults.json"

PANEL_PATH = PIXEL_DIR / "all_events_pixel_panel_v1.parquet"
PANEL_NLCD_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_with_nlcd.parquet"
RECOVERY_PATH = PIXEL_DIR / "recovery_daily_panel_v1.parquet"
CLOUD_SUMMARY_PATH = OUTPUT_DIR / "cloud_feature_summary.csv"

BUFFER_RADII = {"aerodrome": 1250}
DEFAULT_BUFFER = 750


@dataclass
class RunContext:
    issues: List[Dict[str, object]]


# ----------------------------
# Utility and logging
# ----------------------------

def ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def ensure_directories() -> None:
    paths = [
        PIXEL_DIR,
        OUTPUT_DIR,
        REPORT_DIR,
        FIG_DIR / "ols",
        FIG_DIR / "mixedlm",
        FIG_DIR / "logit",
        FIG_DIR / "cox",
        TRACKING_DIR / "progress_record",
        TRACKING_DIR / "future_plan",
    ]
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def append_progress(message: str) -> None:
    progress_file = TRACKING_DIR / "progress_record" / "00_bootstrap.md"
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    if not progress_file.exists():
        progress_file.write_text("# Modeling Pipeline Progress\n\n", encoding="utf-8")
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts()}] {message}\n")


def init_tracking_files() -> None:
    bootstrap = TRACKING_DIR / "progress_record" / "00_bootstrap.md"
    issue_log = TRACKING_DIR / "progress_record" / "issue_log.md"
    roadmap = TRACKING_DIR / "future_plan" / "00_roadmap.md"

    if not bootstrap.exists():
        bootstrap.write_text(
            "# Modeling Bootstrap Log\n\n"
            "## Scope\n"
            "- Branch: modeling-6events\n"
            "- Data line: teammate/main six-event pipeline\n"
            "- Report language: Chinese main text + English technical terms\n\n"
            "## Execution Log\n",
            encoding="utf-8",
        )
    if not issue_log.exists():
        issue_log.write_text(
            "# Modeling Issue Log\n\n"
            "All issues here correspond to `project/modeling/output/model_issue_log.csv`.\n",
            encoding="utf-8",
        )
    if not roadmap.exists():
        roadmap.write_text(
            "# Future Roadmap\n\n"
            "## Completed in this round\n"
            "- Baseline pixel panel and 4-model chain\n"
            "- Figures and bilingual reports\n"
            "- Robustness loops\n\n"
            "## Planned next\n"
            "- NLCD export/import completion if missing\n"
            "- BEAST as robustness extension\n"
            "- Spatial autocorrelation diagnostics\n",
            encoding="utf-8",
        )


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def log_issue(
    ctx: RunContext,
    stage: str,
    model: str,
    event_id: str,
    issue_type: str,
    symptom: str,
    fix_action: str,
    impact: str,
    status: str,
) -> None:
    ctx.issues.append(
        {
            "stage": stage,
            "model": model,
            "event_id": event_id,
            "issue_type": issue_type,
            "symptom": symptom,
            "fix_action": fix_action,
            "impact": impact,
            "status": status,
        }
    )


def save_issue_log(ctx: RunContext) -> None:
    out_csv = OUTPUT_DIR / "model_issue_log.csv"
    if ctx.issues:
        pd.DataFrame(ctx.issues).to_csv(out_csv, index=False)
    else:
        pd.DataFrame(
            columns=[
                "stage",
                "model",
                "event_id",
                "issue_type",
                "symptom",
                "fix_action",
                "impact",
                "status",
            ]
        ).to_csv(out_csv, index=False)

    md_file = TRACKING_DIR / "progress_record" / "issue_log.md"
    with md_file.open("a", encoding="utf-8") as f:
        f.write(f"\n## Update {ts()}\n")
        if not ctx.issues:
            f.write("- No critical issue observed in this run.\n")
        else:
            for it in ctx.issues:
                f.write(
                    "- "
                    f"[{it['model']}] {it['event_id']} | {it['issue_type']} | "
                    f"symptom={it['symptom']} | fix={it['fix_action']} | impact={it['impact']} | status={it['status']}\n"
                )


# ----------------------------
# Data building
# ----------------------------

def _parse_date_from_name(path: Path) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    return m.group(1) if m else path.name


def list_daily_tifs(directory: Path) -> List[Path]:
    tifs = [p for p in directory.glob("*.tif") if "composite" not in p.name.lower()]
    return sorted(tifs, key=lambda x: _parse_date_from_name(x))


def _find_cloud_screening_file(event_id: str) -> Optional[Path]:
    script_dir = PROJECT_DIR / "script"
    if not script_dir.exists():
        return None

    candidates = {
        "maria_sanjuan": [
            "maria_sanjuan_cloud_screening.csv",
            "hurricane_maria_sanjuan_cloud_screening.csv",
        ],
        "michael_panamacity": [
            "michael_panamacity_cloud_screening.csv",
            "Michael_FL_cloud_screening.csv",
        ],
        "earthquake_sanjuan": [
            "earthquake_sanjuan_cloud_screening.csv",
            "Earthquake_sanjuan_cloud_screening.csv",
        ],
        "ida_neworleans": [
            "ida_neworleans_cloud_screening.csv",
            "hurricane_ida_neworleans_cloud_screening.csv",
        ],
        "laura_lakecharles": [
            "laura_lakecharles_cloud_screening.csv",
        ],
        "irma_miami": [
            "irma_miami_cloud_screening.csv",
        ],
    }
    for name in candidates.get(event_id, []):
        p = script_dir / name
        if p.exists():
            return p
    return None


def attach_cloud_features(
    ctx: RunContext,
    panel_path: Path = PANEL_PATH,
    output_path: Path = PANEL_PATH,
) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path).copy()

    feature_cols = [
        "pre_valid_ratio",
        "post_valid_ratio",
        "cloud_pre_mean",
        "cloud_post_mean",
        "cloud_window_mean",
        "missing_weather_flag",
    ]
    for col in feature_cols:
        if col not in panel.columns:
            panel[col] = np.nan

    events_cfg = load_json(CONFIG_EVENTS)
    summary_rows: List[Dict[str, object]] = []

    for event_id in events_cfg.keys():
        event_mask = panel["event_id"] == event_id
        if event_mask.sum() == 0:
            continue

        csv_path = _find_cloud_screening_file(event_id)
        if csv_path is None:
            panel.loc[event_mask, "missing_weather_flag"] = 1
            summary_rows.append(
                {
                    "event_id": event_id,
                    "cloud_file": "",
                    "n_rows": 0,
                    "pre_valid_ratio": np.nan,
                    "post_valid_ratio": np.nan,
                    "cloud_pre_mean": np.nan,
                    "cloud_post_mean": np.nan,
                    "cloud_window_mean": np.nan,
                    "missing_weather_flag": 1,
                }
            )
            log_issue(
                ctx,
                stage="attach_cloud_features",
                model="all",
                event_id=event_id,
                issue_type="missing_cloud_screening",
                symptom="No cloud screening CSV found under project/script",
                fix_action="set missing_weather_flag=1 and continue",
                impact="weather controls unavailable for this event",
                status="monitor",
            )
            continue

        df = pd.read_csv(csv_path)
        if "period" not in df.columns or "cloud_fraction" not in df.columns:
            panel.loc[event_mask, "missing_weather_flag"] = 1
            summary_rows.append(
                {
                    "event_id": event_id,
                    "cloud_file": str(csv_path),
                    "n_rows": len(df),
                    "pre_valid_ratio": np.nan,
                    "post_valid_ratio": np.nan,
                    "cloud_pre_mean": np.nan,
                    "cloud_post_mean": np.nan,
                    "cloud_window_mean": np.nan,
                    "missing_weather_flag": 1,
                }
            )
            log_issue(
                ctx,
                stage="attach_cloud_features",
                model="all",
                event_id=event_id,
                issue_type="malformed_cloud_screening",
                symptom=f"Missing required columns in {csv_path.name}",
                fix_action="set missing_weather_flag=1 and continue",
                impact="weather controls unavailable for this event",
                status="monitor",
            )
            continue

        work = df.copy()
        work["period"] = work["period"].astype(str).str.lower().str.strip()
        if "usable" in work.columns:
            usable = work["usable"].astype(str).str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0}
            )
        else:
            usable = pd.Series(np.nan, index=work.index)

        pre = work[work["period"] == "pre"]
        post = work[work["period"] == "post"]

        pre_valid_ratio = float(usable.loc[pre.index].mean()) if len(pre) else np.nan
        post_valid_ratio = float(usable.loc[post.index].mean()) if len(post) else np.nan
        cloud_pre_mean = float(pre["cloud_fraction"].mean()) if len(pre) else np.nan
        cloud_post_mean = float(post["cloud_fraction"].mean()) if len(post) else np.nan
        cloud_window_mean = float(work["cloud_fraction"].mean()) if len(work) else np.nan

        panel.loc[event_mask, "pre_valid_ratio"] = pre_valid_ratio
        panel.loc[event_mask, "post_valid_ratio"] = post_valid_ratio
        panel.loc[event_mask, "cloud_pre_mean"] = cloud_pre_mean
        panel.loc[event_mask, "cloud_post_mean"] = cloud_post_mean
        panel.loc[event_mask, "cloud_window_mean"] = cloud_window_mean
        panel.loc[event_mask, "missing_weather_flag"] = 0

        summary_rows.append(
            {
                "event_id": event_id,
                "cloud_file": str(csv_path),
                "n_rows": len(work),
                "pre_valid_ratio": pre_valid_ratio,
                "post_valid_ratio": post_valid_ratio,
                "cloud_pre_mean": cloud_pre_mean,
                "cloud_post_mean": cloud_post_mean,
                "cloud_window_mean": cloud_window_mean,
                "missing_weather_flag": 0,
            }
        )

    panel.to_parquet(output_path, index=False)
    pd.DataFrame(summary_rows).to_csv(CLOUD_SUMMARY_PATH, index=False)
    append_progress(
        f"Attached cloud features to panel. Summary saved to {CLOUD_SUMMARY_PATH.relative_to(ROOT)}"
    )
    return panel


def _read_stack_mean(paths: Sequence[Path]) -> Tuple[np.ndarray, rasterio.Affine, str]:
    if not paths:
        raise FileNotFoundError("No GeoTIFF found for stack mean.")

    arrays: List[np.ndarray] = []
    transform = None
    crs = None
    shape = None
    for p in paths:
        with rasterio.open(p) as src:
            arr = src.read(1).astype("float64")
            if shape is None:
                shape = arr.shape
                transform = src.transform
                crs = src.crs.to_string() if src.crs else "EPSG:4326"
            elif arr.shape != shape:
                raise ValueError(f"Shape mismatch in {p}")
            arrays.append(arr)
    stack = np.stack(arrays, axis=0)
    mean_arr = np.nanmean(stack, axis=0)
    return mean_arr, transform, crs


def _pixel_grid(transform: rasterio.Affine, height: int, width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.indices((height, width))
    rows_f = rows.ravel()
    cols_f = cols.ravel()
    lons, lats = rasterio.transform.xy(transform, rows_f, cols_f, offset="center")
    return rows_f, cols_f, np.array(lons), np.array(lats)


def standardize_facility_type(raw_type: str) -> str:
    t = (raw_type or "").strip().lower()
    if t == "plant":
        return "power_plant"
    return t


def _load_poi(csv_path: Path, exclude_types: Optional[Sequence[str]] = None) -> pd.DataFrame:
    poi = pd.read_csv(csv_path)
    if "type" not in poi.columns:
        raise KeyError(f"POI file missing `type`: {csv_path}")
    required = {"lat", "lon"}
    miss = [c for c in required if c not in poi.columns]
    if miss:
        raise KeyError(f"POI file missing columns {miss}: {csv_path}")

    poi = poi.copy()
    poi["facility_type_raw"] = poi["type"].astype(str).str.lower().str.strip()
    poi["facility_type_std"] = poi["facility_type_raw"].map(standardize_facility_type)
    if exclude_types:
        ex = {x.strip().lower() for x in exclude_types}
        poi = poi[~poi["facility_type_std"].isin(ex)].copy()
    return poi


def _mask_and_distance(
    rows: np.ndarray,
    cols: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    transform: rasterio.Affine,
    shape: Tuple[int, int],
    raster_crs: str,
    poi_df: pd.DataFrame,
    metric_crs: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gdf = gpd.GeoDataFrame(
        poi_df[["facility_type_std"]].copy(),
        geometry=gpd.points_from_xy(poi_df["lon"], poi_df["lat"]),
        crs="EPSG:4326",
    )
    gdf_m = gdf.to_crs(metric_crs)
    gdf_m["buffer_radius"] = np.where(gdf_m["facility_type_std"] == "aerodrome", BUFFER_RADII["aerodrome"], DEFAULT_BUFFER)
    gdf_m["buffer_geom"] = gdf_m.geometry.buffer(gdf_m["buffer_radius"])

    gdf_buffer_wgs = gpd.GeoDataFrame(geometry=gdf_m["buffer_geom"], crs=metric_crs).to_crs(raster_crs)
    shapes_binary = [(geom, 1) for geom in gdf_buffer_wgs.geometry if geom is not None and not geom.is_empty]
    shapes_count = [(geom, 1) for geom in gdf_buffer_wgs.geometry if geom is not None and not geom.is_empty]

    in_buffer = rasterize(
        shapes_binary,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).ravel()

    n_in_buffer = rasterize(
        shapes_count,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint16",
        all_touched=True,
        merge_alg=MergeAlg.add,
    ).ravel()

    poi_xy = np.column_stack([gdf_m.geometry.x.values, gdf_m.geometry.y.values])
    tree = cKDTree(poi_xy)

    transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    xpix, ypix = transformer.transform(lons, lats)
    pix_xy = np.column_stack([xpix, ypix])
    dist, idx = tree.query(pix_xy, k=1)
    nearest_types = gdf_m["facility_type_std"].iloc[idx].to_numpy()

    return in_buffer, n_in_buffer, nearest_types, dist


def build_pixel_panel(
    ctx: RunContext,
    pre_threshold: float,
    damage_threshold: float,
    exclude_types: Optional[Sequence[str]] = None,
    output_path: Path = PANEL_PATH,
) -> pd.DataFrame:
    events_cfg = load_json(CONFIG_EVENTS)
    rows_all: List[pd.DataFrame] = []
    event_stats: List[Dict[str, object]] = []

    for event_id, cfg in events_cfg.items():
        pre_dir = ROOT / cfg["pre_dir"]
        post_dir = ROOT / cfg["post_dir"]
        poi_csv = ROOT / cfg["poi_csv"]
        metric_crs = cfg["metric_crs"]

        pre_tifs = list_daily_tifs(pre_dir)
        post_tifs = list_daily_tifs(post_dir)

        if not pre_tifs or not post_tifs:
            log_issue(
                ctx,
                stage="build_pixel_panel",
                model="all",
                event_id=event_id,
                issue_type="missing_tif",
                symptom=f"pre={len(pre_tifs)}, post={len(post_tifs)}",
                fix_action="skip this event for panel build",
                impact="event dropped from model sample",
                status="open",
            )
            continue

        pre_mean, transform, raster_crs = _read_stack_mean(pre_tifs)
        post_mean, transform2, raster_crs2 = _read_stack_mean(post_tifs)
        if pre_mean.shape != post_mean.shape:
            raise ValueError(f"Shape mismatch pre/post for {event_id}")
        if transform != transform2:
            log_issue(
                ctx,
                stage="build_pixel_panel",
                model="all",
                event_id=event_id,
                issue_type="transform_mismatch",
                symptom="pre/post transform differ",
                fix_action="use pre transform for indexing and continue",
                impact="potential minor geo offset",
                status="monitor",
            )

        poi_df = _load_poi(poi_csv, exclude_types=exclude_types)
        if poi_df.empty:
            log_issue(
                ctx,
                stage="build_pixel_panel",
                model="all",
                event_id=event_id,
                issue_type="empty_poi",
                symptom="POI empty after filtering",
                fix_action="skip this event",
                impact="event dropped",
                status="open",
            )
            continue

        h, w = pre_mean.shape
        rows, cols, lons, lats = _pixel_grid(transform, h, w)
        in_buffer, n_in_buffer, nearest_types, nearest_dist = _mask_and_distance(
            rows, cols, lons, lats, transform, (h, w), raster_crs, poi_df, metric_crs
        )

        pre_flat = pre_mean.ravel()
        post_flat = post_mean.ravel()
        delta = np.where(pre_flat > 0, (post_flat - pre_flat) / pre_flat, np.nan)

        df = pd.DataFrame(
            {
                "event_id": event_id,
                "row": rows.astype(int),
                "col": cols.astype(int),
                "lon": lons,
                "lat": lats,
                "pre_mean_ntl": pre_flat,
                "post_mean_ntl": post_flat,
                "delta_ntl": delta,
                "in_buffer": in_buffer.astype(int),
                "n_facilities_in_buffer": n_in_buffer.astype(int),
                "nearest_facility_type": nearest_types,
                "distance_to_nearest": nearest_dist,
            }
        )
        df["facility_type_std"] = df["nearest_facility_type"].astype(str).str.lower().str.strip()
        df["is_damaged"] = (df["delta_ntl"] < damage_threshold).astype(int)
        df["pixel_id"] = (
            df["event_id"].astype(str)
            + "_"
            + df["row"].astype(str)
            + "_"
            + df["col"].astype(str)
        )

        before = len(df)
        df = df[np.isfinite(df["pre_mean_ntl"]) & np.isfinite(df["post_mean_ntl"]) & np.isfinite(df["delta_ntl"])].copy()
        df = df[df["pre_mean_ntl"] > pre_threshold].copy()
        after = len(df)

        rows_all.append(df)
        event_stats.append(
            {
                "event_id": event_id,
                "event_name": cfg["event_name"],
                "n_pre_images": len(pre_tifs),
                "n_post_images": len(post_tifs),
                "n_pixels_before_filter": before,
                "n_pixels_after_filter": after,
                "in_buffer_ratio": float(df["in_buffer"].mean()) if after else np.nan,
                "damage_ratio": float(df["is_damaged"].mean()) if after else np.nan,
            }
        )

    if not rows_all:
        raise RuntimeError("No valid event data available to build pixel panel.")

    panel = pd.concat(rows_all, ignore_index=True)
    ordered_cols = [
        "pixel_id",
        "event_id",
        "row",
        "col",
        "lon",
        "lat",
        "pre_mean_ntl",
        "post_mean_ntl",
        "delta_ntl",
        "in_buffer",
        "nearest_facility_type",
        "distance_to_nearest",
        "n_facilities_in_buffer",
        "facility_type_std",
        "is_damaged",
    ]
    panel = panel[ordered_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    pd.DataFrame(event_stats).to_csv(OUTPUT_DIR / "pixel_panel_event_summary.csv", index=False)

    append_progress(
        f"Built pixel panel {output_path.relative_to(ROOT)} with {len(panel):,} rows across {panel['event_id'].nunique()} events"
    )
    return panel


# ----------------------------
# NLCD stage
# ----------------------------

def attach_nlcd(
    ctx: RunContext,
    panel_path: Path = PANEL_PATH,
    output_path: Path = PANEL_NLCD_PATH,
    nlcd_dir: Optional[Path] = None,
) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    panel = panel.copy()
    panel["land_use"] = np.nan

    if nlcd_dir is None:
        nlcd_dir = PROJECT_DIR / "data" / "nlcd"

    events_cfg = load_json(CONFIG_EVENTS)
    coverage_rows: List[Dict[str, object]] = []

    for event_id, cfg in events_cfg.items():
        event_mask = panel["event_id"] == event_id
        if event_mask.sum() == 0:
            continue

        candidates = [
            nlcd_dir / f"nlcd_{event_id}.tif",
            PROJECT_DIR / "data" / "processed" / f"nlcd_{event_id}.tif",
            PROJECT_DIR / "data" / "nlcd" / f"nlcd_{event_id}.tif",
        ]
        nlcd_file = next((p for p in candidates if p.exists()), None)

        if nlcd_file is None:
            log_issue(
                ctx,
                stage="attach_nlcd",
                model="all",
                event_id=event_id,
                issue_type="missing_nlcd",
                symptom="No NLCD raster found for event",
                fix_action="keep land_use as NaN and continue",
                impact="NLCD-enhanced model skipped for this event",
                status="open",
            )
            coverage_rows.append({"event_id": event_id, "nlcd_file": "", "coverage": 0.0})
            continue

        pre_dir = ROOT / cfg["pre_dir"]
        pre_tifs = list_daily_tifs(pre_dir)
        sub = panel.loc[event_mask, ["row", "col", "lon", "lat"]].copy()
        nodata_val = np.nan

        if pre_tifs:
            with rasterio.open(pre_tifs[0]) as target_src, rasterio.open(nlcd_file) as nlcd_src:
                target_shape = (target_src.height, target_src.width)
                dst = np.full(target_shape, np.nan, dtype="float32")

                reproject(
                    source=rasterio.band(nlcd_src, 1),
                    destination=dst,
                    src_transform=nlcd_src.transform,
                    src_crs=nlcd_src.crs,
                    dst_transform=target_src.transform,
                    dst_crs=target_src.crs,
                    resampling=Resampling.nearest,
                )

                land_use_full = dst.ravel()
                nodata_val = nlcd_src.nodata if nlcd_src.nodata is not None else np.nan

            idx = (sub["row"].astype(int) * target_shape[1] + sub["col"].astype(int)).to_numpy()
            vals = np.full(len(idx), np.nan, dtype="float64")
            valid_idx = (idx >= 0) & (idx < land_use_full.size)
            if valid_idx.any():
                vals[valid_idx] = land_use_full[idx[valid_idx]]
        else:
            log_issue(
                ctx,
                stage="attach_nlcd",
                model="all",
                event_id=event_id,
                issue_type="missing_pre_tif",
                symptom="cannot infer target raster grid for NLCD align",
                fix_action="fallback to lon/lat sampling from panel coordinates",
                impact="land_use attached without event-grid reprojection",
                status="resolved",
            )
            with rasterio.open(nlcd_file) as nlcd_src:
                nodata_val = nlcd_src.nodata if nlcd_src.nodata is not None else np.nan
                coords = sub[["lon", "lat"]].astype(float).to_numpy()
                if coords.size == 0:
                    vals = np.array([], dtype="float64")
                else:
                    if nlcd_src.crs and str(nlcd_src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
                        transformer = Transformer.from_crs("EPSG:4326", nlcd_src.crs, always_xy=True)
                        xs, ys = transformer.transform(coords[:, 0], coords[:, 1])
                        sample_coords = list(zip(xs, ys))
                    else:
                        sample_coords = [tuple(v) for v in coords]
                    vals = np.array([float(v[0]) for v in nlcd_src.sample(sample_coords)], dtype="float64")

        if np.isfinite(nodata_val):
            vals[np.isclose(vals, nodata_val)] = np.nan
        vals[vals <= 0] = np.nan
        panel.loc[event_mask, "land_use"] = vals

        valid = np.isfinite(vals)
        coverage = float(valid.mean()) if len(vals) else 0.0
        coverage_rows.append({"event_id": event_id, "nlcd_file": str(nlcd_file), "coverage": coverage})

    panel["is_high_density"] = panel["land_use"].isin([23, 24]).astype(int)
    panel.to_parquet(output_path, index=False)
    pd.DataFrame(coverage_rows).to_csv(OUTPUT_DIR / "nlcd_coverage.csv", index=False)

    append_progress(
        f"Attached NLCD to panel. Coverage summary saved to {(OUTPUT_DIR / 'nlcd_coverage.csv').relative_to(ROOT)}"
    )
    return panel


# ----------------------------
# Model helpers
# ----------------------------

def _coef_table_from_result(result, model_name: str, variant: str, kind: str = "linear") -> pd.DataFrame:
    ci = result.conf_int()
    table = pd.DataFrame(
        {
            "model": model_name,
            "variant": variant,
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "stat": result.tvalues.values if hasattr(result, "tvalues") else result.zvalues.values,
            "p_value": result.pvalues.values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_high": ci.iloc[:, 1].values,
            "kind": kind,
        }
    )
    return table


def _build_formula(include_land_use: bool, include_event: bool = True) -> str:
    terms = ["in_buffer * pre_mean_ntl"]
    if include_event:
        terms.append("C(event_id)")
    if include_land_use:
        terms.append("C(land_use)")
    return "delta_ntl ~ " + " + ".join(terms)


def fit_ols_and_mixed(
    ctx: RunContext,
    df: pd.DataFrame,
    variant: str,
    include_land_use: bool,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    formula_ols = _build_formula(include_land_use=include_land_use, include_event=True)
    model_ols = smf.ols(formula_ols, data=df).fit(cov_type="HC1")
    ols_coef = _coef_table_from_result(model_ols, "OLS", variant, kind="linear")
    ols_pred = df[["pixel_id", "event_id", "delta_ntl"]].copy()
    ols_pred["predicted"] = model_ols.predict(df)
    ols_pred["residual"] = ols_pred["delta_ntl"] - ols_pred["predicted"]
    ols_pred.to_csv(OUTPUT_DIR / f"ols_predictions_{variant}.csv", index=False)

    formula_mixed = _build_formula(include_land_use=include_land_use, include_event=False)
    mixed_model = smf.mixedlm(formula_mixed, data=df, groups=df["event_id"])
    mixed_result = None
    mixed_error = None

    try:
        mixed_result = mixed_model.fit(method="lbfgs", reml=False)
    except Exception as e_lbfgs:
        mixed_error = str(e_lbfgs)
        try:
            mixed_result = mixed_model.fit(method="powell", reml=False)
            log_issue(
                ctx,
                stage="fit_mixedlm",
                model="MixedLM",
                event_id="all",
                issue_type="optimizer_fallback",
                symptom=f"lbfgs failed: {mixed_error}",
                fix_action="fallback to powell optimizer",
                impact="estimate retained with slower convergence",
                status="resolved",
            )
        except Exception as e_powell:
            mixed_error = f"lbfgs={mixed_error}; powell={e_powell}"
            log_issue(
                ctx,
                stage="fit_mixedlm",
                model="MixedLM",
                event_id="all",
                issue_type="model_fit_failed",
                symptom=mixed_error,
                fix_action="skip mixedlm in this variant",
                impact="mixed model unavailable",
                status="open",
            )

    if mixed_result is not None:
        mixed_coef = _coef_table_from_result(mixed_result, "MixedLM", variant, kind="linear")
        re_rows = []
        random_effects_failed = False
        try:
            re_obj = mixed_result.random_effects
            for ev, re_dict in re_obj.items():
                if isinstance(re_dict, pd.Series):
                    val = float(re_dict.iloc[0])
                elif isinstance(re_dict, dict):
                    val = float(next(iter(re_dict.values())))
                else:
                    val = float(re_dict[0]) if hasattr(re_dict, "__len__") else float(re_dict)
                re_rows.append({"event_id": ev, "random_intercept": val, "variant": variant})
        except Exception as e_re:
            random_effects_failed = True
            log_issue(
                ctx,
                stage="fit_mixedlm",
                model="MixedLM",
                event_id="all",
                issue_type="random_effect_extraction_failed",
                symptom=str(e_re),
                fix_action="keep fixed effects and predictions; skip random-intercept export",
                impact="random-effect chart unavailable for this variant",
                status="monitor",
            )
        random_effects = pd.DataFrame(re_rows)

        mixed_pred = df[["pixel_id", "event_id", "delta_ntl"]].copy()
        mixed_pred["predicted"] = mixed_result.predict(df)
        mixed_pred["residual"] = mixed_pred["delta_ntl"] - mixed_pred["predicted"]
        mixed_pred.to_csv(OUTPUT_DIR / f"mixedlm_predictions_{variant}.csv", index=False)
        if not random_effects_failed:
            random_effects.to_csv(OUTPUT_DIR / f"mixedlm_random_effects_{variant}.csv", index=False)
    else:
        mixed_coef = pd.DataFrame(
            columns=["model", "variant", "term", "coef", "std_err", "stat", "p_value", "ci_low", "ci_high", "kind"]
        )
        random_effects = pd.DataFrame(columns=["event_id", "random_intercept", "variant"])

    out["ols_coef"] = ols_coef
    out["mixed_coef"] = mixed_coef
    out["random_effects"] = random_effects
    return out


def fit_logit(
    ctx: RunContext,
    df: pd.DataFrame,
    variant: str,
    include_land_use: bool,
    damage_threshold: float,
) -> Dict[str, pd.DataFrame]:
    work = df.copy()
    work["is_damaged"] = (work["delta_ntl"] < damage_threshold).astype(int)

    terms = ["in_buffer * pre_mean_ntl", "C(event_id)"]
    if include_land_use:
        terms.append("C(land_use)")
    formula = "is_damaged ~ " + " + ".join(terms)

    result = None
    try:
        result = smf.logit(formula=formula, data=work).fit(disp=False, maxiter=200)
    except Exception as e_logit:
        log_issue(
            ctx,
            stage="fit_logit",
            model="Logit",
            event_id="all",
            issue_type="model_fit_failed",
            symptom=str(e_logit),
            fix_action="skip logit for this variant",
            impact="logit unavailable",
            status="open",
        )
        empty = pd.DataFrame()
        return {
            "coef": empty,
            "marginal": empty,
            "roc": empty,
            "calibration": empty,
            "predictions": empty,
        }

    coef = _coef_table_from_result(result, "Logit", variant, kind="logit")
    coef["odds_ratio"] = np.exp(coef["coef"])
    coef["or_ci_low"] = np.exp(coef["ci_low"])
    coef["or_ci_high"] = np.exp(coef["ci_high"])

    pred_prob = result.predict(work)
    pred_df = work[["pixel_id", "event_id", "is_damaged"]].copy()
    pred_df["pred_prob"] = pred_prob
    pred_df.to_csv(OUTPUT_DIR / f"logit_predictions_{variant}.csv", index=False)

    marginal = result.get_margeff(at="overall").summary_frame().reset_index().rename(columns={"index": "term"})
    marginal["model"] = "Logit"
    marginal["variant"] = variant

    fpr, tpr, thresholds = roc_curve(work["is_damaged"], pred_prob)
    roc_df = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
            "auc": auc(fpr, tpr),
            "variant": variant,
        }
    )

    cal = pd.DataFrame({"pred_prob": pred_prob, "is_damaged": work["is_damaged"]})
    cal["bin"] = pd.qcut(cal["pred_prob"], q=10, duplicates="drop")
    calibration = (
        cal.groupby("bin", observed=True)
        .agg(pred_mean=("pred_prob", "mean"), obs_rate=("is_damaged", "mean"), n=("is_damaged", "size"))
        .reset_index()
    )
    calibration["variant"] = variant

    return {
        "coef": coef,
        "marginal": marginal,
        "roc": roc_df,
        "calibration": calibration,
        "predictions": pred_df,
    }


def build_recovery_panel(
    ctx: RunContext,
    panel: pd.DataFrame,
    threshold: float,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    events_cfg = load_json(CONFIG_EVENTS)
    rows_all = []

    for event_id, cfg in events_cfg.items():
        sub = panel[panel["event_id"] == event_id].copy()
        if sub.empty:
            continue

        post_dir = ROOT / cfg["post_dir"]
        post_tifs = list_daily_tifs(post_dir)
        if not post_tifs:
            log_issue(
                ctx,
                stage="build_recovery_panel",
                model="Cox",
                event_id=event_id,
                issue_type="missing_post_tif",
                symptom="No post-event daily tifs",
                fix_action="skip event in cox panel",
                impact="reduced sample",
                status="open",
            )
            continue

        arrs = []
        shape = None
        for p in post_tifs:
            with rasterio.open(p) as src:
                arr = src.read(1).astype("float64")
                if shape is None:
                    shape = arr.shape
                elif arr.shape != shape:
                    raise ValueError(f"Post tif shape mismatch in {event_id}: {p}")
                arrs.append(arr)

        stack = np.stack(arrs, axis=0)  # D x H x W
        dcount, _, width = stack.shape[0], stack.shape[1], stack.shape[2]

        rr = sub["row"].astype(int).to_numpy()
        cc = sub["col"].astype(int).to_numpy()
        seq = stack[:, rr, cc].T  # N x D
        targets = sub["pre_mean_ntl"].to_numpy() * threshold
        recovered = seq >= targets[:, None]
        observed = recovered.any(axis=1)
        first_day = np.where(observed, recovered.argmax(axis=1) + 1, dcount)

        out = sub.copy()
        out["recovery_days"] = first_day.astype(int)
        out["event_observed"] = observed.astype(int)
        out["recovery_threshold"] = threshold
        rows_all.append(out)

    if not rows_all:
        raise RuntimeError("No recovery rows built.")

    rec = pd.concat(rows_all, ignore_index=True)
    if output_path is not None:
        rec.to_parquet(output_path, index=False)
    return rec


def fit_cox(
    ctx: RunContext,
    recovery_df: pd.DataFrame,
    variant: str,
    include_land_use: bool,
) -> Dict[str, pd.DataFrame]:
    cols = ["recovery_days", "event_observed", "in_buffer", "pre_mean_ntl", "event_id"]
    if include_land_use and "land_use" in recovery_df.columns:
        cols.append("land_use")

    work = recovery_df[cols].copy()
    work = work[np.isfinite(work["recovery_days"]) & np.isfinite(work["pre_mean_ntl"])].copy()

    design = work[["recovery_days", "event_observed", "in_buffer", "pre_mean_ntl"]].copy()
    design = pd.concat([design, pd.get_dummies(work["event_id"], prefix="event", drop_first=True)], axis=1)
    if include_land_use and "land_use" in work.columns and work["land_use"].notna().any():
        design = pd.concat([design, pd.get_dummies(work["land_use"].astype("Int64"), prefix="lu", drop_first=True)], axis=1)

    cph = CoxPHFitter()
    try:
        cph.fit(design, duration_col="recovery_days", event_col="event_observed")
    except Exception as e_cox:
        log_issue(
            ctx,
            stage="fit_cox",
            model="Cox",
            event_id="all",
            issue_type="model_fit_failed",
            symptom=str(e_cox),
            fix_action="skip cox for this variant",
            impact="cox unavailable",
            status="open",
        )
        return {"coef": pd.DataFrame(), "km": pd.DataFrame(), "ph": pd.DataFrame()}

    summary = cph.summary.reset_index().rename(columns={"index": "term"})
    summary["model"] = "Cox"
    summary["variant"] = variant
    summary["hazard_ratio"] = np.exp(summary["coef"])

    km_rows = []
    kmf = KaplanMeierFitter()
    for grp in [0, 1]:
        grp_df = recovery_df[recovery_df["in_buffer"] == grp]
        if grp_df.empty:
            continue
        kmf.fit(grp_df["recovery_days"], grp_df["event_observed"], label=f"in_buffer_{grp}")
        sf = kmf.survival_function_.reset_index().rename(columns={"timeline": "day", kmf._label: "survival"})
        sf["group"] = grp
        sf["variant"] = variant
        km_rows.append(sf)
    km_df = pd.concat(km_rows, ignore_index=True) if km_rows else pd.DataFrame()

    try:
        ph = proportional_hazard_test(cph, design, time_transform="rank")
        ph_df = ph.summary.reset_index().rename(columns={"index": "term"})
        ph_df["variant"] = variant
    except Exception as e_ph:
        log_issue(
            ctx,
            stage="fit_cox",
            model="Cox",
            event_id="all",
            issue_type="ph_test_failed",
            symptom=str(e_ph),
            fix_action="skip PH diagnostic table",
            impact="PH diagnostic unavailable",
            status="monitor",
        )
        ph_df = pd.DataFrame()

    return {"coef": summary, "km": km_df, "ph": ph_df}


# ----------------------------
# Robustness
# ----------------------------

def run_robustness(
    ctx: RunContext,
    base_panel: pd.DataFrame,
    defaults: Dict[str, object],
    include_land_use: bool,
) -> pd.DataFrame:
    rows = []

    # Pre-threshold sensitivity on OLS core coefficient
    for thr in defaults["pre_threshold_scenarios"]:
        sub = base_panel[base_panel["pre_mean_ntl"] > float(thr)].copy()
        if len(sub) < 100:
            continue
        formula = "delta_ntl ~ in_buffer * pre_mean_ntl + C(event_id)"
        if include_land_use and "land_use" in sub.columns and sub["land_use"].notna().any():
            formula += " + C(land_use)"
        res = smf.ols(formula, data=sub).fit(cov_type="HC1")
        term = "in_buffer"
        if term in res.params.index:
            rows.append(
                {
                    "scenario_type": "pre_threshold",
                    "scenario_value": thr,
                    "model": "OLS",
                    "term": term,
                    "coef": res.params[term],
                    "p_value": res.pvalues[term],
                    "n_obs": len(sub),
                }
            )

    # Damage threshold sensitivity on logit in_buffer odds
    for dthr in defaults["damage_threshold_scenarios"]:
        sub = base_panel.copy()
        sub["is_damaged"] = (sub["delta_ntl"] < float(dthr)).astype(int)
        formula = "is_damaged ~ in_buffer * pre_mean_ntl + C(event_id)"
        if include_land_use and "land_use" in sub.columns and sub["land_use"].notna().any():
            formula += " + C(land_use)"
        try:
            res = smf.logit(formula, data=sub).fit(disp=False, maxiter=200)
            term = "in_buffer"
            if term in res.params.index:
                rows.append(
                    {
                        "scenario_type": "damage_threshold",
                        "scenario_value": dthr,
                        "model": "Logit",
                        "term": term,
                        "coef": res.params[term],
                        "odds_ratio": math.exp(res.params[term]),
                        "p_value": res.pvalues[term],
                        "n_obs": len(sub),
                    }
                )
        except Exception as e:
            log_issue(
                ctx,
                stage="robustness",
                model="Logit",
                event_id="all",
                issue_type="scenario_fit_failed",
                symptom=f"damage_threshold={dthr}: {e}",
                fix_action="skip scenario",
                impact="one robustness scenario missing",
                status="monitor",
            )

    # Facility exclusion scenario
    excludes = set(defaults["facility_exclusion_for_robustness"])
    sub = base_panel[~base_panel["facility_type_std"].isin(excludes)].copy()
    if len(sub) > 100:
        res = smf.ols("delta_ntl ~ in_buffer * pre_mean_ntl + C(event_id)", data=sub).fit(cov_type="HC1")
        if "in_buffer" in res.params.index:
            rows.append(
                {
                    "scenario_type": "exclude_facility_types",
                    "scenario_value": ",".join(sorted(excludes)),
                    "model": "OLS",
                    "term": "in_buffer",
                    "coef": res.params["in_buffer"],
                    "p_value": res.pvalues["in_buffer"],
                    "n_obs": len(sub),
                }
            )

    # Recovery threshold sensitivity on Cox
    for rthr in defaults["recovery_threshold_scenarios"]:
        try:
            rec = build_recovery_panel(ctx=ctx, panel=base_panel, threshold=float(rthr), output_path=None)
        except Exception as e_rec:
            log_issue(
                ctx,
                stage="robustness",
                model="Cox",
                event_id="all",
                issue_type="recovery_panel_unavailable",
                symptom=f"recovery_threshold={rthr}: {e_rec}",
                fix_action="skip this recovery-threshold scenario",
                impact="partial robustness table for Cox recovery thresholds",
                status="open",
            )
            continue
        cox_res = fit_cox(ctx=ctx, recovery_df=rec, variant=f"robust_recovery_{rthr}", include_land_use=False)
        coef = cox_res["coef"]
        if not coef.empty and "in_buffer" in coef["covariate"].values:
            row = coef.loc[coef["covariate"] == "in_buffer"].iloc[0]
            rows.append(
                {
                    "scenario_type": "recovery_threshold",
                    "scenario_value": rthr,
                    "model": "Cox",
                    "term": "in_buffer",
                    "coef": row["coef"],
                    "hazard_ratio": row["hazard_ratio"],
                    "p_value": row["p"],
                    "n_obs": len(rec),
                }
            )

    robust_df = pd.DataFrame(rows)
    robust_df.to_csv(OUTPUT_DIR / "robustness_summary.csv", index=False)
    append_progress("Completed robustness scenarios and exported robustness_summary.csv")
    return robust_df


# ----------------------------
# Figure generation
# ----------------------------

def _coef_plot(df: pd.DataFrame, value_col: str, low_col: str, high_col: str, title: str, out_file: Path, x_label: str) -> None:
    if df.empty:
        return
    d = df.copy()
    d = d.sort_values(value_col)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(d))))
    ax.errorbar(
        d[value_col],
        np.arange(len(d)),
        xerr=[d[value_col] - d[low_col], d[high_col] - d[value_col]],
        fmt="o",
        color="#1f77b4",
        ecolor="#7f7f7f",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["term"])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    plt.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def generate_figures() -> None:
    sns.set_theme(style="whitegrid")

    # OLS figures
    ols_path = OUTPUT_DIR / "ols_results.csv"
    if not ols_path.exists():
        append_progress("Figure generation skipped: missing ols_results.csv")
        return
    ols_coef = pd.read_csv(ols_path)
    ols_base = ols_coef[(ols_coef["model"] == "OLS") & (ols_coef["variant"] == "no_nlcd")].copy()
    _coef_plot(
        ols_base,
        value_col="coef",
        low_col="ci_low",
        high_col="ci_high",
        title="OLS Coefficients (No NLCD)",
        out_file=FIG_DIR / "ols" / "ols_coefficients.png",
        x_label="Coefficient",
    )

    pred_path = OUTPUT_DIR / "ols_predictions_no_nlcd.csv"
    if not pred_path.exists():
        append_progress("Figure generation warning: missing ols_predictions_no_nlcd.csv")
        return
    pred = pd.read_csv(pred_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pred["predicted"], pred["delta_ntl"], s=8, alpha=0.35, color="#2c7fb8")
    lo = min(pred["predicted"].min(), pred["delta_ntl"].min())
    hi = max(pred["predicted"].max(), pred["delta_ntl"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_title("OLS Predicted vs Actual")
    ax.set_xlabel("Predicted delta_ntl")
    ax.set_ylabel("Actual delta_ntl")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ols" / "ols_pred_vs_actual.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pred["predicted"], pred["residual"], s=8, alpha=0.3, color="#d95f02")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("OLS Residual Diagnostic")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ols" / "ols_residual_diagnostic.png", dpi=220)
    plt.close(fig)

    # MixedLM figures
    mixed_path = OUTPUT_DIR / "mixedlm_results.csv"
    if mixed_path.exists():
        mixed_coef = pd.read_csv(mixed_path)
        mixed_base = mixed_coef[(mixed_coef["model"] == "MixedLM") & (mixed_coef["variant"] == "no_nlcd")].copy()
        _coef_plot(
            mixed_base,
            value_col="coef",
            low_col="ci_low",
            high_col="ci_high",
            title="MixedLM Fixed Effects (No NLCD)",
            out_file=FIG_DIR / "mixedlm" / "mixedlm_fixed_effects.png",
            x_label="Coefficient",
        )

    re_file = OUTPUT_DIR / "mixedlm_random_effects_no_nlcd.csv"
    if re_file.exists():
        re_df = pd.read_csv(re_file)
    else:
        re_df = pd.DataFrame(columns=["event_id", "random_intercept"])
    if not re_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        re_df = re_df.sort_values("random_intercept")
        ax.barh(re_df["event_id"], re_df["random_intercept"], color="#7570b3")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title("MixedLM Random Intercepts by Event")
        ax.set_xlabel("Random intercept")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "mixedlm" / "mixedlm_random_effects.png", dpi=220)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Random effects unavailable\\n(singular covariance structure)",
            ha="center",
            va="center",
            fontsize=11,
        )
        fig.savefig(FIG_DIR / "mixedlm" / "mixedlm_random_effects.png", dpi=220)
        plt.close(fig)

    mpred_file = OUTPUT_DIR / "mixedlm_predictions_no_nlcd.csv"
    if mpred_file.exists():
        mpred = pd.read_csv(mpred_file)
        grp = mpred.groupby("event_id", observed=True).agg(actual=("delta_ntl", "mean"), predicted=("predicted", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(grp))
        ax.plot(x, grp["actual"], marker="o", label="Actual mean")
        ax.plot(x, grp["predicted"], marker="s", label="Predicted mean")
        ax.set_xticks(x)
        ax.set_xticklabels(grp["event_id"], rotation=25, ha="right")
        ax.set_title("MixedLM Group-level Fit")
        ax.set_ylabel("Mean delta_ntl")
        ax.legend()
        plt.tight_layout()
        fig.savefig(FIG_DIR / "mixedlm" / "mixedlm_group_fit.png", dpi=220)
        plt.close(fig)

    # Logit figures
    logit_coef = pd.read_csv(OUTPUT_DIR / "logit_results.csv")
    logit_base = logit_coef[(logit_coef["model"] == "Logit") & (logit_coef["variant"] == "no_nlcd")].copy()
    if not logit_base.empty:
        d = logit_base.sort_values("odds_ratio")
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(d))))
        ax.errorbar(
            d["odds_ratio"],
            np.arange(len(d)),
            xerr=[d["odds_ratio"] - d["or_ci_low"], d["or_ci_high"] - d["odds_ratio"]],
            fmt="o",
            color="#1b9e77",
            ecolor="#7f7f7f",
            capsize=3,
        )
        ax.axvline(1, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(d["term"])
        ax.set_title("Logit Odds Ratios (No NLCD)")
        ax.set_xlabel("Odds Ratio")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "logit" / "logit_odds_ratio.png", dpi=220)
        plt.close(fig)

    roc_path = OUTPUT_DIR / "logit_roc_no_nlcd.csv"
    if not roc_path.exists():
        append_progress("Figure generation warning: missing logit_roc_no_nlcd.csv")
        return
    roc = pd.read_csv(roc_path)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(roc["fpr"], roc["tpr"], color="#d95f02", label=f"AUC={roc['auc'].iloc[0]:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("Logit ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "logit" / "logit_roc_curve.png", dpi=220)
    plt.close(fig)

    cal_path = OUTPUT_DIR / "logit_calibration_no_nlcd.csv"
    if not cal_path.exists():
        append_progress("Figure generation warning: missing logit_calibration_no_nlcd.csv")
        return
    cal = pd.read_csv(cal_path)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(cal["pred_mean"], cal["obs_rate"], marker="o", color="#7570b3")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("Logit Calibration")
    ax.set_xlabel("Predicted damage probability")
    ax.set_ylabel("Observed damage rate")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "logit" / "logit_calibration.png", dpi=220)
    plt.close(fig)

    # Cox figures
    cox_path = OUTPUT_DIR / "cox_results.csv"
    if not cox_path.exists():
        append_progress("Figure generation warning: missing cox_results.csv")
        return
    cox_coef = pd.read_csv(cox_path)
    cox_base = cox_coef[(cox_coef["model"] == "Cox") & (cox_coef["variant"] == "no_nlcd")].copy()

    if not cox_base.empty:
        d = cox_base.sort_values("hazard_ratio")
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(d))))
        ax.errorbar(
            d["hazard_ratio"],
            np.arange(len(d)),
            xerr=[d["hazard_ratio"] - np.exp(d["coef lower 95%"]), np.exp(d["coef upper 95%"]) - d["hazard_ratio"]],
            fmt="o",
            color="#66a61e",
            ecolor="#7f7f7f",
            capsize=3,
        )
        ax.axvline(1, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(d["covariate"])
        ax.set_title("Cox Hazard Ratios (No NLCD)")
        ax.set_xlabel("Hazard Ratio")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "cox" / "cox_hazard_ratio.png", dpi=220)
        plt.close(fig)

    km_path = OUTPUT_DIR / "cox_km_no_nlcd.csv"
    if not km_path.exists():
        append_progress("Figure generation warning: missing cox_km_no_nlcd.csv")
        return
    km = pd.read_csv(km_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for g in sorted(km["group"].unique()):
        sub = km[km["group"] == g]
        label = "Buffer=1" if g == 1 else "Buffer=0"
        ax.plot(sub["day"], sub["survival"], label=label)
    ax.set_title("Kaplan-Meier Recovery Curves")
    ax.set_xlabel("Days after disaster")
    ax.set_ylabel("Not recovered proportion")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "cox" / "cox_km_curve.png", dpi=220)
    plt.close(fig)

    ph_file = OUTPUT_DIR / "cox_ph_test_no_nlcd.csv"
    if ph_file.exists():
        ph = pd.read_csv(ph_file)
        if not ph.empty and "p" in ph.columns:
            ph = ph.sort_values("p")
            cov_col = "covariate"
            if cov_col not in ph.columns:
                if "term" in ph.columns:
                    cov_col = "term"
                elif "index" in ph.columns:
                    cov_col = "index"
                else:
                    cov_col = ph.columns[0]
            fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(ph))))
            ax.barh(ph[cov_col].astype(str), ph["p"], color="#e7298a")
            ax.axvline(0.05, color="black", linestyle="--", linewidth=1)
            ax.set_title("Cox PH Test p-values")
            ax.set_xlabel("p-value")
            plt.tight_layout()
            fig.savefig(FIG_DIR / "cox" / "cox_ph_test.png", dpi=220)
            plt.close(fig)

    append_progress("Generated model figures for OLS, MixedLM, Logit, and Cox")


# ----------------------------
# Reporting
# ----------------------------

def _read_issue_rows(model_name: str) -> pd.DataFrame:
    issue_file = OUTPUT_DIR / "model_issue_log.csv"
    if not issue_file.exists():
        return pd.DataFrame()
    issues = pd.read_csv(issue_file)
    if issues.empty:
        return issues
    return issues[issues["model"].str.lower() == model_name.lower()].copy()


def _build_problem_section(model_name: str) -> str:
    issues = _read_issue_rows(model_name)
    if issues.empty:
        return (
            "No critical issue observed.\n\n"
            "Residual risk: sample composition and unobserved confounders may still influence effect size; "
            "this risk is tracked in robustness outputs."
        )

    lines = []
    keys = ["issue_type", "symptom", "fix_action", "impact", "status"]
    grouped = (
        issues.groupby(keys, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "issue_type"], ascending=[False, True])
    )

    for _, r in grouped.iterrows():
        lines.append(
            f"- 发生次数 Count: {int(r['count'])}\n"
            f"  症状 Symptom: {r['symptom']}\n"
            f"  原因 Cause: {r['issue_type']}\n"
            f"  修复 Fix: {r['fix_action']}\n"
            f"  影响 Impact: {r['impact']}\n"
            f"  状态 Status: {r['status']}"
        )
    return "\n".join(lines)


def _write_model_report(
    filename: Path,
    title: str,
    objective: str,
    data_features: str,
    model_spec: str,
    results_text: str,
    figures: List[Tuple[str, str]],
    interpretation: str,
    limitations: str,
    next_step: str,
    problem_section: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        "## Objective",
        objective,
        "",
        "## Data & Features",
        data_features,
        "",
        "## Model Spec",
        model_spec,
        "",
        "## Problems & Fixes",
        problem_section,
        "",
        "## Results",
        results_text,
        "",
        "## Figures",
    ]
    for caption, relpath in figures:
        lines.append(f"- {caption}: `{relpath}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            interpretation,
            "",
            "## Limitations",
            limitations,
            "",
            "## Next Step",
            next_step,
            "",
        ]
    )
    filename.write_text("\n".join(lines), encoding="utf-8")


def _safe_pick(df: pd.DataFrame, cond, col: str, default=np.nan):
    sub = df.loc[cond]
    if sub.empty or col not in sub.columns:
        return default
    return sub.iloc[0][col]


def build_model_summary_for_report() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    # OLS
    ols_file = OUTPUT_DIR / "ols_results.csv"
    if ols_file.exists():
        ols = pd.read_csv(ols_file)
        for _, r in ols.iterrows():
            if r["term"] == "in_buffer":
                rows.append(
                    {
                        "model": r["model"],
                        "variant": r["variant"],
                        "key_metric": "coef_in_buffer",
                        "term": r["term"],
                        "value": r["coef"],
                        "p_value": r["p_value"],
                        "ci_low": r["ci_low"],
                        "ci_high": r["ci_high"],
                        "n_obs": np.nan,
                        "notes": "linear effect",
                    }
                )

    # MixedLM
    mixed_file = OUTPUT_DIR / "mixedlm_results.csv"
    if mixed_file.exists():
        mixed = pd.read_csv(mixed_file)
        for _, r in mixed.iterrows():
            if r["term"] == "in_buffer":
                rows.append(
                    {
                        "model": r["model"],
                        "variant": r["variant"],
                        "key_metric": "coef_in_buffer",
                        "term": r["term"],
                        "value": r["coef"],
                        "p_value": r["p_value"],
                        "ci_low": r["ci_low"],
                        "ci_high": r["ci_high"],
                        "n_obs": np.nan,
                        "notes": "mixed-effect fixed coefficient",
                    }
                )

    # Logit
    logit_file = OUTPUT_DIR / "logit_results.csv"
    if logit_file.exists():
        logit = pd.read_csv(logit_file)
        for _, r in logit.iterrows():
            if r["term"] == "in_buffer":
                rows.append(
                    {
                        "model": "Logit",
                        "variant": r["variant"],
                        "key_metric": "odds_ratio_in_buffer",
                        "term": r["term"],
                        "value": r.get("odds_ratio", np.nan),
                        "p_value": r["p_value"],
                        "ci_low": r.get("or_ci_low", np.nan),
                        "ci_high": r.get("or_ci_high", np.nan),
                        "n_obs": np.nan,
                        "notes": "odds ratio",
                    }
                )

    for variant in ["no_nlcd", "with_nlcd"]:
        roc_file = OUTPUT_DIR / f"logit_roc_{variant}.csv"
        if roc_file.exists():
            roc = pd.read_csv(roc_file)
            if not roc.empty:
                rows.append(
                    {
                        "model": "Logit",
                        "variant": variant,
                        "key_metric": "auc",
                        "term": "AUC",
                        "value": roc["auc"].iloc[0],
                        "p_value": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n_obs": np.nan,
                        "notes": "ROC AUC",
                    }
                )

    # Cox
    cox_file = OUTPUT_DIR / "cox_results.csv"
    if cox_file.exists():
        cox = pd.read_csv(cox_file)
        for _, r in cox.iterrows():
            if r.get("covariate", "") == "in_buffer":
                rows.append(
                    {
                        "model": "Cox",
                        "variant": r["variant"],
                        "key_metric": "hazard_ratio_in_buffer",
                        "term": "in_buffer",
                        "value": r["hazard_ratio"],
                        "p_value": r["p"],
                        "ci_low": float(np.exp(r["coef lower 95%"])),
                        "ci_high": float(np.exp(r["coef upper 95%"])),
                        "n_obs": np.nan,
                        "notes": "hazard ratio",
                    }
                )

    summary = pd.DataFrame(
        rows,
        columns=[
            "model",
            "variant",
            "key_metric",
            "term",
            "value",
            "p_value",
            "ci_low",
            "ci_high",
            "n_obs",
            "notes",
        ],
    )
    summary.to_csv(OUTPUT_DIR / "model_summary_for_report.csv", index=False)
    return summary


def generate_reports() -> None:
    summary = pd.read_csv(OUTPUT_DIR / "model_summary_for_report.csv")

    def _metric(model: str, variant: str, key_metric: str) -> Tuple[float, float]:
        sub = summary[
            (summary["model"] == model)
            & (summary["variant"] == variant)
            & (summary["key_metric"] == key_metric)
        ]
        if sub.empty:
            return np.nan, np.nan
        row = sub.iloc[0]
        return row.get("value", np.nan), row.get("p_value", np.nan)

    def _fmt(v: float, digits: int = 4) -> str:
        return "N/A" if pd.isna(v) else f"{v:.{digits}f}"

    def _fmt_p(v: float) -> str:
        return "N/A" if pd.isna(v) else f"{v:.4g}"

    # OLS report
    ols_coef_no, ols_p_no = _metric("OLS", "no_nlcd", "coef_in_buffer")
    ols_coef_lu, ols_p_lu = _metric("OLS", "with_nlcd", "coef_in_buffer")
    ols_delta = ols_coef_lu - ols_coef_no if pd.notna(ols_coef_lu) and pd.notna(ols_coef_no) else np.nan

    _write_model_report(
        filename=REPORT_DIR / "01_ols_report.md",
        title="OLS Model Report / OLS 模型报告",
        objective="评估在控制基线亮度与事件差异后，缓冲区像素是否表现出更高韧性 (resilience).",
        data_features="`no_nlcd` 使用 `all_events_pixel_panel_v1.parquet`；`with_nlcd` 使用 `all_events_pixel_panel_v1_with_nlcd.parquet` 并加入 `land_use`。",
        model_spec="`no_nlcd`: `delta_ntl ~ in_buffer * pre_mean_ntl + C(event_id)`; `with_nlcd`: `+ C(land_use)` (HC1 robust SE).",
        results_text=(
            f"`in_buffer` (no_nlcd) = {_fmt(ols_coef_no)}, p={_fmt_p(ols_p_no)}; "
            f"(with_nlcd) = {_fmt(ols_coef_lu)}, p={_fmt_p(ols_p_lu)}; "
            f"change(with-no) = {_fmt(ols_delta)}."
        ),
        figures=[
            ("系数图 Coefficient Plot", "project/modeling_report/figures/ols/ols_coefficients.png"),
            ("预测-实际 Predicted vs Actual", "project/modeling_report/figures/ols/ols_pred_vs_actual.png"),
            ("残差诊断 Residual Diagnostic", "project/modeling_report/figures/ols/ols_residual_diagnostic.png"),
        ],
        interpretation="若 `in_buffer` 为正且显著，说明在同等基线下，设施缓冲区像素夜光下降更少。",
        limitations="OLS 假设独立同分布；像素空间相关可能造成标准误低估。",
        next_step="与 MixedLM 对照随机效应后确认结论稳健性。",
        problem_section=_build_problem_section("OLS"),
    )

    # Mixed report
    mixed_coef_no, mixed_p_no = _metric("MixedLM", "no_nlcd", "coef_in_buffer")
    mixed_coef_lu, mixed_p_lu = _metric("MixedLM", "with_nlcd", "coef_in_buffer")
    mixed_delta = mixed_coef_lu - mixed_coef_no if pd.notna(mixed_coef_lu) and pd.notna(mixed_coef_no) else np.nan

    _write_model_report(
        filename=REPORT_DIR / "02_mixedlm_report.md",
        title="Mixed-Effects Model Report / 混合效应模型报告",
        objective="在事件层级随机截距下估计缓冲区效应，处理像素嵌套结构。",
        data_features="同 OLS 两个数据版本，模型在 `event_id` 层加入 random intercept。",
        model_spec="`no_nlcd`: `delta_ntl ~ in_buffer * pre_mean_ntl`; `with_nlcd`: `+ C(land_use)`, `groups=event_id`.",
        results_text=(
            f"`in_buffer` (no_nlcd) = {_fmt(mixed_coef_no)}, p={_fmt_p(mixed_p_no)}; "
            f"(with_nlcd) = {_fmt(mixed_coef_lu)}, p={_fmt_p(mixed_p_lu)}; "
            f"change(with-no) = {_fmt(mixed_delta)}."
        ),
        figures=[
            ("固定效应系数 Fixed Effects", "project/modeling_report/figures/mixedlm/mixedlm_fixed_effects.png"),
            ("随机截距 Random Intercepts", "project/modeling_report/figures/mixedlm/mixedlm_random_effects.png"),
            ("组内拟合 Group-level Fit", "project/modeling_report/figures/mixedlm/mixedlm_group_fit.png"),
        ],
        interpretation="若 MixedLM 与 OLS 的 `in_buffer` 方向一致，说明结论不依赖单一方差假设。",
        limitations="仅建模事件随机截距，未引入空间随机场。",
        next_step="在 NLCD 接入后重跑 mixed model 对比系数变化。",
        problem_section=_build_problem_section("MixedLM"),
    )

    # Logit report
    logit_or_no, logit_p_no = _metric("Logit", "no_nlcd", "odds_ratio_in_buffer")
    logit_or_lu, logit_p_lu = _metric("Logit", "with_nlcd", "odds_ratio_in_buffer")
    auc_no, _ = _metric("Logit", "no_nlcd", "auc")
    auc_lu, _ = _metric("Logit", "with_nlcd", "auc")
    logit_delta = logit_or_lu - logit_or_no if pd.notna(logit_or_lu) and pd.notna(logit_or_no) else np.nan

    _write_model_report(
        filename=REPORT_DIR / "03_logit_report.md",
        title="Logistic Model Report / Logit 模型报告",
        objective="评估缓冲区位置是否降低像素受损概率 (damage probability).",
        data_features="因变量 `is_damaged = 1(delta_ntl < threshold)`，阈值基线 -10%。",
        model_spec="`no_nlcd`: `is_damaged ~ in_buffer * pre_mean_ntl + C(event_id)`; `with_nlcd`: `+ C(land_use)`.",
        results_text=(
            f"`in_buffer` OR (no_nlcd) = {_fmt(logit_or_no)}, p={_fmt_p(logit_p_no)}, AUC={_fmt(auc_no)}; "
            f"(with_nlcd) = {_fmt(logit_or_lu)}, p={_fmt_p(logit_p_lu)}, AUC={_fmt(auc_lu)}; "
            f"OR change(with-no) = {_fmt(logit_delta)}."
        ),
        figures=[
            ("优势比图 Odds Ratio Plot", "project/modeling_report/figures/logit/logit_odds_ratio.png"),
            ("ROC 曲线 ROC Curve", "project/modeling_report/figures/logit/logit_roc_curve.png"),
            ("校准图 Calibration", "project/modeling_report/figures/logit/logit_calibration.png"),
        ],
        interpretation="OR<1 表示缓冲区像素受损 odds 更低；AUC 反映分类区分能力。",
        limitations="损害阈值定义会影响绝对概率，需结合阈值敏感性解释。",
        next_step="在 robustness 中比较 -5/-10/-15/-20% 阈值结果一致性。",
        problem_section=_build_problem_section("Logit"),
    )

    # Cox report
    cox_hr_no, cox_p_no = _metric("Cox", "no_nlcd", "hazard_ratio_in_buffer")
    cox_hr_lu, cox_p_lu = _metric("Cox", "with_nlcd", "hazard_ratio_in_buffer")
    cox_delta = cox_hr_lu - cox_hr_no if pd.notna(cox_hr_lu) and pd.notna(cox_hr_no) else np.nan

    _write_model_report(
        filename=REPORT_DIR / "04_cox_report.md",
        title="Cox PH Model Report / Cox 生存模型报告",
        objective="比较缓冲区与非缓冲区像素恢复速度 (recovery speed) 差异。",
        data_features="使用 `recovery_days` 与 `event_observed`（右删失处理）。",
        model_spec="Cox proportional hazards on in_buffer + baseline + event dummies; `with_nlcd` 额外加入 land-use dummies.",
        results_text=(
            f"`in_buffer` HR (no_nlcd) = {_fmt(cox_hr_no)}, p={_fmt_p(cox_p_no)}; "
            f"(with_nlcd) = {_fmt(cox_hr_lu)}, p={_fmt_p(cox_p_lu)}; "
            f"HR change(with-no) = {_fmt(cox_delta)} (threshold 90%)."
        ),
        figures=[
            ("Kaplan-Meier 曲线", "project/modeling_report/figures/cox/cox_km_curve.png"),
            ("风险比图 Hazard Ratio Plot", "project/modeling_report/figures/cox/cox_hazard_ratio.png"),
            ("PH 检验图 Proportional Hazard Test", "project/modeling_report/figures/cox/cox_ph_test.png"),
        ],
        interpretation="HR>1 表示更快达到恢复阈值；若显著则支持缓冲区恢复优势。",
        limitations="恢复定义受阈值影响；观测窗口长度不一致会影响删失比例。",
        next_step="执行 80/90/95% 阈值敏感性并与主结果对照。",
        problem_section=_build_problem_section("Cox"),
    )

    # index
    idx_lines = [
        "# Modeling Report Index",
        "",
        "## Deliverables",
        "- `project/modeling_report/01_ols_report.md`",
        "- `project/modeling_report/02_mixedlm_report.md`",
        "- `project/modeling_report/03_logit_report.md`",
        "- `project/modeling_report/04_cox_report.md`",
        "",
        "## Cross-model key points",
    ]

    for _, r in summary.iterrows():
        idx_lines.append(
            f"- {r['model']} ({r['variant']}): {r['key_metric']} = {r['value']:.4f}"
            + (f", p={r['p_value']:.4g}" if pd.notna(r['p_value']) else "")
        )

    idx_lines.extend(
        [
            "",
            "## Land-use Control Delta (with_nlcd - no_nlcd)",
        ]
    )

    compare_targets = [
        ("OLS", "coef_in_buffer"),
        ("MixedLM", "coef_in_buffer"),
        ("Logit", "odds_ratio_in_buffer"),
        ("Cox", "hazard_ratio_in_buffer"),
    ]
    for model_name, metric in compare_targets:
        v0, _ = _metric(model_name, "no_nlcd", metric)
        v1, _ = _metric(model_name, "with_nlcd", metric)
        diff = v1 - v0 if pd.notna(v0) and pd.notna(v1) else np.nan
        idx_lines.append(f"- {model_name} `{metric}`: no_nlcd={_fmt(v0)}, with_nlcd={_fmt(v1)}, delta={_fmt(diff)}")

    idx_lines.extend(
        [
            "",
            "## Consistency & Conflict",
            "- 若 OLS/MixedLM 与 Logit/Cox 对 in_buffer 的方向一致，则支持备用发电机韧性信号存在。",
            "- 若方向冲突，优先检查阈值敏感性、样本构成和删失结构。",
            "",
            "## Citation-ready statement",
            "- 在六事件统一像素框架下，控制基线亮度与事件异质性后，关键设施缓冲区在夜光恢复/损伤概率上展现出可检验的韧性差异。",
        ]
    )

    (REPORT_DIR / "index.md").write_text("\n".join(idx_lines), encoding="utf-8")
    append_progress("Generated four standalone model reports and report index")


# ----------------------------
# Save consolidated outputs
# ----------------------------

def save_model_outputs(
    ols_mixed_tables: List[Dict[str, pd.DataFrame]],
    logit_tables: List[Dict[str, pd.DataFrame]],
    cox_tables: List[Dict[str, pd.DataFrame]],
) -> None:
    ols_rows = []
    mixed_rows = []
    logit_rows = []
    marg_rows = []
    cox_rows = []

    for t in ols_mixed_tables:
        if not t["ols_coef"].empty:
            ols_rows.append(t["ols_coef"])
        if not t["mixed_coef"].empty:
            mixed_rows.append(t["mixed_coef"])

    for t in logit_tables:
        if not t["coef"].empty:
            logit_rows.append(t["coef"])
        if not t["marginal"].empty:
            marg_rows.append(t["marginal"])

    for t in cox_tables:
        if not t["coef"].empty:
            cox_rows.append(t["coef"])

    if ols_rows:
        pd.concat(ols_rows, ignore_index=True).to_csv(OUTPUT_DIR / "ols_results.csv", index=False)
    else:
        pd.DataFrame(
            columns=["model", "variant", "term", "coef", "std_err", "stat", "p_value", "ci_low", "ci_high", "kind"]
        ).to_csv(OUTPUT_DIR / "ols_results.csv", index=False)

    if mixed_rows:
        pd.concat(mixed_rows, ignore_index=True).to_csv(OUTPUT_DIR / "mixedlm_results.csv", index=False)
    else:
        pd.DataFrame(
            columns=["model", "variant", "term", "coef", "std_err", "stat", "p_value", "ci_low", "ci_high", "kind"]
        ).to_csv(OUTPUT_DIR / "mixedlm_results.csv", index=False)

    if logit_rows:
        pd.concat(logit_rows, ignore_index=True).to_csv(OUTPUT_DIR / "logit_results.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "logit_results.csv", index=False)

    if marg_rows:
        pd.concat(marg_rows, ignore_index=True).to_csv(OUTPUT_DIR / "logit_marginal_effects.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "logit_marginal_effects.csv", index=False)

    if cox_rows:
        pd.concat(cox_rows, ignore_index=True).to_csv(OUTPUT_DIR / "cox_results.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "cox_results.csv", index=False)


# ----------------------------
# Pipeline orchestrator
# ----------------------------

def run_pipeline() -> None:
    ensure_directories()
    init_tracking_files()
    append_progress("Pipeline started")

    defaults = load_json(CONFIG_DEFAULTS)
    ctx = RunContext(issues=[])

    pre_thr = float(defaults["pre_ntl_threshold"])
    dmg_thr = float(defaults["damage_threshold"])
    rec_thr = float(defaults["recovery_threshold"])

    # Stage 1: baseline panel and models
    panel = build_pixel_panel(
        ctx,
        pre_threshold=pre_thr,
        damage_threshold=dmg_thr,
        exclude_types=None,
        output_path=PANEL_PATH,
    )

    panel = attach_cloud_features(ctx=ctx, panel_path=PANEL_PATH, output_path=PANEL_PATH)

    ols_mixed_tables: List[Dict[str, pd.DataFrame]] = []
    logit_tables: List[Dict[str, pd.DataFrame]] = []
    cox_tables: List[Dict[str, pd.DataFrame]] = []

    om_base = fit_ols_and_mixed(ctx, panel, variant="no_nlcd", include_land_use=False)
    ols_mixed_tables.append(om_base)

    logit_base = fit_logit(ctx, panel, variant="no_nlcd", include_land_use=False, damage_threshold=dmg_thr)
    logit_tables.append(logit_base)
    logit_base["roc"].to_csv(OUTPUT_DIR / "logit_roc_no_nlcd.csv", index=False)
    logit_base["calibration"].to_csv(OUTPUT_DIR / "logit_calibration_no_nlcd.csv", index=False)

    rec_base = build_recovery_panel(ctx, panel, threshold=rec_thr, output_path=RECOVERY_PATH)
    cox_base = fit_cox(ctx, rec_base, variant="no_nlcd", include_land_use=False)
    cox_tables.append(cox_base)
    cox_base["km"].to_csv(OUTPUT_DIR / "cox_km_no_nlcd.csv", index=False)
    cox_base["ph"].to_csv(OUTPUT_DIR / "cox_ph_test_no_nlcd.csv", index=False)

    # Stage 2: NLCD attach and rerun if available
    panel_nlcd = attach_nlcd(ctx, panel_path=PANEL_PATH, output_path=PANEL_NLCD_PATH)
    has_land_use = panel_nlcd["land_use"].notna().mean() > 0.2

    if has_land_use:
        panel_nlcd = panel_nlcd[panel_nlcd["land_use"].isin([21, 22, 23, 24])].copy()

        om_lu = fit_ols_and_mixed(ctx, panel_nlcd, variant="with_nlcd", include_land_use=True)
        ols_mixed_tables.append(om_lu)

        logit_lu = fit_logit(ctx, panel_nlcd, variant="with_nlcd", include_land_use=True, damage_threshold=dmg_thr)
        logit_tables.append(logit_lu)
        logit_lu["roc"].to_csv(OUTPUT_DIR / "logit_roc_with_nlcd.csv", index=False)
        logit_lu["calibration"].to_csv(OUTPUT_DIR / "logit_calibration_with_nlcd.csv", index=False)

        rec_lu = build_recovery_panel(ctx, panel_nlcd, threshold=rec_thr, output_path=None)
        cox_lu = fit_cox(ctx, rec_lu, variant="with_nlcd", include_land_use=True)
        cox_tables.append(cox_lu)
        cox_lu["km"].to_csv(OUTPUT_DIR / "cox_km_with_nlcd.csv", index=False)
        cox_lu["ph"].to_csv(OUTPUT_DIR / "cox_ph_test_with_nlcd.csv", index=False)

        append_progress("NLCD coverage sufficient; reran all 4 models with land_use controls")
    else:
        log_issue(
            ctx,
            stage="run_pipeline",
            model="all",
            event_id="all",
            issue_type="insufficient_nlcd_coverage",
            symptom="land_use coverage <= 20%",
            fix_action="skip with_nlcd model reruns",
            impact="only no_nlcd model variant available",
            status="open",
        )

    save_model_outputs(ols_mixed_tables, logit_tables, cox_tables)

    # Stage 3: robustness
    run_robustness(ctx, panel, defaults=defaults, include_land_use=False)

    # Stage 4: figures and reports
    build_model_summary_for_report()
    save_issue_log(ctx)
    generate_figures()
    generate_reports()

    append_progress("Pipeline finished")
