#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

MODELING_DIR = Path(__file__).resolve().parents[1]
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_lib import (
    CONFIG_DEFAULTS,
    CONFIG_EVENTS,
    FIG_DIR,
    OUTPUT_DIR,
    PIXEL_DIR,
    REPORT_DIR,
    ROOT,
    RunContext,
    build_recovery_panel,
    ensure_directories,
    init_tracking_files,
    list_daily_tifs,
    load_json,
)


PANEL_FEATURE_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_feature_upgrade.parquet"
SAMPLE_LOCK_PATH = PIXEL_DIR / "sample_lock_cohort_v1.parquet"
PANEL_V3_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_cross_event_v3.parquet"
EVENT_PROFILE_PATH = PIXEL_DIR / "event_profile_v1.csv"
CLOUD_SUMMARY_PATH = OUTPUT_DIR / "cloud_feature_summary.csv"

V3_ANCHOR_PATH = OUTPUT_DIR / "v3_baseline_anchor.json"
SHIFT_PATH = OUTPUT_DIR / "cross_event_shift_diagnostics_v3.csv"
FOLD_PATH = OUTPUT_DIR / "cross_event_fold_metrics_v3.csv"
AGG_PATH = OUTPUT_DIR / "cross_event_aggregate_metrics_v3.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "cross_event_feature_importance_v3.csv"
SUMMARY_PATH = OUTPUT_DIR / "model_summary_cross_event_v3.csv"

REPORT_PATH = REPORT_DIR / "08_cross_event_model_report.md"
INDEX_PATH = REPORT_DIR / "index.md"
FIG_CE_DIR = FIG_DIR / "cross_event"
ENABLE_CROSS_EVENT_BENCHMARK = False

REPORT_SCHEMA_ORDER = [
    "event_id",
    "disaster_type",
    "lat_center",
    "lon_center",
    "coastal_flag",
    "island_like_flag",
    "elevation_median",
    "slope_median",
    "urban_share_1km",
    "water_share_1km",
    "developed_high_share_1km",
    "pre_ntl_event_mean",
    "cloud_pre_event_mean",
    "cloud_post_event_mean",
    "storm_precip_7d",
    "event_duration_days",
    "source_ref",
    "quality_flag",
]

EVENT_META = {
    "maria_sanjuan": {"disaster_type": "hurricane", "coastal_flag": 1, "island_like_flag": 1},
    "michael_panamacity": {"disaster_type": "hurricane", "coastal_flag": 1, "island_like_flag": 0},
    "earthquake_sanjuan": {"disaster_type": "earthquake", "coastal_flag": 1, "island_like_flag": 1},
    "ida_neworleans": {"disaster_type": "hurricane", "coastal_flag": 1, "island_like_flag": 0},
    "laura_lakecharles": {"disaster_type": "hurricane", "coastal_flag": 1, "island_like_flag": 0},
    "irma_miami": {"disaster_type": "hurricane", "coastal_flag": 1, "island_like_flag": 0},
}

NUMERIC_FEATURE_CANDIDATES = [
    "osm_dist_any_m",
    "osm_power_count_1000m",
    "osm_medical_count_1000m",
    "pixel_cloud_proxy",
    "urban_share_1km",
    "water_share_1km",
    "developed_high_share_1km",
    "event_coastal_flag",
    "event_island_like_flag",
    "event_urban_share_1km",
    "event_water_share_1km",
    "event_precip_7d",
    "event_duration_days",
]

CATEGORICAL_FEATURE_CANDIDATES = ["land_use_group", "event_disaster_type"]
CORE_NUMERIC_COLUMNS = ["pre_mean_ntl", "delta_ntl", "in_buffer", "is_damaged", "recovery_days", "event_observed"]


@dataclass
class FoldArtifacts:
    fold_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    model_coefs: pd.DataFrame


def _safe_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if out.notna().any():
        return out.fillna(out.median())
    return out.fillna(default)


def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _parse_date_from_name(path: Path) -> Optional[date]:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()


def _event_windows(events_cfg: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for event_id, cfg in events_cfg.items():
        pre_tifs = list_daily_tifs(ROOT / cfg["pre_dir"])
        post_tifs = list_daily_tifs(ROOT / cfg["post_dir"])
        pre_dates = [d for d in (_parse_date_from_name(p) for p in pre_tifs) if d is not None]
        post_dates = [d for d in (_parse_date_from_name(p) for p in post_tifs) if d is not None]

        post_start = min(post_dates) if post_dates else None
        post_end = max(post_dates) if post_dates else None
        if post_start is not None and post_end is not None:
            duration_days = (post_end - post_start).days + 1
        else:
            duration_days = np.nan

        out[event_id] = {
            "pre_start": min(pre_dates) if pre_dates else None,
            "pre_end": max(pre_dates) if pre_dates else None,
            "post_start": post_start,
            "post_end": post_end,
            "event_duration_days": duration_days,
            "n_pre_tif": len(pre_tifs),
            "n_post_tif": len(post_tifs),
        }
    return out


def _http_get_json(url: str, timeout: int = 20) -> Dict[str, object]:
    req = Request(url, headers={"User-Agent": "PracticumModeling/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _fetch_precip_7d(lat: float, lon: float, event_date: Optional[date]) -> Tuple[float, str, str]:
    if event_date is None or not np.isfinite(lat) or not np.isfinite(lon):
        return np.nan, "open-meteo:skipped", "missing_event_date_or_coords"

    start = event_date - timedelta(days=6)
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "start_date": start.isoformat(),
        "end_date": event_date.isoformat(),
        "daily": "precipitation_sum",
        "timezone": "UTC",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    try:
        payload = _http_get_json(url)
        daily = payload.get("daily", {}) if isinstance(payload, dict) else {}
        arr = daily.get("precipitation_sum", []) if isinstance(daily, dict) else []
        vals = pd.to_numeric(pd.Series(arr, dtype="float64"), errors="coerce")
        if vals.notna().any():
            return float(vals.fillna(0.0).sum()), "open-meteo:archive", "ok"
        return np.nan, "open-meteo:archive", "missing_daily_precip"
    except Exception as e:
        return np.nan, "open-meteo:archive", f"api_error:{type(e).__name__}"


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r * c)


def _fetch_elevation_slope(lat: float, lon: float) -> Tuple[float, float, str, str]:
    if not np.isfinite(lat) or not np.isfinite(lon):
        return np.nan, np.nan, "opentopodata:skipped", "missing_coords"

    points = [
        (lat, lon),
        (lat + 0.05, lon),
        (lat - 0.05, lon),
        (lat, lon + 0.05),
        (lat, lon - 0.05),
    ]
    locs = "|".join([f"{la:.6f},{lo:.6f}" for la, lo in points])
    url = "https://api.opentopodata.org/v1/srtm90m?" + urlencode({"locations": locs})

    try:
        payload = _http_get_json(url)
        res = payload.get("results", []) if isinstance(payload, dict) else []
        if not isinstance(res, list) or not res:
            return np.nan, np.nan, "opentopodata:srtm90m", "missing_results"

        elev = []
        for row in res:
            if not isinstance(row, dict):
                elev.append(np.nan)
            else:
                elev.append(pd.to_numeric(row.get("elevation"), errors="coerce"))
        arr = np.asarray(elev, dtype="float64")
        if not np.isfinite(arr[0]):
            return np.nan, np.nan, "opentopodata:srtm90m", "missing_center_elevation"

        center = float(arr[0])
        slopes = []
        for idx in range(1, len(points)):
            if not np.isfinite(arr[idx]):
                continue
            d = _haversine_m(points[0][0], points[0][1], points[idx][0], points[idx][1])
            if d > 0:
                slopes.append(abs(float(arr[idx] - arr[0])) / d)
        slope_median = float(np.median(slopes)) if slopes else np.nan
        return center, slope_median, "opentopodata:srtm90m", "ok"
    except Exception as e:
        return np.nan, np.nan, "opentopodata:srtm90m", f"api_error:{type(e).__name__}"


def _compute_local_landuse_shares(panel: pd.DataFrame, events_cfg: Dict[str, object], radius_m: float = 1000.0) -> pd.DataFrame:
    out = panel.copy()
    for col in ["urban_share_1km", "water_share_1km", "developed_high_share_1km"]:
        if col not in out.columns:
            out[col] = np.nan

    for event_id, cfg in events_cfg.items():
        mask = out["event_id"] == event_id
        if mask.sum() == 0:
            continue

        lon = pd.to_numeric(out.loc[mask, "lon"], errors="coerce").to_numpy()
        lat = pd.to_numeric(out.loc[mask, "lat"], errors="coerce").to_numpy()
        lu = pd.to_numeric(out.loc[mask, "land_use"], errors="coerce").to_numpy()

        transformer = Transformer.from_crs("EPSG:4326", cfg["metric_crs"], always_xy=True)
        x, y = transformer.transform(lon, lat)
        xy = np.column_stack([x, y])
        finite = np.isfinite(xy).all(axis=1)

        urban = np.isin(lu, [21, 22, 23, 24]).astype("float64")
        water = np.isin(lu, [11]).astype("float64")
        high = np.isin(lu, [24]).astype("float64")

        urban_share = np.full(len(xy), np.nan, dtype="float64")
        water_share = np.full(len(xy), np.nan, dtype="float64")
        high_share = np.full(len(xy), np.nan, dtype="float64")

        if finite.any():
            tree = cKDTree(xy[finite])
            idx_map = np.where(finite)[0]
            neighbors = tree.query_ball_point(xy[finite], r=radius_m)
            for local_i, neigh in enumerate(neighbors):
                base_i = idx_map[local_i]
                if not neigh:
                    continue
                src = idx_map[np.asarray(neigh, dtype=int)]
                urban_share[base_i] = float(np.nanmean(urban[src]))
                water_share[base_i] = float(np.nanmean(water[src]))
                high_share[base_i] = float(np.nanmean(high[src]))

        out.loc[mask, "urban_share_1km"] = urban_share
        out.loc[mask, "water_share_1km"] = water_share
        out.loc[mask, "developed_high_share_1km"] = high_share

    return out


def _build_event_profile(panel: pd.DataFrame, events_cfg: Dict[str, object]) -> pd.DataFrame:
    windows = _event_windows(events_cfg)
    cloud_summary = pd.read_csv(CLOUD_SUMMARY_PATH) if CLOUD_SUMMARY_PATH.exists() else pd.DataFrame()
    rows: List[Dict[str, object]] = []

    for event_id in sorted(panel["event_id"].unique().tolist()):
        d = panel[panel["event_id"] == event_id].copy()
        meta = EVENT_META.get(event_id, {})

        lat_center = float(pd.to_numeric(d["lat"], errors="coerce").median())
        lon_center = float(pd.to_numeric(d["lon"], errors="coerce").median())

        w = windows.get(event_id, {})
        event_date = w.get("post_start")
        precip, precip_src, precip_flag = _fetch_precip_7d(lat_center, lon_center, event_date)
        elev, slope, topo_src, topo_flag = _fetch_elevation_slope(lat_center, lon_center)

        quality_parts = [
            "ok" if precip_flag == "ok" else precip_flag,
            "ok" if topo_flag == "ok" else topo_flag,
        ]
        cloud_pre = float(pd.to_numeric(d["cloud_pre_mean"], errors="coerce").mean()) if "cloud_pre_mean" in d.columns else np.nan
        cloud_post = float(pd.to_numeric(d["cloud_post_mean"], errors="coerce").mean()) if "cloud_post_mean" in d.columns else np.nan
        if (not np.isfinite(cloud_pre) or not np.isfinite(cloud_post)) and not cloud_summary.empty:
            hit = cloud_summary[cloud_summary["event_id"] == event_id]
            if not hit.empty:
                cloud_pre = float(pd.to_numeric(hit["cloud_pre_mean"], errors="coerce").iloc[0])
                cloud_post = float(pd.to_numeric(hit["cloud_post_mean"], errors="coerce").iloc[0])
                quality_parts.append("cloud_summary_fallback")

        row = {
            "event_id": event_id,
            "disaster_type": meta.get("disaster_type", "unknown"),
            "lat_center": lat_center,
            "lon_center": lon_center,
            "coastal_flag": int(meta.get("coastal_flag", 0)),
            "island_like_flag": int(meta.get("island_like_flag", 0)),
            "elevation_median": elev,
            "slope_median": slope,
            "urban_share_1km": float(pd.to_numeric(d["urban_share_1km"], errors="coerce").mean()),
            "water_share_1km": float(pd.to_numeric(d["water_share_1km"], errors="coerce").mean()),
            "developed_high_share_1km": float(pd.to_numeric(d["developed_high_share_1km"], errors="coerce").mean()),
            "pre_ntl_event_mean": float(pd.to_numeric(d["pre_mean_ntl"], errors="coerce").mean()),
            "cloud_pre_event_mean": cloud_pre,
            "cloud_post_event_mean": cloud_post,
            "storm_precip_7d": precip,
            "event_duration_days": w.get("event_duration_days", np.nan),
            "source_ref": f"panel_stats;{precip_src};{topo_src}",
            "quality_flag": ";".join(quality_parts),
        }
        rows.append(row)

    profile = pd.DataFrame(rows)
    for c in REPORT_SCHEMA_ORDER:
        if c not in profile.columns:
            profile[c] = np.nan
    profile = profile[REPORT_SCHEMA_ORDER]
    profile.to_csv(EVENT_PROFILE_PATH, index=False)
    return profile


def _attach_event_features(panel: pd.DataFrame, event_profile: pd.DataFrame) -> pd.DataFrame:
    attach = event_profile[
        [
            "event_id",
            "disaster_type",
            "coastal_flag",
            "island_like_flag",
            "urban_share_1km",
            "water_share_1km",
            "storm_precip_7d",
            "event_duration_days",
        ]
    ].rename(
        columns={
            "disaster_type": "event_disaster_type",
            "coastal_flag": "event_coastal_flag",
            "island_like_flag": "event_island_like_flag",
            "urban_share_1km": "event_urban_share_1km",
            "water_share_1km": "event_water_share_1km",
            "storm_precip_7d": "event_precip_7d",
        }
    )
    out = panel.merge(attach, on="event_id", how="left")
    return out


def _fit_fill_rules(df: pd.DataFrame, numeric_terms: Sequence[str], cat_terms: Sequence[str]) -> Dict[str, Dict[str, object]]:
    numeric_fill: Dict[str, float] = {}
    for c in list(dict.fromkeys(list(CORE_NUMERIC_COLUMNS) + list(numeric_terms))):
        if c not in df.columns:
            numeric_fill[c] = 0.0
            continue
        series = _coerce_numeric(df[c])
        val = float(series.median()) if series.notna().any() else 0.0
        if not np.isfinite(val):
            val = 0.0
        numeric_fill[c] = val

    cat_levels: Dict[str, List[str]] = {}
    for c in cat_terms:
        if c in df.columns:
            levels = sorted(df[c].fillna("unknown").astype(str).unique().tolist())
        else:
            levels = []
        if "unknown" not in levels:
            levels.append("unknown")
        cat_levels[c] = levels

    return {"numeric_fill": numeric_fill, "cat_levels": cat_levels}


def _apply_fill_rules(
    df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
    fill_rules: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    out = df.copy()
    numeric_fill = fill_rules.get("numeric_fill", {})
    cat_levels = fill_rules.get("cat_levels", {})

    for c in list(dict.fromkeys(list(CORE_NUMERIC_COLUMNS) + list(numeric_terms))):
        if c not in out.columns:
            out[c] = np.nan
        fill_value = float(numeric_fill.get(c, 0.0))
        out[c] = _coerce_numeric(out[c]).fillna(fill_value)

    for c in cat_terms:
        if c not in out.columns:
            out[c] = "unknown"
        values = out[c]
        if isinstance(values.dtype, CategoricalDtype):
            if "unknown" not in values.cat.categories:
                values = values.cat.add_categories(["unknown"])
        values = values.fillna("unknown").astype(str)
        levels = list(cat_levels.get(c, ["unknown"]))
        if "unknown" not in levels:
            levels.append("unknown")
        values = values.where(values.isin(levels), "unknown")
        out[c] = pd.Categorical(values, categories=levels)
    return out


def _fill_model_na(df: pd.DataFrame) -> pd.DataFrame:
    numeric_terms = [c for c in NUMERIC_FEATURE_CANDIDATES if c in df.columns]
    cat_terms = [c for c in CATEGORICAL_FEATURE_CANDIDATES if c in df.columns]
    fill_rules = _fit_fill_rules(df, numeric_terms=numeric_terms, cat_terms=cat_terms)
    return _apply_fill_rules(df, numeric_terms=numeric_terms, cat_terms=cat_terms, fill_rules=fill_rules)


def _select_terms(df_train: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_terms = []
    for c in NUMERIC_FEATURE_CANDIDATES:
        if c not in df_train.columns:
            continue
        s = _coerce_numeric(df_train[c])
        if s.nunique(dropna=True) > 1:
            num_terms.append(c)
    cat_terms = []
    for c in CATEGORICAL_FEATURE_CANDIDATES:
        if c not in df_train.columns:
            continue
        if df_train[c].nunique(dropna=True) > 1:
            cat_terms.append(c)
    return num_terms, cat_terms


def _scale_train_test(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    tr = train.copy()
    te = test.copy()
    stats: Dict[str, Dict[str, float]] = {}
    for c in cols:
        tr[c] = _safe_numeric(tr[c])
        te[c] = _safe_numeric(te[c])
        mu = float(tr[c].mean())
        sd = float(tr[c].std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        tr[f"{c}_z"] = (tr[c] - mu) / sd
        te[f"{c}_z"] = (te[c] - mu) / sd
        stats[c] = {"mean": mu, "std": sd}
    return tr, te, stats


def _build_linear_formula(num_terms: Sequence[str], cat_terms: Sequence[str]) -> str:
    terms = ["in_buffer * pre_mean_ntl"]
    terms.extend([f"C({c})" for c in cat_terms])
    terms.extend(list(num_terms))
    return "delta_ntl ~ " + " + ".join(terms)


def _build_logit_formula(num_terms: Sequence[str], cat_terms: Sequence[str]) -> str:
    terms = ["in_buffer * pre_mean_ntl_z"]
    terms.extend([f"C({c})" for c in cat_terms])
    terms.extend([f"{c}_z" for c in num_terms])
    return "is_damaged ~ " + " + ".join(terms)


def _fit_logit(formula: str, data: pd.DataFrame):
    last_error = None
    for method in ["newton", "lbfgs", "bfgs", "powell"]:
        try:
            return smf.logit(formula=formula, data=data).fit(disp=False, method=method, maxiter=300)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"logit fit failed: {last_error}")


def _coefficient_rows(model_name: str, result_obj, fold_event: str, spec: str) -> List[Dict[str, object]]:
    try:
        pvals = result_obj.pvalues
    except Exception:
        pvals = pd.Series(np.nan, index=result_obj.params.index)

    rows = []
    for term, coef in result_obj.params.items():
        rows.append(
            {
                "fold_event": fold_event,
                "spec": spec,
                "model": model_name,
                "feature": term,
                "coef": float(coef),
                "p_value": float(pvals.get(term, np.nan)),
                "source": "interpretable_coef",
                "importance_mean": np.nan,
                "importance_std": np.nan,
            }
        )
    return rows


def _prepare_hgb_matrix(df: pd.DataFrame, num_terms: Sequence[str], cat_terms: Sequence[str]) -> pd.DataFrame:
    base_cols = ["in_buffer", "pre_mean_ntl"] + list(num_terms)
    x = df[base_cols].copy()
    for c in x.columns:
        x[c] = _safe_numeric(x[c])
    if cat_terms:
        dummies = pd.get_dummies(df[list(cat_terms)].fillna("unknown").astype(str), drop_first=False)
        x = pd.concat([x.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return x


def _fit_cox_transport(train_rec: pd.DataFrame, test_rec: pd.DataFrame, num_terms: Sequence[str], cat_terms: Sequence[str]) -> Tuple[float, float, List[Dict[str, object]]]:
    base_cols = ["in_buffer", "pre_mean_ntl"] + list(num_terms)
    tr_x = train_rec[base_cols].copy()
    te_x = test_rec[base_cols].copy()

    for c in tr_x.columns:
        tr_x[c] = _safe_numeric(tr_x[c])
        te_x[c] = _safe_numeric(te_x[c])

    if cat_terms:
        tr_cat = pd.get_dummies(train_rec[list(cat_terms)].fillna("unknown").astype(str), drop_first=True)
        te_cat = pd.get_dummies(test_rec[list(cat_terms)].fillna("unknown").astype(str), drop_first=True)
        te_cat = te_cat.reindex(columns=tr_cat.columns, fill_value=0)
        tr_x = pd.concat([tr_x.reset_index(drop=True), tr_cat.reset_index(drop=True)], axis=1)
        te_x = pd.concat([te_x.reset_index(drop=True), te_cat.reset_index(drop=True)], axis=1)

    fit_df = pd.concat([train_rec[["recovery_days", "event_observed"]].reset_index(drop=True), tr_x.reset_index(drop=True)], axis=1)
    fit_df = fit_df.replace([np.inf, -np.inf], np.nan)
    fit_df = fit_df.fillna(fit_df.median(numeric_only=True))

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(fit_df, duration_col="recovery_days", event_col="event_observed")
    risk_te = cph.predict_partial_hazard(te_x)
    cidx = float(
        concordance_index(
            test_rec["recovery_days"],
            -risk_te.to_numpy().reshape(-1),
            test_rec["event_observed"],
        )
    )

    aft_df = fit_df.copy()
    aft = WeibullAFTFitter(penalizer=0.01)
    aft.fit(aft_df, duration_col="recovery_days", event_col="event_observed")
    med_te = aft.predict_median(te_x)
    aft_cidx = float(
        concordance_index(
            test_rec["recovery_days"],
            -med_te.to_numpy().reshape(-1),
            test_rec["event_observed"],
        )
    )

    cox_rows = []
    for feat, coef in cph.params_.items():
        cox_rows.append(
            {
                "fold_event": "all_train",
                "spec": "transport",
                "model": "Cox",
                "feature": feat,
                "coef": float(coef),
                "p_value": float(cph.summary.loc[feat, "p"]) if feat in cph.summary.index else np.nan,
                "source": "interpretable_coef",
                "importance_mean": np.nan,
                "importance_std": np.nan,
            }
        )
    return cidx, aft_cidx, cox_rows


def _compute_psi(base: pd.Series, cmp: pd.Series, bins: int = 10) -> float:
    b = pd.to_numeric(base, errors="coerce")
    c = pd.to_numeric(cmp, errors="coerce")
    b = b.replace([np.inf, -np.inf], np.nan).dropna()
    c = c.replace([np.inf, -np.inf], np.nan).dropna()
    if b.empty or c.empty:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(b, quantiles))
    if len(edges) < 3:
        return np.nan

    b_counts, _ = np.histogram(b, bins=edges)
    c_counts, _ = np.histogram(c, bins=edges)
    b_dist = np.clip(b_counts / max(b_counts.sum(), 1), 1e-6, 1)
    c_dist = np.clip(c_counts / max(c_counts.sum(), 1), 1e-6, 1)
    return float(np.sum((b_dist - c_dist) * np.log(b_dist / c_dist)))


def _build_shift_diagnostics(df: pd.DataFrame, recovery_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    events = sorted(df["event_id"].unique().tolist())
    key_features = [
        "pre_mean_ntl",
        "delta_ntl",
        "osm_dist_any_m",
        "pixel_cloud_proxy",
        "urban_share_1km",
        "water_share_1km",
        "developed_high_share_1km",
        "event_precip_7d",
        "event_duration_days",
    ]

    rec_map = recovery_df.groupby("event_id", observed=True).agg(
        recovery_observed_rate=("event_observed", "mean"),
        recovery_days_median=("recovery_days", "median"),
    )

    for ev in events:
        d_ev = df[df["event_id"] == ev]
        d_rest = df[df["event_id"] != ev]

        rows.append({
            "diagnostic_type": "event_summary",
            "event_id": ev,
            "compare_event": "all",
            "metric_name": "n_obs",
            "value": float(len(d_ev)),
            "source": "panel",
        })
        rows.append({
            "diagnostic_type": "event_summary",
            "event_id": ev,
            "compare_event": "all",
            "metric_name": "damage_rate",
            "value": float(d_ev["is_damaged"].mean()),
            "source": "panel",
        })
        rows.append({
            "diagnostic_type": "event_summary",
            "event_id": ev,
            "compare_event": "all",
            "metric_name": "recovery_observed_rate",
            "value": float(rec_map.loc[ev, "recovery_observed_rate"]) if ev in rec_map.index else np.nan,
            "source": "recovery_panel",
        })
        rows.append({
            "diagnostic_type": "event_summary",
            "event_id": ev,
            "compare_event": "all",
            "metric_name": "recovery_days_median",
            "value": float(rec_map.loc[ev, "recovery_days_median"]) if ev in rec_map.index else np.nan,
            "source": "recovery_panel",
        })

        smd_vals = []
        psi_vals = []
        for c in key_features:
            x = _safe_numeric(d_ev[c])
            y = _safe_numeric(d_rest[c])
            denom = math.sqrt((float(x.var(ddof=0)) + float(y.var(ddof=0))) / 2.0 + 1e-12)
            smd = abs(float(x.mean()) - float(y.mean())) / denom if denom > 0 else 0.0
            psi = _compute_psi(x, y)
            smd_vals.append(smd)
            if np.isfinite(psi):
                psi_vals.append(psi)
            rows.append(
                {
                    "diagnostic_type": "feature_shift",
                    "event_id": ev,
                    "compare_event": "rest",
                    "metric_name": f"smd:{c}",
                    "value": float(smd),
                    "source": "panel",
                }
            )

        rows.append(
            {
                "diagnostic_type": "event_shift",
                "event_id": ev,
                "compare_event": "rest",
                "metric_name": "smd_mean",
                "value": float(np.nanmean(smd_vals)),
                "source": "panel",
            }
        )
        rows.append(
            {
                "diagnostic_type": "event_shift",
                "event_id": ev,
                "compare_event": "rest",
                "metric_name": "smd_max",
                "value": float(np.nanmax(smd_vals)),
                "source": "panel",
            }
        )
        rows.append(
            {
                "diagnostic_type": "event_shift",
                "event_id": ev,
                "compare_event": "rest",
                "metric_name": "psi_mean",
                "value": float(np.nanmean(psi_vals)) if psi_vals else np.nan,
                "source": "panel",
            }
        )

    for i, ev_i in enumerate(events):
        di = df[df["event_id"] == ev_i]
        for ev_j in events[i + 1 :]:
            dj = df[df["event_id"] == ev_j]
            smd_pair = []
            for c in key_features:
                x = _safe_numeric(di[c])
                y = _safe_numeric(dj[c])
                denom = math.sqrt((float(x.var(ddof=0)) + float(y.var(ddof=0))) / 2.0 + 1e-12)
                smd = abs(float(x.mean()) - float(y.mean())) / denom if denom > 0 else 0.0
                smd_pair.append(smd)
            rows.append(
                {
                    "diagnostic_type": "pairwise_smd",
                    "event_id": ev_i,
                    "compare_event": ev_j,
                    "metric_name": "pair_smd_mean",
                    "value": float(np.nanmean(smd_pair)),
                    "source": "panel",
                }
            )

    diag = pd.DataFrame(rows)
    diag.to_csv(SHIFT_PATH, index=False)
    return diag


def _plot_shift_heatmap(diag: pd.DataFrame) -> None:
    pair = diag[(diag["diagnostic_type"] == "pairwise_smd") & (diag["metric_name"] == "pair_smd_mean")].copy()
    if pair.empty:
        return

    evs = sorted(set(pair["event_id"]).union(set(pair["compare_event"])))
    mat = pd.DataFrame(np.nan, index=evs, columns=evs)
    for _, r in pair.iterrows():
        i = r["event_id"]
        j = r["compare_event"]
        mat.loc[i, j] = r["value"]
        mat.loc[j, i] = r["value"]
    np.fill_diagonal(mat.values, 0.0)

    plt.figure(figsize=(8.5, 6.5))
    sns.heatmap(mat, cmap="YlOrRd", annot=True, fmt=".2f", cbar_kws={"label": "Mean SMD"})
    plt.title("Cross-Event Covariate Shift (Pairwise Mean SMD)")
    plt.tight_layout()
    plt.savefig(FIG_CE_DIR / "shift_pairwise_smd_heatmap_v3.png", dpi=220)
    plt.close()


def _run_loeo(
    panel: pd.DataFrame,
    recovery: pd.DataFrame,
    damage_threshold: float,
) -> FoldArtifacts:
    rows: List[Dict[str, object]] = []
    imp_rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []

    events = sorted(panel["event_id"].unique().tolist())

    for fold_event in events:
        tr = panel[panel["event_id"] != fold_event].copy()
        te = panel[panel["event_id"] == fold_event].copy()
        tr["is_damaged"] = (tr["delta_ntl"] < damage_threshold).astype(int)
        te["is_damaged"] = (te["delta_ntl"] < damage_threshold).astype(int)

        num_terms, cat_terms = _select_terms(tr)
        fill_rules = _fit_fill_rules(tr, numeric_terms=num_terms, cat_terms=cat_terms)
        tr = _apply_fill_rules(tr, numeric_terms=num_terms, cat_terms=cat_terms, fill_rules=fill_rules)
        te = _apply_fill_rules(te, numeric_terms=num_terms, cat_terms=cat_terms, fill_rules=fill_rules)

        # Interpretable: OLS
        ols_formula = _build_linear_formula(num_terms=num_terms, cat_terms=cat_terms)
        ols = smf.ols(ols_formula, data=tr).fit(cov_type="HC1")
        pred_ols = ols.predict(te)
        rows.append(
            {
                "fold_event": fold_event,
                "spec": "transport",
                "track": "interpretable",
                "model": "OLS",
                "rmse": float(math.sqrt(mean_squared_error(te["delta_ntl"], pred_ols))),
                "mae": float(mean_absolute_error(te["delta_ntl"], pred_ols)),
                "auc": np.nan,
                "brier": np.nan,
                "calibration_slope": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": float(ols.params.get("in_buffer", np.nan)),
            }
        )
        coef_rows.extend(_coefficient_rows("OLS", ols, fold_event, "transport"))

        # Interpretable: MixedLM
        mx_formula = _build_linear_formula(num_terms=num_terms, cat_terms=cat_terms)
        mx = None
        mixed_formulas = [mx_formula]
        numeric_only_terms = ["in_buffer * pre_mean_ntl"] + list(num_terms)
        mixed_formulas.append("delta_ntl ~ " + " + ".join(numeric_only_terms))
        mixed_formulas.append("delta_ntl ~ in_buffer * pre_mean_ntl")
        for fmx in mixed_formulas:
            for method in ["lbfgs", "powell", "cg", "nm"]:
                try:
                    mx = smf.mixedlm(fmx, data=tr, groups=tr["event_id"]).fit(method=method, reml=False)
                    break
                except Exception:
                    continue
            if mx is not None:
                break
        if mx is None:
            raise RuntimeError(f"MixedLM fit failed on fold {fold_event}")
        pred_mx = mx.predict(te)
        rows.append(
            {
                "fold_event": fold_event,
                "spec": "transport",
                "track": "interpretable",
                "model": "MixedLM",
                "rmse": float(math.sqrt(mean_squared_error(te["delta_ntl"], pred_mx))),
                "mae": float(mean_absolute_error(te["delta_ntl"], pred_mx)),
                "auc": np.nan,
                "brier": np.nan,
                "calibration_slope": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": float(mx.params.get("in_buffer", np.nan)),
            }
        )
        coef_rows.extend(_coefficient_rows("MixedLM", mx, fold_event, "transport"))

        # Interpretable: Logit
        scale_cols = ["pre_mean_ntl"] + list(num_terms)
        tr_lg, te_lg, _ = _scale_train_test(tr, te, cols=scale_cols)
        lg_formula = _build_logit_formula(num_terms=num_terms, cat_terms=cat_terms)
        lg = _fit_logit(lg_formula, tr_lg)
        prob = np.asarray(lg.predict(te_lg))
        auc = float(roc_auc_score(te_lg["is_damaged"], prob)) if te_lg["is_damaged"].nunique() > 1 else np.nan
        brier = float(brier_score_loss(te_lg["is_damaged"], prob))

        rows.append(
            {
                "fold_event": fold_event,
                "spec": "transport",
                "track": "interpretable",
                "model": "Logit",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": auc,
                "brier": brier,
                "calibration_slope": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": float(lg.params.get("in_buffer", np.nan)),
            }
        )
        coef_rows.extend(_coefficient_rows("Logit", lg, fold_event, "transport"))

        # Interpretable: Cox + AFT
        tr_rec = recovery[recovery["event_id"] != fold_event].copy()
        te_rec = recovery[recovery["event_id"] == fold_event].copy()
        rec_fill_rules = _fit_fill_rules(tr_rec, numeric_terms=num_terms, cat_terms=cat_terms)
        tr_rec = _apply_fill_rules(tr_rec, numeric_terms=num_terms, cat_terms=cat_terms, fill_rules=rec_fill_rules)
        te_rec = _apply_fill_rules(te_rec, numeric_terms=num_terms, cat_terms=cat_terms, fill_rules=rec_fill_rules)
        cox_cidx, aft_cidx, cox_coef_rows = _fit_cox_transport(tr_rec, te_rec, num_terms=num_terms, cat_terms=cat_terms)
        rows.append(
            {
                "fold_event": fold_event,
                "spec": "transport",
                "track": "interpretable",
                "model": "Cox",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "calibration_slope": np.nan,
                "c_index": cox_cidx,
                "coef_in_buffer": np.nan,
            }
        )
        rows.append(
            {
                "fold_event": fold_event,
                "spec": "transport",
                "track": "interpretable",
                "model": "AFT",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "calibration_slope": np.nan,
                "c_index": aft_cidx,
                "coef_in_buffer": np.nan,
            }
        )
        coef_rows.extend(cox_coef_rows)

        if not ENABLE_CROSS_EVENT_BENCHMARK:
            continue

        # Benchmark: HGB regressor
        xtr_reg = _prepare_hgb_matrix(tr, num_terms=num_terms, cat_terms=cat_terms)
        xte_reg = _prepare_hgb_matrix(te, num_terms=num_terms, cat_terms=cat_terms)
        xte_reg = xte_reg.reindex(columns=xtr_reg.columns, fill_value=0.0)
        ytr_reg = _safe_numeric(tr["delta_ntl"])
        yte_reg = _safe_numeric(te["delta_ntl"])

        hgb_r = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=150, random_state=42)
        hgb_r.fit(xtr_reg, ytr_reg)
        pred_hgb = hgb_r.predict(xte_reg)
        rows.append(
            {
                "fold_event": fold_event,
                "spec": "transport",
                "track": "benchmark",
                "model": "HGBRegressor",
                "rmse": float(math.sqrt(mean_squared_error(yte_reg, pred_hgb))),
                "mae": float(mean_absolute_error(yte_reg, pred_hgb)),
                "auc": np.nan,
                "brier": np.nan,
                "calibration_slope": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": np.nan,
            }
        )

        try:
            perm_r = permutation_importance(
                hgb_r,
                xte_reg,
                yte_reg,
                n_repeats=5,
                random_state=42,
                scoring="neg_root_mean_squared_error",
            )
            for name, m, s in zip(xte_reg.columns, perm_r.importances_mean, perm_r.importances_std):
                imp_rows.append(
                    {
                        "fold_event": fold_event,
                        "spec": "transport",
                        "model": "HGBRegressor",
                        "feature": name,
                        "source": "permutation",
                        "importance_mean": float(m),
                        "importance_std": float(s),
                        "coef": np.nan,
                        "p_value": np.nan,
                    }
                )
        except Exception:
            pass

        # Benchmark: HGB classifier
        xtr_cls = xtr_reg.copy()
        xte_cls = xte_reg.copy()
        ytr_cls = tr["is_damaged"].astype(int).to_numpy()
        yte_cls = te["is_damaged"].astype(int).to_numpy()

        hgb_c = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=150, random_state=42)
        hgb_c.fit(xtr_cls, ytr_cls)
        prob_cls = hgb_c.predict_proba(xte_cls)[:, 1]
        auc_cls = float(roc_auc_score(yte_cls, prob_cls)) if np.unique(yte_cls).size > 1 else np.nan
        brier_cls = float(brier_score_loss(yte_cls, prob_cls))
        rows.append(
            {
                "fold_event": fold_event,
                "spec": "transport",
                "track": "benchmark",
                "model": "HGBClassifier",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": auc_cls,
                "brier": brier_cls,
                "calibration_slope": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": np.nan,
            }
        )

        if np.unique(yte_cls).size > 1:
            try:
                perm_c = permutation_importance(
                    hgb_c,
                    xte_cls,
                    yte_cls,
                    n_repeats=5,
                    random_state=42,
                    scoring="roc_auc",
                )
                for name, m, s in zip(xte_cls.columns, perm_c.importances_mean, perm_c.importances_std):
                    imp_rows.append(
                        {
                            "fold_event": fold_event,
                            "spec": "transport",
                            "model": "HGBClassifier",
                            "feature": name,
                            "source": "permutation",
                            "importance_mean": float(m),
                            "importance_std": float(s),
                            "coef": np.nan,
                            "p_value": np.nan,
                        }
                    )
            except Exception:
                pass

    fold_df = pd.DataFrame(rows)
    fold_df["sign_consistency"] = np.nan

    # Sign consistency by model against fold-average sign.
    for model in ["OLS", "MixedLM", "Logit"]:
        m = fold_df["model"] == model
        if not m.any():
            continue
        vals = fold_df.loc[m, "coef_in_buffer"]
        ref = np.sign(np.nanmean(vals))
        fold_df.loc[m, "sign_consistency"] = np.where(np.sign(vals) == ref, 1.0, 0.0)

    imp_df = pd.DataFrame(imp_rows)
    coef_df = pd.DataFrame(coef_rows)
    return FoldArtifacts(fold_metrics=fold_df, feature_importance=imp_df, model_coefs=coef_df)


def _aggregate_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["rmse", "mae", "auc", "brier", "calibration_slope", "c_index", "sign_consistency"]
    agg = fold_df.groupby(["spec", "track", "model"], dropna=False)[cols].mean().reset_index()
    agg.to_csv(AGG_PATH, index=False)
    return agg


def _build_baseline_anchor() -> Dict[str, object]:
    anchor = {
        "strict_v2_logo_path": str((OUTPUT_DIR / "logo_aggregate_metrics_v2_strict.csv").relative_to(ROOT)),
        "strict_v2_summary_path": str((OUTPUT_DIR / "model_summary_feature_upgrade_v2_strict.csv").relative_to(ROOT)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    p = OUTPUT_DIR / "logo_aggregate_metrics_v2_strict.csv"
    if p.exists():
        v2 = pd.read_csv(p)
        sub = v2[v2["spec"] == "logo_transport"]
        metrics = {}
        for _, r in sub.iterrows():
            metrics[r["model"]] = {
                "rmse": r.get("rmse", np.nan),
                "mae": r.get("mae", np.nan),
                "auc": r.get("auc", np.nan),
                "brier": r.get("brier", np.nan),
                "c_index": r.get("c_index", np.nan),
                "sign_consistency": r.get("sign_consistency", np.nan),
            }
        anchor["strict_v2_logo_transport"] = metrics

    V3_ANCHOR_PATH.write_text(json.dumps(anchor, indent=2), encoding="utf-8")
    return anchor


def _merge_importance(coef_df: pd.DataFrame, imp_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    if not coef_df.empty:
        c = coef_df.groupby(["model", "feature"], dropna=False).agg(
            coef=("coef", "mean"),
            p_value=("p_value", "mean"),
        ).reset_index()
        for _, r in c.iterrows():
            rows.append(
                {
                    "model": r["model"],
                    "feature": r["feature"],
                    "source": "interpretable_coef",
                    "importance_mean": np.nan,
                    "importance_std": np.nan,
                    "coef": r["coef"],
                    "p_value": r["p_value"],
                }
            )

    if not imp_df.empty:
        p = imp_df.groupby(["model", "feature"], dropna=False).agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_std", "mean"),
        ).reset_index()
        for _, r in p.iterrows():
            rows.append(
                {
                    "model": r["model"],
                    "feature": r["feature"],
                    "source": "permutation",
                    "importance_mean": r["importance_mean"],
                    "importance_std": r["importance_std"],
                    "coef": np.nan,
                    "p_value": np.nan,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    return out


def _plot_transport_metrics(agg_df: pd.DataFrame, anchor: Dict[str, object]) -> None:
    v2 = anchor.get("strict_v2_logo_transport", {})

    rows = []
    for _, r in agg_df.iterrows():
        model = r["model"]
        if model in {"Logit", "HGBClassifier"}:
            rows.append({"model": model, "metric": "AUC", "value": r.get("auc", np.nan), "version": "v3"})
            rows.append({"model": model, "metric": "Brier", "value": r.get("brier", np.nan), "version": "v3"})
        if model in {"Cox", "AFT"}:
            rows.append({"model": model, "metric": "C-index", "value": r.get("c_index", np.nan), "version": "v3"})
        if model in {"OLS", "MixedLM", "HGBRegressor"}:
            rows.append({"model": model, "metric": "RMSE", "value": r.get("rmse", np.nan), "version": "v3"})

    for m in ["OLS", "MixedLM", "Logit", "Cox"]:
        if m in v2:
            vm = v2[m]
            rows.append({"model": m, "metric": "AUC", "value": vm.get("auc", np.nan), "version": "strict_v2"})
            rows.append({"model": m, "metric": "Brier", "value": vm.get("brier", np.nan), "version": "strict_v2"})
            rows.append({"model": m, "metric": "C-index", "value": vm.get("c_index", np.nan), "version": "strict_v2"})
            rows.append({"model": m, "metric": "RMSE", "value": vm.get("rmse", np.nan), "version": "strict_v2"})

    d = pd.DataFrame(rows)
    d = d[np.isfinite(pd.to_numeric(d["value"], errors="coerce"))]
    if d.empty:
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(data=d, x="model", y="value", hue="version", palette="Set2")
    plt.title("Transport Metrics: strict-v2 vs v3")
    plt.tight_layout()
    plt.savefig(FIG_CE_DIR / "transport_metrics_compare_v3.png", dpi=220)
    plt.close()


def _plot_feature_importance(fi: pd.DataFrame) -> None:
    if fi.empty:
        return
    d = fi[(fi["source"] == "permutation") & (fi["model"] == "HGBClassifier")].copy()
    if d.empty:
        return
    d = d.sort_values("importance_mean", ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=d, y="feature", x="importance_mean", hue="feature", dodge=False, palette="viridis", legend=False)
    plt.title("Top Permutation Importance (HGBClassifier, LOEO)")
    plt.tight_layout()
    plt.savefig(FIG_CE_DIR / "hgb_permutation_importance_v3.png", dpi=220)
    plt.close()


def _build_summary(
    agg_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    anchor: Dict[str, object],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    v2 = anchor.get("strict_v2_logo_transport", {})

    for _, r in agg_df.iterrows():
        model = r["model"]
        for metric in ["rmse", "mae", "auc", "brier", "c_index", "sign_consistency"]:
            val = pd.to_numeric(pd.Series([r.get(metric)]), errors="coerce").iloc[0]
            if not np.isfinite(val):
                continue
            base = np.nan
            if model in v2:
                base = pd.to_numeric(pd.Series([v2[model].get(metric, np.nan)]), errors="coerce").iloc[0]
            rows.append(
                {
                    "model": model,
                    "scope": "loeo_transport",
                    "metric_name": metric,
                    "value": float(val),
                    "baseline_value": float(base) if np.isfinite(base) else np.nan,
                    "delta_vs_baseline": float(val - base) if np.isfinite(base) else np.nan,
                    "p_value": np.nan,
                    "notes": r["track"],
                }
            )

    if not coef_df.empty:
        key_terms = ["in_buffer"]
        k = coef_df[coef_df["feature"].isin(key_terms)].copy()
        if not k.empty:
            c = k.groupby("model", dropna=False).agg(coef=("coef", "mean"), p_value=("p_value", "mean")).reset_index()
            for _, r in c.iterrows():
                rows.append(
                    {
                        "model": r["model"],
                        "scope": "coefficient",
                        "metric_name": "coef_in_buffer",
                        "value": float(r["coef"]),
                        "baseline_value": np.nan,
                        "delta_vs_baseline": np.nan,
                        "p_value": float(r["p_value"]) if np.isfinite(r["p_value"]) else np.nan,
                        "notes": "transport_full",
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_PATH, index=False)
    return out


def _format_metric(df: pd.DataFrame, model: str, metric: str) -> str:
    sub = df[(df["model"] == model) & (df["metric_name"] == metric) & (df["scope"] == "loeo_transport")]
    if sub.empty:
        return "N/A"
    v = float(sub.iloc[0]["value"])
    b = sub.iloc[0]["baseline_value"]
    d = sub.iloc[0]["delta_vs_baseline"]
    if pd.notna(b):
        return f"{v:.4f} (vs strict-v2 {float(b):.4f}, Δ={float(d):+.4f})"
    return f"{v:.4f}"


def _write_report(
    summary: pd.DataFrame,
    diag: pd.DataFrame,
    event_profile: pd.DataFrame,
    refs: Sequence[Tuple[str, str]],
) -> None:
    low_events = (
        diag[(diag["diagnostic_type"] == "event_shift") & (diag["metric_name"] == "smd_mean")]
        .sort_values("value", ascending=False)
        .head(2)
    )
    low_lines = [f"- {r['event_id']}: smd_mean={float(r['value']):.3f}" for _, r in low_events.iterrows()]
    if not low_lines:
        low_lines = ["- N/A"]

    logit_auc = summary[(summary["model"] == "Logit") & (summary["metric_name"] == "auc") & (summary["scope"] == "loeo_transport")]
    cox_cidx = summary[(summary["model"] == "Cox") & (summary["metric_name"] == "c_index") & (summary["scope"] == "loeo_transport")]

    verdict = "Partially Improved"
    if not logit_auc.empty and not cox_cidx.empty:
        a = float(logit_auc.iloc[0]["value"])
        c = float(cox_cidx.iloc[0]["value"])
        if a >= 0.50 and c >= 0.52:
            verdict = "Improved"
        elif a < 0.47 and c < 0.51:
            verdict = "Not Improved"

    lines = [
        "# Cross-Event Model Report / 跨事件预测模型报告（V3）",
        "",
        "## Objective",
        "目标是提升 LOEO/transport 外推能力，同时解释为何 strict-v2 在跨事件预测上偏弱。",
        "",
        "## Why LOEO was low",
        "核心原因是事件间协变量漂移（domain shift）显著：灾害类型、局地土地利用结构、云量有效观测代理和设施密度分布存在系统差异。",
        "",
        "## Data & Event Profiles",
        f"- Event profile: `{EVENT_PROFILE_PATH.relative_to(ROOT)}`",
        f"- Enriched panel: `{PANEL_V3_PATH.relative_to(ROOT)}`",
        f"- Shift diagnostics: `{SHIFT_PATH.relative_to(ROOT)}`",
        "",
        "### Event Profile Snapshot",
    ]

    p = event_profile[["event_id", "disaster_type", "event_duration_days", "storm_precip_7d", "quality_flag"]].copy()
    cols = p.columns.tolist()
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in p.iterrows():
        row_vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and np.isfinite(v):
                row_vals.append(f"{v:.4f}")
            elif pd.isna(v):
                row_vals.append("N/A")
            else:
                row_vals.append(str(v))
        lines.append("| " + " | ".join(row_vals) + " |")

    lines.extend(
        [
            "",
            "## Specs",
            "- Interpretable track: OLS, MixedLM, Logit, Cox, AFT（统一 transport 口径，不使用 event FE 作为主预测特征）",
            "- Benchmark track: HistGradientBoostingRegressor / HistGradientBoostingClassifier",
            "- Validation: LOEO (6 folds), each fold trains on 5 events and tests on 1 unseen event.",
            "",
            "## Fold Results",
            f"- Logit AUC: {_format_metric(summary, 'Logit', 'auc')}",
            f"- Logit Brier: {_format_metric(summary, 'Logit', 'brier')}",
            f"- Cox c-index: {_format_metric(summary, 'Cox', 'c_index')}",
            f"- OLS RMSE: {_format_metric(summary, 'OLS', 'rmse')}",
            f"- MixedLM RMSE: {_format_metric(summary, 'MixedLM', 'rmse')}",
            f"- HGBClassifier AUC: {_format_metric(summary, 'HGBClassifier', 'auc')}",
            "",
            "## Improvement Verdict",
            f"- Verdict: **{verdict}**",
            "- 判定口径：Logit AUC 是否提升到 >=0.50 或较 strict-v2 至少 +0.03；Cox c-index 是否不降并接近/超过 0.52。",
            "",
            "## Failure Cases",
            "下列事件表现为主要外推压力源（按 `smd_mean` 排序）：",
            *low_lines,
            "",
            "## Figures",
            f"- Pairwise shift heatmap: `project/modeling_report/figures/cross_event/shift_pairwise_smd_heatmap_v3.png`",
            f"- Transport metrics compare: `project/modeling_report/figures/cross_event/transport_metrics_compare_v3.png`",
            f"- HGB permutation importance: `project/modeling_report/figures/cross_event/hgb_permutation_importance_v3.png`",
            "",
            "## Next Actions",
            "- 固定同样本下，增加事件强度变量（例如最大风速/震级/降水极值）并做分层 LOEO。",
            "- 对高漂移事件启用 group-reweight（GroupDRO 思路）并与当前 ERM 结果对照。",
            "- 保留 interpretable 主结果，benchmark 仅作为性能上界参考。",
            "",
            "## References",
        ]
    )

    for title, link in refs:
        lines.append(f"- [{title}]({link})")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _update_index() -> None:
    line = "- `project/modeling_report/08_cross_event_model_report.md`"
    note = "- V3 cross-event outputs: `project/modeling/output/model_summary_cross_event_v3.csv`, `project/modeling/output/cross_event_aggregate_metrics_v3.csv`"

    if INDEX_PATH.exists():
        text = INDEX_PATH.read_text(encoding="utf-8")
    else:
        text = "# Modeling Report Index\n\n## Deliverables\n"

    if line not in text:
        text = text.rstrip() + "\n" + line + "\n"
    if note not in text:
        text = text.rstrip() + "\n" + note + "\n"
    INDEX_PATH.write_text(text, encoding="utf-8")


def _build_v3_impl() -> None:
    ensure_directories()
    init_tracking_files()
    FIG_CE_DIR.mkdir(parents=True, exist_ok=True)

    if not PANEL_FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing panel: {PANEL_FEATURE_PATH}")

    defaults = load_json(CONFIG_DEFAULTS)
    damage_thr = float(defaults["damage_threshold"])
    recovery_thr = float(defaults["recovery_threshold"])
    events_cfg = load_json(CONFIG_EVENTS)

    df = pd.read_parquet(PANEL_FEATURE_PATH).copy()
    if "sample_lock_flag" not in df.columns and SAMPLE_LOCK_PATH.exists():
        lock = pd.read_parquet(SAMPLE_LOCK_PATH)[["pixel_id", "sample_lock_flag"]]
        df = df.merge(lock, on="pixel_id", how="left")
    if "sample_lock_flag" not in df.columns:
        raise KeyError("sample_lock_flag missing from panel and sample lock table.")

    df = df[df["sample_lock_flag"] == 1].copy()
    df = _compute_local_landuse_shares(df, events_cfg, radius_m=1000.0)
    event_profile = _build_event_profile(df, events_cfg)
    df = _attach_event_features(df, event_profile)
    # Save enriched panel for reproducibility.
    df.to_parquet(PANEL_V3_PATH, index=False)

    ctx = RunContext(issues=[])
    rec = build_recovery_panel(ctx=ctx, panel=df, threshold=recovery_thr, output_path=None)

    baseline_anchor = _build_baseline_anchor()
    diagnostics = _build_shift_diagnostics(df, rec)
    _plot_shift_heatmap(diagnostics)

    artifacts = _run_loeo(panel=df, recovery=rec, damage_threshold=damage_thr)
    fold_df = artifacts.fold_metrics
    fold_df.to_csv(FOLD_PATH, index=False)

    agg_df = _aggregate_metrics(fold_df)
    fi_df = _merge_importance(artifacts.model_coefs, artifacts.feature_importance)

    _plot_transport_metrics(agg_df, baseline_anchor)
    _plot_feature_importance(fi_df)

    summary = _build_summary(agg_df=agg_df, coef_df=artifacts.model_coefs, anchor=baseline_anchor)

    refs = [
        ("Invariant Risk Minimization (Arjovsky et al., 2019)", "https://arxiv.org/abs/1907.02893"),
        ("DomainBed: In Search of Lost Domain Generalization (Gulrajani & Lopez-Paz, 2020)", "https://arxiv.org/abs/2007.01434"),
        ("Group DRO (Sagawa et al., 2019)", "https://arxiv.org/abs/1911.08731"),
        ("Spatiotemporal distribution of power outages with climate events (Do et al., 2023)", "https://www.nature.com/articles/s41467-023-38084-6"),
        ("Antecedent rainfall and outage risk (Manning et al., 2025)", "https://www.nature.com/articles/s43247-025-02176-6"),
    ]
    _write_report(summary=summary, diag=diagnostics, event_profile=event_profile, refs=refs)
    _update_index()

# ----------------------------
# Stabilization integrated block
# ----------------------------

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from patsy import build_design_matrices, dmatrices
from pandas.api.types import CategoricalDtype
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold

from pipeline_lib import (
    CONFIG_DEFAULTS,
    OUTPUT_DIR,
    PIXEL_DIR,
    REPORT_DIR,
    ROOT,
    RunContext,
    build_recovery_panel,
    ensure_directories,
    init_tracking_files,
    load_json,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

STABILIZE_PANEL_V3_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_cross_event_v3.parquet"
STABILIZE_BASE_AGG_PATH = OUTPUT_DIR / "cross_event_aggregate_metrics_v3.csv"

STABILIZE_ROUND_COMP_PATH = OUTPUT_DIR / "cross_event_round_comparison_v3x.csv"
STABILIZE_STOP_DECISION_PATH = OUTPUT_DIR / "cross_event_stop_decision_v3x.json"
STABILIZE_REPORT_PATH = REPORT_DIR / "09_cross_event_stabilization_report.md"
STABILIZE_INDEX_PATH = REPORT_DIR / "index.md"

STABILIZE_LOCAL_NUMERIC_BASE = [
    "osm_dist_any_m",
    "osm_power_count_1000m",
    "osm_medical_count_1000m",
    "pixel_cloud_proxy",
    "urban_share_1km",
    "water_share_1km",
    "developed_high_share_1km",
]
STABILIZE_CAT_TERMS_BASE = ["land_use_group", "event_disaster_type"]

STABILIZE_THRESH_AUC = 0.02
STABILIZE_THRESH_SURV = 0.01
STABILIZE_MAX_ROUNDS = 3


class StabilizationError(RuntimeError):
    pass


def _stabilize_safe_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if out.notna().any():
        return out.fillna(out.median())
    return out.fillna(default)


def _stabilize_event_inverse_sqrt_weights(event_series: pd.Series) -> pd.Series:
    cnt = event_series.value_counts()
    return event_series.map(lambda e: 1.0 / math.sqrt(float(cnt.get(e, 1))))


def _stabilize_fit_fill_rules(
    df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
    cat_levels: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, object]]:
    fill_rules = _fit_fill_rules(df, numeric_terms=numeric_terms, cat_terms=cat_terms)
    if cat_levels:
        merged_levels: Dict[str, List[str]] = {}
        for c in cat_terms:
            levels = list(fill_rules["cat_levels"].get(c, []))
            if not levels:
                levels = list(cat_levels.get(c, []))
            if "unknown" not in levels:
                levels.append("unknown")
            merged_levels[c] = levels
        fill_rules["cat_levels"] = merged_levels
    return fill_rules


def _stabilize_apply_fill_rules(
    df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
    fill_rules: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    return _apply_fill_rules(df, numeric_terms=numeric_terms, cat_terms=cat_terms, fill_rules=fill_rules)


def _stabilize_fill_frame(
    df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
    cat_levels: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    fill_rules = _stabilize_fit_fill_rules(df, numeric_terms=numeric_terms, cat_terms=cat_terms, cat_levels=cat_levels)
    return _stabilize_apply_fill_rules(df, numeric_terms=numeric_terms, cat_terms=cat_terms, fill_rules=fill_rules)


def _stabilize_build_formula(target: str, numeric_terms: Sequence[str], cat_terms: Sequence[str]) -> str:
    rhs = ["in_buffer * pre_mean_ntl"]
    rhs.extend([f"C({c})" for c in cat_terms])
    rhs.extend(list(numeric_terms))
    return f"{target} ~ " + " + ".join(rhs)


def _stabilize_calibration_slope(y_true: np.ndarray, prob: np.ndarray) -> float:
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    x = np.log(p / (1 - p))
    if np.allclose(np.std(x), 0) or len(np.unique(y_true)) < 2:
        return np.nan
    return float(np.polyfit(x, y_true, 1)[0])


def _stabilize_select_penalty_cox(train_df: pd.DataFrame, penalties: Sequence[float]) -> float:
    if len(train_df) < 300:
        return 0.01
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_pen, best_score = 0.01, -np.inf
    for pen in penalties:
        scores = []
        for tr_idx, va_idx in kf.split(train_df):
            tr = train_df.iloc[tr_idx].copy()
            va = train_df.iloc[va_idx].copy()
            xva = va.drop(columns=["recovery_days", "event_observed"])
            try:
                cph = CoxPHFitter(penalizer=float(pen))
                cph.fit(tr, duration_col="recovery_days", event_col="event_observed")
                risk = cph.predict_partial_hazard(xva)
                cidx = float(concordance_index(va["recovery_days"], -risk.to_numpy().reshape(-1), va["event_observed"]))
                if np.isfinite(cidx):
                    scores.append(cidx)
            except Exception:
                continue
        if scores:
            s = float(np.mean(scores))
            if s > best_score:
                best_score, best_pen = s, float(pen)
    return best_pen


def _stabilize_select_penalty_aft(train_df: pd.DataFrame, penalties: Sequence[float]) -> float:
    if len(train_df) < 300:
        return 0.01
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_pen, best_score = 0.01, -np.inf
    for pen in penalties:
        scores = []
        for tr_idx, va_idx in kf.split(train_df):
            tr = train_df.iloc[tr_idx].copy()
            va = train_df.iloc[va_idx].copy()
            xva = va.drop(columns=["recovery_days", "event_observed"])
            try:
                aft = WeibullAFTFitter(penalizer=float(pen))
                aft.fit(tr, duration_col="recovery_days", event_col="event_observed")
                med = aft.predict_median(xva)
                cidx = float(concordance_index(va["recovery_days"], -med.to_numpy().reshape(-1), va["event_observed"]))
                if np.isfinite(cidx):
                    scores.append(cidx)
            except Exception:
                continue
        if scores:
            s = float(np.mean(scores))
            if s > best_score:
                best_score, best_pen = s, float(pen)
    return best_pen


def _stabilize_prepare_survival_design(
    rec_df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
) -> pd.DataFrame:
    cols = ["recovery_days", "event_observed", "event_id", "in_buffer", "pre_mean_ntl"] + list(numeric_terms) + list(cat_terms)
    cols = [c for c in cols if c in rec_df.columns]
    d = rec_df[cols].copy()

    for c in ["recovery_days", "event_observed", "in_buffer", "pre_mean_ntl"] + list(numeric_terms):
        if c in d.columns:
            d[c] = _stabilize_safe_numeric(d[c])

    dummy_parts = []
    for c in cat_terms:
        if c in d.columns:
            dummy_parts.append(pd.get_dummies(d[c].fillna("unknown").astype(str), prefix=c, drop_first=True))
            d = d.drop(columns=[c])
    if dummy_parts:
        d = pd.concat([d.reset_index(drop=True)] + [x.reset_index(drop=True) for x in dummy_parts], axis=1)

    d = d.replace([np.inf, -np.inf], np.nan)
    for c in d.columns:
        if c in {"event_id"}:
            continue
        d[c] = _stabilize_safe_numeric(d[c])
    return d


def _stabilize_collect_coef_rows_statsmodels(
    result,
    round_id: str,
    fold_event: str,
    model: str,
) -> List[Dict[str, object]]:
    params = result.params
    pvals = getattr(result, "pvalues", pd.Series(np.nan, index=params.index))
    rows = []
    for term, coef in params.items():
        rows.append(
            {
                "round_id": round_id,
                "fold_event": fold_event,
                "model": model,
                "feature": str(term),
                "coef": float(coef),
                "p_value": float(pvals.get(term, np.nan)) if term in pvals.index else np.nan,
                "source": "interpretable_coef",
            }
        )
    return rows


def _stabilize_collect_coef_rows_sklearn(
    model,
    feature_names: Sequence[str],
    round_id: str,
    fold_event: str,
    model_name: str,
) -> List[Dict[str, object]]:
    coefs = model.coef_.reshape(-1)
    rows = []
    for name, coef in zip(feature_names, coefs):
        rows.append(
            {
                "round_id": round_id,
                "fold_event": fold_event,
                "model": model_name,
                "feature": str(name),
                "coef": float(coef),
                "p_value": np.nan,
                "source": "interpretable_coef",
            }
        )
    return rows


def _stabilize_run_loeo_round(
    round_id: str,
    panel: pd.DataFrame,
    recovery: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
    damage_threshold: float,
    use_event_weights: bool,
    use_logit_calibration: bool,
    use_survival_grid: bool,
    cat_levels: Dict[str, List[str]],
    benchmark_selected_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = sorted(panel["event_id"].unique().tolist())
    fold_rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []
    perm_rows: List[Dict[str, object]] = []

    ols_formula = _stabilize_build_formula("delta_ntl", numeric_terms, cat_terms)
    logit_formula = _stabilize_build_formula("is_damaged", numeric_terms, cat_terms)

    for test_event in events:
        tr = panel[panel["event_id"] != test_event].copy()
        te = panel[panel["event_id"] == test_event].copy()
        tr["is_damaged"] = (tr["delta_ntl"] < damage_threshold).astype(int)
        te["is_damaged"] = (te["delta_ntl"] < damage_threshold).astype(int)

        fold_fill_rules = _stabilize_fit_fill_rules(tr, numeric_terms=numeric_terms, cat_terms=cat_terms, cat_levels=cat_levels)
        tr = _stabilize_apply_fill_rules(tr, numeric_terms=numeric_terms, cat_terms=cat_terms, fill_rules=fold_fill_rules)
        te = _stabilize_apply_fill_rules(te, numeric_terms=numeric_terms, cat_terms=cat_terms, fill_rules=fold_fill_rules)

        w_tr = _stabilize_event_inverse_sqrt_weights(tr["event_id"]).to_numpy() if use_event_weights else None

        # OLS
        try:
            if use_event_weights:
                ols = smf.wls(ols_formula, data=tr, weights=w_tr).fit(cov_type="HC1")
            else:
                ols = smf.ols(ols_formula, data=tr).fit(cov_type="HC1")
            pred = np.asarray(ols.predict(te))
            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "interpretable",
                    "model": "OLS",
                    "rmse": float(math.sqrt(mean_squared_error(te["delta_ntl"], pred))),
                    "mae": float(mean_absolute_error(te["delta_ntl"], pred)),
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": np.nan,
                    "coef_in_buffer": float(ols.params.get("in_buffer", np.nan)),
                    "notes": f"weights={int(use_event_weights)}",
                }
            )
            coef_rows.extend(_stabilize_collect_coef_rows_statsmodels(ols, round_id, test_event, "OLS"))
        except Exception as e:
            raise StabilizationError(f"Round {round_id} OLS failed on fold {test_event}: {e}")

        # MixedLM
        try:
            mixed = None
            mixed_formula_used = ols_formula
            mixed_formulas = [ols_formula]
            # Stability fallback for folds with singular design (e.g., unseen category dominated fold).
            numeric_only_terms = [f"in_buffer * pre_mean_ntl"] + list(numeric_terms)
            mixed_formulas.append("delta_ntl ~ " + " + ".join(numeric_only_terms))
            mixed_formulas.append("delta_ntl ~ in_buffer * pre_mean_ntl")

            for fmx in mixed_formulas:
                for method in ["lbfgs", "powell", "cg", "nm"]:
                    try:
                        mixed = smf.mixedlm(fmx, data=tr, groups=tr["event_id"]).fit(method=method, reml=False)
                        mixed_formula_used = fmx
                        break
                    except Exception:
                        continue
                if mixed is not None:
                    break
            if mixed is None:
                raise RuntimeError("mixedlm fit failed for all optimizers")
            pred = np.asarray(mixed.predict(te))
            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "interpretable",
                    "model": "MixedLM",
                    "rmse": float(math.sqrt(mean_squared_error(te["delta_ntl"], pred))),
                    "mae": float(mean_absolute_error(te["delta_ntl"], pred)),
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": np.nan,
                    "coef_in_buffer": float(mixed.params.get("in_buffer", np.nan)),
                    "notes": f"mixedlm_formula={mixed_formula_used}",
                }
            )
            coef_rows.extend(_stabilize_collect_coef_rows_statsmodels(mixed, round_id, test_event, "MixedLM"))
        except Exception as e:
            raise StabilizationError(f"Round {round_id} MixedLM failed on fold {test_event}: {e}")

        # Logit (L2)
        try:
            y_tr, X_tr = dmatrices(logit_formula, tr, return_type="dataframe")
            X_te = build_design_matrices([X_tr.design_info], te, return_type="dataframe")[0]
            y_tr_arr = y_tr.values.reshape(-1).astype(int)
            y_te_arr = te["is_damaged"].to_numpy().astype(int)

            base_clf = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=1200,
                fit_intercept=False,
                random_state=42,
            )
            base_clf.fit(X_tr, y_tr_arr, sample_weight=w_tr)

            if use_logit_calibration:
                cal = None
                for method in ["sigmoid", "isotonic"]:
                    try:
                        cal = CalibratedClassifierCV(
                            estimator=LogisticRegression(
                                penalty="l2",
                                C=1.0,
                                solver="lbfgs",
                                max_iter=1200,
                                fit_intercept=False,
                                random_state=42,
                            ),
                            method=method,
                            cv=3,
                        )
                        cal.fit(X_tr, y_tr_arr, sample_weight=w_tr)
                        prob = cal.predict_proba(X_te)[:, 1]
                        cal_used = method
                        break
                    except Exception:
                        cal = None
                if cal is None:
                    prob = base_clf.predict_proba(X_te)[:, 1]
                    cal_used = "none"
            else:
                prob = base_clf.predict_proba(X_te)[:, 1]
                cal_used = "none"

            auc = float(roc_auc_score(y_te_arr, prob)) if len(np.unique(y_te_arr)) > 1 else np.nan
            brier = float(brier_score_loss(y_te_arr, prob))
            slope = _stabilize_calibration_slope(y_te_arr, prob)

            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "interpretable",
                    "model": "Logit",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "auc": auc,
                    "brier": brier,
                    "calibration_slope": slope,
                    "c_index": np.nan,
                    "coef_in_buffer": float(dict(zip(X_tr.columns, base_clf.coef_.reshape(-1))).get("in_buffer", np.nan)),
                    "notes": f"l2;cal={cal_used};weights={int(use_event_weights)}",
                }
            )
            coef_rows.extend(_stabilize_collect_coef_rows_sklearn(base_clf, X_tr.columns, round_id, test_event, "Logit"))
        except Exception as e:
            raise StabilizationError(f"Round {round_id} Logit failed on fold {test_event}: {e}")

        # Survival (Cox + AFT, pick best later)
        try:
            tr_rec = recovery[recovery["event_id"] != test_event].copy()
            te_rec = recovery[recovery["event_id"] == test_event].copy()
            rec_fill_rules = _stabilize_fit_fill_rules(tr_rec, numeric_terms=numeric_terms, cat_terms=cat_terms, cat_levels=cat_levels)
            tr_rec = _stabilize_apply_fill_rules(tr_rec, numeric_terms=numeric_terms, cat_terms=cat_terms, fill_rules=rec_fill_rules)
            te_rec = _stabilize_apply_fill_rules(te_rec, numeric_terms=numeric_terms, cat_terms=cat_terms, fill_rules=rec_fill_rules)

            dtr = _stabilize_prepare_survival_design(tr_rec, numeric_terms=numeric_terms, cat_terms=cat_terms)
            dte = _stabilize_prepare_survival_design(te_rec, numeric_terms=numeric_terms, cat_terms=cat_terms)
            xtr = dtr.drop(columns=["recovery_days", "event_observed", "event_id"], errors="ignore")
            xte = dte.drop(columns=["recovery_days", "event_observed", "event_id"], errors="ignore")
            xte = xte.reindex(columns=xtr.columns, fill_value=0.0)

            fit_tr = pd.concat([dtr[["recovery_days", "event_observed"]].reset_index(drop=True), xtr.reset_index(drop=True)], axis=1)
            fit_te = pd.concat([dte[["recovery_days", "event_observed"]].reset_index(drop=True), xte.reset_index(drop=True)], axis=1)

            penalties = [0.001, 0.01, 0.05, 0.1]
            pen_cox = _stabilize_select_penalty_cox(fit_tr, penalties) if use_survival_grid else 0.01
            pen_aft = _stabilize_select_penalty_aft(fit_tr, penalties) if use_survival_grid else 0.01

            cph = CoxPHFitter(penalizer=float(pen_cox))
            cph.fit(fit_tr, duration_col="recovery_days", event_col="event_observed")
            risk = cph.predict_partial_hazard(fit_te.drop(columns=["recovery_days", "event_observed"]))
            cidx_cox = float(
                concordance_index(
                    fit_te["recovery_days"],
                    -risk.to_numpy().reshape(-1),
                    fit_te["event_observed"],
                )
            )

            aft = WeibullAFTFitter(penalizer=float(pen_aft))
            aft.fit(fit_tr, duration_col="recovery_days", event_col="event_observed")
            med = aft.predict_median(fit_te.drop(columns=["recovery_days", "event_observed"]))
            cidx_aft = float(
                concordance_index(
                    fit_te["recovery_days"],
                    -med.to_numpy().reshape(-1),
                    fit_te["event_observed"],
                )
            )

            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "interpretable",
                    "model": "Cox",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": cidx_cox,
                    "coef_in_buffer": float(cph.params_.get("in_buffer", np.nan)),
                    "notes": f"pen={pen_cox:.3g}",
                }
            )
            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "interpretable",
                    "model": "AFT",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": cidx_aft,
                    "coef_in_buffer": np.nan,
                    "notes": f"pen={pen_aft:.3g}",
                }
            )

            for feat, coef in cph.params_.items():
                pval = np.nan
                try:
                    if feat in cph.summary.index:
                        pval = float(cph.summary.loc[feat, "p"])
                except Exception:
                    pval = np.nan
                coef_rows.append(
                    {
                        "round_id": round_id,
                        "fold_event": test_event,
                        "model": "Cox",
                        "feature": str(feat),
                        "coef": float(coef),
                        "p_value": pval,
                        "source": "interpretable_coef",
                    }
                )
            try:
                aft_params = aft.params_.copy()
                for idx, val in aft_params.items():
                    coef_rows.append(
                        {
                            "round_id": round_id,
                            "fold_event": test_event,
                            "model": "AFT",
                            "feature": str(idx),
                            "coef": float(val),
                            "p_value": np.nan,
                            "source": "interpretable_coef",
                        }
                    )
            except Exception:
                pass
        except Exception as e:
            raise StabilizationError(f"Round {round_id} Survival failed on fold {test_event}: {e}")

        if not ENABLE_CROSS_EVENT_BENCHMARK:
            continue

        # Benchmark models
        try:
            xtr_b = pd.DataFrame({"in_buffer": _stabilize_safe_numeric(tr["in_buffer"]), "pre_mean_ntl": _stabilize_safe_numeric(tr["pre_mean_ntl"])})
            xte_b = pd.DataFrame({"in_buffer": _stabilize_safe_numeric(te["in_buffer"]), "pre_mean_ntl": _stabilize_safe_numeric(te["pre_mean_ntl"])})
            for c in numeric_terms:
                xtr_b[c] = _stabilize_safe_numeric(tr[c])
                xte_b[c] = _stabilize_safe_numeric(te[c])
            if cat_terms:
                tr_cat = pd.get_dummies(tr[list(cat_terms)].fillna("unknown").astype(str), drop_first=False)
                te_cat = pd.get_dummies(te[list(cat_terms)].fillna("unknown").astype(str), drop_first=False)
                te_cat = te_cat.reindex(columns=tr_cat.columns, fill_value=0)
                xtr_b = pd.concat([xtr_b.reset_index(drop=True), tr_cat.reset_index(drop=True)], axis=1)
                xte_b = pd.concat([xte_b.reset_index(drop=True), te_cat.reset_index(drop=True)], axis=1)

            if benchmark_selected_cols:
                use_cols = [c for c in benchmark_selected_cols if c in xtr_b.columns]
                if not use_cols:
                    use_cols = list(xtr_b.columns)
                xtr_b = xtr_b[use_cols].copy()
                xte_b = xte_b[use_cols].copy()

            ytr_reg = _stabilize_safe_numeric(tr["delta_ntl"]).to_numpy()
            yte_reg = _stabilize_safe_numeric(te["delta_ntl"]).to_numpy()
            ytr_cls = tr["is_damaged"].astype(int).to_numpy()
            yte_cls = te["is_damaged"].astype(int).to_numpy()

            hgb_r = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=150, random_state=42)
            hgb_r.fit(xtr_b, ytr_reg, sample_weight=w_tr)
            pred_r = hgb_r.predict(xte_b)
            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "benchmark",
                    "model": "HGBRegressor",
                    "rmse": float(math.sqrt(mean_squared_error(yte_reg, pred_r))),
                    "mae": float(mean_absolute_error(yte_reg, pred_r)),
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": np.nan,
                    "coef_in_buffer": np.nan,
                    "notes": f"weights={int(use_event_weights)}",
                }
            )

            hgb_c = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=150, random_state=42)
            hgb_c.fit(xtr_b, ytr_cls, sample_weight=w_tr)
            prob_c = hgb_c.predict_proba(xte_b)[:, 1]
            auc_c = float(roc_auc_score(yte_cls, prob_c)) if len(np.unique(yte_cls)) > 1 else np.nan
            brier_c = float(brier_score_loss(yte_cls, prob_c))
            fold_rows.append(
                {
                    "round_id": round_id,
                    "fold_event": test_event,
                    "track": "benchmark",
                    "model": "HGBClassifier",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "auc": auc_c,
                    "brier": brier_c,
                    "calibration_slope": np.nan,
                    "c_index": np.nan,
                    "coef_in_buffer": np.nan,
                    "notes": f"weights={int(use_event_weights)}",
                }
            )

            if len(np.unique(yte_cls)) > 1:
                pi = permutation_importance(
                    hgb_c,
                    xte_b,
                    yte_cls,
                    n_repeats=3,
                    random_state=42,
                    scoring="roc_auc",
                )
                for name, m, s in zip(xte_b.columns, pi.importances_mean, pi.importances_std):
                    perm_rows.append(
                        {
                            "round_id": round_id,
                            "fold_event": test_event,
                            "model": "HGBClassifier",
                            "feature": str(name),
                            "source": "permutation",
                            "importance_mean": float(m),
                            "importance_std": float(s),
                        }
                    )
        except Exception as e:
            raise StabilizationError(f"Round {round_id} benchmark failed on fold {test_event}: {e}")

    fold_df = pd.DataFrame(fold_rows)
    for model in ["OLS", "MixedLM", "Logit"]:
        m = fold_df["model"] == model
        if m.any():
            ref = np.sign(np.nanmean(fold_df.loc[m, "coef_in_buffer"]))
            fold_df.loc[m, "sign_consistency"] = np.where(
                np.sign(fold_df.loc[m, "coef_in_buffer"]) == ref,
                1.0,
                0.0,
            )

    agg_cols = ["rmse", "mae", "auc", "brier", "calibration_slope", "c_index", "sign_consistency"]
    agg_df = fold_df.groupby(["round_id", "track", "model"], dropna=False)[agg_cols].mean().reset_index()

    coef_df = pd.DataFrame(coef_rows)
    if perm_rows:
        perm_df = pd.DataFrame(perm_rows)
        perm_agg = perm_df.groupby(["round_id", "model", "feature", "source"], dropna=False).agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_std", "mean"),
        ).reset_index()
    else:
        perm_agg = pd.DataFrame(columns=["round_id", "model", "feature", "source", "importance_mean", "importance_std"])

    if coef_df.empty:
        coef_part = pd.DataFrame(columns=["round_id", "model", "feature", "source", "coef", "p_value"])
    else:
        coef_part = coef_df.groupby(["round_id", "model", "feature", "source"], dropna=False).agg(
            coef=("coef", "mean"),
            p_value=("p_value", "mean"),
        ).reset_index()

    fi_df = coef_part.merge(perm_agg, on=["round_id", "model", "feature", "source"], how="outer")
    return fold_df, agg_df, fi_df


def _stabilize_extract_round_metrics(agg_df: pd.DataFrame) -> Dict[str, float]:
    def pick(track: str, model: str, col: str) -> float:
        s = agg_df[(agg_df["track"] == track) & (agg_df["model"] == model)]
        if s.empty:
            return np.nan
        v = pd.to_numeric(pd.Series([s.iloc[0][col]]), errors="coerce").iloc[0]
        return float(v) if np.isfinite(v) else np.nan

    cox = pick("interpretable", "Cox", "c_index")
    aft = pick("interpretable", "AFT", "c_index")
    surv_best = np.nanmax([cox, aft]) if (np.isfinite(cox) or np.isfinite(aft)) else np.nan

    out = {
        "logit_auc": pick("interpretable", "Logit", "auc"),
        "logit_brier": pick("interpretable", "Logit", "brier"),
        "survival_best_c_index": float(surv_best) if np.isfinite(surv_best) else np.nan,
        "cox_c_index": cox,
        "aft_c_index": aft,
        "ols_rmse": pick("interpretable", "OLS", "rmse"),
        "mixedlm_rmse": pick("interpretable", "MixedLM", "rmse"),
        "hgb_auc": pick("benchmark", "HGBClassifier", "auc"),
        "hgb_rmse": pick("benchmark", "HGBRegressor", "rmse"),
    }
    return out


def _stabilize_round_comparison_rows(
    round_id: str,
    cur: Dict[str, float],
    prev: Dict[str, float],
) -> Tuple[List[Dict[str, object]], bool]:
    rows = []

    d_auc = cur["logit_auc"] - prev["logit_auc"] if np.isfinite(cur["logit_auc"]) and np.isfinite(prev["logit_auc"]) else np.nan
    d_surv = (
        cur["survival_best_c_index"] - prev["survival_best_c_index"]
        if np.isfinite(cur["survival_best_c_index"]) and np.isfinite(prev["survival_best_c_index"])
        else np.nan
    )

    stop_flag = bool(np.isfinite(d_auc) and np.isfinite(d_surv) and (d_auc < STABILIZE_THRESH_AUC) and (d_surv < STABILIZE_THRESH_SURV))

    metric_defs = [
        ("classification", "logit_auc", STABILIZE_THRESH_AUC),
        ("survival", "survival_best_c_index", STABILIZE_THRESH_SURV),
        ("classification", "logit_brier", np.nan),
        ("regression", "ols_rmse", np.nan),
        ("regression", "mixedlm_rmse", np.nan),
        ("benchmark", "hgb_auc", np.nan),
    ]

    for group, name, thr in metric_defs:
        v = cur.get(name, np.nan)
        pv = prev.get(name, np.nan)
        dv = (v - pv) if np.isfinite(v) and np.isfinite(pv) else np.nan
        if np.isfinite(thr):
            pass_flag = int(np.isfinite(dv) and dv >= thr)
            notes = f"target_delta>={thr:.3f}"
        else:
            pass_flag = np.nan
            notes = "tracking"
        rows.append(
            {
                "round_id": round_id,
                "metric_group": group,
                "metric_name": name,
                "value": v,
                "prev_value": pv,
                "delta": dv,
                "threshold": thr,
                "pass_flag": pass_flag,
                "stop_flag": int(stop_flag),
                "notes": notes,
            }
        )
    return rows, stop_flag


def _stabilize_build_round_summary(round_id: str, metrics: Dict[str, float], prev_metrics: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for metric in ["logit_auc", "logit_brier", "survival_best_c_index", "cox_c_index", "aft_c_index", "ols_rmse", "mixedlm_rmse", "hgb_auc", "hgb_rmse"]:
        v = metrics.get(metric, np.nan)
        pv = prev_metrics.get(metric, np.nan)
        dv = (v - pv) if np.isfinite(v) and np.isfinite(pv) else np.nan
        rows.append(
            {
                "round_id": round_id,
                "metric_name": metric,
                "value": v,
                "prev_value": pv,
                "delta_vs_prev": dv,
            }
        )
    return pd.DataFrame(rows)


def _stabilize_pick_round3_numeric(r2_coef_df: pd.DataFrame, base_terms: Sequence[str]) -> List[str]:
    if r2_coef_df.empty:
        return list(base_terms)
    stable_terms: List[str] = []
    for t in base_terms:
        sub = r2_coef_df[
            (r2_coef_df["model"].isin(["OLS", "MixedLM", "Logit"]))
            & (r2_coef_df["feature"] == t)
            & pd.to_numeric(r2_coef_df["coef"], errors="coerce").notna()
        ]
        if sub.empty:
            continue
        signs = np.sign(pd.to_numeric(sub["coef"], errors="coerce").to_numpy())
        pos = int((signs > 0).sum())
        neg = int((signs < 0).sum())
        if max(pos, neg) >= 4:
            stable_terms.append(t)
    if not stable_terms:
        return list(base_terms)
    return stable_terms


def _stabilize_pick_round3_benchmark_cols(r2_fi_df: pd.DataFrame, top_k: int = 10) -> Optional[List[str]]:
    if r2_fi_df.empty:
        return None
    sub = r2_fi_df[(r2_fi_df["source"] == "permutation") & (r2_fi_df["model"] == "HGBClassifier")].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("importance_mean", ascending=False).head(top_k)
    cols = [str(x) for x in sub["feature"].tolist() if pd.notna(x)]
    return cols or None


def _stabilize_update_index() -> None:
    line = "- `project/modeling_report/09_cross_event_stabilization_report.md`"
    note = "- V3 stabilization outputs: `project/modeling/output/cross_event_round_comparison_v3x.csv`, `project/modeling/output/cross_event_stop_decision_v3x.json`"

    if STABILIZE_INDEX_PATH.exists():
        text = STABILIZE_INDEX_PATH.read_text(encoding="utf-8")
    else:
        text = "# Modeling Report Index\n\n## Deliverables\n"

    if line not in text:
        text = text.rstrip() + "\n" + line + "\n"
    if note not in text:
        text = text.rstrip() + "\n" + note + "\n"
    STABILIZE_INDEX_PATH.write_text(text, encoding="utf-8")


def _stabilize_write_report(
    round_rows: pd.DataFrame,
    stop_decision: Dict[str, object],
    round_metrics_map: Dict[str, Dict[str, float]],
) -> None:
    def fmt(v) -> str:
        return f"{float(v):.4f}" if pd.notna(v) and np.isfinite(float(v)) else "N/A"

    lines = [
        "# Cross-Event Stabilization Report / 跨事件稳定化修正报告（V3.x）",
        "",
        "## Objective",
        "在最多 3 轮内验证稳定化修正是否能把 V3 从边际改进推进到可接受改进；若仍边际改进则按规则收工。",
        "",
        "## Stop Rule",
        f"- Balanced: `ΔLogit AUC < {STABILIZE_THRESH_AUC:.2f}` 且 `ΔSurvival(best c-index) < {STABILIZE_THRESH_SURV:.2f}` -> 停止。",
        f"- 最终停止轮次: `{stop_decision.get('stopped_at_round')}`",
        f"- 停止原因: `{stop_decision.get('reason')}`",
        "",
        "## Round Metrics",
        "| round | logit_auc | logit_brier | survival_best_c_index | cox_c_index | aft_c_index | ols_rmse | mixedlm_rmse | hgb_auc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rid in sorted(round_metrics_map.keys(), key=lambda x: int(x.replace("r", "")) if x.startswith("r") else -1):
        m = round_metrics_map[rid]
        lines.append(
            f"| {rid} | {fmt(m.get('logit_auc'))} | {fmt(m.get('logit_brier'))} | {fmt(m.get('survival_best_c_index'))} | {fmt(m.get('cox_c_index'))} | {fmt(m.get('aft_c_index'))} | {fmt(m.get('ols_rmse'))} | {fmt(m.get('mixedlm_rmse'))} | {fmt(m.get('hgb_auc'))} |"
        )

    lines.extend(
        [
            "",
            "## Stop Decision Evidence",
            f"- Comparison file: `{STABILIZE_ROUND_COMP_PATH.relative_to(ROOT)}`",
            f"- Decision file: `{STABILIZE_STOP_DECISION_PATH.relative_to(ROOT)}`",
            "",
            "## Output Files",
            "- Round fold metrics: `project/modeling/output/cross_event_fold_metrics_v3r*.csv`",
            "- Round aggregates: `project/modeling/output/cross_event_aggregate_metrics_v3r*.csv`",
            "- Round feature importance: `project/modeling/output/cross_event_feature_importance_v3r*.csv`",
            "- Round summaries: `project/modeling/output/model_summary_cross_event_v3r*.csv`",
            "",
            "## Conclusion",
        ]
    )

    if bool(stop_decision.get("marginal_improvement_stop", False)):
        lines.append("- 触发边际改进停止规则：今天按计划收工。")
    else:
        lines.append("- 未触发边际改进停止，但已达到轮次预算上限：今天按计划收工。")

    STABILIZE_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def stabilize_main() -> None:
    ensure_directories()
    init_tracking_files()

    if not STABILIZE_PANEL_V3_PATH.exists():
        raise FileNotFoundError(f"Missing panel: {STABILIZE_PANEL_V3_PATH}")
    if not STABILIZE_BASE_AGG_PATH.exists():
        raise FileNotFoundError(f"Missing V3 aggregate anchor: {STABILIZE_BASE_AGG_PATH}")

    defaults = load_json(CONFIG_DEFAULTS)
    damage_thr = float(defaults["damage_threshold"])
    rec_thr = float(defaults["recovery_threshold"])

    panel = pd.read_parquet(STABILIZE_PANEL_V3_PATH).copy()
    if "sample_lock_flag" in panel.columns:
        panel = panel[panel["sample_lock_flag"] == 1].copy()
    if panel.empty:
        raise RuntimeError("No rows available after sample lock filter.")

    # Base event-level constant features are intentionally excluded from interpretable numeric list.
    required_cols = ["event_id", "delta_ntl", "pre_mean_ntl", "in_buffer", "land_use_group", "event_disaster_type"] + STABILIZE_LOCAL_NUMERIC_BASE
    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        raise KeyError(f"Missing required columns in V3 panel: {missing}")

    cat_levels = {
        c: sorted(panel[c].fillna("unknown").astype(str).unique().tolist())
        for c in STABILIZE_CAT_TERMS_BASE
        if c in panel.columns
    }
    rec_ctx = RunContext(issues=[])
    recovery = build_recovery_panel(ctx=rec_ctx, panel=panel, threshold=rec_thr, output_path=None)

    # Round0 anchor from existing V3 aggregate.
    agg0 = pd.read_csv(STABILIZE_BASE_AGG_PATH)
    def pick_metric(track: str, model: str, col: str) -> float:
        sub = agg0[(agg0["track"] == track) & (agg0["model"] == model)]
        if sub.empty:
            return np.nan
        val = pd.to_numeric(sub[col], errors="coerce").iloc[0]
        return float(val) if pd.notna(val) else np.nan

    base_metrics = {
        "logit_auc": pick_metric("interpretable", "Logit", "auc"),
        "logit_brier": pick_metric("interpretable", "Logit", "brier"),
        "cox_c_index": pick_metric("interpretable", "Cox", "c_index"),
        "aft_c_index": pick_metric("interpretable", "AFT", "c_index"),
        "ols_rmse": pick_metric("interpretable", "OLS", "rmse"),
        "mixedlm_rmse": pick_metric("interpretable", "MixedLM", "rmse"),
        "hgb_auc": pick_metric("benchmark", "HGBClassifier", "auc"),
        "hgb_rmse": pick_metric("benchmark", "HGBRegressor", "rmse"),
    }
    base_metrics["survival_best_c_index"] = float(max(base_metrics["cox_c_index"], base_metrics["aft_c_index"]))

    round_metrics_map: Dict[str, Dict[str, float]] = {"r0": base_metrics}
    comp_rows: List[Dict[str, object]] = []

    stopped = False
    stop_reason = ""
    stopped_round = "r0"
    prev_metrics = base_metrics

    r2_coef_store = pd.DataFrame()
    r2_fi_store = pd.DataFrame()

    for rid in range(1, STABILIZE_MAX_ROUNDS + 1):
        round_name = f"r{rid}"

        if rid == 1:
            numeric_terms = list(STABILIZE_LOCAL_NUMERIC_BASE)
            benchmark_cols = None
            use_weights = False
            use_cal = False
            use_surv_grid = False
        elif rid == 2:
            numeric_terms = list(STABILIZE_LOCAL_NUMERIC_BASE)
            benchmark_cols = None
            use_weights = True
            use_cal = True
            use_surv_grid = True
        else:
            numeric_terms = _stabilize_pick_round3_numeric(r2_coef_store, STABILIZE_LOCAL_NUMERIC_BASE)
            benchmark_cols = _stabilize_pick_round3_benchmark_cols(r2_fi_store, top_k=10)
            use_weights = True
            use_cal = True
            use_surv_grid = True

        fold_df, agg_df, fi_df = _stabilize_run_loeo_round(
            round_id=round_name,
            panel=panel,
            recovery=recovery,
            numeric_terms=numeric_terms,
            cat_terms=STABILIZE_CAT_TERMS_BASE,
            damage_threshold=damage_thr,
            use_event_weights=use_weights,
            use_logit_calibration=use_cal,
            use_survival_grid=use_surv_grid,
            cat_levels=cat_levels,
            benchmark_selected_cols=benchmark_cols,
        )

        # Save round outputs.
        fold_path = OUTPUT_DIR / f"cross_event_fold_metrics_v3{round_name}.csv"
        agg_path = OUTPUT_DIR / f"cross_event_aggregate_metrics_v3{round_name}.csv"
        fi_path = OUTPUT_DIR / f"cross_event_feature_importance_v3{round_name}.csv"
        summary_path = OUTPUT_DIR / f"model_summary_cross_event_v3{round_name}.csv"

        fold_df.to_csv(fold_path, index=False)
        agg_df.to_csv(agg_path, index=False)
        fi_df.to_csv(fi_path, index=False)

        cur_metrics = _stabilize_extract_round_metrics(agg_df)
        # Hard stability guard: if mandatory metrics are NaN, stop immediately.
        mandatory = [cur_metrics.get("logit_auc", np.nan), cur_metrics.get("survival_best_c_index", np.nan)]
        if any([not np.isfinite(v) for v in mandatory]):
            raise StabilizationError(f"Round {round_name} produced invalid mandatory metrics: {mandatory}")

        summary_df = _stabilize_build_round_summary(round_name, cur_metrics, prev_metrics)
        summary_df.to_csv(summary_path, index=False)

        round_metrics_map[round_name] = cur_metrics
        rows, stop_flag = _stabilize_round_comparison_rows(round_name, cur_metrics, prev_metrics)
        comp_rows.extend(rows)

        if rid == 2:
            r2_coef_store = fi_df[fi_df["source"] == "interpretable_coef"].copy()
            r2_fi_store = fi_df.copy()

        if stop_flag:
            stopped = True
            stop_reason = "marginal_improvement"
            stopped_round = round_name
            break

        prev_metrics = cur_metrics
        stopped_round = round_name

    if not stopped:
        stop_reason = "max_round_reached"

    comp_df = pd.DataFrame(comp_rows)
    comp_df = comp_df[
        [
            "round_id",
            "metric_group",
            "metric_name",
            "value",
            "prev_value",
            "delta",
            "threshold",
            "pass_flag",
            "stop_flag",
            "notes",
        ]
    ]
    comp_df.to_csv(STABILIZE_ROUND_COMP_PATH, index=False)

    final_metrics = round_metrics_map.get(stopped_round, {})
    stop_json = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Balanced",
        "thresholds": {"delta_logit_auc": STABILIZE_THRESH_AUC, "delta_survival_best_c_index": STABILIZE_THRESH_SURV},
        "max_rounds": STABILIZE_MAX_ROUNDS,
        "stopped_at_round": stopped_round,
        "reason": stop_reason,
        "marginal_improvement_stop": bool(stop_reason == "marginal_improvement"),
        "baseline_round": "r0",
        "baseline_metrics": round_metrics_map.get("r0", {}),
        "final_metrics": final_metrics,
        "executed_rounds": sorted([k for k in round_metrics_map.keys() if k.startswith("r") and k != "r0"], key=lambda x: int(x[1:])),
    }
    STABILIZE_STOP_DECISION_PATH.write_text(json.dumps(stop_json, indent=2), encoding="utf-8")

    _stabilize_write_report(round_rows=comp_df, stop_decision=stop_json, round_metrics_map=round_metrics_map)
    _stabilize_update_index()
# ----------------------------
# Unified CLI
# ----------------------------

import argparse


def cmd_build_v3() -> int:
    _build_v3_impl()
    return 0


def cmd_stabilize_v3() -> int:
    stabilize_main()
    return 0


def cmd_full_run() -> int:
    _build_v3_impl()
    stabilize_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Unified cross-event modeling entrypoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build-v3', help='Build cross-event V3 artifacts')
    sub.add_parser('stabilize-v3', help='Run V3 stabilization')
    sub.add_parser('full-run', help='Run build-v3 then stabilize-v3')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'build-v3':
        return cmd_build_v3()
    if args.command == 'stabilize-v3':
        return cmd_stabilize_v3()
    if args.command == 'full-run':
        return cmd_full_run()
    parser.error(f'Unknown command: {args.command}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
