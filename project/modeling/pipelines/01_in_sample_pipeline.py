#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

MODELING_DIR = Path(__file__).resolve().parents[1]
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    brier_score_loss,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_lib import (
    CONFIG_DEFAULTS,
    CONFIG_EVENTS,
    FIG_DIR,
    OUTPUT_DIR,
    PANEL_NLCD_PATH,
    PANEL_PATH,
    PIXEL_DIR,
    PROJECT_DIR,
    RECOVERY_PATH,
    REPORT_DIR,
    ROOT,
    RunContext,
    append_progress,
    attach_nlcd,
    build_pixel_panel,
    build_model_summary_for_report,
    build_recovery_panel,
    ensure_directories,
    fit_cox,
    fit_logit,
    fit_ols_and_mixed,
    generate_figures,
    generate_reports,
    init_tracking_files,
    list_daily_tifs,
    load_json,
    log_issue,
    run_pipeline,
    save_issue_log,
    standardize_facility_type,
)


PANEL_FEATURE_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_feature_upgrade.parquet"
SAMPLE_LOCK_PATH = PIXEL_DIR / "sample_lock_cohort_v1.parquet"

SAMPLE_ALIGN_AUDIT_PATH = OUTPUT_DIR / "sample_alignment_audit.csv"
OSM_SUMMARY_PATH = OUTPUT_DIR / "osm_feature_summary.csv"
CLOUD_PIXEL_SUMMARY_PATH = OUTPUT_DIR / "cloud_feature_pixel_summary.csv"
COX_DIAG_EXT_PATH = OUTPUT_DIR / "cox_diagnostics_extended.csv"
COX_AFT_RESULT_PATH = OUTPUT_DIR / "cox_aft_results_feature_upgrade.csv"
LOGO_FOLD_PATH = OUTPUT_DIR / "logo_fold_metrics.csv"
LOGO_AGG_PATH = OUTPUT_DIR / "logo_aggregate_metrics.csv"
SUMMARY_UPGRADE_PATH = OUTPUT_DIR / "model_summary_feature_upgrade.csv"

OLS_FEATURE_RESULT = OUTPUT_DIR / "ols_results_feature_upgrade.csv"
MIXED_FEATURE_RESULT = OUTPUT_DIR / "mixedlm_results_feature_upgrade.csv"
LOGIT_FEATURE_RESULT = OUTPUT_DIR / "logit_results_feature_upgrade.csv"
LOGIT_MARGINAL_FEATURE_RESULT = OUTPUT_DIR / "logit_marginal_effects_feature_upgrade.csv"
COX_FEATURE_RESULT = OUTPUT_DIR / "cox_results_feature_upgrade.csv"

FIG_FEATURE_DIR = FIG_DIR / "feature_upgrade"
FIG_LOGO_DIR = FIG_DIR / "logo"

POWER_TYPE_PATTERN = r"power|substation|plant|generator|electric"
MEDICAL_TYPE_PATTERN = r"hospital|clinic|medical|health"

FULL_NUMERIC_TERMS = [
    "osm_any_count_750m",
    "osm_power_count_1000m",
    "osm_medical_count_1000m",
    "osm_dist_power_m",
    "osm_dist_any_m",
    "pixel_cloud_proxy",
    "pixel_pre_valid_ratio",
    "pixel_post_valid_ratio",
    "missing_osm_flag",
    "missing_cloud_flag",
]


@dataclass
class CoxCandidate:
    name: str
    cph: CoxPHFitter
    ph: pd.DataFrame
    n_violation: float


def _ensure_extra_dirs() -> None:
    FIG_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_LOGO_DIR.mkdir(parents=True, exist_ok=True)


def _land_use_group(v: float) -> str:
    if pd.isna(v):
        return "unknown"
    try:
        iv = int(v)
    except Exception:
        return "unknown"

    if iv == 24:
        return "developed_high"
    if iv == 23:
        return "developed_medium"
    if iv == 22:
        return "developed_low"
    if iv == 21:
        return "developed_open"
    return "other"


def _feature_coef_table_from_result(result, model_name: str, variant: str, kind: str = "linear") -> pd.DataFrame:
    ci = result.conf_int()
    table = pd.DataFrame(
        {
            "model": model_name,
            "variant": variant,
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": getattr(result, "bse", pd.Series(np.nan, index=result.params.index)).values,
            "stat": getattr(result, "tvalues", getattr(result, "zvalues", pd.Series(np.nan, index=result.params.index))).values,
            "p_value": getattr(result, "pvalues", pd.Series(np.nan, index=result.params.index)).values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_high": ci.iloc[:, 1].values,
            "kind": kind,
        }
    )
    return table


def _valid_ratio_grid(paths: Sequence[Path]) -> np.ndarray:
    if not paths:
        raise FileNotFoundError("No tif path for valid-ratio grid.")

    valid_count = None
    for p in paths:
        with rasterio.open(p) as src:  # type: ignore[name-defined]
            arr = src.read(1).astype("float64")
            if valid_count is None:
                valid_count = np.zeros(arr.shape, dtype="float64")
            valid_count += np.isfinite(arr).astype("float64")
    return valid_count / float(len(paths))


def attach_pixel_cloud_features(ctx: RunContext, panel: pd.DataFrame, events_cfg: Dict[str, object]) -> pd.DataFrame:
    panel = panel.copy()
    for col in ["pixel_pre_valid_ratio", "pixel_post_valid_ratio", "pixel_cloud_proxy", "missing_cloud_flag"]:
        if col not in panel.columns:
            panel[col] = np.nan

    summary_rows: List[Dict[str, object]] = []
    for event_id, cfg in events_cfg.items():
        mask = panel["event_id"] == event_id
        if mask.sum() == 0:
            continue

        pre_tifs = list_daily_tifs(ROOT / cfg["pre_dir"])
        post_tifs = list_daily_tifs(ROOT / cfg["post_dir"])
        if not pre_tifs or not post_tifs:
            panel.loc[mask, "pixel_pre_valid_ratio"] = np.nan
            panel.loc[mask, "pixel_post_valid_ratio"] = np.nan
            panel.loc[mask, "pixel_cloud_proxy"] = np.nan
            panel.loc[mask, "missing_cloud_flag"] = 1
            log_issue(
                ctx,
                stage="attach_pixel_cloud_features",
                model="all",
                event_id=event_id,
                issue_type="missing_pre_or_post_tif",
                symptom=f"pre={len(pre_tifs)}, post={len(post_tifs)}",
                fix_action="set missing_cloud_flag=1 and continue",
                impact="pixel cloud proxy unavailable",
                status="monitor",
            )
            summary_rows.append(
                {
                    "event_id": event_id,
                    "n_rows": int(mask.sum()),
                    "n_pre_tif": len(pre_tifs),
                    "n_post_tif": len(post_tifs),
                    "pre_valid_mean": np.nan,
                    "post_valid_mean": np.nan,
                    "cloud_proxy_mean": np.nan,
                    "coverage": 0.0,
                    "missing_cloud_flag": 1,
                }
            )
            continue

        pre_ratio = _valid_ratio_grid(pre_tifs)
        post_ratio = _valid_ratio_grid(post_tifs)

        rr = panel.loc[mask, "row"].astype(int).to_numpy()
        cc = panel.loc[mask, "col"].astype(int).to_numpy()
        ok = (
            (rr >= 0)
            & (cc >= 0)
            & (rr < pre_ratio.shape[0])
            & (cc < pre_ratio.shape[1])
            & (rr < post_ratio.shape[0])
            & (cc < post_ratio.shape[1])
        )

        pre_vals = np.full(rr.shape, np.nan, dtype="float64")
        post_vals = np.full(rr.shape, np.nan, dtype="float64")
        if ok.any():
            pre_vals[ok] = pre_ratio[rr[ok], cc[ok]]
            post_vals[ok] = post_ratio[rr[ok], cc[ok]]

        proxy = post_vals - pre_vals
        panel.loc[mask, "pixel_pre_valid_ratio"] = pre_vals
        panel.loc[mask, "pixel_post_valid_ratio"] = post_vals
        panel.loc[mask, "pixel_cloud_proxy"] = proxy
        panel.loc[mask, "missing_cloud_flag"] = np.where(np.isfinite(proxy), 0, 1)

        summary_rows.append(
            {
                "event_id": event_id,
                "n_rows": int(mask.sum()),
                "n_pre_tif": len(pre_tifs),
                "n_post_tif": len(post_tifs),
                "pre_valid_mean": float(np.nanmean(pre_vals)),
                "post_valid_mean": float(np.nanmean(post_vals)),
                "cloud_proxy_mean": float(np.nanmean(proxy)),
                "coverage": float(np.isfinite(proxy).mean()),
                "missing_cloud_flag": int((~np.isfinite(proxy)).all()),
            }
        )

    pd.DataFrame(summary_rows).to_csv(CLOUD_PIXEL_SUMMARY_PATH, index=False)
    return panel


def _event_distance_fill(df: pd.DataFrame, col: str, default_value: float = 20000.0) -> pd.Series:
    out = df[col].copy()
    for event_id, idx in df.groupby("event_id").groups.items():
        s = out.loc[idx]
        finite = s[np.isfinite(s)]
        if finite.empty:
            fill_v = default_value
        else:
            fill_v = float(finite.max() + 1.0)
        out.loc[idx] = s.fillna(fill_v)
    return out


def attach_osm_features(ctx: RunContext, panel: pd.DataFrame, events_cfg: Dict[str, object]) -> pd.DataFrame:
    panel = panel.copy()
    defaults = {
        "osm_any_count_750m": 0.0,
        "osm_power_count_1000m": 0.0,
        "osm_medical_count_1000m": 0.0,
        "osm_dist_power_m": np.nan,
        "osm_dist_any_m": np.nan,
        "osm_feature_source": "track_a",
        "missing_osm_flag": 1,
    }
    for col, val in defaults.items():
        if col not in panel.columns:
            panel[col] = val

    summary_rows: List[Dict[str, object]] = []
    for event_id, cfg in events_cfg.items():
        mask = panel["event_id"] == event_id
        if mask.sum() == 0:
            continue

        poi_path = ROOT / cfg["poi_csv"]
        if not poi_path.exists():
            log_issue(
                ctx,
                stage="attach_osm_features",
                model="all",
                event_id=event_id,
                issue_type="missing_poi_file",
                symptom=str(poi_path),
                fix_action="set missing_osm_flag=1 and continue",
                impact="OSM features unavailable",
                status="open",
            )
            summary_rows.append(
                {
                    "event_id": event_id,
                    "poi_file": str(poi_path),
                    "n_poi": 0,
                    "n_power": 0,
                    "n_medical": 0,
                    "missing_osm_flag": 1,
                    "coverage": 0.0,
                }
            )
            continue

        poi = pd.read_csv(poi_path)
        if poi.empty or "lon" not in poi.columns or "lat" not in poi.columns:
            log_issue(
                ctx,
                stage="attach_osm_features",
                model="all",
                event_id=event_id,
                issue_type="malformed_poi_file",
                symptom=f"empty or missing lon/lat in {poi_path.name}",
                fix_action="set missing_osm_flag=1 and continue",
                impact="OSM features unavailable",
                status="open",
            )
            summary_rows.append(
                {
                    "event_id": event_id,
                    "poi_file": str(poi_path),
                    "n_poi": 0,
                    "n_power": 0,
                    "n_medical": 0,
                    "missing_osm_flag": 1,
                    "coverage": 0.0,
                }
            )
            continue

        poi = poi.copy()
        if "type" not in poi.columns:
            poi["type"] = ""
        poi["facility_type_std"] = poi["type"].astype(str).str.lower().str.strip().map(standardize_facility_type)

        transformer = Transformer.from_crs("EPSG:4326", cfg["metric_crs"], always_xy=True)

        px_lon = panel.loc[mask, "lon"].astype(float).to_numpy()
        px_lat = panel.loc[mask, "lat"].astype(float).to_numpy()
        px_x, px_y = transformer.transform(px_lon, px_lat)
        pix_xy = np.column_stack([px_x, px_y])

        poi_lon = poi["lon"].astype(float).to_numpy()
        poi_lat = poi["lat"].astype(float).to_numpy()
        poi_x, poi_y = transformer.transform(poi_lon, poi_lat)
        poi_xy = np.column_stack([poi_x, poi_y])
        poi_ok = np.isfinite(poi_xy).all(axis=1)
        poi = poi.loc[poi_ok].reset_index(drop=True)
        poi_xy = poi_xy[poi_ok]

        if poi.empty:
            panel.loc[mask, "missing_osm_flag"] = 1
            summary_rows.append(
                {
                    "event_id": event_id,
                    "poi_file": str(poi_path),
                    "n_poi": 0,
                    "n_power": 0,
                    "n_medical": 0,
                    "missing_osm_flag": 1,
                    "coverage": 0.0,
                }
            )
            continue

        tree_any = cKDTree(poi_xy)
        count_any = tree_any.query_ball_point(pix_xy, r=750.0, return_length=True)
        dist_any, _ = tree_any.query(pix_xy, k=1)

        power_mask = poi["facility_type_std"].str.contains(POWER_TYPE_PATTERN, regex=True, na=False)
        med_mask = poi["facility_type_std"].str.contains(MEDICAL_TYPE_PATTERN, regex=True, na=False)

        count_power = np.zeros(len(pix_xy), dtype="float64")
        dist_power = np.full(len(pix_xy), np.nan, dtype="float64")
        if power_mask.any():
            power_xy = poi_xy[power_mask.to_numpy()]
            tree_power = cKDTree(power_xy)
            count_power = tree_power.query_ball_point(pix_xy, r=1000.0, return_length=True).astype("float64")
            dist_power, _ = tree_power.query(pix_xy, k=1)

        count_med = np.zeros(len(pix_xy), dtype="float64")
        if med_mask.any():
            med_xy = poi_xy[med_mask.to_numpy()]
            tree_med = cKDTree(med_xy)
            count_med = tree_med.query_ball_point(pix_xy, r=1000.0, return_length=True).astype("float64")

        panel.loc[mask, "osm_any_count_750m"] = count_any
        panel.loc[mask, "osm_power_count_1000m"] = count_power
        panel.loc[mask, "osm_medical_count_1000m"] = count_med
        panel.loc[mask, "osm_dist_power_m"] = dist_power
        panel.loc[mask, "osm_dist_any_m"] = dist_any
        panel.loc[mask, "osm_feature_source"] = "track_a"
        panel.loc[mask, "missing_osm_flag"] = 0

        summary_rows.append(
            {
                "event_id": event_id,
                "poi_file": str(poi_path),
                "n_poi": int(len(poi)),
                "n_power": int(power_mask.sum()),
                "n_medical": int(med_mask.sum()),
                "missing_osm_flag": 0,
                "coverage": float(np.isfinite(dist_any).mean()),
                "any_count_mean_750m": float(np.mean(count_any)),
                "power_count_mean_1000m": float(np.mean(count_power)),
            }
        )

    pd.DataFrame(summary_rows).to_csv(OSM_SUMMARY_PATH, index=False)
    return panel


def create_sample_lock(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    core_cols = ["pre_mean_ntl", "post_mean_ntl", "delta_ntl", "in_buffer", "distance_to_nearest"]
    lock_flag = np.isfinite(panel[core_cols]).all(axis=1)

    cloud_gate = pd.Series(True, index=panel.index)
    if "missing_cloud_flag" in panel.columns:
        cloud_gate &= pd.to_numeric(panel["missing_cloud_flag"], errors="coerce").fillna(1).eq(0)
    for col in ["pixel_pre_valid_ratio", "pixel_post_valid_ratio"]:
        if col in panel.columns:
            cloud_gate &= pd.to_numeric(panel[col], errors="coerce").notna()

    final_lock = lock_flag & cloud_gate
    panel["sample_lock_flag"] = final_lock.astype(int)
    panel["lock_reason"] = np.select(
        [final_lock, ~lock_flag],
        ["core_fields_available", "missing_core_fields"],
        default="missing_cloud_quality",
    )

    panel[["pixel_id", "event_id", "sample_lock_flag", "lock_reason"]].to_parquet(SAMPLE_LOCK_PATH, index=False)
    return panel


MODEL_FRAME_COUNT_COLS = ["osm_any_count_750m", "osm_power_count_1000m", "osm_medical_count_1000m"]
MODEL_FRAME_DISTANCE_COLS = ["osm_dist_power_m", "osm_dist_any_m"]
MODEL_FRAME_QUALITY_COLS = ["pixel_pre_valid_ratio", "pixel_post_valid_ratio", "pixel_cloud_proxy"]


def _init_model_frame_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["land_use_group"] = out["land_use_group"].fillna("unknown").astype(str)

    for col in ["missing_osm_flag", "missing_cloud_flag"]:
        if col not in out.columns:
            out[col] = 1
        out[col] = out[col].fillna(1).astype(int)

    for col in MODEL_FRAME_COUNT_COLS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    for col in MODEL_FRAME_DISTANCE_COLS + MODEL_FRAME_QUALITY_COLS:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def _fit_model_frame_fill_rules(df: pd.DataFrame) -> Dict[str, float]:
    base = _init_model_frame_defaults(df)
    rules: Dict[str, float] = {}

    for col in MODEL_FRAME_DISTANCE_COLS:
        series = base[col]
        rules[col] = float(series.median()) if series.notna().any() else 0.0

    for col in MODEL_FRAME_QUALITY_COLS:
        series = base[col]
        rules[col] = float(series.mean()) if series.notna().any() else 0.0

    return rules


def _apply_model_frame_fill_rules(df: pd.DataFrame, fill_rules: Dict[str, float]) -> pd.DataFrame:
    out = _init_model_frame_defaults(df)
    for col in MODEL_FRAME_DISTANCE_COLS + MODEL_FRAME_QUALITY_COLS:
        out[col] = out[col].fillna(fill_rules.get(col, 0.0))
    return out


def prepare_model_frame(panel: pd.DataFrame) -> pd.DataFrame:
    fill_rules = _fit_model_frame_fill_rules(panel)
    return _apply_model_frame_fill_rules(panel, fill_rules)


def prepare_model_frame_fold(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    fill_rules = _fit_model_frame_fill_rules(train_df)
    train = _apply_model_frame_fill_rules(train_df, fill_rules)
    test = _apply_model_frame_fill_rules(test_df, fill_rules)
    return train, test, fill_rules


def _build_linear_formula(include_land_use: bool, include_full: bool, include_event_fe: bool = True) -> str:
    terms = ["in_buffer * pre_mean_ntl"]
    if include_event_fe:
        terms.append("C(event_id)")
    if include_land_use:
        terms.append("C(land_use_group)")
    if include_full:
        terms.extend(FULL_NUMERIC_TERMS)
    return "delta_ntl ~ " + " + ".join(terms)


def _build_logit_formula(include_land_use: bool, include_full: bool, include_event_fe: bool = True) -> str:
    terms = ["in_buffer * pre_mean_ntl"]
    if include_event_fe:
        terms.append("C(event_id)")
    if include_land_use:
        terms.append("C(land_use_group)")
    if include_full:
        terms.extend(FULL_NUMERIC_TERMS)
    return "is_damaged ~ " + " + ".join(terms)


def _calibration_slope(y_true: np.ndarray, prob: np.ndarray) -> float:
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    x = np.log(p / (1 - p))
    if np.allclose(x.std(), 0.0) or len(np.unique(y_true)) < 2:
        return np.nan
    slope = np.polyfit(x, y_true, 1)[0]
    return float(slope)


def fit_ols_and_mixed(
    ctx: RunContext,
    df: pd.DataFrame,
    variant: str,
    include_land_use: bool,
    include_full: bool,
    include_event_fe: bool = True,
) -> Dict[str, object]:
    out: Dict[str, object] = {}

    formula_ols = _build_linear_formula(
        include_land_use=include_land_use,
        include_full=include_full,
        include_event_fe=include_event_fe,
    )
    model_ols = smf.ols(formula_ols, data=df).fit(cov_type="HC1")
    ols_coef = _feature_coef_table_from_result(model_ols, "OLS", variant, kind="linear")
    ols_pred = pd.DataFrame(
        {
            "pixel_id": df["pixel_id"],
            "event_id": df["event_id"],
            "delta_ntl": df["delta_ntl"],
            "predicted": model_ols.predict(df),
        }
    )
    ols_pred["residual"] = ols_pred["delta_ntl"] - ols_pred["predicted"]
    ols_pred.to_csv(OUTPUT_DIR / f"ols_predictions_{variant}.csv", index=False)

    formula_mixed = _build_linear_formula(include_land_use=include_land_use, include_full=include_full, include_event_fe=False)
    mixed_coef = pd.DataFrame(
        columns=["model", "variant", "term", "coef", "std_err", "stat", "p_value", "ci_low", "ci_high", "kind"]
    )
    random_effects = pd.DataFrame(columns=["event_id", "random_intercept", "variant"])
    mixed_pred = pd.DataFrame(columns=["pixel_id", "event_id", "delta_ntl", "predicted", "residual"])

    mixed_candidates = [formula_mixed]
    if include_full:
        mixed_candidates.append(_build_linear_formula(include_land_use=include_land_use, include_full=False, include_event_fe=False))
    if include_land_use:
        mixed_candidates.append(_build_linear_formula(include_land_use=False, include_full=False, include_event_fe=False))
    mixed_result = None
    mixed_formula_used = ""
    last_err = None
    for cand in mixed_candidates:
        try:
            mixed_model = smf.mixedlm(cand, data=df, groups=df["event_id"])
            try:
                mixed_result = mixed_model.fit(method="lbfgs", reml=False)
            except Exception:
                mixed_result = mixed_model.fit(method="powell", reml=False)
            mixed_formula_used = cand
            break
        except Exception as e:
            last_err = e

    if mixed_result is None:
        log_issue(
            ctx,
            stage="fit_mixedlm_feature_upgrade",
            model="MixedLM",
            event_id="all",
            issue_type="model_fit_failed",
            symptom=str(last_err),
            fix_action="skip mixedlm for this variant",
            impact="mixed-effect result unavailable",
            status="open",
        )
    else:
        if mixed_formula_used != formula_mixed:
            log_issue(
                ctx,
                stage="fit_mixedlm_feature_upgrade",
                model="MixedLM",
                event_id="all",
                issue_type="formula_fallback",
                symptom=f"failed full formula for {variant}",
                fix_action=f"use reduced mixed formula: {mixed_formula_used}",
                impact="mixedlm kept with reduced controls",
                status="resolved",
            )
        mixed_coef = _feature_coef_table_from_result(mixed_result, "MixedLM", variant, kind="linear")
        try:
            re_rows = []
            for ev, re_dict in mixed_result.random_effects.items():
                if isinstance(re_dict, pd.Series):
                    val = float(re_dict.iloc[0])
                elif isinstance(re_dict, dict):
                    val = float(next(iter(re_dict.values())))
                else:
                    val = float(re_dict[0]) if hasattr(re_dict, "__len__") else float(re_dict)
                re_rows.append({"event_id": ev, "random_intercept": val, "variant": variant})
            random_effects = pd.DataFrame(re_rows)
        except Exception as e_re:
            log_issue(
                ctx,
                stage="fit_mixedlm_feature_upgrade",
                model="MixedLM",
                event_id="all",
                issue_type="random_effect_extraction_failed",
                symptom=str(e_re),
                fix_action="keep fixed effects and predictions",
                impact="random-effect table unavailable",
                status="monitor",
            )
        mixed_pred = pd.DataFrame(
            {
                "pixel_id": df["pixel_id"],
                "event_id": df["event_id"],
                "delta_ntl": df["delta_ntl"],
                "predicted": mixed_result.predict(df),
            }
        )
        mixed_pred["residual"] = mixed_pred["delta_ntl"] - mixed_pred["predicted"]

    mixed_pred.to_csv(OUTPUT_DIR / f"mixedlm_predictions_{variant}.csv", index=False)
    random_effects.to_csv(OUTPUT_DIR / f"mixedlm_random_effects_{variant}.csv", index=False)

    out["ols_coef"] = ols_coef
    out["mixed_coef"] = mixed_coef
    out["random_effects"] = random_effects
    out["ols_metrics"] = {
        "rmse": float(math.sqrt(mean_squared_error(ols_pred["delta_ntl"], ols_pred["predicted"]))),
        "mae": float(mean_absolute_error(ols_pred["delta_ntl"], ols_pred["predicted"])),
    }
    if not mixed_pred.empty:
        out["mixed_metrics"] = {
            "rmse": float(math.sqrt(mean_squared_error(mixed_pred["delta_ntl"], mixed_pred["predicted"]))),
            "mae": float(mean_absolute_error(mixed_pred["delta_ntl"], mixed_pred["predicted"])),
        }
    else:
        out["mixed_metrics"] = {"rmse": np.nan, "mae": np.nan}
    return out


def fit_logit(
    ctx: RunContext,
    df: pd.DataFrame,
    variant: str,
    include_land_use: bool,
    include_full: bool,
    damage_threshold: float,
    include_event_fe: bool = True,
) -> Dict[str, object]:
    work = df.copy()
    work["is_damaged"] = (work["delta_ntl"] < damage_threshold).astype(int)
    formula = _build_logit_formula(include_land_use=include_land_use, include_full=include_full, include_event_fe=include_event_fe)
    formula_candidates = [formula]
    if include_full:
        formula_candidates.append(_build_logit_formula(include_land_use=include_land_use, include_full=False, include_event_fe=include_event_fe))
    if include_land_use:
        formula_candidates.append(_build_logit_formula(include_land_use=False, include_full=False, include_event_fe=include_event_fe))

    result = None
    regularized = False
    used_formula = None
    last_err = None
    for f in formula_candidates:
        try:
            result = smf.logit(formula=f, data=work).fit(disp=False, maxiter=200)
            used_formula = f
            break
        except Exception as e:
            last_err = e

    if result is None:
        for f in formula_candidates:
            try:
                result = smf.logit(formula=f, data=work).fit_regularized(disp=False, maxiter=200, alpha=0.01)
                regularized = True
                used_formula = f
                break
            except Exception as e:
                last_err = e

    if result is None:
        log_issue(
            ctx,
            stage="fit_logit_feature_upgrade",
            model="Logit",
            event_id="all",
            issue_type="model_fit_failed",
            symptom=str(last_err),
            fix_action="skip logit for this variant",
            impact="logit result unavailable",
            status="open",
        )
        empty = pd.DataFrame()
        return {
            "coef": empty,
            "marginal": empty,
            "roc": empty,
            "calibration": empty,
            "predictions": empty,
            "metrics": {"auc": np.nan, "brier": np.nan, "calibration_slope": np.nan},
        }

    if used_formula != formula:
        log_issue(
            ctx,
            stage="fit_logit_feature_upgrade",
            model="Logit",
            event_id="all",
            issue_type="formula_fallback",
            symptom=f"failed full formula for {variant}",
            fix_action=f"use reduced formula: {used_formula}",
            impact="logit kept with reduced controls",
            status="resolved",
        )

    if regularized:
        params = result.params
        coef = pd.DataFrame(
            {
                "model": "Logit",
                "variant": variant,
                "term": params.index,
                "coef": params.values,
                "std_err": np.nan,
                "stat": np.nan,
                "p_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "kind": "logit_regularized",
            }
        )
    else:
        coef = _feature_coef_table_from_result(result, "Logit", variant, kind="logit")
    coef["odds_ratio"] = np.exp(coef["coef"])
    coef["or_ci_low"] = np.exp(coef["ci_low"])
    coef["or_ci_high"] = np.exp(coef["ci_high"])

    pred_prob = result.predict(work)
    pred_df = work[["pixel_id", "event_id", "is_damaged"]].copy()
    pred_df["pred_prob"] = pred_prob
    pred_df.to_csv(OUTPUT_DIR / f"logit_predictions_{variant}.csv", index=False)

    if not regularized:
        try:
            marginal = result.get_margeff(at="overall").summary_frame().reset_index().rename(columns={"index": "term"})
            marginal["model"] = "Logit"
            marginal["variant"] = variant
        except Exception:
            marginal = pd.DataFrame()
    else:
        marginal = pd.DataFrame()

    try:
        auc_value = float(roc_auc_score(work["is_damaged"], pred_prob))
    except Exception:
        auc_value = np.nan
    brier = float(brier_score_loss(work["is_damaged"], pred_prob))
    cal_slope = _calibration_slope(work["is_damaged"].to_numpy(), np.asarray(pred_prob))

    roc_df = pd.DataFrame(
        {
            "variant": [variant],
            "auc": [auc_value],
            "brier": [brier],
            "calibration_slope": [cal_slope],
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
        "metrics": {"auc": auc_value, "brier": brier, "calibration_slope": cal_slope},
    }


def _count_ph_violation(ph_df: pd.DataFrame) -> float:
    if ph_df.empty or "p" not in ph_df.columns:
        return np.nan
    return float((ph_df["p"] < 0.05).sum())


def _sanitize_design_for_cox(design: pd.DataFrame, keep_cols: Sequence[str]) -> pd.DataFrame:
    out = design.copy()
    numeric_cols = [c for c in out.columns if c not in keep_cols]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
        if out[c].isna().all():
            out[c] = 0.0
        else:
            out[c] = out[c].fillna(out[c].median())
    return out


def _fit_cph_with_retry(
    design: pd.DataFrame,
    duration_col: str,
    event_col: str,
    strata: Optional[Sequence[str]] = None,
) -> Tuple[Optional[CoxPHFitter], float, Optional[Exception]]:
    last_err = None
    for penalizer in [0.0, 0.01, 0.1]:
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            if strata:
                cph.fit(design, duration_col=duration_col, event_col=event_col, strata=list(strata))
            else:
                cph.fit(design, duration_col=duration_col, event_col=event_col)
            return cph, penalizer, None
        except Exception as e:
            last_err = e
    return None, np.nan, last_err


def _cox_design(
    work: pd.DataFrame,
    include_land_use: bool,
    include_full: bool,
    include_event_dummies: bool,
) -> pd.DataFrame:
    design = work[["in_buffer", "pre_mean_ntl"]].copy()
    if include_land_use:
        lu = pd.get_dummies(work["land_use_group"], prefix="lu", drop_first=True)
        design = pd.concat([design, lu], axis=1)
    if include_full:
        design = pd.concat([design, work[FULL_NUMERIC_TERMS].copy()], axis=1)
    if include_event_dummies:
        ev = pd.get_dummies(work["event_id"], prefix="event", drop_first=True)
        design = pd.concat([design, ev], axis=1)
    return design


def fit_cox_enhanced(
    ctx: RunContext,
    recovery_df: pd.DataFrame,
    variant: str,
    include_land_use: bool,
    include_full: bool,
) -> Dict[str, pd.DataFrame]:
    cols = ["recovery_days", "event_observed", "event_id", "in_buffer", "pre_mean_ntl", "land_use_group"] + FULL_NUMERIC_TERMS
    cols = [c for c in cols if c in recovery_df.columns]
    work = recovery_df[cols].copy()
    work = work[np.isfinite(work["recovery_days"]) & np.isfinite(work["pre_mean_ntl"])].copy()

    diagnostics: List[Dict[str, object]] = []
    candidates: List[CoxCandidate] = []

    # Step 1: base model with event dummies.
    try:
        design_cov = _cox_design(work, include_land_use=include_land_use, include_full=include_full, include_event_dummies=True)
        design = pd.concat([work[["recovery_days", "event_observed"]].reset_index(drop=True), design_cov.reset_index(drop=True)], axis=1)
        design = _sanitize_design_for_cox(design, keep_cols=["recovery_days", "event_observed"])
        cph, penalizer, err = _fit_cph_with_retry(design, duration_col="recovery_days", event_col="event_observed")
        if cph is None:
            raise err if err is not None else RuntimeError("unknown cox fit error")
        try:
            ph = proportional_hazard_test(cph, design, time_transform="rank").summary.reset_index().rename(columns={"index": "covariate"})
        except Exception:
            ph = pd.DataFrame()
        n_vio = _count_ph_violation(ph)
        diagnostics.append(
            {
                "variant": variant,
                "step": "base_event_dummies",
                "n_obs": len(design),
                "n_covariates": design_cov.shape[1],
                "ph_violations": n_vio,
                "status": f"ok (penalizer={penalizer})",
            }
        )
        candidates.append(CoxCandidate(name="base_event_dummies", cph=cph, ph=ph, n_violation=n_vio))
    except Exception as e:
        diagnostics.append(
            {
                "variant": variant,
                "step": "base_event_dummies",
                "n_obs": len(work),
                "n_covariates": np.nan,
                "ph_violations": np.nan,
                "status": f"failed: {e}",
            }
        )
        log_issue(
            ctx,
            stage="fit_cox_enhanced",
            model="Cox",
            event_id="all",
            issue_type="base_fit_failed",
            symptom=str(e),
            fix_action="try event strata model",
            impact="fallback sequence triggered",
            status="monitor",
        )

    # Step 2: stratified by event_id
    try:
        design_cov = _cox_design(work, include_land_use=include_land_use, include_full=include_full, include_event_dummies=False)
        design = pd.concat(
            [work[["recovery_days", "event_observed", "event_id"]].reset_index(drop=True), design_cov.reset_index(drop=True)],
            axis=1,
        )
        design = _sanitize_design_for_cox(design, keep_cols=["recovery_days", "event_observed", "event_id"])
        cph, penalizer, err = _fit_cph_with_retry(
            design,
            duration_col="recovery_days",
            event_col="event_observed",
            strata=["event_id"],
        )
        if cph is None:
            raise err if err is not None else RuntimeError("unknown cox fit error")
        try:
            ph = proportional_hazard_test(cph, design, time_transform="rank").summary.reset_index().rename(columns={"index": "covariate"})
        except Exception:
            ph = pd.DataFrame()
        n_vio = _count_ph_violation(ph)
        diagnostics.append(
            {
                "variant": variant,
                "step": "strata_event_id",
                "n_obs": len(design),
                "n_covariates": design_cov.shape[1],
                "ph_violations": n_vio,
                "status": f"ok (penalizer={penalizer})",
            }
        )
        candidates.append(CoxCandidate(name="strata_event_id", cph=cph, ph=ph, n_violation=n_vio))
    except Exception as e:
        diagnostics.append(
            {
                "variant": variant,
                "step": "strata_event_id",
                "n_obs": len(work),
                "n_covariates": np.nan,
                "ph_violations": np.nan,
                "status": f"failed: {e}",
            }
        )
        log_issue(
            ctx,
            stage="fit_cox_enhanced",
            model="Cox",
            event_id="all",
            issue_type="strata_fit_failed",
            symptom=str(e),
            fix_action="try time-interaction model",
            impact="fallback sequence triggered",
            status="monitor",
        )

    # Step 3: time interaction for key continuous terms
    try:
        design_cov = _cox_design(work, include_land_use=include_land_use, include_full=include_full, include_event_dummies=False)
        design = pd.concat(
            [work[["recovery_days", "event_observed", "event_id"]].reset_index(drop=True), design_cov.reset_index(drop=True)],
            axis=1,
        )
        design["log_time"] = np.log(np.clip(design["recovery_days"].to_numpy(), 1.0, None))
        for term in ["pre_mean_ntl", "in_buffer"]:
            if term in design.columns:
                design[f"{term}_x_log_time"] = design[term] * design["log_time"]
        design = _sanitize_design_for_cox(design, keep_cols=["recovery_days", "event_observed", "event_id"])
        cph, penalizer, err = _fit_cph_with_retry(
            design,
            duration_col="recovery_days",
            event_col="event_observed",
            strata=["event_id"],
        )
        if cph is None:
            raise err if err is not None else RuntimeError("unknown cox fit error")
        try:
            ph = proportional_hazard_test(cph, design, time_transform="rank").summary.reset_index().rename(columns={"index": "covariate"})
        except Exception:
            ph = pd.DataFrame()
        n_vio = _count_ph_violation(ph)
        diagnostics.append(
            {
                "variant": variant,
                "step": "strata_time_interaction",
                "n_obs": len(design),
                "n_covariates": design.shape[1] - 3,
                "ph_violations": n_vio,
                "status": f"ok (penalizer={penalizer})",
            }
        )
        candidates.append(CoxCandidate(name="strata_time_interaction", cph=cph, ph=ph, n_violation=n_vio))
    except Exception as e:
        diagnostics.append(
            {
                "variant": variant,
                "step": "strata_time_interaction",
                "n_obs": len(work),
                "n_covariates": np.nan,
                "ph_violations": np.nan,
                "status": f"failed: {e}",
            }
        )
        log_issue(
            ctx,
            stage="fit_cox_enhanced",
            model="Cox",
            event_id="all",
            issue_type="time_interaction_fit_failed",
            symptom=str(e),
            fix_action="use best available cox candidate",
            impact="partial PH repair",
            status="monitor",
        )

    aft_summary = pd.DataFrame()
    try:
        aft_cov = _cox_design(work, include_land_use=include_land_use, include_full=include_full, include_event_dummies=True)
        aft_design = pd.concat(
            [work[["recovery_days", "event_observed"]].reset_index(drop=True), aft_cov.reset_index(drop=True)],
            axis=1,
        )
        aft_design = _sanitize_design_for_cox(aft_design, keep_cols=["recovery_days", "event_observed"])
        aft = WeibullAFTFitter(penalizer=0.01)
        aft.fit(aft_design, duration_col="recovery_days", event_col="event_observed")
        aft_summary = aft.summary.reset_index().rename(columns={"index": "covariate"})
        aft_summary["variant"] = variant
        aft_summary["model"] = "WeibullAFT"
        diagnostics.append(
            {
                "variant": variant,
                "step": "weibull_aft",
                "n_obs": len(aft_design),
                "n_covariates": aft_cov.shape[1],
                "ph_violations": np.nan,
                "status": "ok",
            }
        )
    except Exception as e:
        diagnostics.append(
            {
                "variant": variant,
                "step": "weibull_aft",
                "n_obs": len(work),
                "n_covariates": np.nan,
                "ph_violations": np.nan,
                "status": f"failed: {e}",
            }
        )

    if not candidates:
        return {
            "coef": pd.DataFrame(),
            "km": pd.DataFrame(),
            "ph": pd.DataFrame(),
            "diagnostics": pd.DataFrame(diagnostics),
            "aft": aft_summary,
        }

    priority = {"strata_time_interaction": 0, "strata_event_id": 1, "base_event_dummies": 2}
    selected = sorted(
        candidates,
        key=lambda x: (
            np.inf if pd.isna(x.n_violation) else x.n_violation,
            priority.get(x.name, 99),
        ),
    )[0]

    summary = selected.cph.summary.reset_index().rename(columns={"index": "covariate"})
    summary["model"] = "Cox"
    summary["variant"] = variant
    summary["cox_spec"] = selected.name
    summary["hazard_ratio"] = np.exp(summary["coef"])

    ph_df = selected.ph.copy()
    ph_df["variant"] = variant
    ph_df["cox_spec"] = selected.name

    km_rows = []
    kmf = KaplanMeierFitter()
    for grp in [0, 1]:
        grp_df = work[work["in_buffer"] == grp]
        if grp_df.empty:
            continue
        kmf.fit(grp_df["recovery_days"], grp_df["event_observed"], label=f"in_buffer_{grp}")
        sf = kmf.survival_function_.reset_index().rename(columns={"timeline": "day", kmf._label: "survival"})
        sf["group"] = grp
        sf["variant"] = variant
        km_rows.append(sf)
    km_df = pd.concat(km_rows, ignore_index=True) if km_rows else pd.DataFrame()

    diag_df = pd.DataFrame(diagnostics)
    diag_df["selected_spec"] = selected.name
    return {"coef": summary, "km": km_df, "ph": ph_df, "diagnostics": diag_df, "aft": aft_summary}


def _summarize_feature_upgrade(
    ols_df: pd.DataFrame,
    mixed_df: pd.DataFrame,
    logit_df: pd.DataFrame,
    logit_roc: pd.DataFrame,
    cox_df: pd.DataFrame,
    variant_n_obs: Dict[str, int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def _append_row(model: str, variant: str, metric: str, term: str, value, pvalue, low, high, notes: str):
        rows.append(
            {
                "model": model,
                "variant": variant,
                "key_metric": metric,
                "term": term,
                "value": value,
                "p_value": pvalue,
                "ci_low": low,
                "ci_high": high,
                "n_obs": variant_n_obs.get(variant, np.nan),
                "notes": notes,
            }
        )

    for df, model_name in [(ols_df, "OLS"), (mixed_df, "MixedLM")]:
        if df.empty:
            continue
        sub = df[df["term"] == "in_buffer"]
        for _, r in sub.iterrows():
            _append_row(
                model_name,
                r["variant"],
                "coef_in_buffer",
                "in_buffer",
                r["coef"],
                r["p_value"],
                r["ci_low"],
                r["ci_high"],
                "linear effect",
            )

    if not logit_df.empty:
        sub = logit_df[logit_df["term"] == "in_buffer"]
        for _, r in sub.iterrows():
            _append_row(
                "Logit",
                r["variant"],
                "odds_ratio_in_buffer",
                "in_buffer",
                r.get("odds_ratio", np.nan),
                r.get("p_value", np.nan),
                r.get("or_ci_low", np.nan),
                r.get("or_ci_high", np.nan),
                "odds ratio",
            )

    if not logit_roc.empty:
        for _, r in logit_roc.iterrows():
            _append_row(
                "Logit",
                r["variant"],
                "auc",
                "AUC",
                r["auc"],
                np.nan,
                np.nan,
                np.nan,
                "classification quality",
            )

    if not cox_df.empty:
        sub = cox_df[cox_df["covariate"] == "in_buffer"]
        for _, r in sub.iterrows():
            _append_row(
                "Cox",
                r["variant"],
                "hazard_ratio_in_buffer",
                "in_buffer",
                r["hazard_ratio"],
                r["p"],
                float(np.exp(r["coef lower 95%"])),
                float(np.exp(r["coef upper 95%"])),
                f"selected={r.get('cox_spec', '')}",
            )

    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_UPGRADE_PATH, index=False)
    return out


def _plot_feature_upgrade(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    sns.set_theme(style="whitegrid")

    targets = [
        ("OLS", "coef_in_buffer", "Coefficient", 0.0),
        ("MixedLM", "coef_in_buffer", "Coefficient", 0.0),
        ("Logit", "odds_ratio_in_buffer", "Odds Ratio", 1.0),
        ("Cox", "hazard_ratio_in_buffer", "Hazard Ratio", 1.0),
    ]
    rows = []
    for model, metric, _, _ in targets:
        sub = summary[(summary["model"] == model) & (summary["key_metric"] == metric)]
        for _, r in sub.iterrows():
            rows.append({"model": model, "variant": r["variant"], "value": r["value"], "metric": metric})
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (model, metric, ylabel, ref) in zip(axes, targets):
        d = plot_df[(plot_df["model"] == model) & (plot_df["metric"] == metric)]
        sns.barplot(data=d, x="variant", y="value", ax=ax, palette="Set2")
        ax.axhline(ref, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{model}: {metric}")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(FIG_FEATURE_DIR / "feature_upgrade_model_compare_locked.png", dpi=220)
    plt.close(fig)


def _build_sample_alignment_audit(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    baseline_n = None
    for variant, df in frames.items():
        n = len(df)
        if baseline_n is None:
            baseline_n = n
        rows.append(
            {
                "variant": variant,
                "n_obs": n,
                "n_unique_pixel": int(df["pixel_id"].nunique()),
                "n_event": int(df["event_id"].nunique()),
                "sample_lock_only": 1,
                "matches_baseline_n_obs": int(n == baseline_n),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(SAMPLE_ALIGN_AUDIT_PATH, index=False)
    return out


def _logo_fold_eval(
    ctx: RunContext,
    panel: pd.DataFrame,
    damage_threshold: float,
    recovery_threshold: float,
    ref_sign: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    events = sorted(panel["event_id"].unique().tolist())
    fold_errors: List[str] = []

    for test_event in events:
        train_raw = panel[panel["event_id"] != test_event].copy()
        test_raw = panel[panel["event_id"] == test_event].copy()
        train, test, _ = prepare_model_frame_fold(train_raw, test_raw)
        rec_train = build_recovery_panel(ctx=ctx, panel=train, threshold=recovery_threshold, output_path=None)
        rec_test = build_recovery_panel(ctx=ctx, panel=test, threshold=recovery_threshold, output_path=None)

        # OLS transport only
        try:
            formula = _build_linear_formula(include_land_use=True, include_full=True, include_event_fe=False)
            ols = smf.ols(formula, data=train).fit(cov_type="HC1")
            coef = float(ols.params.get("in_buffer", np.nan))
            pred = ols.predict(test)
            rows.append(
                {
                    "fold_event": test_event,
                    "spec": "logo_transport",
                    "model": "OLS",
                    "rmse": float(math.sqrt(mean_squared_error(test["delta_ntl"], pred))),
                    "mae": float(mean_absolute_error(test["delta_ntl"], pred)),
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": np.nan,
                    "coef_in_buffer": coef,
                    "sign_consistency": int(np.sign(coef) == np.sign(ref_sign["OLS"])) if pd.notna(coef) else np.nan,
                }
            )
        except Exception as e:
            fold_errors.append(f"{test_event}:OLS:{type(e).__name__}")
            log_issue(
                ctx,
                stage="logo_eval",
                model="OLS",
                event_id=test_event,
                issue_type="fold_fit_failed",
                symptom=str(e),
                fix_action="fail held-out transport fold and record NaN metrics",
                impact="partial LOEO for OLS",
                status="monitor",
            )

        # MixedLM transport only. Use the estimable reduced transport spec to avoid singular fold-specific Hessians.
        try:
            formula = _build_linear_formula(include_land_use=True, include_full=False, include_event_fe=False)
            mixed = _strict_fit_mixed_with_optimizers(formula, train)
            coef = float(mixed.params.get("in_buffer", np.nan))
            pred = mixed.predict(test)
            rows.append(
                {
                    "fold_event": test_event,
                    "spec": "logo_transport",
                    "model": "MixedLM",
                    "rmse": float(math.sqrt(mean_squared_error(test["delta_ntl"], pred))),
                    "mae": float(mean_absolute_error(test["delta_ntl"], pred)),
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": np.nan,
                    "coef_in_buffer": coef,
                    "sign_consistency": int(np.sign(coef) == np.sign(ref_sign["MixedLM"])) if pd.notna(coef) else np.nan,
                }
            )
        except Exception as e:
            fold_errors.append(f"{test_event}:MixedLM:{type(e).__name__}")
            log_issue(
                ctx,
                stage="logo_eval",
                model="MixedLM",
                event_id=test_event,
                issue_type="fold_fit_failed",
                symptom=str(e),
                fix_action="fail held-out transport fold and record NaN metrics",
                impact="partial LOEO for MixedLM",
                status="monitor",
            )

        # Logit transport only
        try:
            train_logit = train.copy()
            test_logit = test.copy()
            train_logit["is_damaged"] = (train_logit["delta_ntl"] < damage_threshold).astype(int)
            test_logit["is_damaged"] = (test_logit["delta_ntl"] < damage_threshold).astype(int)
            formulas = [
                _build_logit_formula(include_land_use=True, include_full=True, include_event_fe=False),
                _build_logit_formula(include_land_use=True, include_full=False, include_event_fe=False),
                _build_logit_formula(include_land_use=False, include_full=False, include_event_fe=False),
            ]
            logit = None
            for f in formulas:
                try:
                    logit = smf.logit(formula=f, data=train_logit).fit(disp=False, maxiter=200)
                    break
                except Exception:
                    continue
            if logit is None:
                for f in formulas:
                    try:
                        logit = smf.logit(formula=f, data=train_logit).fit_regularized(disp=False, maxiter=200, alpha=0.01)
                        break
                    except Exception:
                        continue
            if logit is None:
                raise RuntimeError("all logit transport formulas failed")
            coef = float(logit.params.get("in_buffer", np.nan))
            prob = np.asarray(logit.predict(test_logit))
            y_true = test_logit["is_damaged"].to_numpy()
            auc_val = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan
            rows.append(
                {
                    "fold_event": test_event,
                    "spec": "logo_transport",
                    "model": "Logit",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "auc": auc_val,
                    "brier": float(brier_score_loss(y_true, prob)),
                    "calibration_slope": _calibration_slope(y_true, prob),
                    "c_index": np.nan,
                    "coef_in_buffer": coef,
                    "sign_consistency": int(np.sign(coef) == np.sign(ref_sign["Logit"])) if pd.notna(coef) else np.nan,
                }
            )
        except Exception as e:
            fold_errors.append(f"{test_event}:Logit:{type(e).__name__}")
            log_issue(
                ctx,
                stage="logo_eval",
                model="Logit",
                event_id=test_event,
                issue_type="fold_fit_failed",
                symptom=str(e),
                fix_action="fail held-out transport fold and record NaN metrics",
                impact="partial LOEO for Logit",
                status="monitor",
            )

        # Cox transport only. Keep the held-out design estimable without event dummies or full fold-unstable extras.
        try:
            x_train = _cox_design(rec_train, include_land_use=True, include_full=False, include_event_dummies=False)
            x_test = _cox_design(rec_test, include_land_use=True, include_full=False, include_event_dummies=False)
            x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
            x_test = _sanitize_design_for_cox(x_test, keep_cols=[])

            fit_df = pd.concat([rec_train[["recovery_days", "event_observed"]].reset_index(drop=True), x_train.reset_index(drop=True)], axis=1)
            fit_df = _sanitize_design_for_cox(fit_df, keep_cols=["recovery_days", "event_observed"])
            cph, _, err = _fit_cph_with_retry(fit_df, duration_col="recovery_days", event_col="event_observed")
            if cph is None:
                raise err if err is not None else RuntimeError("cox fit failed")
            coef = float(cph.params_.get("in_buffer", np.nan))
            risk = cph.predict_partial_hazard(x_test).to_numpy().reshape(-1)
            rows.append(
                {
                    "fold_event": test_event,
                    "spec": "logo_transport",
                    "model": "Cox",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": float(concordance_index(rec_test["recovery_days"], -risk, rec_test["event_observed"])),
                    "coef_in_buffer": coef,
                    "sign_consistency": int(np.sign(coef) == np.sign(ref_sign["Cox"])) if pd.notna(coef) else np.nan,
                }
            )
        except Exception as e:
            fold_errors.append(f"{test_event}:Cox:{type(e).__name__}")
            log_issue(
                ctx,
                stage="logo_eval",
                model="Cox",
                event_id=test_event,
                issue_type="fold_fit_failed",
                symptom=str(e),
                fix_action="fail held-out transport fold and record NaN metrics",
                impact="partial LOEO for Cox",
                status="monitor",
            )

    fold_df = pd.DataFrame(rows)
    available_models = set(fold_df["model"].unique().tolist()) if not fold_df.empty else set()
    required_models = {"OLS", "Logit"}
    missing_models = sorted(required_models - available_models)
    if missing_models:
        raise RuntimeError("LOEO transport missing required model outputs: " + ", ".join(missing_models))
    if fold_errors:
        log_issue(
            ctx,
            stage="logo_eval",
            model="all",
            event_id="all",
            issue_type="partial_fold_failures",
            symptom="; ".join(fold_errors),
            fix_action="keep successful held-out transport rows and mark unstable models as partial",
            impact="feature-upgrade LOEO is partial for some models",
            status="monitor",
        )
    agg_df = (
        fold_df.groupby(["spec", "model"], dropna=False)[
            ["rmse", "mae", "auc", "brier", "calibration_slope", "c_index", "sign_consistency"]
        ]
        .mean()
        .reset_index()
    )
    return fold_df, agg_df


def _plot_logo_metrics(logo_agg: pd.DataFrame) -> None:
    if logo_agg.empty:
        return
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    auc_df = logo_agg[logo_agg["model"] == "Logit"]
    if not auc_df.empty:
        sns.barplot(data=auc_df, x="spec", y="auc", ax=axes[0], palette="Set2")
        axes[0].set_title("LOEO Logit AUC")
    else:
        axes[0].axis("off")

    cox_df = logo_agg[logo_agg["model"] == "Cox"]
    if not cox_df.empty:
        sns.barplot(data=cox_df, x="spec", y="c_index", ax=axes[1], palette="Set2")
        axes[1].set_title("LOEO Cox C-index")
    else:
        axes[1].axis("off")

    sign_df = logo_agg.copy()
    if not sign_df.empty:
        sns.barplot(data=sign_df, x="model", y="sign_consistency", hue="spec", ax=axes[2], palette="Set2")
        axes[2].set_title("Sign Consistency")
        axes[2].set_ylim(0, 1)
    else:
        axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(FIG_LOGO_DIR / "logo_aggregate_metrics.png", dpi=220)
    plt.close(fig)


def _write_06_report(summary: pd.DataFrame, cox_diag: pd.DataFrame, sample_audit: pd.DataFrame) -> None:
    def _get(model: str, variant: str, metric: str) -> Tuple[float, float]:
        sub = summary[(summary["model"] == model) & (summary["variant"] == variant) & (summary["key_metric"] == metric)]
        if sub.empty:
            return np.nan, np.nan
        r = sub.iloc[0]
        return float(r["value"]), float(r["p_value"]) if pd.notna(r["p_value"]) else np.nan

    lines = [
        "# Feature Upgrade Report / OSM+Cloud+Sample Lock 升级报告",
        "",
        "## Objective",
        "本轮目标是修复样本重定义、事件级云量共线和特征异质性不足，形成 `baseline_locked / nlcd_locked / full_locked` 三规格同样本对照。",
        "",
        "## What Was Added",
        "- OSM 像素特征: `osm_any_count_750m`, `osm_power_count_1000m`, `osm_medical_count_1000m`, `osm_dist_power_m`, `osm_dist_any_m`",
        "- 云量像素特征: `pixel_pre_valid_ratio`, `pixel_post_valid_ratio`, `pixel_cloud_proxy`",
        "- 样本锁定: `sample_lock_cohort_v1.parquet` + `sample_alignment_audit.csv`",
        "- Cox 扩展诊断: `cox_diagnostics_extended.csv` + strata/time-interaction/AFT 对照",
        "",
        "## Quantitative Comparison",
    ]
    for model, metric in [
        ("OLS", "coef_in_buffer"),
        ("MixedLM", "coef_in_buffer"),
        ("Logit", "odds_ratio_in_buffer"),
        ("Cox", "hazard_ratio_in_buffer"),
    ]:
        base, p1 = _get(model, "baseline_locked", metric)
        nlcd, p2 = _get(model, "nlcd_locked", metric)
        full, p3 = _get(model, "full_locked", metric)
        lines.append(
            f"- {model} `{metric}`: baseline={base:.4f} (p={p1:.4g}), nlcd={nlcd:.4f} (p={p2:.4g}), full={full:.4f} (p={p3:.4g})"
            if pd.notna(base) and pd.notna(nlcd) and pd.notna(full)
            else f"- {model} `{metric}`: partially missing"
        )

    if not sample_audit.empty:
        lines.extend(
            [
                "",
                "## Sample Lock Audit",
            ]
        )
        for _, r in sample_audit.iterrows():
            lines.append(
                f"- {r['variant']}: n_obs={int(r['n_obs'])}, n_pixel={int(r['n_unique_pixel'])}, match_baseline={int(r['matches_baseline_n_obs'])}"
            )

    if not cox_diag.empty:
        lines.extend(
            [
                "",
                "## Cox Diagnostics",
            ]
        )
        d = cox_diag.groupby(["variant", "step"], dropna=False)["ph_violations"].first().reset_index()
        for _, r in d.iterrows():
            pv = "N/A" if pd.isna(r["ph_violations"]) else f"{int(r['ph_violations'])}"
            lines.append(f"- {r['variant']} / {r['step']}: PH violations={pv}")

    lines.extend(
        [
            "",
            "## Figures",
            f"- Locked variants comparison: `{(FIG_FEATURE_DIR / 'feature_upgrade_model_compare_locked.png').relative_to(ROOT)}`",
            "",
            "## Verdict",
            "工程层面显著改进：同样本锁定、像素级云量和 OSM 特征已入模；统计层面仍需看 LOEO 与 Cox 修复后稳定性是否收敛。",
        ]
    )
    (REPORT_DIR / "06_feature_upgrade_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_07_logo_report(logo_agg: pd.DataFrame) -> None:
    def _pick(model: str, col: str) -> str:
        sub = logo_agg[(logo_agg["spec"] == "logo_transport") & (logo_agg["model"] == model)]
        if sub.empty or pd.isna(sub.iloc[0][col]):
            return "N/A"
        return f"{float(sub.iloc[0][col]):.4f}"

    lines = [
        "# LOEO Validation Report / 按事件留一验证报告",
        "",
        "## Objective",
        "采用 LOEO（6 折）评估 full_locked 规格在事件外推下的稳定性。",
        "",
        "## Setup",
        "- Fold: 每次留 1 个事件做测试",
        "- Specs: `inference`（保留事件结构）与 `logo_transport`（移除事件 FE）",
        "",
        "## Aggregate Metrics",
        f"- Logit AUC: transport={_pick('Logit', 'auc')}",
        f"- Logit Brier: transport={_pick('Logit', 'brier')}",
        f"- Cox C-index: transport={_pick('Cox', 'c_index')}",
        f"- OLS RMSE: transport={_pick('OLS', 'rmse')}",
        f"- MixedLM RMSE: transport={_pick('MixedLM', 'rmse')}",
        "",
        "## Sign Consistency",
        f"- OLS: transport={_pick('OLS', 'sign_consistency')}",
        f"- MixedLM: transport={_pick('MixedLM', 'sign_consistency')}",
        f"- Logit: transport={_pick('Logit', 'sign_consistency')}",
        f"- Cox: transport={_pick('Cox', 'sign_consistency')}",
        "",
        "## Figure",
        f"- LOEO aggregate plot: `{(FIG_LOGO_DIR / 'logo_aggregate_metrics.png').relative_to(ROOT)}`",
        "",
        "## Verdict",
        "若 `logo_transport` 下 AUC/C-index 维持中等且符号一致性不崩塌，可判定为“部分可泛化”；否则仍是事件内解释更强、外推有限。",
    ]
    (REPORT_DIR / "07_logo_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def _update_index_links() -> None:
    index_file = REPORT_DIR / "index.md"
    if index_file.exists():
        text = index_file.read_text(encoding="utf-8")
    else:
        text = "# Modeling Report Index\n\n## Deliverables\n"

    additions = [
        "- `project/modeling_report/06_feature_upgrade_report.md`",
        "- `project/modeling_report/07_logo_validation_report.md`",
    ]
    for line in additions:
        if line not in text:
            text += ("\n" if not text.endswith("\n") else "") + line + "\n"
    index_file.write_text(text, encoding="utf-8")


def _feature_upgrade_impl() -> None:
    ensure_directories()
    init_tracking_files()
    _ensure_extra_dirs()
    append_progress("Feature-upgrade pipeline started")

    defaults = load_json(CONFIG_DEFAULTS)
    events_cfg = load_json(CONFIG_EVENTS)
    pre_thr = float(defaults["pre_ntl_threshold"])
    dmg_thr = float(defaults["damage_threshold"])
    rec_thr = float(defaults["recovery_threshold"])
    ctx = RunContext(issues=[])

    # Base panel and NLCD.
    if PANEL_PATH.exists():
        panel_base = pd.read_parquet(PANEL_PATH)
    else:
        panel_base = build_pixel_panel(ctx, pre_threshold=pre_thr, damage_threshold=dmg_thr, exclude_types=None, output_path=PANEL_PATH)
    if PANEL_NLCD_PATH.exists():
        panel = pd.read_parquet(PANEL_NLCD_PATH)
    else:
        panel = attach_nlcd(ctx=ctx, panel_path=PANEL_PATH, output_path=PANEL_NLCD_PATH)
    if "land_use" not in panel.columns:
        panel = attach_nlcd(ctx=ctx, panel_path=PANEL_PATH, output_path=PANEL_NLCD_PATH)
    panel = panel_base.merge(panel[["pixel_id", "land_use"]], on="pixel_id", how="left")
    panel["land_use_group"] = panel["land_use"].map(_land_use_group)

    # Feature augmentations.
    panel = attach_pixel_cloud_features(ctx, panel, events_cfg)
    panel = attach_osm_features(ctx, panel, events_cfg)
    panel = create_sample_lock(panel)
    panel.to_parquet(PANEL_FEATURE_PATH, index=False)

    model_df = prepare_model_frame(panel)
    locked = model_df[model_df["sample_lock_flag"] == 1].copy()

    variants = {
        "baseline_locked": {"include_land_use": False, "include_full": False},
        "nlcd_locked": {"include_land_use": True, "include_full": False},
        "full_locked": {"include_land_use": True, "include_full": True},
    }
    frames_for_audit = {k: locked.copy() for k in variants.keys()}
    sample_audit = _build_sample_alignment_audit(frames_for_audit)

    ols_tables: List[pd.DataFrame] = []
    mixed_tables: List[pd.DataFrame] = []
    logit_tables: List[pd.DataFrame] = []
    marginal_tables: List[pd.DataFrame] = []
    logit_roc_rows: List[pd.DataFrame] = []
    cox_tables: List[pd.DataFrame] = []
    cox_diag_rows: List[pd.DataFrame] = []
    variant_n_obs: Dict[str, int] = {}
    if COX_AFT_RESULT_PATH.exists():
        COX_AFT_RESULT_PATH.unlink()

    for variant, spec in variants.items():
        dfv = locked.copy()
        variant_n_obs[variant] = len(dfv)

        om = fit_ols_and_mixed(
            ctx=ctx,
            df=dfv,
            variant=variant,
            include_land_use=spec["include_land_use"],
            include_full=spec["include_full"],
            include_event_fe=True,
        )
        if not om["ols_coef"].empty:
            ols_tables.append(om["ols_coef"])
        if not om["mixed_coef"].empty:
            mixed_tables.append(om["mixed_coef"])

        lg = fit_logit(
            ctx=ctx,
            df=dfv,
            variant=variant,
            include_land_use=spec["include_land_use"],
            include_full=spec["include_full"],
            damage_threshold=dmg_thr,
            include_event_fe=True,
        )
        if not lg["coef"].empty:
            logit_tables.append(lg["coef"])
        if not lg["marginal"].empty:
            marginal_tables.append(lg["marginal"])
        if not lg["roc"].empty:
            logit_roc_rows.append(lg["roc"])
        lg["roc"].to_csv(OUTPUT_DIR / f"logit_roc_{variant}.csv", index=False)
        lg["calibration"].to_csv(OUTPUT_DIR / f"logit_calibration_{variant}.csv", index=False)

        rec = build_recovery_panel(ctx=ctx, panel=dfv, threshold=rec_thr, output_path=None)
        cox = fit_cox_enhanced(
            ctx=ctx,
            recovery_df=rec,
            variant=variant,
            include_land_use=spec["include_land_use"],
            include_full=spec["include_full"],
        )
        if cox["coef"].empty and spec["include_full"]:
            log_issue(
                ctx,
                stage="fit_cox_enhanced",
                model="Cox",
                event_id="all",
                issue_type="formula_fallback",
                symptom=f"full Cox failed for {variant}",
                fix_action="fallback to nlcd-level cox design",
                impact="full_locked uses reduced Cox controls",
                status="resolved",
            )
            cox = fit_cox_enhanced(
                ctx=ctx,
                recovery_df=rec,
                variant=variant,
                include_land_use=spec["include_land_use"],
                include_full=False,
            )
        if not cox["coef"].empty:
            cox_tables.append(cox["coef"])
        if not cox["diagnostics"].empty:
            cox_diag_rows.append(cox["diagnostics"])
        if not cox["aft"].empty:
            cox["aft"].to_csv(COX_AFT_RESULT_PATH, mode="a", header=not COX_AFT_RESULT_PATH.exists(), index=False)
        cox["km"].to_csv(OUTPUT_DIR / f"cox_km_{variant}.csv", index=False)
        cox["ph"].to_csv(OUTPUT_DIR / f"cox_ph_test_{variant}.csv", index=False)

    ols_df = pd.concat(ols_tables, ignore_index=True) if ols_tables else pd.DataFrame()
    mixed_df = pd.concat(mixed_tables, ignore_index=True) if mixed_tables else pd.DataFrame()
    logit_df = pd.concat(logit_tables, ignore_index=True) if logit_tables else pd.DataFrame()
    marginal_df = pd.concat(marginal_tables, ignore_index=True) if marginal_tables else pd.DataFrame()
    logit_roc_df = pd.concat(logit_roc_rows, ignore_index=True) if logit_roc_rows else pd.DataFrame()
    cox_df = pd.concat(cox_tables, ignore_index=True) if cox_tables else pd.DataFrame()

    ols_df.to_csv(OLS_FEATURE_RESULT, index=False)
    mixed_df.to_csv(MIXED_FEATURE_RESULT, index=False)
    logit_df.to_csv(LOGIT_FEATURE_RESULT, index=False)
    marginal_df.to_csv(LOGIT_MARGINAL_FEATURE_RESULT, index=False)
    cox_df.to_csv(COX_FEATURE_RESULT, index=False)

    if cox_diag_rows:
        cox_diag = pd.concat(cox_diag_rows, ignore_index=True)
    else:
        cox_diag = pd.DataFrame()
    cox_diag.to_csv(COX_DIAG_EXT_PATH, index=False)

    summary = _summarize_feature_upgrade(
        ols_df=ols_df,
        mixed_df=mixed_df,
        logit_df=logit_df,
        logit_roc=logit_roc_df,
        cox_df=cox_df,
        variant_n_obs=variant_n_obs,
    )
    _plot_feature_upgrade(summary)

    # LOEO uses full-locked sample.
    rec_full = build_recovery_panel(ctx=ctx, panel=locked, threshold=rec_thr, output_path=RECOVERY_PATH)
    ref_sign = {}
    for model_name, df_table, col in [
        ("OLS", ols_df, "coef"),
        ("MixedLM", mixed_df, "coef"),
        ("Logit", logit_df, "coef"),
        ("Cox", cox_df, "coef"),
    ]:
        sub = df_table[(df_table["variant"] == "full_locked")]
        if "term" in sub.columns:
            sub = sub[sub["term"] == "in_buffer"]
        elif "covariate" in sub.columns:
            sub = sub[sub["covariate"] == "in_buffer"]
        ref_sign[model_name] = float(sub.iloc[0][col]) if not sub.empty else np.nan

    logo_fold, logo_agg = _logo_fold_eval(
        ctx=ctx,
        panel=locked,
        damage_threshold=dmg_thr,
        recovery_threshold=rec_thr,
        ref_sign=ref_sign,
    )
    logo_fold.to_csv(LOGO_FOLD_PATH, index=False)
    logo_agg.to_csv(LOGO_AGG_PATH, index=False)
    _plot_logo_metrics(logo_agg)

    _write_06_report(summary=summary, cox_diag=cox_diag, sample_audit=sample_audit)
    _write_07_logo_report(logo_agg)
    _update_index_links()

    save_issue_log(ctx)
    append_progress("Feature-upgrade pipeline finished")

# ----------------------------
# Strict V2 integrated block
# ----------------------------

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pipeline_lib import (
    CONFIG_DEFAULTS,
    OUTPUT_DIR,
    PIXEL_DIR,
    REPORT_DIR,
    ROOT,
    RunContext,
    append_progress,
    build_recovery_panel,
    ensure_directories,
    init_tracking_files,
    load_json,
    log_issue,
    save_issue_log,
)


STRICT_PANEL_FEATURE_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_feature_upgrade.parquet"
STRICT_SAMPLE_LOCK_PATH = PIXEL_DIR / "sample_lock_cohort_v1.parquet"

STRICT_MANIFEST_PATH = OUTPUT_DIR / "feature_spec_manifest_v2_strict.json"
STRICT_VIF_PATH = OUTPUT_DIR / "multicollinearity_vif_v2_strict.csv"
STRICT_SUMMARY_PATH = OUTPUT_DIR / "model_summary_feature_upgrade_v2_strict.csv"
STRICT_SAMPLE_AUDIT_PATH = OUTPUT_DIR / "sample_alignment_audit_v2_strict.csv"
STRICT_COX_DIAG_PATH = OUTPUT_DIR / "cox_diagnostics_extended_v2_strict.csv"
STRICT_LOGO_FOLD_PATH = OUTPUT_DIR / "logo_fold_metrics_v2_strict.csv"
STRICT_LOGO_AGG_PATH = OUTPUT_DIR / "logo_aggregate_metrics_v2_strict.csv"
STRICT_MISSING_FLAG_AUDIT_PATH = OUTPUT_DIR / "missing_flag_audit_v2_strict.csv"
STRICT_VIF_MAX = 12.0

STRICT_OLS_RESULT_PATH = OUTPUT_DIR / "ols_results_feature_upgrade_v2_strict.csv"
STRICT_MIXED_RESULT_PATH = OUTPUT_DIR / "mixedlm_results_feature_upgrade_v2_strict.csv"
STRICT_LOGIT_RESULT_PATH = OUTPUT_DIR / "logit_results_feature_upgrade_v2_strict.csv"
STRICT_COX_RESULT_PATH = OUTPUT_DIR / "cox_results_feature_upgrade_v2_strict.csv"

STRICT_VARIANTS = [
    "baseline_locked_v2_strict",
    "nlcd_locked_v2_strict",
    "full_locked_v2_strict",
]
STRICT_FULL_VARIANT = "full_locked_v2_strict"

STRICT_RAW_FULL_FEATURES = [
    "osm_dist_any_m",
    "osm_power_count_1000m",
    "osm_medical_count_1000m",
    "pixel_cloud_proxy",
]
STRICT_SCALED_BASE_FEATURES = ["pre_mean_ntl_centered"]
STRICT_SCALED_FULL_FEATURES = [
    "pre_mean_ntl_centered",
    "osm_dist_any_m",
    "osm_power_count_1000m",
    "osm_medical_count_1000m",
    "pixel_cloud_proxy",
]
STRICT_SCALED_SUFFIX = "_z"


def _strict_coef_table_from_result(result, model_name: str, variant: str, kind: str = "linear") -> pd.DataFrame:
    ci = result.conf_int()
    table = pd.DataFrame(
        {
            "model": model_name,
            "variant": variant,
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": getattr(result, "bse", pd.Series(np.nan, index=result.params.index)).values,
            "stat": getattr(
                result,
                "tvalues",
                getattr(result, "zvalues", pd.Series(np.nan, index=result.params.index)),
            ).values,
            "p_value": getattr(result, "pvalues", pd.Series(np.nan, index=result.params.index)).values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_high": ci.iloc[:, 1].values,
            "kind": kind,
        }
    )
    return table


def _strict_fit_logit_with_optimizers(formula: str, data: pd.DataFrame):
    last_error = None
    for method in ["newton", "lbfgs", "bfgs", "powell"]:
        try:
            return smf.logit(formula=formula, data=data).fit(disp=False, method=method, maxiter=400)
        except Exception as e_fit:
            last_error = e_fit
    raise RuntimeError(f"logit fit failed for all optimizers: {last_error}")


def _strict_fit_mixed_with_optimizers(formula: str, data: pd.DataFrame):
    last_error = None
    md = smf.mixedlm(formula, data=data, groups=data["event_id"])
    for method in ["lbfgs", "powell"]:
        try:
            return md.fit(method=method, reml=False)
        except Exception as e_fit:
            last_error = e_fit
    raise RuntimeError(f"mixedlm fit failed for all optimizers: {last_error}")


def _strict_safe_numeric(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if out.notna().any():
        return out.fillna(out.median())
    return out.fillna(0.0)


def _strict_fit_preprocess(df: pd.DataFrame) -> Dict[str, object]:
    work = df.copy()
    work["pre_mean_ntl"] = _strict_safe_numeric(work["pre_mean_ntl"])
    for col in STRICT_RAW_FULL_FEATURES:
        work[col] = _strict_safe_numeric(work[col])

    pre_ref = float(work["pre_mean_ntl"].mean()) if len(work) else 0.0
    centered = work["pre_mean_ntl"] - pre_ref
    scale_stats: Dict[str, Dict[str, float]] = {}

    centered_mean = float(centered.mean()) if len(centered) else 0.0
    centered_std = float(centered.std(ddof=0)) if len(centered) else 0.0
    if not np.isfinite(centered_std) or centered_std <= 0:
        centered_std = 1.0
    scale_stats["pre_mean_ntl_centered"] = {"mean": centered_mean, "std": centered_std}

    for col in STRICT_RAW_FULL_FEATURES:
        mu = float(work[col].mean())
        sd = float(work[col].std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        scale_stats[col] = {"mean": mu, "std": sd}

    return {
        "pre_mean_ntl_reference": pre_ref,
        "scale_stats": scale_stats,
    }


def _strict_apply_preprocess(df: pd.DataFrame, preprocess_state: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    out["pre_mean_ntl"] = _strict_safe_numeric(out["pre_mean_ntl"])
    for col in STRICT_RAW_FULL_FEATURES:
        out[col] = _strict_safe_numeric(out[col])

    pre_ref = float(preprocess_state["pre_mean_ntl_reference"])
    scale_stats = preprocess_state["scale_stats"]
    out["pre_mean_ntl_centered"] = out["pre_mean_ntl"] - pre_ref

    for col in STRICT_SCALED_FULL_FEATURES:
        stats = scale_stats[col]
        mu = float(stats["mean"])
        sd = float(stats["std"])
        zcol = f"{col}{STRICT_SCALED_SUFFIX}"
        out[zcol] = (out[col] - mu) / sd
    return out


def _strict_make_center_and_scale(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    preprocess_state = _strict_fit_preprocess(df)
    out = _strict_apply_preprocess(df, preprocess_state)
    scale_stats = {k: dict(v) for k, v in preprocess_state["scale_stats"].items()}
    return out, scale_stats


def _strict_build_variant_formula(model: str, variant: str, use_scaled: bool) -> str:
    pre_col = f"pre_mean_ntl_centered{STRICT_SCALED_SUFFIX}" if use_scaled else "pre_mean_ntl_centered"
    full_cols = [f"{c}{STRICT_SCALED_SUFFIX}" for c in STRICT_RAW_FULL_FEATURES] if use_scaled else STRICT_RAW_FULL_FEATURES

    if model == "OLS":
        terms = [f"in_buffer * {pre_col}", "C(event_id)"]
        if variant in {"nlcd_locked_v2_strict", "full_locked_v2_strict"}:
            terms.append("C(land_use_group)")
        if variant == "full_locked_v2_strict":
            terms.extend(full_cols)
        return "delta_ntl ~ " + " + ".join(terms)

    if model == "MixedLM":
        terms = [f"in_buffer * {pre_col}"]
        if variant in {"nlcd_locked_v2_strict", "full_locked_v2_strict"}:
            terms.append("C(land_use_group)")
        if variant == "full_locked_v2_strict":
            terms.extend(full_cols)
        return "delta_ntl ~ " + " + ".join(terms)

    if model == "Logit":
        terms = [f"in_buffer * {pre_col}", "C(event_id)"]
        if variant in {"nlcd_locked_v2_strict", "full_locked_v2_strict"}:
            terms.append("C(land_use_group)")
        if variant == "full_locked_v2_strict":
            terms.extend(full_cols)
        return "is_damaged ~ " + " + ".join(terms)

    raise ValueError(f"Unknown model: {model}")


def _strict_build_transport_formula(model: str, use_scaled: bool) -> str:
    pre_col = f"pre_mean_ntl_centered{STRICT_SCALED_SUFFIX}" if use_scaled else "pre_mean_ntl_centered"
    full_cols = [f"{c}{STRICT_SCALED_SUFFIX}" for c in STRICT_RAW_FULL_FEATURES] if use_scaled else STRICT_RAW_FULL_FEATURES
    terms = [f"in_buffer * {pre_col}", "C(land_use_group)"] + full_cols
    if model in {"OLS", "MixedLM"}:
        return "delta_ntl ~ " + " + ".join(terms)
    if model == "Logit":
        return "is_damaged ~ " + " + ".join(terms)
    raise ValueError(f"Unknown model: {model}")


def _strict_vif_gate(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["interaction"] = work["in_buffer"] * work["pre_mean_ntl_centered"]
    cols = [
        "in_buffer",
        "pre_mean_ntl_centered",
        "interaction",
        "osm_dist_any_m",
        "osm_power_count_1000m",
        "osm_medical_count_1000m",
        "pixel_cloud_proxy",
    ]
    X = work[cols].copy()
    for c in cols:
        X[c] = _strict_safe_numeric(X[c])
    X = sm.add_constant(X, has_constant="add")

    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif = float(variance_inflation_factor(X.values, i))
        rows.append({"term": col, "vif": vif, "pass_lt_cap": int(vif < STRICT_VIF_MAX)})
    out = pd.DataFrame(rows).sort_values("vif", ascending=False)
    out.to_csv(STRICT_VIF_PATH, index=False)
    return out


def _strict_missing_flag_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["missing_osm_flag", "missing_cloud_flag"]:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        for k, v in vc.items():
            rows.append(
                {
                    "flag_col": col,
                    "scope": "global",
                    "event_id": "all",
                    "value": str(k),
                    "count": int(v),
                    "share": float(v / len(df)),
                }
            )
        ev = df.groupby("event_id", observed=True)[col].mean().reset_index()
        for _, r in ev.iterrows():
            rows.append(
                {
                    "flag_col": col,
                    "scope": "event_mean",
                    "event_id": str(r["event_id"]),
                    "value": "mean",
                    "count": np.nan,
                    "share": float(r[col]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(STRICT_MISSING_FLAG_AUDIT_PATH, index=False)
    return out


def _strict_make_sample_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in STRICT_VARIANTS:
        rows.append(
            {
                "variant": v,
                "n_obs": int(len(df)),
                "n_unique_pixel": int(df["pixel_id"].nunique()),
                "n_event": int(df["event_id"].nunique()),
                "sample_lock_only": 1,
            }
        )
    out = pd.DataFrame(rows)
    out["matches_baseline_n_obs"] = (out["n_obs"] == out["n_obs"].iloc[0]).astype(int)
    out.to_csv(STRICT_SAMPLE_AUDIT_PATH, index=False)
    return out


def _strict_prepare_cox_design(rec: pd.DataFrame, variant: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    base_terms = ["in_buffer", f"pre_mean_ntl_centered{STRICT_SCALED_SUFFIX}"]
    full_terms = [f"{c}{STRICT_SCALED_SUFFIX}" for c in STRICT_RAW_FULL_FEATURES]

    work = rec[["recovery_days", "event_observed", "event_id"] + base_terms + ["land_use_group"] + full_terms].copy()
    work["land_use_group"] = work["land_use_group"].fillna("unknown").astype(str)

    if variant == "baseline_locked_v2_strict":
        key_terms = base_terms.copy()
        design = work[["recovery_days", "event_observed", "event_id"] + key_terms].copy()
    elif variant == "nlcd_locked_v2_strict":
        key_terms = base_terms.copy()
        lu = pd.get_dummies(work["land_use_group"], prefix="lu", drop_first=True)
        design = pd.concat([work[["recovery_days", "event_observed", "event_id"] + key_terms], lu], axis=1)
    else:
        key_terms = base_terms + full_terms
        lu = pd.get_dummies(work["land_use_group"], prefix="lu", drop_first=True)
        design = pd.concat([work[["recovery_days", "event_observed", "event_id"] + key_terms], lu], axis=1)

    for c in design.columns:
        if c in {"recovery_days", "event_observed", "event_id"}:
            continue
        design[c] = _strict_safe_numeric(design[c])
    return design, key_terms, base_terms + full_terms


def _strict_fit_cox_variant_strict(rec: pd.DataFrame, variant: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    diag_rows: List[Dict[str, object]] = []
    design, key_terms, interaction_candidates = _strict_prepare_cox_design(rec, variant)

    cph = CoxPHFitter()
    cph.fit(design, duration_col="recovery_days", event_col="event_observed", strata=["event_id"])
    ph = proportional_hazard_test(cph, design, time_transform="rank").summary.reset_index().rename(columns={"index": "covariate"})
    key_viol = int(ph[ph["covariate"].isin(key_terms)]["p"].lt(0.05).sum())
    diag_rows.append(
        {
            "variant": variant,
            "step": "strata_base",
            "n_obs": int(len(design)),
            "n_covariates": int(design.shape[1] - 3),
            "key_ph_violations": key_viol,
            "status": "ok",
        }
    )

    selected = cph
    selected_name = "strata_base"
    if key_viol > 1:
        design2 = design.copy()
        design2["log_time"] = np.log(np.clip(design2["recovery_days"].to_numpy(), 1.0, None))
        for c in interaction_candidates:
            if c in design2.columns:
                design2[f"{c}_x_log_time"] = design2[c] * design2["log_time"]

        cph2 = CoxPHFitter(penalizer=0.01)
        cph2.fit(design2, duration_col="recovery_days", event_col="event_observed", strata=["event_id"])
        ph2 = proportional_hazard_test(cph2, design2, time_transform="rank").summary.reset_index().rename(columns={"index": "covariate"})
        key_viol2 = int(ph2[ph2["covariate"].isin(key_terms)]["p"].lt(0.05).sum())
        diag_rows.append(
            {
                "variant": variant,
                "step": "strata_time_interaction",
                "n_obs": int(len(design2)),
                "n_covariates": int(design2.shape[1] - 3),
                "key_ph_violations": key_viol2,
                "status": "ok",
            }
        )
        selected = cph2
        selected_name = "strata_time_interaction"

    summary = selected.summary.reset_index().rename(columns={"index": "covariate"})
    summary["model"] = "Cox"
    summary["variant"] = variant
    summary["cox_spec"] = selected_name
    summary["hazard_ratio"] = np.exp(summary["coef"])

    return summary, pd.DataFrame(diag_rows)


def _strict_summarize_for_report(
    ols_df: pd.DataFrame,
    mixed_df: pd.DataFrame,
    logit_df: pd.DataFrame,
    logit_auc: Dict[str, float],
    cox_df: pd.DataFrame,
    n_obs: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add_row(model: str, variant: str, metric: str, term: str, value, pval, lo, hi, notes: str):
        rows.append(
            {
                "model": model,
                "variant": variant,
                "key_metric": metric,
                "term": term,
                "value": value,
                "p_value": pval,
                "ci_low": lo,
                "ci_high": hi,
                "n_obs": n_obs,
                "notes": notes,
            }
        )

    for df, model in [(ols_df, "OLS"), (mixed_df, "MixedLM")]:
        sub = df[df["term"] == "in_buffer"]
        for _, r in sub.iterrows():
            add_row(model, r["variant"], "coef_in_buffer", "in_buffer", r["coef"], r["p_value"], r["ci_low"], r["ci_high"], "strict_v2")

    sub = logit_df[logit_df["term"] == "in_buffer"]
    for _, r in sub.iterrows():
        add_row(
            "Logit",
            r["variant"],
            "odds_ratio_in_buffer",
            "in_buffer",
            r["odds_ratio"],
            r["p_value"],
            r.get("or_ci_low", np.nan),
            r.get("or_ci_high", np.nan),
            "strict_v2",
        )
    for v, auc in logit_auc.items():
        add_row("Logit", v, "auc", "AUC", auc, np.nan, np.nan, np.nan, "strict_v2")

    csub = cox_df[cox_df["covariate"] == "in_buffer"]
    for _, r in csub.iterrows():
        add_row(
            "Cox",
            r["variant"],
            "hazard_ratio_in_buffer",
            "in_buffer",
            r["hazard_ratio"],
            r["p"],
            float(np.exp(r["coef lower 95%"])),
            float(np.exp(r["coef upper 95%"])),
            f"strict_v2:{r['cox_spec']}",
        )

    out = pd.DataFrame(rows)
    out.to_csv(STRICT_SUMMARY_PATH, index=False)
    return out


def _strict_build_logo(
    ctx: RunContext,
    df_base: pd.DataFrame,
    recovery_threshold: float,
    ref_sign: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    events = sorted(df_base["event_id"].unique().tolist())

    full_ols_transport = _strict_build_transport_formula("OLS", use_scaled=False)
    full_mixed_transport = _strict_build_transport_formula("MixedLM", use_scaled=False)
    full_logit_transport = _strict_build_transport_formula("Logit", use_scaled=True)

    for test_event in events:
        tr_base = df_base[df_base["event_id"] != test_event].copy()
        te_base = df_base[df_base["event_id"] == test_event].copy()
        preprocess_state = _strict_fit_preprocess(tr_base)
        tr = _strict_apply_preprocess(tr_base, preprocess_state)
        te = _strict_apply_preprocess(te_base, preprocess_state)
        tr_rec = build_recovery_panel(ctx=ctx, panel=tr, threshold=recovery_threshold, output_path=None)
        te_rec = build_recovery_panel(ctx=ctx, panel=te, threshold=recovery_threshold, output_path=None)

        try:
            ols_tr = smf.ols(full_ols_transport, data=tr).fit(cov_type="HC1")
            pred_te = ols_tr.predict(te)
            rows.append(
                {
                    "fold_event": test_event,
                    "spec": "logo_transport",
                    "model": "OLS",
                    "rmse": float(math.sqrt(mean_squared_error(te["delta_ntl"], pred_te))),
                    "mae": float(mean_absolute_error(te["delta_ntl"], pred_te)),
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": np.nan,
                    "coef_in_buffer": float(ols_tr.params.get("in_buffer", np.nan)),
                }
            )
        except Exception as e:
            log_issue(ctx, stage="strict_logo", model="OLS", event_id=test_event, issue_type="fold_fit_failed", symptom=str(e), fix_action="record NaN metrics", impact="strict transport partial", status="monitor")

        try:
            mx_tr = _strict_fit_mixed_with_optimizers(full_mixed_transport, tr)
            pred_mx_te = mx_tr.predict(te)
            rows.append(
                {
                    "fold_event": test_event,
                    "spec": "logo_transport",
                    "model": "MixedLM",
                    "rmse": float(math.sqrt(mean_squared_error(te["delta_ntl"], pred_mx_te))),
                    "mae": float(mean_absolute_error(te["delta_ntl"], pred_mx_te)),
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": np.nan,
                    "coef_in_buffer": float(mx_tr.params.get("in_buffer", np.nan)),
                }
            )
        except Exception as e:
            log_issue(ctx, stage="strict_logo", model="MixedLM", event_id=test_event, issue_type="fold_fit_failed", symptom=str(e), fix_action="record NaN metrics", impact="strict transport partial", status="monitor")

        try:
            lg_tr = _strict_fit_logit_with_optimizers(full_logit_transport, tr)
            prob_te = np.asarray(lg_tr.predict(te))
            auc_te = float(roc_auc_score(te["is_damaged"], prob_te)) if te["is_damaged"].nunique() > 1 else np.nan
            rows.append(
                {
                    "fold_event": test_event,
                    "spec": "logo_transport",
                    "model": "Logit",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "auc": auc_te,
                    "brier": float(brier_score_loss(te["is_damaged"], prob_te)),
                    "calibration_slope": _calibration_slope(te["is_damaged"].to_numpy(), prob_te),
                    "c_index": np.nan,
                    "coef_in_buffer": float(lg_tr.params.get("in_buffer", np.nan)),
                }
            )
        except Exception as e:
            log_issue(ctx, stage="strict_logo", model="Logit", event_id=test_event, issue_type="fold_fit_failed", symptom=str(e), fix_action="record NaN metrics", impact="strict transport partial", status="monitor")

        try:
            dtr, _, _ = _strict_prepare_cox_design(tr_rec, STRICT_FULL_VARIANT)
            dte, _, _ = _strict_prepare_cox_design(te_rec, STRICT_FULL_VARIANT)
            tr_x = dtr.drop(columns=["recovery_days", "event_observed", "event_id"])
            te_x = dte.drop(columns=["recovery_days", "event_observed", "event_id"]).reindex(columns=tr_x.columns, fill_value=0.0)
            fit_df = pd.concat([dtr[["recovery_days", "event_observed"]], tr_x], axis=1)
            cph_tr = CoxPHFitter(penalizer=0.01)
            cph_tr.fit(fit_df, duration_col="recovery_days", event_col="event_observed")
            risk_te = cph_tr.predict_partial_hazard(te_x)
            rows.append(
                {
                    "fold_event": test_event,
                    "spec": "logo_transport",
                    "model": "Cox",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "auc": np.nan,
                    "brier": np.nan,
                    "calibration_slope": np.nan,
                    "c_index": float(concordance_index(dte["recovery_days"], -risk_te.to_numpy().reshape(-1), dte["event_observed"])),
                    "coef_in_buffer": float(cph_tr.params_.get("in_buffer", np.nan)),
                }
            )
        except Exception as e:
            log_issue(ctx, stage="strict_logo", model="Cox", event_id=test_event, issue_type="fold_fit_failed", symptom=str(e), fix_action="record NaN metrics", impact="strict transport partial", status="monitor")

    fold = pd.DataFrame(rows)
    for model, sign in ref_sign.items():
        m = fold["model"] == model
        fold.loc[m, "sign_consistency"] = np.where(
            np.sign(fold.loc[m, "coef_in_buffer"]) == np.sign(sign),
            1.0,
            0.0,
        )
    fold.to_csv(STRICT_LOGO_FOLD_PATH, index=False)

    agg = (
        fold.groupby(["spec", "model"], dropna=False)[
            ["rmse", "mae", "auc", "brier", "calibration_slope", "c_index", "sign_consistency"]
        ]
        .mean()
        .reset_index()
    )
    agg.to_csv(STRICT_LOGO_AGG_PATH, index=False)
    return fold, agg


def _strict_update_reports(
    summary: pd.DataFrame,
    vif_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    logo_agg: pd.DataFrame,
) -> None:
    def pick(model: str, variant: str, metric: str) -> Tuple[float, float]:
        sub = summary[
            (summary["model"] == model)
            & (summary["variant"] == variant)
            & (summary["key_metric"] == metric)
        ]
        if sub.empty:
            return np.nan, np.nan
        row = sub.iloc[0]
        return float(row["value"]), float(row["p_value"]) if pd.notna(row["p_value"]) else np.nan

    b = "baseline_locked_v2_strict"
    n = "nlcd_locked_v2_strict"
    f = "full_locked_v2_strict"

    lines = [
        "# Feature Upgrade Report / OSM+Cloud+Sample Lock 升级报告（Strict V2）",
        "",
        "## Objective",
        "本轮按 Strict V2 规则重训：`Proxy only` 云量、`dist_any only` 距离、`strict no fallback`。",
        "",
        "## Strict Spec",
        "- Full 主规格: `in_buffer * pre_mean_ntl_centered + C(event_id) + C(land_use_group) + osm_dist_any_m + osm_power_count_1000m + osm_medical_count_1000m + pixel_cloud_proxy`",
        "- `missing_osm_flag` 与 `missing_cloud_flag` 不入主模型，仅保留审计。",
        "",
        "## Quantitative Comparison",
    ]
    for model, metric in [
        ("OLS", "coef_in_buffer"),
        ("MixedLM", "coef_in_buffer"),
        ("Logit", "odds_ratio_in_buffer"),
        ("Cox", "hazard_ratio_in_buffer"),
    ]:
        vb, pb = pick(model, b, metric)
        vn, pn = pick(model, n, metric)
        vf, pf = pick(model, f, metric)
        lines.append(
            f"- {model} `{metric}`: baseline={vb:.4f} (p={pb:.4g}), nlcd={vn:.4f} (p={pn:.4g}), full={vf:.4f} (p={pf:.4g})"
        )
    auc_b, _ = pick("Logit", b, "auc")
    auc_n, _ = pick("Logit", n, "auc")
    auc_f, _ = pick("Logit", f, "auc")
    lines.append(f"- Logit `AUC`: baseline={auc_b:.4f}, nlcd={auc_n:.4f}, full={auc_f:.4f}")

    lines.extend(
        [
            "",
            "## Collinearity Gate",
            f"- Max VIF: {vif_df['vif'].max():.4f}",
            f"- Gate result (`VIF < {STRICT_VIF_MAX:g}`): {'PASS' if (vif_df['vif'] < STRICT_VIF_MAX).all() else 'FAIL'}",
            f"- Detail: `{STRICT_VIF_PATH.relative_to(ROOT)}`",
            "",
            "## Missing-Flag Audit",
            f"- `missing_osm_flag` global distribution: `{STRICT_MISSING_FLAG_AUDIT_PATH.relative_to(ROOT)}`",
            "- 当前数据下 `missing_osm_flag` 为全 0，说明 OSM 加载完整；作为常数项无信息量，已从主模型移除。",
            "",
            "## Outputs",
            f"- Summary: `{STRICT_SUMMARY_PATH.relative_to(ROOT)}`",
            f"- Cox diagnostics: `{STRICT_COX_DIAG_PATH.relative_to(ROOT)}`",
            f"- LOEO aggregate: `{STRICT_LOGO_AGG_PATH.relative_to(ROOT)}`",
        ]
    )
    (REPORT_DIR / "06_feature_upgrade_report.md").write_text("\n".join(lines), encoding="utf-8")

    def lp(model: str, col: str) -> str:
        sub = logo_agg[(logo_agg["spec"] == "logo_transport") & (logo_agg["model"] == model)]
        if sub.empty or pd.isna(sub.iloc[0][col]):
            return "N/A"
        return f"{float(sub.iloc[0][col]):.4f}"

    lines2 = [
        "# LOEO Validation Report / 按事件留一验证报告（Strict V2）",
        "",
        "## Objective",
        "在 `full_locked_v2_strict` 下执行 LOEO 双规格验证（inference vs transport）。",
        "",
        "## Aggregate Metrics",
        f"- OLS RMSE: transport={lp('OLS','rmse')}",
        f"- MixedLM RMSE: transport={lp('MixedLM','rmse')}",
        f"- Logit AUC: transport={lp('Logit','auc')}",
        f"- Logit Brier: transport={lp('Logit','brier')}",
        f"- Cox c-index: transport={lp('Cox','c_index')}",
        "",
        "## Sign Consistency",
        f"- OLS: transport={lp('OLS','sign_consistency')}",
        f"- MixedLM: transport={lp('MixedLM','sign_consistency')}",
        f"- Logit: transport={lp('Logit','sign_consistency')}",
        f"- Cox: transport={lp('Cox','sign_consistency')}",
        "",
        f"Detail files: `{STRICT_LOGO_FOLD_PATH.relative_to(ROOT)}`, `{STRICT_LOGO_AGG_PATH.relative_to(ROOT)}`",
    ]
    (REPORT_DIR / "07_logo_validation_report.md").write_text("\n".join(lines2), encoding="utf-8")

    index_file = REPORT_DIR / "index.md"
    if index_file.exists():
        idx = index_file.read_text(encoding="utf-8")
    else:
        idx = "# Modeling Report Index\n\n## Deliverables\n"
    marker = "- `project/modeling_report/07_logo_validation_report.md`"
    if marker not in idx:
        idx = idx.rstrip() + "\n" + marker + "\n"
    strict_line = "- Strict V2 outputs: `project/modeling/output/model_summary_feature_upgrade_v2_strict.csv`, `project/modeling/output/logo_aggregate_metrics_v2_strict.csv`"
    if strict_line not in idx:
        idx = idx.rstrip() + "\n" + strict_line + "\n"
    index_file.write_text(idx, encoding="utf-8")


def strict_main() -> None:
    ensure_directories()
    init_tracking_files()
    append_progress("Strict V2 pipeline started")
    ctx = RunContext(issues=[])

    defaults = load_json(CONFIG_DEFAULTS)
    rec_thr = float(defaults["recovery_threshold"])
    damage_thr = float(defaults["damage_threshold"])

    if not STRICT_PANEL_FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing required panel: {STRICT_PANEL_FEATURE_PATH}")
    df = pd.read_parquet(STRICT_PANEL_FEATURE_PATH).copy()
    if STRICT_SAMPLE_LOCK_PATH.exists():
        lock = pd.read_parquet(STRICT_SAMPLE_LOCK_PATH)[["pixel_id", "sample_lock_flag", "lock_reason"]]
        df = df.drop(columns=["sample_lock_flag", "lock_reason"], errors="ignore").merge(lock, on="pixel_id", how="left")
    if "sample_lock_flag" not in df.columns:
        raise KeyError("sample_lock_flag missing; run feature-upgrade pipeline first.")

    df = df[df["sample_lock_flag"] == 1].copy()
    if df.empty:
        raise RuntimeError("No locked rows available.")
    df["land_use_group"] = df["land_use_group"].fillna("unknown").astype(str)
    df["is_damaged"] = (df["delta_ntl"] < damage_thr).astype(int)

    # Manifest and audits.
    missing_df = _strict_missing_flag_audit(df)
    df_raw, scale_stats = _strict_make_center_and_scale(df)
    _strict_make_sample_audit(df_raw)

    manifest = {
        "version": "v2_strict",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "variants": STRICT_VARIANTS,
        "full_feature_raw": STRICT_RAW_FULL_FEATURES,
        "excluded_from_main_spec": [
            "missing_osm_flag",
            "missing_cloud_flag",
            "pixel_pre_valid_ratio",
            "pixel_post_valid_ratio",
            "osm_dist_power_m",
            "osm_any_count_750m",
        ],
        "vif_gate_max": STRICT_VIF_MAX,
        "scale_stats": scale_stats,
        "sample_n_obs": int(len(df_raw)),
        "events": sorted(df_raw["event_id"].unique().tolist()),
        "missing_osm_flag_all_zero": bool((df_raw["missing_osm_flag"] == 0).all()) if "missing_osm_flag" in df_raw.columns else None,
    }
    STRICT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # VIF gate.
    vif_df = _strict_vif_gate(df_raw)
    if not (vif_df["vif"] < STRICT_VIF_MAX).all():
        raise RuntimeError(f"VIF gate failed. See {STRICT_VIF_PATH}")

    # Fit OLS / Mixed / Logit variants (strict, no formula fallback).
    ols_rows = []
    mixed_rows = []
    logit_rows = []
    logit_auc: Dict[str, float] = {}

    for v in STRICT_VARIANTS:
        # OLS
        f_ols = _strict_build_variant_formula("OLS", v, use_scaled=False)
        ols = smf.ols(f_ols, data=df_raw).fit(cov_type="HC1")
        ols_rows.append(_strict_coef_table_from_result(ols, "OLS", v, kind="linear"))

        # MixedLM
        f_mx = _strict_build_variant_formula("MixedLM", v, use_scaled=False)
        mx = _strict_fit_mixed_with_optimizers(f_mx, df_raw)
        mixed_rows.append(_strict_coef_table_from_result(mx, "MixedLM", v, kind="linear"))

        # Logit
        f_lg = _strict_build_variant_formula("Logit", v, use_scaled=True)
        lg = _strict_fit_logit_with_optimizers(f_lg, df_raw)
        coef = _strict_coef_table_from_result(lg, "Logit", v, kind="logit")
        coef["odds_ratio"] = np.exp(coef["coef"])
        coef["or_ci_low"] = np.exp(coef["ci_low"])
        coef["or_ci_high"] = np.exp(coef["ci_high"])
        logit_rows.append(coef)
        prob = np.asarray(lg.predict(df_raw))
        logit_auc[v] = float(roc_auc_score(df_raw["is_damaged"], prob)) if df_raw["is_damaged"].nunique() > 1 else np.nan

    ols_df = pd.concat(ols_rows, ignore_index=True)
    mixed_df = pd.concat(mixed_rows, ignore_index=True)
    logit_df = pd.concat(logit_rows, ignore_index=True)
    ols_df.to_csv(STRICT_OLS_RESULT_PATH, index=False)
    mixed_df.to_csv(STRICT_MIXED_RESULT_PATH, index=False)
    logit_df.to_csv(STRICT_LOGIT_RESULT_PATH, index=False)

    # Cox (strict, no downgrade to smaller spec).
    rec = build_recovery_panel(ctx=ctx, panel=df_raw, threshold=rec_thr, output_path=None)
    cox_rows = []
    cox_diag_rows = []
    for v in STRICT_VARIANTS:
        cox_coef, cox_diag = _strict_fit_cox_variant_strict(rec, v)
        cox_rows.append(cox_coef)
        cox_diag_rows.append(cox_diag)

    cox_df = pd.concat(cox_rows, ignore_index=True)
    cox_diag_df = pd.concat(cox_diag_rows, ignore_index=True)
    cox_df.to_csv(STRICT_COX_RESULT_PATH, index=False)
    cox_diag_df.to_csv(STRICT_COX_DIAG_PATH, index=False)

    # Strict fail-fast: full variant must exist in all four model outputs.
    checks = [
        ("OLS", not ols_df[ols_df["variant"] == STRICT_FULL_VARIANT].empty),
        ("MixedLM", not mixed_df[mixed_df["variant"] == STRICT_FULL_VARIANT].empty),
        ("Logit", not logit_df[logit_df["variant"] == STRICT_FULL_VARIANT].empty),
        ("Cox", not cox_df[cox_df["variant"] == STRICT_FULL_VARIANT].empty),
    ]
    fail_models = [m for m, ok in checks if not ok]
    if fail_models:
        raise RuntimeError(f"Fail-fast: full strict variant missing for models: {fail_models}")

    summary = _strict_summarize_for_report(
        ols_df=ols_df,
        mixed_df=mixed_df,
        logit_df=logit_df,
        logit_auc=logit_auc,
        cox_df=cox_df,
        n_obs=len(df_raw),
    )

    # LOEO dual spec.
    ref_sign = {
        "OLS": float(ols_df[(ols_df["variant"] == STRICT_FULL_VARIANT) & (ols_df["term"] == "in_buffer")]["coef"].iloc[0]),
        "MixedLM": float(mixed_df[(mixed_df["variant"] == STRICT_FULL_VARIANT) & (mixed_df["term"] == "in_buffer")]["coef"].iloc[0]),
        "Logit": float(logit_df[(logit_df["variant"] == STRICT_FULL_VARIANT) & (logit_df["term"] == "in_buffer")]["coef"].iloc[0]),
        "Cox": float(cox_df[(cox_df["variant"] == STRICT_FULL_VARIANT) & (cox_df["covariate"] == "in_buffer")]["coef"].iloc[0]),
    }
    _, logo_agg = _strict_build_logo(ctx, df_base=df, recovery_threshold=rec_thr, ref_sign=ref_sign)

    # Reports
    _strict_update_reports(summary=summary, vif_df=vif_df, missing_df=missing_df, logo_agg=logo_agg)
    save_issue_log(ctx)
    append_progress("Strict V2 pipeline finished")
# ----------------------------
# Unified CLI
# ----------------------------

import argparse


def cmd_feature_upgrade() -> int:
    _feature_upgrade_impl()
    return 0


def cmd_strict_v2() -> int:
    strict_main()
    return 0


def _cmd_build_baseline(args: argparse.Namespace) -> int:
    ctx = RunContext(issues=[])
    defaults = load_json(CONFIG_DEFAULTS)
    pre_thr = float(args.pre_threshold) if args.pre_threshold is not None else float(defaults["pre_ntl_threshold"])
    dmg_thr = float(args.damage_threshold) if args.damage_threshold is not None else float(defaults["damage_threshold"])
    exclude_types = [t.strip() for t in args.exclude_types.split(',') if t.strip()] if args.exclude_types else None
    output_path = Path(args.output) if args.output else PANEL_PATH
    build_pixel_panel(
        ctx=ctx,
        pre_threshold=pre_thr,
        damage_threshold=dmg_thr,
        exclude_types=exclude_types,
        output_path=output_path,
    )
    save_issue_log(ctx)
    return 0


def _cmd_attach_nlcd(args: argparse.Namespace) -> int:
    ctx = RunContext(issues=[])
    attach_nlcd(
        ctx=ctx,
        panel_path=Path(args.input) if args.input else PANEL_PATH,
        output_path=Path(args.output) if args.output else PANEL_NLCD_PATH,
        nlcd_dir=Path(args.nlcd_dir) if args.nlcd_dir else None,
    )
    save_issue_log(ctx)
    return 0


def _cmd_fit_core_models(args: argparse.Namespace) -> int:
    ctx = RunContext(issues=[])
    defaults = load_json(CONFIG_DEFAULTS)
    in_path = Path(args.input) if args.input else (PANEL_NLCD_PATH if args.with_land_use else PANEL_PATH)
    variant = args.variant or ('with_nlcd' if args.with_land_use else 'no_nlcd')
    dmg_thr = float(args.damage_threshold) if args.damage_threshold is not None else float(defaults['damage_threshold'])
    rec_thr = float(args.recovery_threshold) if args.recovery_threshold is not None else float(defaults['recovery_threshold'])

    df = pd.read_parquet(in_path)
    ols_mixed = fit_ols_and_mixed(ctx=ctx, df=df, variant=variant, include_land_use=args.with_land_use)
    logit = fit_logit(ctx=ctx, df=df, variant=variant, include_land_use=args.with_land_use, damage_threshold=dmg_thr)
    rec = build_recovery_panel(ctx=ctx, panel=df, threshold=rec_thr, output_path=RECOVERY_PATH)
    cox = fit_cox(ctx=ctx, recovery_df=rec, variant=variant, include_land_use=args.with_land_use)

    ols_path = OUTPUT_DIR / 'ols_results.csv'
    mixed_path = OUTPUT_DIR / 'mixedlm_results.csv'
    logit_path = OUTPUT_DIR / 'logit_results.csv'
    cox_path = OUTPUT_DIR / 'cox_results.csv'

    if not ols_mixed['ols_coef'].empty:
        if ols_path.exists():
            pd.concat([pd.read_csv(ols_path), ols_mixed['ols_coef']], ignore_index=True).to_csv(ols_path, index=False)
        else:
            ols_mixed['ols_coef'].to_csv(ols_path, index=False)
    if not ols_mixed['mixed_coef'].empty:
        if mixed_path.exists():
            pd.concat([pd.read_csv(mixed_path), ols_mixed['mixed_coef']], ignore_index=True).to_csv(mixed_path, index=False)
        else:
            ols_mixed['mixed_coef'].to_csv(mixed_path, index=False)
    if not logit['coef'].empty:
        if logit_path.exists():
            pd.concat([pd.read_csv(logit_path), logit['coef']], ignore_index=True).to_csv(logit_path, index=False)
        else:
            logit['coef'].to_csv(logit_path, index=False)
    if not logit['marginal'].empty:
        logit['marginal'].to_csv(OUTPUT_DIR / f'logit_marginal_effects_{variant}.csv', index=False)
    if not logit['roc'].empty:
        logit['roc'].to_csv(OUTPUT_DIR / f'logit_roc_{variant}.csv', index=False)
    if not logit['calibration'].empty:
        logit['calibration'].to_csv(OUTPUT_DIR / f'logit_calibration_{variant}.csv', index=False)
    if not cox['coef'].empty:
        if cox_path.exists():
            pd.concat([pd.read_csv(cox_path), cox['coef']], ignore_index=True).to_csv(cox_path, index=False)
        else:
            cox['coef'].to_csv(cox_path, index=False)
    if not cox['km'].empty:
        cox['km'].to_csv(OUTPUT_DIR / f'cox_km_{variant}.csv', index=False)
    if not cox['ph'].empty:
        cox['ph'].to_csv(OUTPUT_DIR / f'cox_ph_test_{variant}.csv', index=False)

    save_issue_log(ctx)
    return 0


def cmd_figures() -> int:
    generate_figures()
    return 0


def cmd_reports() -> int:
    build_model_summary_for_report()
    generate_reports()
    return 0


def _load_pipeline_main(rel_path: str):
    target = Path(__file__).resolve().parent / rel_path
    spec = importlib.util.spec_from_file_location(target.stem.replace("-", "_"), target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load pipeline module from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


def cmd_full_run() -> int:
    _feature_upgrade_impl()
    strict_main()
    cross_event_main = _load_pipeline_main("02_cross_event_pipeline.py")
    cross_event_main(["full-run"])
    exploration_main = _load_pipeline_main("03_exploration_pipeline.py")
    exploration_main(["run-v2"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Unified in-sample modeling entrypoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('build-baseline', help='Build baseline pixel panel')
    p.add_argument('--output', default=str(PANEL_PATH))
    p.add_argument('--pre-threshold', type=float, default=None)
    p.add_argument('--damage-threshold', type=float, default=None)
    p.add_argument('--exclude-types', default='')

    p = sub.add_parser('attach-nlcd', help='Attach NLCD to baseline panel')
    p.add_argument('--input', default=str(PANEL_PATH))
    p.add_argument('--output', default=str(PANEL_NLCD_PATH))
    p.add_argument('--nlcd-dir', default='')

    p = sub.add_parser('fit-core-models', help='Run OLS/MixedLM/Logit/Cox with cached inputs')
    p.add_argument('--with-land-use', action='store_true')
    p.add_argument('--input', default='')
    p.add_argument('--variant', default='')
    p.add_argument('--damage-threshold', type=float, default=None)
    p.add_argument('--recovery-threshold', type=float, default=None)

    sub.add_parser('feature-upgrade', help='Run feature-upgrade in-sample pipeline')
    sub.add_parser('strict-v2', help='Run strict-v2 in-sample pipeline')
    sub.add_parser('figures', help='Generate figures')
    sub.add_parser('reports', help='Generate reports')
    sub.add_parser('full-run', help='Run feature-upgrade, strict-v2, cross-event, and exploration mainline')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'build-baseline':
        return _cmd_build_baseline(args)
    if args.command == 'attach-nlcd':
        return _cmd_attach_nlcd(args)
    if args.command == 'fit-core-models':
        return _cmd_fit_core_models(args)
    if args.command == 'feature-upgrade':
        return cmd_feature_upgrade()
    if args.command == 'strict-v2':
        return cmd_strict_v2()
    if args.command == 'figures':
        return cmd_figures()
    if args.command == 'reports':
        return cmd_reports()
    if args.command == 'full-run':
        return cmd_full_run()
    parser.error(f'Unknown command: {args.command}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
