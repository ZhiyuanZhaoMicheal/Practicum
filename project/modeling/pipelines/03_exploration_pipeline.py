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
import shutil
import subprocess
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import importlib.util
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import urlopen

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import requests
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import box
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pipeline_lib as pipeline_lib_mod

from pipeline_lib import (
    BUFFER_RADII,
    CONFIG_DEFAULTS,
    CONFIG_EVENTS,
    DEFAULT_BUFFER,
    OUTPUT_DIR,
    PIXEL_DIR,
    REPORT_DIR,
    ROOT,
    FIG_DIR,
    RunContext,
    append_progress,
    ensure_directories,
    init_tracking_files,
    list_daily_tifs,
    load_json,
    save_issue_log,
    standardize_facility_type,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PANEL_IN_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_cross_event_v3.parquet"
RECOVERY_IN_PATH = PIXEL_DIR / "recovery_daily_panel_v1.parquet"
PANEL_OUT_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_exploration_v2.parquet"
EVENT_PROFILE_V1_PATH = PIXEL_DIR / "event_profile_v1.csv"

FIG_EXP_DIR = FIG_DIR / "exploration_v2"

MASTER_PLAN_PATH = ROOT / "project" / "modeling_tracking" / "future_plan" / "03_exploration_master_plan.md"
CLOUD_PLAN_PATH = ROOT / "project" / "modeling_tracking" / "future_plan" / "04_cloud_coverage_importance_plan.md"
MASK_PLAN_PATH = ROOT / "project" / "modeling_tracking" / "future_plan" / "05_noise_masking_plan.md"
URBAN_PLAN_PATH = ROOT / "project" / "modeling_tracking" / "future_plan" / "06_urban_rural_population_plan.md"
SPATIAL_PLAN_PATH = ROOT / "project" / "modeling_tracking" / "future_plan" / "07_spatial_autocorr_and_contribution_plan.md"
EXTREME_PLAN_PATH = ROOT / "project" / "modeling_tracking" / "future_plan" / "08_extreme_event_sensitivity_plan.md"

REPORT_PATH = REPORT_DIR / "10_exploration_upgrade_report.md"
INDEX_PATH = REPORT_DIR / "index.md"

# Cloud ablation outputs
CLOUD_FOLD_PATH = OUTPUT_DIR / "cloud_ablation_fold_metrics.csv"
CLOUD_AGG_PATH = OUTPUT_DIR / "cloud_ablation_aggregate_metrics.csv"
CLOUD_IMPORTANCE_PATH = OUTPUT_DIR / "cloud_feature_importance.csv"
CLOUD_COEF_PATH = OUTPUT_DIR / "cloud_ablation_coefficients.csv"

# Noise mask outputs
MASK_COVERAGE_PATH = OUTPUT_DIR / "noise_mask_coverage_by_event.csv"
MASK_METRICS_PATH = OUTPUT_DIR / "noise_mask_experiment_metrics.csv"
MASK_COEF_PATH = OUTPUT_DIR / "noise_mask_effect_on_coefficients.csv"
MASK_FULL_COEF_PATH = OUTPUT_DIR / "noise_mask_coefficients.csv"

# Urban-rural outputs
URBAN_SPLIT_PATH = OUTPUT_DIR / "urban_rural_split_summary.csv"
POP_QUALITY_PATH = OUTPUT_DIR / "pop_density_feature_quality.csv"
URBAN_MODEL_PATH = OUTPUT_DIR / "urban_rural_model_comparison.csv"
URBAN_COEF_PATH = OUTPUT_DIR / "urban_rural_coefficients.csv"

# Spatial / contribution outputs
MORAN_PATH = OUTPUT_DIR / "spatial_autocorr_morans_i.csv"
SPATIAL_SE_PATH = OUTPUT_DIR / "spatial_se_comparison.csv"
CONTRIB_SCORE_PATH = OUTPUT_DIR / "feature_contribution_scorecard.csv"
CONTRIB_RANK_PATH = OUTPUT_DIR / "feature_contribution_rank_by_model.csv"

# Extreme sensitivity outputs
EXTREME_CANDIDATE_PATH = OUTPUT_DIR / "extreme_event_candidates_v1.csv"
EXTREME_SCORE_PATH = OUTPUT_DIR / "extreme_event_scorecard_v1.csv"
EXTREME_DROP_METRIC_PATH = OUTPUT_DIR / "extreme_event_drop_metrics_v1.csv"
EXTREME_DROP_AGG_PATH = OUTPUT_DIR / "extreme_event_drop_aggregate_v1.csv"
EXTREME_DECISION_PATH = OUTPUT_DIR / "extreme_event_stop_or_keep_v1.json"

# Existing references
SHIFT_V3_PATH = OUTPUT_DIR / "cross_event_shift_diagnostics_v3.csv"
FOLD_V3R1_PATH = OUTPUT_DIR / "cross_event_fold_metrics_v3r1.csv"

# Quality / matched-design outputs
PANEL_QUALITY_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_quality_v1.parquet"
RECOVERY_V2_PATH = PIXEL_DIR / "recovery_daily_panel_v2.parquet"
FACILITY_CONTEXT_PATH = PIXEL_DIR / "facility_context_panel_v1.parquet"

TARGET_QUALITY_AUDIT_PATH = OUTPUT_DIR / "target_quality_audit.csv"
QUALITY_TRANSPORT_FOLD_PATH = OUTPUT_DIR / "quality_transport_fold_metrics_v1.csv"
QUALITY_TRANSPORT_AGG_PATH = OUTPUT_DIR / "quality_transport_aggregate_metrics_v1.csv"
SPATIAL_BLOCK_CV_PATH = OUTPUT_DIR / "spatial_block_cv_metrics_v1.csv"
FACILITY_MATCH_QUALITY_PATH = OUTPUT_DIR / "facility_match_quality.csv"
FACILITY_CENTERED_SUMMARY_PATH = OUTPUT_DIR / "facility_centered_model_summary.csv"
MODEL_ROLE_MATRIX_PATH = OUTPUT_DIR / "model_role_matrix_v1.csv"

QUALITY_REPORT_PATH = REPORT_DIR / "11_quality_matched_report.md"
QUALITY_FIG_PATH = FIG_EXP_DIR / "quality_matched_compare_v1.png"


def _pretty_stage_label(label: str) -> str:
    if label == "baseline_6":
        return "Baseline\n(6 events)"
    m = re.match(r"stage_(\d+)_(.+)", label)
    if not m:
        return label.replace("_", " ")
    stage_num = m.group(1)
    event_name = m.group(2).replace("_", " ").title()
    return f"Stage {stage_num}\n{event_name}"


def _pretty_event_label(label: str) -> str:
    label = str(label)
    if label == "baseline":
        return "Baseline"
    return label.replace("_", "\n").title()

PANEL_HAZARD_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_hazard_v1.parquet"
HAZARD_TRANSPORT_FOLD_PATH = OUTPUT_DIR / "hazard_transport_fold_metrics_v1.csv"
HAZARD_TRANSPORT_AGG_PATH = OUTPUT_DIR / "hazard_transport_aggregate_metrics_v1.csv"
HAZARD_FEATURE_SUMMARY_PATH = OUTPUT_DIR / "hazard_transport_feature_summary_v1.csv"
EVENT_SELECTION_PATH = OUTPUT_DIR / "event_selection_scorecard_v1.csv"
HAZARD_REPORT_PATH = REPORT_DIR / "12_hazard_exposure_transport_report.md"
HAZARD_FIG_PATH = FIG_EXP_DIR / "hazard_transport_compare_v1.png"
EVENT_READINESS_SCORE_PATH = OUTPUT_DIR / "event_readiness_score_v1.csv"
HAZARD_READY_FOLD_PATH = OUTPUT_DIR / "hazard_transport_readiness_fold_metrics_v1.csv"
HAZARD_READY_AGG_PATH = OUTPUT_DIR / "hazard_transport_readiness_aggregate_metrics_v1.csv"
HAZARD_READY_FEATURE_SUMMARY_PATH = OUTPUT_DIR / "hazard_transport_readiness_feature_summary_v1.csv"
HAZARD_READY_EVENTS_PATH = OUTPUT_DIR / "hazard_transport_readiness_events_v1.csv"
HAZARD_READY_REPORT_PATH = REPORT_DIR / "hazard_transport_readiness_report_v1.md"
HAZARD_READY_FIG_PATH = FIG_EXP_DIR / "hazard_transport_readiness_compare_v1.png"

BUG_TRANSPORT_PANEL_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_bug_transport_v1.parquet"
BUG_TRANSPORT_FOLD_PATH = OUTPUT_DIR / "bug_transport_fold_metrics_v1.csv"
BUG_TRANSPORT_AGG_PATH = OUTPUT_DIR / "bug_transport_aggregate_metrics_v1.csv"
BUG_TRANSPORT_FEATURE_SUMMARY_PATH = OUTPUT_DIR / "bug_transport_feature_summary_v1.csv"
BUG_TRANSPORT_FEATURE_AUDIT_PATH = OUTPUT_DIR / "bug_transport_feature_audit_v1.csv"
BUG_TRANSPORT_REPORT_PATH = REPORT_DIR / "bug_transport_report.md"
BUG_TRANSPORT_FIG_PATH = FIG_EXP_DIR / "bug_transport_compare_v1.png"

# Event-increment outputs
EVENTS10_PATH = MODELING_DIR / "config" / "events_10.json"
EVENT_INCREMENT_PLAN_PATH = MODELING_DIR / "config" / "event_increment_plan_v1.json"
BUG_PRIOR_CONFIG_PATH = MODELING_DIR / "config" / "bug_prior_lookup_v1.json"
HAZARD_MAINLINE_CANDIDATES_PATH = MODELING_DIR / "config" / "hazard_mainline_candidates_v1.json"
BUG2_PILOT_CONFIG_PATH = MODELING_DIR / "config" / "bug2_pilot_plan_v1.json"
REMOTE_REF = "teammate/main"
FIG_EVENT_INCREMENT_DIR = FIG_DIR / "event_increment"
EVENT_INCREMENT_REPORT_PATH = REPORT_DIR / "13_event_increment_report.md"

NEW_EVENT_SYNC_MANIFEST_PATH = OUTPUT_DIR / "new_event_sync_manifest_v1.csv"
NEW_EVENT_SYNC_LOG_PATH = OUTPUT_DIR / "new_event_sync_log_v1.csv"
NEW_EVENT_INPUT_GATE_PATH = OUTPUT_DIR / "new_event_input_gate_v1.csv"
NEW_EVENT_ACQ_MANIFEST_PATH = OUTPUT_DIR / "new_event_acquisition_manifest_v1.csv"
NEW_EVENT_POI_QUALITY_PATH = OUTPUT_DIR / "new_event_poi_quality_v1.csv"
COVARIATE_SOURCE_MANIFEST_PATH = OUTPUT_DIR / "covariate_source_manifest_v1.csv"
EVENT_INCREMENT_MANIFEST_PATH = OUTPUT_DIR / "event_increment_manifest_v1.csv"
EVENT_INCREMENT_METRICS_PATH = OUTPUT_DIR / "event_increment_model_metrics_v1.csv"
EVENT_INCREMENT_ISSUE_PATH = OUTPUT_DIR / "event_increment_issue_log_v1.csv"
EVENT_TYPE_GAP_PATH = OUTPUT_DIR / "event_type_gap_recommendation_v1.csv"

EVENT_INCREMENT_BOOTSTRAP_PATH = ROOT / "project" / "modeling_tracking" / "progress_record" / "04_event_increment_bootstrap.md"
EVENT_INCREMENT_ISSUE_MD_PATH = ROOT / "project" / "modeling_tracking" / "progress_record" / "05_event_increment_issue_log.md"
EVENT_INCREMENT_NEXT_PATH = ROOT / "project" / "modeling_tracking" / "future_plan" / "09_post_event_increment_next_steps.md"

FEATURE_PANEL_BASE_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_feature_upgrade.parquet"
QUALITY_PANEL_BASE_PATH = PIXEL_DIR / "all_events_pixel_panel_v1_quality_v1.parquet"
RECOVERY_V2_BASE_PATH = PIXEL_DIR / "recovery_daily_panel_v2.parquet"
STRICT_BASE_SUMMARY_PATH = OUTPUT_DIR / "model_summary_feature_upgrade_v2_strict.csv"
STRICT_BASE_LOGO_PATH = OUTPUT_DIR / "logo_aggregate_metrics_v2_strict.csv"

NEW_EVENT_ORDER = [
    "ian_fortmyers",
    "ian_charlotteharbor",
    "earthquake_hatay",
    "dorian_freeport",
]

BUG_INVENTORY_DIR = ROOT / "project" / "data" / "external" / "bug_inventory"
BUG_INVENTORY_RAW_DIR = BUG_INVENTORY_DIR / "raw"
BUG_INVENTORY_CANONICAL_DIR = BUG_INVENTORY_DIR / "canonical"
BUG2_ACQ_BACKLOG_PATH = OUTPUT_DIR / "bug2_pilot_acquisition_backlog_v1.csv"
BUG2_PR_CANONICAL_PATH = BUG_INVENTORY_CANONICAL_DIR / "bug_inventory_pr_pilot_v1.csv"
BUG2_PR_CANONICAL_TEMPLATE_PATH = BUG_INVENTORY_CANONICAL_DIR / "bug_inventory_pr_pilot_v1_template.csv"
BUG2_QA_PATH = OUTPUT_DIR / "bug2_pr_pilot_qa_v1.csv"
BUG2_FEATURE_AUDIT_PATH = OUTPUT_DIR / "bug2_pr_feature_audit_v1.csv"
BUG2_FOLD_PATH = OUTPUT_DIR / "bug2_pr_pilot_fold_metrics_v1.csv"
BUG2_AGG_PATH = OUTPUT_DIR / "bug2_pr_pilot_aggregate_metrics_v1.csv"
BUG2_FEATURE_SUMMARY_PATH = OUTPUT_DIR / "bug2_pr_pilot_feature_summary_v1.csv"
BUG2_REPORT_PATH = REPORT_DIR / "bug2_pr_pilot_report.md"
BUG2_FIG_PATH = FIG_EXP_DIR / "bug2_pr_pilot_compare_v1.png"

SYNC_MANIFEST_COLS = ["stage", "remote_path", "local_path", "event_id", "file_type", "exists_before", "action", "status", "size_bytes", "source_commit"]
INPUT_GATE_COLS = ["event_id", "has_pre_dir", "has_post_dir", "pre_tif_n", "post_tif_n", "has_cloud_csv", "has_poi_csv_before", "gate_status", "notes"]
ACQ_MANIFEST_COLS = ["event_id", "indicator_type", "source_name", "source_priority", "request_status", "download_status", "local_output_path", "coverage_metric", "quality_flag", "notes"]
POI_QUALITY_COLS = ["event_id", "poi_source", "poi_count", "lat_valid_share", "lon_valid_share", "type_missing_share", "facility_type_unique_n", "quality_flag"]
COV_SOURCE_COLS = ["event_id", "covariate_name", "source_name", "source_type", "spatial_resolution", "temporal_reference", "is_us_only", "used_in_mainline", "quality_flag"]
EVENT_INCREMENT_ISSUE_COLS = ["stage_id", "event_id", "bundle", "issue_type", "symptom", "fix_action", "impact", "status"]
EVENT_INCREMENT_METRIC_COLS = ["stage_id", "event_count", "new_event_id", "bundle", "model", "metric_name", "value", "delta_vs_prev", "delta_vs_baseline", "status", "notes"]

US_EVENT_TO_STATE = {
    "maria_sanjuan": "72",
    "earthquake_sanjuan": "72",
    "ida_neworleans": "22",
    "laura_lakecharles": "22",
    "michael_panamacity": "12",
    "irma_miami": "12",
    "ian_fortmyers": "12",
    "ian_charlotteharbor": "12",
}

# Added fields required by plan
ADDED_FIELDS = [
    "noise_mask_group",
    "is_cbsa",
    "is_urban_area",
    "urban_rural_stratum",
    "pop_density_per_km2",
    "pop_density_log1p",
    "missing_pop_flag",
]

BASE_NUMERIC = [
    "osm_dist_any_m",
    "osm_power_count_1000m",
    "osm_medical_count_1000m",
    "urban_share_1km",
    "water_share_1km",
    "developed_high_share_1km",
]

BUG_PRIOR_NUMERIC = [
    "bug_prior_count_750m",
    "bug_prior_count_1250m",
    "bug_prior_capacity_proxy_1km",
    "bug_prior_hours_proxy_1km",
    "bug_prior_min_dist_m",
]

QUALITY_GUARD_NUMERIC = [
    "pixel_cloud_proxy",
    "pixel_pre_valid_ratio",
    "pixel_post_valid_ratio",
    "recovery_obs_quality_score",
    "high_censoring_risk_flag",
]

CLOUD_VARIANTS = {
    "C0": [],
    "C1": ["pixel_cloud_proxy"],
    "C2": ["pixel_cloud_proxy", "pixel_pre_valid_ratio", "pixel_post_valid_ratio"],
    "C3": ["pixel_cloud_proxy", "pixel_pre_valid_ratio", "pixel_post_valid_ratio", "missing_cloud_flag"],
}

MASK_CLASSES_M1 = {11, 12, 90, 95}
MASK_CLASSES_M2_EXTRA = {31, 41, 42, 43, 52, 71}

MODEL_ORDER = ["OLS", "MixedLM", "Logit", "Cox", "AFT"]

EVENT_TO_STATE = {
    "maria_sanjuan": "72",
    "earthquake_sanjuan": "72",
    "ida_neworleans": "22",
    "laura_lakecharles": "22",
    "michael_panamacity": "12",
    "irma_miami": "12",
}


@dataclass
class SpecResult:
    fold_df: pd.DataFrame
    agg_df: pd.DataFrame
    coef_df: pd.DataFrame


def _load_dynamic_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get_pipeline_module(tag: str, file_name: str):
    module_name = f"_event_increment_{tag}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    return _load_dynamic_module(module_name, MODELING_DIR / "pipelines" / file_name)


@contextmanager
def _override_attrs(target, mapping: Dict[str, object]):
    old = {}
    for key, value in mapping.items():
        old[key] = getattr(target, key)
        setattr(target, key, value)
    try:
        yield
    finally:
        for key, value in old.items():
            setattr(target, key, value)


@contextmanager
def _override_globals(mapping: Dict[str, object]):
    g = globals()
    old = {key: g[key] for key in mapping}
    g.update(mapping)
    try:
        yield
    finally:
        g.update(old)


def _stage_suffix_path(path: Path, stage_tag: str) -> Path:
    if not stage_tag:
        return path
    return path.with_name(f"{path.stem}_{stage_tag}{path.suffix}")


def _cloud_csv_for_event(event_id: str) -> Path:
    return ROOT / "project" / "script" / f"{event_id}_cloud_screening.csv"


def _stage_paths(stage_tag: str) -> Dict[str, Path]:
    return {
        "feature_panel": _stage_suffix_path(FEATURE_PANEL_BASE_PATH, stage_tag),
        "sample_lock": _stage_suffix_path(PIXEL_DIR / "sample_lock_cohort_v1.parquet", stage_tag),
        "quality_panel": _stage_suffix_path(PANEL_QUALITY_PATH, stage_tag),
        "recovery_v2": _stage_suffix_path(RECOVERY_V2_PATH, stage_tag),
        "target_audit": _stage_suffix_path(TARGET_QUALITY_AUDIT_PATH, stage_tag),
        "quality_fold": _stage_suffix_path(QUALITY_TRANSPORT_FOLD_PATH, stage_tag),
        "quality_agg": _stage_suffix_path(QUALITY_TRANSPORT_AGG_PATH, stage_tag),
        "spatial_block": _stage_suffix_path(SPATIAL_BLOCK_CV_PATH, stage_tag),
        "facility_panel": _stage_suffix_path(FACILITY_CONTEXT_PATH, stage_tag),
        "facility_quality": _stage_suffix_path(FACILITY_MATCH_QUALITY_PATH, stage_tag),
        "facility_summary": _stage_suffix_path(FACILITY_CENTERED_SUMMARY_PATH, stage_tag),
        "role_matrix": _stage_suffix_path(MODEL_ROLE_MATRIX_PATH, stage_tag),
        "hazard_panel": _stage_suffix_path(PANEL_HAZARD_PATH, stage_tag),
        "hazard_fold": _stage_suffix_path(HAZARD_TRANSPORT_FOLD_PATH, stage_tag),
        "hazard_agg": _stage_suffix_path(HAZARD_TRANSPORT_AGG_PATH, stage_tag),
        "hazard_feature": _stage_suffix_path(HAZARD_FEATURE_SUMMARY_PATH, stage_tag),
        "event_selection": _stage_suffix_path(EVENT_SELECTION_PATH, stage_tag),
        "event_profile": _stage_suffix_path(EVENT_PROFILE_V1_PATH, stage_tag),
        "strict_summary": _stage_suffix_path(STRICT_BASE_SUMMARY_PATH, stage_tag),
        "strict_logo": _stage_suffix_path(STRICT_BASE_LOGO_PATH, stage_tag),
        "strict_manifest": _stage_suffix_path(OUTPUT_DIR / "feature_spec_manifest_v2_strict.json", stage_tag),
        "strict_vif": _stage_suffix_path(OUTPUT_DIR / "multicollinearity_vif_v2_strict.csv", stage_tag),
        "strict_sample_audit": _stage_suffix_path(OUTPUT_DIR / "sample_alignment_audit_v2_strict.csv", stage_tag),
        "strict_cox_diag": _stage_suffix_path(OUTPUT_DIR / "cox_diagnostics_extended_v2_strict.csv", stage_tag),
        "strict_logo_fold": _stage_suffix_path(OUTPUT_DIR / "logo_fold_metrics_v2_strict.csv", stage_tag),
        "strict_missing_audit": _stage_suffix_path(OUTPUT_DIR / "missing_flag_audit_v2_strict.csv", stage_tag),
        "strict_ols": _stage_suffix_path(OUTPUT_DIR / "ols_results_feature_upgrade_v2_strict.csv", stage_tag),
        "strict_mixed": _stage_suffix_path(OUTPUT_DIR / "mixedlm_results_feature_upgrade_v2_strict.csv", stage_tag),
        "strict_logit": _stage_suffix_path(OUTPUT_DIR / "logit_results_feature_upgrade_v2_strict.csv", stage_tag),
        "strict_cox": _stage_suffix_path(OUTPUT_DIR / "cox_results_feature_upgrade_v2_strict.csv", stage_tag),
    }


def _run_git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def _list_remote_paths(prefix: str) -> List[str]:
    out = _run_git("ls-tree", "-r", "--name-only", REMOTE_REF, "--", prefix)
    return [line.strip() for line in out.splitlines() if line.strip()]


def _export_remote_path(remote_path: str, local_path: Path) -> int:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as fh:
        proc = subprocess.run(
            ["git", "show", f"{REMOTE_REF}:{remote_path}"],
            cwd=ROOT,
            check=True,
            stdout=fh,
            stderr=subprocess.PIPE,
        )
    return proc.returncode


def _metric_value(df: pd.DataFrame, model: str, col: str) -> float:
    if df.empty or model not in df["model"].astype(str).values or col not in df.columns:
        return float("nan")
    sub = df[df["model"].astype(str) == model]
    return float(pd.to_numeric(sub[col], errors="coerce").mean())


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def _read_csv_or_empty(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(columns))
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(columns))


def _event_bbox(cfg: Dict[str, object]) -> Tuple[float, float, float, float]:
    vals = cfg.get("bounds")
    if not isinstance(vals, list) or len(vals) != 4:
        raise KeyError(f"Event config missing bounds: {cfg.get('event_name', '')}")
    west, south, east, north = [float(v) for v in vals]
    return west, south, east, north


def _parse_date_from_name_local(path: Path) -> Optional[pd.Timestamp]:
    m = re.search(r"(\\d{4}-\\d{2}-\\d{2})", path.name)
    if not m:
        return None
    return pd.Timestamp(m.group(1))


def _safe_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if out.notna().any():
        return out.fillna(out.median())
    return out.fillna(default)


def _capacity_band_scalar(band: str) -> float:
    mapping = {
        "low": 0.75,
        "medium": 1.5,
        "high": 3.0,
        "very_high": 4.0,
        "critical": 4.5,
    }
    return float(mapping.get(str(band).strip().lower(), 1.0))


def _load_bug_prior_lookup() -> pd.DataFrame:
    cfg = load_json(BUG_PRIOR_CONFIG_PATH)
    default = cfg.get("default", {}) if isinstance(cfg, dict) else {}
    facility_types = {}
    if isinstance(cfg, dict):
        if isinstance(cfg.get("facility_types"), dict):
            facility_types = cfg.get("facility_types", {})
        else:
            facility_types = {k: v for k, v in cfg.items() if k != "default" and isinstance(v, dict)}
    rows: List[Dict[str, object]] = []
    for facility_type, payload in facility_types.items():
        row = dict(default)
        if isinstance(payload, dict):
            row.update(payload)
        row["facility_type_std"] = str(facility_type)
        rows.append(row)
    lookup = pd.DataFrame(rows)
    if lookup.empty:
        raise RuntimeError(f"BUG prior lookup is empty: {BUG_PRIOR_CONFIG_PATH}")
    for c in ["bug_propensity_weight", "night_use_weight", "detectability_weight"]:
        lookup[c] = _safe_numeric(lookup[c], default=float(default.get(c, 0.25)))
    lookup["capacity_weight"] = _safe_numeric(lookup.get("capacity_weight", pd.Series(dtype=float)), default=1.0)
    lookup["capacity_prior_band"] = lookup["capacity_prior_band"].fillna(default.get("capacity_prior_band", "low")).astype(str)
    return lookup


def _bug2_required_columns() -> List[str]:
    return [
        "source_dataset",
        "jurisdiction",
        "state",
        "county_or_district",
        "record_id",
        "facility_name",
        "facility_type_raw",
        "facility_type_std",
        "fuel_type",
        "capacity_kw",
        "operating_hours_annual",
        "address_raw",
        "lat",
        "lon",
        "geo_quality_flag",
        "attribute_quality_flag",
        "source_url",
    ]


def _load_hazard_mainline_candidates() -> List[str]:
    if HAZARD_MAINLINE_CANDIDATES_PATH.exists():
        cfg = load_json(HAZARD_MAINLINE_CANDIDATES_PATH)
        if isinstance(cfg, dict):
            events = cfg.get("event_ids", [])
            if isinstance(events, list) and events:
                return [str(v) for v in events]
    if EVENT_READINESS_SCORE_PATH.exists():
        ready = pd.read_csv(EVENT_READINESS_SCORE_PATH)
        if {"event_id", "readiness_band"}.issubset(ready.columns):
            allow = ready.loc[ready["readiness_band"].astype(str) == "mainline_ready", "event_id"].astype(str).tolist()
            if allow:
                return allow
    return [
        "ian_charlotteharbor",
        "earthquake_sanjuan",
        "ida_neworleans",
        "irma_miami",
        "laura_lakecharles",
    ]


def _load_bug2_pilot_config() -> Dict[str, object]:
    cfg = load_json(BUG2_PILOT_CONFIG_PATH) if BUG2_PILOT_CONFIG_PATH.exists() else {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg.setdefault("pilot_state", "PR")
    cfg.setdefault("pilot_label", "Puerto Rico official BUG pilot")
    cfg.setdefault("pilot_event_ids", ["earthquake_sanjuan", "maria_sanjuan"])
    cfg.setdefault("source_dataset", "pr_official_bug_pilot")
    cfg.setdefault("jurisdiction", "Puerto Rico")
    cfg.setdefault("county_or_district", "San Juan")
    cfg.setdefault("source_url", "")
    cfg.setdefault("source_path", _safe_rel(BUG_INVENTORY_RAW_DIR / "bug_inventory_pr_pilot_raw.csv"))
    cfg.setdefault("canonical_path", _safe_rel(BUG2_PR_CANONICAL_PATH))
    cfg.setdefault("coverage_gate", {"min_records": 25, "min_geo_coverage": 0.75, "min_event_hits": 2})
    cfg.setdefault("canonical_columns", _bug2_required_columns())
    return cfg


def _zscore(x: pd.Series) -> pd.Series:
    v = _safe_numeric(x, default=0.0)
    std = float(v.std(ddof=0))
    if std == 0 or not np.isfinite(std):
        return pd.Series(np.zeros(len(v)), index=v.index)
    return (v - float(v.mean())) / std


def _fetch_json(url: str, timeout: int = 30):
    with urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _land_use_to_noise_group(v: float) -> str:
    if pd.isna(v):
        return "kept"
    try:
        code = int(v)
    except Exception:
        return "kept"
    if code in MASK_CLASSES_M1:
        return "masked_m1"
    if code in MASK_CLASSES_M2_EXTRA:
        return "masked_m2"
    return "kept"


def _build_formula(target: str, numeric_terms: Sequence[str], cat_terms: Sequence[str]) -> str:
    rhs = ["in_buffer * pre_mean_ntl"]
    rhs.extend(list(numeric_terms))
    rhs.extend([f"C({c})" for c in cat_terms])
    return f"{target} ~ " + " + ".join(rhs)


def _filter_sample_lock(df: pd.DataFrame) -> pd.DataFrame:
    if "sample_lock_flag" not in df.columns:
        return df.copy()
    flags = pd.to_numeric(df["sample_lock_flag"], errors="coerce").fillna(1).astype(int)
    return df.loc[flags == 1].copy()


def _prepare_columns(
    df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
    required_numeric: Sequence[str] = (),
) -> pd.DataFrame:
    out = df.copy()
    base_num = ["delta_ntl", "pre_mean_ntl", "in_buffer", "is_damaged", "recovery_days", "event_observed"]
    required = {"pre_mean_ntl", "in_buffer"} | set(required_numeric)

    for c in base_num + list(numeric_terms):
        if c in out.columns:
            out[c] = _safe_numeric(out[c])
        elif c in required:
            raise KeyError(f"Missing required numeric column: {c}")
        else:
            out[c] = 0.0

    for c in cat_terms:
        if c not in out.columns:
            out[c] = "unknown"
        out[c] = out[c].fillna("unknown").astype(str)

    out["in_buffer_x_pre"] = out["in_buffer"] * out["pre_mean_ntl"]
    return out


def _prune_terms(
    train_df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
) -> Tuple[List[str], List[str]]:
    kept_num: List[str] = []
    kept_cat: List[str] = []

    for c in numeric_terms:
        if c not in train_df.columns:
            continue
        vals = _safe_numeric(train_df[c])
        if vals.nunique(dropna=True) > 1:
            kept_num.append(c)

    for c in cat_terms:
        if c not in train_df.columns:
            continue
        nuniq = train_df[c].fillna("unknown").astype(str).nunique(dropna=True)
        if nuniq > 1:
            kept_cat.append(c)

    return kept_num, kept_cat


def _build_logit_design(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    base_cols = ["in_buffer", "pre_mean_ntl", "in_buffer_x_pre"]
    num_cols = base_cols + list(numeric_terms)

    tr = train_df.copy()
    te = test_df.copy()
    for c in num_cols:
        if c not in tr.columns:
            tr[c] = 0.0
            te[c] = 0.0
        tr[c] = _safe_numeric(tr[c])
        med = float(tr[c].median()) if tr[c].notna().any() else 0.0
        te[c] = _safe_numeric(te[c]).fillna(med)

    tr_num = tr[num_cols].copy()
    te_num = te[num_cols].copy()

    dtr = pd.get_dummies(tr[cat_terms], prefix=cat_terms, drop_first=True) if cat_terms else pd.DataFrame(index=tr.index)
    dte = pd.get_dummies(te[cat_terms], prefix=cat_terms, drop_first=True) if cat_terms else pd.DataFrame(index=te.index)
    dte = dte.reindex(columns=dtr.columns, fill_value=0)

    xtr = pd.concat([tr_num.reset_index(drop=True), dtr.reset_index(drop=True)], axis=1)
    xte = pd.concat([te_num.reset_index(drop=True), dte.reset_index(drop=True)], axis=1)
    return xtr, xte, xtr.columns.tolist()


def _build_survival_design(
    df: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
) -> pd.DataFrame:
    keep = ["recovery_days", "event_observed", "event_id", "in_buffer", "pre_mean_ntl", "in_buffer_x_pre"]
    keep += list(numeric_terms)
    keep += list(cat_terms)
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    for c in ["recovery_days", "event_observed", "in_buffer", "pre_mean_ntl", "in_buffer_x_pre"] + list(numeric_terms):
        if c in out.columns:
            out[c] = _safe_numeric(out[c])

    dummy_parts = []
    for c in cat_terms:
        if c in out.columns:
            dummy_parts.append(pd.get_dummies(out[c].fillna("unknown").astype(str), prefix=c, drop_first=True))
            out = out.drop(columns=[c])
    if dummy_parts:
        out = pd.concat([out.reset_index(drop=True)] + [d.reset_index(drop=True) for d in dummy_parts], axis=1)

    out = out.replace([np.inf, -np.inf], np.nan)
    for c in out.columns:
        if c == "event_id":
            continue
        out[c] = _safe_numeric(out[c])
    return out


def _evaluate_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_rec: pd.DataFrame,
    test_rec: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
    experiment_family: str,
    spec_id: str,
    fold_event: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []

    num_terms, cats = _prune_terms(train_df, numeric_terms, cat_terms)
    formula = _build_formula("delta_ntl", num_terms, cats)

    # OLS
    try:
        ols = smf.ols(formula, data=train_df).fit(cov_type="HC1")
        pred = ols.predict(test_df)
        rmse = float(np.sqrt(mean_squared_error(test_df["delta_ntl"], pred)))
        mae = float(mean_absolute_error(test_df["delta_ntl"], pred))
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "OLS",
                "rmse": rmse,
                "mae": mae,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": float(ols.params.get("in_buffer", np.nan)),
                "notes": "ok",
            }
        )
        for term, val in ols.params.items():
            coef_rows.append(
                {
                    "experiment_family": experiment_family,
                    "spec_id": spec_id,
                    "fold_event": fold_event,
                    "model": "OLS",
                    "feature": term,
                    "coef": float(val),
                    "p_value": float(ols.pvalues.get(term, np.nan)),
                }
            )
    except Exception as e:
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "OLS",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": np.nan,
                "notes": f"fail:{type(e).__name__}",
            }
        )

    # MixedLM
    try:
        mixed = smf.mixedlm(formula, data=train_df, groups=train_df["event_id"]).fit(reml=False, method="lbfgs", maxiter=200)
        pred = mixed.predict(test_df)
        rmse = float(np.sqrt(mean_squared_error(test_df["delta_ntl"], pred)))
        mae = float(mean_absolute_error(test_df["delta_ntl"], pred))
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "MixedLM",
                "rmse": rmse,
                "mae": mae,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": float(mixed.params.get("in_buffer", np.nan)),
                "notes": "ok",
            }
        )
        for term, val in mixed.params.items():
            if term == "Group Var":
                continue
            coef_rows.append(
                {
                    "experiment_family": experiment_family,
                    "spec_id": spec_id,
                    "fold_event": fold_event,
                    "model": "MixedLM",
                    "feature": term,
                    "coef": float(val),
                    "p_value": float(mixed.pvalues.get(term, np.nan)),
                }
            )
    except Exception as e:
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "MixedLM",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": np.nan,
                "notes": f"fail:{type(e).__name__}",
            }
        )

    # Logit
    try:
        xtr, xte, feat_names = _build_logit_design(train_df, test_df, num_terms, cats)
        ytr = _safe_numeric(train_df["is_damaged"]).astype(int)
        yte = _safe_numeric(test_df["is_damaged"]).astype(int)

        logit = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", class_weight="balanced")
        logit.fit(xtr, ytr)

        prob = logit.predict_proba(xte)[:, 1]
        auc = float(roc_auc_score(yte, prob)) if yte.nunique() > 1 else np.nan
        brier = float(brier_score_loss(yte, prob))

        coef_in_buffer = np.nan
        if "in_buffer" in feat_names:
            coef_in_buffer = float(logit.coef_[0, feat_names.index("in_buffer")])

        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "Logit",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": auc,
                "brier": brier,
                "c_index": np.nan,
                "coef_in_buffer": coef_in_buffer,
                "notes": "ok",
            }
        )

        for idx, name in enumerate(feat_names):
            coef_rows.append(
                {
                    "experiment_family": experiment_family,
                    "spec_id": spec_id,
                    "fold_event": fold_event,
                    "model": "Logit",
                    "feature": name,
                    "coef": float(logit.coef_[0, idx]),
                    "p_value": np.nan,
                }
            )
    except Exception as e:
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "Logit",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": np.nan,
                "notes": f"fail:{type(e).__name__}",
            }
        )

    # Cox / AFT
    try:
        surv_num, surv_cat = _prune_terms(train_rec, num_terms, cats)
        s_tr = _build_survival_design(train_rec, surv_num, surv_cat)
        s_te = _build_survival_design(test_rec, surv_num, surv_cat)

        # Align columns
        target_cols = [c for c in s_tr.columns if c not in {"recovery_days", "event_observed", "event_id"}]
        for c in target_cols:
            if c not in s_te.columns:
                s_te[c] = 0.0
        s_te = s_te.reindex(columns=s_tr.columns, fill_value=0.0)

        cox = CoxPHFitter(penalizer=0.01)
        cox.fit(s_tr.drop(columns=["event_id"], errors="ignore"), duration_col="recovery_days", event_col="event_observed")
        risk = cox.predict_partial_hazard(s_te.drop(columns=["recovery_days", "event_observed", "event_id"], errors="ignore"))
        c_idx = float(
            concordance_index(
                s_te["recovery_days"],
                -risk.to_numpy().reshape(-1),
                s_te["event_observed"],
            )
        )
        coef_in = float(cox.params_.get("in_buffer", np.nan))
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "Cox",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": c_idx,
                "coef_in_buffer": coef_in,
                "notes": "ok",
            }
        )
        for term, val in cox.params_.items():
            coef_rows.append(
                {
                    "experiment_family": experiment_family,
                    "spec_id": spec_id,
                    "fold_event": fold_event,
                    "model": "Cox",
                    "feature": term,
                    "coef": float(val),
                    "p_value": float(cox.summary.loc[term, "p"]) if term in cox.summary.index else np.nan,
                }
            )

        aft = WeibullAFTFitter(penalizer=0.01)
        aft.fit(s_tr.drop(columns=["event_id"], errors="ignore"), duration_col="recovery_days", event_col="event_observed")
        med = aft.predict_median(s_te.drop(columns=["recovery_days", "event_observed", "event_id"], errors="ignore"))
        c_idx_aft = float(
            concordance_index(
                s_te["recovery_days"],
                -med.to_numpy().reshape(-1),
                s_te["event_observed"],
            )
        )

        cands = [i for i in aft.params_.index if i[0] == "lambda_" and i[1] == "in_buffer"]
        coef_in_aft = float(aft.params_.loc[cands[0]]) if cands else np.nan
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "AFT",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": c_idx_aft,
                "coef_in_buffer": coef_in_aft,
                "notes": "ok",
            }
        )
    except Exception as e:
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "Cox",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": np.nan,
                "notes": f"fail:{type(e).__name__}",
            }
        )
        rows.append(
            {
                "experiment_family": experiment_family,
                "spec_id": spec_id,
                "fold_event": fold_event,
                "model": "AFT",
                "rmse": np.nan,
                "mae": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "c_index": np.nan,
                "coef_in_buffer": np.nan,
                "notes": f"fail:{type(e).__name__}",
            }
        )

    return rows, coef_rows


def _aggregate_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for (fam, spec, model), grp in fold_df.groupby(["experiment_family", "spec_id", "model"], dropna=False):
        out_rows.append(
            {
                "experiment_family": fam,
                "spec_id": spec,
                "model": model,
                "rmse": float(grp["rmse"].mean(skipna=True)),
                "mae": float(grp["mae"].mean(skipna=True)),
                "auc": float(grp["auc"].mean(skipna=True)),
                "brier": float(grp["brier"].mean(skipna=True)),
                "c_index": float(grp["c_index"].mean(skipna=True)),
                "coef_in_buffer": float(grp["coef_in_buffer"].mean(skipna=True)),
                "n_folds": int(grp["fold_event"].nunique()),
            }
        )
    return pd.DataFrame(out_rows)


def run_loeo_spec(
    panel: pd.DataFrame,
    recovery: pd.DataFrame,
    numeric_terms: Sequence[str],
    cat_terms: Sequence[str],
    experiment_family: str,
    spec_id: str,
    allowed_events: Optional[Sequence[str]] = None,
) -> SpecResult:
    df = panel.copy()
    rec = recovery.copy()

    if allowed_events is not None:
        allowed = set(allowed_events)
        df = df[df["event_id"].isin(allowed)].copy()
        rec = rec[rec["event_id"].isin(allowed)].copy()

    events = sorted(df["event_id"].dropna().unique().tolist())
    fold_rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []

    for fold_event in events:
        tr = df[df["event_id"] != fold_event].copy()
        te = df[df["event_id"] == fold_event].copy()
        tr_rec = rec[rec["event_id"] != fold_event].copy()
        te_rec = rec[rec["event_id"] == fold_event].copy()

        tr = _prepare_columns(tr, numeric_terms, cat_terms, required_numeric=("delta_ntl", "is_damaged"))
        te = _prepare_columns(te, numeric_terms, cat_terms, required_numeric=("delta_ntl", "is_damaged"))
        tr_rec = _prepare_columns(tr_rec, numeric_terms, cat_terms, required_numeric=("recovery_days", "event_observed"))
        te_rec = _prepare_columns(te_rec, numeric_terms, cat_terms, required_numeric=("recovery_days", "event_observed"))

        rows, coefs = _evaluate_fold(
            tr,
            te,
            tr_rec,
            te_rec,
            numeric_terms=numeric_terms,
            cat_terms=cat_terms,
            experiment_family=experiment_family,
            spec_id=spec_id,
            fold_event=fold_event,
        )
        fold_rows.extend(rows)
        coef_rows.extend(coefs)

    fold_df = pd.DataFrame(fold_rows)
    agg_df = _aggregate_metrics(fold_df)
    coef_df = pd.DataFrame(coef_rows)
    return SpecResult(fold_df=fold_df, agg_df=agg_df, coef_df=coef_df)


def _prepare_noise_groups(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["noise_mask_group"] = out["land_use"].apply(_land_use_to_noise_group)
    return out


def _read_first_geo(urls: Sequence[str]) -> gpd.GeoDataFrame:
    errs: List[str] = []
    for u in urls:
        try:
            return gpd.read_file(u).to_crs("EPSG:4326")
        except Exception as e:
            errs.append(f"{u}:{type(e).__name__}")
    raise RuntimeError("all_urls_failed:" + ";".join(errs))


def _download_cbsa_ua() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    cbsa = _read_first_geo(
        [
            "https://www2.census.gov/geo/tiger/TIGER2023/CBSA/tl_2023_us_cbsa.zip",
            "https://www2.census.gov/geo/tiger/TIGER2024/CBSA/tl_2024_us_cbsa.zip",
            "https://www2.census.gov/geo/tiger/TIGER2021/CBSA/tl_2021_us_cbsa.zip",
        ]
    )
    ua = _read_first_geo(
        [
            "https://www2.census.gov/geo/tiger/TIGER2023/UAC20/tl_2023_us_uac20.zip",
            "https://www2.census.gov/geo/tiger/TIGER2022/UAC/tl_2022_us_uac10.zip",
        ]
    )

    cbsa_id_col = "GEOID" if "GEOID" in cbsa.columns else cbsa.columns[0]
    ua_id_col = "GEOID20" if "GEOID20" in ua.columns else ("GEOID10" if "GEOID10" in ua.columns else ua.columns[0])
    return cbsa[[cbsa_id_col, "geometry"]].rename(columns={cbsa_id_col: "GEOID"}), ua[[ua_id_col, "geometry"]].rename(
        columns={ua_id_col: "GEOID20"}
    )


def _download_tracts_for_states(states: Iterable[str]) -> gpd.GeoDataFrame:
    pieces = []
    for st in sorted(set(states)):
        url = f"https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_{st}_tract.zip"
        g = gpd.read_file(url).to_crs("EPSG:4326")
        pieces.append(g[["STATEFP", "COUNTYFP", "TRACTCE", "GEOID", "ALAND", "geometry"]])
    if not pieces:
        return gpd.GeoDataFrame(columns=["STATEFP", "COUNTYFP", "TRACTCE", "GEOID", "ALAND", "geometry"], geometry="geometry", crs="EPSG:4326")
    return pd.concat(pieces, ignore_index=True)


def _fetch_acs_population(states: Iterable[str]) -> pd.DataFrame:
    rows = []
    for st in sorted(set(states)):
        url = f"https://api.census.gov/data/2022/acs/acs5?get=B01003_001E&for=tract:*&in=state:{st}&in=county:*"
        data = _fetch_json(url)
        if not isinstance(data, list) or len(data) <= 1:
            continue
        header = data[0]
        for rec in data[1:]:
            item = dict(zip(header, rec))
            geoid = f"{item.get('state','')}{item.get('county','')}{item.get('tract','')}"
            pop = pd.to_numeric(item.get("B01003_001E"), errors="coerce")
            rows.append({"GEOID": geoid, "population": pop})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["GEOID"], keep="last")
    return out


def attach_urban_population(panel: pd.DataFrame, ctx: RunContext) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    out["is_cbsa"] = 0
    out["is_urban_area"] = 0
    out["urban_rural_stratum"] = "rural"
    out["pop_density_per_km2"] = np.nan
    out["pop_density_log1p"] = np.nan
    out["missing_pop_flag"] = 1

    quality_rows = []

    gpts = gpd.GeoDataFrame(
        out[["pixel_id", "event_id", "lon", "lat"]].copy(),
        geometry=gpd.points_from_xy(out["lon"], out["lat"]),
        crs="EPSG:4326",
    )

    # CBSA + UA
    try:
        cbsa, ua = _download_cbsa_ua()
        cbsa_join = gpd.sjoin(gpts[["pixel_id", "geometry"]], cbsa, how="left", predicate="within")
        ua_join = gpd.sjoin(gpts[["pixel_id", "geometry"]], ua, how="left", predicate="within")

        cbsa_ids = set(cbsa_join.loc[cbsa_join["GEOID"].notna(), "pixel_id"].astype(str))
        ua_ids = set(ua_join.loc[ua_join["GEOID20"].notna(), "pixel_id"].astype(str))

        out.loc[out["pixel_id"].astype(str).isin(cbsa_ids), "is_cbsa"] = 1
        out.loc[out["pixel_id"].astype(str).isin(ua_ids), "is_urban_area"] = 1
        quality_rows.append(
            {
                "feature": "cbsa_urban_join",
                "status": "ok",
                "coverage_ratio": float((out["is_cbsa"] == 1).mean()),
                "notes": "Census TIGER 2022",
            }
        )
    except Exception as e:
        quality_rows.append(
            {
                "feature": "cbsa_urban_join",
                "status": f"fallback:{type(e).__name__}",
                "coverage_ratio": 0.0,
                "notes": "set is_cbsa/is_urban_area default 0",
            }
        )

    out["urban_rural_stratum"] = np.where(
        out["is_urban_area"] == 1,
        "urban",
        np.where(out["is_cbsa"] == 1, "suburban", "rural"),
    )

    # ACS tract population
    try:
        state_list = sorted({EVENT_TO_STATE.get(e) for e in out["event_id"].dropna().unique().tolist() if EVENT_TO_STATE.get(e)})
        tracts = _download_tracts_for_states(state_list)
        tracts = tracts.copy()
        tracts["area_km2"] = _safe_numeric(tracts["ALAND"]) / 1_000_000.0

        join = gpd.sjoin(gpts[["pixel_id", "event_id", "geometry"]], tracts[["GEOID", "area_km2", "geometry"]], how="left", predicate="within")
        pop = _fetch_acs_population(state_list)
        if pop.empty:
            raise RuntimeError("ACS population table is empty")

        join = join.merge(pop, on="GEOID", how="left")
        join["population"] = _safe_numeric(join["population"])
        join["pop_density_per_km2"] = np.where(join["area_km2"] > 0, join["population"] / join["area_km2"], np.nan)

        pop_map = join[["pixel_id", "pop_density_per_km2"]].drop_duplicates(subset=["pixel_id"])
        out = out.merge(pop_map, on="pixel_id", how="left", suffixes=("", "_new"))
        out["pop_density_per_km2"] = _safe_numeric(out.get("pop_density_per_km2_new", out["pop_density_per_km2"]))
        out = out.drop(columns=[c for c in ["pop_density_per_km2_new"] if c in out.columns])

        coverage = float(out["pop_density_per_km2"].notna().mean())
        quality_rows.append(
            {
                "feature": "acs_pop_density",
                "status": "ok",
                "coverage_ratio": coverage,
                "notes": "ACS 2022 B01003 + TIGER tract ALAND",
            }
        )
    except Exception as e:
        quality_rows.append(
            {
                "feature": "acs_pop_density",
                "status": f"fallback:{type(e).__name__}",
                "coverage_ratio": float(out["pop_density_per_km2"].notna().mean()),
                "notes": "keep missing and continue",
            }
        )

    out["pop_density_per_km2"] = _safe_numeric(out["pop_density_per_km2"])
    out["pop_density_log1p"] = np.log1p(out["pop_density_per_km2"].clip(lower=0.0))
    out["missing_pop_flag"] = np.where(out["pop_density_per_km2"].isna(), 1, 0).astype(int)
    quality = pd.DataFrame(quality_rows)
    return out, quality


def build_recovery_from_panel(panel: pd.DataFrame, ctx: Optional[RunContext] = None) -> pd.DataFrame:
    defaults = load_json(CONFIG_DEFAULTS)
    threshold = float(defaults["recovery_threshold"])
    local_ctx = ctx if ctx is not None else RunContext(issues=[])
    rec = pipeline_lib_mod.build_recovery_panel(local_ctx, panel=panel, threshold=threshold, output_path=None)
    if "sample_lock_flag" not in rec.columns:
        rec["sample_lock_flag"] = 1
    return rec


def run_cloud_ablation(panel: pd.DataFrame, recovery: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_parts, agg_parts, coef_parts = [], [], []

    cat_terms = ["land_use_group", "event_disaster_type"]
    for spec_id, cloud_terms in CLOUD_VARIANTS.items():
        num_terms = BASE_NUMERIC + cloud_terms
        res = run_loeo_spec(
            panel,
            recovery,
            numeric_terms=num_terms,
            cat_terms=cat_terms,
            experiment_family="cloud_ablation",
            spec_id=spec_id,
        )
        fold_parts.append(res.fold_df)
        agg_parts.append(res.agg_df)
        coef_parts.append(res.coef_df)

    fold = pd.concat(fold_parts, ignore_index=True)
    agg = pd.concat(agg_parts, ignore_index=True)
    coef = pd.concat(coef_parts, ignore_index=True)

    # Cloud feature importance from Logit coefficients (mean abs coef over folds)
    cloud_feats = ["pixel_cloud_proxy", "pixel_pre_valid_ratio", "pixel_post_valid_ratio", "missing_cloud_flag"]
    imp = (
        coef[(coef["model"] == "Logit") & (coef["feature"].isin(cloud_feats))]
        .assign(abs_coef=lambda d: d["coef"].abs())
        .groupby(["spec_id", "feature"], as_index=False)["abs_coef"]
        .mean()
        .rename(columns={"abs_coef": "importance"})
    )
    if imp.empty:
        imp = pd.DataFrame(columns=["spec_id", "feature", "importance"])

    fold.to_csv(CLOUD_FOLD_PATH, index=False)
    agg.to_csv(CLOUD_AGG_PATH, index=False)
    imp.to_csv(CLOUD_IMPORTANCE_PATH, index=False)
    coef.to_csv(CLOUD_COEF_PATH, index=False)
    return fold, agg, coef


def run_noise_mask(panel: pd.DataFrame, recovery: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = panel.copy()
    p["land_use_int"] = pd.to_numeric(p["land_use"], errors="coerce").astype("Int64")

    subsets = {
        "M0": p,
        "M1": p[~p["land_use_int"].isin(list(MASK_CLASSES_M1))].copy(),
        "M2": p[~p["land_use_int"].isin(list(MASK_CLASSES_M1 | MASK_CLASSES_M2_EXTRA))].copy(),
    }

    coverage_rows = []
    for name, sdf in subsets.items():
        for event_id, grp in p.groupby("event_id"):
            kept = int((sdf["event_id"] == event_id).sum())
            total = int((p["event_id"] == event_id).sum())
            coverage_rows.append(
                {
                    "spec_id": name,
                    "event_id": event_id,
                    "n_kept": kept,
                    "n_total": total,
                    "kept_ratio": float(kept / total) if total > 0 else np.nan,
                }
            )
    coverage = pd.DataFrame(coverage_rows)

    fold_parts, agg_parts, coef_parts = [], [], []
    num_terms = BASE_NUMERIC + ["pixel_cloud_proxy"]
    cat_terms = ["land_use_group", "event_disaster_type"]

    for spec_id, sdf in subsets.items():
        rsub = recovery[recovery["pixel_id"].isin(sdf["pixel_id"])].copy()
        res = run_loeo_spec(
            sdf,
            rsub,
            numeric_terms=num_terms,
            cat_terms=cat_terms,
            experiment_family="noise_mask",
            spec_id=spec_id,
        )
        fold_parts.append(res.fold_df)
        agg_parts.append(res.agg_df)
        coef_parts.append(res.coef_df)

    fold = pd.concat(fold_parts, ignore_index=True)
    agg = pd.concat(agg_parts, ignore_index=True)
    coef = pd.concat(coef_parts, ignore_index=True)

    coef_effect = (
        coef[coef["feature"] == "in_buffer"]
        .groupby(["spec_id", "model"], as_index=False)["coef"]
        .mean()
        .rename(columns={"coef": "coef_in_buffer_mean"})
    )

    coverage.to_csv(MASK_COVERAGE_PATH, index=False)
    agg.to_csv(MASK_METRICS_PATH, index=False)
    coef_effect.to_csv(MASK_COEF_PATH, index=False)
    coef.to_csv(MASK_FULL_COEF_PATH, index=False)
    return fold, agg, coef


def run_urban_population(panel: pd.DataFrame, recovery: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = panel.copy()

    split_summary = (
        p.groupby(["event_id", "urban_rural_stratum"], as_index=False)
        .agg(
            n_obs=("pixel_id", "count"),
            damage_rate=("is_damaged", "mean"),
            pre_ntl_mean=("pre_mean_ntl", "mean"),
            pop_density_mean=("pop_density_per_km2", "mean"),
        )
    )

    fold_parts, agg_parts, coef_parts = [], [], []

    num_terms = BASE_NUMERIC + ["pixel_cloud_proxy", "is_cbsa", "is_urban_area", "pop_density_log1p", "missing_pop_flag"]
    cat_terms = ["land_use_group", "event_disaster_type", "urban_rural_stratum"]

    # Full
    res_full = run_loeo_spec(
        p,
        recovery,
        numeric_terms=num_terms,
        cat_terms=cat_terms,
        experiment_family="urban_rural",
        spec_id="UR_full",
    )
    fold_parts.append(res_full.fold_df)
    agg_parts.append(res_full.agg_df)
    coef_parts.append(res_full.coef_df)

    # Urban only
    p_urban = p[p["urban_rural_stratum"] == "urban"].copy()
    r_urban = recovery[recovery["pixel_id"].isin(p_urban["pixel_id"])].copy()
    if p_urban["event_id"].nunique() >= 3:
        res_urban = run_loeo_spec(
            p_urban,
            r_urban,
            numeric_terms=num_terms,
            cat_terms=["land_use_group", "event_disaster_type"],
            experiment_family="urban_rural",
            spec_id="UR_urban",
        )
        fold_parts.append(res_urban.fold_df)
        agg_parts.append(res_urban.agg_df)
        coef_parts.append(res_urban.coef_df)

    # Rural only
    p_rural = p[p["urban_rural_stratum"] == "rural"].copy()
    r_rural = recovery[recovery["pixel_id"].isin(p_rural["pixel_id"])].copy()
    if p_rural["event_id"].nunique() >= 3:
        res_rural = run_loeo_spec(
            p_rural,
            r_rural,
            numeric_terms=num_terms,
            cat_terms=["land_use_group", "event_disaster_type"],
            experiment_family="urban_rural",
            spec_id="UR_rural",
        )
        fold_parts.append(res_rural.fold_df)
        agg_parts.append(res_rural.agg_df)
        coef_parts.append(res_rural.coef_df)

    fold = pd.concat(fold_parts, ignore_index=True)
    agg = pd.concat(agg_parts, ignore_index=True)

    split_summary.to_csv(URBAN_SPLIT_PATH, index=False)
    agg.to_csv(URBAN_MODEL_PATH, index=False)
    coef = pd.concat(coef_parts, ignore_index=True)
    coef.to_csv(URBAN_COEF_PATH, index=False)
    return fold, agg, coef


def _moran_i_knn(df: pd.DataFrame, value_col: str, k: int = 8, n_perm: int = 199) -> Tuple[float, float]:
    if len(df) < (k + 2):
        return np.nan, np.nan

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    coords = np.column_stack([x, y])

    z = _safe_numeric(df[value_col]).to_numpy().astype(float)
    z = z - z.mean()
    denom = float((z ** 2).sum())
    if denom == 0:
        return np.nan, np.nan

    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=min(k + 1, len(df)))
    nbrs = idx[:, 1:]

    n = len(df)
    row_std_w = np.full(nbrs.shape, 1.0 / max(nbrs.shape[1], 1), dtype=float)
    s0 = float(row_std_w.sum())

    num = 0.0
    for i in range(n):
        zi = z[i]
        jidx = nbrs[i]
        wij = row_std_w[i]
        num += float((wij * zi * z[jidx]).sum())

    I = float((n / s0) * (num / denom)) if s0 > 0 else np.nan

    # Permutation p-value (two-sided)
    perm_vals = []
    for _ in range(n_perm):
        zp = np.random.permutation(z)
        nnum = 0.0
        for i in range(n):
            zi = zp[i]
            jidx = nbrs[i]
            wij = row_std_w[i]
            nnum += float((wij * zi * zp[jidx]).sum())
        perm_vals.append(float((n / s0) * (nnum / denom)) if s0 > 0 else np.nan)

    perm_arr = np.asarray(perm_vals, dtype=float)
    pval = float((np.sum(np.abs(perm_arr) >= abs(I)) + 1) / (len(perm_arr) + 1))
    return I, pval


def run_spatial_and_contribution(panel: pd.DataFrame, coef_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = _prepare_columns(
        panel,
        numeric_terms=BASE_NUMERIC + ["pixel_cloud_proxy", "is_cbsa", "is_urban_area", "pop_density_log1p", "missing_pop_flag"],
        cat_terms=["land_use_group", "event_disaster_type", "urban_rural_stratum"],
    )

    formula = _build_formula(
        "delta_ntl",
        numeric_terms=BASE_NUMERIC + ["pixel_cloud_proxy", "is_cbsa", "is_urban_area", "pop_density_log1p"],
        cat_terms=["land_use_group", "event_disaster_type", "urban_rural_stratum"],
    )

    ols = smf.ols(formula, data=p).fit(cov_type="HC1")
    p["ols_resid"] = ols.resid

    moran_rows = []
    for event_id, grp in p.groupby("event_id"):
        I, pval = _moran_i_knn(grp, "ols_resid", k=8, n_perm=199)
        moran_rows.append({"event_id": event_id, "moran_i": I, "p_value": pval, "n_obs": int(len(grp))})
    moran = pd.DataFrame(moran_rows)

    # Spatial cluster-robust SE (5km bins)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = tr.transform(p["lon"].to_numpy(), p["lat"].to_numpy())
    p["cluster_5km"] = (np.floor(x / 5000).astype(int).astype(str) + "_" + np.floor(y / 5000).astype(int).astype(str))

    ols_cluster = smf.ols(formula, data=p).fit(cov_type="cluster", cov_kwds={"groups": p["cluster_5km"]})
    se_cmp = pd.DataFrame(
        [
            {
                "term": "in_buffer",
                "coef_hc1": float(ols.params.get("in_buffer", np.nan)),
                "se_hc1": float(ols.bse.get("in_buffer", np.nan)),
                "p_hc1": float(ols.pvalues.get("in_buffer", np.nan)),
                "coef_cluster": float(ols_cluster.params.get("in_buffer", np.nan)),
                "se_cluster": float(ols_cluster.bse.get("in_buffer", np.nan)),
                "p_cluster": float(ols_cluster.pvalues.get("in_buffer", np.nan)),
            }
        ]
    )

    # Contribution scorecards
    keep_models = {"OLS", "MixedLM", "Logit", "Cox", "AFT"}
    cf = coef_all[coef_all["model"].isin(keep_models)].copy()
    if cf.empty:
        score = pd.DataFrame(
            columns=[
                "model",
                "spec_id",
                "feature",
                "effect_direction",
                "effect_size",
                "uncertainty",
                "loeo_sign_consistency",
                "net_contribution_label",
                "evidence_ref",
            ]
        )
        rank = pd.DataFrame(columns=["model", "feature", "rank", "effect_size_abs"])
    else:
        rows = []
        for (model, spec, feat), grp in cf.groupby(["model", "spec_id", "feature"], dropna=False):
            vals = _safe_numeric(grp["coef"])
            mean_coef = float(vals.mean())
            std_coef = float(vals.std(ddof=0))
            sign_cons = float(np.mean(np.sign(vals) == np.sign(mean_coef))) if len(vals) else np.nan

            if not np.isfinite(mean_coef):
                direction = "unknown"
            elif mean_coef > 0:
                direction = "positive"
            elif mean_coef < 0:
                direction = "negative"
            else:
                direction = "neutral"

            if not np.isfinite(sign_cons) or sign_cons < 0.6:
                label = "unstable"
            elif direction == "positive":
                label = "positive_contribution"
            elif direction == "negative":
                label = "negative_contribution"
            else:
                label = "neutral"

            rows.append(
                {
                    "model": model,
                    "spec_id": spec,
                    "feature": feat,
                    "effect_direction": direction,
                    "effect_size": mean_coef,
                    "uncertainty": std_coef,
                    "loeo_sign_consistency": sign_cons,
                    "net_contribution_label": label,
                    "evidence_ref": "coef_loeo",
                }
            )
        score = pd.DataFrame(rows)
        rank = (
            score.assign(effect_size_abs=lambda d: d["effect_size"].abs())
            .sort_values(["model", "effect_size_abs"], ascending=[True, False])
            .groupby("model", as_index=False)
            .head(30)
        )
        rank["rank"] = rank.groupby("model").cumcount() + 1
        rank = rank[["model", "feature", "rank", "effect_size_abs"]]

    moran.to_csv(MORAN_PATH, index=False)
    se_cmp.to_csv(SPATIAL_SE_PATH, index=False)
    score.to_csv(CONTRIB_SCORE_PATH, index=False)
    rank.to_csv(CONTRIB_RANK_PATH, index=False)
    return moran, se_cmp, score, rank


def _compute_extreme_candidates() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not SHIFT_V3_PATH.exists() or not FOLD_V3R1_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    shift = pd.read_csv(SHIFT_V3_PATH)
    fold = pd.read_csv(FOLD_V3R1_PATH)

    ev_shift = shift[shift["diagnostic_type"] == "event_shift"].pivot_table(
        index="event_id", columns="metric_name", values="value", aggfunc="mean"
    )
    ev_shift = ev_shift.reset_index()
    if "smd_mean" not in ev_shift.columns:
        ev_shift["smd_mean"] = np.nan
    if "psi_mean" not in ev_shift.columns:
        ev_shift["psi_mean"] = np.nan

    # Fold performance by event
    perf = fold.pivot_table(
        index="fold_event",
        columns="model",
        values=["auc", "c_index", "rmse"],
        aggfunc="mean",
    )
    perf.columns = [f"{a}_{b}" for a, b in perf.columns]
    perf = perf.reset_index().rename(columns={"fold_event": "event_id"})

    df = ev_shift.merge(perf, on="event_id", how="left")

    smd_thr = float(_safe_numeric(df["smd_mean"]).quantile(2 / 3))
    psi_thr = float(_safe_numeric(df["psi_mean"]).quantile(2 / 3))

    auc_q = float(_safe_numeric(df.get("auc_Logit", pd.Series(np.nan))).quantile(1 / 3))
    cox_q = float(_safe_numeric(df.get("c_index_Cox", pd.Series(np.nan))).quantile(1 / 3))
    aft_q = float(_safe_numeric(df.get("c_index_AFT", pd.Series(np.nan))).quantile(1 / 3))
    rmse_q = float(_safe_numeric(df.get("rmse_OLS", pd.Series(np.nan))).quantile(2 / 3))

    bad_count = []
    for _, r in df.iterrows():
        c = 0
        if pd.notna(r.get("auc_Logit")) and float(r.get("auc_Logit")) <= auc_q:
            c += 1
        c_best = np.nanmax([r.get("c_index_Cox", np.nan), r.get("c_index_AFT", np.nan)])
        surv_thr = max(cox_q, aft_q)
        if np.isfinite(c_best) and c_best <= surv_thr:
            c += 1
        if pd.notna(r.get("rmse_OLS")) and float(r.get("rmse_OLS")) >= rmse_q:
            c += 1
        bad_count.append(c)
    df["bad_metric_count"] = bad_count

    df["high_shift_flag"] = ((df["smd_mean"] >= smd_thr) | (df["psi_mean"] >= psi_thr)).astype(int)
    df["poor_perf_flag"] = (df["bad_metric_count"] >= 2).astype(int)
    df["extreme_candidate"] = ((df["high_shift_flag"] == 1) & (df["poor_perf_flag"] == 1)).astype(int)

    shift_rank = _zscore(df["smd_mean"].rank(method="average")) + _zscore(df["psi_mean"].rank(method="average"))
    perf_rank = _zscore(df["bad_metric_count"].rank(method="average"))
    df["extreme_score"] = shift_rank + perf_rank
    df = df.sort_values("extreme_score", ascending=False).copy()
    df["extreme_candidate_soft"] = 0
    if int(df["extreme_candidate"].sum()) == 0 and len(df) > 0:
        soft_n = min(2, len(df))
        df.loc[df.index[:soft_n], "extreme_candidate_soft"] = 1

    candidates = df[["event_id", "high_shift_flag", "poor_perf_flag", "extreme_candidate"]].copy()
    candidates["extreme_candidate_soft"] = df["extreme_candidate_soft"].to_numpy()
    score = df.copy()
    score["source_ref"] = "cross_event_shift_diagnostics_v3 + cross_event_fold_metrics_v3r1"

    candidates.to_csv(EXTREME_CANDIDATE_PATH, index=False)
    score.to_csv(EXTREME_SCORE_PATH, index=False)
    return candidates, score


def run_extreme_drop_sensitivity(
    panel: pd.DataFrame,
    recovery: pd.DataFrame,
    score: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    all_events = sorted(panel["event_id"].unique().tolist())

    # Main full spec for this sensitivity
    num_terms = BASE_NUMERIC + ["pixel_cloud_proxy", "is_cbsa", "is_urban_area", "pop_density_log1p"]
    cat_terms = ["land_use_group", "event_disaster_type", "urban_rural_stratum"]

    drop_scenarios: Dict[str, List[str]] = {}
    drop_scenarios["drop_none"] = all_events.copy()

    score = score.sort_values("extreme_score", ascending=False).copy()
    extreme_events = score.loc[score["extreme_candidate"] == 1, "event_id"].tolist()
    used_soft = False
    if len(extreme_events) == 0:
        extreme_events = score.loc[score.get("extreme_candidate_soft", 0) == 1, "event_id"].tolist()
        used_soft = True

    for ev in extreme_events:
        drop_scenarios[f"drop1_{ev}"] = [e for e in all_events if e != ev]

    top2 = extreme_events[:2]
    if len(top2) == 2:
        drop_scenarios["drop2_top_extreme"] = [e for e in all_events if e not in set(top2)]

    non_ext = score.loc[score["extreme_candidate"] == 0, "event_id"].tolist()
    if non_ext:
        ctrl_event = non_ext[0]
        drop_scenarios[f"control_drop_{ctrl_event}"] = [e for e in all_events if e != ctrl_event]
    else:
        ctrl_event = None

    fold_parts, agg_parts = [], []
    for scenario, kept_events in drop_scenarios.items():
        if len(kept_events) < 3:
            continue
        res = run_loeo_spec(
            panel,
            recovery,
            numeric_terms=num_terms,
            cat_terms=cat_terms,
            experiment_family="extreme_drop",
            spec_id=scenario,
            allowed_events=kept_events,
        )
        fold_parts.append(res.fold_df)
        agg_parts.append(res.agg_df)

    fold = pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()
    agg = pd.concat(agg_parts, ignore_index=True) if agg_parts else pd.DataFrame()

    if not fold.empty:
        fold.to_csv(EXTREME_DROP_METRIC_PATH, index=False)
    else:
        pd.DataFrame(columns=["experiment_family", "spec_id", "fold_event", "model", "rmse", "mae", "auc", "brier", "c_index", "coef_in_buffer", "notes"]).to_csv(
            EXTREME_DROP_METRIC_PATH, index=False
        )

    if not agg.empty:
        agg.to_csv(EXTREME_DROP_AGG_PATH, index=False)
    else:
        pd.DataFrame(columns=["experiment_family", "spec_id", "model", "rmse", "mae", "auc", "brier", "c_index", "coef_in_buffer", "n_folds"]).to_csv(
            EXTREME_DROP_AGG_PATH, index=False
        )

    decision = {
        "logic": "extreme_drop_improves_and_control_not",
        "extreme_candidates": extreme_events,
        "used_soft_candidates": used_soft,
        "control_drop_event": ctrl_event,
        "decision": "insufficient_evidence",
        "evidence": {},
    }

    try:
        if not agg.empty:
            base = agg[(agg["spec_id"] == "drop_none") & (agg["model"] == "Logit")]["auc"].mean()
            extreme_best = agg[agg["spec_id"].str.startswith("drop1_", na=False) & (agg["model"] == "Logit")]["auc"].max()
            control_auc = agg[agg["spec_id"].str.startswith("control_drop_", na=False) & (agg["model"] == "Logit")]["auc"].mean()
            decision["evidence"] = {
                "base_logit_auc": None if pd.isna(base) else float(base),
                "best_extreme_drop_logit_auc": None if pd.isna(extreme_best) else float(extreme_best),
                "control_drop_logit_auc": None if pd.isna(control_auc) else float(control_auc),
            }
            if pd.notna(base) and pd.notna(extreme_best):
                if (extreme_best - base) >= 0.02 and (pd.isna(control_auc) or (control_auc - base) < 0.01):
                    decision["decision"] = "extreme_event_dominance_supported"
                else:
                    decision["decision"] = "structural_shift_not_single_event"
    except Exception:
        pass

    EXTREME_DECISION_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    return fold, agg, decision


def _extract_pixel_stack_values(paths: Sequence[Path], sub: pd.DataFrame) -> np.ndarray:
    if not paths or sub.empty:
        return np.empty((len(sub), 0), dtype=float)
    rr = sub["row"].astype(int).to_numpy()
    cc = sub["col"].astype(int).to_numpy()
    series = []
    for path in paths:
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float64")
            series.append(arr[rr, cc])
    return np.vstack(series).T


def build_target_quality_panel(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    defaults = load_json(CONFIG_DEFAULTS)
    recovery_thr = float(defaults["recovery_threshold"])
    damage_thr = float(defaults["damage_threshold"])
    events_cfg = load_json(CONFIG_EVENTS)

    out = panel.copy()
    out["delta_ntl_raw"] = out["delta_ntl"]
    out["valid_pre_days"] = np.nan
    out["valid_post_days"] = np.nan
    out["post_obs_span_days"] = np.nan
    out["recovery_day_first_valid"] = np.nan
    out["recovery_day_first_threshold_hit"] = np.nan
    out["recovery_obs_quality_score"] = np.nan
    out["high_censoring_risk_flag"] = 0
    out["delta_ntl_obs_adjusted"] = out["delta_ntl"]

    rec_parts: List[pd.DataFrame] = []
    audit_rows: List[Dict[str, object]] = []

    for event_id, cfg in events_cfg.items():
        mask = out["event_id"] == event_id
        if mask.sum() == 0:
            continue
        sub = out.loc[mask].copy()
        pre_paths = list_daily_tifs(ROOT / cfg["pre_dir"])
        post_paths = list_daily_tifs(ROOT / cfg["post_dir"])
        if not pre_paths or not post_paths:
            continue

        pre_vals = _extract_pixel_stack_values(pre_paths, sub)
        post_vals = _extract_pixel_stack_values(post_paths, sub)

        valid_pre_days = np.isfinite(pre_vals).sum(axis=1).astype(float)
        valid_post_days = np.isfinite(post_vals).sum(axis=1).astype(float)
        total_pre = float(pre_vals.shape[1]) if pre_vals.ndim == 2 else 0.0
        total_post = float(post_vals.shape[1]) if post_vals.ndim == 2 else 0.0

        pre_ratio = np.divide(valid_pre_days, total_pre, out=np.zeros_like(valid_pre_days), where=total_pre > 0)
        post_ratio = np.divide(valid_post_days, total_post, out=np.zeros_like(valid_post_days), where=total_post > 0)
        obs_weight = np.sqrt(np.clip(pre_ratio, 0.0, 1.0) * np.clip(post_ratio, 0.0, 1.0))

        event_ref = float(sub.loc[obs_weight >= 0.8, "delta_ntl"].median()) if np.any(obs_weight >= 0.8) else float(sub["delta_ntl"].median())
        delta_adj = obs_weight * sub["delta_ntl"].to_numpy() + (1.0 - obs_weight) * event_ref

        targets = sub["pre_mean_ntl"].to_numpy(dtype=float) * recovery_thr
        valid_mask = np.isfinite(post_vals)
        hit_mask = valid_mask & (post_vals >= targets[:, None])
        observed = hit_mask.any(axis=1).astype(int)
        first_valid = np.where(valid_mask.any(axis=1), valid_mask.argmax(axis=1) + 1, np.ceil(total_post).astype(int))
        first_hit = np.where(observed == 1, hit_mask.argmax(axis=1) + 1, np.ceil(total_post).astype(int))
        quality_score = np.divide(valid_post_days, total_post, out=np.zeros_like(valid_post_days), where=total_post > 0)
        high_censor = ((quality_score < 0.60) | ((observed == 0) & (quality_score < 0.85))).astype(int)

        out.loc[mask, "valid_pre_days"] = valid_pre_days
        out.loc[mask, "valid_post_days"] = valid_post_days
        out.loc[mask, "post_obs_span_days"] = total_post
        out.loc[mask, "recovery_day_first_valid"] = first_valid
        out.loc[mask, "recovery_day_first_threshold_hit"] = first_hit
        out.loc[mask, "recovery_obs_quality_score"] = quality_score
        out.loc[mask, "high_censoring_risk_flag"] = high_censor
        out.loc[mask, "delta_ntl_obs_adjusted"] = delta_adj

        event_rec = sub[["pixel_id", "event_id"]].copy()
        event_rec["recovery_days"] = first_hit.astype(int)
        event_rec["event_observed"] = observed.astype(int)
        event_rec["recovery_threshold"] = recovery_thr
        event_rec["valid_pre_days"] = valid_pre_days
        event_rec["valid_post_days"] = valid_post_days
        event_rec["post_obs_span_days"] = total_post
        event_rec["recovery_day_first_valid"] = first_valid
        event_rec["recovery_day_first_threshold_hit"] = first_hit
        event_rec["recovery_obs_quality_score"] = quality_score
        event_rec["high_censoring_risk_flag"] = high_censor
        rec_parts.append(event_rec)

        audit_rows.append(
            {
                "event_id": event_id,
                "n_obs": int(len(sub)),
                "total_pre_days": int(total_pre),
                "total_post_days": int(total_post),
                "mean_pre_valid_ratio_raw": float(pre_ratio.mean()),
                "mean_post_valid_ratio_raw": float(post_ratio.mean()),
                "mean_obs_weight": float(obs_weight.mean()),
                "event_ref_delta": event_ref,
                "delta_shift_mean": float(np.mean(delta_adj - sub["delta_ntl"].to_numpy())),
                "delta_shift_abs_mean": float(np.mean(np.abs(delta_adj - sub["delta_ntl"].to_numpy()))),
                "observed_rate_v2": float(observed.mean()),
                "high_censoring_share": float(high_censor.mean()),
            }
        )

    out["is_damaged_raw"] = out["is_damaged"]
    out["is_damaged"] = (out["delta_ntl_obs_adjusted"] < damage_thr).astype(int)
    out["delta_ntl"] = out["delta_ntl_obs_adjusted"]
    out["pixel_pre_valid_ratio"] = _safe_numeric(out["pixel_pre_valid_ratio"])
    out["pixel_post_valid_ratio"] = _safe_numeric(out["pixel_post_valid_ratio"])
    out["recovery_obs_quality_score"] = _safe_numeric(out["recovery_obs_quality_score"])
    out["high_censoring_risk_flag"] = _safe_numeric(out["high_censoring_risk_flag"]).astype(int)

    rec = pd.concat(rec_parts, ignore_index=True)
    merge_cols = [
        "pixel_id",
        "event_id",
        "in_buffer",
        "pre_mean_ntl",
        "land_use_group",
        "event_disaster_type",
        "osm_dist_any_m",
        "osm_power_count_1000m",
        "osm_medical_count_1000m",
        "pixel_cloud_proxy",
        "pixel_pre_valid_ratio",
        "pixel_post_valid_ratio",
        "urban_share_1km",
        "water_share_1km",
        "developed_high_share_1km",
        "recovery_obs_quality_score",
        "high_censoring_risk_flag",
    ]
    merge_cols = [c for c in merge_cols if c in out.columns]
    rec = rec.merge(out[merge_cols].drop_duplicates(subset=["pixel_id"]), on=["pixel_id", "event_id"], how="left")
    audit = pd.DataFrame(audit_rows)

    out.to_parquet(PANEL_QUALITY_PATH, index=False)
    rec.to_parquet(RECOVERY_V2_PATH, index=False)
    audit.to_csv(TARGET_QUALITY_AUDIT_PATH, index=False)
    return out, rec, audit


def run_quality_transport(panel: pd.DataFrame, recovery: pd.DataFrame) -> SpecResult:
    numeric_terms = BASE_NUMERIC + [
        "pixel_cloud_proxy",
        "pixel_pre_valid_ratio",
        "pixel_post_valid_ratio",
        "recovery_obs_quality_score",
        "high_censoring_risk_flag",
    ]
    cat_terms = ["land_use_group", "event_disaster_type"]
    res = run_loeo_spec(
        panel,
        recovery,
        numeric_terms=numeric_terms,
        cat_terms=cat_terms,
        experiment_family="quality_transport",
        spec_id="QT1",
    )
    res.fold_df.to_csv(QUALITY_TRANSPORT_FOLD_PATH, index=False)
    res.agg_df.to_csv(QUALITY_TRANSPORT_AGG_PATH, index=False)
    return res


def run_spatial_block_cv(panel: pd.DataFrame, recovery: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = tr.transform(p["lon"].to_numpy(), p["lat"].to_numpy())
    p["spatial_block_id"] = (
        p["event_id"].astype(str)
        + "_"
        + np.floor(x / 5000).astype(int).astype(str)
        + "_"
        + np.floor(y / 5000).astype(int).astype(str)
    )

    groups = p["spatial_block_id"].astype(str)
    unique_groups = groups.nunique()
    if unique_groups < 3:
        out = pd.DataFrame(columns=["experiment_family", "spec_id", "model", "rmse", "mae", "auc", "brier", "c_index", "coef_in_buffer", "n_folds"])
        out.to_csv(SPATIAL_BLOCK_CV_PATH, index=False)
        return out

    numeric_terms = BASE_NUMERIC + [
        "pixel_cloud_proxy",
        "pixel_pre_valid_ratio",
        "pixel_post_valid_ratio",
        "recovery_obs_quality_score",
        "high_censoring_risk_flag",
    ]
    cat_terms = ["land_use_group", "event_disaster_type"]
    n_splits = min(5, unique_groups)
    gkf = GroupKFold(n_splits=n_splits)
    fold_rows: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(p, groups=groups), start=1):
        tr_panel = _prepare_columns(p.iloc[train_idx].copy(), numeric_terms, cat_terms, required_numeric=("delta_ntl", "is_damaged"))
        te_panel = _prepare_columns(p.iloc[test_idx].copy(), numeric_terms, cat_terms, required_numeric=("delta_ntl", "is_damaged"))
        train_pixels = set(tr_panel["pixel_id"].astype(str))
        test_pixels = set(te_panel["pixel_id"].astype(str))
        tr_rec = _prepare_columns(recovery[recovery["pixel_id"].astype(str).isin(train_pixels)].copy(), numeric_terms, cat_terms, required_numeric=("recovery_days", "event_observed"))
        te_rec = _prepare_columns(recovery[recovery["pixel_id"].astype(str).isin(test_pixels)].copy(), numeric_terms, cat_terms, required_numeric=("recovery_days", "event_observed"))
        rows, _ = _evaluate_fold(
            tr_panel,
            te_panel,
            tr_rec,
            te_rec,
            numeric_terms=numeric_terms,
            cat_terms=cat_terms,
            experiment_family="spatial_block_cv",
            spec_id="SB1",
            fold_event=f"block_fold_{fold_idx}",
        )
        fold_rows.extend(rows)

    fold_df = pd.DataFrame(fold_rows)
    agg_df = _aggregate_metrics(fold_df)
    agg_df.to_csv(SPATIAL_BLOCK_CV_PATH, index=False)
    return agg_df


def build_facility_context_panel(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    events_cfg = load_json(CONFIG_EVENTS)
    matched_rows: List[pd.DataFrame] = []
    quality_rows: List[Dict[str, object]] = []
    match_features = ["pre_mean_ntl", "urban_share_1km", "water_share_1km", "developed_high_share_1km"]

    for event_id, cfg in events_cfg.items():
        sub = panel[panel["event_id"] == event_id].copy()
        if sub.empty:
            continue

        poi = pd.read_csv(ROOT / cfg["poi_csv"]).copy()
        if poi.empty:
            continue
        poi["facility_id"] = poi["osm_id"].astype(str) if "osm_id" in poi.columns else [f"{event_id}_{i}" for i in range(len(poi))]
        poi["facility_type_std"] = poi["type"].astype(str).map(standardize_facility_type)
        poi["buffer_radius_m"] = np.where(poi["facility_type_std"] == "aerodrome", BUFFER_RADII["aerodrome"], DEFAULT_BUFFER)

        transformer = Transformer.from_crs("EPSG:4326", cfg["metric_crs"], always_xy=True)
        px, py = transformer.transform(sub["lon"].to_numpy(), sub["lat"].to_numpy())
        qx, qy = transformer.transform(poi["lon"].to_numpy(), poi["lat"].to_numpy())
        pix_xy = np.column_stack([px, py])
        poi_xy = np.column_stack([qx, qy])
        tree = cKDTree(poi_xy)
        dist, idx = tree.query(pix_xy, k=1)

        sub["facility_id"] = poi.iloc[idx]["facility_id"].to_numpy()
        sub["facility_local_context_type"] = poi.iloc[idx]["facility_type_std"].to_numpy()
        sub["nearest_facility_buffer_m"] = poi.iloc[idx]["buffer_radius_m"].to_numpy(dtype=float)
        sub["nearest_facility_distance_m"] = dist.astype(float)
        sub["within_facility_buffer"] = (sub["nearest_facility_distance_m"] <= sub["nearest_facility_buffer_m"]).astype(int)
        sub["within_facility_ring"] = (sub["nearest_facility_distance_m"] <= 2500.0).astype(int)

        for c in match_features:
            sub[c] = _safe_numeric(sub[c])
            std = float(sub[c].std(ddof=0))
            sub[f"{c}_z"] = 0.0 if std == 0 or not np.isfinite(std) else (sub[c] - float(sub[c].mean())) / std

        event_pairs: List[pd.DataFrame] = []
        for facility_id, grp in sub.groupby("facility_id", observed=True):
            treated = grp[(grp["within_facility_buffer"] == 1)].copy()
            controls = grp[(grp["within_facility_buffer"] == 0) & (grp["within_facility_ring"] == 1)].copy()
            if treated.empty or controls.empty:
                quality_rows.append(
                    {
                        "event_id": event_id,
                        "facility_id": facility_id,
                        "facility_type": grp["facility_local_context_type"].iloc[0],
                        "n_treated": int(len(treated)),
                        "n_control_candidates": int(len(controls)),
                        "matched_pairs": 0,
                        "mean_match_distance": np.nan,
                    }
                )
                continue

            ctrl_feat = controls[[f"{c}_z" for c in match_features]].to_numpy(dtype=float)
            matched_local: List[pd.DataFrame] = []
            match_scores: List[float] = []
            for _, trow in treated.iterrows():
                candidates = controls[controls["land_use_group"] == trow["land_use_group"]].copy()
                if candidates.empty:
                    candidates = controls.copy()
                cand_feat = candidates[[f"{c}_z" for c in match_features]].to_numpy(dtype=float)
                tgt = trow[[f"{c}_z" for c in match_features]].to_numpy(dtype=float)
                d = np.sqrt(((cand_feat - tgt) ** 2).sum(axis=1))
                best_idx = int(np.argmin(d))
                best = candidates.iloc[[best_idx]].copy()
                score = float(d[best_idx])
                group_id = f"{event_id}_{facility_id}_{trow['pixel_id']}"

                t_df = pd.DataFrame([trow]).copy()
                t_df["match_group_id"] = group_id
                t_df["match_distance_score"] = score
                t_df["matched_control_count"] = int(len(candidates))

                best["match_group_id"] = group_id
                best["match_distance_score"] = score
                best["matched_control_count"] = int(len(candidates))
                matched_local.extend([t_df, best])
                match_scores.append(score)

            if matched_local:
                event_pairs.append(pd.concat(matched_local, ignore_index=True))
            quality_rows.append(
                {
                    "event_id": event_id,
                    "facility_id": facility_id,
                    "facility_type": grp["facility_local_context_type"].iloc[0],
                    "n_treated": int(len(treated)),
                    "n_control_candidates": int(len(controls)),
                    "matched_pairs": int(len(match_scores)),
                    "mean_match_distance": float(np.mean(match_scores)) if match_scores else np.nan,
                }
            )

        if event_pairs:
            matched_rows.append(pd.concat(event_pairs, ignore_index=True))

    fac_panel = pd.concat(matched_rows, ignore_index=True) if matched_rows else pd.DataFrame()
    quality = pd.DataFrame(quality_rows)
    if not fac_panel.empty:
        fac_panel.to_parquet(FACILITY_CONTEXT_PATH, index=False)
    else:
        pd.DataFrame().to_parquet(FACILITY_CONTEXT_PATH, index=False)
    quality.to_csv(FACILITY_MATCH_QUALITY_PATH, index=False)
    return fac_panel, quality


def fit_facility_centered_models(fac_panel: pd.DataFrame) -> pd.DataFrame:
    if fac_panel.empty:
        out = pd.DataFrame(columns=["model", "metric_name", "term", "value", "p_value", "ci_low", "ci_high", "n_obs", "notes"])
        out.to_csv(FACILITY_CENTERED_SUMMARY_PATH, index=False)
        return out

    formula = (
        "delta_ntl_obs_adjusted ~ in_buffer + pre_mean_ntl + urban_share_1km + "
        "water_share_1km + developed_high_share_1km + C(event_id) + C(land_use_group)"
    )
    ols = smf.ols(formula, data=fac_panel).fit(cov_type="cluster", cov_kwds={"groups": fac_panel["facility_id"]})
    logit = smf.glm(
        "is_damaged ~ in_buffer + pre_mean_ntl + urban_share_1km + water_share_1km + developed_high_share_1km + C(event_id) + C(land_use_group)",
        data=fac_panel,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": fac_panel["facility_id"]})

    pair_diff = (
        fac_panel.groupby("match_group_id", observed=True)
        .apply(lambda g: float(g.loc[g["in_buffer"] == 1, "delta_ntl_obs_adjusted"].mean() - g.loc[g["in_buffer"] == 0, "delta_ntl_obs_adjusted"].mean()))
        .rename("paired_delta_diff")
        .reset_index()
    )

    rows = []
    ols_ci = ols.conf_int().loc["in_buffer"]
    rows.append(
        {
            "model": "FacilityMatchedOLS",
            "metric_name": "coef_in_buffer",
            "term": "in_buffer",
            "value": float(ols.params["in_buffer"]),
            "p_value": float(ols.pvalues["in_buffer"]),
            "ci_low": float(ols_ci.iloc[0]),
            "ci_high": float(ols_ci.iloc[1]),
            "n_obs": int(len(fac_panel)),
            "notes": "clustered_by_facility",
        }
    )
    logit_ci = logit.conf_int().loc["in_buffer"]
    rows.append(
        {
            "model": "FacilityMatchedLogit",
            "metric_name": "odds_ratio_in_buffer",
            "term": "in_buffer",
            "value": float(np.exp(logit.params["in_buffer"])),
            "p_value": float(logit.pvalues["in_buffer"]),
            "ci_low": float(np.exp(logit_ci.iloc[0])),
            "ci_high": float(np.exp(logit_ci.iloc[1])),
            "n_obs": int(len(fac_panel)),
            "notes": "glm_binomial_clustered_by_facility",
        }
    )
    rows.append(
        {
            "model": "FacilityPairedATT",
            "metric_name": "mean_delta_diff",
            "term": "treated_minus_control",
            "value": float(pair_diff["paired_delta_diff"].mean()) if not pair_diff.empty else np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_obs": int(pair_diff["match_group_id"].nunique()) if not pair_diff.empty else 0,
            "notes": "positive_means_treated_higher_delta_ntl",
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(FACILITY_CENTERED_SUMMARY_PATH, index=False)
    return out


def build_model_role_matrix(quality_agg: pd.DataFrame, spatial_block_agg: pd.DataFrame, facility_summary: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "model": "MixedLM",
            "role": "explanatory_main",
            "scope": "event-aware_in_sample",
            "primary_metric": "coef_in_buffer",
            "justification": "best explanatory random-effects model in strict-v2 baseline",
        },
        {
            "model": "Logit",
            "role": "damage_transport_main",
            "scope": "loeo_transport_quality",
            "primary_metric": "auc",
            "justification": "most interpretable damage probability model under cross-event transport",
        },
        {
            "model": "AFT",
            "role": "recovery_transport_main",
            "scope": "loeo_transport_quality",
            "primary_metric": "c_index",
            "justification": "more stable than Cox under censoring and PH risk",
        },
        {
            "model": "Cox",
            "role": "recovery_interpretive_secondary",
            "scope": "loeo_transport_quality",
            "primary_metric": "hazard_ratio_or_c_index",
            "justification": "kept as explanatory survival comparison, not the primary transport KPI",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(MODEL_ROLE_MATRIX_PATH, index=False)
    return out


def _plot_quality_matched_summary(quality_agg: pd.DataFrame, spatial_block_agg: pd.DataFrame) -> None:
    baseline_auc = 0.4827375101952784
    baseline_surv = 0.5213379750676524
    q_logit = float(quality_agg.loc[quality_agg["model"] == "Logit", "auc"].mean()) if not quality_agg.empty else np.nan
    q_aft = float(quality_agg.loc[quality_agg["model"] == "AFT", "c_index"].mean()) if not quality_agg.empty else np.nan
    sb_logit = float(spatial_block_agg.loc[spatial_block_agg["model"] == "Logit", "auc"].mean()) if not spatial_block_agg.empty else np.nan
    sb_aft = float(spatial_block_agg.loc[spatial_block_agg["model"] == "AFT", "c_index"].mean()) if not spatial_block_agg.empty else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].bar(["v3r1", "quality_loeo", "spatial_block"], [baseline_auc, q_logit, sb_logit], color=["#8c8c8c", "#2c7fb8", "#d95f02"])
    axes[0].set_title("Logit AUC Comparison")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].bar(["v3r1", "quality_loeo", "spatial_block"], [baseline_surv, q_aft, sb_aft], color=["#8c8c8c", "#2c7fb8", "#d95f02"])
    axes[1].set_title("AFT c-index Comparison")
    axes[1].set_ylim(0.0, 1.0)
    plt.tight_layout()
    fig.savefig(QUALITY_FIG_PATH, dpi=220)
    plt.close(fig)


def _update_quality_index() -> None:
    line = "- `project/modeling_report/11_quality_matched_report.md`"
    note = "- Appendix quality/matched outputs: `project/modeling/output/quality_transport_aggregate_metrics_v1.csv`, `project/modeling/output/facility_centered_model_summary.csv`, `project/modeling/output/spatial_block_cv_metrics_v1.csv`"
    text = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else "# Modeling Report Index\n\n## Deliverables\n"
    if line not in text:
        text = text.rstrip() + "\n" + line + "\n"
    if note not in text:
        text = text.rstrip() + "\n" + note + "\n"
    INDEX_PATH.write_text(text, encoding="utf-8")


def _write_quality_report(target_audit: pd.DataFrame, quality_agg: pd.DataFrame, spatial_block_agg: pd.DataFrame, facility_summary: pd.DataFrame, match_quality: pd.DataFrame) -> None:
    def _metric(df: pd.DataFrame, model: str, col: str) -> float:
        if df.empty:
            return float("nan")
        sub = df[df["model"] == model]
        return float(sub[col].mean()) if not sub.empty and col in sub.columns else float("nan")

    q_auc = _metric(quality_agg, "Logit", "auc")
    q_brier = _metric(quality_agg, "Logit", "brier")
    q_aft = _metric(quality_agg, "AFT", "c_index")
    q_cox = _metric(quality_agg, "Cox", "c_index")
    q_ols = _metric(quality_agg, "OLS", "rmse")
    q_mx = _metric(quality_agg, "MixedLM", "rmse")
    sb_auc = _metric(spatial_block_agg, "Logit", "auc")
    sb_aft = _metric(spatial_block_agg, "AFT", "c_index")

    ols_row = facility_summary[facility_summary["model"] == "FacilityMatchedOLS"]
    logit_row = facility_summary[facility_summary["model"] == "FacilityMatchedLogit"]
    att_row = facility_summary[facility_summary["model"] == "FacilityPairedATT"]

    lines = [
        "# Quality + Matched Upgrade Report / 质量控制与设施匹配升级报告",
        "",
        "## Objective",
        "- 在不扩事件的前提下，优先提升 target/recovery 质量、空间约束和 buffer 对照可比性。",
        "",
        "## What Was Added",
        "- `delta_ntl_obs_adjusted`：低观测质量像素向事件高质量参考值收缩。",
        "- `recovery_daily_panel_v2.parquet`：显式记录 valid days、first valid day、first threshold hit 和 censoring risk。",
        "- `facility_context_panel_v1.parquet`：按最近关键设施生成局地 matched design。",
        "- `spatial_block_cv_metrics_v1.csv`：把空间依赖从诊断升级为 block-level 验证。",
        "",
        "## Target Quality Audit",
        f"- Mean high-censoring share: {target_audit['high_censoring_share'].mean():.3f}" if not target_audit.empty else "- Mean high-censoring share: NA",
        f"- Mean absolute delta adjustment: {target_audit['delta_shift_abs_mean'].mean():.4f}" if not target_audit.empty else "- Mean absolute delta adjustment: NA",
        f"- Worst observed-rate event: {target_audit.sort_values('observed_rate_v2').iloc[0]['event_id']} ({target_audit.sort_values('observed_rate_v2').iloc[0]['observed_rate_v2']:.3f})" if not target_audit.empty else "- Worst observed-rate event: NA",
        "",
        "## Appendix: Post-Hoc Quality Adjustment",
        f"- Logit AUC: {q_auc:.4f}",
        f"- Logit Brier: {q_brier:.4f}",
        f"- AFT c-index: {q_aft:.4f}",
        f"- Cox c-index: {q_cox:.4f}",
        f"- OLS RMSE: {q_ols:.4f}",
        f"- MixedLM RMSE: {q_mx:.4f}",
        "",
        "## Spatial Block Validation",
        f"- Logit AUC (block CV): {sb_auc:.4f}",
        f"- AFT c-index (block CV): {sb_aft:.4f}",
        "- 若 block CV 明显低于 LOEO，则说明原模型仍受空间近邻泄漏影响。",
        "",
        "## Facility-Centered Matched Results",
        f"- Matched OLS coef(in_buffer): {float(ols_row['value'].iloc[0]):.4f}, p={float(ols_row['p_value'].iloc[0]):.4g}" if not ols_row.empty else "- Matched OLS coef(in_buffer): NA",
        f"- Matched Logit OR(in_buffer): {float(logit_row['value'].iloc[0]):.4f}, p={float(logit_row['p_value'].iloc[0]):.4g}" if not logit_row.empty else "- Matched Logit OR(in_buffer): NA",
        f"- Paired ATT (treated-control delta_ntl): {float(att_row['value'].iloc[0]):.4f}" if not att_row.empty else "- Paired ATT: NA",
        f"- Mean matched pairs per facility: {match_quality['matched_pairs'].mean():.2f}" if not match_quality.empty else "- Mean matched pairs per facility: NA",
        "",
        "## Appendix Interpretation",
        "- 如果 quality-aware transport 比当前 v3r1 更稳，说明瓶颈的一部分来自 target/recovery 噪声。",
        "- 如果 matched 结果仍保持 buffer 正向信号，说明关键设施局地韧性并非完全由土地利用与城市密度混淆驱动。",
        "- 如果 spatial block CV 下分数明显回落，说明后续论文口径必须强调空间依赖修正。",
        "",
        "## Output Files",
        "- `project/modeling/output/target_quality_audit.csv`",
        "- `project/modeling/output/quality_transport_aggregate_metrics_v1.csv`",
        "- `project/modeling/output/spatial_block_cv_metrics_v1.csv`",
        "- `project/modeling/output/facility_centered_model_summary.csv`",
        "- `project/modeling/output/model_role_matrix_v1.csv`",
        "",
        "## Figure",
        "- `project/modeling_report/figures/exploration_v2/quality_matched_compare_v1.png`",
    ]
    QUALITY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_quality_matched_v1_impl() -> int:
    ensure_directories()
    init_tracking_files()
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(issues=[])

    append_progress("Quality+matched V1 started")
    panel = _filter_sample_lock(pd.read_parquet(PANEL_IN_PATH))

    append_progress("Quality+matched V1: build quality-adjusted targets and recovery v2")
    panel_q, rec_v2, target_audit = build_target_quality_panel(panel)

    append_progress("Quality+matched V1: LOEO quality-aware transport")
    quality_res = run_quality_transport(panel_q, rec_v2)

    append_progress("Quality+matched V1: spatial block CV")
    spatial_block = run_spatial_block_cv(panel_q, rec_v2)

    append_progress("Quality+matched V1: facility-centered matched design")
    fac_panel, match_quality = build_facility_context_panel(panel_q)
    facility_summary = fit_facility_centered_models(fac_panel)

    append_progress("Quality+matched V1: role matrix and report")
    role_matrix = build_model_role_matrix(quality_res.agg_df, spatial_block, facility_summary)
    _plot_quality_matched_summary(quality_res.agg_df, spatial_block)
    _write_quality_report(target_audit, quality_res.agg_df, spatial_block, facility_summary, match_quality)
    _update_quality_index()

    save_issue_log(ctx)
    append_progress("Quality+matched V1 completed")
    return 0


def attach_bug_prior_features(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    events_cfg = load_json(CONFIG_EVENTS)
    lookup = _load_bug_prior_lookup()
    default_row = {
        "bug_propensity_weight": 0.15,
        "night_use_weight": 0.25,
        "detectability_weight": 0.25,
        "capacity_prior_band": "low",
    }
    if BUG_PRIOR_CONFIG_PATH.exists():
        cfg = load_json(BUG_PRIOR_CONFIG_PATH)
        if isinstance(cfg, dict) and isinstance(cfg.get("default"), dict):
            default_row.update(cfg["default"])

    feature_defaults = {
        "bug_prior_count_750m": 0.0,
        "bug_prior_count_1250m": 0.0,
        "bug_prior_capacity_proxy_1km": 0.0,
        "bug_prior_hours_proxy_1km": 0.0,
        "bug_prior_min_dist_m": np.nan,
        "high_conf_bug_buffer": 0,
    }
    for col, default in feature_defaults.items():
        if col not in out.columns:
            out[col] = default

    audit_rows: List[Dict[str, object]] = []

    for event_id, cfg in events_cfg.items():
        mask = out["event_id"].astype(str) == str(event_id)
        if mask.sum() == 0:
            continue

        poi_path = ROOT / str(cfg["poi_csv"])
        if not poi_path.exists():
            out.loc[mask, "bug_prior_count_750m"] = 0.0
            out.loc[mask, "bug_prior_count_1250m"] = 0.0
            out.loc[mask, "bug_prior_capacity_proxy_1km"] = 0.0
            out.loc[mask, "bug_prior_hours_proxy_1km"] = 0.0
            out.loc[mask, "bug_prior_min_dist_m"] = np.nan
            out.loc[mask, "high_conf_bug_buffer"] = 0
            audit_rows.append(
                {
                    "event_id": event_id,
                    "n_obs": int(mask.sum()),
                    "poi_status": "missing_poi_csv",
                    "poi_count": 0,
                    "high_conf_poi_count": 0,
                    "high_conf_share": 0.0,
                    "bug_prior_count_750m_constant": 1,
                    "bug_prior_count_1250m_constant": 1,
                    "bug_prior_capacity_proxy_1km_constant": 1,
                    "bug_prior_hours_proxy_1km_constant": 1,
                    "high_conf_bug_buffer_share": 0.0,
                }
            )
            continue

        poi = pd.read_csv(poi_path).copy()
        if poi.empty:
            out.loc[mask, "bug_prior_count_750m"] = 0.0
            out.loc[mask, "bug_prior_count_1250m"] = 0.0
            out.loc[mask, "bug_prior_capacity_proxy_1km"] = 0.0
            out.loc[mask, "bug_prior_hours_proxy_1km"] = 0.0
            out.loc[mask, "bug_prior_min_dist_m"] = np.nan
            out.loc[mask, "high_conf_bug_buffer"] = 0
            audit_rows.append(
                {
                    "event_id": event_id,
                    "n_obs": int(mask.sum()),
                    "poi_status": "empty_poi_csv",
                    "poi_count": 0,
                    "high_conf_poi_count": 0,
                    "high_conf_share": 0.0,
                    "bug_prior_count_750m_constant": 1,
                    "bug_prior_count_1250m_constant": 1,
                    "bug_prior_capacity_proxy_1km_constant": 1,
                    "bug_prior_hours_proxy_1km_constant": 1,
                    "high_conf_bug_buffer_share": 0.0,
                }
            )
            continue

        poi["facility_type_std"] = poi["type"].astype(str).map(standardize_facility_type)
        poi = poi.merge(lookup, on="facility_type_std", how="left")
        for c, default in [
            ("bug_propensity_weight", float(default_row["bug_propensity_weight"])),
            ("night_use_weight", float(default_row["night_use_weight"])),
            ("detectability_weight", float(default_row["detectability_weight"])),
        ]:
            poi[c] = _safe_numeric(poi[c], default=default)
        poi["capacity_weight"] = _safe_numeric(poi.get("capacity_weight", pd.Series(dtype=float)), default=1.0)
        poi["capacity_prior_band"] = poi["capacity_prior_band"].fillna(str(default_row["capacity_prior_band"])).astype(str)
        poi["capacity_band_scalar"] = poi["capacity_prior_band"].map(_capacity_band_scalar)
        poi["bug_prior_weight"] = (
            _safe_numeric(poi["bug_propensity_weight"]) *
            _safe_numeric(poi["night_use_weight"]) *
            _safe_numeric(poi["detectability_weight"])
        )
        poi["bug_capacity_weight"] = poi["bug_prior_weight"] * _safe_numeric(poi["capacity_weight"], default=1.0) * _safe_numeric(poi["capacity_band_scalar"], default=1.0)
        poi["bug_hours_weight"] = poi["bug_prior_weight"] * _safe_numeric(poi["night_use_weight"], default=0.25)
        poi["buffer_radius_m"] = np.where(poi["facility_type_std"] == "aerodrome", BUFFER_RADII["aerodrome"], DEFAULT_BUFFER)
        poi["high_conf_bug_flag"] = (
            (poi["bug_propensity_weight"] >= 0.8) &
            (poi["night_use_weight"] >= 0.8) &
            (poi["detectability_weight"] >= 0.65)
        ).astype(int)

        sub = out.loc[mask].copy()
        transformer = Transformer.from_crs("EPSG:4326", str(cfg["metric_crs"]), always_xy=True)
        px, py = transformer.transform(sub["lon"].to_numpy(), sub["lat"].to_numpy())
        qx, qy = transformer.transform(poi["lon"].to_numpy(), poi["lat"].to_numpy())
        pix_xy = np.column_stack([px, py])
        poi_xy = np.column_stack([qx, qy])
        tree = cKDTree(poi_xy)

        idx_750 = tree.query_ball_point(pix_xy, r=750.0)
        idx_1000 = tree.query_ball_point(pix_xy, r=1000.0)
        idx_1250 = tree.query_ball_point(pix_xy, r=1250.0)

        prior_weight = poi["bug_prior_weight"].to_numpy(dtype=float)
        cap_weight = poi["bug_capacity_weight"].to_numpy(dtype=float)
        hour_weight = poi["bug_hours_weight"].to_numpy(dtype=float)

        def _weighted_sum(ix: Sequence[int], weights: np.ndarray) -> float:
            if not ix:
                return 0.0
            return float(weights[np.asarray(ix, dtype=int)].sum())

        sub["bug_prior_count_750m"] = [_weighted_sum(ix, prior_weight) for ix in idx_750]
        sub["bug_prior_count_1250m"] = [_weighted_sum(ix, prior_weight) for ix in idx_1250]
        sub["bug_prior_capacity_proxy_1km"] = [_weighted_sum(ix, cap_weight) for ix in idx_1000]
        sub["bug_prior_hours_proxy_1km"] = [_weighted_sum(ix, hour_weight) for ix in idx_1000]

        if len(poi_xy):
            dist_all, _ = tree.query(pix_xy, k=1)
            sub["bug_prior_min_dist_m"] = dist_all.astype(float)
        else:
            sub["bug_prior_min_dist_m"] = np.nan

        high_conf = poi[poi["high_conf_bug_flag"] == 1].copy()
        if not high_conf.empty:
            high_xy = np.column_stack([qx[high_conf.index.to_numpy()], qy[high_conf.index.to_numpy()]])
            high_tree = cKDTree(high_xy)
            dist_high, idx_high = high_tree.query(pix_xy, k=1)
            high_radii = high_conf["buffer_radius_m"].to_numpy(dtype=float)
            sub["high_conf_bug_buffer"] = (dist_high.astype(float) <= high_radii[np.asarray(idx_high, dtype=int)]).astype(int)
        else:
            sub["high_conf_bug_buffer"] = 0

        out.loc[mask, "bug_prior_count_750m"] = sub["bug_prior_count_750m"].to_numpy(dtype=float)
        out.loc[mask, "bug_prior_count_1250m"] = sub["bug_prior_count_1250m"].to_numpy(dtype=float)
        out.loc[mask, "bug_prior_capacity_proxy_1km"] = sub["bug_prior_capacity_proxy_1km"].to_numpy(dtype=float)
        out.loc[mask, "bug_prior_hours_proxy_1km"] = sub["bug_prior_hours_proxy_1km"].to_numpy(dtype=float)
        out.loc[mask, "bug_prior_min_dist_m"] = sub["bug_prior_min_dist_m"].to_numpy(dtype=float)
        out.loc[mask, "high_conf_bug_buffer"] = sub["high_conf_bug_buffer"].to_numpy(dtype=int)

        audit_rows.append(
            {
                "event_id": event_id,
                "n_obs": int(len(sub)),
                "poi_status": "ok",
                "poi_count": int(len(poi)),
                "high_conf_poi_count": int(high_conf.shape[0]),
                "high_conf_share": float(high_conf.shape[0] / len(poi)) if len(poi) else 0.0,
                "bug_prior_count_750m_constant": int(sub["bug_prior_count_750m"].nunique(dropna=True) <= 1),
                "bug_prior_count_1250m_constant": int(sub["bug_prior_count_1250m"].nunique(dropna=True) <= 1),
                "bug_prior_capacity_proxy_1km_constant": int(sub["bug_prior_capacity_proxy_1km"].nunique(dropna=True) <= 1),
                "bug_prior_hours_proxy_1km_constant": int(sub["bug_prior_hours_proxy_1km"].nunique(dropna=True) <= 1),
                "high_conf_bug_buffer_share": float(sub["high_conf_bug_buffer"].mean()),
            }
        )

    out["high_conf_bug_buffer"] = _safe_numeric(out["high_conf_bug_buffer"]).astype(int)
    out["bug_prior_min_dist_m"] = _safe_numeric(out["bug_prior_min_dist_m"], default=5000.0)
    out.to_parquet(BUG_TRANSPORT_PANEL_PATH, index=False)
    audit = pd.DataFrame(audit_rows).sort_values("event_id") if audit_rows else pd.DataFrame()
    audit.to_csv(BUG_TRANSPORT_FEATURE_AUDIT_PATH, index=False)
    return out, audit


def _merge_bug_features_into_recovery(panel: pd.DataFrame, recovery: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [
        "pixel_id",
        "event_id",
        "high_conf_bug_buffer",
        "official_bug_count_1km",
        "official_bug_kw_sum_1km",
        "official_bug_hours_proxy_1km",
        "official_bug_min_dist_m",
        "official_bug_coverage_flag",
    ] + BUG_PRIOR_NUMERIC
    merge_cols = [c for c in merge_cols if c in panel.columns]
    rec = recovery.drop(columns=[c for c in merge_cols if c in recovery.columns and c not in {"pixel_id", "event_id"}], errors="ignore")
    return rec.merge(panel[merge_cols].drop_duplicates(subset=["pixel_id"]), on=["pixel_id", "event_id"], how="left")


def run_bug_aware_transport(panel: pd.DataFrame, recovery: pd.DataFrame) -> SpecResult:
    specs: List[Tuple[str, pd.DataFrame, pd.DataFrame, List[str]]] = []

    specs.append(("BUG0", panel.copy(), recovery.copy(), BASE_NUMERIC + QUALITY_GUARD_NUMERIC))
    specs.append(("BUG1A", panel.copy(), recovery.copy(), BASE_NUMERIC + QUALITY_GUARD_NUMERIC + BUG_PRIOR_NUMERIC))

    panel_b = panel.copy()
    rec_b = recovery.copy()
    panel_b["legacy_in_buffer"] = panel_b["in_buffer"]
    panel_b["in_buffer"] = _safe_numeric(panel_b["high_conf_bug_buffer"]).astype(int)
    rec_b["legacy_in_buffer"] = rec_b["in_buffer"]
    rec_b["in_buffer"] = _safe_numeric(rec_b["high_conf_bug_buffer"]).astype(int)
    specs.append(("BUG1B", panel_b, rec_b, BASE_NUMERIC + QUALITY_GUARD_NUMERIC + BUG_PRIOR_NUMERIC))

    panel_c = panel.copy()
    rec_c = recovery.copy()
    panel_c["legacy_in_buffer"] = panel_c["in_buffer"]
    panel_c["in_buffer"] = _safe_numeric(panel_c["high_conf_bug_buffer"]).astype(int)
    rec_c["legacy_in_buffer"] = rec_c["in_buffer"]
    rec_c["in_buffer"] = _safe_numeric(rec_c["high_conf_bug_buffer"]).astype(int)
    specs.append(("BUG1C", panel_c, rec_c, QUALITY_GUARD_NUMERIC + BUG_PRIOR_NUMERIC))

    fold_parts: List[pd.DataFrame] = []
    agg_parts: List[pd.DataFrame] = []
    coef_parts: List[pd.DataFrame] = []
    for spec_id, spec_panel, spec_recovery, numeric_terms in specs:
        res = run_loeo_spec(
            spec_panel,
            spec_recovery,
            numeric_terms=numeric_terms,
            cat_terms=["land_use_group", "event_disaster_type"],
            experiment_family="bug_transport",
            spec_id=spec_id,
        )
        fold_parts.append(res.fold_df)
        agg_parts.append(res.agg_df)
        coef_parts.append(res.coef_df)

    fold_df = pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()
    agg_df = pd.concat(agg_parts, ignore_index=True) if agg_parts else pd.DataFrame()
    coef_df = pd.concat(coef_parts, ignore_index=True) if coef_parts else pd.DataFrame()
    fold_df.to_csv(BUG_TRANSPORT_FOLD_PATH, index=False)
    agg_df.to_csv(BUG_TRANSPORT_AGG_PATH, index=False)
    return SpecResult(fold_df=fold_df, agg_df=agg_df, coef_df=coef_df)


def summarize_bug_transport_features(coef_df: pd.DataFrame) -> pd.DataFrame:
    if coef_df.empty:
        out = pd.DataFrame(columns=["spec_id", "model", "feature", "mean_coef", "mean_abs_coef", "sign_consistency", "folds"])
        out.to_csv(BUG_TRANSPORT_FEATURE_SUMMARY_PATH, index=False)
        return out

    tracked = set(BUG_PRIOR_NUMERIC) | {"in_buffer", "high_conf_bug_buffer"}
    sub = coef_df[coef_df["feature"].isin(tracked)].copy()
    if sub.empty:
        out = pd.DataFrame(columns=["spec_id", "model", "feature", "mean_coef", "mean_abs_coef", "sign_consistency", "folds"])
        out.to_csv(BUG_TRANSPORT_FEATURE_SUMMARY_PATH, index=False)
        return out

    rows = []
    for (spec_id, model, feature), grp in sub.groupby(["spec_id", "model", "feature"], dropna=False):
        coef = pd.to_numeric(grp["coef"], errors="coerce").dropna()
        if coef.empty:
            continue
        pos = float((coef > 0).mean())
        neg = float((coef < 0).mean())
        rows.append(
            {
                "spec_id": spec_id,
                "model": model,
                "feature": feature,
                "mean_coef": float(coef.mean()),
                "mean_abs_coef": float(coef.abs().mean()),
                "sign_consistency": max(pos, neg),
                "folds": int(coef.shape[0]),
            }
        )
    out = pd.DataFrame(rows).sort_values(["spec_id", "model", "mean_abs_coef"], ascending=[True, True, False])
    out.to_csv(BUG_TRANSPORT_FEATURE_SUMMARY_PATH, index=False)
    return out


def attach_hazard_exposure_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if not EVENT_PROFILE_V1_PATH.exists():
        raise FileNotFoundError(f"Missing event profile: {EVENT_PROFILE_V1_PATH}")

    prof = pd.read_csv(EVENT_PROFILE_V1_PATH).copy()
    prof["event_disaster_type"] = prof["disaster_type"].fillna("unknown").astype(str)
    prof["event_island_like_flag"] = _safe_numeric(prof["island_like_flag"])
    prof["event_cloud_shift"] = _safe_numeric(prof["cloud_post_event_mean"]) - _safe_numeric(prof["cloud_pre_event_mean"])
    prof["event_precip_log1p"] = np.log1p(_safe_numeric(prof["storm_precip_7d"]).clip(lower=0))
    prof["event_duration_log1p"] = np.log1p(_safe_numeric(prof["event_duration_days"]).clip(lower=0))
    prof["event_elevation_log1p"] = np.log1p(_safe_numeric(prof["elevation_median"]).clip(lower=0))
    prof["event_slope_milli"] = _safe_numeric(prof["slope_median"]).clip(lower=0) * 1000.0
    prof["event_urban_context"] = _safe_numeric(prof["urban_share_1km"])
    prof["event_water_context"] = _safe_numeric(prof["water_share_1km"])

    keep = [
        "event_id",
        "event_disaster_type",
        "event_island_like_flag",
        "event_cloud_shift",
        "event_precip_log1p",
        "event_duration_log1p",
        "event_elevation_log1p",
        "event_slope_milli",
        "event_urban_context",
        "event_water_context",
    ]
    out = out.drop(columns=[c for c in keep if c in out.columns and c != "event_id"], errors="ignore")
    out = out.merge(prof[keep], on="event_id", how="left")

    out["event_disaster_type"] = out["event_disaster_type"].fillna("unknown").astype(str)
    for c in [
        "event_island_like_flag",
        "event_cloud_shift",
        "event_precip_log1p",
        "event_duration_log1p",
        "event_elevation_log1p",
        "event_slope_milli",
        "event_urban_context",
        "event_water_context",
    ]:
        out[c] = _safe_numeric(out[c])

    out["island_local_water"] = out["event_island_like_flag"] * _safe_numeric(out["water_share_1km"])
    out["island_local_urban"] = out["event_island_like_flag"] * _safe_numeric(out["urban_share_1km"])
    out["hazard_cloud_water"] = out["event_cloud_shift"] * _safe_numeric(out["water_share_1km"])
    out["hazard_precip_urban"] = out["event_precip_log1p"] * _safe_numeric(out["urban_share_1km"])
    out.to_parquet(PANEL_HAZARD_PATH, index=False)
    return out


def run_hazard_aware_transport(
    panel: pd.DataFrame,
    recovery: pd.DataFrame,
    spec_id: str = "HZ1",
    experiment_family: str = "hazard_transport",
    fold_path: Path = HAZARD_TRANSPORT_FOLD_PATH,
    agg_path: Path = HAZARD_TRANSPORT_AGG_PATH,
) -> SpecResult:
    numeric_terms = BASE_NUMERIC + [
        "pixel_cloud_proxy",
        "recovery_obs_quality_score",
        "event_cloud_shift",
        "event_precip_log1p",
        "event_duration_log1p",
        "event_elevation_log1p",
        "event_slope_milli",
        "island_local_water",
        "island_local_urban",
        "hazard_cloud_water",
        "hazard_precip_urban",
    ]
    cat_terms = ["land_use_group", "event_disaster_type"]
    res = run_loeo_spec(
        panel,
        recovery,
        numeric_terms=numeric_terms,
        cat_terms=cat_terms,
        experiment_family=experiment_family,
        spec_id=spec_id,
    )
    res.fold_df.to_csv(fold_path, index=False)
    res.agg_df.to_csv(agg_path, index=False)
    return res


def summarize_hazard_features(coef_df: pd.DataFrame, output_path: Path = HAZARD_FEATURE_SUMMARY_PATH) -> pd.DataFrame:
    if coef_df.empty:
        out = pd.DataFrame(columns=["model", "feature", "mean_coef", "mean_abs_coef", "sign_consistency", "folds"])
        out.to_csv(output_path, index=False)
        return out

    hazard_features = {
        "event_cloud_shift",
        "event_precip_log1p",
        "event_duration_log1p",
        "event_elevation_log1p",
        "event_slope_milli",
        "island_local_water",
        "island_local_urban",
        "hazard_cloud_water",
        "hazard_precip_urban",
    }
    sub = coef_df[coef_df["feature"].isin(hazard_features)].copy()
    if sub.empty:
        out = pd.DataFrame(columns=["model", "feature", "mean_coef", "mean_abs_coef", "sign_consistency", "folds"])
        out.to_csv(output_path, index=False)
        return out

    rows = []
    for (model, feature), grp in sub.groupby(["model", "feature"], dropna=False):
        coef = pd.to_numeric(grp["coef"], errors="coerce").dropna()
        if coef.empty:
            continue
        pos = float((coef > 0).mean())
        neg = float((coef < 0).mean())
        rows.append(
            {
                "model": model,
                "feature": feature,
                "mean_coef": float(coef.mean()),
                "mean_abs_coef": float(coef.abs().mean()),
                "sign_consistency": max(pos, neg),
                "folds": int(coef.shape[0]),
            }
        )
    out = pd.DataFrame(rows).sort_values(["model", "mean_abs_coef"], ascending=[True, False])
    out.to_csv(output_path, index=False)
    return out


def _prepare_hazard_transport_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if PANEL_QUALITY_PATH.exists() and RECOVERY_V2_PATH.exists():
        panel_q = pd.read_parquet(PANEL_QUALITY_PATH)
        rec_v2 = pd.read_parquet(RECOVERY_V2_PATH)
        target_audit = pd.read_csv(TARGET_QUALITY_AUDIT_PATH) if TARGET_QUALITY_AUDIT_PATH.exists() else pd.DataFrame()
    else:
        panel = _filter_sample_lock(pd.read_parquet(PANEL_IN_PATH))
        panel_q, rec_v2, target_audit = build_target_quality_panel(panel)

    panel_h = attach_hazard_exposure_features(panel_q)
    merge_cols = [
        c
        for c in panel_h.columns
        if c.startswith("event_") or c in ["island_local_water", "island_local_urban", "hazard_cloud_water", "hazard_precip_urban"]
    ]
    merge_cols = ["pixel_id", "event_id"] + [c for c in merge_cols if c not in {"pixel_id", "event_id"}]
    rec_h = rec_v2.drop(columns=[c for c in merge_cols if c in rec_v2.columns and c not in {"pixel_id", "event_id"}], errors="ignore")
    rec_h = rec_h.merge(panel_h[merge_cols].drop_duplicates(subset=["pixel_id"]), on=["pixel_id", "event_id"], how="left")
    return panel_h, rec_h, target_audit


def _filter_event_allowlist(
    panel: pd.DataFrame,
    recovery: pd.DataFrame,
    allow_events: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    allow = {str(v) for v in allow_events}
    panel_sub = panel[panel["event_id"].astype(str).isin(allow)].copy()
    recovery_sub = recovery[recovery["event_id"].astype(str).isin(allow)].copy()
    return panel_sub, recovery_sub


def _load_bug_inventory_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported inventory file type: {path}")


def canonicalize_bug_inventory(df: pd.DataFrame, pilot_cfg: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "facility_type": "facility_type_raw",
        "facility_type_clean": "facility_type_std",
        "generator_type": "facility_type_raw",
        "EQUIPMENT_TYPE": "facility_type_raw",
        "PERMIT_DESCRIPTION": "facility_type_raw",
        "capacity": "capacity_kw",
        "kw": "capacity_kw",
        "Capacity listed (kW)": "capacity_kw",
        "Total Capacity (kW)": "capacity_kw",
        "operating_hours": "operating_hours_annual",
        "hours": "operating_hours_annual",
        "LAST READING - FIRST READING": "operating_hours_annual",
        "address": "address_raw",
        "address_line": "address_raw",
        "EQUIP_LOCATION_ADDRESS": "address_raw",
        "latitude": "lat",
        "longitude": "lon",
        "Latitude": "lat",
        "Longitude": "lon",
        "DBA": "facility_name",
        "Regulated Entity Name": "facility_name",
        "Fuel": "fuel_type",
        "agency_url": "source_url",
        "Source": "source_url",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    for col in _bug2_required_columns():
        if col not in out.columns:
            out[col] = np.nan
    out["source_dataset"] = out["source_dataset"].fillna(str(pilot_cfg.get("source_dataset", str(pilot_cfg.get("pilot_state", "PR")).lower() + "_pilot")))
    out["jurisdiction"] = out["jurisdiction"].fillna(str(pilot_cfg.get("jurisdiction", pilot_cfg.get("pilot_state", "PR"))))
    out["state"] = out["state"].fillna(str(pilot_cfg.get("pilot_state", "PR")))
    out["county_or_district"] = out["county_or_district"].fillna(str(pilot_cfg.get("county_or_district", "")))
    out["facility_type_raw"] = out["facility_type_raw"].fillna(out["facility_type_std"]).fillna("unknown").astype(str)
    mask_std = out["facility_type_std"].isna() | (out["facility_type_std"].astype(str).str.strip() == "")
    if mask_std.any():
        out.loc[mask_std, "facility_type_std"] = out.loc[mask_std, "facility_type_raw"].map(standardize_facility_type)
    out["facility_type_std"] = out["facility_type_std"].fillna("other").astype(str)
    out["record_id"] = out["record_id"].fillna(out.index.astype(str)).astype(str)
    out["facility_name"] = out["facility_name"].fillna("unknown_facility").astype(str)
    out["fuel_type"] = out["fuel_type"].fillna("unknown").astype(str)
    out["address_raw"] = out["address_raw"].fillna("").astype(str)
    out["source_url"] = out["source_url"].fillna(str(pilot_cfg.get("source_url", ""))).astype(str)
    out["capacity_kw"] = _safe_numeric(out["capacity_kw"])
    out["operating_hours_annual"] = _safe_numeric(out["operating_hours_annual"])
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["geo_quality_flag"] = out["geo_quality_flag"].fillna(np.where(np.isfinite(out["lat"]) & np.isfinite(out["lon"]), "coords_present", "missing_coords")).astype(str)
    out["attribute_quality_flag"] = out["attribute_quality_flag"].fillna(
        np.where((out["facility_type_std"] != "other") | (out["capacity_kw"] > 0) | (out["operating_hours_annual"] > 0), "usable", "sparse")
    ).astype(str)
    keep = _bug2_required_columns()
    return out[keep].copy()


def audit_bug_inventory(df: pd.DataFrame, pilot_cfg: Dict[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    pilot_events = [str(v) for v in pilot_cfg.get("pilot_event_ids", [])]
    geo_cov = float((np.isfinite(df["lat"]) & np.isfinite(df["lon"])).mean()) if not df.empty else 0.0
    attr_cov = float(((df["capacity_kw"] > 0) | (df["operating_hours_annual"] > 0) | (df["facility_type_std"] != "other")).mean()) if not df.empty else 0.0
    rows.append(
        {
            "pilot_state": str(pilot_cfg.get("pilot_state", "PR")),
            "pilot_event_ids": ";".join(pilot_events),
            "records_n": int(df.shape[0]),
            "geo_coverage": geo_cov,
            "attribute_coverage": attr_cov,
            "capacity_nonzero_share": float((df["capacity_kw"] > 0).mean()) if not df.empty else 0.0,
            "hours_nonzero_share": float((df["operating_hours_annual"] > 0).mean()) if not df.empty else 0.0,
            "distinct_facility_types": int(df["facility_type_std"].astype(str).nunique()) if not df.empty else 0,
            "duplicate_record_share": float(df["record_id"].astype(str).duplicated().mean()) if not df.empty else 0.0,
            "gate_pass": int(
                (df.shape[0] >= int(pilot_cfg["coverage_gate"]["min_records"])) and
                (geo_cov >= float(pilot_cfg["coverage_gate"]["min_geo_coverage"]))
            ),
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(BUG2_QA_PATH, index=False)
    return out


def attach_official_bug_features(
    panel: pd.DataFrame,
    inventory: pd.DataFrame,
    pilot_cfg: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    feature_defaults = {
        "official_bug_count_1km": 0.0,
        "official_bug_kw_sum_1km": 0.0,
        "official_bug_hours_proxy_1km": 0.0,
        "official_bug_min_dist_m": np.nan,
        "official_bug_coverage_flag": 0,
    }
    for col, default in feature_defaults.items():
        if col not in out.columns:
            out[col] = default

    audit_rows: List[Dict[str, object]] = []
    events_cfg = load_json(CONFIG_EVENTS)
    pilot_events = {str(v) for v in pilot_cfg.get("pilot_event_ids", [])}
    inv = inventory[np.isfinite(inventory["lat"]) & np.isfinite(inventory["lon"])].copy()

    for event_id, cfg in events_cfg.items():
        mask = out["event_id"].astype(str) == str(event_id)
        if mask.sum() == 0:
            continue
        if event_id not in pilot_events or inv.empty:
            out.loc[mask, "official_bug_coverage_flag"] = 0
            audit_rows.append({"event_id": event_id, "inventory_records": 0, "coverage_flag": 0, "feature_nonzero_share": 0.0})
            continue

        transformer = Transformer.from_crs("EPSG:4326", str(cfg["metric_crs"]), always_xy=True)
        sub = out.loc[mask].copy()
        px, py = transformer.transform(sub["lon"].to_numpy(), sub["lat"].to_numpy())
        qx, qy = transformer.transform(inv["lon"].to_numpy(), inv["lat"].to_numpy())
        pix_xy = np.column_stack([px, py])
        inv_xy = np.column_stack([qx, qy])
        tree = cKDTree(inv_xy)
        idx_1000 = tree.query_ball_point(pix_xy, r=1000.0)
        kw = inv["capacity_kw"].to_numpy(dtype=float)
        hrs = inv["operating_hours_annual"].to_numpy(dtype=float)

        def _sum(ix: Sequence[int], arr: np.ndarray) -> float:
            if not ix:
                return 0.0
            return float(arr[np.asarray(ix, dtype=int)].sum())

        sub["official_bug_count_1km"] = [float(len(ix)) for ix in idx_1000]
        sub["official_bug_kw_sum_1km"] = [_sum(ix, kw) for ix in idx_1000]
        sub["official_bug_hours_proxy_1km"] = [_sum(ix, hrs) for ix in idx_1000]
        if len(inv_xy):
            dist_all, _ = tree.query(pix_xy, k=1)
            sub["official_bug_min_dist_m"] = dist_all.astype(float)
        else:
            sub["official_bug_min_dist_m"] = np.nan
        sub["official_bug_coverage_flag"] = 1

        out.loc[mask, "official_bug_count_1km"] = sub["official_bug_count_1km"].to_numpy(dtype=float)
        out.loc[mask, "official_bug_kw_sum_1km"] = sub["official_bug_kw_sum_1km"].to_numpy(dtype=float)
        out.loc[mask, "official_bug_hours_proxy_1km"] = sub["official_bug_hours_proxy_1km"].to_numpy(dtype=float)
        out.loc[mask, "official_bug_min_dist_m"] = sub["official_bug_min_dist_m"].to_numpy(dtype=float)
        out.loc[mask, "official_bug_coverage_flag"] = 1
        audit_rows.append(
            {
                "event_id": event_id,
                "inventory_records": int(inv.shape[0]),
                "coverage_flag": 1,
                "feature_nonzero_share": float((sub["official_bug_count_1km"] > 0).mean()),
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(BUG2_FEATURE_AUDIT_PATH, index=False)
    return out, audit


def build_event_selection_scorecard(
    hazard_fold: pd.DataFrame,
    target_audit: pd.DataFrame,
) -> pd.DataFrame:
    prof = pd.read_csv(EVENT_PROFILE_V1_PATH).copy()
    shift = pd.read_csv(SHIFT_V3_PATH) if SHIFT_V3_PATH.exists() else pd.DataFrame()

    logit = hazard_fold[hazard_fold["model"] == "Logit"][["fold_event", "auc", "brier"]].rename(
        columns={"fold_event": "event_id", "auc": "logit_auc_hz", "brier": "logit_brier_hz"}
    )
    cox = hazard_fold[hazard_fold["model"] == "Cox"][["fold_event", "c_index"]].rename(
        columns={"fold_event": "event_id", "c_index": "cox_c_index_hz"}
    )
    aft = hazard_fold[hazard_fold["model"] == "AFT"][["fold_event", "c_index"]].rename(
        columns={"fold_event": "event_id", "c_index": "aft_c_index_hz"}
    )

    out = prof.merge(logit, on="event_id", how="left").merge(cox, on="event_id", how="left").merge(aft, on="event_id", how="left")
    out["survival_best_hz"] = out[["cox_c_index_hz", "aft_c_index_hz"]].max(axis=1)
    if not target_audit.empty:
        out = out.merge(target_audit[["event_id", "observed_rate_v2", "high_censoring_share"]], on="event_id", how="left")
    if not shift.empty and "event_id" in shift.columns:
        shift_keep = [c for c in ["event_id", "smd_mean", "psi_mean"] if c in shift.columns]
        shift_small = shift[shift_keep].copy()
        shift_small = shift_small.groupby("event_id", as_index=False).mean(numeric_only=True)
        out = out.merge(shift_small, on="event_id", how="left")

    out["urban_bin"] = pd.cut(
        _safe_numeric(out["urban_share_1km"]),
        bins=[-np.inf, 0.68, 0.75, np.inf],
        labels=["low_urban", "mid_urban", "high_urban"],
    ).astype(str)
    out["water_bin"] = pd.cut(
        _safe_numeric(out["water_share_1km"]),
        bins=[-np.inf, 0.10, 0.15, np.inf],
        labels=["low_water", "mid_water", "high_water"],
    ).astype(str)

    disaster_counts = out["disaster_type"].value_counts(dropna=False).to_dict()
    urban_counts = out["urban_bin"].value_counts(dropna=False).to_dict()
    island_counts = out["island_like_flag"].value_counts(dropna=False).to_dict()

    recs = []
    for row in out.itertuples(index=False):
        reasons: List[str] = []
        if disaster_counts.get(row.disaster_type, 0) <= 1:
            reasons.append(f"add_more_{row.disaster_type}")
        if row.island_like_flag == 1 and island_counts.get(1, 0) <= 2:
            reasons.append("add_non_sanjuan_island_like_event")
        if urban_counts.get(row.urban_bin, 0) <= 1:
            reasons.append(f"add_more_{row.urban_bin}_events")
        if pd.notna(row.logit_auc_hz) and row.logit_auc_hz < 0.45:
            reasons.append("poor_damage_transport_holdout")
        if pd.notna(row.survival_best_hz) and row.survival_best_hz < 0.50:
            reasons.append("poor_recovery_transport_holdout")
        if pd.notna(getattr(row, "observed_rate_v2", np.nan)) and row.observed_rate_v2 < 0.80:
            reasons.append("low_observation_quality_neighbor_needed")
        recs.append(";".join(reasons) if reasons else "representative_keep")

    out["selection_signal"] = recs
    keep_cols = [
        "event_id",
        "disaster_type",
        "island_like_flag",
        "urban_bin",
        "water_bin",
        "storm_precip_7d",
        "event_duration_days",
        "logit_auc_hz",
        "survival_best_hz",
        "observed_rate_v2",
        "smd_mean",
        "psi_mean",
        "selection_signal",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].drop_duplicates(subset=["event_id"]).sort_values(["logit_auc_hz", "survival_best_hz"], na_position="last")
    out.to_csv(EVENT_SELECTION_PATH, index=False)
    return out


def _plot_hazard_transport_summary(hazard_agg: pd.DataFrame) -> None:
    q = pd.read_csv(QUALITY_TRANSPORT_AGG_PATH) if QUALITY_TRANSPORT_AGG_PATH.exists() else pd.DataFrame()
    base = pd.read_csv(OUTPUT_DIR / "model_summary_cross_event_v3r1.csv") if (OUTPUT_DIR / "model_summary_cross_event_v3r1.csv").exists() else pd.DataFrame()
    base_map = {r["metric_name"]: r["value"] for _, r in base.iterrows()} if not base.empty else {}
    hazard_logit = float(hazard_agg.loc[hazard_agg["model"] == "Logit", "auc"].mean()) if not hazard_agg.empty else np.nan
    hazard_surv = float(hazard_agg.loc[hazard_agg["model"].isin(["Cox", "AFT"]), "c_index"].max()) if not hazard_agg.empty else np.nan
    quality_logit = float(q.loc[q["model"] == "Logit", "auc"].mean()) if not q.empty else np.nan
    quality_surv = float(q.loc[q["model"].isin(["Cox", "AFT"]), "c_index"].max()) if not q.empty else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].bar(
        ["v3r1", "quality", "hazard"],
        [base_map.get("logit_auc", np.nan), quality_logit, hazard_logit],
        color=["#8c8c8c", "#2c7fb8", "#1b9e77"],
    )
    axes[0].set_title("Transport Logit AUC")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].bar(
        ["v3r1", "quality", "hazard"],
        [base_map.get("survival_best_c_index", np.nan), quality_surv, hazard_surv],
        color=["#8c8c8c", "#2c7fb8", "#1b9e77"],
    )
    axes[1].set_title("Transport Survival Best c-index")
    axes[1].set_ylim(0.0, 1.0)
    plt.tight_layout()
    fig.savefig(HAZARD_FIG_PATH, dpi=220)
    plt.close(fig)


def _write_hazard_transport_report(
    hazard_agg: pd.DataFrame,
    feature_summary: pd.DataFrame,
    selection_scorecard: pd.DataFrame,
) -> None:
    base = pd.read_csv(OUTPUT_DIR / "model_summary_cross_event_v3r1.csv") if (OUTPUT_DIR / "model_summary_cross_event_v3r1.csv").exists() else pd.DataFrame()
    quality = pd.read_csv(QUALITY_TRANSPORT_AGG_PATH) if QUALITY_TRANSPORT_AGG_PATH.exists() else pd.DataFrame()
    base_map = {r["metric_name"]: r["value"] for _, r in base.iterrows()} if not base.empty else {}

    hazard_logit = float(hazard_agg.loc[hazard_agg["model"] == "Logit", "auc"].mean()) if not hazard_agg.empty else np.nan
    hazard_brier = float(hazard_agg.loc[hazard_agg["model"] == "Logit", "brier"].mean()) if not hazard_agg.empty else np.nan
    hazard_surv = float(hazard_agg.loc[hazard_agg["model"].isin(["Cox", "AFT"]), "c_index"].max()) if not hazard_agg.empty else np.nan
    quality_logit = float(quality.loc[quality["model"] == "Logit", "auc"].mean()) if not quality.empty else np.nan
    quality_surv = float(quality.loc[quality["model"].isin(["Cox", "AFT"]), "c_index"].max()) if not quality.empty else np.nan

    top_logit = feature_summary[feature_summary["model"] == "Logit"].head(5)
    weak_events = selection_scorecard.head(3) if not selection_scorecard.empty else pd.DataFrame()

    lines = [
        "# Hazard/Exposure Transport Report / 灾害强度与暴露特征主线报告",
        "",
        "## Objective",
        "- 在 quality-adjusted target 基础上，加入低维 hazard/exposure 特征，重做跨事件 transport 主线。",
        "",
        "## Added Mainline Features",
        "- `event_cloud_shift`",
        "- `event_precip_log1p`",
        "- `event_duration_log1p`",
        "- `event_elevation_log1p`",
        "- `event_slope_milli`",
        "- `island_local_water`",
        "- `island_local_urban`",
        "- `hazard_cloud_water`",
        "- `hazard_precip_urban`",
        "",
        "## Metric Comparison",
        f"- v3r1 Logit AUC: {base_map.get('logit_auc', np.nan):.4f}",
        f"- quality Logit AUC: {quality_logit:.4f}",
        f"- hazard Logit AUC: {hazard_logit:.4f}",
        f"- v3r1 survival best: {base_map.get('survival_best_c_index', np.nan):.4f}",
        f"- quality survival best: {quality_surv:.4f}",
        f"- hazard survival best: {hazard_surv:.4f}",
        f"- hazard Logit Brier: {hazard_brier:.4f}",
        "",
        "## Top Hazard Features (Logit)",
    ]
    if top_logit.empty:
        lines.append("- NA")
    else:
        for _, r in top_logit.iterrows():
            lines.append(
                f"- {r['feature']}: mean_coef={r['mean_coef']:.4f}, abs={r['mean_abs_coef']:.4f}, sign_consistency={r['sign_consistency']:.2f}"
            )

    lines.extend(["", "## Event Selection Signals"])
    if weak_events.empty:
        lines.append("- NA")
    else:
        for _, r in weak_events.iterrows():
            lines.append(
                f"- {r['event_id']}: damage_auc={r.get('logit_auc_hz', np.nan):.4f}, survival_best={r.get('survival_best_hz', np.nan):.4f}, signal={r.get('selection_signal', 'NA')}"
            )

    lines.extend(
        [
            "",
            "## Recommendation",
            "- 本轮 hazard/exposure 主线显著提升了 damage ranking（AUC），说明事件级暴露差异确实是当前跨事件主线缺失的信息。",
            "- 但 hazard Logit 的 Brier 明显变差，说明它更会排序、但概率更不稳；后续若用于预测，应加校准或减弱事件级强特征。",
            "- 后续扩事件时，应优先补足灾种、岛屿性、城市层级和中等水域暴露这几个维度。",
            "",
            "## Outputs",
            "- `project/modeling/output/hazard_transport_aggregate_metrics_v1.csv`",
            "- `project/modeling/output/hazard_transport_feature_summary_v1.csv`",
            "- `project/modeling/output/event_selection_scorecard_v1.csv`",
            "- `project/modeling_report/figures/exploration_v2/hazard_transport_compare_v1.png`",
        ]
    )
    HAZARD_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_hazard_index() -> None:
    line = "- `project/modeling_report/12_hazard_exposure_transport_report.md`"
    note = "- Appendix hazard/exposure outputs: `project/modeling/output/hazard_transport_aggregate_metrics_v1.csv`, `project/modeling/output/event_selection_scorecard_v1.csv`"
    text = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else "# Modeling Report Index\n\n## Deliverables\n"
    if line not in text:
        text = text.rstrip() + "\n" + line + "\n"
    if note not in text:
        text = text.rstrip() + "\n" + note + "\n"
    INDEX_PATH.write_text(text, encoding="utf-8")


def _plot_hazard_readiness_summary(hazard_ready_agg: pd.DataFrame) -> None:
    hazard = pd.read_csv(HAZARD_TRANSPORT_AGG_PATH) if HAZARD_TRANSPORT_AGG_PATH.exists() else pd.DataFrame()
    quality = pd.read_csv(QUALITY_TRANSPORT_AGG_PATH) if QUALITY_TRANSPORT_AGG_PATH.exists() else pd.DataFrame()

    def _metric(df: pd.DataFrame, spec_id: str, model: str, col: str) -> float:
        if df.empty:
            return np.nan
        if "spec_id" in df.columns:
            sub = df[(df["spec_id"] == spec_id) & (df["model"] == model)]
        else:
            sub = df[df["model"] == model]
        return float(pd.to_numeric(sub[col], errors="coerce").mean()) if not sub.empty and col in sub.columns else np.nan

    labels = ["QT1", "HZ1", "HZ1_READY"]
    auc_vals = [
        _metric(quality, "QT1", "Logit", "auc"),
        _metric(hazard, "HZ1", "Logit", "auc"),
        _metric(hazard_ready_agg, "HZ1_READY", "Logit", "auc"),
    ]
    brier_vals = [
        _metric(quality, "QT1", "Logit", "brier"),
        _metric(hazard, "HZ1", "Logit", "brier"),
        _metric(hazard_ready_agg, "HZ1_READY", "Logit", "brier"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    axes[0].bar(labels, auc_vals, color=["#8c8c8c", "#1b9e77", "#4c78a8"])
    axes[0].set_title("Damage Transport Logit AUC")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].bar(labels, brier_vals, color=["#8c8c8c", "#1b9e77", "#4c78a8"])
    axes[1].set_title("Damage Transport Logit Brier")
    axes[1].set_ylim(0.0, max([v for v in brier_vals if pd.notna(v)] + [0.6]) * 1.15)
    plt.tight_layout()
    fig.savefig(HAZARD_READY_FIG_PATH, dpi=220)
    plt.close(fig)


def _write_hazard_readiness_report(
    readiness_agg: pd.DataFrame,
    feature_summary: pd.DataFrame,
    allow_df: pd.DataFrame,
) -> None:
    hazard = pd.read_csv(HAZARD_TRANSPORT_AGG_PATH) if HAZARD_TRANSPORT_AGG_PATH.exists() else pd.DataFrame()
    quality = pd.read_csv(QUALITY_TRANSPORT_AGG_PATH) if QUALITY_TRANSPORT_AGG_PATH.exists() else pd.DataFrame()

    def _metric(df: pd.DataFrame, spec_id: str, model: str, col: str) -> float:
        if df.empty:
            return np.nan
        if "spec_id" in df.columns:
            sub = df[(df["spec_id"] == spec_id) & (df["model"] == model)]
        else:
            sub = df[df["model"] == model]
        return float(pd.to_numeric(sub[col], errors="coerce").mean()) if not sub.empty and col in sub.columns else np.nan

    hz1_auc = _metric(hazard, "HZ1", "Logit", "auc")
    ready_auc = _metric(readiness_agg, "HZ1_READY", "Logit", "auc")
    hz1_brier = _metric(hazard, "HZ1", "Logit", "brier")
    ready_brier = _metric(readiness_agg, "HZ1_READY", "Logit", "brier")
    delta_auc = ready_auc - hz1_auc if np.isfinite(ready_auc) and np.isfinite(hz1_auc) else np.nan
    delta_brier = ready_brier - hz1_brier if np.isfinite(ready_brier) and np.isfinite(hz1_brier) else np.nan
    top_logit = feature_summary[feature_summary["model"] == "Logit"].head(5)
    if np.isfinite(delta_auc) and np.isfinite(delta_brier):
        if delta_auc >= 0:
            verdict = "HZ1_READY preserves or improves ranking on the cleaner event pool and can be kept as the predictive anchor candidate."
        elif delta_brier < 0:
            verdict = "HZ1_READY improves calibration but loses ranking power versus the full-event HZ1 line, so it should remain a robustness subset rather than the main predictive anchor."
        else:
            verdict = "HZ1_READY underperforms the full-event HZ1 line on both ranking and calibration and should remain sensitivity-only."
    else:
        verdict = "HZ1_READY verdict unavailable because one or more key metrics are missing."

    lines = [
        "# Hazard Readiness-Filtered Transport Report",
        "",
        "## Objective",
        "- Re-run the HZ1 hazard/exposure transport line on the current mainline-ready event subset instead of the full mixed event pool.",
        "",
        "## Mainline Event Allowlist",
    ]
    if allow_df.empty:
        lines.append("- NA")
    else:
        for _, row in allow_df.iterrows():
            lines.append(f"- {row['event_id']}: source={row.get('allowlist_source', 'config')}, readiness_band={row.get('readiness_band', 'NA')}")

    lines.extend(
        [
            "",
            "## Metric Comparison",
            f"- QT1 Logit AUC: {_metric(quality, 'QT1', 'Logit', 'auc'):.4f}",
            f"- full-event HZ1 Logit AUC: {hz1_auc:.4f}",
            f"- readiness-filtered HZ1_READY Logit AUC: {ready_auc:.4f}",
            f"- HZ1_READY vs HZ1 AUC delta: {delta_auc:.4f}",
            f"- full-event HZ1 Logit Brier: {hz1_brier:.4f}",
            f"- HZ1_READY Logit Brier: {ready_brier:.4f}",
            f"- HZ1_READY vs HZ1 Brier delta: {delta_brier:.4f}",
            "",
            "## Top Hazard Features (Logit)",
        ]
    )
    if top_logit.empty:
        lines.append("- NA")
    else:
        for _, row in top_logit.iterrows():
            lines.append(
                f"- {row['feature']}: mean_coef={row['mean_coef']:.4f}, abs={row['mean_abs_coef']:.4f}, sign_consistency={row['sign_consistency']:.2f}"
            )

    lines.extend(
        [
            "",
            "## Recommendation",
            f"- {verdict}",
            "- Treat the readiness subset as a cleaner benchmark, not as a replacement for explanatory models.",
            "",
            "## Outputs",
            "- `project/modeling/output/hazard_transport_readiness_aggregate_metrics_v1.csv`",
            "- `project/modeling/output/hazard_transport_readiness_feature_summary_v1.csv`",
            "- `project/modeling/output/hazard_transport_readiness_events_v1.csv`",
            "- `project/modeling_report/figures/exploration_v2/hazard_transport_readiness_compare_v1.png`",
        ]
    )
    HAZARD_READY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_hazard_readiness_index() -> None:
    line = "- `project/modeling_report/hazard_transport_readiness_report_v1.md`"
    note = "- Appendix readiness-filtered hazard outputs: `project/modeling/output/hazard_transport_readiness_aggregate_metrics_v1.csv`, `project/modeling/output/hazard_transport_readiness_events_v1.csv`"
    text = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else "# Modeling Report Index\n\n## Deliverables\n"
    if line not in text:
        text = text.rstrip() + "\n" + line + "\n"
    if note not in text:
        text = text.rstrip() + "\n" + note + "\n"
    INDEX_PATH.write_text(text, encoding="utf-8")


def _run_hazard_mainline_v1_impl() -> int:
    ensure_directories()
    init_tracking_files()
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(issues=[])

    append_progress("Hazard mainline V1 started")
    append_progress("Hazard mainline V1: prepare quality-adjusted hazard inputs")
    panel_h, rec_h, target_audit = _prepare_hazard_transport_inputs()

    append_progress("Hazard mainline V1: LOEO transport")
    hazard_res = run_hazard_aware_transport(panel_h, rec_h)

    append_progress("Hazard mainline V1: summarize features and event selection")
    feature_summary = summarize_hazard_features(hazard_res.coef_df)
    selection_scorecard = build_event_selection_scorecard(hazard_res.fold_df, target_audit)
    _plot_hazard_transport_summary(hazard_res.agg_df)
    _write_hazard_transport_report(hazard_res.agg_df, feature_summary, selection_scorecard)
    _update_hazard_index()

    save_issue_log(ctx)
    append_progress("Hazard mainline V1 completed")
    return 0


def _run_hazard_readiness_v1_impl() -> int:
    ensure_directories()
    init_tracking_files()
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(issues=[])

    append_progress("Hazard readiness V1 started")
    panel_h, rec_h, _ = _prepare_hazard_transport_inputs()
    allow_events = _load_hazard_mainline_candidates()
    allow_df = pd.DataFrame({"event_id": allow_events, "allowlist_source": "config"})
    if EVENT_READINESS_SCORE_PATH.exists():
        readiness = pd.read_csv(EVENT_READINESS_SCORE_PATH)
        allow_df = allow_df.merge(readiness, on="event_id", how="left")
    allow_df.to_csv(HAZARD_READY_EVENTS_PATH, index=False)

    append_progress("Hazard readiness V1: filter mainline-ready events")
    panel_ready, rec_ready = _filter_event_allowlist(panel_h, rec_h, allow_events)
    if panel_ready.empty or rec_ready.empty:
        raise RuntimeError("hazard_readiness_empty_subset")

    append_progress("Hazard readiness V1: LOEO transport")
    ready_res = run_hazard_aware_transport(
        panel_ready,
        rec_ready,
        spec_id="HZ1_READY",
        experiment_family="hazard_transport_readiness",
        fold_path=HAZARD_READY_FOLD_PATH,
        agg_path=HAZARD_READY_AGG_PATH,
    )

    append_progress("Hazard readiness V1: summarize and report")
    feature_summary = summarize_hazard_features(ready_res.coef_df, output_path=HAZARD_READY_FEATURE_SUMMARY_PATH)
    _plot_hazard_readiness_summary(ready_res.agg_df)
    _write_hazard_readiness_report(ready_res.agg_df, feature_summary, allow_df)
    _update_hazard_readiness_index()

    save_issue_log(ctx)
    append_progress("Hazard readiness V1 completed")
    return 0


def _plot_bug_transport_summary(bug_agg: pd.DataFrame) -> None:
    quality = pd.read_csv(QUALITY_TRANSPORT_AGG_PATH) if QUALITY_TRANSPORT_AGG_PATH.exists() else pd.DataFrame()
    hazard = pd.read_csv(HAZARD_TRANSPORT_AGG_PATH) if HAZARD_TRANSPORT_AGG_PATH.exists() else pd.DataFrame()

    def _metric(df: pd.DataFrame, spec_id: str, model: str, col: str) -> float:
        if df.empty:
            return np.nan
        sub = df[(df["spec_id"] == spec_id) & (df["model"] == model)]
        return float(pd.to_numeric(sub[col], errors="coerce").mean()) if not sub.empty and col in sub.columns else np.nan

    labels = ["QT1", "HZ1", "BUG0", "BUG1A", "BUG1B", "BUG1C"]
    auc_vals = [
        float(pd.to_numeric(quality.loc[quality["model"] == "Logit", "auc"], errors="coerce").mean()) if not quality.empty else np.nan,
        float(pd.to_numeric(hazard.loc[hazard["model"] == "Logit", "auc"], errors="coerce").mean()) if not hazard.empty else np.nan,
        _metric(bug_agg, "BUG0", "Logit", "auc"),
        _metric(bug_agg, "BUG1A", "Logit", "auc"),
        _metric(bug_agg, "BUG1B", "Logit", "auc"),
        _metric(bug_agg, "BUG1C", "Logit", "auc"),
    ]
    brier_vals = [
        float(pd.to_numeric(quality.loc[quality["model"] == "Logit", "brier"], errors="coerce").mean()) if not quality.empty else np.nan,
        float(pd.to_numeric(hazard.loc[hazard["model"] == "Logit", "brier"], errors="coerce").mean()) if not hazard.empty else np.nan,
        _metric(bug_agg, "BUG0", "Logit", "brier"),
        _metric(bug_agg, "BUG1A", "Logit", "brier"),
        _metric(bug_agg, "BUG1B", "Logit", "brier"),
        _metric(bug_agg, "BUG1C", "Logit", "brier"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(labels, auc_vals, color=["#8c8c8c", "#1b9e77", "#4c78a8", "#59a14f", "#f28e2b", "#e15759"])
    axes[0].set_title("Damage Transport Logit AUC")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].bar(labels, brier_vals, color=["#8c8c8c", "#1b9e77", "#4c78a8", "#59a14f", "#f28e2b", "#e15759"])
    axes[1].set_title("Damage Transport Logit Brier")
    axes[1].set_ylim(0.0, max([v for v in brier_vals if pd.notna(v)] + [0.6]) * 1.15)
    plt.tight_layout()
    fig.savefig(BUG_TRANSPORT_FIG_PATH, dpi=220)
    plt.close(fig)


def _write_bug_transport_report(bug_agg: pd.DataFrame, feature_summary: pd.DataFrame, feature_audit: pd.DataFrame) -> None:
    quality = pd.read_csv(QUALITY_TRANSPORT_AGG_PATH) if QUALITY_TRANSPORT_AGG_PATH.exists() else pd.DataFrame()
    hazard = pd.read_csv(HAZARD_TRANSPORT_AGG_PATH) if HAZARD_TRANSPORT_AGG_PATH.exists() else pd.DataFrame()

    def _metric(df: pd.DataFrame, spec_id: str, model: str, col: str) -> float:
        if df.empty:
            return np.nan
        sub = df[(df["spec_id"] == spec_id) & (df["model"] == model)]
        return float(pd.to_numeric(sub[col], errors="coerce").mean()) if not sub.empty and col in sub.columns else np.nan

    def _family_metric(df: pd.DataFrame, model: str, col: str) -> float:
        if df.empty:
            return np.nan
        sub = df[df["model"] == model]
        return float(pd.to_numeric(sub[col], errors="coerce").mean()) if not sub.empty and col in sub.columns else np.nan

    top_logit = feature_summary[(feature_summary["model"] == "Logit") & (feature_summary["spec_id"].isin(["BUG1A", "BUG1B", "BUG1C"]))].head(8)
    bug0_auc = _metric(bug_agg, "BUG0", "Logit", "auc")
    bug1a_auc = _metric(bug_agg, "BUG1A", "Logit", "auc")
    bug1a_brier = _metric(bug_agg, "BUG1A", "Logit", "brier")
    improvement = bug1a_auc - bug0_auc if np.isfinite(bug1a_auc) and np.isfinite(bug0_auc) else np.nan
    brier_delta = bug1a_brier - _metric(bug_agg, "BUG0", "Logit", "brier") if np.isfinite(bug1a_brier) else np.nan

    lines = [
        "# BUG-aware Transport Report",
        "",
        "## Objective",
        "- Keep the existing mainline untouched and test whether BUG-aware proxy features improve damage transport stability.",
        "",
        "## Compared Specs",
        "- `BUG0`: quality-adjusted transport baseline rerun under the BUG family",
        "- `BUG1A`: baseline plus prior-weighted BUG features",
        "- `BUG1B`: replace binary `in_buffer` with `high_conf_bug_buffer` and keep baseline controls",
        "- `BUG1C`: BUG prior features plus quality guards, without the legacy baseline spatial context block",
        "",
        "## Metric Comparison",
        f"- QT1 Logit AUC: {_family_metric(quality, 'Logit', 'auc'):.4f}",
        f"- HZ1 Logit AUC: {_family_metric(hazard, 'Logit', 'auc'):.4f}",
        f"- BUG0 Logit AUC: {bug0_auc:.4f}",
        f"- BUG1A Logit AUC: {bug1a_auc:.4f}",
        f"- BUG1A vs BUG0 AUC delta: {improvement:.4f}",
        f"- BUG1A Logit Brier: {bug1a_brier:.4f}",
        f"- BUG1A vs BUG0 Brier delta: {brier_delta:.4f}",
        "",
        "## Feature Audit",
    ]
    if feature_audit.empty:
        lines.append("- No feature audit rows generated.")
    else:
        for _, row in feature_audit.iterrows():
            lines.append(
                f"- {row['event_id']}: poi_status={row['poi_status']}, poi_count={int(row['poi_count'])}, "
                f"high_conf_share={float(row['high_conf_share']):.2f}, high_conf_bug_buffer_share={float(row['high_conf_bug_buffer_share']):.3f}"
            )

    lines.extend(["", "## Top BUG Features (Logit)"])
    if top_logit.empty:
        lines.append("- NA")
    else:
        for _, row in top_logit.iterrows():
            lines.append(
                f"- {row['spec_id']} | {row['feature']}: mean_coef={row['mean_coef']:.4f}, "
                f"mean_abs={row['mean_abs_coef']:.4f}, sign_consistency={row['sign_consistency']:.2f}"
            )

    if np.isfinite(improvement) and np.isfinite(brier_delta):
        if improvement > 0 and brier_delta <= 0.02:
            verdict = "BUG1A improves damage ranking without material probability deterioration."
        elif improvement > 0:
            verdict = "BUG1A improves ranking but looks more like a ranking-only line because Brier worsens."
        else:
            verdict = "BUG1A does not improve transport enough to justify moving to official inventory work yet."
    else:
        verdict = "BUG1A verdict unavailable because one or more key metrics are missing."

        lines.extend(
        [
            "",
            "## Recommendation",
            f"- {verdict}",
            "- Freeze `BUG1` as a proxy-refinement test and do not keep tuning prior weights or BUG-only proxy thresholds.",
            "- Keep `strict-v2`, `quality_transport`, and `hazard_transport` unchanged; only a minimal official-inventory pilot should remain open for the BUG mechanism line.",
            "",
            "## Outputs",
            "- `project/modeling/output/bug_transport_fold_metrics_v1.csv`",
            "- `project/modeling/output/bug_transport_aggregate_metrics_v1.csv`",
            "- `project/modeling/output/bug_transport_feature_summary_v1.csv`",
            "- `project/modeling/output/bug_transport_feature_audit_v1.csv`",
            "- `project/modeling_report/figures/exploration_v2/bug_transport_compare_v1.png`",
        ]
    )
    BUG_TRANSPORT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_bug_transport_index() -> None:
    line = "- `project/modeling_report/bug_transport_report.md`"
    note = "- Appendix BUG-aware outputs: `project/modeling/output/bug_transport_aggregate_metrics_v1.csv`, `project/modeling/output/bug_transport_feature_audit_v1.csv`"
    text = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else "# Modeling Report Index\n\n## Deliverables\n"
    if line not in text:
        text = text.rstrip() + "\n" + line + "\n"
    if note not in text:
        text = text.rstrip() + "\n" + note + "\n"
    INDEX_PATH.write_text(text, encoding="utf-8")


def _run_bug_transport_v1_impl() -> int:
    ensure_directories()
    init_tracking_files()
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(issues=[])

    append_progress("BUG transport V1 started")
    if PANEL_QUALITY_PATH.exists() and RECOVERY_V2_PATH.exists():
        panel_q = pd.read_parquet(PANEL_QUALITY_PATH)
        rec_v2 = pd.read_parquet(RECOVERY_V2_PATH)
    else:
        panel = _filter_sample_lock(pd.read_parquet(PANEL_IN_PATH))
        panel_q, rec_v2, _ = build_target_quality_panel(panel)

    append_progress("BUG transport V1: attach BUG prior features")
    panel_bug, feature_audit = attach_bug_prior_features(panel_q)
    rec_bug = _merge_bug_features_into_recovery(panel_bug, rec_v2)

    append_progress("BUG transport V1: LOEO transport")
    bug_res = run_bug_aware_transport(panel_bug, rec_bug)

    append_progress("BUG transport V1: summarize features and report")
    feature_summary = summarize_bug_transport_features(bug_res.coef_df)
    _plot_bug_transport_summary(bug_res.agg_df)
    _write_bug_transport_report(bug_res.agg_df, feature_summary, feature_audit)
    _update_bug_transport_index()

    save_issue_log(ctx)
    append_progress("BUG transport V1 completed")
    return 0


def _build_bug2_acquisition_backlog() -> pd.DataFrame:
    tracker_path = ROOT / "BUG dataset tracker.xlsx"
    pilot_cfg = _load_bug2_pilot_config()
    priority_states = [str(v) for v in pilot_cfg.get("pilot_priority_order", ["PR", "FL", "LA"])]
    columns = [
        "State",
        "Priority state?",
        "Agency responsible for permitting BUGs",
        "Source for agency responsible for permitting BUGs",
        "Data availability",
        "Source for data availability",
        "Notes",
    ]
    fallback = pd.DataFrame(pilot_cfg.get("acquisition_backlog", []))
    if not tracker_path.exists():
        out = fallback.copy()
        if out.empty:
            out = pd.DataFrame(columns=columns)
        out.to_csv(BUG2_ACQ_BACKLOG_PATH, index=False)
        return out
    try:
        tracker = pd.read_excel(tracker_path, sheet_name="State level tracker")
        keep = [c for c in columns if c in tracker.columns]
        out = tracker[tracker["State"].astype(str).isin(priority_states)][keep].copy()
    except Exception:
        out = fallback.copy()
        if out.empty:
            out = pd.DataFrame(columns=columns)
    out = out.sort_values(["Priority state?", "State"], na_position="last")
    out.to_csv(BUG2_ACQ_BACKLOG_PATH, index=False)
    return out


def summarize_bug2_features(coef_df: pd.DataFrame) -> pd.DataFrame:
    tracked = {
        "official_bug_count_1km",
        "official_bug_kw_sum_1km",
        "official_bug_hours_proxy_1km",
        "official_bug_min_dist_m",
        "official_bug_coverage_flag",
    }
    if coef_df.empty:
        out = pd.DataFrame(columns=["spec_id", "model", "feature", "mean_coef", "mean_abs_coef", "sign_consistency", "folds"])
        out.to_csv(BUG2_FEATURE_SUMMARY_PATH, index=False)
        return out
    sub = coef_df[coef_df["feature"].isin(tracked)].copy()
    if sub.empty:
        out = pd.DataFrame(columns=["spec_id", "model", "feature", "mean_coef", "mean_abs_coef", "sign_consistency", "folds"])
        out.to_csv(BUG2_FEATURE_SUMMARY_PATH, index=False)
        return out
    rows = []
    for (spec_id, model, feature), grp in sub.groupby(["spec_id", "model", "feature"], dropna=False):
        coef = pd.to_numeric(grp["coef"], errors="coerce").dropna()
        if coef.empty:
            continue
        pos = float((coef > 0).mean())
        neg = float((coef < 0).mean())
        rows.append(
            {
                "spec_id": spec_id,
                "model": model,
                "feature": feature,
                "mean_coef": float(coef.mean()),
                "mean_abs_coef": float(coef.abs().mean()),
                "sign_consistency": max(pos, neg),
                "folds": int(coef.shape[0]),
            }
        )
    out = pd.DataFrame(rows).sort_values(["spec_id", "model", "mean_abs_coef"], ascending=[True, True, False])
    out.to_csv(BUG2_FEATURE_SUMMARY_PATH, index=False)
    return out


def run_bug2_pilot_transport(panel: pd.DataFrame, recovery: pd.DataFrame) -> SpecResult:
    official_terms = [
        "official_bug_count_1km",
        "official_bug_kw_sum_1km",
        "official_bug_hours_proxy_1km",
        "official_bug_min_dist_m",
        "official_bug_coverage_flag",
    ]
    specs = [
        ("BUG2_BASE", panel.copy(), recovery.copy(), BASE_NUMERIC + QUALITY_GUARD_NUMERIC),
        ("BUG2_OFFICIAL", panel.copy(), recovery.copy(), BASE_NUMERIC + QUALITY_GUARD_NUMERIC + official_terms),
    ]
    fold_parts: List[pd.DataFrame] = []
    agg_parts: List[pd.DataFrame] = []
    coef_parts: List[pd.DataFrame] = []
    for spec_id, spec_panel, spec_recovery, numeric_terms in specs:
        res = run_loeo_spec(
            spec_panel,
            spec_recovery,
            numeric_terms=numeric_terms,
            cat_terms=["land_use_group", "event_disaster_type"],
            experiment_family="bug2_pr_pilot",
            spec_id=spec_id,
        )
        fold_parts.append(res.fold_df)
        agg_parts.append(res.agg_df)
        coef_parts.append(res.coef_df)
    fold_df = pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()
    agg_df = pd.concat(agg_parts, ignore_index=True) if agg_parts else pd.DataFrame()
    coef_df = pd.concat(coef_parts, ignore_index=True) if coef_parts else pd.DataFrame()
    fold_df.to_csv(BUG2_FOLD_PATH, index=False)
    agg_df.to_csv(BUG2_AGG_PATH, index=False)
    return SpecResult(fold_df=fold_df, agg_df=agg_df, coef_df=coef_df)


def _plot_bug2_pilot_summary(bug2_agg: pd.DataFrame) -> None:
    def _metric(spec_id: str, model: str, col: str) -> float:
        if bug2_agg.empty:
            return np.nan
        sub = bug2_agg[(bug2_agg["spec_id"] == spec_id) & (bug2_agg["model"] == model)]
        return float(pd.to_numeric(sub[col], errors="coerce").mean()) if not sub.empty and col in sub.columns else np.nan

    labels = ["BUG2_BASE", "BUG2_OFFICIAL"]
    auc_vals = [_metric("BUG2_BASE", "Logit", "auc"), _metric("BUG2_OFFICIAL", "Logit", "auc")]
    brier_vals = [_metric("BUG2_BASE", "Logit", "brier"), _metric("BUG2_OFFICIAL", "Logit", "brier")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    axes[0].bar(labels, auc_vals, color=["#8c8c8c", "#1b9e77"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("PR Pilot Logit AUC")
    axes[1].bar(labels, brier_vals, color=["#8c8c8c", "#1b9e77"])
    axes[1].set_ylim(0.0, max([v for v in brier_vals if pd.notna(v)] + [0.6]) * 1.15)
    axes[1].set_title("PR Pilot Logit Brier")
    plt.tight_layout()
    fig.savefig(BUG2_FIG_PATH, dpi=220)
    plt.close(fig)


def _write_bug2_pilot_report(
    pilot_cfg: Dict[str, object],
    qa_df: pd.DataFrame,
    backlog_df: pd.DataFrame,
    feature_audit: pd.DataFrame,
    agg_df: Optional[pd.DataFrame] = None,
    feature_summary: Optional[pd.DataFrame] = None,
    status: str = "awaiting_inventory",
) -> None:
    gate_pass = int(qa_df["gate_pass"].iloc[0]) if not qa_df.empty and "gate_pass" in qa_df.columns else 0
    lines = [
        "# BUG2 Puerto Rico Pilot Report",
        "",
        "## Objective",
        "- Move the BUG line from proxy refinement to official inventory validation for the Puerto Rico pilot jurisdiction.",
        "",
        "## Pilot Scope",
        f"- pilot_state: {pilot_cfg.get('pilot_state', 'PR')}",
        f"- pilot_events: {';'.join([str(v) for v in pilot_cfg.get('pilot_event_ids', [])])}",
        f"- status: {status}",
        "",
        "## Acquisition Backlog",
    ]
    if backlog_df.empty:
        lines.append("- No tracker-derived backlog rows available.")
    else:
        for _, row in backlog_df.iterrows():
            lines.append(f"- {row['State']}: availability={row.get('Data availability', 'NA')}, notes={row.get('Notes', 'NA')}")

    lines.extend(["", "## QA Gate"])
    if qa_df.empty:
        lines.append("- QA not available.")
    else:
        row = qa_df.iloc[0]
        lines.extend(
            [
                f"- records_n: {int(row.get('records_n', 0))}",
                f"- geo_coverage: {float(row.get('geo_coverage', 0.0)):.3f}",
                f"- attribute_coverage: {float(row.get('attribute_coverage', 0.0)):.3f}",
                f"- gate_pass: {gate_pass}",
            ]
        )

    lines.extend(["", "## Feature Coverage"])
    if feature_audit.empty:
        lines.append("- Official inventory features not attached yet.")
    else:
        for _, row in feature_audit.iterrows():
            lines.append(
                f"- {row['event_id']}: coverage_flag={int(row.get('coverage_flag', 0))}, "
                f"inventory_records={int(row.get('inventory_records', 0))}, "
                f"feature_nonzero_share={float(row.get('feature_nonzero_share', 0.0)):.3f}"
            )

    if agg_df is not None and not agg_df.empty:
        def _metric(spec_id: str, model: str, col: str) -> float:
            sub = agg_df[(agg_df["spec_id"] == spec_id) & (agg_df["model"] == model)]
            return float(pd.to_numeric(sub[col], errors="coerce").mean()) if not sub.empty and col in sub.columns else np.nan

        lines.extend(
            [
                "",
                "## Model Comparison",
                f"- BUG2_BASE Logit AUC: {_metric('BUG2_BASE', 'Logit', 'auc'):.4f}",
                f"- BUG2_OFFICIAL Logit AUC: {_metric('BUG2_OFFICIAL', 'Logit', 'auc'):.4f}",
                f"- BUG2_BASE Logit Brier: {_metric('BUG2_BASE', 'Logit', 'brier'):.4f}",
                f"- BUG2_OFFICIAL Logit Brier: {_metric('BUG2_OFFICIAL', 'Logit', 'brier'):.4f}",
            ]
        )
        top_logit = feature_summary[feature_summary["model"] == "Logit"].head(5) if feature_summary is not None and not feature_summary.empty else pd.DataFrame()
        lines.extend(["", "## Top Official BUG Features"])
        if top_logit.empty:
            lines.append("- NA")
        else:
            for _, row in top_logit.iterrows():
                lines.append(
                    f"- {row['spec_id']} | {row['feature']}: mean_coef={row['mean_coef']:.4f}, abs={row['mean_abs_coef']:.4f}, sign_consistency={row['sign_consistency']:.2f}"
                )

    lines.extend(
        [
            "",
            "## Recommendation",
            "- Expand beyond Puerto Rico only if the QA gate passes and the official BUG features show a clear local increment over the baseline.",
            "- If the pilot remains blocked on data, keep BUG2 as an acquisition-and-validation track rather than a main modeling branch.",
            "",
            "## Outputs",
            "- `project/modeling/output/bug2_pilot_acquisition_backlog_v1.csv`",
            "- `project/modeling/output/bug2_pr_pilot_qa_v1.csv`",
            "- `project/modeling/output/bug2_pr_feature_audit_v1.csv`",
            "- `project/modeling/output/bug2_pr_pilot_aggregate_metrics_v1.csv`",
        ]
    )
    BUG2_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_bug2_index() -> None:
    line = "- `project/modeling_report/bug2_pr_pilot_report.md`"
    note = "- Appendix BUG2 pilot outputs: `project/modeling/output/bug2_pilot_acquisition_backlog_v1.csv`, `project/modeling/output/bug2_pr_pilot_qa_v1.csv`"
    text = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else "# Modeling Report Index\n\n## Deliverables\n"
    if line not in text:
        text = text.rstrip() + "\n" + line + "\n"
    if note not in text:
        text = text.rstrip() + "\n" + note + "\n"
    INDEX_PATH.write_text(text, encoding="utf-8")


def _run_bug2_pr_pilot_v1_impl() -> int:
    ensure_directories()
    init_tracking_files()
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)
    BUG_INVENTORY_RAW_DIR.mkdir(parents=True, exist_ok=True)
    BUG_INVENTORY_CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(issues=[])

    append_progress("BUG2 PR pilot V1 started")
    pilot_cfg = _load_bug2_pilot_config()
    backlog_df = _build_bug2_acquisition_backlog()
    source_path = ROOT / str(pilot_cfg.get("source_path", _safe_rel(BUG_INVENTORY_RAW_DIR / "bug_inventory_pr_pilot_raw.csv")))
    canonical_path = ROOT / str(pilot_cfg.get("canonical_path", _safe_rel(BUG2_PR_CANONICAL_PATH)))

    if not BUG2_PR_CANONICAL_TEMPLATE_PATH.exists():
        pd.DataFrame(columns=_bug2_required_columns()).to_csv(BUG2_PR_CANONICAL_TEMPLATE_PATH, index=False)

    if not canonical_path.exists() and source_path.exists():
        append_progress("BUG2 PR pilot V1: canonicalize raw inventory")
        inventory_raw = _load_bug_inventory_frame(source_path)
        inventory = canonicalize_bug_inventory(inventory_raw, pilot_cfg)
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        inventory.to_csv(canonical_path, index=False)

    if not canonical_path.exists():
        pd.DataFrame(columns=["experiment_family", "spec_id", "fold_event", "model", "rmse", "mae", "auc", "brier", "c_index", "coef_in_buffer", "notes"]).to_csv(BUG2_FOLD_PATH, index=False)
        pd.DataFrame(columns=["experiment_family", "spec_id", "model", "rmse", "mae", "auc", "brier", "c_index", "coef_in_buffer", "n_folds"]).to_csv(BUG2_AGG_PATH, index=False)
        pd.DataFrame(columns=["spec_id", "model", "feature", "mean_coef", "mean_abs_coef", "sign_consistency", "folds"]).to_csv(BUG2_FEATURE_SUMMARY_PATH, index=False)
        qa_df = pd.DataFrame([{
            "pilot_state": str(pilot_cfg.get("pilot_state", "PR")),
            "pilot_event_ids": ";".join([str(v) for v in pilot_cfg.get("pilot_event_ids", [])]),
            "records_n": 0,
            "geo_coverage": 0.0,
            "attribute_coverage": 0.0,
            "capacity_nonzero_share": 0.0,
            "hours_nonzero_share": 0.0,
            "distinct_facility_types": 0,
            "duplicate_record_share": 0.0,
            "gate_pass": 0,
        }])
        qa_df.to_csv(BUG2_QA_PATH, index=False)
        empty_audit = pd.DataFrame(columns=["event_id", "inventory_records", "coverage_flag", "feature_nonzero_share"])
        empty_audit.to_csv(BUG2_FEATURE_AUDIT_PATH, index=False)
        _write_bug2_pilot_report(pilot_cfg, qa_df, backlog_df, empty_audit, status="awaiting_inventory")
        _update_bug2_index()
        save_issue_log(ctx)
        append_progress("BUG2 PR pilot V1 completed: awaiting canonical inventory")
        return 0

    append_progress("BUG2 PR pilot V1: load canonical inventory")
    inventory_raw = _load_bug_inventory_frame(canonical_path)
    inventory = canonicalize_bug_inventory(inventory_raw, pilot_cfg)
    inventory.to_csv(canonical_path, index=False)
    qa_df = audit_bug_inventory(inventory, pilot_cfg)

    append_progress("BUG2 PR pilot V1: attach official features")
    panel_h, rec_h, _ = _prepare_hazard_transport_inputs()
    panel_sub, rec_sub = _filter_event_allowlist(panel_h, rec_h, [str(v) for v in pilot_cfg.get("pilot_event_ids", [])])
    panel_official, feature_audit = attach_official_bug_features(panel_sub, inventory, pilot_cfg)
    rec_official = _merge_bug_features_into_recovery(panel_official, rec_sub)

    gate_pass = int(qa_df["gate_pass"].iloc[0]) if not qa_df.empty else 0
    if gate_pass != 1 or panel_official["event_id"].astype(str).nunique() < 2:
        pd.DataFrame(columns=["experiment_family", "spec_id", "fold_event", "model", "rmse", "mae", "auc", "brier", "c_index", "coef_in_buffer", "notes"]).to_csv(BUG2_FOLD_PATH, index=False)
        pd.DataFrame(columns=["experiment_family", "spec_id", "model", "rmse", "mae", "auc", "brier", "c_index", "coef_in_buffer", "n_folds"]).to_csv(BUG2_AGG_PATH, index=False)
        pd.DataFrame(columns=["spec_id", "model", "feature", "mean_coef", "mean_abs_coef", "sign_consistency", "folds"]).to_csv(BUG2_FEATURE_SUMMARY_PATH, index=False)
        _write_bug2_pilot_report(pilot_cfg, qa_df, backlog_df, feature_audit, status="qa_failed_or_sparse")
        _update_bug2_index()
        save_issue_log(ctx)
        append_progress("BUG2 PR pilot V1 completed: QA gate failed or insufficient pilot events")
        return 0

    append_progress("BUG2 PR pilot V1: LOEO transport")
    bug2_res = run_bug2_pilot_transport(panel_official, rec_official)
    feature_summary = summarize_bug2_features(bug2_res.coef_df)
    _plot_bug2_pilot_summary(bug2_res.agg_df)
    _write_bug2_pilot_report(
        pilot_cfg,
        qa_df,
        backlog_df,
        feature_audit,
        agg_df=bug2_res.agg_df,
        feature_summary=feature_summary,
        status="modeled",
    )
    _update_bug2_index()
    save_issue_log(ctx)
    append_progress("BUG2 PR pilot V1 completed")
    return 0


def _write_future_plan_files() -> None:
    MASTER_PLAN_PATH.write_text(
        "# Exploration Master Plan (V2)\n\n"
        "- Anchor: v3r1 outputs as round0\n"
        "- Sample lock: `sample_lock_flag=1`\n"
        "- Validation: LOEO by event (6 folds)\n"
        "- Stop rule: if two consecutive lines are marginal (<+0.02 AUC and <+0.01 survival best), stop\n"
        "- This round adds six lines: cloud, mask, urban+pop, spatial, contribution, extreme-drop sensitivity\n",
        encoding="utf-8",
    )

    CLOUD_PLAN_PATH.write_text(
        "# Cloud/Coverage Importance Plan\n\n"
        "Specs:\n"
        "- C0: no cloud\n"
        "- C1: pixel_cloud_proxy\n"
        "- C2: + pre/post valid ratio\n"
        "- C3: + missing_cloud_flag (robustness)\n\n"
        "Outputs:\n"
        "- cloud_ablation_fold_metrics.csv\n"
        "- cloud_ablation_aggregate_metrics.csv\n"
        "- cloud_feature_importance.csv\n",
        encoding="utf-8",
    )

    MASK_PLAN_PATH.write_text(
        "# Noise Masking Plan\n\n"
        "- M1 hard mask: 11,12,90,95\n"
        "- M2 extended mask: +31,41,42,43,52,71\n"
        "- Agriculture (81,82) remains kept in this round\n"
        "- Compare coverage and model metrics under M0/M1/M2\n",
        encoding="utf-8",
    )

    URBAN_PLAN_PATH.write_text(
        "# Urban-Rural + Population Plan\n\n"
        "- Urban split: CBSA + Census Urban Area\n"
        "- Population: ACS tract (B01003 / ALAND)\n"
        "- Added fields: is_cbsa, is_urban_area, urban_rural_stratum, pop_density_per_km2, pop_density_log1p\n"
        "- Models: full + urban-only + rural-only\n",
        encoding="utf-8",
    )

    SPATIAL_PLAN_PATH.write_text(
        "# Spatial Autocorr + Contribution Plan\n\n"
        "- Moran's I on event-level OLS residuals\n"
        "- If spatial dependence appears, compare HC1 vs spatial-clustered SE\n"
        "- Build feature contribution scorecard by model\n",
        encoding="utf-8",
    )

    EXTREME_PLAN_PATH.write_text(
        "# Extreme Event Sensitivity Plan\n\n"
        "Candidate rule (double threshold):\n"
        "- High shift: smd_mean or psi_mean in top 33%\n"
        "- Poor prediction: >=2 task metrics in worst 33%\n"
        "- Candidate if both true\n\n"
        "Sensitivity only:\n"
        "- drop-1 for each candidate\n"
        "- drop-2 for top2 candidates\n"
        "- control-drop for one non-extreme event\n"
        "- Keep main conclusion unchanged\n",
        encoding="utf-8",
    )


POI_TAGS = {
    "amenity": ["hospital", "clinic", "fire_station", "police"],
    "power": ["plant", "generator", "substation"],
    "man_made": ["wastewater_plant", "water_works", "pumping_station", "mast", "tower"],
    "aeroway": ["aerodrome"],
    "shop": ["supermarket"],
    "tourism": ["hotel", "resort"],
    "landuse": ["industrial"],
}

LAND_TAGS = {
    "landuse": True,
    "natural": ["water", "wetland", "wood", "scrub", "grassland", "heath", "bare_rock", "sand"],
    "leisure": ["park", "garden", "recreation_ground", "golf_course", "nature_reserve"],
    "water": True,
}


def _load_event_increment_plan() -> List[Dict[str, object]]:
    return list(load_json(EVENT_INCREMENT_PLAN_PATH))


def _remote_event_paths(event_id: str, cfg: Dict[str, object]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for key in ["pre_dir", "post_dir"]:
        for item in _list_remote_paths(str(cfg[key])):
            if item.lower().endswith(".tif"):
                rows.append((item, "ntl_tif"))
    cloud_csv = str(Path("project/script") / f"{event_id}_cloud_screening.csv")
    for item in _list_remote_paths(cloud_csv):
        rows.append((item, "cloud_screening"))
    for item in _list_remote_paths("project/script/multi_event_ntl_download_v2.ipynb"):
        rows.append((item, "download_script"))
    return rows


def _sync_new_event_assets(stage_id: str, event_ids: Sequence[str], events_cfg: Dict[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for event_id in event_ids:
        for remote_path, file_type in _remote_event_paths(event_id, events_cfg[event_id]):
            local_path = ROOT / remote_path
            exists_before = int(local_path.exists())
            action = "keep_local"
            status = "keep_local"
            size_bytes = int(local_path.stat().st_size) if local_path.exists() else 0
            if not local_path.exists():
                action = "sync_from_teammate"
                try:
                    _export_remote_path(remote_path, local_path)
                    status = "copied"
                    size_bytes = int(local_path.stat().st_size) if local_path.exists() else 0
                except Exception as exc:  # noqa: BLE001
                    status = f"failed:{type(exc).__name__}"
                    action = "sync_failed"
            rows.append(
                {
                    "stage": stage_id,
                    "remote_path": remote_path,
                    "local_path": str(local_path),
                    "event_id": event_id,
                    "file_type": file_type,
                    "exists_before": exists_before,
                    "action": action,
                    "status": status,
                    "size_bytes": size_bytes,
                    "source_commit": "2431f55",
                }
            )
    out = pd.DataFrame(rows, columns=SYNC_MANIFEST_COLS)
    existing = _read_csv_or_empty(NEW_EVENT_SYNC_MANIFEST_PATH, SYNC_MANIFEST_COLS)
    merged = pd.concat([existing, out], ignore_index=True).drop_duplicates(
        subset=["stage", "remote_path", "local_path", "event_id", "file_type"],
        keep="last",
    )
    merged.to_csv(NEW_EVENT_SYNC_MANIFEST_PATH, index=False)
    merged.to_csv(NEW_EVENT_SYNC_LOG_PATH, index=False)
    return out


def _build_input_gate(event_ids: Sequence[str], events_cfg: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for event_id in event_ids:
        cfg = events_cfg[event_id]
        pre_dir = ROOT / str(cfg["pre_dir"])
        post_dir = ROOT / str(cfg["post_dir"])
        poi_csv = ROOT / str(cfg["poi_csv"])
        pre_tif_n = len(list_daily_tifs(pre_dir)) if pre_dir.exists() else 0
        post_tif_n = len(list_daily_tifs(post_dir)) if post_dir.exists() else 0
        has_cloud = _cloud_csv_for_event(event_id).exists()
        gate_status = "pass" if pre_tif_n > 0 and post_tif_n > 0 and has_cloud else "needs_attention"
        rows.append(
            {
                "event_id": event_id,
                "has_pre_dir": int(pre_dir.exists()),
                "has_post_dir": int(post_dir.exists()),
                "pre_tif_n": pre_tif_n,
                "post_tif_n": post_tif_n,
                "has_cloud_csv": int(has_cloud),
                "has_poi_csv_before": int(poi_csv.exists()),
                "gate_status": gate_status,
                "notes": "",
            }
        )
    out = pd.DataFrame(rows, columns=INPUT_GATE_COLS)
    existing = _read_csv_or_empty(NEW_EVENT_INPUT_GATE_PATH, INPUT_GATE_COLS)
    merged = pd.concat([existing, out], ignore_index=True).drop_duplicates(subset=["event_id"], keep="last")
    merged.to_csv(NEW_EVENT_INPUT_GATE_PATH, index=False)
    return out


def _fetch_osm_bbox(bbox: Tuple[float, float, float, float], tags: Dict[str, object]) -> gpd.GeoDataFrame:
    import osmnx as ox

    last_error = None
    for _ in range(3):
        try:
            gdf = ox.features_from_bbox(bbox=bbox, tags=tags)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            return gdf.to_crs("EPSG:4326")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"osm_query_failed:{type(last_error).__name__}:{last_error}")


def _row_value(row: pd.Series, key: str) -> str:
    if key not in row or pd.isna(row[key]):
        return ""
    return str(row[key]).strip().lower()


def _geom_centroid_latlon(geom) -> Tuple[float, float]:
    if geom is None or geom.is_empty:
        return np.nan, np.nan
    c = geom.centroid
    return float(c.y), float(c.x)


def _classify_poi_type(row: pd.Series) -> str:
    amenity = _row_value(row, "amenity")
    power = _row_value(row, "power")
    man_made = _row_value(row, "man_made")
    aeroway = _row_value(row, "aeroway")
    shop = _row_value(row, "shop")
    tourism = _row_value(row, "tourism")
    landuse = _row_value(row, "landuse")
    name = _row_value(row, "name")

    if power in {"plant", "generator", "substation"}:
        return power
    if amenity in {"hospital", "clinic", "fire_station", "police"}:
        return amenity
    if aeroway == "aerodrome":
        return "aerodrome"
    if man_made in {"wastewater_plant", "water_works", "pumping_station", "mast", "tower"}:
        return man_made
    if landuse == "industrial":
        return "industrial"
    if shop == "supermarket":
        return "supermarket"
    if tourism in {"hotel", "resort"}:
        return tourism
    if "hospital" in name:
        return "hospital"
    return ""


def _build_poi_from_osm(event_id: str, cfg: Dict[str, object]) -> pd.DataFrame:
    gdf = _fetch_osm_bbox(_event_bbox(cfg), POI_TAGS)
    if gdf.empty:
        return pd.DataFrame(columns=["osm_id", "name", "type", "lat", "lon"])
    work = gdf.reset_index().copy()
    osm_id = work["id"] if "id" in work.columns else work.get("osmid", pd.Series(index=work.index, dtype=float))
    work["type"] = work.apply(_classify_poi_type, axis=1)
    work = work[work["type"] != ""].copy()
    if work.empty:
        return pd.DataFrame(columns=["osm_id", "name", "type", "lat", "lon"])
    latlon = work["geometry"].apply(_geom_centroid_latlon)
    work["lat"] = [v[0] for v in latlon]
    work["lon"] = [v[1] for v in latlon]
    work["name"] = work.get("name", pd.Series([""] * len(work))).fillna("").astype(str)
    osm_id_num = pd.to_numeric(osm_id, errors="coerce")
    fallback_ids = pd.Series(np.arange(1, len(work) + 1), index=work.index, dtype="int64")
    work["osm_id"] = osm_id_num.where(osm_id_num.notna(), fallback_ids).astype(int)
    out = work[["osm_id", "name", "type", "lat", "lon"]].copy()
    out = out[np.isfinite(out["lat"]) & np.isfinite(out["lon"])].copy()
    return out.drop_duplicates(subset=["type", "lat", "lon", "name"], keep="first").sort_values(["type", "name", "osm_id"]).reset_index(drop=True)


def _generate_event_poi(event_id: str, cfg: Dict[str, object], acq_rows: List[Dict[str, object]], poi_rows: List[Dict[str, object]]) -> Path:
    poi_path = ROOT / str(cfg["poi_csv"])
    if poi_path.exists():
        existing = pd.read_csv(poi_path)
        if not existing.empty and {"type", "lat", "lon"}.issubset(existing.columns):
            poi_rows.append(
                {
                    "event_id": event_id,
                    "poi_source": "local_existing",
                    "poi_count": int(len(existing)),
                    "lat_valid_share": float(pd.to_numeric(existing["lat"], errors="coerce").notna().mean()),
                    "lon_valid_share": float(pd.to_numeric(existing["lon"], errors="coerce").notna().mean()),
                    "type_missing_share": float(existing["type"].isna().mean()),
                    "facility_type_unique_n": int(existing["type"].astype(str).nunique()),
                    "quality_flag": "ok",
                }
            )
            return poi_path

    df = _build_poi_from_osm(event_id, cfg)
    poi_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(poi_path, index=False)
    poi_rows.append(
        {
            "event_id": event_id,
            "poi_source": "osm_overpass",
            "poi_count": int(len(df)),
            "lat_valid_share": float(pd.to_numeric(df["lat"], errors="coerce").notna().mean()) if not df.empty else 0.0,
            "lon_valid_share": float(pd.to_numeric(df["lon"], errors="coerce").notna().mean()) if not df.empty else 0.0,
            "type_missing_share": float(df["type"].isna().mean()) if not df.empty else 1.0,
            "facility_type_unique_n": int(df["type"].astype(str).nunique()) if not df.empty else 0,
            "quality_flag": "ok" if not df.empty else "empty",
        }
    )
    acq_rows.append(
        {
            "event_id": event_id,
            "indicator_type": "poi",
            "source_name": "osm_overpass",
            "source_priority": 4,
            "request_status": "ok",
            "download_status": "ok" if not df.empty else "empty",
            "local_output_path": _safe_rel(poi_path),
            "coverage_metric": float(len(df)),
            "quality_flag": "ok" if not df.empty else "empty",
            "notes": "",
        }
    )
    return poi_path


def _classify_land_proxy(row: pd.Series) -> Tuple[int, str, int]:
    landuse = _row_value(row, "landuse")
    natural = _row_value(row, "natural")
    leisure = _row_value(row, "leisure")
    water = _row_value(row, "water")
    if water or natural in {"water", "wetland"} or landuse in {"reservoir", "basin"}:
        return 11, "other", 100
    if landuse in {"industrial", "commercial", "retail", "port", "quarry"}:
        return 24, "developed_high", 90
    if landuse in {"construction", "military", "brownfield"}:
        return 23, "developed_medium", 85
    if landuse == "residential":
        return 22, "developed_low", 80
    if leisure in {"park", "garden", "recreation_ground", "golf_course", "nature_reserve"} or landuse in {"recreation_ground", "cemetery", "grass"}:
        return 21, "developed_open", 70
    return 31, "other", 10


def _compute_local_landuse_shares_proxy(panel: pd.DataFrame, events_cfg: Dict[str, object], radius_m: float = 1000.0) -> pd.DataFrame:
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
        transformer = Transformer.from_crs("EPSG:4326", str(cfg["metric_crs"]), always_xy=True)
        x, y = transformer.transform(lon, lat)
        xy = np.column_stack([x, y])
        finite = np.isfinite(xy).all(axis=1)
        urban = np.isin(lu, [21, 22, 23, 24]).astype("float64")
        water_mask = np.isin(lu, [11]).astype("float64")
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
                src = idx_map[np.asarray(neigh, dtype=int)] if neigh else np.array([], dtype=int)
                if src.size == 0:
                    continue
                urban_share[base_i] = float(np.nanmean(urban[src]))
                water_share[base_i] = float(np.nanmean(water_mask[src]))
                high_share[base_i] = float(np.nanmean(high[src]))
        out.loc[mask, "urban_share_1km"] = urban_share
        out.loc[mask, "water_share_1km"] = water_share
        out.loc[mask, "developed_high_share_1km"] = high_share
    return out


def _attach_osm_land_proxy(panel: pd.DataFrame, event_ids: Sequence[str], events_cfg: Dict[str, object], acq_rows: List[Dict[str, object]], cov_rows: List[Dict[str, object]]) -> pd.DataFrame:
    out = panel.copy()
    if "land_use" not in out.columns:
        out["land_use"] = np.nan
    if "land_use_group" not in out.columns:
        out["land_use_group"] = "unknown"
    for event_id in event_ids:
        cfg = events_cfg[event_id]
        mask = out["event_id"] == event_id
        if mask.sum() == 0:
            continue
        try:
            land = _fetch_osm_bbox(_event_bbox(cfg), LAND_TAGS)
            land = land[land.geometry.notna()].copy()
            land = land[land.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            if land.empty:
                raise RuntimeError("no_land_polygons")
            classified = land.apply(_classify_land_proxy, axis=1, result_type="expand")
            land["land_use"] = classified[0]
            land["land_use_group"] = classified[1]
            land["priority"] = classified[2]
            pts = gpd.GeoDataFrame(
                out.loc[mask, ["pixel_id", "lon", "lat"]].copy(),
                geometry=gpd.points_from_xy(out.loc[mask, "lon"], out.loc[mask, "lat"]),
                crs="EPSG:4326",
            )
            joined = gpd.sjoin(pts, land[["land_use", "land_use_group", "priority", "geometry"]], how="left", predicate="within")
            joined = joined.sort_values(["pixel_id", "priority"], ascending=[True, False]).drop_duplicates(subset=["pixel_id"], keep="first")
            joined = joined.rename(columns={"land_use": "land_use_new", "land_use_group": "land_use_group_new"})
            out = out.merge(joined[["pixel_id", "land_use_new", "land_use_group_new"]], on="pixel_id", how="left")
            out.loc[mask, "land_use"] = pd.to_numeric(out.loc[mask, "land_use_new"], errors="coerce").fillna(31.0)
            out.loc[mask, "land_use_group"] = out.loc[mask, "land_use_group_new"].fillna("other").astype(str)
            out = out.drop(columns=[c for c in ["land_use_new", "land_use_group_new"] if c in out.columns])
            acq_rows.append(
                {
                    "event_id": event_id,
                    "indicator_type": "urban_boundary",
                    "source_name": "osm_landuse_proxy",
                    "source_priority": 1,
                    "request_status": "ok",
                    "download_status": "ok",
                    "local_output_path": "",
                    "coverage_metric": float(mask.sum()),
                    "quality_flag": "ok",
                    "notes": "landuse/natural/leisure polygons",
                }
            )
        except Exception as exc:  # noqa: BLE001
            out.loc[mask, "land_use"] = out.loc[mask, "land_use"].fillna(31.0)
            out.loc[mask, "land_use_group"] = out.loc[mask, "land_use_group"].fillna("unknown")
            acq_rows.append(
                {
                    "event_id": event_id,
                    "indicator_type": "urban_boundary",
                    "source_name": "osm_landuse_proxy",
                    "source_priority": 1,
                    "request_status": f"failed:{type(exc).__name__}",
                    "download_status": "failed",
                    "local_output_path": "",
                    "coverage_metric": 0.0,
                    "quality_flag": "fallback_unknown",
                    "notes": str(exc),
                }
            )
        for cov_name in ["land_use_group", "urban_share_1km", "water_share_1km", "developed_high_share_1km"]:
            cov_rows.append(
                {
                    "event_id": event_id,
                    "covariate_name": cov_name,
                    "source_name": "osm_landuse_proxy",
                    "source_type": "vector_overpass",
                    "spatial_resolution": "pixel_proxy",
                    "temporal_reference": "current_osm_snapshot",
                    "is_us_only": 0,
                    "used_in_mainline": 1,
                    "quality_flag": "ok",
                }
            )
    out = _compute_local_landuse_shares_proxy(out, {eid: events_cfg[eid] for eid in event_ids})
    return out


def _sync_stage_and_gate(stage_id: str, event_ids: Sequence[str], events_cfg: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sync_df = _sync_new_event_assets(stage_id, event_ids, events_cfg)
    gate_df = _build_input_gate(event_ids, events_cfg)
    return sync_df, gate_df


def _apply_population_sources(panel: pd.DataFrame, stage_tag: str, cov_rows: List[Dict[str, object]], acq_rows: List[Dict[str, object]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ctx = RunContext(issues=[])
    event_ids = sorted(panel["event_id"].dropna().astype(str).unique().tolist())
    saved = EVENT_TO_STATE.copy()
    EVENT_TO_STATE.update(US_EVENT_TO_STATE)
    try:
        out, quality = attach_urban_population(panel, ctx)
    finally:
        EVENT_TO_STATE.clear()
        EVENT_TO_STATE.update(saved)

    events_cfg = load_json(EVENTS10_PATH)
    for event_id in event_ids:
        country_scope = str(events_cfg[event_id].get("country_scope", "US"))
        mask = out["event_id"] == event_id
        if country_scope != "US":
            urban_share = _safe_numeric(out.loc[mask, "urban_share_1km"])
            out.loc[mask, "is_urban_area"] = (urban_share >= 0.30).astype(int)
            out.loc[mask, "is_cbsa"] = 0
            out.loc[mask, "urban_rural_stratum"] = np.where(urban_share >= 0.55, "urban", np.where(urban_share >= 0.25, "suburban", "rural"))

            bbox = _event_bbox(events_cfg[event_id])
            pop_density = np.nan
            try:
                west, south, east, north = bbox
                payload = {
                    "type": "Polygon",
                    "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
                }
                req = requests.get(
                    "https://api.worldpop.org/v1/services/stats",
                    params={"dataset": "wpgppop", "year": "2020", "geojson": json.dumps(payload)},
                    timeout=60,
                )
                req.raise_for_status()
                resp = req.json()
                total_pop = pd.to_numeric(resp.get("total_population"), errors="coerce")
                area = gpd.GeoSeries([box(west, south, east, north)], crs="EPSG:4326").to_crs(events_cfg[event_id]["metric_crs"]).area.iloc[0] / 1_000_000.0
                if pd.notna(total_pop) and area > 0:
                    pop_density = float(total_pop) / float(area)
                    out.loc[mask, "pop_density_per_km2"] = pop_density
                    out.loc[mask, "pop_density_log1p"] = np.log1p(max(pop_density, 0.0))
                    out.loc[mask, "missing_pop_flag"] = 0
                    acq_rows.append(
                        {
                            "event_id": event_id,
                            "indicator_type": "population_density",
                            "source_name": "worldpop_stats_api",
                            "source_priority": 1,
                            "request_status": "ok",
                            "download_status": "ok",
                            "local_output_path": "",
                            "coverage_metric": pop_density,
                            "quality_flag": "event_level_proxy",
                            "notes": "WorldPop 2020 bbox density",
                        }
                    )
                else:
                    raise RuntimeError("empty_worldpop_population")
            except Exception as exc:  # noqa: BLE001
                out.loc[mask, "missing_pop_flag"] = 1
                acq_rows.append(
                    {
                        "event_id": event_id,
                        "indicator_type": "population_density",
                        "source_name": "worldpop_stats_api",
                        "source_priority": 1,
                        "request_status": f"failed:{type(exc).__name__}",
                        "download_status": "failed",
                        "local_output_path": "",
                        "coverage_metric": 0.0,
                        "quality_flag": "missing",
                        "notes": str(exc),
                    }
                )
            cov_rows.append(
                {
                    "event_id": event_id,
                    "covariate_name": "pop_density_per_km2",
                    "source_name": "worldpop_stats_api" if np.isfinite(pop_density) else "missing",
                    "source_type": "api_event_level",
                    "spatial_resolution": "event_bbox",
                    "temporal_reference": "2020",
                    "is_us_only": 0,
                    "used_in_mainline": 0,
                    "quality_flag": "event_level_proxy" if np.isfinite(pop_density) else "missing",
                }
            )
            cov_rows.append(
                {
                    "event_id": event_id,
                    "covariate_name": "urban_rural_stratum",
                    "source_name": "osm_landuse_proxy",
                    "source_type": "vector_overpass",
                    "spatial_resolution": "pixel_proxy",
                    "temporal_reference": "current_osm_snapshot",
                    "is_us_only": 0,
                    "used_in_mainline": 0,
                    "quality_flag": "proxy",
                }
            )
        else:
            cov_rows.append(
                {
                    "event_id": event_id,
                    "covariate_name": "pop_density_per_km2",
                    "source_name": "acs_2022_b01003_tiger_2022",
                    "source_type": "census_api_plus_tiger",
                    "spatial_resolution": "tract",
                    "temporal_reference": "2022",
                    "is_us_only": 1,
                    "used_in_mainline": 0,
                    "quality_flag": "ok",
                }
            )
            cov_rows.append(
                {
                    "event_id": event_id,
                    "covariate_name": "urban_rural_stratum",
                    "source_name": "cbsa_uac20_tiger",
                    "source_type": "vector_tiger",
                    "spatial_resolution": "cbsa_urban_area",
                    "temporal_reference": "2022/2023",
                    "is_us_only": 1,
                    "used_in_mainline": 0,
                    "quality_flag": "ok",
                }
            )

    return out, quality


def _build_stage_feature_panel(
    stage_id: str,
    stage_tag: str,
    events_in_scope: Sequence[str],
    new_event_ids: Sequence[str],
    events_cfg: Dict[str, object],
) -> Tuple[pd.DataFrame, List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], pd.DataFrame]:
    p01 = _get_pipeline_module("p01", "01_in_sample_pipeline.py")
    stage_paths = _stage_paths(stage_tag)
    acq_rows: List[Dict[str, object]] = []
    poi_rows: List[Dict[str, object]] = []
    cov_rows: List[Dict[str, object]] = []

    base_panel = pd.read_parquet(FEATURE_PANEL_BASE_PATH).copy()
    base_panel = base_panel[base_panel["event_id"].isin([e for e in events_in_scope if e not in new_event_ids])].copy()

    if new_event_ids:
        tmp_cfg = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        try:
            json.dump({eid: events_cfg[eid] for eid in new_event_ids}, tmp_cfg, ensure_ascii=False, indent=2)
            tmp_cfg.close()
            tmp_cfg_path = Path(tmp_cfg.name)

            for event_id in new_event_ids:
                _generate_event_poi(event_id, events_cfg[event_id], acq_rows, poi_rows)

            sync_df, gate_df = _sync_stage_and_gate(stage_id, new_event_ids, events_cfg)
            append_progress(f"{stage_id}: synced {len(sync_df)} teammate asset rows")

            defaults = load_json(CONFIG_DEFAULTS)
            pre_thr = float(defaults.get("pre_threshold", defaults.get("pre_ntl_threshold", 0.5)))
            dmg_thr = float(defaults["damage_threshold"])
            ctx = RunContext(issues=[])
            with _override_attrs(pipeline_lib_mod, {"CONFIG_EVENTS": tmp_cfg_path}):
                new_panel = pipeline_lib_mod.build_pixel_panel(
                    ctx=ctx,
                    pre_threshold=pre_thr,
                    damage_threshold=dmg_thr,
                    exclude_types=None,
                    output_path=stage_paths["feature_panel"],
                )

            with _override_attrs(p01, {"SAMPLE_LOCK_PATH": stage_paths["sample_lock"]}):
                new_panel = p01.attach_pixel_cloud_features(ctx, new_panel, {eid: events_cfg[eid] for eid in new_event_ids})
                new_panel = _attach_osm_land_proxy(new_panel, new_event_ids, events_cfg, acq_rows, cov_rows)
                new_panel = p01.attach_osm_features(ctx, new_panel, {eid: events_cfg[eid] for eid in new_event_ids})
                new_panel = p01.create_sample_lock(new_panel)
                new_panel = p01.prepare_model_frame(new_panel)
            new_panel["event_disaster_type"] = new_panel["event_id"].map({eid: events_cfg[eid].get("disaster_type", "unknown") for eid in new_event_ids}).fillna("unknown").astype(str)

            full_panel = pd.concat([base_panel, new_panel], ignore_index=True, sort=False)
            full_panel, pop_quality = _apply_population_sources(full_panel, stage_tag, cov_rows, acq_rows)
        finally:
            try:
                Path(tmp_cfg.name).unlink(missing_ok=True)
            except Exception:
                pass
    else:
        full_panel = base_panel.copy()
        pop_quality = pd.DataFrame()
        gate_df = _build_input_gate([], events_cfg)

    full_panel.to_parquet(stage_paths["feature_panel"], index=False)
    if "sample_lock_flag" in full_panel.columns:
        full_panel[["pixel_id", "event_id", "sample_lock_flag", "lock_reason"]].to_parquet(stage_paths["sample_lock"], index=False)
    pd.DataFrame(acq_rows, columns=ACQ_MANIFEST_COLS).to_csv(NEW_EVENT_ACQ_MANIFEST_PATH, index=False)
    pd.DataFrame(poi_rows, columns=POI_QUALITY_COLS).to_csv(NEW_EVENT_POI_QUALITY_PATH, index=False)
    pd.DataFrame(cov_rows, columns=COV_SOURCE_COLS).to_csv(COVARIATE_SOURCE_MANIFEST_PATH, index=False)
    return full_panel, acq_rows, poi_rows, cov_rows, pop_quality


def _fetch_precip_7d_stage(lat: float, lon: float, event_date: Optional[pd.Timestamp]) -> Tuple[float, str]:
    if event_date is None or not np.isfinite(lat) or not np.isfinite(lon):
        return np.nan, "missing_event_date_or_coords"
    start = (event_date - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    end = event_date.strftime("%Y-%m-%d")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat:.6f}&longitude={lon:.6f}&start_date={start}&end_date={end}&daily=precipitation_sum&timezone=UTC"
    )
    try:
        payload = _fetch_json(url)
        vals = payload.get("daily", {}).get("precipitation_sum", [])
        arr = pd.to_numeric(pd.Series(vals), errors="coerce")
        return float(arr.sum()), "ok"
    except Exception as exc:  # noqa: BLE001
        return np.nan, f"api_error:{type(exc).__name__}"


def _fetch_elevation_slope_stage(lat: float, lon: float) -> Tuple[float, float, str]:
    if not np.isfinite(lat) or not np.isfinite(lon):
        return np.nan, np.nan, "missing_coords"
    pts = [
        (lat, lon),
        (lat + 0.02, lon),
        (lat - 0.02, lon),
        (lat, lon + 0.02),
        (lat, lon - 0.02),
    ]
    locs = "|".join([f"{la:.6f},{lo:.6f}" for la, lo in pts])
    url = f"https://api.opentopodata.org/v1/srtm90m?locations={locs}"
    try:
        payload = _fetch_json(url)
        elev = pd.to_numeric(pd.DataFrame(payload.get("results", []))["elevation"], errors="coerce").to_numpy(dtype=float)
        if elev.size == 0 or not np.isfinite(elev[0]):
            return np.nan, np.nan, "missing_center_elevation"
        center = float(elev[0])
        slopes = []
        for idx in range(1, len(pts)):
            if not np.isfinite(elev[idx]):
                continue
            d = float(np.hypot((pts[idx][0] - pts[0][0]) * 111_000.0, (pts[idx][1] - pts[0][1]) * 111_000.0))
            if d > 0:
                slopes.append(abs(float(elev[idx] - elev[0])) / d)
        slope = float(np.median(slopes)) if slopes else np.nan
        return center, slope, "ok"
    except Exception as exc:  # noqa: BLE001
        return np.nan, np.nan, f"api_error:{type(exc).__name__}"


def _build_stage_event_profile(panel: pd.DataFrame, events_cfg: Dict[str, object], events_in_scope: Sequence[str], output_path: Path, acq_rows: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for event_id in events_in_scope:
        sub = panel[panel["event_id"] == event_id].copy()
        if sub.empty:
            continue
        cfg = events_cfg[event_id]
        pre_tifs = list_daily_tifs(ROOT / str(cfg["pre_dir"]))
        post_tifs = list_daily_tifs(ROOT / str(cfg["post_dir"]))
        pre_dates = [d for d in (_parse_date_from_name_local(p) for p in pre_tifs) if d is not None]
        post_dates = [d for d in (_parse_date_from_name_local(p) for p in post_tifs) if d is not None]
        event_date = min(post_dates) if post_dates else None
        lat_center = float(pd.to_numeric(sub["lat"], errors="coerce").median())
        lon_center = float(pd.to_numeric(sub["lon"], errors="coerce").median())
        precip, precip_flag = _fetch_precip_7d_stage(lat_center, lon_center, event_date)
        elev, slope, topo_flag = _fetch_elevation_slope_stage(lat_center, lon_center)
        acq_rows.append(
            {
                "event_id": event_id,
                "indicator_type": "hazard_summary",
                "source_name": "open-meteo+opentopodata",
                "source_priority": 1,
                "request_status": "ok" if precip_flag == "ok" or topo_flag == "ok" else "partial",
                "download_status": "ok" if precip_flag == "ok" or topo_flag == "ok" else "partial",
                "local_output_path": _safe_rel(output_path),
                "coverage_metric": float(len(sub)),
                "quality_flag": f"precip={precip_flag};topo={topo_flag}",
                "notes": "",
            }
        )
        rows.append(
            {
                "event_id": event_id,
                "disaster_type": cfg.get("disaster_type", "unknown"),
                "lat_center": lat_center,
                "lon_center": lon_center,
                "coastal_flag": int(cfg.get("coastal_flag", 0)),
                "island_like_flag": int(cfg.get("island_like_flag", 0)),
                "elevation_median": elev,
                "slope_median": slope,
                "urban_share_1km": float(pd.to_numeric(sub["urban_share_1km"], errors="coerce").mean()),
                "water_share_1km": float(pd.to_numeric(sub["water_share_1km"], errors="coerce").mean()),
                "developed_high_share_1km": float(pd.to_numeric(sub["developed_high_share_1km"], errors="coerce").mean()),
                "pre_ntl_event_mean": float(pd.to_numeric(sub["pre_mean_ntl"], errors="coerce").mean()),
                "cloud_pre_event_mean": float(pd.to_numeric(sub["cloud_pre_mean"], errors="coerce").mean()),
                "cloud_post_event_mean": float(pd.to_numeric(sub["cloud_post_mean"], errors="coerce").mean()),
                "storm_precip_7d": precip,
                "event_duration_days": (int((max(post_dates) - min(post_dates)).days + 1) if post_dates else np.nan),
                "source_ref": "panel_stats;open-meteo;opentopodata",
                "quality_flag": f"precip={precip_flag};topo={topo_flag}",
            }
        )
    profile = pd.DataFrame(rows)
    profile.to_csv(output_path, index=False)
    return profile


def _run_stage_strict_v2(stage_paths: Dict[str, Path]) -> pd.DataFrame:
    p01 = _get_pipeline_module("p01", "01_in_sample_pipeline.py")
    overrides = {
        "STRICT_PANEL_FEATURE_PATH": stage_paths["feature_panel"],
        "STRICT_SAMPLE_LOCK_PATH": stage_paths["sample_lock"],
        "STRICT_MANIFEST_PATH": stage_paths["strict_manifest"],
        "STRICT_VIF_PATH": stage_paths["strict_vif"],
        "STRICT_SUMMARY_PATH": stage_paths["strict_summary"],
        "STRICT_SAMPLE_AUDIT_PATH": stage_paths["strict_sample_audit"],
        "STRICT_COX_DIAG_PATH": stage_paths["strict_cox_diag"],
        "STRICT_LOGO_FOLD_PATH": stage_paths["strict_logo_fold"],
        "STRICT_LOGO_AGG_PATH": stage_paths["strict_logo"],
        "STRICT_MISSING_FLAG_AUDIT_PATH": stage_paths["strict_missing_audit"],
        "STRICT_OLS_RESULT_PATH": stage_paths["strict_ols"],
        "STRICT_MIXED_RESULT_PATH": stage_paths["strict_mixed"],
        "STRICT_LOGIT_RESULT_PATH": stage_paths["strict_logit"],
        "STRICT_COX_RESULT_PATH": stage_paths["strict_cox"],
    }
    with _override_attrs(p01, overrides):
        try:
            p01.strict_main()
        except Exception as exc:  # noqa: BLE001
            if stage_paths["strict_summary"].exists():
                append_progress(
                    f"strict-v2 stage fallback: using summary before downstream LOGO failure ({type(exc).__name__}: {exc})"
                )
            else:
                raise
    return pd.read_csv(stage_paths["strict_summary"])


def _collect_baseline_metric_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    strict = pd.read_csv(STRICT_BASE_SUMMARY_PATH) if STRICT_BASE_SUMMARY_PATH.exists() else pd.DataFrame()
    hazard = pd.read_csv(HAZARD_TRANSPORT_AGG_PATH) if HAZARD_TRANSPORT_AGG_PATH.exists() else pd.DataFrame()
    quality = pd.read_csv(QUALITY_TRANSPORT_AGG_PATH) if QUALITY_TRANSPORT_AGG_PATH.exists() else pd.DataFrame()
    facility = pd.read_csv(FACILITY_CENTERED_SUMMARY_PATH) if FACILITY_CENTERED_SUMMARY_PATH.exists() else pd.DataFrame()

    for model, metric_name in [
        ("OLS", "coef_in_buffer"),
        ("MixedLM", "coef_in_buffer"),
        ("Logit", "odds_ratio_in_buffer"),
        ("Logit", "auc"),
        ("Cox", "hazard_ratio_in_buffer"),
    ]:
        if metric_name in {"auc"}:
            val = _metric_value(hazard, model, metric_name)
            pval = np.nan
        else:
            sub = strict[(strict["model"] == model) & (strict["variant"] == "full_locked_v2_strict") & (strict["key_metric"] == metric_name)]
            val = float(pd.to_numeric(sub["value"], errors="coerce").iloc[0]) if not sub.empty else np.nan
            pval = float(pd.to_numeric(sub["p_value"], errors="coerce").iloc[0]) if not sub.empty else np.nan
        rows.append({"stage_id": "baseline_6", "event_count": 6, "new_event_id": "", "bundle": "strict_v2", "model": model, "metric_name": metric_name, "value": val, "p_value": pval, "status": "ok", "notes": ""})

    for model, metric_name, col in [
        ("Logit", "auc", "auc"),
        ("Logit", "brier", "brier"),
        ("Cox", "c_index", "c_index"),
        ("AFT", "c_index", "c_index"),
        ("OLS", "rmse", "rmse"),
        ("MixedLM", "rmse", "rmse"),
    ]:
        rows.append({"stage_id": "baseline_6", "event_count": 6, "new_event_id": "", "bundle": "hazard_mainline", "model": model, "metric_name": metric_name, "value": _metric_value(hazard, model, col), "p_value": np.nan, "status": "ok", "notes": ""})

    for model, metric_name in [
        ("FacilityMatchedOLS", "coef_in_buffer"),
        ("FacilityMatchedLogit", "odds_ratio_in_buffer"),
        ("FacilityPairedATT", "mean_delta_diff"),
    ]:
        sub = facility[(facility["model"] == model) & (facility["metric_name"] == metric_name)]
        rows.append({"stage_id": "baseline_6", "event_count": 6, "new_event_id": "", "bundle": "quality_matched", "model": model, "metric_name": metric_name, "value": float(pd.to_numeric(sub["value"], errors="coerce").iloc[0]) if not sub.empty else np.nan, "p_value": float(pd.to_numeric(sub["p_value"], errors="coerce").iloc[0]) if not sub.empty else np.nan, "status": "ok", "notes": ""})
    if not quality.empty:
        rows.append({"stage_id": "baseline_6", "event_count": 6, "new_event_id": "", "bundle": "quality_matched", "model": "Logit", "metric_name": "auc", "value": _metric_value(quality, "Logit", "auc"), "p_value": np.nan, "status": "ok", "notes": ""})
    return rows


def _stage_metric_rows(stage_id: str, event_count: int, new_event_id: str, strict_df: pd.DataFrame, hazard_df: pd.DataFrame, facility_df: Optional[pd.DataFrame] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model, metric_name in [
        ("OLS", "coef_in_buffer"),
        ("MixedLM", "coef_in_buffer"),
        ("Logit", "odds_ratio_in_buffer"),
        ("Cox", "hazard_ratio_in_buffer"),
    ]:
        sub = strict_df[(strict_df["model"] == model) & (strict_df["variant"] == "full_locked_v2_strict") & (strict_df["key_metric"] == metric_name)]
        rows.append({"stage_id": stage_id, "event_count": event_count, "new_event_id": new_event_id, "bundle": "strict_v2", "model": model, "metric_name": metric_name, "value": float(pd.to_numeric(sub["value"], errors="coerce").iloc[0]) if not sub.empty else np.nan, "p_value": float(pd.to_numeric(sub["p_value"], errors="coerce").iloc[0]) if not sub.empty else np.nan, "status": "ok", "notes": ""})
    rows.append({"stage_id": stage_id, "event_count": event_count, "new_event_id": new_event_id, "bundle": "strict_v2", "model": "Logit", "metric_name": "auc", "value": _metric_value(hazard_df, "Logit", "auc"), "p_value": np.nan, "status": "ok", "notes": "transport_auc_reference"})

    for model, metric_name, col in [
        ("Logit", "auc", "auc"),
        ("Logit", "brier", "brier"),
        ("Cox", "c_index", "c_index"),
        ("AFT", "c_index", "c_index"),
        ("OLS", "rmse", "rmse"),
        ("MixedLM", "rmse", "rmse"),
    ]:
        rows.append({"stage_id": stage_id, "event_count": event_count, "new_event_id": new_event_id, "bundle": "hazard_mainline", "model": model, "metric_name": metric_name, "value": _metric_value(hazard_df, model, col), "p_value": np.nan, "status": "ok", "notes": ""})

    if facility_df is not None and not facility_df.empty:
        for model, metric_name in [
            ("FacilityMatchedOLS", "coef_in_buffer"),
            ("FacilityMatchedLogit", "odds_ratio_in_buffer"),
            ("FacilityPairedATT", "mean_delta_diff"),
        ]:
            sub = facility_df[(facility_df["model"] == model) & (facility_df["metric_name"] == metric_name)]
            rows.append({"stage_id": stage_id, "event_count": event_count, "new_event_id": new_event_id, "bundle": "quality_matched", "model": model, "metric_name": metric_name, "value": float(pd.to_numeric(sub["value"], errors="coerce").iloc[0]) if not sub.empty else np.nan, "p_value": float(pd.to_numeric(sub["p_value"], errors="coerce").iloc[0]) if not sub.empty else np.nan, "status": "ok", "notes": ""})
    return rows


def _finalize_metric_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delta_vs_prev"] = np.nan
    out["delta_vs_baseline"] = np.nan
    for (bundle, model, metric_name), grp in out.groupby(["bundle", "model", "metric_name"], dropna=False):
        idx = grp.sort_values("event_count").index.tolist()
        baseline_val = pd.to_numeric(out.loc[idx[0], "value"], errors="coerce")
        prev = None
        for i in idx:
            val = pd.to_numeric(out.loc[i, "value"], errors="coerce")
            out.loc[i, "delta_vs_baseline"] = val - baseline_val if pd.notna(val) and pd.notna(baseline_val) else np.nan
            out.loc[i, "delta_vs_prev"] = np.nan if prev is None or pd.isna(val) or pd.isna(prev) else val - prev
            prev = val
    out.to_csv(EVENT_INCREMENT_METRICS_PATH, index=False)
    return out


def _plot_event_increment(metrics: pd.DataFrame, stage_plan: List[Dict[str, object]]) -> None:
    FIG_EVENT_INCREMENT_DIR.mkdir(parents=True, exist_ok=True)
    order = [str(x["stage_id"]) for x in stage_plan]

    def _line(bundle: str, model: str, metric: str, path: Path, title: str, ylabel: str):
        sub = metrics[(metrics["bundle"] == bundle) & (metrics["model"] == model) & (metrics["metric_name"] == metric)].copy()
        if sub.empty:
            return
        sub["stage_id"] = pd.Categorical(sub["stage_id"], categories=order, ordered=True)
        sub = sub.sort_values("stage_id")
        labels = [_pretty_stage_label(x) for x in sub["stage_id"].astype(str)]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        ax.plot(x, pd.to_numeric(sub["value"], errors="coerce"), marker="o", color="#1d3557", linewidth=2.2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Stage")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.grid(alpha=0.25)
        fig.subplots_adjust(bottom=0.22, left=0.10, right=0.98, top=0.90)
        fig.savefig(path, dpi=220)
        plt.close(fig)

    _line("hazard_mainline", "Logit", "auc", FIG_EVENT_INCREMENT_DIR / "logit_auc_by_stage.png", "Hazard Mainline Logit AUC by Stage", "AUC")
    aft = metrics[(metrics["bundle"] == "hazard_mainline") & (metrics["metric_name"] == "c_index") & (metrics["model"].isin(["Cox", "AFT"]))].copy()
    if not aft.empty:
        pivot = aft.pivot_table(index="stage_id", columns="model", values="value", aggfunc="mean").reset_index()
        pivot["survival_best"] = pivot[["AFT", "Cox"]].max(axis=1)
        pivot["stage_id"] = pd.Categorical(pivot["stage_id"], categories=order, ordered=True)
        pivot = pivot.sort_values("stage_id")
        labels = [_pretty_stage_label(x) for x in pivot["stage_id"].astype(str)]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        ax.plot(x, pivot["survival_best"], marker="o", color="#457b9d", linewidth=2.2)
        ax.set_title("Hazard Mainline Survival Best by Stage")
        ax.set_ylabel("c-index")
        ax.set_xlabel("Stage")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.grid(alpha=0.25)
        fig.subplots_adjust(bottom=0.22, left=0.10, right=0.98, top=0.90)
        fig.savefig(FIG_EVENT_INCREMENT_DIR / "survival_best_by_stage.png", dpi=220)
        plt.close(fig)

    _line("strict_v2", "MixedLM", "coef_in_buffer", FIG_EVENT_INCREMENT_DIR / "strict_v2_in_buffer_by_stage.png", "Strict V2 MixedLM coef(in_buffer) by Stage", "coef")
    _line("quality_matched", "FacilityMatchedLogit", "odds_ratio_in_buffer", FIG_EVENT_INCREMENT_DIR / "matched_logit_or_by_stage.png", "Matched Logit OR(in_buffer) by Stage", "odds ratio")

    summary = pd.DataFrame(stage_plan)[["stage_id", "new_event_id", "event_count", "group_tag"]].copy()
    summary["new_event_id"] = summary["new_event_id"].replace("", "baseline")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = summary["group_tag"].map({"baseline": "#6c757d", "us_only": "#2a9d8f", "intl_addition": "#e76f51"}).fillna("#457b9d")
    labels = [_pretty_event_label(x) for x in summary["new_event_id"]]
    ax.bar(labels, summary["event_count"], color=colors)
    ax.set_title("Event Gap Coverage Progress")
    ax.set_ylabel("Event Count")
    ax.set_xlabel("Added Event")
    fig.subplots_adjust(bottom=0.22, left=0.10, right=0.98, top=0.90)
    fig.savefig(FIG_EVENT_INCREMENT_DIR / "event_gap_coverage_map.png", dpi=220)
    plt.close(fig)


def _write_event_increment_report(metrics: pd.DataFrame, manifest: pd.DataFrame, acq: pd.DataFrame, poi_quality: pd.DataFrame, stage_plan: List[Dict[str, object]]) -> None:
    order = [str(x["stage_id"]) for x in stage_plan]
    hz_auc = metrics[(metrics["bundle"] == "hazard_mainline") & (metrics["model"] == "Logit") & (metrics["metric_name"] == "auc")][["stage_id", "value", "delta_vs_prev", "delta_vs_baseline"]].copy()
    if not hz_auc.empty:
        hz_auc["stage_id"] = pd.Categorical(hz_auc["stage_id"], categories=order, ordered=True)
        hz_auc = hz_auc.sort_values("stage_id")
    surv = metrics[(metrics["bundle"] == "hazard_mainline") & (metrics["metric_name"] == "c_index") & (metrics["model"].isin(["Cox", "AFT"]))].copy()
    if not surv.empty:
        surv = surv.pivot_table(index="stage_id", columns="model", values="value", aggfunc="mean").reset_index()
        surv["survival_best"] = surv[["AFT", "Cox"]].max(axis=1)
        surv["stage_id"] = pd.Categorical(surv["stage_id"], categories=order, ordered=True)
        surv = surv.sort_values("stage_id")
    else:
        surv = pd.DataFrame(columns=["stage_id", "survival_best"])

    lines = [
        "# Event Increment Report / 新事件增量接入报告",
        "",
        "## Objective",
        "- 只同步 teammate 新增事件文件，逐步扩展事件集合并比较 strict-v2、hazard-mainline、quality-matched 三条评估线是否改善。",
        "",
        "## Stages",
    ]
    for row in stage_plan:
        lines.append(f"- `{row['stage_id']}`: event_count={row['event_count']}, new_event=`{row['new_event_id'] or 'baseline'}`, group=`{row['group_tag']}`")

    generated_poi = int(
        acq[(acq["indicator_type"] == "poi") & (acq["request_status"] == "ok")]["event_id"].nunique()
    ) if not acq.empty else 0
    sync_rows = int(_read_csv_or_empty(NEW_EVENT_SYNC_MANIFEST_PATH, SYNC_MANIFEST_COLS).shape[0])
    lines.extend(["", "## Sync & Acquisition", f"- synced rows: {sync_rows}", f"- acquisition records: {len(acq)}", f"- generated/new POI files: {generated_poi}", ""])
    lines.append("## Hazard Mainline Logit AUC by Stage")
    if hz_auc.empty:
        lines.append("- NA")
    else:
        for _, r in hz_auc.iterrows():
            lines.append(f"- {r['stage_id']}: auc={float(r['value']):.4f}, delta_prev={float(r['delta_vs_prev']) if pd.notna(r['delta_vs_prev']) else float('nan'):.4f}, delta_baseline={float(r['delta_vs_baseline']):.4f}")

    lines.extend(["", "## Survival Best by Stage"])
    if surv.empty:
        lines.append("- NA")
    else:
        for _, r in surv.iterrows():
            lines.append(f"- {r['stage_id']}: survival_best={float(r['survival_best']):.4f}")

    lines.extend(["", "## Figures", "- `project/modeling_report/figures/event_increment/logit_auc_by_stage.png`", "- `project/modeling_report/figures/event_increment/survival_best_by_stage.png`", "- `project/modeling_report/figures/event_increment/strict_v2_in_buffer_by_stage.png`", "- `project/modeling_report/figures/event_increment/matched_logit_or_by_stage.png`", "- `project/modeling_report/figures/event_increment/event_gap_coverage_map.png`"])
    EVENT_INCREMENT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    text = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else "# Modeling Report Index\n\n"
    line = "- `project/modeling_report/13_event_increment_report.md`"
    if line not in text:
        text = text.rstrip() + "\n" + line + "\n"
        INDEX_PATH.write_text(text, encoding="utf-8")


def _write_event_increment_tracking(metrics: pd.DataFrame, issue_rows: List[Dict[str, object]]) -> None:
    EVENT_INCREMENT_BOOTSTRAP_PATH.write_text(
        "# Event Increment Bootstrap\n\n"
        f"- Generated at: {datetime.utcnow().isoformat()}Z\n"
        f"- Metrics file: `{EVENT_INCREMENT_METRICS_PATH.relative_to(ROOT)}`\n",
        encoding="utf-8",
    )
    issue_df = pd.DataFrame(issue_rows, columns=EVENT_INCREMENT_ISSUE_COLS)
    issue_df.to_csv(EVENT_INCREMENT_ISSUE_PATH, index=False)
    EVENT_INCREMENT_ISSUE_MD_PATH.write_text(
        "# Event Increment Issues\n\n" + ("\n".join([f"- {r['stage_id']} | {r['event_id']} | {r['issue_type']} | {r['symptom']} | {r['status']}" for r in issue_rows]) if issue_rows else "- No critical issue observed\n") + "\n",
        encoding="utf-8",
    )
    EVENT_INCREMENT_NEXT_PATH.write_text(
        "# Post Event-Increment Next Steps\n\n"
        "- Re-check which event types produce positive transport deltas.\n"
        "- Keep hazard-aware transport as the main comparison line.\n"
        "- Use full-stage matched design only for explanatory stability, not as the transport KPI.\n",
        encoding="utf-8",
    )


def _recommend_next_event_types(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "gap_type": "damage_transport",
            "current_coverage": "hurricane-heavy, earthquake-light, island-like sparse",
            "event_ids_current": "earthquake_sanjuan,earthquake_hatay,dorian_freeport,maria_sanjuan",
            "recommended_next_event_type": "non-island earthquake in mid-urban setting",
            "why": "best candidate to further break earthquake=SanJuan coupling",
            "it_addresses": "damage_transport",
            "priority": "high",
        },
        {
            "gap_type": "recovery_transport",
            "current_coverage": "coastal hurricane recovery still unstable",
            "event_ids_current": "michael_panamacity,laura_lakecharles,ian_fortmyers,ian_charlotteharbor",
            "recommended_next_event_type": "coastal hurricane with cleaner post-observation coverage",
            "why": "survival metrics remain more sensitive to observation quality than to simple event count",
            "it_addresses": "recovery_transport",
            "priority": "high",
        },
        {
            "gap_type": "explanatory_balance",
            "current_coverage": "buffer signal mostly coastal and critical-facility dense",
            "event_ids_current": "all_current",
            "recommended_next_event_type": "non-island, medium-density inland event with strong facility map coverage",
            "why": "improves matched-design comparability without over-weighting island/coastal structure",
            "it_addresses": "explanatory_balance",
            "priority": "medium",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(EVENT_TYPE_GAP_PATH, index=False)
    return out


def _run_event_expansion_v1_impl() -> int:
    ensure_directories()
    init_tracking_files()
    FIG_EVENT_INCREMENT_DIR.mkdir(parents=True, exist_ok=True)
    append_progress("Event increment V1 started")

    events_cfg = load_json(EVENTS10_PATH)
    stage_plan = _load_event_increment_plan()
    issue_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    all_acq_rows: List[Dict[str, object]] = []
    all_poi_rows: List[Dict[str, object]] = []
    all_cov_rows: List[Dict[str, object]] = []
    metric_rows = _collect_baseline_metric_rows()

    pd.DataFrame(
        [
            {
                "stage_id": "baseline_6",
                "event_count": 6,
                "new_event_id": "",
                "group_tag": "baseline",
                "events_config_path": _safe_rel(EVENTS10_PATH),
                "panel_feature_path": _safe_rel(FEATURE_PANEL_BASE_PATH),
                "panel_cross_event_path": _safe_rel(PANEL_IN_PATH),
                "event_profile_path": _safe_rel(EVENT_PROFILE_V1_PATH),
                "poi_ready_flag": 1,
                "status": "ok",
            }
        ]
    ).to_csv(EVENT_INCREMENT_MANIFEST_PATH, index=False)

    cumulative_new: List[str] = []
    for stage in stage_plan[1:]:
        stage_id = str(stage["stage_id"])
        new_event_id = str(stage["new_event_id"])
        stage_tag = stage_id
        cumulative_new.append(new_event_id)
        stage_paths = _stage_paths(stage_tag)
        append_progress(f"{stage_id}: build feature panel for {new_event_id}")
        try:
            panel_stage, acq_rows, poi_rows, cov_rows, pop_quality = _build_stage_feature_panel(
                stage_id=stage_id,
                stage_tag=stage_tag,
                events_in_scope=list(stage["events_in_scope"]),
                new_event_ids=cumulative_new,
                events_cfg=events_cfg,
            )
            all_acq_rows.extend(acq_rows)
            all_poi_rows.extend(poi_rows)
            all_cov_rows.extend(cov_rows)
            manifest_rows.append(
                {
                    "stage_id": stage_id,
                    "event_count": int(stage["event_count"]),
                    "new_event_id": new_event_id,
                    "group_tag": str(stage["group_tag"]),
                    "events_config_path": _safe_rel(EVENTS10_PATH),
                    "panel_feature_path": _safe_rel(stage_paths["feature_panel"]),
                    "panel_cross_event_path": "",
                    "event_profile_path": _safe_rel(stage_paths["event_profile"]),
                    "poi_ready_flag": int((ROOT / str(events_cfg[new_event_id]["poi_csv"])).exists()),
                    "status": "ok",
                }
            )

            with _override_globals(
                {
                    "CONFIG_EVENTS": EVENTS10_PATH,
                    "PANEL_QUALITY_PATH": stage_paths["quality_panel"],
                    "RECOVERY_V2_PATH": stage_paths["recovery_v2"],
                    "TARGET_QUALITY_AUDIT_PATH": stage_paths["target_audit"],
                    "QUALITY_TRANSPORT_FOLD_PATH": stage_paths["quality_fold"],
                    "QUALITY_TRANSPORT_AGG_PATH": stage_paths["quality_agg"],
                    "SPATIAL_BLOCK_CV_PATH": stage_paths["spatial_block"],
                    "FACILITY_CONTEXT_PATH": stage_paths["facility_panel"],
                    "FACILITY_MATCH_QUALITY_PATH": stage_paths["facility_quality"],
                    "FACILITY_CENTERED_SUMMARY_PATH": stage_paths["facility_summary"],
                    "MODEL_ROLE_MATRIX_PATH": stage_paths["role_matrix"],
                    "PANEL_HAZARD_PATH": stage_paths["hazard_panel"],
                    "HAZARD_TRANSPORT_FOLD_PATH": stage_paths["hazard_fold"],
                    "HAZARD_TRANSPORT_AGG_PATH": stage_paths["hazard_agg"],
                    "HAZARD_FEATURE_SUMMARY_PATH": stage_paths["hazard_feature"],
                    "EVENT_SELECTION_PATH": stage_paths["event_selection"],
                    "EVENT_PROFILE_V1_PATH": stage_paths["event_profile"],
                }
            ):
                panel_q, rec_v2, target_audit = build_target_quality_panel(panel_stage)
                stage_profile = _build_stage_event_profile(panel_q, events_cfg, list(stage["events_in_scope"]), stage_paths["event_profile"], acq_rows)
                panel_h = attach_hazard_exposure_features(panel_q)
                merge_cols = ["pixel_id", "event_id"] + [
                    c
                    for c in panel_h.columns
                    if c not in {"pixel_id", "event_id"}
                    and (c.startswith("event_") or c in ["island_local_water", "island_local_urban", "hazard_cloud_water", "hazard_precip_urban"])
                ]
                rec_h = rec_v2.drop(columns=[c for c in merge_cols if c in rec_v2.columns and c not in {"pixel_id", "event_id"}], errors="ignore")
                rec_h = rec_h.merge(panel_h[merge_cols].drop_duplicates(subset=["pixel_id"]), on=["pixel_id", "event_id"], how="left")
                hazard_res = run_hazard_aware_transport(panel_h, rec_h)
                summarize_hazard_features(hazard_res.coef_df)
                build_event_selection_scorecard(hazard_res.fold_df, target_audit)

                if stage_id == "stage_10_dorian_freeport":
                    quality_res = run_quality_transport(panel_q, rec_v2)
                    spatial_block = run_spatial_block_cv(panel_q, rec_v2)
                    fac_panel, match_quality = build_facility_context_panel(panel_q)
                    facility_summary = fit_facility_centered_models(fac_panel)
                    build_model_role_matrix(quality_res.agg_df, spatial_block, facility_summary)
                else:
                    facility_summary = None

            strict_df = _run_stage_strict_v2(stage_paths)
            hazard_df = pd.read_csv(stage_paths["hazard_agg"])
            metric_rows.extend(_stage_metric_rows(stage_id, int(stage["event_count"]), new_event_id, strict_df, hazard_df, facility_summary))
            if not pop_quality.empty:
                pop_quality.to_csv(_stage_suffix_path(POP_QUALITY_PATH, stage_tag), index=False)
        except Exception as exc:  # noqa: BLE001
            issue_rows.append(
                {
                    "stage_id": stage_id,
                    "event_id": new_event_id,
                    "bundle": "event_expansion",
                    "issue_type": "stage_failed",
                    "symptom": str(exc),
                    "fix_action": "record failure and continue",
                    "impact": "stage result unavailable",
                    "status": "open",
                }
            )
            manifest_rows.append(
                {
                    "stage_id": stage_id,
                    "event_count": int(stage["event_count"]),
                    "new_event_id": new_event_id,
                    "group_tag": str(stage["group_tag"]),
                    "events_config_path": _safe_rel(EVENTS10_PATH),
                    "panel_feature_path": _safe_rel(stage_paths["feature_panel"]),
                    "panel_cross_event_path": "",
                    "event_profile_path": _safe_rel(stage_paths["event_profile"]),
                    "poi_ready_flag": 0,
                    "status": f"failed:{type(exc).__name__}",
                }
            )

    manifest = pd.concat([_read_csv_or_empty(EVENT_INCREMENT_MANIFEST_PATH, ["stage_id", "event_count", "new_event_id", "group_tag", "events_config_path", "panel_feature_path", "panel_cross_event_path", "event_profile_path", "poi_ready_flag", "status"]), pd.DataFrame(manifest_rows)], ignore_index=True)
    manifest.to_csv(EVENT_INCREMENT_MANIFEST_PATH, index=False)
    pd.DataFrame(all_acq_rows, columns=ACQ_MANIFEST_COLS).drop_duplicates(
        subset=["event_id", "indicator_type", "source_name"],
        keep="last",
    ).to_csv(NEW_EVENT_ACQ_MANIFEST_PATH, index=False)
    pd.DataFrame(all_poi_rows, columns=POI_QUALITY_COLS).drop_duplicates(
        subset=["event_id"],
        keep="last",
    ).to_csv(NEW_EVENT_POI_QUALITY_PATH, index=False)
    pd.DataFrame(all_cov_rows, columns=COV_SOURCE_COLS).drop_duplicates(
        subset=["event_id", "covariate_name", "source_name"],
        keep="last",
    ).to_csv(COVARIATE_SOURCE_MANIFEST_PATH, index=False)
    metrics = _finalize_metric_deltas(pd.DataFrame(metric_rows, columns=EVENT_INCREMENT_METRIC_COLS))
    acq = _read_csv_or_empty(NEW_EVENT_ACQ_MANIFEST_PATH, ACQ_MANIFEST_COLS)
    poi_quality = _read_csv_or_empty(NEW_EVENT_POI_QUALITY_PATH, POI_QUALITY_COLS)
    _plot_event_increment(metrics, stage_plan)
    _write_event_increment_report(metrics, manifest, acq, poi_quality, stage_plan)
    _write_event_increment_tracking(metrics, issue_rows)
    _recommend_next_event_types(metrics)
    append_progress("Event increment V1 completed")
    return 0


def _plot_experiment_summary(cloud_agg: pd.DataFrame, mask_agg: pd.DataFrame, urban_agg: pd.DataFrame) -> None:
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    c = cloud_agg[(cloud_agg["model"] == "Logit") & (cloud_agg["spec_id"].isin(["C0", "C1", "C2", "C3"]))].copy()
    if not c.empty:
        sns.barplot(data=c, x="spec_id", y="auc", ax=axes[0], color="#2a9d8f")
    axes[0].set_title("Cloud Ablation: Logit AUC")
    axes[0].set_xlabel("Spec")
    axes[0].set_ylabel("AUC")

    m = mask_agg[(mask_agg["model"] == "Logit") & (mask_agg["spec_id"].isin(["M0", "M1", "M2"]))].copy()
    if not m.empty:
        sns.barplot(data=m, x="spec_id", y="auc", ax=axes[1], color="#e76f51")
    axes[1].set_title("Noise Mask: Logit AUC")
    axes[1].set_xlabel("Spec")
    axes[1].set_ylabel("AUC")

    u = urban_agg[(urban_agg["model"] == "Logit") & (urban_agg["spec_id"].isin(["UR_full", "UR_urban", "UR_rural"]))].copy()
    if not u.empty:
        sns.barplot(data=u, x="spec_id", y="auc", ax=axes[2], color="#264653")
    axes[2].set_title("Urban Split: Logit AUC")
    axes[2].set_xlabel("Spec")
    axes[2].set_ylabel("AUC")

    fig.tight_layout()
    fig.savefig(FIG_EXP_DIR / "exploration_auc_compare.png", dpi=220)
    plt.close(fig)


def _update_index() -> None:
    if INDEX_PATH.exists():
        text = INDEX_PATH.read_text(encoding="utf-8")
    else:
        text = "# Modeling Report Index\n\n"

    lines_to_add = [
        "- `project/modeling_report/10_exploration_upgrade_report.md`",
        "- Exploration V2 outputs: `project/modeling/output/cloud_ablation_aggregate_metrics.csv`, `project/modeling/output/noise_mask_experiment_metrics.csv`, `project/modeling/output/urban_rural_model_comparison.csv`, `project/modeling/output/spatial_autocorr_morans_i.csv`, `project/modeling/output/extreme_event_drop_aggregate_v1.csv`",
    ]

    for line in lines_to_add:
        if line not in text:
            text = text.rstrip() + "\n" + line + "\n"

    INDEX_PATH.write_text(text, encoding="utf-8")


def _write_report(
    cloud_agg: pd.DataFrame,
    mask_agg: pd.DataFrame,
    urban_agg: pd.DataFrame,
    moran: pd.DataFrame,
    contrib: pd.DataFrame,
    extreme_agg: pd.DataFrame,
    decision: Dict[str, object],
    pop_quality: pd.DataFrame,
) -> None:
    def _metric(df: pd.DataFrame, spec: str, model: str, col: str) -> str:
        if df.empty:
            return "NA"
        s = df[(df["spec_id"] == spec) & (df["model"] == model)][col]
        if s.empty or s.isna().all():
            return "NA"
        return f"{float(s.mean()):.4f}"

    c0_auc = _metric(cloud_agg, "C0", "Logit", "auc")
    c1_auc = _metric(cloud_agg, "C1", "Logit", "auc")
    c2_auc = _metric(cloud_agg, "C2", "Logit", "auc")
    m0_auc = _metric(mask_agg, "M0", "Logit", "auc")
    m2_auc = _metric(mask_agg, "M2", "Logit", "auc")
    ur_full_auc = _metric(urban_agg, "UR_full", "Logit", "auc")

    moran_sig = moran[moran["p_value"] < 0.05] if not moran.empty else pd.DataFrame()
    pop_note = "NA"
    if not pop_quality.empty:
        hit = pop_quality[pop_quality["feature"] == "acs_pop_density"]
        if not hit.empty:
            pop_note = f"status={hit['status'].iloc[0]}, coverage={float(hit['coverage_ratio'].iloc[0]):.3f}"

    contrib_dedup = (
        contrib.groupby(["model", "feature"], as_index=False)
        .agg(
            effect_size=("effect_size", "mean"),
            loeo_sign_consistency=("loeo_sign_consistency", "mean"),
            net_contribution_label=("net_contribution_label", "first"),
        )
    )

    top_pos = (
        contrib_dedup[contrib_dedup["net_contribution_label"] == "positive_contribution"]
        .sort_values("effect_size", ascending=False)
        .head(5)
    )
    top_neg = (
        contrib_dedup[contrib_dedup["net_contribution_label"] == "negative_contribution"]
        .sort_values("effect_size")
        .head(5)
    )

    report = [
        "# Exploration Upgrade Report (V2) / 六事件探索增强报告（V2）",
        "",
        "## Objective",
        "在不扩事件集合前提下，验证云量、噪声 mask、城乡/人口、空间依赖与极端事件敏感性是否能解释并改善跨事件泛化表现。",
        "",
        "## Experiment Matrix",
        "- Cloud: C0/C1/C2/C3",
        "- Noise Mask: M0/M1/M2",
        "- Urban-Rural: UR_full/UR_urban/UR_rural",
        "- Spatial: Moran's I + cluster SE",
        "- Extreme sensitivity: drop-1/drop-2/control-drop",
        "",
        "## Cloud Importance Findings",
        f"- Logit AUC: C0={c0_auc}, C1={c1_auc}, C2={c2_auc}",
        "- 判定逻辑：若 AUC 提升但 Brier 恶化，则归类为有代价提升。",
        "",
        "## Noise Mask Findings",
        f"- Logit AUC: M0={m0_auc}, M2={m2_auc}",
        "- M1/M2 通过去除高噪声地物类别检验图像背景噪声对可泛化性的影响。",
        "",
        "## Urban-Rural + Population Findings",
        f"- UR_full Logit AUC={ur_full_auc}",
        f"- Population source quality: {pop_note}",
        "- 城乡与人口密度用于检验发电机相关韧性是否存在结构性分层。",
        "",
        "## Spatial Autocorrelation Findings",
        f"- Significant Moran's I events: {int(len(moran_sig))}/{int(len(moran)) if len(moran)>0 else 0}",
        "- 已输出 HC1 vs spatial cluster SE 对照，避免显著性高估。",
        "",
        "## Indicator Contribution Verdict",
        "### Positive contribution (top)",
    ]

    if top_pos.empty:
        report.append("- NA")
    else:
        for _, r in top_pos.iterrows():
            report.append(f"- {r['model']} | {r['feature']} | effect={r['effect_size']:.4f} | stability={r['loeo_sign_consistency']:.2f}")

    report.append("")
    report.append("### Negative contribution (top)")
    if top_neg.empty:
        report.append("- NA")
    else:
        for _, r in top_neg.iterrows():
            report.append(f"- {r['model']} | {r['feature']} | effect={r['effect_size']:.4f} | stability={r['loeo_sign_consistency']:.2f}")

    report.extend(
        [
            "",
            "## Extreme-event Identification & Drop Sensitivity",
            f"- Decision: {decision.get('decision', 'NA')}",
            f"- Extreme candidates: {', '.join(decision.get('extreme_candidates', [])) if decision.get('extreme_candidates') else 'none'}",
            "- 该结论仅用于敏感性，不改变主规格口径。",
            "",
            "## Key Outputs",
            "- `project/modeling/output/cloud_ablation_aggregate_metrics.csv`",
            "- `project/modeling/output/noise_mask_experiment_metrics.csv`",
            "- `project/modeling/output/urban_rural_model_comparison.csv`",
            "- `project/modeling/output/spatial_autocorr_morans_i.csv`",
            "- `project/modeling/output/feature_contribution_scorecard.csv`",
            "- `project/modeling/output/extreme_event_drop_aggregate_v1.csv`",
            "",
            "## Figures",
            "- `project/modeling_report/figures/exploration_v2/exploration_auc_compare.png`",
        ]
    )

    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def _load_exploration_inputs(ctx: RunContext) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not PANEL_IN_PATH.exists():
        raise FileNotFoundError(f"Missing panel input: {PANEL_IN_PATH}")

    panel = _filter_sample_lock(pd.read_parquet(PANEL_IN_PATH))
    panel = _prepare_noise_groups(panel)
    panel, pop_quality = attach_urban_population(panel, ctx)
    pop_quality.to_csv(POP_QUALITY_PATH, index=False)

    for c in ADDED_FIELDS:
        if c not in panel.columns:
            panel[c] = np.nan

    panel.to_parquet(PANEL_OUT_PATH, index=False)
    rec = build_recovery_from_panel(panel, ctx=ctx)
    return panel, rec, pop_quality


def _run_cloud_ablation_impl() -> int:
    ensure_directories()
    init_tracking_files()
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(issues=[])
    append_progress("Exploration V2: cloud ablation bundle started")
    panel, rec, _ = _load_exploration_inputs(ctx)
    run_cloud_ablation(panel, rec)
    save_issue_log(ctx)
    append_progress("Exploration V2: cloud ablation bundle completed")
    return 0


def _run_noise_mask_impl() -> int:
    ensure_directories()
    init_tracking_files()
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(issues=[])
    append_progress("Exploration V2: noise mask bundle started")
    panel, rec, _ = _load_exploration_inputs(ctx)
    run_noise_mask(panel, rec)
    save_issue_log(ctx)
    append_progress("Exploration V2: noise mask bundle completed")
    return 0


def _run_urban_rural_impl() -> int:
    ensure_directories()
    init_tracking_files()
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(issues=[])
    append_progress("Exploration V2: urban-rural bundle started")
    panel, rec, _ = _load_exploration_inputs(ctx)
    run_urban_population(panel, rec)
    save_issue_log(ctx)
    append_progress("Exploration V2: urban-rural bundle completed")
    return 0


def _load_required_coef_outputs() -> pd.DataFrame:
    required = [CLOUD_COEF_PATH, MASK_FULL_COEF_PATH, URBAN_COEF_PATH]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Spatial diagnostics requires existing cloud-ablation, noise-mask, and urban-rural coefficient outputs. "
            f"Missing: {missing}. Run those bundles first or use `run-v2`."
        )
    coef_parts = [pd.read_csv(CLOUD_COEF_PATH), pd.read_csv(MASK_FULL_COEF_PATH), pd.read_csv(URBAN_COEF_PATH)]
    return pd.concat(coef_parts, ignore_index=True)


def _run_spatial_diagnostics_impl() -> int:
    ensure_directories()
    init_tracking_files()
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(issues=[])
    append_progress("Exploration V2: spatial diagnostics bundle started")
    panel, _, _ = _load_exploration_inputs(ctx)
    coef_all = _load_required_coef_outputs()
    run_spatial_and_contribution(panel, coef_all)
    save_issue_log(ctx)
    append_progress("Exploration V2: spatial diagnostics bundle completed")
    return 0


def _run_extreme_event_sensitivity_impl() -> int:
    ensure_directories()
    init_tracking_files()
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(issues=[])
    append_progress("Exploration V2: extreme-event sensitivity bundle started")
    panel, rec, _ = _load_exploration_inputs(ctx)
    _, score = _compute_extreme_candidates()
    if score.empty:
        pd.DataFrame(columns=["event_id", "high_shift_flag", "poor_perf_flag", "extreme_candidate"]).to_csv(EXTREME_CANDIDATE_PATH, index=False)
        pd.DataFrame(columns=["event_id", "extreme_score", "source_ref"]).to_csv(EXTREME_SCORE_PATH, index=False)
        pd.DataFrame(columns=["experiment_family", "spec_id", "fold_event", "model", "rmse", "mae", "auc", "brier", "c_index", "coef_in_buffer", "notes"]).to_csv(EXTREME_DROP_METRIC_PATH, index=False)
        pd.DataFrame(columns=["experiment_family", "spec_id", "model", "rmse", "mae", "auc", "brier", "c_index", "coef_in_buffer", "n_folds"]).to_csv(EXTREME_DROP_AGG_PATH, index=False)
        EXTREME_DECISION_PATH.write_text(json.dumps({"decision": "missing_baseline_files", "extreme_candidates": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        run_extreme_drop_sensitivity(panel, rec, score)
    save_issue_log(ctx)
    append_progress("Exploration V2: extreme-event sensitivity bundle completed")
    return 0


def _run_v2_impl() -> int:
    ensure_directories()
    init_tracking_files()
    FIG_EXP_DIR.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(issues=[])

    append_progress("Exploration V2 pipeline started")

    panel, rec, pop_quality = _load_exploration_inputs(ctx)

    append_progress("Exploration V2: cloud ablation")
    cloud_fold, cloud_agg, cloud_coef = run_cloud_ablation(panel, rec)

    append_progress("Exploration V2: noise masking")
    mask_fold, mask_agg, mask_coef = run_noise_mask(panel, rec)

    append_progress("Exploration V2: urban-rural + population")
    urban_fold, urban_agg, urban_coef = run_urban_population(panel, rec)

    coef_all = pd.concat([cloud_coef, mask_coef, urban_coef], ignore_index=True)

    append_progress("Exploration V2: spatial diagnostics + contribution")
    moran, se_cmp, contrib, contrib_rank = run_spatial_and_contribution(panel, coef_all)

    append_progress("Exploration V2: extreme-event sensitivity")
    candidates, score = _compute_extreme_candidates()
    if score.empty:
        # safe fallback if baseline files unavailable
        pd.DataFrame(columns=["event_id", "high_shift_flag", "poor_perf_flag", "extreme_candidate"]).to_csv(EXTREME_CANDIDATE_PATH, index=False)
        pd.DataFrame(columns=["event_id", "extreme_score", "source_ref"]).to_csv(EXTREME_SCORE_PATH, index=False)
        extreme_fold = pd.DataFrame()
        extreme_agg = pd.DataFrame()
        decision = {"decision": "missing_baseline_files", "extreme_candidates": []}
        EXTREME_DECISION_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        extreme_fold, extreme_agg, decision = run_extreme_drop_sensitivity(panel, rec, score)

    append_progress("Exploration V2: write plan docs/report/index")
    _write_future_plan_files()
    _plot_experiment_summary(cloud_agg, mask_agg, urban_agg)
    _write_report(cloud_agg, mask_agg, urban_agg, moran, contrib, extreme_agg, decision, pop_quality)
    _update_index()

    save_issue_log(ctx)
    append_progress("Exploration V2 pipeline completed")

    return 0
# ----------------------------
# Unified CLI
# ----------------------------

import argparse


def cmd_run_v2() -> int:
    return _run_v2_impl()


def cmd_quality_matched_v1() -> int:
    return _run_quality_matched_v1_impl()


def cmd_hazard_mainline_v1() -> int:
    return _run_hazard_mainline_v1_impl()


def cmd_hazard_readiness_v1() -> int:
    return _run_hazard_readiness_v1_impl()


def cmd_bug2_pr_pilot_v1() -> int:
    return _run_bug2_pr_pilot_v1_impl()


def cmd_event_expansion_v1() -> int:
    return _run_event_expansion_v1_impl()


def cmd_full_run() -> int:
    return _run_v2_impl()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Unified exploration/sensitivity entrypoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('run-v2', help='Run full exploration V2 pipeline')
    sub.add_parser('cloud-ablation', help='Run exploration V2 cloud ablation bundle')
    sub.add_parser('noise-mask', help='Run exploration V2 noise mask bundle')
    sub.add_parser('urban-rural', help='Run exploration V2 urban-rural bundle')
    sub.add_parser('spatial-diagnostics', help='Run exploration V2 spatial diagnostics bundle')
    sub.add_parser('extreme-event-sensitivity', help='Run exploration V2 extreme-event sensitivity bundle')
    sub.add_parser('quality-matched-v1', help='Run quality-adjusted target + spatial block + facility-matched bundle')
    sub.add_parser('hazard-mainline-v1', help='Run hazard/exposure-aware transport mainline on quality-adjusted panel')
    sub.add_parser('hazard-readiness-v1', help='Run readiness-filtered HZ1 rerun on the current mainline-ready event subset')
    sub.add_parser('bug-transport-v1', help='Run BUG-aware transport family on the quality-adjusted panel')
    sub.add_parser('bug2-pr-pilot-v1', help='Run the Puerto Rico official-inventory pilot setup and, if data exists, the local BUG2 model')
    sub.add_parser('event-expansion-v1', help='Run staged event expansion with selective sync, online acquisition, and retraining')
    sub.add_parser('full-run', help='Run full exploration V2 pipeline')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'cloud-ablation':
        return _run_cloud_ablation_impl()
    if args.command == 'noise-mask':
        return _run_noise_mask_impl()
    if args.command == 'urban-rural':
        return _run_urban_rural_impl()
    if args.command == 'spatial-diagnostics':
        return _run_spatial_diagnostics_impl()
    if args.command == 'extreme-event-sensitivity':
        return _run_extreme_event_sensitivity_impl()
    if args.command == 'quality-matched-v1':
        return _run_quality_matched_v1_impl()
    if args.command == 'hazard-mainline-v1':
        return _run_hazard_mainline_v1_impl()
    if args.command == 'hazard-readiness-v1':
        return _run_hazard_readiness_v1_impl()
    if args.command == 'bug-transport-v1':
        return _run_bug_transport_v1_impl()
    if args.command == 'bug2-pr-pilot-v1':
        return _run_bug2_pr_pilot_v1_impl()
    if args.command == 'event-expansion-v1':
        return _run_event_expansion_v1_impl()
    if args.command in {'run-v2', 'full-run'}:
        return _run_v2_impl()
    parser.error(f'Unknown command: {args.command}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
