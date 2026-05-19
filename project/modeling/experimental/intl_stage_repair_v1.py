#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Geod
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_PATH = Path(__file__).resolve()
MODELING_DIR = SCRIPT_PATH.parents[1]
ROOT = SCRIPT_PATH.parents[3]
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))


def _load_exploration_module():
    path = MODELING_DIR / "pipelines" / "03_exploration_pipeline.py"
    spec = importlib.util.spec_from_file_location("_intl_repair_exp3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load exploration module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP = _load_exploration_module()

OUTPUT_DIR = ROOT / "project" / "modeling" / "output"
PIXEL_DIR = ROOT / "project" / "modeling" / "pixel_data"
REPORT_DIR = ROOT / "project" / "modeling_report"
FIG_DIR = REPORT_DIR / "figures" / "intl_stage_repair_v1"
CONFIG_DIR = ROOT / "project" / "modeling" / "config"
WORLDPOP_DIR = ROOT / "project" / "data" / "external" / "worldpop"

CONFIG_PATH = CONFIG_DIR / "intl_stage_repair_v1.json"
RULES_PATH = CONFIG_DIR / "readiness_score_rules_v1.json"
EVENTS10_PATH = CONFIG_DIR / "events_10.json"
INPUT_GATE_PATH = OUTPUT_DIR / "new_event_input_gate_v1.csv"
NEW_POI_QUALITY_PATH = OUTPUT_DIR / "new_event_poi_quality_v1.csv"
EVENT_INCREMENT_METRICS_PATH = OUTPUT_DIR / "event_increment_model_metrics_v1.csv"
REPORT_INDEX_PATH = REPORT_DIR / "index.md"

INTL_COV_MANIFEST_PATH = OUTPUT_DIR / "intl_covariate_manifest_v1.csv"
INTL_COV_QUALITY_PATH = OUTPUT_DIR / "intl_covariate_quality_v1.csv"
INTL_COMPARE_PATH = OUTPUT_DIR / "intl_stage_repair_comparison_v1.csv"
READINESS_COMPONENTS_PATH = OUTPUT_DIR / "event_readiness_components_v1.csv"
READINESS_SCORE_PATH = OUTPUT_DIR / "event_readiness_score_v1.csv"
TRAINING_DECISION_PATH = OUTPUT_DIR / "event_training_decision_v1.csv"
REPORT_PATH = REPORT_DIR / "14_intl_stage_repair_report.md"

GEOD = Geod(ellps="WGS84")
WORLDPOP_META_TMPL = "https://www.worldpop.org/rest/data/pop/wpgp?iso3={iso3}"

HZ2_NUMERIC = [
    "pixel_cloud_proxy",
    "recovery_obs_quality_score",
    "urban_share_1km",
    "water_share_1km",
    "developed_high_share_1km",
    "pop_density_log1p_v2",
    "in_buffer_x_pop_v2",
]
HZ2_SURV_NUMERIC = [
    "pixel_cloud_proxy",
    "recovery_obs_quality_score",
    "urban_share_1km",
    "water_share_1km",
    "developed_high_share_1km",
    "pop_density_log1p_v2",
]
HZ2_CATEGORICAL = ["land_use_group", "event_disaster_type"]

STAGE_IDS = ["stage_9_earthquake_hatay", "stage_10_dorian_freeport"]


def _pretty_stage_label(label: str) -> str:
    label = str(label)
    m = label.split("_", 2)
    if len(m) >= 3 and m[0] == "stage":
        suffix = m[2].replace("_", " ").title()
        return f"Stage {m[1]}\n{suffix}"
    return label.replace("_", " ")


def _pretty_event_label(label: str) -> str:
    return str(label).replace("_", "\n").title()


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


FEATURE_STAGE_PATHS = {
    stage: PIXEL_DIR / f"all_events_pixel_panel_v1_feature_upgrade_{stage}.parquet" for stage in STAGE_IDS
}
QUALITY_STAGE_PATHS = {
    stage: PIXEL_DIR / f"all_events_pixel_panel_v1_quality_v1_{stage}.parquet" for stage in STAGE_IDS
}
RECOVERY_STAGE_PATHS = {
    stage: PIXEL_DIR / f"recovery_daily_panel_v2_{stage}.parquet" for stage in STAGE_IDS
}
PROFILE_STAGE_PATHS = {
    stage: PIXEL_DIR / f"event_profile_v1_{stage}.csv" for stage in STAGE_IDS
}
TARGET_AUDIT_STAGE_PATHS = {
    stage: OUTPUT_DIR / f"target_quality_audit_{stage}.csv" for stage in STAGE_IDS
}
HZ1_AGG_STAGE_PATHS = {
    stage: OUTPUT_DIR / f"hazard_transport_aggregate_metrics_v1_{stage}.csv" for stage in STAGE_IDS
}
STRICT_SUMMARY_STAGE_PATHS = {
    stage: OUTPUT_DIR / f"model_summary_feature_upgrade_v2_strict_{stage}.csv" for stage in STAGE_IDS
}

REPAIRED_FEATURE_STAGE_PATHS = {
    stage: PIXEL_DIR / f"all_events_pixel_panel_v1_feature_upgrade_{stage}_intl_repair_v1.parquet" for stage in STAGE_IDS
}
REPAIRED_QUALITY_STAGE_PATHS = {
    stage: PIXEL_DIR / f"all_events_pixel_panel_v1_quality_v1_{stage}_intl_repair_v1.parquet" for stage in STAGE_IDS
}
REPAIRED_RECOVERY_STAGE_PATHS = {
    stage: PIXEL_DIR / f"recovery_daily_panel_v2_{stage}_intl_repair_v1.parquet" for stage in STAGE_IDS
}
REPAIRED_PROFILE_STAGE_PATHS = {
    stage: PIXEL_DIR / f"event_profile_v1_{stage}_intl_repair_v1.csv" for stage in STAGE_IDS
}
HZ2_AGG_STAGE_PATHS = {
    stage: OUTPUT_DIR / f"hazard_transport_aggregate_metrics_v2_{stage}.csv" for stage in STAGE_IDS
}
HZ2_FOLD_STAGE_PATHS = {
    stage: OUTPUT_DIR / f"hazard_transport_fold_metrics_v2_{stage}.csv" for stage in STAGE_IDS
}
HZ2_FEATURE_STAGE_PATHS = {
    stage: OUTPUT_DIR / f"hazard_transport_feature_summary_v2_{stage}.csv" for stage in STAGE_IDS
}


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dirs() -> None:
    WORLDPOP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (MODELING_DIR / "experimental").mkdir(parents=True, exist_ok=True)


def _append_progress(message: str) -> None:
    try:
        EXP.append_progress(message)
    except Exception:
        pass
    print(message)


def _worldpop_item(iso3: str, year: int) -> Dict[str, object]:
    resp = requests.get(WORLDPOP_META_TMPL.format(iso3=iso3), timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise RuntimeError(f"worldpop_metadata_missing_data:{iso3}")
    for item in data:
        if str(item.get("popyear")) == str(year):
            return item
    raise RuntimeError(f"worldpop_year_not_found:{iso3}:{year}")


def _download_worldpop_raster(iso3: str, year: int, cache_path: Path) -> Dict[str, object]:
    item = _worldpop_item(iso3, year)
    files = item.get("files") if isinstance(item.get("files"), list) else []
    raster_url = files[0] if files else f"https://data.worldpop.org/{item['data_file']}"
    metadata_url = WORLDPOP_META_TMPL.format(iso3=iso3)

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return {
            "iso3": iso3,
            "year": year,
            "metadata_url": metadata_url,
            "raster_url": raster_url,
            "cache_path": cache_path,
            "request_status": "ok",
            "download_status": "cached",
            "file_size_bytes": int(cache_path.stat().st_size),
        }

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    with requests.get(raster_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp_path.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp_path.replace(cache_path)
    return {
        "iso3": iso3,
        "year": year,
        "metadata_url": metadata_url,
        "raster_url": raster_url,
        "cache_path": cache_path,
        "request_status": "ok",
        "download_status": "downloaded",
        "file_size_bytes": int(cache_path.stat().st_size),
    }


def _cell_area_km2(src: rasterio.DatasetReader, row: int) -> float:
    x0, y0 = src.transform * (0, row)
    x1, y1 = src.transform * (1, row + 1)
    lon = [x0, x1, x1, x0]
    lat = [y0, y0, y1, y1]
    area_m2, _ = GEOD.polygon_area_perimeter(lon, lat)
    return abs(area_m2) / 1_000_000.0


def _sample_worldpop_density(cache_path: Path, points: pd.DataFrame) -> pd.DataFrame:
    out = points[["pixel_id", "event_id", "lon", "lat"]].copy()
    out["pop_density_per_km2_v2"] = np.nan
    out["pop_density_log1p_v2"] = np.nan
    out["missing_pop_flag_v2"] = 1
    if out.empty:
        return out

    with rasterio.open(cache_path) as src:
        nodata = src.nodata
        coords = list(zip(pd.to_numeric(out["lon"], errors="coerce"), pd.to_numeric(out["lat"], errors="coerce")))
        samples = np.array([val[0] for val in src.sample(coords)], dtype="float64")
        rc = [src.index(lon, lat) if np.isfinite(lon) and np.isfinite(lat) else (-1, -1) for lon, lat in coords]
        rows = np.array([r for r, _ in rc], dtype=int)
        cols = np.array([c for _, c in rc], dtype=int)
        row_area_cache: Dict[int, float] = {}
        density = np.full(len(out), np.nan, dtype="float64")

        for i, (sample, row, col) in enumerate(zip(samples, rows, cols)):
            if row < 0 or col < 0:
                continue
            if nodata is not None and np.isclose(sample, nodata):
                continue
            if not np.isfinite(sample):
                continue
            if row not in row_area_cache:
                row_area_cache[row] = _cell_area_km2(src, row)
            area = row_area_cache[row]
            if area <= 0 or not np.isfinite(area):
                continue
            density[i] = max(float(sample), 0.0) / area

    out["pop_density_per_km2_v2"] = density
    out["pop_density_log1p_v2"] = np.log1p(np.clip(density, a_min=0.0, a_max=None))
    out["missing_pop_flag_v2"] = np.where(np.isfinite(density), 0, 1).astype(int)
    return out


def _legacy_pop_issue(df: pd.DataFrame, event_id: str) -> Dict[str, object]:
    sub = df[df["event_id"] == event_id].copy()
    vals = pd.to_numeric(sub.get("pop_density_per_km2"), errors="coerce") if "pop_density_per_km2" in sub.columns else pd.Series(dtype=float)
    flags = pd.to_numeric(sub.get("missing_pop_flag"), errors="coerce") if "missing_pop_flag" in sub.columns else pd.Series(dtype=float)
    nonmissing = int(vals.notna().sum())
    mismatch = bool(nonmissing > 0 and float(flags.fillna(1).mean()) >= 0.99)
    return {
        "legacy_nonmissing_count": nonmissing,
        "legacy_unique_nonmissing": int(vals.dropna().nunique()) if nonmissing else 0,
        "legacy_missing_pop_flag_mean": float(flags.fillna(1).mean()) if len(flags) else np.nan,
        "legacy_issue_detected": int(mismatch),
    }


def _map_us_v2_columns(df: pd.DataFrame, events_cfg: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    numeric_defaults = {
        "pop_density_per_km2_v2": np.nan,
        "pop_density_log1p_v2": np.nan,
        "missing_pop_flag_v2": np.nan,
        "covariate_integrity_flag_v2": np.nan,
    }
    string_defaults = {
        "pop_source_v2": "",
        "urban_source_v2": "",
    }
    for col, default in {**numeric_defaults, **string_defaults}.items():
        if col not in out.columns:
            out[col] = default
    us_mask = out["event_id"].map(lambda x: str(events_cfg[str(x)].get("country_scope", "US")) == "US")
    out.loc[us_mask, "pop_density_per_km2_v2"] = pd.to_numeric(out.loc[us_mask, "pop_density_per_km2"], errors="coerce")
    out.loc[us_mask, "pop_density_log1p_v2"] = pd.to_numeric(out.loc[us_mask, "pop_density_log1p"], errors="coerce")
    out.loc[us_mask, "missing_pop_flag_v2"] = pd.to_numeric(out.loc[us_mask, "missing_pop_flag"], errors="coerce").fillna(1).astype(int)
    out.loc[us_mask, "pop_source_v2"] = "acs_2022_existing"
    out.loc[us_mask, "urban_source_v2"] = "cbsa_uac20_tiger_existing"
    out.loc[us_mask, "covariate_integrity_flag_v2"] = np.where(out.loc[us_mask, "missing_pop_flag_v2"] == 0, 1, 0)
    return out


def _repair_stage_panel(stage_id: str, worldpop_cache: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, object]], List[Dict[str, object]]]:
    events_cfg = _read_json(EVENTS10_PATH)
    feature_df = pd.read_parquet(FEATURE_STAGE_PATHS[stage_id])
    quality_df = pd.read_parquet(QUALITY_STAGE_PATHS[stage_id])
    recovery_df = pd.read_parquet(RECOVERY_STAGE_PATHS[stage_id])
    profile_df = pd.read_csv(PROFILE_STAGE_PATHS[stage_id])

    feature_df = _map_us_v2_columns(feature_df, events_cfg)
    quality_df = _map_us_v2_columns(quality_df, events_cfg)

    cfg = _read_json(CONFIG_PATH)
    repair_events = cfg["stages"][stage_id]["repair_events"]
    quality_rows: List[Dict[str, object]] = []
    source_rows: List[Dict[str, object]] = []

    for event_id in repair_events:
        info = cfg["worldpop"]["countries"][event_id]
        cache_path = worldpop_cache[event_id]
        for df_name, df in [("feature", feature_df), ("quality", quality_df)]:
            mask = df["event_id"] == event_id
            sampled = _sample_worldpop_density(cache_path, df.loc[mask, ["pixel_id", "event_id", "lon", "lat"]])
            sampled["pixel_id"] = sampled["pixel_id"].astype(str)
            sampled = sampled.set_index("pixel_id")
            df.loc[mask, "pop_density_per_km2_v2"] = df.loc[mask, "pixel_id"].astype(str).map(sampled["pop_density_per_km2_v2"])
            df.loc[mask, "pop_density_log1p_v2"] = df.loc[mask, "pixel_id"].astype(str).map(sampled["pop_density_log1p_v2"])
            df.loc[mask, "missing_pop_flag_v2"] = df.loc[mask, "pixel_id"].astype(str).map(sampled["missing_pop_flag_v2"]).fillna(1).astype(int)
            df.loc[mask, "pop_source_v2"] = "worldpop_raster_2020"
            df.loc[mask, "urban_source_v2"] = "osm_landuse_proxy_existing"
            df.loc[mask, "covariate_integrity_flag_v2"] = np.where(df.loc[mask, "missing_pop_flag_v2"] == 0, 1, 0)
            if df_name == "quality":
                legacy = _legacy_pop_issue(df, event_id)
                sub = df.loc[mask]
                quality_rows.append(
                    {
                        "stage_id": stage_id,
                        "event_id": event_id,
                        **legacy,
                        "v2_missing_pop_flag_mean": float(pd.to_numeric(sub["missing_pop_flag_v2"], errors="coerce").mean()),
                        "v2_unique_nonmissing": int(pd.to_numeric(sub["pop_density_per_km2_v2"], errors="coerce").dropna().nunique()),
                        "v2_mean": float(pd.to_numeric(sub["pop_density_per_km2_v2"], errors="coerce").dropna().mean()) if pd.to_numeric(sub["pop_density_per_km2_v2"], errors="coerce").notna().any() else np.nan,
                        "quality_status": "ok" if float(pd.to_numeric(sub["missing_pop_flag_v2"], errors="coerce").mean()) < 0.05 and int(pd.to_numeric(sub["pop_density_per_km2_v2"], errors="coerce").dropna().nunique()) > 1 else "repair_first",
                    }
                )
                source_rows.append(
                    {
                        "event_id": event_id,
                        "covariate_name": "pop_density_per_km2_v2",
                        "source_name": f"worldpop_{info['iso3']}_{info['year']}",
                        "source_type": "raster_sample",
                        "spatial_resolution": "3_arc_sec",
                        "temporal_reference": str(info["year"]),
                        "is_us_only": 0,
                        "used_in_mainline": 1,
                        "quality_flag": quality_rows[-1]["quality_status"],
                    }
                )

    feature_df.to_parquet(REPAIRED_FEATURE_STAGE_PATHS[stage_id], index=False)
    quality_df.to_parquet(REPAIRED_QUALITY_STAGE_PATHS[stage_id], index=False)

    if "recovery_obs_quality_score_x" in recovery_df.columns and "recovery_obs_quality_score" not in recovery_df.columns:
        recovery_df = recovery_df.rename(columns={"recovery_obs_quality_score_x": "recovery_obs_quality_score"})
    if "high_censoring_risk_flag_x" in recovery_df.columns and "high_censoring_risk_flag" not in recovery_df.columns:
        recovery_df = recovery_df.rename(columns={"high_censoring_risk_flag_x": "high_censoring_risk_flag"})

    merge_cols = [
        "pixel_id",
        "event_id",
        "in_buffer",
        "pre_mean_ntl",
        "land_use_group",
        "event_disaster_type",
        "pixel_cloud_proxy",
        "recovery_obs_quality_score",
        "urban_share_1km",
        "water_share_1km",
        "developed_high_share_1km",
        "high_censoring_risk_flag",
        "pop_density_per_km2_v2",
        "pop_density_log1p_v2",
        "missing_pop_flag_v2",
        "pop_source_v2",
        "urban_source_v2",
        "covariate_integrity_flag_v2",
    ]
    merge_cols = [c for c in merge_cols if c in quality_df.columns]
    recovery_core = recovery_df.drop(columns=[c for c in merge_cols if c in recovery_df.columns and c not in {"pixel_id", "event_id"}], errors="ignore")
    recovery_repaired = recovery_core.merge(quality_df[merge_cols].drop_duplicates(subset=["pixel_id", "event_id"]), on=["pixel_id", "event_id"], how="left")
    recovery_repaired.to_parquet(REPAIRED_RECOVERY_STAGE_PATHS[stage_id], index=False)

    profile_df = profile_df.copy()
    quality_stats = (
        quality_df.groupby("event_id", as_index=False)
        .agg(
            pop_source_v2=("pop_source_v2", lambda s: next((str(v) for v in s if pd.notna(v) and str(v) != ""), "missing")),
            urban_source_v2=("urban_source_v2", lambda s: next((str(v) for v in s if pd.notna(v) and str(v) != ""), "missing")),
            covariate_integrity_flag_v2=("covariate_integrity_flag_v2", "mean"),
            pop_missing_share_v2=("missing_pop_flag_v2", "mean"),
            pop_density_mean_v2=("pop_density_per_km2_v2", "mean"),
        )
    )
    profile_df = profile_df.merge(quality_stats, on="event_id", how="left")
    profile_df["covariate_integrity_flag_v2"] = np.where(pd.to_numeric(profile_df["covariate_integrity_flag_v2"], errors="coerce") >= 0.95, 1, 0)
    profile_df.to_csv(REPAIRED_PROFILE_STAGE_PATHS[stage_id], index=False)

    return feature_df, quality_df, recovery_repaired, profile_df, quality_rows, source_rows


def _evaluate_fold_hz2(train_df: pd.DataFrame, test_df: pd.DataFrame, train_rec: pd.DataFrame, test_rec: pd.DataFrame, fold_event: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []

    linear_num, linear_cat = EXP._prune_terms(train_df, HZ2_NUMERIC, HZ2_CATEGORICAL)
    formula = EXP._build_formula("delta_ntl", linear_num, linear_cat)

    try:
        ols = smf.ols(formula, data=train_df).fit(cov_type="HC1")
        pred = ols.predict(test_df)
        rows.append({
            "experiment_family": "hazard_transport",
            "spec_id": "HZ2",
            "fold_event": fold_event,
            "model": "OLS",
            "rmse": float(np.sqrt(mean_squared_error(test_df["delta_ntl"], pred))),
            "mae": float(mean_absolute_error(test_df["delta_ntl"], pred)),
            "auc": np.nan,
            "brier": np.nan,
            "c_index": np.nan,
            "coef_in_buffer": float(ols.params.get("in_buffer", np.nan)),
            "notes": "ok",
        })
        for term, val in ols.params.items():
            coef_rows.append({"experiment_family": "hazard_transport", "spec_id": "HZ2", "fold_event": fold_event, "model": "OLS", "feature": term, "coef": float(val), "p_value": float(ols.pvalues.get(term, np.nan))})
    except Exception as exc:  # noqa: BLE001
        rows.append({"experiment_family": "hazard_transport", "spec_id": "HZ2", "fold_event": fold_event, "model": "OLS", "rmse": np.nan, "mae": np.nan, "auc": np.nan, "brier": np.nan, "c_index": np.nan, "coef_in_buffer": np.nan, "notes": f"fail:{type(exc).__name__}"})

    try:
        mixed = smf.mixedlm(formula, data=train_df, groups=train_df["event_id"]).fit(reml=False, method="lbfgs", maxiter=200)
        pred = mixed.predict(test_df)
        rows.append({
            "experiment_family": "hazard_transport",
            "spec_id": "HZ2",
            "fold_event": fold_event,
            "model": "MixedLM",
            "rmse": float(np.sqrt(mean_squared_error(test_df["delta_ntl"], pred))),
            "mae": float(mean_absolute_error(test_df["delta_ntl"], pred)),
            "auc": np.nan,
            "brier": np.nan,
            "c_index": np.nan,
            "coef_in_buffer": float(mixed.params.get("in_buffer", np.nan)),
            "notes": "ok",
        })
        for term, val in mixed.params.items():
            if term == "Group Var":
                continue
            coef_rows.append({"experiment_family": "hazard_transport", "spec_id": "HZ2", "fold_event": fold_event, "model": "MixedLM", "feature": term, "coef": float(val), "p_value": float(mixed.pvalues.get(term, np.nan))})
    except Exception as exc:  # noqa: BLE001
        rows.append({"experiment_family": "hazard_transport", "spec_id": "HZ2", "fold_event": fold_event, "model": "MixedLM", "rmse": np.nan, "mae": np.nan, "auc": np.nan, "brier": np.nan, "c_index": np.nan, "coef_in_buffer": np.nan, "notes": f"fail:{type(exc).__name__}"})

    try:
        xtr, xte, feat_names = EXP._build_logit_design(train_df, test_df, linear_num, linear_cat)
        ytr = EXP._safe_numeric(train_df["is_damaged"]).astype(int)
        yte = EXP._safe_numeric(test_df["is_damaged"]).astype(int)
        logit = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", class_weight="balanced")
        logit.fit(xtr, ytr)
        prob = logit.predict_proba(xte)[:, 1]
        auc = float(roc_auc_score(yte, prob)) if yte.nunique() > 1 else np.nan
        brier = float(brier_score_loss(yte, prob))
        coef_in_buffer = float(logit.coef_[0, feat_names.index("in_buffer")]) if "in_buffer" in feat_names else np.nan
        rows.append({"experiment_family": "hazard_transport", "spec_id": "HZ2", "fold_event": fold_event, "model": "Logit", "rmse": np.nan, "mae": np.nan, "auc": auc, "brier": brier, "c_index": np.nan, "coef_in_buffer": coef_in_buffer, "notes": "ok"})
        for idx, name in enumerate(feat_names):
            coef_rows.append({"experiment_family": "hazard_transport", "spec_id": "HZ2", "fold_event": fold_event, "model": "Logit", "feature": name, "coef": float(logit.coef_[0, idx]), "p_value": np.nan})
    except Exception as exc:  # noqa: BLE001
        rows.append({"experiment_family": "hazard_transport", "spec_id": "HZ2", "fold_event": fold_event, "model": "Logit", "rmse": np.nan, "mae": np.nan, "auc": np.nan, "brier": np.nan, "c_index": np.nan, "coef_in_buffer": np.nan, "notes": f"fail:{type(exc).__name__}"})

    try:
        surv_num, surv_cat = EXP._prune_terms(train_rec, HZ2_SURV_NUMERIC, HZ2_CATEGORICAL)
        s_tr = EXP._build_survival_design(train_rec, surv_num, surv_cat)
        s_te = EXP._build_survival_design(test_rec, surv_num, surv_cat)
        for c in [c for c in s_tr.columns if c not in {"recovery_days", "event_observed", "event_id"}]:
            if c not in s_te.columns:
                s_te[c] = 0.0
        s_te = s_te.reindex(columns=s_tr.columns, fill_value=0.0)

        cox = CoxPHFitter(penalizer=0.01)
        cox.fit(s_tr.drop(columns=["event_id"], errors="ignore"), duration_col="recovery_days", event_col="event_observed")
        risk = cox.predict_partial_hazard(s_te.drop(columns=["recovery_days", "event_observed", "event_id"], errors="ignore"))
        c_idx = float(concordance_index(s_te["recovery_days"], -risk.to_numpy().reshape(-1), s_te["event_observed"]))
        rows.append({"experiment_family": "hazard_transport", "spec_id": "HZ2", "fold_event": fold_event, "model": "Cox", "rmse": np.nan, "mae": np.nan, "auc": np.nan, "brier": np.nan, "c_index": c_idx, "coef_in_buffer": float(cox.params_.get("in_buffer", np.nan)), "notes": "ok"})
        for term, val in cox.params_.items():
            coef_rows.append({"experiment_family": "hazard_transport", "spec_id": "HZ2", "fold_event": fold_event, "model": "Cox", "feature": term, "coef": float(val), "p_value": float(cox.summary.loc[term, "p"]) if term in cox.summary.index else np.nan})

        aft = WeibullAFTFitter(penalizer=0.01)
        aft.fit(s_tr.drop(columns=["event_id"], errors="ignore"), duration_col="recovery_days", event_col="event_observed")
        med = aft.predict_median(s_te.drop(columns=["recovery_days", "event_observed", "event_id"], errors="ignore"))
        c_idx_aft = float(concordance_index(s_te["recovery_days"], -med.to_numpy().reshape(-1), s_te["event_observed"]))
        cands = [i for i in aft.params_.index if i[0] == "lambda_" and i[1] == "in_buffer"]
        rows.append({"experiment_family": "hazard_transport", "spec_id": "HZ2", "fold_event": fold_event, "model": "AFT", "rmse": np.nan, "mae": np.nan, "auc": np.nan, "brier": np.nan, "c_index": c_idx_aft, "coef_in_buffer": float(aft.params_.loc[cands[0]]) if cands else np.nan, "notes": "ok"})
    except Exception as exc:  # noqa: BLE001
        rows.append({"experiment_family": "hazard_transport", "spec_id": "HZ2", "fold_event": fold_event, "model": "Cox", "rmse": np.nan, "mae": np.nan, "auc": np.nan, "brier": np.nan, "c_index": np.nan, "coef_in_buffer": np.nan, "notes": f"fail:{type(exc).__name__}"})
        rows.append({"experiment_family": "hazard_transport", "spec_id": "HZ2", "fold_event": fold_event, "model": "AFT", "rmse": np.nan, "mae": np.nan, "auc": np.nan, "brier": np.nan, "c_index": np.nan, "coef_in_buffer": np.nan, "notes": f"fail:{type(exc).__name__}"})

    return rows, coef_rows


def _run_hz2(stage_id: str, quality_df: pd.DataFrame, recovery_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = quality_df.copy()
    recovery = recovery_df.copy()
    for df in [panel, recovery]:
        df["in_buffer_x_pop_v2"] = EXP._safe_numeric(df["in_buffer"]) * EXP._safe_numeric(df["pop_density_log1p_v2"])

    fold_rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []
    events = sorted(panel["event_id"].dropna().astype(str).unique())
    for fold_event in events:
        tr = panel[panel["event_id"] != fold_event].copy()
        te = panel[panel["event_id"] == fold_event].copy()
        tr_rec = recovery[recovery["event_id"] != fold_event].copy()
        te_rec = recovery[recovery["event_id"] == fold_event].copy()
        tr = EXP._prepare_columns(tr, HZ2_NUMERIC, HZ2_CATEGORICAL)
        te = EXP._prepare_columns(te, HZ2_NUMERIC, HZ2_CATEGORICAL)
        tr_rec = EXP._prepare_columns(tr_rec, HZ2_SURV_NUMERIC, HZ2_CATEGORICAL)
        te_rec = EXP._prepare_columns(te_rec, HZ2_SURV_NUMERIC, HZ2_CATEGORICAL)
        rows, coefs = _evaluate_fold_hz2(tr, te, tr_rec, te_rec, fold_event)
        fold_rows.extend(rows)
        coef_rows.extend(coefs)

    fold_df = pd.DataFrame(fold_rows)
    agg_df = EXP._aggregate_metrics(fold_df)
    coef_df = pd.DataFrame(coef_rows)
    fold_df.to_csv(HZ2_FOLD_STAGE_PATHS[stage_id], index=False)
    agg_df.to_csv(HZ2_AGG_STAGE_PATHS[stage_id], index=False)

    feature_summary = (
        coef_df.groupby(["model", "feature"], as_index=False)
        .agg(mean_coef=("coef", "mean"), mean_abs_coef=("coef", lambda s: np.mean(np.abs(pd.to_numeric(s, errors="coerce")))), folds=("fold_event", "nunique"))
        .sort_values(["model", "mean_abs_coef"], ascending=[True, False])
    ) if not coef_df.empty else pd.DataFrame(columns=["model", "feature", "mean_coef", "mean_abs_coef", "folds"])
    feature_summary.to_csv(HZ2_FEATURE_STAGE_PATHS[stage_id], index=False)
    return fold_df, agg_df, feature_summary


def _quality_status_from_row(row: pd.Series) -> str:
    miss = float(pd.to_numeric(pd.Series([row.get("v2_missing_pop_flag_mean")]), errors="coerce").iloc[0]) if pd.notna(row.get("v2_missing_pop_flag_mean")) else 1.0
    uniq = int(row.get("v2_unique_nonmissing", 0) or 0)
    return "ok" if miss < 0.05 and uniq > 1 else "repair_first"


def _prepare_covariates() -> pd.DataFrame:
    _ensure_dirs()
    cfg = _read_json(CONFIG_PATH)
    rows = []
    for event_id, info in cfg["worldpop"]["countries"].items():
        cache_path = WORLDPOP_DIR / str(info["cache_name"])
        item = _download_worldpop_raster(str(info["iso3"]), int(info["year"]), cache_path)
        rows.append({
            "event_id": event_id,
            "iso3": item["iso3"],
            "year": item["year"],
            "metadata_url": item["metadata_url"],
            "raster_url": item["raster_url"],
            "cache_path": _safe_rel(item["cache_path"]),
            "request_status": item["request_status"],
            "download_status": item["download_status"],
            "file_size_bytes": item["file_size_bytes"],
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(INTL_COV_MANIFEST_PATH, index=False)
    return manifest


def _repair_stage(stage_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = _read_json(CONFIG_PATH)
    cache = {
        event_id: WORLDPOP_DIR / str(info["cache_name"])
        for event_id, info in cfg["worldpop"]["countries"].items()
    }
    for p in cache.values():
        if not p.exists():
            raise FileNotFoundError(f"Missing WorldPop cache: {p}")
    feature_df, quality_df, recovery_df, profile_df, quality_rows, _ = _repair_stage_panel(stage_id, cache)
    existing = pd.read_csv(INTL_COV_QUALITY_PATH) if INTL_COV_QUALITY_PATH.exists() else pd.DataFrame()
    merged = pd.concat([existing, pd.DataFrame(quality_rows)], ignore_index=True)
    merged = merged.drop_duplicates(subset=["stage_id", "event_id"], keep="last")
    merged.to_csv(INTL_COV_QUALITY_PATH, index=False)
    return feature_df, quality_df, recovery_df, profile_df


def _survival_best(agg: pd.DataFrame) -> float:
    sub = agg[agg["model"].isin(["Cox", "AFT"])].copy()
    if sub.empty:
        return np.nan
    return float(pd.to_numeric(sub["c_index"], errors="coerce").max())


def _metric_from_agg(agg: pd.DataFrame, model: str, col: str) -> float:
    sub = agg[agg["model"] == model]
    if sub.empty or col not in sub.columns:
        return np.nan
    return float(pd.to_numeric(sub[col], errors="coerce").mean())


def _build_comparison() -> pd.DataFrame:
    rows = []
    for stage_id in STAGE_IDS:
        old = pd.read_csv(HZ1_AGG_STAGE_PATHS[stage_id])
        new = pd.read_csv(HZ2_AGG_STAGE_PATHS[stage_id])
        metric_specs = [
            ("Logit", "auc"),
            ("Logit", "brier"),
            ("Cox", "c_index"),
            ("AFT", "c_index"),
            ("OLS", "rmse"),
            ("MixedLM", "rmse"),
        ]
        for model, metric in metric_specs:
            old_val = _metric_from_agg(old, model, metric)
            new_val = _metric_from_agg(new, model, metric)
            delta = new_val - old_val if np.isfinite(old_val) and np.isfinite(new_val) else np.nan
            if metric == "auc":
                status = "improved" if delta >= 0.01 else ("flat" if np.isfinite(delta) and delta >= -0.01 else "worse")
            elif model in {"Cox", "AFT"} and metric == "c_index":
                status = "improved" if delta >= 0.01 else ("flat" if np.isfinite(delta) and delta >= -0.01 else "worse")
            elif metric in {"rmse", "brier"}:
                status = "improved" if delta <= -0.01 else ("flat" if np.isfinite(delta) and delta < 0.01 else "worse")
            else:
                status = "flat"
            rows.append({"stage_id": stage_id, "model": model, "metric_name": metric, "hz1_value": old_val, "hz2_value": new_val, "delta": delta, "status": status})
        old_surv = _survival_best(old)
        new_surv = _survival_best(new)
        d_surv = new_surv - old_surv if np.isfinite(old_surv) and np.isfinite(new_surv) else np.nan
        status = "improved" if d_surv >= 0.01 else ("flat" if np.isfinite(d_surv) and d_surv >= -0.01 else "worse")
        rows.append({"stage_id": stage_id, "model": "best", "metric_name": "survival_best", "hz1_value": old_surv, "hz2_value": new_surv, "delta": d_surv, "status": status})
    out = pd.DataFrame(rows)
    out.to_csv(INTL_COMPARE_PATH, index=False)
    return out


def _poi_counts_all(events_cfg: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for event_id, cfg in events_cfg.items():
        poi_path = ROOT / str(cfg["poi_csv"])
        count = 0
        if poi_path.exists():
            try:
                count = int(len(pd.read_csv(poi_path)))
            except Exception:
                count = 0
        rows.append({"event_id": event_id, "poi_count": count})
    return pd.DataFrame(rows)


def _post_tif_counts(events_cfg: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for event_id, cfg in events_cfg.items():
        pre_dir = ROOT / str(cfg["pre_dir"])
        post_dir = ROOT / str(cfg["post_dir"])
        rows.append({
            "event_id": event_id,
            "pre_tif_n": len(EXP.list_daily_tifs(pre_dir)) if pre_dir.exists() else 0,
            "post_tif_n": len(EXP.list_daily_tifs(post_dir)) if post_dir.exists() else 0,
        })
    return pd.DataFrame(rows)


def _score_obs(row: pd.Series) -> int:
    obs = float(row.get("observed_rate_v2", 0.0) or 0.0)
    censor = float(row.get("high_censoring_share", 1.0) or 1.0)
    if obs >= 0.99 and censor <= 0.01:
        return 30
    if obs >= 0.97:
        return 24
    if obs >= 0.95:
        return 18
    return 0


def _score_post_coverage(post_tif_n: int) -> int:
    if post_tif_n >= 40:
        return 20
    if post_tif_n >= 25:
        return 12
    if post_tif_n >= 15:
        return 6
    return 0


def _score_poi(count: int) -> int:
    if count >= 150:
        return 20
    if count >= 80:
        return 12
    if count >= 40:
        return 6
    return 0


def _score_covariate(missing_mean: float, is_proxy_only: bool) -> int:
    if np.isfinite(missing_mean) and missing_mean <= 0.05:
        return 20
    if is_proxy_only:
        return 10
    return 0


def _score_integrity(integrity_ok: bool) -> int:
    return 10 if integrity_ok else 0


def _increment_labels_from_metrics(comparison: pd.DataFrame) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    base = pd.read_csv(EVENT_INCREMENT_METRICS_PATH)
    auc_rows = base[(base["bundle"] == "hazard_mainline") & (base["model"] == "Logit") & (base["metric_name"] == "auc")].copy()
    for event_id, stage_id in [("ian_fortmyers", "stage_7_ian_fortmyers"), ("ian_charlotteharbor", "stage_8_ian_charlotteharbor")]:
        sub = auc_rows[auc_rows["stage_id"] == stage_id]
        delta = float(pd.to_numeric(sub["delta_vs_prev"], errors="coerce").iloc[0]) if not sub.empty else np.nan
        if np.isfinite(delta) and delta >= -0.01:
            labels[event_id] = "flat_or_helpful"
        else:
            labels[event_id] = "hurts_transport"

    stage9 = comparison[(comparison["stage_id"] == "stage_9_earthquake_hatay") & (comparison["model"] == "Logit") & (comparison["metric_name"] == "auc")]
    if not stage9.empty and float(stage9["hz2_value"].iloc[0]) - float(auc_rows[auc_rows["stage_id"] == "stage_8_ian_charlotteharbor"]["value"].iloc[0]) >= -0.01:
        labels["earthquake_hatay"] = "flat_or_helpful"
    else:
        labels["earthquake_hatay"] = "hurts_transport"

    stage10 = comparison[(comparison["stage_id"] == "stage_10_dorian_freeport") & (comparison["model"] == "Logit") & (comparison["metric_name"] == "auc")]
    stage9_hz2 = comparison[(comparison["stage_id"] == "stage_9_earthquake_hatay") & (comparison["model"] == "Logit") & (comparison["metric_name"] == "auc")]
    if not stage10.empty and not stage9_hz2.empty and float(stage10["hz2_value"].iloc[0]) - float(stage9_hz2["hz2_value"].iloc[0]) >= -0.01:
        labels["dorian_freeport"] = "flat_or_helpful"
    else:
        labels["dorian_freeport"] = "hurts_transport"
    return labels


def _score_readiness() -> Tuple[pd.DataFrame, pd.DataFrame]:
    events_cfg = _read_json(EVENTS10_PATH)
    audit = pd.read_csv(TARGET_AUDIT_STAGE_PATHS["stage_10_dorian_freeport"])
    post_cov = _post_tif_counts(events_cfg)
    poi = _poi_counts_all(events_cfg)
    repaired = pd.read_parquet(REPAIRED_QUALITY_STAGE_PATHS["stage_10_dorian_freeport"])
    comparison = pd.read_csv(INTL_COMPARE_PATH)
    inc_labels = _increment_labels_from_metrics(comparison)

    cov_stats = (
        repaired.groupby("event_id", as_index=False)
        .agg(
            missing_pop_flag_v2_mean=("missing_pop_flag_v2", "mean"),
            nonmissing_pop_v2=("pop_density_per_km2_v2", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            pop_nunique_v2=("pop_density_per_km2_v2", lambda s: int(pd.to_numeric(s, errors="coerce").dropna().nunique())),
            pop_source_v2=("pop_source_v2", lambda s: next((str(v) for v in s if pd.notna(v) and str(v) != ""), "missing")),
        )
    )
    cov_stats["integrity_ok"] = ~((cov_stats["nonmissing_pop_v2"] > 0) & (cov_stats["missing_pop_flag_v2_mean"] >= 0.99))

    ready = audit.merge(post_cov, on="event_id", how="left").merge(poi, on="event_id", how="left").merge(cov_stats, on="event_id", how="left")
    ready["event_count"] = ready["n_obs"]
    ready["obs_quality_score"] = ready.apply(_score_obs, axis=1)
    ready["post_coverage_score"] = ready["post_tif_n"].fillna(0).astype(int).apply(_score_post_coverage)
    ready["poi_score"] = ready["poi_count"].fillna(0).astype(int).apply(_score_poi)
    ready["is_proxy_only"] = ready["pop_source_v2"].fillna("missing").astype(str).str.contains("proxy|missing", case=False)
    ready["covariate_score"] = [
        _score_covariate(float(m) if pd.notna(m) else 1.0, bool(p))
        for m, p in zip(ready["missing_pop_flag_v2_mean"], ready["is_proxy_only"])
    ]
    ready["integrity_score"] = [
        _score_integrity(bool(v)) for v in ready["integrity_ok"].fillna(False)
    ]
    ready["total_score"] = ready[["obs_quality_score", "post_coverage_score", "poi_score", "covariate_score", "integrity_score"]].sum(axis=1)
    ready["readiness_band"] = np.where(ready["total_score"] >= 80, "mainline_ready", np.where(ready["total_score"] >= 60, "sensitivity_only", "repair_first"))
    ready["notes"] = ""

    comp_out = ready[["event_id", "event_count", "obs_quality_score", "post_coverage_score", "poi_score", "covariate_score", "integrity_score", "total_score", "readiness_band", "notes"]].copy()
    comp_out = comp_out.sort_values(["total_score", "event_id"], ascending=[False, True])
    comp_out.to_csv(READINESS_COMPONENTS_PATH, index=False)

    decision_rows = []
    for row in ready.itertuples(index=False):
        event_id = str(row.event_id)
        impact = inc_labels.get(event_id, "baseline_or_existing")
        if row.readiness_band == "mainline_ready" and impact != "hurts_transport":
            role = "mainline_candidate"
            why = "quality and covariates are ready, and incremental transport impact is not clearly harmful"
        elif row.readiness_band == "mainline_ready" and impact == "hurts_transport":
            role = "sensitivity_only"
            why = "data quality is acceptable, but current transport delta is still harmful"
        else:
            role = "repair_first"
            why = "covariate or observation readiness is not high enough for mainline training"
        decision_rows.append({
            "event_id": event_id,
            "total_score": float(row.total_score),
            "readiness_band": row.readiness_band,
            "increment_impact_label": impact,
            "recommended_role": role,
            "why": why,
        })
    decisions = pd.DataFrame(decision_rows).sort_values(["total_score", "event_id"], ascending=[False, True])
    decisions.to_csv(TRAINING_DECISION_PATH, index=False)
    score = decisions[["event_id", "total_score", "readiness_band"]].copy()
    score.to_csv(READINESS_SCORE_PATH, index=False)
    return comp_out, decisions


def _plot_summary(comparison: pd.DataFrame, readiness: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    auc = comparison[(comparison["model"] == "Logit") & (comparison["metric_name"] == "auc")].copy()
    surv = comparison[comparison["metric_name"] == "survival_best"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    x = np.arange(len(auc))
    axes[0].bar(x - 0.18, auc["hz1_value"], width=0.36, label="HZ1", color="#8d99ae")
    axes[0].bar(x + 0.18, auc["hz2_value"], width=0.36, label="HZ2", color="#1d3557")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([_pretty_stage_label(x) for x in auc["stage_id"]], rotation=0, ha="center")
    axes[0].set_title("Stage 9/10 Logit AUC")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()

    x2 = np.arange(len(surv))
    axes[1].bar(x2 - 0.18, surv["hz1_value"], width=0.36, label="HZ1", color="#8d99ae")
    axes[1].bar(x2 + 0.18, surv["hz2_value"], width=0.36, label="HZ2", color="#457b9d")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([_pretty_stage_label(x) for x in surv["stage_id"]], rotation=0, ha="center")
    axes[1].set_title("Stage 9/10 Survival Best")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    fig.subplots_adjust(bottom=0.22, left=0.06, right=0.98, top=0.90, wspace=0.10)
    fig.savefig(FIG_DIR / "stage9_10_hz1_vs_hz2.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    rd = readiness.sort_values("total_score", ascending=False)
    labels = [_pretty_event_label(x) for x in rd["event_id"]]
    ax.bar(labels, rd["total_score"], color="#2a9d8f")
    ax.axhline(80, color="#264653", linestyle="--", linewidth=1)
    ax.axhline(60, color="#e9c46a", linestyle="--", linewidth=1)
    ax.set_title("Event Readiness Score")
    ax.set_ylabel("score")
    ax.tick_params(axis="x", labelrotation=0)
    fig.subplots_adjust(bottom=0.25, left=0.08, right=0.98, top=0.90)
    fig.savefig(FIG_DIR / "event_readiness_score_v1.png", dpi=220)
    plt.close(fig)


def _write_report(comparison: pd.DataFrame, readiness: pd.DataFrame, decisions: pd.DataFrame) -> None:
    qual = pd.read_csv(INTL_COV_QUALITY_PATH) if INTL_COV_QUALITY_PATH.exists() else pd.DataFrame()
    strict9 = pd.read_csv(STRICT_SUMMARY_STAGE_PATHS["stage_9_earthquake_hatay"])
    strict10 = pd.read_csv(STRICT_SUMMARY_STAGE_PATHS["stage_10_dorian_freeport"])
    matched10 = pd.read_csv(OUTPUT_DIR / "facility_centered_model_summary_stage_10_dorian_freeport.csv")

    def comp_line(stage: str, metric: str, model: str = "Logit") -> str:
        sub = comparison[(comparison["stage_id"] == stage) & (comparison["metric_name"] == metric) & (comparison["model"] == model)]
        if sub.empty:
            return f"- {stage} {model} {metric}: NA"
        r = sub.iloc[0]
        return f"- {stage} {model} {metric}: HZ1={float(r['hz1_value']):.4f}, HZ2={float(r['hz2_value']):.4f}, delta={float(r['delta']):.4f}, status={r['status']}"

    surv_lines = []
    for stage in STAGE_IDS:
        sub = comparison[(comparison["stage_id"] == stage) & (comparison["metric_name"] == "survival_best")]
        if sub.empty:
            surv_lines.append(f"- {stage} survival_best: NA")
        else:
            r = sub.iloc[0]
            surv_lines.append(f"- {stage} survival_best: HZ1={float(r['hz1_value']):.4f}, HZ2={float(r['hz2_value']):.4f}, delta={float(r['delta']):.4f}, status={r['status']}")

    top_ready = readiness.sort_values("total_score", ascending=False).head(5)
    repair_first = decisions[decisions["recommended_role"] == "repair_first"]

    lines = [
        "# International Stage Repair Report / Stage 9/10 国际协变量修补报告",
        "",
        "## Objective",
        "- 用 WorldPop 栅格采样替换 Stage 9/10 国际事件的旧事件级常数人口值，并用更瘦的 HZ2 transport 规格重跑。",
        "",
        "## Legacy Issue Confirmed",
    ]
    if qual.empty:
        lines.append("- No covariate quality audit found.")
    else:
        for _, r in qual.iterrows():
            lines.append(
                f"- {r['stage_id']} | {r['event_id']}: legacy_issue_detected={int(r['legacy_issue_detected'])}, old_missing_mean={float(r['legacy_missing_pop_flag_mean']):.3f}, v2_missing_mean={float(r['v2_missing_pop_flag_mean']):.3f}, v2_unique={int(r['v2_unique_nonmissing'])}, status={r['quality_status']}"
            )
    lines.extend([
        "",
        "## HZ1 vs HZ2 Comparison",
        comp_line("stage_9_earthquake_hatay", "auc", "Logit"),
        comp_line("stage_10_dorian_freeport", "auc", "Logit"),
        *surv_lines,
        comp_line("stage_9_earthquake_hatay", "brier", "Logit"),
        comp_line("stage_10_dorian_freeport", "brier", "Logit"),
        "",
        "## Strict-V2 Reference (not rerun)",
        f"- Stage 9 MixedLM coef(in_buffer): {float(strict9[(strict9['model']=='MixedLM') & (strict9['variant']=='full_locked_v2_strict')]['value'].iloc[0]):.4f}",
        f"- Stage 10 MixedLM coef(in_buffer): {float(strict10[(strict10['model']=='MixedLM') & (strict10['variant']=='full_locked_v2_strict')]['value'].iloc[0]):.4f}",
        "",
        "## Matched Reference (Stage 10, not rerun)",
        f"- Matched Logit OR(in_buffer): {float(matched10[matched10['model']=='FacilityMatchedLogit']['value'].iloc[0]):.4f}",
        "",
        "## Readiness Ranking (top)",
    ])
    for _, r in top_ready.iterrows():
        lines.append(f"- {r['event_id']}: score={float(r['total_score']):.1f}, band={r['readiness_band']}")
    lines.extend(["", "## Repair-First Events"])
    if repair_first.empty:
        lines.append("- None")
    else:
        for _, r in repair_first.iterrows():
            lines.append(f"- {r['event_id']}: {r['why']}")
    both_bad = comparison[comparison["status"] == "improved"].empty
    verdict = "Negative result" if both_bad else "Partially improved"
    lines.extend([
        "",
        "## Verdict",
        f"- {verdict}",
        "- If HZ2 does not recover Stage 9/10, the evidence favors a structural transport issue rather than just a bad international population proxy.",
        "",
        "## Figures",
        "- `project/modeling_report/figures/intl_stage_repair_v1/stage9_10_hz1_vs_hz2.png`",
        "- `project/modeling_report/figures/intl_stage_repair_v1/event_readiness_score_v1.png`",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    text = REPORT_INDEX_PATH.read_text(encoding="utf-8") if REPORT_INDEX_PATH.exists() else "# Modeling Report Index\n\n"
    line = "- `project/modeling_report/14_intl_stage_repair_report.md`"
    if line not in text:
        REPORT_INDEX_PATH.write_text(text.rstrip() + "\n" + line + "\n", encoding="utf-8")


def cmd_prepare_covariates() -> int:
    _ensure_dirs()
    _append_progress("intl-stage-repair: prepare covariates")
    _prepare_covariates()
    return 0


def cmd_repair_stage9() -> int:
    _ensure_dirs()
    _append_progress("intl-stage-repair: repair stage 9")
    if not INTL_COV_MANIFEST_PATH.exists():
        _prepare_covariates()
    _repair_stage("stage_9_earthquake_hatay")
    return 0


def cmd_repair_stage10() -> int:
    _ensure_dirs()
    _append_progress("intl-stage-repair: repair stage 10")
    if not INTL_COV_MANIFEST_PATH.exists():
        _prepare_covariates()
    if not REPAIRED_QUALITY_STAGE_PATHS["stage_9_earthquake_hatay"].exists():
        _repair_stage("stage_9_earthquake_hatay")
    _repair_stage("stage_10_dorian_freeport")
    return 0


def cmd_fit_hz2() -> int:
    _ensure_dirs()
    _append_progress("intl-stage-repair: fit HZ2")
    for stage in STAGE_IDS:
        if not REPAIRED_QUALITY_STAGE_PATHS[stage].exists() or not REPAIRED_RECOVERY_STAGE_PATHS[stage].exists():
            _repair_stage(stage)
        quality_df = pd.read_parquet(REPAIRED_QUALITY_STAGE_PATHS[stage])
        recovery_df = pd.read_parquet(REPAIRED_RECOVERY_STAGE_PATHS[stage])
        _run_hz2(stage, quality_df, recovery_df)
    _build_comparison()
    return 0


def cmd_score_readiness() -> int:
    _ensure_dirs()
    _append_progress("intl-stage-repair: score readiness")
    if not INTL_COMPARE_PATH.exists():
        _build_comparison()
    _score_readiness()
    return 0


def cmd_report() -> int:
    _ensure_dirs()
    _append_progress("intl-stage-repair: write report")
    comparison = pd.read_csv(INTL_COMPARE_PATH)
    readiness = pd.read_csv(READINESS_COMPONENTS_PATH)
    decisions = pd.read_csv(TRAINING_DECISION_PATH)
    _plot_summary(comparison, readiness)
    _write_report(comparison, readiness, decisions)
    return 0


def cmd_full_run() -> int:
    _ensure_dirs()
    _append_progress("intl-stage-repair: full run start")
    manifest = _prepare_covariates()
    if manifest.empty:
        raise RuntimeError("worldpop_manifest_empty")
    _repair_stage("stage_9_earthquake_hatay")
    _repair_stage("stage_10_dorian_freeport")
    for stage in STAGE_IDS:
        quality_df = pd.read_parquet(REPAIRED_QUALITY_STAGE_PATHS[stage])
        recovery_df = pd.read_parquet(REPAIRED_RECOVERY_STAGE_PATHS[stage])
        _run_hz2(stage, quality_df, recovery_df)
    comparison = _build_comparison()
    readiness, decisions = _score_readiness()
    _plot_summary(comparison, readiness)
    _write_report(comparison, readiness, decisions)
    _append_progress("intl-stage-repair: full run completed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent Stage 9/10 international covariate repair line")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-covariates")
    sub.add_parser("repair-stage9")
    sub.add_parser("repair-stage10")
    sub.add_parser("fit-hz2")
    sub.add_parser("score-readiness")
    sub.add_parser("report")
    sub.add_parser("full-run")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-covariates":
        return cmd_prepare_covariates()
    if args.command == "repair-stage9":
        return cmd_repair_stage9()
    if args.command == "repair-stage10":
        return cmd_repair_stage10()
    if args.command == "fit-hz2":
        return cmd_fit_hz2()
    if args.command == "score-readiness":
        return cmd_score_readiness()
    if args.command == "report":
        return cmd_report()
    if args.command == "full-run":
        return cmd_full_run()
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
