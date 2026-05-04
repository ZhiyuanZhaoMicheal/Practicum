"""
regen_modelD_prob_maps.py
=========================
Train Model D (pure-NTL, no spatial proximity features) on the 25-event panel
and regenerate probability maps for each event in the same 3-band GeoTIFF
format Stage 3 expects ({event}_prob_map_modelD.tif).

Model D specification mirrors stage2_15events_modelD.ipynb but trains on the
full 25-event panel for downstream Stage 3 use.

Output:
    data/result/stage2/{event_id}_prob_map_modelD.tif   (RF, XGB, Ensemble)
    data/result/stage2/rf_modelD.pkl
    data/result/stage2/xgb_modelD.pkl
    data/result/stage2/feature_importance_modelD.csv
"""

import os, glob
import numpy as np
import pandas as pd
import rasterio
import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'data', 'result', 'stage2')
PANEL_PATH   = os.path.join(OUTPUT_DIR, 'pixel_panel.parquet')

# ─── Hyperparameters (identical to Stage 2) ──────────────────────────
RF_PARAMS = {
    "n_estimators":     500,
    "max_depth":        5,
    "min_samples_leaf": 20,
    "max_features":     "sqrt",
    "class_weight":     "balanced",
    "n_jobs":           -1,
    "random_state":     42,
}

XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "scale_pos_weight": 5,
    "random_state":     42,
    "verbosity":        0,
}

# ─── Event configuration: pre folder for each event ──────────────────
# Maps event_id (in pixel_panel) → folder name under data/processed/
PRE_FOLDER = {
    "Maria_SanJuan":         "Maria-VNP46A2-pre",
    "Irma_Miami":            "Irma_Miami-VNP46A2-pre",
    "Ida_NewOrleans":        "Ida_NewOrleans-VNP46A2-pre",
    "Laura_LakeCharles":     "Laura_LakeCharles-VNP46A2-pre",
    "Michael_PanamaCity":    "Michael-VNP46A2-pre",
    "Earthquake_SanJuan":    "Earthquake-VNP46A2-pre",
    "Ian_CharlotteHarbor":   "Ian_CharlotteHarbor-VNP46A2-pre",
    "Ian_FortMyers":         "Ian_FortMyers-VNP46A2-pre",
    "Earthquake_Hatay":      "Earthquake_Hatay-VNP46A2-pre",
    "Florence_Wilmington":   "Florence_Wilmington-VNP46A2-pre",
    "Irma_Savannah":         "Irma_Savannah-VNP46A2-pre",
    "Isaias_Newark":         "Isaias_Newark-VNP46A2-pre",
    "Matthew_Jacksonville":  "Matthew_Jacksonville-VNP46A2-pre",
    "Zeta_Atlanta":          "Zeta_Atlanta-VNP46A2-pre",
    "Zeta_Birmingham":       "Zeta_Birmingham-VNP46A2-pre",
    "Matthew_Fayetteville":  "Matthew_Fayetteville-VNP46A2-pre",
    "Florence_MyrtleBeach":  "Florence_MyrtleBeach-VNP46A2-pre",
    "Isaias_Westchester":    "Isaias_Westchester-VNP46A2-pre",
    "Uri_Houston":           "Uri_Houston-VNP46A2-pre",
    "Derecho_Chicago":       "Derecho_Chicago-VNP46A2-pre",
    "Severe_Detroit":        "Severe_Detroit-VNP46A2-pre",
    "Noreaster_Boston":      "Noreaster_Boston-VNP46A2-pre",
    "IceStorm_OKC":          "IceStorm_OKC-VNP46A2-pre",
    "Severe_Nashville":      "Severe_Nashville-VNP46A2-pre",
    "Atmos_Seattle":         "Atmos_Seattle-VNP46A2-pre",
}


def engineer_features_modelD(df):
    """Mirror stage2_15events_modelD.ipynb. 10 features, no spatial proximity."""
    df = df.copy()
    df["drop_magnitude"]    = -df["delta_ntl"].clip(upper=0)
    df["log_pre_ntl"]       = np.log1p(df["pre_mean_ntl"])
    df["log_post_ntl"]      = np.log1p(df["post_mean_ntl"])
    df["log_city_pre_mean"] = np.log1p(df["city_pre_mean"])
    df["ntl_relative"]      = df["pre_mean_ntl"] / (df["city_pre_mean"] + 1e-6)

    city_median = df.groupby("event_id")["pre_mean_ntl"].transform("median")
    df["below_city_median"] = (df["pre_mean_ntl"] < city_median).astype(np.uint8)

    df["city_size_code"] = df["city_size"].map(
        {"large": 0, "medium": 1, "small": 2}).fillna(1)
    df["is_hurricane"]   = (df["disaster_type"] == "hurricane").astype(np.uint8)
    df["is_earthquake"]  = (df["disaster_type"] == "earthquake").astype(np.uint8)

    features = [
        "drop_magnitude", "delta_ntl",
        "log_pre_ntl", "log_post_ntl",
        "log_city_pre_mean", "ntl_relative",
        "below_city_median",
        "city_size_code", "is_hurricane", "is_earthquake",
    ]
    return df, features


def main():
    print("=" * 70)
    print("Model D · regenerate per-event probability maps for Stage 3")
    print("=" * 70)

    # ── Load panel ────────────────────────────────────────────────
    df = pd.read_parquet(PANEL_PATH)
    print(f"\nPanel: {len(df):,} pixels × {df['event_id'].nunique()} events")

    df, features = engineer_features_modelD(df)
    print(f"Model D features ({len(features)}): {features}")

    # ── Train final models on all 25 events ──────────────────────
    X = df[features].fillna(0).values
    y = df["in_buffer_strict"].values
    print(f"\nTraining: n={len(y):,}  positives={y.sum():,} ({100*y.mean():.1f}%)")

    rf  = RandomForestClassifier(**RF_PARAMS).fit(X, y)
    print("  RF trained ✓")
    xgbm = xgb.XGBClassifier(**XGB_PARAMS).fit(X, y)
    print("  XGBoost trained ✓")

    importance_df = pd.DataFrame({
        "feature": features,
        "rf_imp":  rf.feature_importances_,
        "xgb_imp": xgbm.feature_importances_,
    })
    importance_df["avg_imp"] = (importance_df["rf_imp"] + importance_df["xgb_imp"]) / 2
    importance_df = importance_df.sort_values("avg_imp", ascending=False).reset_index(drop=True)

    joblib.dump(rf,   os.path.join(OUTPUT_DIR, "rf_modelD.pkl"))
    joblib.dump(xgbm, os.path.join(OUTPUT_DIR, "xgb_modelD.pkl"))
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_modelD.csv"), index=False)
    print(f"\nTop features (Model D):\n{importance_df.head(10).to_string(index=False)}")

    # ── Generate per-event probability maps ──────────────────────
    print("\nGenerating Model D probability maps ...")
    for event_id in sorted(df["event_id"].unique()):
        event_df = df[df["event_id"] == event_id]
        if event_df.empty:
            continue

        pre_folder = PRE_FOLDER.get(event_id)
        if pre_folder is None:
            print(f"  [SKIP] {event_id}: no pre folder mapping"); continue

        pre_tifs = sorted(glob.glob(os.path.join(DATA_DIR, pre_folder, "*.tif")))
        if not pre_tifs:
            print(f"  [SKIP] {event_id}: no pre TIFs in {pre_folder}"); continue

        with rasterio.open(pre_tifs[0]) as src:
            profile = src.profile.copy()
            height, width = src.height, src.width

        X_ev     = event_df[features].fillna(0).values
        rf_prob  = rf.predict_proba(X_ev)[:, 1]
        xgb_prob = xgbm.predict_proba(X_ev)[:, 1]
        ens_prob = rf_prob * 0.7 + xgb_prob * 0.3

        rows = event_df["row"].values
        cols = event_df["col"].values

        def to_grid(probs):
            arr = np.full((height, width), np.nan, dtype=np.float32)
            arr[rows, cols] = probs
            return arr

        grid_rf  = to_grid(rf_prob)
        grid_xgb = to_grid(xgb_prob)
        grid_ens = to_grid(ens_prob)

        profile.update(count=3, dtype="float32", nodata=0.0, compress="lzw")
        out_tif = os.path.join(OUTPUT_DIR, f"{event_id}_prob_map_modelD.tif")
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(np.nan_to_num(grid_rf,  nan=0.0), 1)
            dst.write(np.nan_to_num(grid_xgb, nan=0.0), 2)
            dst.write(np.nan_to_num(grid_ens, nan=0.0), 3)
        print(f"  ✓ {event_id}_prob_map_modelD.tif  (n_pixels={len(event_df):,})")

    print("\n" + "=" * 70)
    print("Model D regeneration complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
