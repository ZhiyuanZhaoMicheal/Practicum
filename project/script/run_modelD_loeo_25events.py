"""
run_modelD_loeo_25events.py
===========================
Leave-One-Event-Out CV for the Production Model (Model D, pure-NTL features)
on the full 25-event panel.

Each fold:
  · Hold out one event
  · Train RF + XGBoost + Logit on the remaining 24 events
  · Compute AUC + PR-AUC on held-out event

Output:
  data/result/stage2/loeo_modelD_25events.csv
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STAGE2_DIR   = os.path.join(PROJECT_ROOT, 'data', 'result', 'stage2')

RF_PARAMS = {
    "n_estimators":     500, "max_depth": 5, "min_samples_leaf": 20,
    "max_features":     "sqrt", "class_weight": "balanced",
    "n_jobs":           -1, "random_state": 42,
}
XGB_PARAMS = {
    "n_estimators":     500, "max_depth": 4, "learning_rate": 0.05,
    "subsample":        0.8, "colsample_bytree": 0.8, "min_child_weight": 20,
    "scale_pos_weight": 5, "early_stopping_rounds": 50,
    "eval_metric":      "auc", "random_state": 42, "verbosity": 0,
}


def engineer_features_modelD(df):
    df = df.copy()
    df["drop_magnitude"]    = -df["delta_ntl"].clip(upper=0)
    df["log_pre_ntl"]       = np.log1p(df["pre_mean_ntl"])
    df["log_post_ntl"]      = np.log1p(df["post_mean_ntl"])
    df["log_city_pre_mean"] = np.log1p(df["city_pre_mean"])
    df["ntl_relative"]      = df["pre_mean_ntl"] / (df["city_pre_mean"] + 1e-6)
    cm = df.groupby("event_id")["pre_mean_ntl"].transform("median")
    df["below_city_median"] = (df["pre_mean_ntl"] < cm).astype(np.uint8)
    df["city_size_code"]    = df["city_size"].map({"large": 0, "medium": 1, "small": 2}).fillna(1)
    df["is_hurricane"]      = (df["disaster_type"] == "hurricane").astype(np.uint8)
    df["is_earthquake"]     = (df["disaster_type"] == "earthquake").astype(np.uint8)
    feats = [
        "drop_magnitude", "delta_ntl",
        "log_pre_ntl", "log_post_ntl",
        "log_city_pre_mean", "ntl_relative",
        "below_city_median",
        "city_size_code", "is_hurricane", "is_earthquake",
    ]
    return df, feats


def safe_proba(model, X_te, y_te):
    if len(model.classes_) < 2: return None, False
    if len(np.unique(y_te)) < 2: return None, False
    return model.predict_proba(X_te)[:, 1], True


def main():
    print("=" * 65)
    print("Model D · LOEO cross-validation on the 25-event panel")
    print("=" * 65)

    df = pd.read_parquet(os.path.join(STAGE2_DIR, 'pixel_panel.parquet'))
    df, feats = engineer_features_modelD(df)
    print(f"Panel: {len(df):,} pixels  ·  {df['event_id'].nunique()} events")
    print(f"Features ({len(feats)}): {feats}")
    label = "in_buffer_strict"

    events = sorted(df["event_id"].unique())
    rows = []
    for i, held in enumerate(events, 1):
        train_idx = df.index[df["event_id"] != held]
        test_idx  = df.index[df["event_id"] == held]
        X_tr = df.loc[train_idx, feats].fillna(0).values
        y_tr = df.loc[train_idx, label].values
        X_te = df.loc[test_idx,  feats].fillna(0).values
        y_te = df.loc[test_idx,  label].values

        # ── RF ────────────────────────────────────────────────
        rf = RandomForestClassifier(**RF_PARAMS).fit(X_tr, y_tr)
        rfp, rfok = safe_proba(rf, X_te, y_te)
        rf_auc = roc_auc_score(y_te, rfp) if rfok else np.nan
        rf_ap  = average_precision_score(y_te, rfp) if rfok else np.nan

        # ── XGB ───────────────────────────────────────────────
        try:
            X_tr2, X_val, y_tr2, y_val = train_test_split(
                X_tr, y_tr, test_size=0.15, random_state=42,
                stratify=y_tr if len(np.unique(y_tr)) > 1 else None)
            xgbm = xgb.XGBClassifier(**XGB_PARAMS)
            xgbm.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)], verbose=False)
            xp, xok = safe_proba(xgbm, X_te, y_te)
            xgb_auc = roc_auc_score(y_te, xp) if xok else np.nan
            xgb_ap  = average_precision_score(y_te, xp) if xok else np.nan
        except Exception as e:
            xgb_auc, xgb_ap = np.nan, np.nan

        # ── Logit ─────────────────────────────────────────────
        sc = StandardScaler()
        Xs_tr = sc.fit_transform(X_tr); Xs_te = sc.transform(X_te)
        lg = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced').fit(Xs_tr, y_tr)
        lp, lok = safe_proba(lg, Xs_te, y_te)
        lg_auc = roc_auc_score(y_te, lp) if lok else np.nan
        lg_ap  = average_precision_score(y_te, lp) if lok else np.nan

        rows.append({
            "held_out": held,
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "pos_rate_test": float(y_te.mean()),
            "rf_auc": rf_auc, "rf_ap": rf_ap,
            "xgb_auc": xgb_auc, "xgb_ap": xgb_ap,
            "logit_auc": lg_auc, "logit_ap": lg_ap,
        })
        print(f"  [{i:2d}/25] {held:25s}  RF={rf_auc:.3f}  XGB={xgb_auc:.3f}  Logit={lg_auc:.3f}")

    out = pd.DataFrame(rows)
    out_path = os.path.join(STAGE2_DIR, 'loeo_modelD_25events.csv')
    out.to_csv(out_path, index=False)
    print()
    print("─" * 60)
    print(f"Mean AUC  RF={out['rf_auc'].mean():.4f}  "
          f"XGB={out['xgb_auc'].mean():.4f}  "
          f"Logit={out['logit_auc'].mean():.4f}")
    print(f"Mean PR-AUC RF={out['rf_ap'].mean():.4f}  "
          f"XGB={out['xgb_ap'].mean():.4f}  "
          f"Logit={out['logit_ap'].mean():.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
