#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "project" / "modeling" / "output"
FIG_DIR = ROOT / "project" / "modeling_report" / "figures" / "feature_upgrade"


def _ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_model_compare() -> None:
    src = OUTPUT / "model_summary_for_report.csv"
    df = pd.read_csv(src)

    key_rows = []
    for model, metric, key in [
        ("OLS", "coef_in_buffer", "effect"),
        ("MixedLM", "coef_in_buffer", "effect"),
        ("Logit", "odds_ratio_in_buffer", "or"),
        ("Cox", "hazard_ratio_in_buffer", "hr"),
    ]:
        sub = df[(df["model"] == model) & (df["key_metric"] == metric)].copy()
        if sub.empty:
            continue
        for variant in ["no_nlcd", "with_nlcd"]:
            one = sub[sub["variant"] == variant]
            if one.empty:
                continue
            key_rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "value": float(one.iloc[0]["value"]),
                    "metric_type": key,
                }
            )
    plot_df = pd.DataFrame(key_rows)
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1, 1]})
    sns.barplot(
        data=plot_df[plot_df["metric_type"] == "effect"],
        x="model",
        y="value",
        hue="variant",
        ax=axes[0],
        palette="Set2",
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("OLS/MixedLM in_buffer Coef")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Coefficient")

    sns.barplot(
        data=plot_df[plot_df["metric_type"] == "or"],
        x="model",
        y="value",
        hue="variant",
        ax=axes[1],
        palette="Set2",
    )
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Logit Odds Ratio")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("OR")

    sns.barplot(
        data=plot_df[plot_df["metric_type"] == "hr"],
        x="model",
        y="value",
        hue="variant",
        ax=axes[2],
        palette="Set2",
    )
    axes[2].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Cox Hazard Ratio")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("HR")

    for ax in axes:
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Model Metrics Before/After Land-use Controls", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "model_compare_before_after.png", dpi=220)
    plt.close(fig)


def plot_cloud_quality() -> None:
    src = OUTPUT / "cloud_feature_summary.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    if df.empty:
        return

    long = pd.melt(
        df,
        id_vars=["event_id"],
        value_vars=["pre_valid_ratio", "post_valid_ratio"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=long, x="event_id", y="value", hue="metric", palette="deep", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title("Cloud-screened Usable Ratio by Event")
    ax.set_xlabel("")
    ax.set_ylabel("Usable ratio")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "cloud_usable_ratio_by_event.png", dpi=220)
    plt.close(fig)


def plot_sync_summary() -> None:
    gap = OUTPUT / "teammate_reuse_gap.csv"
    log = OUTPUT / "teammate_reuse_sync_log.csv"
    if not gap.exists() or not log.exists():
        return

    gap_df = pd.read_csv(gap)
    log_df = pd.read_csv(log)

    status_counts = log_df["status"].value_counts().to_dict()
    copied = int(status_counts.get("copied", 0))
    kept = int((gap_df["action"] == "keep_local").sum())
    missing_before = int((gap_df["action"] == "reuse_from_teammate").sum())

    plot_df = pd.DataFrame(
        {
            "metric": ["Copied from teammate", "Already local", "Missing before sync"],
            "count": [copied, kept, missing_before],
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=plot_df, x="metric", y="count", palette="muted", ax=ax)
    ax.set_title("Reuse-first Sync Summary")
    ax.set_xlabel("")
    ax.set_ylabel("File count")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "teammate_sync_summary.png", dpi=220)
    plt.close(fig)


def main() -> None:
    _ensure_dir()
    sns.set_theme(style="whitegrid")
    plot_model_compare()
    plot_cloud_quality()
    plot_sync_summary()
    print(f"Feature-upgrade figures generated under: {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
