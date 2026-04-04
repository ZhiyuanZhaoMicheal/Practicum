"""
stage3_ntl_download.py
======================
GEE download script for Stage 3 — 16 new disaster events.
Mirrors the pipeline from multi_event_ntl_download_v2.ipynb.

Usage:
    # In terminal with ee authenticated:
    python stage3_ntl_download.py                    # all events
    python stage3_ntl_download.py --events uri_houston zeta_atlanta
    python stage3_ntl_download.py --step screen      # only cloud screening
    python stage3_ntl_download.py --step export       # only export
    python stage3_ntl_download.py --step status       # check task status

Prerequisites:
    pip install earthengine-api pandas
    earthengine authenticate
"""

import ee
import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# 1. Configuration — 16 new events for Stage 3
# ─────────────────────────────────────────────────────────────────────

from stage3_events import EVENTS  # shared config, no ee dependency

_EVENTS_INLINE = {  # not used at runtime; kept for reference only
    # ═══════════════════════════════════════════════════════════════
    # HURRICANES (9 events) — study areas matched to IBTrACS tracks
    # ═══════════════════════════════════════════════════════════════

    "matthew_jax": {
        "name":       "Hurricane Matthew (Jacksonville, FL)",
        "bounds":     [-81.835, 30.18, -81.485, 30.48],
        "landfall":   "2016-10-07",
        "pre_start":  "2016-09-01",
        "pre_end":    "2016-10-06",
        "post_start": "2016-10-07",
        "post_end":   "2016-11-30",
        "metric_crs": "EPSG:32617",
        "drive_root": "Matthew_Jacksonville",
    },

    "florence_wilm": {
        "name":       "Hurricane Florence (Wilmington, NC)",
        "bounds":     [-78.125, 34.08, -77.775, 34.38],
        "landfall":   "2018-09-14",
        "pre_start":  "2018-08-01",
        "pre_end":    "2018-09-13",
        "post_start": "2018-09-14",
        "post_end":   "2018-11-15",
        "metric_crs": "EPSG:32618",
        "drive_root": "Florence_Wilmington",
    },

    "zeta_atlanta": {
        "name":       "Hurricane Zeta (Atlanta, GA)",
        "bounds":     [-84.565, 33.60, -84.215, 33.90],
        "landfall":   "2020-10-29",
        "pre_start":  "2020-09-15",
        "pre_end":    "2020-10-28",
        "post_start": "2020-10-29",
        "post_end":   "2020-12-31",
        "metric_crs": "EPSG:32616",
        "drive_root": "Zeta_Atlanta",
    },

    "isaias_nj": {
        "name":       "Tropical Storm Isaias (Newark, NJ)",
        "bounds":     [-74.345, 40.59, -73.995, 40.89],
        "landfall":   "2020-08-04",
        "pre_start":  "2020-07-01",
        "pre_end":    "2020-08-03",
        "post_start": "2020-08-04",
        "post_end":   "2020-09-30",
        "metric_crs": "EPSG:32618",
        "drive_root": "Isaias_Newark",
    },

    "irma_savannah": {
        "name":       "Hurricane Irma (Savannah, GA)",
        "bounds":     [-81.265, 31.93, -80.915, 32.23],
        "landfall":   "2017-09-11",
        "pre_start":  "2017-08-01",
        "pre_end":    "2017-09-10",
        "post_start": "2017-09-11",
        "post_end":   "2017-11-15",
        "metric_crs": "EPSG:32617",
        "drive_root": "Irma_Savannah",
    },

    "zeta_birmingham": {
        "name":       "Hurricane Zeta (Birmingham, AL)",
        "bounds":     [-86.975, 33.37, -86.625, 33.67],
        "landfall":   "2020-10-29",
        "pre_start":  "2020-09-15",
        "pre_end":    "2020-10-28",
        "post_start": "2020-10-29",
        "post_end":   "2020-12-31",
        "metric_crs": "EPSG:32616",
        "drive_root": "Zeta_Birmingham",
    },

    "matthew_nc": {
        "name":       "Hurricane Matthew (Fayetteville, NC)",
        "bounds":     [-79.055, 34.91, -78.705, 35.21],
        "landfall":   "2016-10-08",
        "pre_start":  "2016-09-01",
        "pre_end":    "2016-10-07",
        "post_start": "2016-10-08",
        "post_end":   "2016-11-30",
        "metric_crs": "EPSG:32617",
        "drive_root": "Matthew_Fayetteville",
    },

    "florence_sc": {
        "name":       "Hurricane Florence (Myrtle Beach, SC)",
        "bounds":     [-79.065, 33.54, -78.715, 33.84],
        "landfall":   "2018-09-14",
        "pre_start":  "2018-08-01",
        "pre_end":    "2018-09-13",
        "post_start": "2018-09-14",
        "post_end":   "2018-11-15",
        "metric_crs": "EPSG:32617",
        "drive_root": "Florence_MyrtleBeach",
    },

    "isaias_ny": {
        "name":       "Tropical Storm Isaias (Westchester, NY)",
        "bounds":     [-73.935, 40.88, -73.585, 41.18],
        "landfall":   "2020-08-04",
        "pre_start":  "2020-07-01",
        "pre_end":    "2020-08-03",
        "post_start": "2020-08-04",
        "post_end":   "2020-09-30",
        "metric_crs": "EPSG:32618",
        "drive_root": "Isaias_Westchester",
    },

    # ═══════════════════════════════════════════════════════════════
    # NON-HURRICANES (7 events) — study areas centered on metro
    # ═══════════════════════════════════════════════════════════════

    "uri_houston": {
        "name":       "Winter Storm Uri (Houston, TX)",
        "bounds":     [-95.545, 29.61, -95.195, 29.91],
        "landfall":   "2021-02-15",
        "pre_start":  "2021-01-01",
        "pre_end":    "2021-02-14",
        "post_start": "2021-02-15",
        "post_end":   "2021-04-15",
        "metric_crs": "EPSG:32615",
        "drive_root": "Uri_Houston",
    },

    "derecho_chicago": {
        "name":       "Derecho (Chicago, IL)",
        "bounds":     [-87.805, 41.73, -87.455, 42.03],
        "landfall":   "2020-08-10",
        "pre_start":  "2020-07-01",
        "pre_end":    "2020-08-09",
        "post_start": "2020-08-10",
        "post_end":   "2020-10-15",
        "metric_crs": "EPSG:32616",
        "drive_root": "Derecho_Chicago",
    },

    "severe_detroit": {
        "name":       "Severe Storms (Detroit, MI)",
        "bounds":     [-83.225, 42.18, -82.875, 42.48],
        "landfall":   "2019-07-20",
        "pre_start":  "2019-06-15",
        "pre_end":    "2019-07-19",
        "post_start": "2019-07-20",
        "post_end":   "2019-09-30",
        "metric_crs": "EPSG:32617",
        "drive_root": "Severe_Detroit",
    },

    "noreaster_boston": {
        "name":       "Nor'easter (Boston, MA)",
        "bounds":     [-71.235, 42.21, -70.885, 42.51],
        "landfall":   "2021-10-27",
        "pre_start":  "2021-09-15",
        "pre_end":    "2021-10-26",
        "post_start": "2021-10-27",
        "post_end":   "2021-12-31",
        "metric_crs": "EPSG:32619",
        "drive_root": "Noreaster_Boston",
    },

    "icestorm_okc": {
        "name":       "Ice Storm (Oklahoma City, OK)",
        "bounds":     [-97.695, 35.32, -97.345, 35.62],
        "landfall":   "2020-10-27",
        "pre_start":  "2020-09-15",
        "pre_end":    "2020-10-26",
        "post_start": "2020-10-27",
        "post_end":   "2020-12-31",
        "metric_crs": "EPSG:32614",
        "drive_root": "IceStorm_OKC",
    },

    "severe_nashville": {
        "name":       "Severe Storms (Nashville, TN)",
        "bounds":     [-86.955, 36.01, -86.605, 36.31],
        "landfall":   "2023-07-18",
        "pre_start":  "2023-06-01",
        "pre_end":    "2023-07-17",
        "post_start": "2023-07-18",
        "post_end":   "2023-09-30",
        "metric_crs": "EPSG:32616",
        "drive_root": "Severe_Nashville",
    },

    "atmos_seattle": {
        "name":       "Atmospheric River (Seattle, WA)",
        "bounds":     [-122.505, 47.46, -122.155, 47.76],
        "landfall":   "2022-11-04",
        "pre_start":  "2022-09-15",
        "pre_end":    "2022-11-03",
        "post_start": "2022-11-04",
        "post_end":   "2023-01-15",
        "metric_crs": "EPSG:32610",
        "drive_root": "Atmos_Seattle",
    },
}


# ─────────────────────────────────────────────────────────────────────
# 2. GEE setup
# ─────────────────────────────────────────────────────────────────────

VNP46A2_COLLECTION = "NASA/VIIRS/002/VNP46A2"
NTL_BAND   = "Gap_Filled_DNB_BRDF_Corrected_NTL"
CLOUD_BAND = "Mandatory_Quality_Flag"
SCALE_FACTOR = 0.1


def init_gee(project=None):
    """Initialize Earth Engine."""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        print(f"GEE initialized")
    except Exception as e:
        print(f"GEE init failed: {e}")
        print("Run: earthengine authenticate")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────
# 3. Cloud screening functions (identical to v2 notebook)
# ─────────────────────────────────────────────────────────────────────

def extract_bits(img, band, start_bit, end_bit):
    nbits = end_bit - start_bit + 1
    mask = (1 << nbits) - 1
    return img.select(band).rightShift(start_bit).bitwiseAnd(mask)


def compute_cloud_fraction(roi):
    def _inner(img):
        cloud_state = extract_bits(img, CLOUD_BAND, 6, 7)
        cloudy = cloud_state.gte(2)
        shadow = extract_bits(img, CLOUD_BAND, 8, 8).eq(1)
        cirrus = extract_bits(img, CLOUD_BAND, 9, 9).eq(1)
        contaminated = cloudy.Or(shadow).Or(cirrus)

        stats = contaminated.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=500,
            maxPixels=1e8,
        )
        return img.set("cloud_fraction", stats.get(CLOUD_BAND))
    return _inner


def apply_qa_and_scale(img):
    """Mask clouds/shadow/cirrus and scale NTL."""
    cloud_state = extract_bits(img, CLOUD_BAND, 6, 7)
    cloudy = cloud_state.gte(2)
    shadow = extract_bits(img, CLOUD_BAND, 8, 8).eq(1)
    cirrus = extract_bits(img, CLOUD_BAND, 9, 9).eq(1)
    bad = cloudy.Or(shadow).Or(cirrus)
    ntl = img.select(NTL_BAND).multiply(SCALE_FACTOR)
    return ntl.updateMask(bad.Not())


# ─────────────────────────────────────────────────────────────────────
# 4. Cloud screening — query all images, compute cloud fraction
# ─────────────────────────────────────────────────────────────────────

def screen_event(event_id):
    """Query GEE for all images in pre+post window, compute cloud fraction."""
    cfg = EVENTS[event_id]
    roi = ee.Geometry.Rectangle(cfg["bounds"])

    start = cfg["pre_start"]
    end = cfg["post_end"]

    print(f"  Screening {event_id}: {start} → {end} ...")
    col = (
        ee.ImageCollection(VNP46A2_COLLECTION)
        .filterBounds(roi)
        .filterDate(start, end)
        .map(compute_cloud_fraction(roi))
    )

    info = col.aggregate_array("system:time_start").getInfo()
    cf = col.aggregate_array("cloud_fraction").getInfo()
    dates = [datetime.utcfromtimestamp(t / 1000).strftime("%Y-%m-%d") for t in info]

    results = []
    for d, c in zip(dates, cf):
        results.append({
            "date": d,
            "cloud_fraction": c if c is not None else 1.0,
        })

    df = pd.DataFrame(results)
    df["usable"] = df["cloud_fraction"] < 0.3  # <30% cloud = usable

    n_total = len(df)
    n_usable = df["usable"].sum()
    print(f"    {n_total} images, {n_usable} usable ({n_usable/n_total*100:.0f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────
# 5. Export functions
# ─────────────────────────────────────────────────────────────────────

def export_masked_image(event_id, date_str, period_label):
    """Submit one GEE export task: QA-masked VNP46A2 → Google Drive."""
    cfg = EVENTS[event_id]
    roi = ee.Geometry.Rectangle(cfg["bounds"])

    date_ee = ee.Date(date_str)
    img = ee.Image(
        ee.ImageCollection(VNP46A2_COLLECTION)
        .filterBounds(roi)
        .filterDate(date_ee, date_ee.advance(1, "day"))
        .first()
    )
    ntl_clean = apply_qa_and_scale(img).clip(roi)

    fname = f"{cfg['drive_root']}_{period_label}_{date_str}"
    folder = f"{cfg['drive_root']}-VNP46A2-{period_label}"

    task = ee.batch.Export.image.toDrive(
        image=ntl_clean,
        description=fname,
        folder=folder,
        fileNamePrefix=fname,
        region=roi,
        scale=500,
        crs=cfg["metric_crs"],
        maxPixels=1e8,
    )
    task.start()
    return task, fname


def build_download_lists(event_id, screening_df):
    """From screening results, pick usable dates for pre and post."""
    cfg = EVENTS[event_id]
    usable = screening_df[screening_df["usable"]].copy()

    pre_dates = usable[
        (usable["date"] >= cfg["pre_start"]) &
        (usable["date"] <= cfg["pre_end"])
    ]["date"].tolist()

    post_dates = usable[
        (usable["date"] >= cfg["post_start"]) &
        (usable["date"] <= cfg["post_end"])
    ]["date"].tolist()

    return pre_dates, post_dates


# ─────────────────────────────────────────────────────────────────────
# 6. Task monitoring
# ─────────────────────────────────────────────────────────────────────

def check_tasks():
    """Print GEE task status summary."""
    tasks = ee.batch.Task.list()
    states = {}
    for t in tasks[:200]:  # last 200 tasks
        s = t.status()["state"]
        states[s] = states.get(s, 0) + 1
    print("GEE task status:")
    for s, n in sorted(states.items()):
        print(f"  {s}: {n}")


# ─────────────────────────────────────────────────────────────────────
# 7. Main pipeline
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 3 NTL download via GEE")
    parser.add_argument("--events", nargs="*", default=None,
                        help="Event IDs to process (default: all)")
    parser.add_argument("--step", choices=["screen", "export", "status", "all"],
                        default="all", help="Pipeline step to run")
    parser.add_argument("--project", default="deductive-tempo-485113-n8",
                        help="GEE Cloud project ID")
    parser.add_argument("--output-dir",
                        default=os.path.join(os.path.dirname(__file__), "data"),
                        help="Directory for cloud screening CSVs")
    args = parser.parse_args()

    init_gee(args.project)

    event_ids = args.events or list(EVENTS.keys())
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Stage 3 NTL Download Pipeline")
    print(f"Events: {len(event_ids)}")
    print(f"Step:   {args.step}")
    print(f"{'='*60}\n")

    # ── Status check ──
    if args.step == "status":
        check_tasks()
        return

    # ── Cloud screening ──
    screening = {}
    if args.step in ("screen", "all"):
        print("[1/2] Cloud screening ...")
        for eid in event_ids:
            csv_path = os.path.join(args.output_dir, f"{eid}_cloud_screening.csv")
            if os.path.exists(csv_path) and args.step != "screen":
                screening[eid] = pd.read_csv(csv_path)
                print(f"  {eid}: loaded cached screening ({len(screening[eid])} days)")
            else:
                screening[eid] = screen_event(eid)
                screening[eid].to_csv(csv_path, index=False)
                print(f"  {eid}: saved → {csv_path}")
    else:
        # Load from cache for export step
        for eid in event_ids:
            csv_path = os.path.join(args.output_dir, f"{eid}_cloud_screening.csv")
            if os.path.exists(csv_path):
                screening[eid] = pd.read_csv(csv_path)
            else:
                print(f"  [WARN] No screening for {eid} — run --step screen first")

    # ── Export ──
    if args.step in ("export", "all"):
        print(f"\n[2/2] Submitting GEE export tasks ...")
        total_tasks = 0
        for eid in event_ids:
            if eid not in screening:
                print(f"  {eid}: SKIP (no screening data)")
                continue

            pre_dates, post_dates = build_download_lists(eid, screening[eid])
            print(f"\n  {eid}: {len(pre_dates)} pre + {len(post_dates)} post = {len(pre_dates)+len(post_dates)} tasks")

            for d in pre_dates:
                export_masked_image(eid, d, "pre")
                total_tasks += 1

            for d in post_dates:
                export_masked_image(eid, d, "post")
                total_tasks += 1

            # Small delay to avoid API throttle
            time.sleep(1)

        print(f"\n{'='*60}")
        print(f"Submitted {total_tasks} export tasks to Google Drive.")
        print(f"Monitor with: python {__file__} --step status")
        print(f"{'='*60}")

    # ── Summary ──
    if screening:
        print(f"\n{'Event':<22} {'Pre':>4} {'Post':>5} {'Total':>6} {'Usable%':>8}")
        print("-" * 50)
        grand_pre = grand_post = 0
        for eid in event_ids:
            if eid in screening:
                pre_dates, post_dates = build_download_lists(eid, screening[eid])
                grand_pre += len(pre_dates)
                grand_post += len(post_dates)
                total = len(pre_dates) + len(post_dates)
                pct = len(screening[eid][screening[eid]["usable"]]) / len(screening[eid]) * 100
                print(f"  {eid:<22} {len(pre_dates):>3} {len(post_dates):>5} {total:>6} {pct:>7.0f}%")
        print("-" * 50)
        print(f"  {'TOTAL':<22} {grand_pre:>3} {grand_post:>5} {grand_pre+grand_post:>6}")


if __name__ == "__main__":
    main()
