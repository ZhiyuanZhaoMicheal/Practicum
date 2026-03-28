"""
Export per-event daily cloud coverage and NTL stats for the docs page.
Reads pre/post GeoTIFFs, computes:
  - cloud_free_pct: fraction of valid (non-NaN, >0) pixels per day
  - mean_ntl: spatial mean NTL per day
  - median_ntl: spatial median NTL per day
Outputs: public/data/cloud_stats.json
"""
import os, glob, json
import numpy as np
import rasterio

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUT_DIR = os.path.join(os.path.dirname(__file__), "public", "data")

EVENTS = {
    "Maria_SanJuan":       {"pre": "Maria-VNP46A2-pre",               "post": "Maria-VNP46A2-post"},
    "Irma_Miami":          {"pre": "Irma_Miami-VNP46A2-pre",          "post": "Irma_Miami-VNP46A2-post"},
    "Ida_NewOrleans":      {"pre": "Ida_NewOrleans-VNP46A2-pre",      "post": "Ida_NewOrleans-VNP46A2-post"},
    "Laura_LakeCharles":   {"pre": "Laura_LakeCharles-VNP46A2-pre",   "post": "Laura_LakeCharles-VNP46A2-post"},
    "Michael_PanamaCity":  {"pre": "Michael-VNP46A2-pre",             "post": "Michael-VNP46A2-post"},
    "Earthquake_SanJuan":  {"pre": "Earthquake-VNP46A2-pre",          "post": "Earthquake-VNP46A2-post"},
    "Ian_CharlotteHarbor": {"pre": "Ian_CharlotteHarbor-VNP46A2-pre", "post": "Ian_CharlotteHarbor-VNP46A2-post"},
    "Ian_FortMyers":       {"pre": "Ian_FortMyers-VNP46A2-pre",       "post": "Ian_FortMyers-VNP46A2-post"},
    "Earthquake_Hatay":    {"pre": "Earthquake_Hatay-VNP46A2-pre",    "post": "Earthquake_Hatay-VNP46A2-post"},
}

DASH_IDS = {
    "Maria_SanJuan": "maria", "Irma_Miami": "irma", "Ida_NewOrleans": "ida",
    "Laura_LakeCharles": "laura", "Michael_PanamaCity": "michael",
    "Earthquake_SanJuan": "eq-pr", "Ian_CharlotteHarbor": "ian-charlotte",
    "Ian_FortMyers": "ian-fortmyers", "Earthquake_Hatay": "eq-hatay",
}

def process_tifs(folder_path):
    """Process all TIFs in a folder, return list of daily stats."""
    tifs = sorted(glob.glob(os.path.join(folder_path, "*.tif")))
    days = []
    for tif_path in tifs:
        fname = os.path.splitext(os.path.basename(tif_path))[0]
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)

        total_pixels = arr.size
        # Valid = not NaN and > 0 (0 or NaN = cloud/nodata)
        valid_mask = np.isfinite(arr) & (arr > 0)
        valid_count = int(valid_mask.sum())
        cloud_free_pct = round(valid_count / total_pixels * 100, 1)

        valid_vals = arr[valid_mask]
        mean_ntl = round(float(np.mean(valid_vals)), 2) if valid_vals.size > 0 else 0
        median_ntl = round(float(np.median(valid_vals)), 2) if valid_vals.size > 0 else 0

        days.append({
            "date": fname,
            "cloud_free_pct": cloud_free_pct,
            "mean_ntl": mean_ntl,
            "median_ntl": median_ntl,
            "valid_pixels": valid_count,
            "total_pixels": total_pixels,
        })
    return days


def main():
    all_stats = {}

    for event_id, dirs in EVENTS.items():
        dash_id = DASH_IDS[event_id]
        pre_path = os.path.join(DATA_DIR, dirs["pre"])
        post_path = os.path.join(DATA_DIR, dirs["post"])

        print(f"\n[{event_id}] → {dash_id}")

        pre_days = []
        post_days = []

        if os.path.isdir(pre_path):
            pre_days = process_tifs(pre_path)
            print(f"  Pre:  {len(pre_days)} days, avg cloud-free: {np.mean([d['cloud_free_pct'] for d in pre_days]):.1f}%")
        else:
            print(f"  Pre:  [MISSING] {pre_path}")

        if os.path.isdir(post_path):
            post_days = process_tifs(post_path)
            print(f"  Post: {len(post_days)} days, avg cloud-free: {np.mean([d['cloud_free_pct'] for d in post_days]):.1f}%")
        else:
            print(f"  Post: [MISSING] {post_path}")

        # Summary stats
        all_cloud = [d["cloud_free_pct"] for d in pre_days + post_days]
        below_30 = sum(1 for c in all_cloud if c < 30)

        all_stats[dash_id] = {
            "event_id": event_id,
            "pre": pre_days,
            "post": post_days,
            "summary": {
                "total_days": len(pre_days) + len(post_days),
                "pre_days": len(pre_days),
                "post_days": len(post_days),
                "avg_cloud_free": round(np.mean(all_cloud), 1) if all_cloud else 0,
                "days_below_30pct": below_30,
                "pre_mean_ntl": round(np.mean([d["mean_ntl"] for d in pre_days]), 2) if pre_days else 0,
                "post_mean_ntl": round(np.mean([d["mean_ntl"] for d in post_days]), 2) if post_days else 0,
            }
        }

    out_path = os.path.join(OUT_DIR, "cloud_stats.json")
    with open(out_path, "w") as f:
        json.dump(all_stats, f, separators=(",", ":"))
    print(f"\nSaved: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
