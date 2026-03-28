"""
Export Maria San Juan NTL GeoTIFFs as PNG frames for the dashboard animation.
Outputs:
  public/data/frames/maria_<date>.png  — hot colormap NTL
  public/data/frames/maria_bau.png     — pre-disaster median composite
  public/data/frames/maria_delta_<date>.png — delta = (day - BAU) / BAU
  public/data/frames/maria_frames.json — metadata
"""
import os, glob, json
import numpy as np
import rasterio
from PIL import Image

PROJECT = os.path.join(os.path.dirname(__file__), "..")
PRE_DIR = os.path.join(PROJECT, "data/processed/Maria-VNP46A2-pre")
POST_DIR = os.path.join(PROJECT, "data/processed/Maria-VNP46A2-post")
OUT_DIR = os.path.join(os.path.dirname(__file__), "public/data/frames")
os.makedirs(OUT_DIR, exist_ok=True)

VMAX = 7.0  # NTL clamp max for colormap

def hot_colormap(val):
    """Map 0-1 to black-red-orange-yellow-white (hot)."""
    r = np.clip(val * 3, 0, 1)
    g = np.clip(val * 3 - 1, 0, 1)
    b = np.clip(val * 3 - 2, 0, 1)
    return r, g, b

def diverging_colormap(val):
    """Map -1..0..+1 to blue..black..red."""
    r = np.where(val > 0, np.clip(val, 0, 1), 0)
    b = np.where(val < 0, np.clip(-val, 0, 1), 0)
    g = np.zeros_like(val)
    return r, g, b

def arr_to_png(arr, path, cmap='hot', vmin=0, vmax=VMAX):
    """Convert 2D array to PNG with given colormap."""
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)
    norm = np.where(np.isfinite(arr) & (arr > 0), norm, 0)

    if cmap == 'hot':
        r, g, b = hot_colormap(norm)
    else:  # diverging
        norm_div = np.clip(arr / max(abs(vmin), abs(vmax), 1e-9), -1, 1)
        norm_div = np.where(np.isfinite(arr), norm_div, 0)
        r, g, b = diverging_colormap(norm_div)

    alpha = np.where(np.isfinite(arr) & (arr != 0), 255, 0).astype(np.uint8)
    rgba = np.stack([
        (r * 255).astype(np.uint8),
        (g * 255).astype(np.uint8),
        (b * 255).astype(np.uint8),
        alpha,
    ], axis=-1)

    img = Image.fromarray(rgba, 'RGBA')
    img.save(path, optimize=True)

def load_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        arr[arr <= 0] = np.nan
    return arr

def main():
    # Load pre-disaster composite (median)
    composite_path = os.path.join(PRE_DIR, "maria_sanjuan_pre_composite_median.tif")
    if os.path.exists(composite_path):
        bau = load_tif(composite_path)
    else:
        # Compute from individual pre days
        pre_tifs = sorted(glob.glob(os.path.join(PRE_DIR, "*_pre_2017*.tif")))
        stack = np.stack([load_tif(f) for f in pre_tifs])
        bau = np.nanmedian(stack, axis=0)

    # Save BAU
    arr_to_png(bau, os.path.join(OUT_DIR, "maria_bau.png"))
    print(f"BAU composite saved")

    frames = []

    # Pre-disaster frames
    pre_tifs = sorted(glob.glob(os.path.join(PRE_DIR, "*_pre_2017*.tif")))
    for tif in pre_tifs:
        fname = os.path.basename(tif)
        date = fname.split("_pre_")[1].replace(".tif", "")
        arr = load_tif(tif)

        # NTL frame
        png_name = f"maria_{date}.png"
        arr_to_png(arr, os.path.join(OUT_DIR, png_name))

        # Delta frame
        delta = np.where(bau > 0.5, (arr - bau) / bau, 0)
        delta_name = f"maria_delta_{date}.png"
        arr_to_png(delta, os.path.join(OUT_DIR, delta_name), cmap='diverging', vmin=-1, vmax=1)

        frames.append({
            "date": date,
            "ntl": png_name,
            "delta": delta_name,
            "phase": "pre",
            "mean_ntl": round(float(np.nanmean(arr)), 2),
        })
        print(f"  {date} [pre]")

    # Post-disaster frames
    post_tifs = sorted(glob.glob(os.path.join(POST_DIR, "*_post_2017*.tif")))
    for tif in post_tifs:
        fname = os.path.basename(tif)
        date = fname.split("_post_")[1].replace(".tif", "")
        arr = load_tif(tif)

        png_name = f"maria_{date}.png"
        arr_to_png(arr, os.path.join(OUT_DIR, png_name))

        delta = np.where(bau > 0.5, (arr - bau) / bau, 0)
        delta_name = f"maria_delta_{date}.png"
        arr_to_png(delta, os.path.join(OUT_DIR, delta_name), cmap='diverging', vmin=-1, vmax=1)

        frames.append({
            "date": date,
            "ntl": png_name,
            "delta": delta_name,
            "phase": "post",
            "mean_ntl": round(float(np.nanmean(arr)), 2),
        })
        print(f"  {date} [post]")

    # Save metadata
    meta = {
        "event": "maria",
        "disaster_date": "2017-09-20",
        "bau": "maria_bau.png",
        "vmax": VMAX,
        "frames": frames,
    }
    with open(os.path.join(OUT_DIR, "maria_frames.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {len(frames)} frames exported to {OUT_DIR}")

if __name__ == "__main__":
    main()
