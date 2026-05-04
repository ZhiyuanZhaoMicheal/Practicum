"""
miami_dade_groundtruth_validation.py
====================================
Sanity check: at the 148 known Miami-Dade generator permit locations that
fall inside our Irma_Miami study area, what probabilities did the model
actually assign?

Caveats acknowledged:
  · 592 permits total, only ~148 fall within our small ROI bbox
  · Ground truth is INCOMPLETE — only addresses that filed a permit
    (many real generators are unpermitted). Absence of permit ≠ no generator.
  · Therefore we report only "did the model assign meaningful probability
    to these known points" — NOT AUC / precision / recall, which require
    true negatives.

Compares Model A (rf_final) vs Model D (rf_modelD) on the same 148 points.
"""

import os, json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pyproj import Transformer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
STAGE2_DIR   = os.path.join(DATA_DIR, 'result', 'stage2')

DADE_SHP   = os.path.join(DATA_DIR, 'dade_test', 'miamidade_filtered.shp')
PROB_A_TIF = os.path.join(STAGE2_DIR, 'Irma_Miami_prob_map.tif')
PROB_D_TIF = os.path.join(STAGE2_DIR, 'Irma_Miami_prob_map_modelD.tif')

print("="*70)
print("Miami-Dade ground-truth sanity check · Hurricane Irma 2017")
print("="*70)
print("Note: ground truth INCOMPLETE — reporting probability distribution")
print("      at known points only, NOT AUC / precision metrics.")

# ─── Load generator permits ────────────────────────────────────────
gen = gpd.read_file(DADE_SHP).to_crs('EPSG:4326')
gen = gen[gen.geometry.notna() & gen.geometry.is_valid].copy()
print(f"\nTotal generator permit records: {len(gen)}")
print(f"  Type: {gen['DESC1'].iloc[0]}")
print(f"  RESCOMM split (whole county): "
      f"R={int((gen['RESCOMM']=='R').sum())}, "
      f"C={int((gen['RESCOMM']=='C').sum())}")

# Filter to study area bbox
LON_MIN, LON_MAX = -80.40, -80.10
LAT_MIN, LAT_MAX = 25.70,  25.90
mask = ((gen.geometry.x.between(LON_MIN, LON_MAX)) &
        (gen.geometry.y.between(LAT_MIN, LAT_MAX)))
gen_in = gen[mask].copy()
print(f"  Within Irma_Miami study area ({LAT_MIN}-{LAT_MAX}°N, {LON_MIN} to {LON_MAX}°W): {len(gen_in)}")
print(f"    RESCOMM split (in bbox): "
      f"R={int((gen_in['RESCOMM']=='R').sum())}, "
      f"C={int((gen_in['RESCOMM']=='C').sum())}")

def sample_at_points(tif_path, points_gdf, band=1):
    with rasterio.open(tif_path) as src:
        if str(src.crs).upper() != 'EPSG:4326':
            t = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
            xs, ys = t.transform(points_gdf.geometry.x.values, points_gdf.geometry.y.values)
            coords = list(zip(xs, ys))
        else:
            coords = list(zip(points_gdf.geometry.x.values, points_gdf.geometry.y.values))
        vals = [v[band-1] if v[band-1] is not None else np.nan
                for v in src.sample(coords, indexes=[band])]
    return np.array(vals, dtype=np.float32)

def event_median_prob(tif_path, band=1):
    """Median prob over all valid pixels in this event."""
    with rasterio.open(tif_path) as src:
        arr = src.read(band)
        valid = arr[(arr > 0) & np.isfinite(arr)]
    return float(np.median(valid))

def evaluate(model_name, tif_path, gen_gdf):
    print(f"\n─── {model_name} ─────────────────────────────────")
    if not os.path.exists(tif_path):
        print(f"  [SKIP] {tif_path} not found"); return None

    p = sample_at_points(tif_path, gen_gdf, band=1)
    valid = np.isfinite(p) & (p > 0)
    p_v = p[valid]
    n_total   = len(gen_gdf)
    n_in_grid = int(valid.sum())
    n_outside = n_total - n_in_grid

    if n_in_grid == 0:
        print(f"  [WARN] No probability data at any generator point"); return None

    ev_med = event_median_prob(tif_path)

    # Distribution stats
    quartiles = np.percentile(p_v, [0, 25, 50, 75, 100])
    n_above_50  = int((p_v > 0.50).sum())
    n_above_70  = int((p_v > 0.70).sum())
    n_above_med = int((p_v > ev_med).sum())

    print(f"  Generator points sampled:   {n_in_grid} / {n_total}")
    if n_outside:
        print(f"    (skipped {n_outside} points with no NTL data — outside processed pixels)")
    print(f"  Event-wide median prob:     {ev_med:.3f}")
    print(f"  Probability distribution at known generator locations:")
    print(f"    min={quartiles[0]:.3f}  q25={quartiles[1]:.3f}  median={quartiles[2]:.3f}  q75={quartiles[3]:.3f}  max={quartiles[4]:.3f}")
    print(f"    > 0.50:  {n_above_50}/{n_in_grid} ({100*n_above_50/n_in_grid:.0f}%)")
    print(f"    > 0.70:  {n_above_70}/{n_in_grid} ({100*n_above_70/n_in_grid:.0f}%)")
    print(f"    > event median ({ev_med:.3f}):  {n_above_med}/{n_in_grid} ({100*n_above_med/n_in_grid:.0f}%)")

    return {
        'model':        model_name,
        'n_total':      n_total,
        'n_in_bbox':    int(len(gen_gdf)),
        'n_sampled':    n_in_grid,
        'event_median': round(ev_med, 4),
        'quartiles':    [round(float(q), 4) for q in quartiles],
        'pct_above_0_5': round(100 * n_above_50 / n_in_grid, 1),
        'pct_above_0_7': round(100 * n_above_70 / n_in_grid, 1),
        'pct_above_event_median': round(100 * n_above_med / n_in_grid, 1),
    }

results = {
    'note': 'Sanity check only. Ground truth incomplete (permits ⊂ all real generators).',
    'n_total_permits': int(len(gen)),
    'n_within_study_bbox': int(len(gen_in)),
    'study_bbox': {'lon': [LON_MIN, LON_MAX], 'lat': [LAT_MIN, LAT_MAX]},
    'rescomm_split': {
        'R': int((gen_in['RESCOMM']=='R').sum()),
        'C': int((gen_in['RESCOMM']=='C').sum()),
    },
}

gen_R = gen_in[gen_in['RESCOMM'] == 'R'].copy()
gen_C = gen_in[gen_in['RESCOMM'] == 'C'].copy()

print("\n╔══════════════════════════════════════════════════════════╗")
print("║ ALL generators (residential + commercial combined)        ║")
print("╚══════════════════════════════════════════════════════════╝")
results['all_modelA']        = evaluate('Model A · ALL', PROB_A_TIF, gen_in)
results['all_modelD']        = evaluate('Model D · ALL', PROB_D_TIF, gen_in)

print("\n╔══════════════════════════════════════════════════════════╗")
print("║ RESIDENTIAL generators only (RESCOMM == 'R')              ║")
print("╚══════════════════════════════════════════════════════════╝")
results['residential_modelA'] = evaluate('Model A · R', PROB_A_TIF, gen_R)
results['residential_modelD'] = evaluate('Model D · R', PROB_D_TIF, gen_R)

print("\n╔══════════════════════════════════════════════════════════╗")
print("║ COMMERCIAL generators only (RESCOMM == 'C')               ║")
print("║ This is the cohort the original Word doc reports          ║")
print("╚══════════════════════════════════════════════════════════╝")
results['commercial_modelA']  = evaluate('Model A · C', PROB_A_TIF, gen_C)
results['commercial_modelD']  = evaluate('Model D · C', PROB_D_TIF, gen_C)

# Save individual point probs for further inspection
detail_rows = []
for _, r in gen_in.iterrows():
    detail_rows.append({
        'lon': float(r.geometry.x),
        'lat': float(r.geometry.y),
        'address': r.get('STNDADDR'),
        'folio':   r.get('FOLIO'),
        'rescomm': r.get('RESCOMM'),
        'propuse': r.get('PROPUSE'),
    })
det = pd.DataFrame(detail_rows)
det['prob_modelA'] = sample_at_points(PROB_A_TIF, gen_in, band=1)
det['prob_modelD'] = sample_at_points(PROB_D_TIF, gen_in, band=1)
det.to_csv(os.path.join(STAGE2_DIR, 'miami_dade_pointwise_probs.csv'), index=False)
print(f"\nSaved pointwise probs to: miami_dade_pointwise_probs.csv  ({len(det)} rows)")

out = os.path.join(STAGE2_DIR, 'miami_dade_validation.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
print(f"Saved summary to: {out}")
