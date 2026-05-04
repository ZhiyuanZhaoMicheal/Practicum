"""
miami_dade_visualization.py
===========================
Render Irma_Miami probability map (Model D) as base layer, with all 169
Miami-Dade generator permits overlaid (split commercial vs residential).

Output:  project/nightlight-dashboard/public/data/miami_generator_validation.png
         (replaces the old Model A image at the same path)
"""

import os
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pyproj import Transformer
from rasterio.warp import reproject, Resampling

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DADE_SHP   = os.path.join(PROJECT_ROOT, 'data', 'dade_test', 'miamidade_filtered.shp')
PROB_TIF   = os.path.join(PROJECT_ROOT, 'data', 'result', 'stage2',
                          'Irma_Miami_prob_map_modelD.tif')
OUT_PNG    = os.path.join(PROJECT_ROOT, 'nightlight-dashboard', 'public', 'data',
                          'miami_generator_validation.png')

# ─── Load probability map (band 1 = RF) ───────────────────────────
with rasterio.open(PROB_TIF) as src:
    prob = src.read(1).astype(np.float32)
    prob[prob == 0] = np.nan         # nodata
    bounds = src.bounds              # (left, bottom, right, top) in WGS84
    transform = src.transform
    print(f"Probability map: {prob.shape}, bounds={bounds}")
    print(f"  CRS: {src.crs}")
    print(f"  Valid prob range: {np.nanmin(prob):.3f} – {np.nanmax(prob):.3f}")

# ─── Load generator permits ──────────────────────────────────────
gen = gpd.read_file(DADE_SHP).to_crs('EPSG:4326')
gen = gen[gen.geometry.notna() & gen.geometry.is_valid].copy()

# Filter to ROI bbox (matches Stage 2 Irma_Miami study area)
LON_MIN, LON_MAX = bounds.left,   bounds.right
LAT_MIN, LAT_MAX = bounds.bottom, bounds.top
gen_in = gen[(gen.geometry.x.between(LON_MIN, LON_MAX)) &
             (gen.geometry.y.between(LAT_MIN, LAT_MAX))].copy()
gen_R = gen_in[gen_in['RESCOMM'] == 'R']
gen_C = gen_in[gen_in['RESCOMM'] == 'C']
print(f"Permits in ROI: {len(gen_in)} (R={len(gen_R)}, C={len(gen_C)})")

# ─── Render ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9), facecolor='#0d1626')
ax.set_facecolor('#0d1626')

extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

# Custom diverging-style colormap: dark for low prob, cyan-yellow for high prob
cmap = mcolors.LinearSegmentedColormap.from_list(
    'prob_cmap',
    ['#0a1f3d', '#143a6b', '#1f6dab', '#2db5e4', '#a4f0a4', '#ffd84d', '#ff5e3a'],
    N=256,
)

im = ax.imshow(prob, cmap=cmap, vmin=0, vmax=1, extent=extent,
               origin='upper', interpolation='nearest', aspect='equal',
               alpha=0.92)

# Plot residential first (so commercial sits on top)
ax.scatter(gen_R.geometry.x, gen_R.geometry.y,
           s=70, marker='o',
           facecolors='#ff8c42', edgecolors='black', linewidths=0.7,
           label=f'Residential permits (n={len(gen_R)})',
           zorder=4, alpha=0.85)
ax.scatter(gen_C.geometry.x, gen_C.geometry.y,
           s=160, marker='D',
           facecolors='#fff14a', edgecolors='black', linewidths=1.0,
           label=f'Commercial permits (n={len(gen_C)})',
           zorder=5)

# Style
ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_xlabel('Longitude', color='#cfd9e6', fontsize=11)
ax.set_ylabel('Latitude',  color='#cfd9e6', fontsize=11)
ax.tick_params(colors='#9caebc', labelsize=9)
for spine in ax.spines.values():
    spine.set_color('#2a3d5a')

ax.set_title(
    'Hurricane Irma · Miami-Dade Ground-Truth Validation',
    color='#ffffff', fontsize=15, fontweight='bold', pad=14)
ax.text(0.5, 1.005,
        'Base: Model D probability (pure NTL).  '
        'Yellow ◆ = commercial generator permits  ·  Orange ● = residential.',
        transform=ax.transAxes, ha='center', va='bottom',
        color='#9caebc', fontsize=10)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label('P(backup power) — Model D', color='#cfd9e6', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='#9caebc', labelcolor='#cfd9e6')
cbar.outline.set_edgecolor('#2a3d5a')

# Legend
leg = ax.legend(loc='upper right', facecolor='#0d1626', edgecolor='#2a3d5a',
                labelcolor='#e7eef7', fontsize=10, framealpha=0.92)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=170, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f"\n✓ Saved {OUT_PNG}")
print(f"  Size: {os.path.getsize(OUT_PNG)/1024:.0f} KB")
