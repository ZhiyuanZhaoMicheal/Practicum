"""
year_end_show.py
================
Generate the Year End Show panel materials:

  01_hero_ntl_before_after.png    — Hurricane Maria · pre vs post NTL drama shot
  02_facility_signal_miami.png    — Miami-Dade Production Model + permits overlay
  03_probability_heatmap.png      — three-event Production Model probability panel
  04_national_coverage.png        — 25-event US map with disaster-type encoding

  captions.md                     — captions for the four images
  project_intro.md                — 100-200 word project intro
  qr_code_dashboard.png           — QR code linking to the live dashboard
  Practicum_Backup_Power_Detection_<members>.zip  — final exhibit bundle

Output folder: docs/year_end_show/
"""

import os, glob, shutil, zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
from matplotlib.lines import Line2D
import qrcode

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED    = os.path.join(PROJECT_ROOT, 'data', 'processed')
STAGE2_DIR   = os.path.join(PROJECT_ROOT, 'data', 'result', 'stage2')
DADE_SHP     = os.path.join(PROJECT_ROOT, 'data', 'dade_test', 'miamidade_filtered.shp')
OUT_DIR      = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'docs', 'year_end_show'))
os.makedirs(OUT_DIR, exist_ok=True)

DASHBOARD_URL = 'https://zhiyuanzhaomicheal.github.io/Practicum/'

# ─── Aesthetic ────────────────────────────────────────────────────────
BG          = '#06101e'
GRID        = '#0e2238'
INK         = '#e8eef7'
INK_DIM     = '#9caebc'
ACCENT_CYAN = '#00d4ff'
ACCENT_AMB  = '#ffaa33'
ACCENT_RED  = '#ff5e3a'

# Custom NTL palette: dark navy → cyan → yellow → orange (high-contrast for panels)
PROB_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'prob_cmap',
    ['#0a1f3d', '#143a6b', '#1f6dab', '#2db5e4', '#a4f0a4', '#ffd84d', '#ff5e3a'],
    N=256,
)
NTL_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'ntl_cmap',
    ['#000814', '#001d3d', '#003566', '#5a189a', '#ffaa33', '#ffe680', '#ffffff'],
    N=256,
)


def style_axis(ax, dark=True):
    bg = BG if dark else 'white'
    fg = INK if dark else '#222'
    fg_dim = INK_DIM if dark else '#777'
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg_dim, labelsize=9)
    for spine in ax.spines.values(): spine.set_color('#1f3a5e' if dark else '#ccc')
    return fg, fg_dim


def stack_mean(folder, low_pct=None, high_pct=None):
    """Mean-stack all TIFs in a folder, with optional percentile clipping."""
    tifs = sorted(glob.glob(os.path.join(folder, '*.tif')))
    arrs = []
    for t in tifs:
        with rasterio.open(t) as src:
            arr = src.read(1).astype('float32')
            if src.nodata is not None: arr[arr == src.nodata] = np.nan
            arr[arr < 0] = np.nan
            arrs.append(arr)
    if not arrs: return None, None
    stack = np.nanmean(arrs, axis=0)
    with rasterio.open(tifs[0]) as src: bounds = src.bounds
    if low_pct is not None and high_pct is not None:
        lo, hi = np.nanpercentile(stack, [low_pct, high_pct])
        stack = np.clip(stack, lo, hi)
    return stack, bounds


# ════════════════════════════════════════════════════════════════════
# FIGURE 1 · HERO · Maria pre vs post NTL drama shot
# ════════════════════════════════════════════════════════════════════
def fig01_hero():
    print("\n[1/4] HERO — Hurricane Maria NTL before/after")
    pre,  bounds = stack_mean(os.path.join(PROCESSED, 'Maria-VNP46A2-pre'),  low_pct=2, high_pct=98)
    post, _      = stack_mean(os.path.join(PROCESSED, 'Maria-VNP46A2-post'), low_pct=2, high_pct=98)
    if pre is None or post is None:
        print("   [SKIP] Maria TIFs missing"); return

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    vmax = max(np.nanpercentile(pre, 99), np.nanpercentile(post, 99))
    vmin = 0

    fig = plt.figure(figsize=(18, 10.5), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.18, 1.0], width_ratios=[1, 1],
                          hspace=0.06, wspace=0.07,
                          left=0.04, right=0.97, top=0.94, bottom=0.04)

    # Title row
    ax_t = fig.add_subplot(gs[0, :]); ax_t.set_facecolor(BG); ax_t.axis('off')
    ax_t.text(0.5, 0.78, 'When the Lights Go Out',
              ha='center', va='top', color=INK, fontsize=42, fontweight='bold',
              family='sans-serif')
    ax_t.text(0.5, 0.25,
              'Hurricane Maria · San Juan, Puerto Rico ·  satellite nighttime light, before vs after landfall (Sep 20, 2017)',
              ha='center', va='top', color=INK_DIM, fontsize=15)

    for col, (arr, label, sub) in enumerate([
        (pre,  'BEFORE',  '2017-08-01 → 2017-09-19   ·   pre-disaster mean NTL'),
        (post, 'AFTER',   '2017-09-20 → 2017-12-31   ·   post-landfall mean NTL'),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor('#000814')
        ax.imshow(arr, cmap=NTL_CMAP, vmin=vmin, vmax=vmax,
                  extent=extent, origin='upper', interpolation='bilinear', aspect='equal')
        ax.text(0.025, 0.965, label,
                transform=ax.transAxes, ha='left', va='top', color=INK,
                fontsize=28, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#06101eee',
                          edgecolor=ACCENT_CYAN, linewidth=2))
        ax.text(0.025, 0.04, sub, transform=ax.transAxes, ha='left', va='bottom',
                color=INK_DIM, fontsize=11, family='monospace')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color('#0e2238')

    # Annotation arrow on AFTER panel pointing at residual bright spots
    ax_after = fig.axes[2]
    ax_after.annotate(
        'lights stay on  →  candidate backup-power signature',
        xy=(-66.07, 18.41), xytext=(-66.18, 18.34),
        color=ACCENT_AMB, fontsize=11.5, ha='left', va='top', fontweight='bold',
        arrowprops=dict(arrowstyle='-|>', color=ACCENT_AMB, lw=1.6,
                        connectionstyle='arc3,rad=0.18'),
    )

    out = os.path.join(OUT_DIR, '01_hero_ntl_before_after.png')
    plt.savefig(out, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"   ✓ {out}")


# ════════════════════════════════════════════════════════════════════
# FIGURE 2 · Miami-Dade Production Model + permit overlay
# ════════════════════════════════════════════════════════════════════
def fig02_miami():
    print("\n[2/4] FACILITY SIGNAL — Miami-Dade Production Model + permits")
    tif = os.path.join(STAGE2_DIR, 'Irma_Miami_prob_map_modelD.tif')
    with rasterio.open(tif) as src:
        prob = src.read(1).astype('float32')
        prob[prob == 0] = np.nan
        bounds = src.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    gen = gpd.read_file(DADE_SHP).to_crs('EPSG:4326')
    gen = gen[gen.geometry.notna() & gen.geometry.is_valid].copy()
    gen = gen[(gen.geometry.x.between(bounds.left, bounds.right)) &
              (gen.geometry.y.between(bounds.bottom, bounds.top))]
    gen_R = gen[gen['RESCOMM'] == 'R']
    gen_C = gen[gen['RESCOMM'] == 'C']

    fig = plt.figure(figsize=(15, 10), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.16, 1.0],
                          hspace=0.04, left=0.05, right=0.97, top=0.95, bottom=0.05)

    ax_t = fig.add_subplot(gs[0]); ax_t.set_facecolor(BG); ax_t.axis('off')
    ax_t.text(0.5, 0.85, 'Where Backup Power Runs',
              ha='center', va='top', color=INK, fontsize=36, fontweight='bold')
    ax_t.text(0.5, 0.30,
              'Hurricane Irma · Miami-Dade County · Production Model probability + 148 permitted standalone generators',
              ha='center', va='top', color=INK_DIM, fontsize=14)

    ax = fig.add_subplot(gs[1]); ax.set_facecolor('#000814')
    im = ax.imshow(prob, cmap=PROB_CMAP, vmin=0, vmax=1, extent=extent,
                   origin='upper', interpolation='nearest', aspect='equal', alpha=0.96)

    if len(gen_R):
        ax.scatter(gen_R.geometry.x, gen_R.geometry.y,
                   marker='o', s=55, facecolors='#ff8c42',
                   edgecolors='black', linewidths=0.6, zorder=4, alpha=0.85,
                   label=f'Residential permits (n = {len(gen_R)})')
    if len(gen_C):
        ax.scatter(gen_C.geometry.x, gen_C.geometry.y,
                   marker='D', s=190, facecolors='#fff14a',
                   edgecolors='black', linewidths=1.0, zorder=5,
                   label=f'Commercial permits (n = {len(gen_C)})')

    ax.set_xlabel('Longitude', color=INK_DIM, fontsize=11)
    ax.set_ylabel('Latitude',  color=INK_DIM, fontsize=11)
    style_axis(ax, dark=True)
    cbar = plt.colorbar(im, ax=ax, fraction=0.034, pad=0.015)
    cbar.set_label('Predicted P(backup power)', color=INK, fontsize=11)
    cbar.ax.yaxis.set_tick_params(color=INK_DIM, labelcolor=INK_DIM)
    cbar.outline.set_edgecolor('#1f3a5e')

    leg = ax.legend(loc='upper right', facecolor=BG, edgecolor='#1f3a5e',
                    labelcolor=INK, fontsize=11, framealpha=0.92)

    # Headline result inset
    ax.text(0.025, 0.96,
            '83% of commercial permits  ·  above event median',
            transform=ax.transAxes, ha='left', va='top', color='#fff14a',
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.55', facecolor='#06101eee',
                      edgecolor='#fff14a', linewidth=1.4))
    ax.text(0.025, 0.90,
            'only 14% of residential permits do',
            transform=ax.transAxes, ha='left', va='top', color='#ff8c42',
            fontsize=11.5,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#06101eee',
                      edgecolor='#ff8c42', linewidth=1.2))

    out = os.path.join(OUT_DIR, '02_facility_signal_miami.png')
    plt.savefig(out, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"   ✓ {out}")


# ════════════════════════════════════════════════════════════════════
# FIGURE 3 · 3-event Production Model probability panel
# ════════════════════════════════════════════════════════════════════
def fig03_probability_panel():
    print("\n[3/4] PROBABILITY PANEL — three events (Maria · Ian · Hatay)")
    events = [
        ('Maria_SanJuan',         'Hurricane Maria',    'San Juan, PR · 2017'),
        ('Ian_FortMyers',         'Hurricane Ian',      'Fort Myers, FL · 2022'),
        ('Matthew_Jacksonville',  'Hurricane Matthew',  'Jacksonville, FL · 2016'),
    ]

    fig = plt.figure(figsize=(20, 9), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.22, 1.0],
                          hspace=0.05, wspace=0.05,
                          left=0.03, right=0.985, top=0.94, bottom=0.04)

    ax_t = fig.add_subplot(gs[0, :]); ax_t.set_facecolor(BG); ax_t.axis('off')
    ax_t.text(0.5, 0.78, 'Predicting Backup-Power Activity from Space',
              ha='center', va='top', color=INK, fontsize=38, fontweight='bold')
    ax_t.text(0.5, 0.20,
              'Production Model  ·  pure NTL temporal pattern  ·  10 features  ·  '
              'Leave-One-Event-Out AUC = 0.704 across 25 disasters',
              ha='center', va='top', color=ACCENT_CYAN, fontsize=14, family='monospace')

    for col, (eid, name, sub) in enumerate(events):
        tif = os.path.join(STAGE2_DIR, f'{eid}_prob_map_modelD.tif')
        if not os.path.exists(tif): continue
        with rasterio.open(tif) as src:
            prob = src.read(1).astype('float32'); prob[prob==0] = np.nan
            bounds = src.bounds
        ax = fig.add_subplot(gs[1, col]); ax.set_facecolor('#000814')
        im = ax.imshow(prob, cmap=PROB_CMAP, vmin=0, vmax=1,
                       extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                       origin='upper', interpolation='nearest', aspect='equal', alpha=0.96)
        ax.text(0.03, 0.96, name, transform=ax.transAxes,
                ha='left', va='top', color=INK, fontsize=18, fontweight='bold')
        ax.text(0.03, 0.91, sub, transform=ax.transAxes,
                ha='left', va='top', color=INK_DIM, fontsize=11, family='monospace')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color('#0e2238')

    # Single shared colorbar at bottom
    cbar_ax = fig.add_axes([0.30, 0.025, 0.40, 0.012])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Predicted P(backup power)', color=INK, fontsize=11)
    cbar.ax.tick_params(colors=INK_DIM, labelcolor=INK_DIM)
    cbar.outline.set_edgecolor('#1f3a5e')

    out = os.path.join(OUT_DIR, '03_probability_heatmap.png')
    plt.savefig(out, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"   ✓ {out}")


# ════════════════════════════════════════════════════════════════════
# FIGURE 4 · 25-event national coverage + finding inset
# ════════════════════════════════════════════════════════════════════
def fig04_national():
    print("\n[4/4] NATIONAL COVERAGE — 25 events + key finding")
    # Pull event metadata from regen_modelD_prob_maps PRE_FOLDER + bounds from TIFs
    events_meta = [
        ('Maria_SanJuan',         'Hurricane Maria · San Juan, PR',          -66.07, 18.40, 'hurricane'),
        ('Irma_Miami',            'Hurricane Irma · Miami, FL',              -80.20, 25.80, 'hurricane'),
        ('Ida_NewOrleans',        'Hurricane Ida · New Orleans, LA',         -90.05, 29.95, 'hurricane'),
        ('Laura_LakeCharles',     'Hurricane Laura · Lake Charles, LA',      -93.20, 30.20, 'hurricane'),
        ('Michael_PanamaCity',    'Hurricane Michael · Panama City, FL',     -85.65, 30.16, 'hurricane'),
        ('Earthquake_SanJuan',    'Earthquake · San Juan, PR',               -66.07, 18.40, 'earthquake'),
        ('Ian_CharlotteHarbor',   'Hurricane Ian · Charlotte Harbor, FL',    -82.07, 26.92, 'hurricane'),
        ('Ian_FortMyers',         'Hurricane Ian · Fort Myers, FL',          -81.86, 26.64, 'hurricane'),
        ('Earthquake_Hatay',      'Türkiye Earthquake · Hatay',               36.16, 36.20, 'earthquake'),
        ('Florence_Wilmington',   'Hurricane Florence · Wilmington, NC',     -77.95, 34.23, 'hurricane'),
        ('Irma_Savannah',         'Hurricane Irma · Savannah, GA',           -81.10, 32.08, 'hurricane'),
        ('Isaias_Newark',         'TS Isaias · Newark, NJ',                  -74.17, 40.74, 'hurricane'),
        ('Matthew_Jacksonville',  'Hurricane Matthew · Jacksonville, FL',    -81.66, 30.33, 'hurricane'),
        ('Zeta_Atlanta',          'Hurricane Zeta · Atlanta, GA',            -84.39, 33.75, 'hurricane'),
        ('Zeta_Birmingham',       'Hurricane Zeta · Birmingham, AL',         -86.81, 33.52, 'hurricane'),
        ('Matthew_Fayetteville',  'Hurricane Matthew · Fayetteville, NC',    -78.88, 35.05, 'hurricane'),
        ('Florence_MyrtleBeach',  'Hurricane Florence · Myrtle Beach, SC',   -78.89, 33.69, 'hurricane'),
        ('Isaias_Westchester',    'TS Isaias · Westchester, NY',             -73.79, 41.13, 'hurricane'),
        ('Uri_Houston',           'Winter Storm Uri · Houston, TX',          -95.40, 29.74, 'winter_storm'),
        ('Derecho_Chicago',       'Derecho · Chicago, IL',                   -87.65, 41.85, 'derecho'),
        ('Severe_Detroit',        'Severe Storms · Detroit, MI',             -83.05, 42.33, 'severe_storm'),
        ('Noreaster_Boston',      "Nor'easter · Boston, MA",                 -71.06, 42.36, 'winter_storm'),
        ('IceStorm_OKC',          'Ice Storm · Oklahoma City, OK',           -97.52, 35.46, 'ice_storm'),
        ('Severe_Nashville',      'Severe Storms · Nashville, TN',           -86.78, 36.16, 'severe_storm'),
        ('Atmos_Seattle',         'Atmospheric River · Seattle, WA',         -122.33, 47.61, 'winter_storm'),
    ]
    df = pd.DataFrame(events_meta, columns=['eid','label','lon','lat','type'])

    color_map = {
        'hurricane':    '#ff5e3a',
        'earthquake':   '#c084fc',
        'winter_storm': '#60a5fa',
        'derecho':      '#a4f0a4',
        'severe_storm': '#ffaa33',
        'ice_storm':    '#7df9ff',
    }

    fig = plt.figure(figsize=(20, 11), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.16, 1.0], width_ratios=[2.4, 1.0],
                          hspace=0.04, wspace=0.05,
                          left=0.03, right=0.985, top=0.95, bottom=0.04)

    ax_t = fig.add_subplot(gs[0, :]); ax_t.set_facecolor(BG); ax_t.axis('off')
    ax_t.text(0.5, 0.85, 'A National-Scale Test',
              ha='center', va='top', color=INK, fontsize=40, fontweight='bold')
    ax_t.text(0.5, 0.30,
              '25 disasters across 17 U.S. states + Puerto Rico + Türkiye  ·  2016 – 2023  ·  ~33,700 satellite pixels',
              ha='center', va='top', color=INK_DIM, fontsize=14)

    # ── Map (CONUS + PR + small Türkiye inset) ────────────────────────
    ax_m = fig.add_subplot(gs[1, 0]); ax_m.set_facecolor('#02060d')
    try:
        states = gpd.read_file('https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json')
        states.boundary.plot(ax=ax_m, color='#1f3a5e', linewidth=0.6)
    except Exception:
        pass

    df_us = df[df['type'].notna() & (df['lon'] > -130) & (df['lat'] < 50) & (df['lat'] > 17)]
    for tp, g in df_us.groupby('type'):
        ax_m.scatter(g['lon'], g['lat'], s=320, c=color_map[tp],
                     edgecolors='black', linewidths=1.2, alpha=0.92, zorder=4,
                     label=f'{tp.replace("_"," ").title()}  ({len(df[df["type"]==tp])})')

    ax_m.set_xlim(-128, -64); ax_m.set_ylim(17, 50)
    ax_m.set_xticks([]); ax_m.set_yticks([])
    for s in ax_m.spines.values(): s.set_color('#0e2238')

    leg = ax_m.legend(loc='lower left', facecolor=BG, edgecolor='#1f3a5e',
                      labelcolor=INK, fontsize=11, framealpha=0.95, title='Disaster type',
                      title_fontsize=11)
    leg.get_title().set_color(INK)

    # ── Side panel with key finding ────────────────────────────────────
    ax_s = fig.add_subplot(gs[1, 1]); ax_s.set_facecolor(BG); ax_s.axis('off')
    ax_s.text(0.0, 1.00, 'KEY VALIDATION RESULT', color=ACCENT_CYAN, fontsize=12,
              fontweight='bold', family='monospace', transform=ax_s.transAxes)
    ax_s.text(0.0, 0.93, 'Predicted P(backup power) ↔ EAGLE-I outage severity',
              color=INK, fontsize=14, fontweight='bold', transform=ax_s.transAxes)

    bullets = [
        ('1,002', 'ZIP-event observations across 22 U.S. events'),
        ('β std ≈ 0.08', 'standardized effect on outage severity'),
        ('p = 0.012', 'after Census + event fixed effects'),
        ('+ direction', 'consistent with where backup is deployed'),
    ]
    y = 0.80
    for big, small in bullets:
        ax_s.text(0.0, y,   big,   color=ACCENT_AMB, fontsize=22,
                  fontweight='bold', transform=ax_s.transAxes)
        ax_s.text(0.0, y-0.06, small, color=INK_DIM, fontsize=11.5,
                  transform=ax_s.transAxes)
        y -= 0.16

    ax_s.text(0.0, 0.15, 'GROUND-TRUTH CHECK', color=ACCENT_CYAN, fontsize=12,
              fontweight='bold', family='monospace', transform=ax_s.transAxes)
    ax_s.text(0.0, 0.08,
              'Miami-Dade generator permits (Hurricane Irma):',
              color=INK, fontsize=12, transform=ax_s.transAxes)
    ax_s.text(0.0, 0.025,
              '83 % commercial detected   ·   only 14 % residential',
              color='#fff14a', fontsize=12.5, fontweight='bold',
              transform=ax_s.transAxes,
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#06101e',
                        edgecolor='#fff14a', linewidth=1.2))

    out = os.path.join(OUT_DIR, '04_national_coverage.png')
    plt.savefig(out, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"   ✓ {out}")


# ════════════════════════════════════════════════════════════════════
# Captions, intro, QR
# ════════════════════════════════════════════════════════════════════
def write_captions():
    txt = """# Image Captions · Year End Show

## 01_hero_ntl_before_after.png — *When the Lights Go Out*
Hurricane Maria (Sep 20, 2017) · San Juan, Puerto Rico.
Mean nighttime light brightness before and after landfall, captured by the NASA VIIRS Black Marble (VNP46A2) sensor at 500 m resolution. The before image shows a fully lit metropolitan area; the after image shows months of widespread darkness with isolated bright pockets — the ones that retained electricity through backup power. These persistent bright pockets are the signal our project detects.

## 02_facility_signal_miami.png — *Where Backup Power Runs*
Hurricane Irma (Sep 2017) · Miami-Dade County. The heatmap is the Production Model's per-pixel predicted probability of backup-power activity. Overlaid are 148 standalone-generator permits filed with the county: 30 commercial (yellow diamonds) and 118 residential (orange dots). 83% of permitted commercial generator locations score above the event-wide median probability — vs. only 14% of residential permits — confirming the 500 m sensor resolves institutional-scale backup behavior but not household units.

## 03_probability_heatmap.png — *Predicting Backup-Power Activity from Space*
Three U.S. hurricanes spanning a decade and a range of urban scales: Hurricane Maria (San Juan, 2017), Hurricane Ian (Fort Myers, 2022), and Hurricane Matthew (Jacksonville, 2016). The Production Model uses only nighttime light temporal patterns — no facility-location features — and achieves Leave-One-Event-Out AUC = 0.704 across all 25 disasters in the panel.

## 04_national_coverage.png — *A National-Scale Test*
The geographic spread of the 25 disasters: 18 hurricanes, 2 earthquakes, 3 winter storms, 2 severe-storm events, plus a derecho and an ice storm, across 17 U.S. states, Puerto Rico, and Türkiye (2016 – 2023). The side panel summarizes the closed-loop validation: the model's predicted backup-power probability is positively associated with EAGLE-I-recorded outage severity (standardized β ≈ 0.08, p = 0.012) — consistent with the natural deployment direction that backup power gets installed where outages strike most.
"""
    with open(os.path.join(OUT_DIR, 'captions.md'), 'w') as f: f.write(txt)
    print(f"   ✓ captions.md")


def write_intro():
    txt = """# Project Introduction

**Detecting Backup Power from Space During Disasters**

When a hurricane knocks out the grid, hospitals, airports, and fire stations
stay lit on backup generators — yet no public registry records where these
generators are or whether they actually work. We ask whether NASA's VIIRS
Black Marble nighttime-light satellite (500 m resolution, daily revisit) can
detect the brightness signature of backup power running during major outages.

Across **25 disasters** in **17 U.S. states, Puerto Rico, and Türkiye
(2016 – 2023)**, we extract a panel of ~33,700 pixels, treat critical
infrastructure (hospitals, airports, fire stations, police, power plants) as
proxy labels, and train a tree-based model on **purely temporal nighttime-light
features** — no spatial proximity to facilities — so its predictions reflect
satellite signal alone. Across all events the model achieves **LOEO
AUC = 0.704**; against actual generator-permit records in Miami-Dade,
**83 % of commercial installations** score above the event-wide median
probability (vs. 14 % of residential), confirming the resolution boundary.
At zip-code scale, model probability is positively associated with real
outage severity (standardized β ≈ 0.08, p = 0.012).

**Authors:** Zhiyuan Zhao · Qiushi Yu  ·  University of Pennsylvania
"""
    with open(os.path.join(OUT_DIR, 'project_intro.md'), 'w') as f: f.write(txt)
    print(f"   ✓ project_intro.md")


def write_qr():
    qr = qrcode.QRCode(version=4, error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=12, border=4)
    qr.add_data(DASHBOARD_URL); qr.make(fit=True)
    img = qr.make_image(fill_color='#06101e', back_color='white').convert('RGB')

    fig, ax = plt.subplots(figsize=(7, 7.6), facecolor='white')
    ax.imshow(np.asarray(img))
    ax.axis('off')
    ax.set_title('Explore the live interactive dashboard',
                 color='#06101e', fontsize=15, fontweight='bold', pad=14)
    fig.text(0.5, 0.04, DASHBOARD_URL,
             ha='center', va='bottom', color='#3a7bb8', fontsize=10, family='monospace')
    out = os.path.join(OUT_DIR, 'qr_code_dashboard.png')
    plt.savefig(out, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"   ✓ {out}  →  {DASHBOARD_URL}")


# ════════════════════════════════════════════════════════════════════
# Bundle into ZIP
# ════════════════════════════════════════════════════════════════════
def make_zip():
    name = 'Practicum_Backup_Power_Detection_Zhao_Yu.zip'
    out_zip = os.path.join(OUT_DIR, name)
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(OUT_DIR)):
            if fname.endswith('.zip'): continue
            full = os.path.join(OUT_DIR, fname)
            if os.path.isfile(full):
                zf.write(full, arcname=fname)
    print(f"\n✓ Bundle: {out_zip}  ({os.path.getsize(out_zip)/1024:.0f} KB)")


# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 70)
    print("Year End Show · panel material generation")
    print("=" * 70)
    fig01_hero()
    fig02_miami()
    fig03_probability_panel()
    fig04_national()
    write_captions()
    write_intro()
    write_qr()
    make_zip()
    print("\n" + "=" * 70)
    print(f"All materials in: {OUT_DIR}")
    print("=" * 70)
