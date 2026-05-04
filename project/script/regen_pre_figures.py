"""
regen_pre_figures.py
====================
Re-generate the docs/pre_figures/ image set for the Model D era.

Replaces / adds:
  03_feature_importance.png      → Model D's 10-feature importance
  06_prob_by_facility_group.png  → Model D in-sample probability by facility group
  07_irma_miami_prob_map.png     → Model D Irma_Miami prob map
  08_ian_fortmyers_prob_map.png  → Model D Ian_FortMyers prob map
  09_hatay_earthquake_prob_map.png → Model D Earthquake_Hatay prob map
  10_miami_generator_validation.png → Model D Miami-Dade R/C overlay (already produced)
  12_equity_gap.png              → NEW: Q1/Q2/Q3 facility-density bar chart (63 % gap)
  13_miami_dade_rc_bar.png       → NEW: Commercial 83 % vs Residential 14 % bar chart

Deletes (they expose Model A/B/C ablation):
  01_model_ABC_comparison.png
  02_modelA_vs_modelB.png
  05_extended_evaluation.png
"""

import os, glob, shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STAGE2_DIR   = os.path.join(PROJECT_ROOT, 'data', 'result', 'stage2')
STAGE3_DIR   = os.path.join(PROJECT_ROOT, 'data', 'result', 'stage3')
DADE_SHP     = os.path.join(PROJECT_ROOT, 'data', 'dade_test', 'miamidade_filtered.shp')
FIG_DIR      = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'docs', 'pre_figures'))

# Custom diverging palette (low → dark blue, high → orange)
PROB_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'prob_cmap',
    ['#0a1f3d', '#143a6b', '#1f6dab', '#2db5e4', '#a4f0a4', '#ffd84d', '#ff5e3a'],
    N=256,
)

# ─────────────────────────────────────────────────────────────────
# 1. Delete A/B/C ablation figures
# ─────────────────────────────────────────────────────────────────
for fname in ['01_model_ABC_comparison.png',
              '02_modelA_vs_modelB.png',
              '05_extended_evaluation.png']:
    p = os.path.join(FIG_DIR, fname)
    if os.path.exists(p):
        os.remove(p)
        print(f"deleted: {fname}")


# ─────────────────────────────────────────────────────────────────
# 2. 03 · Model D feature importance
# ─────────────────────────────────────────────────────────────────
imp = pd.read_csv(os.path.join(STAGE2_DIR, 'feature_importance_modelD.csv'))
imp = imp.sort_values('avg_imp', ascending=True)

fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
y = np.arange(len(imp))
w = 0.4
ax.barh(y - w/2, imp['rf_imp'],  height=w, color='#3a7bb8', label='Random Forest')
ax.barh(y + w/2, imp['xgb_imp'], height=w, color='#e8623a', label='XGBoost')
ax.set_yticks(y)
ax.set_yticklabels(imp['feature'], fontsize=11)
ax.set_xlabel('Feature Importance (Gini / Gain)', fontsize=11)
ax.set_title('Stage 2 Model · Feature Importance (10 NTL behavior features)',
             fontsize=14, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, axis='x', alpha=0.3)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '03_feature_importance.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 03_feature_importance.png")


# ─────────────────────────────────────────────────────────────────
# 3. 06 · Model D probability by facility group (in-sample)
# ─────────────────────────────────────────────────────────────────
import joblib
panel = pd.read_parquet(os.path.join(STAGE2_DIR, 'pixel_panel.parquet'))

# Build Model D features
def engineer_features_modelD(df):
    df = df.copy()
    df['drop_magnitude']    = -df['delta_ntl'].clip(upper=0)
    df['log_pre_ntl']       = np.log1p(df['pre_mean_ntl'])
    df['log_post_ntl']      = np.log1p(df['post_mean_ntl'])
    df['log_city_pre_mean'] = np.log1p(df['city_pre_mean'])
    df['ntl_relative']      = df['pre_mean_ntl'] / (df['city_pre_mean'] + 1e-6)
    cm = df.groupby('event_id')['pre_mean_ntl'].transform('median')
    df['below_city_median'] = (df['pre_mean_ntl'] < cm).astype(np.uint8)
    df['city_size_code']    = df['city_size'].map({'large':0,'medium':1,'small':2}).fillna(1)
    df['is_hurricane']      = (df['disaster_type'] == 'hurricane').astype(np.uint8)
    df['is_earthquake']     = (df['disaster_type'] == 'earthquake').astype(np.uint8)
    feats = ['drop_magnitude','delta_ntl','log_pre_ntl','log_post_ntl',
             'log_city_pre_mean','ntl_relative','below_city_median',
             'city_size_code','is_hurricane','is_earthquake']
    return df, feats

panel_d, feats = engineer_features_modelD(panel)
rf  = joblib.load(os.path.join(STAGE2_DIR, 'rf_modelD.pkl'))
xgb = joblib.load(os.path.join(STAGE2_DIR, 'xgb_modelD.pkl'))
X = panel_d[feats].fillna(0).values
panel_d['rf_prob']  = rf.predict_proba(X)[:, 1]
panel_d['xgb_prob'] = xgb.predict_proba(X)[:, 1]

HIGH = ['hospital', 'aerodrome', 'power_plant']
MED  = ['fire_station', 'police']
EXCL = ['government', 'substation', 'water_works']

def assign_group(row):
    if row['in_buffer_strict'] == 1:
        if row['nearest_fac_type'] in HIGH:    return 'Group 1\n(hospital/airport/\npower_plant)'
        if row['nearest_fac_type'] in MED:     return 'Group 2\n(fire_station/\npolice)'
    if row['in_buffer'] == 1 and row['nearest_fac_type'] in EXCL:
        return 'Group 3\n(excluded types only)'
    if row['in_buffer'] == 0:
        return 'Outside buffer'
    return None

panel_d['group'] = panel_d.apply(assign_group, axis=1)
plot_df = panel_d.dropna(subset=['group'])
order = ['Group 1\n(hospital/airport/\npower_plant)',
         'Group 2\n(fire_station/\npolice)',
         'Group 3\n(excluded types only)',
         'Outside buffer']

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), facecolor='white')
group_colors = ['#3a7bb8', '#e8623a', '#888888', '#cccccc']

for ax, prob_col, title in zip(axes, ['rf_prob', 'xgb_prob'], ['Random Forest', 'XGBoost']):
    data = [plot_df[plot_df['group'] == g][prob_col].values for g in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True, widths=0.55,
                    medianprops=dict(color='black', lw=1.2),
                    flierprops=dict(marker='o', markersize=2, markerfacecolor='gray', alpha=0.4))
    for patch, c in zip(bp['boxes'], group_colors):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    ax.axhline(0.5, ls='--', color='gray', alpha=0.6, label='0.5 threshold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted P(backup power present)', fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='lower right', fontsize=9)

fig.suptitle('Predicted Probability by Facility Group (Stage 2 Model)',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '06_prob_by_facility_group.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 06_prob_by_facility_group.png")


# ─────────────────────────────────────────────────────────────────
# 4. Per-event probability maps (Model D)
# ─────────────────────────────────────────────────────────────────
def render_prob_map(event_id, dash_label, out_filename):
    tif_path = os.path.join(STAGE2_DIR, f'{event_id}_prob_map_modelD.tif')
    poi_path = os.path.join(STAGE2_DIR, 'poi_cache', f'{event_id}_poi.csv')
    if not os.path.exists(tif_path):
        print(f"  [skip] no tif for {event_id}"); return

    with rasterio.open(tif_path) as src:
        prob = src.read(1).astype(np.float32)
        prob[prob == 0] = np.nan
        bounds = src.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, ax = plt.subplots(figsize=(11, 8), facecolor='#0d1626')
    ax.set_facecolor('#0d1626')
    im = ax.imshow(prob, cmap=PROB_CMAP, vmin=0, vmax=1, extent=extent,
                   origin='upper', interpolation='nearest', aspect='equal',
                   alpha=0.92)

    # Overlay POIs by facility type
    if os.path.exists(poi_path):
        pois = pd.read_csv(poi_path)
        pois = pois[(pois['lon'].between(bounds.left, bounds.right)) &
                    (pois['lat'].between(bounds.bottom, bounds.top))]
        style = {
            'hospital':     ('+',  '#00e5a0', 110, 'Hospital'),
            'aerodrome':    ('^',  '#00d4ff',  90, 'Airport'),
            'power_plant':  ('D',  '#c084fc',  70, 'Power plant'),
            'fire_station': ('s',  '#ffaa00',  60, 'Fire station'),
            'police':       ('o',  '#60a5fa',  50, 'Police'),
        }
        for ftype, (m, c, s, lbl) in style.items():
            sub = pois[pois['facility_type'] == ftype]
            if not sub.empty:
                ax.scatter(sub['lon'], sub['lat'], marker=m, s=s, c=c,
                           edgecolors='black', linewidths=0.6,
                           label=f'{lbl} ({len(sub)})', zorder=5)

    ax.set_xlabel('Longitude', color='#cfd9e6', fontsize=11)
    ax.set_ylabel('Latitude',  color='#cfd9e6', fontsize=11)
    ax.tick_params(colors='#9caebc', labelsize=9)
    for spine in ax.spines.values(): spine.set_color('#2a3d5a')

    ax.set_title(f'{dash_label} — Predicted Backup Power Probability',
                 color='#ffffff', fontsize=14, fontweight='bold', pad=14)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('P(backup power)', color='#cfd9e6', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='#9caebc', labelcolor='#cfd9e6')
    cbar.outline.set_edgecolor('#2a3d5a')

    if os.path.exists(poi_path):
        leg = ax.legend(loc='upper right', facecolor='#0d1626',
                        edgecolor='#2a3d5a', labelcolor='#e7eef7',
                        fontsize=9, framealpha=0.92, ncol=1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, out_filename),
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ {out_filename}")

render_prob_map('Irma_Miami',       'Hurricane Irma · Miami, FL',           '07_irma_miami_prob_map.png')
render_prob_map('Ian_FortMyers',    'Hurricane Ian · Fort Myers, FL',       '08_ian_fortmyers_prob_map.png')
render_prob_map('Earthquake_Hatay', 'Türkiye Earthquake · Hatay (2023)',    '09_hatay_earthquake_prob_map.png')


# ─────────────────────────────────────────────────────────────────
# 5. 10 · Replace with Model D Miami-Dade R/C overlay
# ─────────────────────────────────────────────────────────────────
src_viz = os.path.join(PROJECT_ROOT, 'nightlight-dashboard', 'public',
                       'data', 'miami_generator_validation.png')
if os.path.exists(src_viz):
    shutil.copyfile(src_viz, os.path.join(FIG_DIR, '10_miami_generator_validation.png'))
    print("✓ 10_miami_generator_validation.png  (copied from dashboard public/data)")


# ─────────────────────────────────────────────────────────────────
# 6. NEW · 12_equity_gap.png — Q1/Q2/Q3 facility-density bar
# ─────────────────────────────────────────────────────────────────
tiers = [f'Low outage (Q1)\nn = 366', f'Medium (Q2)\nn = 265', f'High outage (Q3)\nn = 304']
fac_density = [0.490, 0.311, 0.309]
colors = ['#3a7bb8', '#9aa0a6', '#e8623a']

fig, ax = plt.subplots(figsize=(8, 5.5), facecolor='white')
bars = ax.bar(tiers, fac_density, color=colors, alpha=0.9,
              edgecolor='black', linewidth=0.8, width=0.55)
for b, v in zip(bars, fac_density):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.2f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Annotate the gap
ax.annotate('', xy=(2, 0.32), xytext=(0, 0.48),
            arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=1.8))
ax.text(1, 0.40, '63 % of Q1\n(t = 2.56, p = 0.011)', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff5f0',
                  edgecolor='#c0392b', linewidth=1.2))

ax.set_ylabel('Critical-facility density  (per km²)', fontsize=11)
ax.set_title('Infrastructure Equity Gap (Stage 3)\nMost outage-vulnerable ZIPs have only 63 % of the facility density of least-affected ZIPs',
             fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(0, 0.6)
ax.tick_params(axis='x', labelsize=10)
ax.grid(True, axis='y', alpha=0.3)
ax.set_axisbelow(True)
for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '12_equity_gap.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 12_equity_gap.png  (NEW)")


# ─────────────────────────────────────────────────────────────────
# 7. NEW · 13_miami_dade_rc_bar.png
# ─────────────────────────────────────────────────────────────────
groups = ['Commercial\n(n = 30)', 'Residential\n(n = 106)']
pct_above_med = [83, 14]
colors_rc = ['#fff14a', '#ff8c42']

fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor='white')
bars = ax.bar(groups, pct_above_med, color=colors_rc, alpha=0.92,
              edgecolor='black', linewidth=0.8, width=0.55)
for b, v in zip(bars, pct_above_med):
    ax.text(b.get_x() + b.get_width()/2, v + 2.5, f'{v} %',
            ha='center', va='bottom', fontsize=15, fontweight='bold')

ax.axhline(50, ls='--', color='gray', alpha=0.55, lw=1)
ax.text(1.45, 51.5, 'event-wide median (50 %)', ha='right',
        fontsize=9, color='gray', style='italic')

ax.set_ylabel('% of permit locations scoring above the\nevent-wide median predicted probability',
              fontsize=10.5)
ax.set_title('Miami-Dade Generator Permits · Ground-Truth Sanity Check\n(Hurricane Irma 2017)',
             fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(0, 105)
ax.grid(True, axis='y', alpha=0.3)
ax.set_axisbelow(True)
for spine in ['top', 'right']: ax.spines[spine].set_visible(False)

ax.text(0.5, -0.20,
        'Commercial generators (hospitals, hotels, offices) are detectable.\n'
        'Residential generators remain below the 500 m / 25-hectare noise floor.',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=10, color='#444', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '13_miami_dade_rc_bar.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 13_miami_dade_rc_bar.png  (NEW)")


print("\n=== Done. ===")
print(f"Figures in {FIG_DIR}:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.endswith('.png'):
        sz = os.path.getsize(os.path.join(FIG_DIR, f)) / 1024
        print(f"  {f:<45s} {sz:>8.0f} KB")
