"""
make_modelD_loeo_heatmap.py
===========================
Render the Production Model (Model D, in_buffer_strict label) LOEO heatmap
for the pre_figures slide deck.

Layout:
  · Rows  = 25 events (sorted by RF AUC, descending)
  · Cols  = RF / XGBoost / Logit
  · Cells = LOEO AUC, color-coded
  · A small bar at the bottom-right shows mean AUC per algorithm.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOEO_CSV     = os.path.join(PROJECT_ROOT, 'data', 'result', 'stage2',
                            'loeo_modelD_25events.csv')
OUT_PNG      = os.path.abspath(os.path.join(
    PROJECT_ROOT, '..', 'docs', 'pre_figures', '04_loeo_heatmap.png'))

df = pd.read_csv(LOEO_CSV)
df = df.sort_values('rf_auc', ascending=False).reset_index(drop=True)
auc_mat = df[['rf_auc', 'xgb_auc', 'logit_auc']].values
events  = df['held_out'].values
algos   = ['Random Forest', 'XGBoost', 'Logit']
mean_aucs = auc_mat.mean(axis=0)

cmap = mcolors.LinearSegmentedColormap.from_list(
    'auc', ['#a32f2f', '#d97a2c', '#e5c552', '#a4cf6b', '#3d8f4a'], N=256)

fig, (ax, ax_bar) = plt.subplots(
    1, 2, figsize=(11, 9.5), facecolor='white',
    gridspec_kw=dict(width_ratios=[2.4, 1.0], wspace=0.6))

# ── heatmap ────────────────────────────────────────────────
im = ax.imshow(auc_mat, cmap=cmap, vmin=0.55, vmax=0.85, aspect='auto')

ax.set_xticks(range(len(algos)))
ax.set_xticklabels(algos, fontsize=11, fontweight='bold')
ax.set_yticks(range(len(events)))
ax.set_yticklabels(events, fontsize=9)
ax.tick_params(axis='x', length=0, top=True, labeltop=True, bottom=False, labelbottom=False)
ax.tick_params(axis='y', length=0)

for i in range(len(events)):
    for j in range(len(algos)):
        v = auc_mat[i, j]
        text_color = 'white' if v < 0.66 or v > 0.78 else 'black'
        ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                fontsize=9, color=text_color, fontweight='bold')

for spine in ax.spines.values(): spine.set_visible(False)
ax.set_title('Production Model · Per-Event LOEO AUC (25 disasters · strict label)',
             fontsize=13, fontweight='bold', pad=18)

cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label('LOEO AUC', fontsize=10)

# ── side bar: mean AUC per algo ────────────────────────────
bar_colors = ['#3a7bb8', '#e8623a', '#7d8a93']
bars = ax_bar.barh(range(len(algos)), mean_aucs, color=bar_colors,
                   alpha=0.9, edgecolor='black', linewidth=0.6, height=0.6)
for i, v in enumerate(mean_aucs):
    ax_bar.text(v + 0.005, i, f'{v:.3f}', va='center',
                fontsize=12, fontweight='bold', color='black')

ax_bar.set_yticks(range(len(algos)))
ax_bar.set_yticklabels(algos, fontsize=11)
ax_bar.invert_yaxis()
ax_bar.set_xlim(0.55, 0.85)
ax_bar.set_xlabel('Mean AUC across 25 events', fontsize=10)
ax_bar.set_title('Algorithm-level\nmean AUC',
                 fontsize=11, fontweight='bold', pad=18)
ax_bar.axvline(0.5, color='gray', ls='--', alpha=0.5, lw=0.8)
ax_bar.text(0.555, len(algos) - 0.4, 'random (0.5)', fontsize=8,
            color='gray', style='italic')
ax_bar.grid(True, axis='x', alpha=0.3)
ax_bar.set_axisbelow(True)
for spine in ['top', 'right']: ax_bar.spines[spine].set_visible(False)

fig.text(0.5, 0.02,
         'Ensemble (0.7·RF + 0.3·XGB) reaches 0.704 mean LOEO AUC — '
         'real but modest signal at 500 m / 25-hectare resolution.',
         ha='center', fontsize=10, color='#555', style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 1])
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: {OUT_PNG}  ({os.path.getsize(OUT_PNG)/1024:.0f} KB)")
