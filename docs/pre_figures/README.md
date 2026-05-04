# Presentation Figures (Model D era)

Figure set for the 10–15 min pre, aligned with the current MODELS.md narrative.
All Stage 2 maps and downstream products use the **Model D** (pure-NTL, no spatial proximity) headline.

| # | File | What it shows | Where to use |
|---|------|---------------|--------------|
| 03 | `03_feature_importance.png` | Top features driving Stage 2 predictions (10 NTL behavior features). `log_pre_ntl` and `log_post_ntl` lead, followed by city / disaster controls. | Methods → "what the model uses" |
| 04 | `04_loeo_heatmap.png` | Production Model · per-event LOEO AUC heatmap (25 disasters · strict label · RF / XGB / Logit) with side bar showing algorithm-level mean AUC. Headline: ensemble mean = 0.704 | Stage 2 robustness / "consistent across events" |
| 06 | `06_prob_by_facility_group.png` | Predicted probability box plot by facility group (Stage 2 Model). Group 1 (hospital/airport/power_plant) and Group 2 (fire/police) sit higher than excluded types and outside-buffer pixels. | Validation / interpretation |
| 07 | `07_irma_miami_prob_map.png` | Per-pixel probability map — Hurricane Irma, Miami | Example output #1 |
| 08 | `08_ian_fortmyers_prob_map.png` | Per-pixel probability map — Hurricane Ian, Fort Myers | Example output #2 |
| 09 | `09_hatay_earthquake_prob_map.png` | Per-pixel probability map — 2023 Türkiye earthquake (Hatay) | Example output #3 (non-hurricane / international) |
| 10 | `10_miami_generator_validation.png` | Hurricane Irma probability heatmap with **all 148 Miami-Dade generator permits overlaid** (yellow ◆ = commercial, orange ● = residential) | Ground-truth validation #1 |
| 11 | `11_dashboard_preview.png` | Screenshot of the interactive dashboard | Dashboard intro / live-demo placeholder |
| 12 | `12_equity_gap.png` | Stage 3 bar chart: ZIPs in the highest-outage tertile have only **63 % of the facility density** of the lowest-outage tertile (t = 2.56, p = 0.011) | Stage 3 main policy finding |
| 13 | `13_miami_dade_rc_bar.png` | Commercial vs Residential generator-permit detection: **83 %** of commercial permits score above the event-wide median probability vs only **14 %** of residential | Ground-truth validation #2 / "what the model can / can't see" |

## Removed from previous set (intentionally)

- `01_model_ABC_comparison.png` · `02_modelA_vs_modelB.png` · `05_extended_evaluation.png`

These compared internal feature-ablation variants (Model A / B / C) and are not needed for the public-facing narrative — Model D is the headline model and the slides should not draw attention to feature-set ablation.

## Suggested slide flow

| Slide | Figure(s) | Talking point |
|-------|-----------|---------------|
| Methods | (no fig — short text only) | Three stages of analysis: interpretive, predictive, ZIP-level |
| Stage 2 — what the model learns | 03 + 06 | NTL temporal features dominate; predictions higher near critical-facility groups |
| Stage 2 — example outputs | 07 / 08 / 09 | Pick 1–3 events; Irma + Ian + Hatay span hurricane / earthquake / domestic / international |
| Ground-truth validation | 10 + 13 | Visual overlay (10) then the bar (13) — commercial detected, residential not |
| Stage 3 — equity finding | 12 | The 63 % gap headline |
| Dashboard | 11 | Lead into live demo or recorded clip |

## Source / regeneration

```
project/script/regen_pre_figures.py
project/script/miami_dade_visualization.py     # original source for figure 10
project/script/run_modelD_loeo_25events.py     # produces loeo_modelD_25events.csv
project/script/make_modelD_loeo_heatmap.py     # renders figure 04
```

Inputs read from:
- `project/data/result/stage2/feature_importance_modelD.csv`
- `project/data/result/stage2/{Event}_prob_map_modelD.tif`
- `project/data/result/stage2/poi_cache/{Event}_poi.csv`
- `project/data/result/stage2/pixel_panel.parquet`
- `project/data/dade_test/miamidade_filtered.shp`

To regenerate everything, run:
```bash
python project/script/regen_pre_figures.py
```
