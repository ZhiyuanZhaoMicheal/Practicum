# NightLight · Detecting Backup Power Deployment from Satellite Nighttime Light

> Capstone project handoff documentation
> Authors: Zhiyuan Zhao · Qiushi Yu (Spring 2026)
> Purpose: Identify post-disaster backup-generator deployment patterns from VIIRS nighttime-light imagery + critical-facility POIs

🌏 [中文版 README](README.zh.md)

## One-line summary

Across 25 disaster events (hurricanes / earthquakes / winter storms / etc.), we predict at 500 m pixel resolution whether each pixel shows backup-power activity (LOEO AUC = 0.704), then regress ZIP-level facility density against outage severity.

Full methodology and results: [MODELS.md](MODELS.md).
Pipeline overview: [PIPELINE.md](PIPELINE.md) (note: some file references in PIPELINE.md use older names — treat MODELS.md as the authoritative narrative).

---

## Authorship

- **Stage 0 (NTL download)**, **Stage 1 (EDA)**, **Stage 1.5 (interpretive modeling — OLS / MixedLM / Logit / Cox PH)** under `project/modeling/` were authored by **Qiushi Yu** and are merged in as-is. See `project/modeling/README.md` for his entry point.
- **Stage 2 (pixel-level prediction)**, **Stage 3 (ZIP-level regression)**, the dashboard, and figure / report assets were authored by **Zhiyuan Zhao**.

**Before extending `project/modeling/`** — the legacy result writers (`legacy/02_fit_ols_mixed.py`, `legacy/03_fit_logit.py`, `legacy/05_fit_cox.py`) append to result CSVs across re-runs rather than overwriting (`pd.concat([old, new]).to_csv(...)`). Re-running silently duplicates rows. Switch to overwrite or version-stamped output before iterating.

---

## Repository layout

```
.
├── MODELS.md                          ← Methodology + results (authoritative)
├── PIPELINE.md                        ← Pipeline overview
├── README.md                          ← This file (English)
├── README.zh.md                       ← Chinese version
│
├── project/
│   ├── modeling/                      ← Stage 1.5 · OLS / MixedLM / Logit / Cox PH (Qiushi Yu)
│   │   ├── pipeline_lib.py                Shared library — model fitting + IO + figures
│   │   ├── pipelines/                     Current entry points (01 in-sample / 02 cross-event / 03 exploration)
│   │   ├── legacy/                        Original 4-model fitters (02_fit_ols_mixed / 03_fit_logit / 05_fit_cox …)
│   │   ├── output/                        Result CSVs (ols_results, mixedlm_results, logit_results, cox_results)
│   │   ├── config/                        JSON configs (events_6/10.json, model_defaults.json …)
│   │   ├── experimental/, support/, pixel_data/
│   │   └── README.md                      His pipeline doc
│   │
│   ├── modeling_tracking/             ← Modeling progress / issue logs (Qiushi Yu)
│   │
│   ├── script/                        ← All analysis scripts (cleaned — current versions only)
│   │   ├── multi_event_ntl_download_v2.ipynb    Stage 0 · Download NTL for 25 events via GEE
│   │   ├── multi_event_eda.ipynb                Stage 1 · EDA / buffer / resilience curves
│   │   ├── stage2_25events.ipynb                Stage 2 · Train pixel-level model (main)
│   │   ├── stage2_15events_modelD.ipynb         Stage 2 · Model D held-out validation
│   │   ├── regen_modelD_prob_maps.py            Stage 2 · Regenerate 25 prob_map_modelD.tif
│   │   ├── run_modelD_loeo_25events.py          Stage 2 · LOEO cross-validation (25 folds)
│   │   ├── make_modelD_loeo_heatmap.py          Stage 2 · Render LOEO heatmap
│   │   ├── miami_dade_visualization.py          Stage 2 · Miami-Dade generator-permit overlay
│   │   ├── miami_dade_groundtruth_validation.py Stage 2 · Residential vs commercial split
│   │   ├── stage3_events.py                     Stage 3 · Event definitions and windows
│   │   ├── stage3_ntl_download.py               Stage 3 · ZIP-level NTL time series download
│   │   ├── stage3_osm_download.py               Stage 3 · OSM POI fetch (includes km² area)
│   │   ├── stage3_export_all.py                 Stage 3 · Batch export regression inputs
│   │   ├── stage3_zipcode_analysis_modelD.py    Stage 3 · Build zipcode_panel_modelD.parquet
│   │   ├── stage3_modelD_full_regressions.py    Stage 3 · Main regressions (Model 1–9)
│   │   ├── stage3_modelD_extra_regressions.py   Stage 3 · Extra regressions + reverse fairness
│   │   ├── regen_pre_figures.py                 Figures · Regenerate docs/pre_figures/
│   │   ├── year_end_show.py                     Figures · Year End Show panel (4 hero plots)
│   │   └── data/                                Cloud-screening CSVs per event
│   │
│   ├── data/
│   │   ├── raw/                       ← Raw data (1.2 GB — transfer separately)
│   │   │   ├── Outage_Dataset_R1/     EAGLE-I outage records 2014–2023
│   │   │   ├── POI/                   OSM critical-facility POIs
│   │   │   ├── counties/              Census county shapefile
│   │   │   ├── zcta520/               Census ZCTA 2020 shapefile
│   │   │   ├── acs_zcta_2022.csv      ACS population / income
│   │   │   ├── ibtracs_NA.csv         IBTrACS hurricane tracks
│   │   │   ├── generator.csv          Generic generator permits
│   │   │   ├── generator_houston_dallas.csv  TX subset
│   │   │   ├── stage3_event_configs.json     Stage 3 event configs
│   │   │   └── POI/                          Region-specific infrastructure extractors (TX + PR w/ HIFLD retry)
│   │   │
│   │   ├── processed/                 Stage 0 NTL TIFs · 25 events × pre/post
│   │   │   └── {Event}-VNP46A2-{pre,post}/*.tif
│   │   │
│   │   ├── dade_test/                 Miami-Dade generator-permit shapefile
│   │   │
│   │   └── result/
│   │       ├── stage2/                ← Current results (all Model D)
│   │       │   ├── pixel_panel.parquet                Stage 2 pixel table
│   │       │   ├── building_coverage_panel.parquet
│   │       │   ├── rf_modelD.pkl / xgb_modelD.pkl     Trained models
│   │       │   ├── feature_importance_modelD.csv
│   │       │   ├── loeo_modelD_25events.csv           LOEO results
│   │       │   ├── {Event}_prob_map_modelD.tif × 25   Per-event probability maps
│   │       │   ├── poi_cache/                         OSM cache
│   │       │   ├── miami_dade_pointwise_probs.csv     Miami-Dade validation
│   │       │   ├── miami_dade_validation.json
│   │       │   └── precision_recall_results.csv
│   │       │
│   │       └── stage3/
│   │           ├── zipcode_panel_modelD.parquet       Stage 3 ZIP panel
│   │           ├── regression_results_modelD_full.json
│   │           ├── regression_results_modelD_extra.json
│   │           └── poi_cache/
│   │
│   ├── nightlight-dashboard/          ← Vue 3 + MapLibre interactive dashboard
│   │   ├── src/                       Vue source
│   │   ├── public/data/               Exported GeoJSON / TIF
│   │   ├── export_to_dashboard_modelD.py  Stage 2 prob maps → dashboard data
│   │   ├── export_cloud_stats.py
│   │   ├── export_ntl_frames.py
│   │   ├── enrich_poi_names.py
│   │   └── package.json
│   │
│   └── export_geojson_facilities.py   POI → GeoJSON
│
├── docs/
│   ├── pre_figures/                   Presentation figures (10 Model D-era plots)
│   ├── year_end_show/                 Year End Show exhibit (4 plots + intro + QR)
│   ├── BackupGen_Zhiyuan_Qiushi/      Final PPT + report
│   ├── reference/                     Reference PDFs
│   ├── Capstone_Speech_10min.docx     Speaker script (10-min version)
│   ├── Capstone_Speech_with_Models.docx
│   ├── NTL_speaker_script.docx
│   ├── Stage3_汇报完整版.docx
│   ├── Models ZH v6.docx
│   ├── dashboardvideo.mov             Dashboard demo screen recording
│   ├── mappreview.png
│   └── pre_prompt.md
│
└── .gitignore
```

---

## End-to-end pipeline

> Assumes raw data is already in `project/data/raw/` and `project/data/processed/`.

```bash
# Stage 2 · Retrain Model D + regenerate 25 prob maps
python project/script/regen_modelD_prob_maps.py

# Stage 2 · LOEO cross-validation (25 folds, ~30 min)
python project/script/run_modelD_loeo_25events.py
python project/script/make_modelD_loeo_heatmap.py

# Stage 3 · Build ZIP panel + run Model 1–9
python project/script/stage3_zipcode_analysis_modelD.py
python project/script/stage3_modelD_full_regressions.py
python project/script/stage3_modelD_extra_regressions.py

# Regenerate the full figure set under docs/pre_figures/
python project/script/regen_pre_figures.py

# Dashboard
cd project/nightlight-dashboard
npm install
npm run dev          # local dev
python export_to_dashboard_modelD.py   # regenerate dashboard data
npm run build        # production build
```

---

## Known pitfalls (we hit these so you don't have to)

1. **Earthquake_SanJuan bbox fallback bug**
   `export_to_dashboard_modelD.py` resolves NTL pre/post directories via substring match. `Earthquake_SanJuan` would accidentally match `Earthquake_Hatay-VNP46A2-pre/` (a Turkey directory, longitude ~36°E), causing all 215 Puerto Rico POIs to be filtered out by the bbox check. **The fix is the `EXPLICIT_PRE_DIR` dict at the top of `export_to_dashboard_modelD.py`** — add new events there explicitly.

2. **Model D probabilities cluster in 0.5–0.7**
   Model D drops all spatial-proximity features, so AUC is 0.704 and predicted probabilities concentrate in the middle. The dashboard's MapView uses **per-event quantile color stops** (min / p10 / p50 / midpoint / p90 / max) instead of a fixed 0–1 ramp — otherwise the whole map looks uniformly colored. If you change the coloring logic, keep the strict-monotonicity guard in the code.

3. **CRS must be EPSG:5070 (US Equal Area) in Stage 2/3**
   Earlier iterations computed km² area in EPSG:4326 and got everything wrong. After fetching OSM POIs in Stage 3, always reproject to 5070 before computing density.

4. **M7 raw β looks alarming (+123.8) — don't panic**
   This is the regression of `outage_severity` on `mean_prob`. `outage_severity` SD ≈ 274 (log customer-hours) and `mean_prob` SD ≈ 0.176, so the **standardized β ≈ 0.08** — a small-to-moderate effect. When presenting, quote standardized β, not raw.

5. **The equity finding is descriptive, not normative**
   The "high-outage ZIPs have 63 % of the facility density of low-outage ZIPs" headline is a **cross-event** cross-sectional co-occurrence pattern. Within events (M5/M6/M7 with event fixed effects), facility density positively predicts both mean probability and severity. These two findings are not contradictory — it's Simpson's paradox. A normative reading ("disaster risk is unfairly distributed") requires stronger causal assumptions; a direct alternative explanation is "places that get hit constantly never grew into cities."

6. **R is a time series, not a scalar**
   Resilience Ratio `R(t) = NTL_post(t) / BAU` is computed per day. Slides sometimes say "median BAU" while the code uses "mean BAU" — align this before presenting.

---

## Data sources

| Dataset | Source | License |
|---|---|---|
| VIIRS NTL VNP46A2 | NASA via Google Earth Engine | Public |
| EAGLE-I outages | DOE Oak Ridge National Lab | Public |
| OSM POI | OpenStreetMap via Overpass API | ODbL |
| Census ACS / ZCTA | US Census Bureau | Public |
| IBTrACS | NOAA | Public |
| Miami-Dade generator permits | Miami-Dade Open Data Portal | Public |

---

## External downloads (not in this repo)

To keep the repo lightweight, these large files are not version-controlled. Drop them into the paths below before running the pipeline:

| File | Path to drop into | Source |
|---|---|---|
| `tl_2020_us_zcta520.{shp,shx,dbf,prj,cpg}` (~820 MB) | `project/data/raw/zcta520/` | [Census TIGER 2020 ZCTA](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layers=ZCTA520) |
| `tl_2020_us_county.{shp,shx,dbf,prj,cpg}` (~130 MB) | `project/data/raw/counties/` | [Census TIGER 2020 Counties](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layers=COUNTY) |
| `ibtracs_NA.csv` (~57 MB) | `project/data/raw/` | [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) |
| Final presentation deck and demo video | n/a — ask the authors | (see contact below) |

---

## Contact

- Zhiyuan Zhao · michaelzhao576@gmail.com
- Repository / dashboard link: [docs/Project Website Link.txt](docs/Project%20Website%20Link.txt)

For most questions, the limitations and FAQ-style discussion in sections 8–11 of [MODELS.md](MODELS.md) covers it. If you're stuck, those sections are the right place to start.
