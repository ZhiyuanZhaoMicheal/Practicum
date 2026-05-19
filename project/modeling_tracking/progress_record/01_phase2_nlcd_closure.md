# Phase-2 NLCD Closure Log

## Run Window
- Date: 2026-02-23
- Pipeline entrypoint: `project/modeling/run_pipeline.py`

## What Was Done
- Added NLCD files under `project/data/nlcd/` and fixed event coverage extraction.
- Implemented fallback in `attach_nlcd`: when event pre-tif is missing, sample NLCD by `lon/lat` from panel.
- Implemented fallback in `run_pipeline`: if raw-event rebuild fails, load cached
  `all_events_pixel_panel_v1.parquet` and `recovery_daily_panel_v1.parquet`.
- Re-ran full pipeline and regenerated outputs, figures, and reports.
- Upgraded reports to include `no_nlcd` vs `with_nlcd` comparison.

## Validation Snapshot
- NLCD coverage (`project/modeling/output/nlcd_coverage.csv`):
  - min coverage = 0.9973
  - all six events > 0.8 (and > 0.9)
- with_nlcd artifacts generated:
  - OLS/MixedLM/Logit/Cox result tables
  - ROC/calibration/KM/PH diagnostics
  - Updated model summary + four model reports + index report

## Notes
- Current repo does not contain six-event raw pre/post tif stacks, so rebuild-from-raw stages are skipped via fallback logic.
- Cox recovery-threshold robustness scenarios (80/90/95) are partially unavailable without raw post-event stacks.
