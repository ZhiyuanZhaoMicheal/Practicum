# Legacy Modeling Scripts

Active execution now lives in `project/modeling/pipelines/01_in_sample_pipeline.py`, `project/modeling/pipelines/02_cross_event_pipeline.py`, and `project/modeling/pipelines/03_exploration_pipeline.py`.

## Still-kept legacy stage scripts
- `01_build_pixel_panel.py` -> `pipelines/01_in_sample_pipeline.py build-baseline`
- `02_fit_ols_mixed.py` -> `pipelines/01_in_sample_pipeline.py fit-core-models`
- `03_fit_logit.py` -> `pipelines/01_in_sample_pipeline.py fit-core-models`
- `04_build_recovery_panel.py` -> `pipelines/01_in_sample_pipeline.py fit-core-models`
- `05_fit_cox.py` -> `pipelines/01_in_sample_pipeline.py fit-core-models`
- `06_robustness.py` -> covered by in-sample pipelines
- `07_generate_figures.py` -> `pipelines/01_in_sample_pipeline.py figures`
- `08_generate_reports.py` -> `pipelines/01_in_sample_pipeline.py reports`
- `10_attach_nlcd.py` -> `pipelines/01_in_sample_pipeline.py attach-nlcd`
- `14_generate_feature_upgrade_figures.py` -> covered by in-sample pipelines

## Archived pre-merge implementations
The former active implementations for `15-19` were used to bootstrap the 3 consolidated pipeline files and are retained only for reference under `project/modeling/legacy/archive_premerge/`.

## Mapping for archived files
- `archive_premerge/15_feature_upgrade_pipeline.py` -> integrated into `pipelines/01_in_sample_pipeline.py feature-upgrade`
- `archive_premerge/16_strict_v2_pipeline.py` -> integrated into `pipelines/01_in_sample_pipeline.py strict-v2`
- `archive_premerge/17_cross_event_v3_pipeline.py` -> integrated into `pipelines/02_cross_event_pipeline.py build-v3`
- `archive_premerge/18_cross_event_v3_stabilization.py` -> integrated into `pipelines/02_cross_event_pipeline.py stabilize-v3`
- `archive_premerge/19_exploration_v2_pipeline.py` -> integrated into `pipelines/03_exploration_pipeline.py run-v2`
