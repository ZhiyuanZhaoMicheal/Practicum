# Modeling Pipelines

Use `project/modeling/README.md` as the canonical entrypoint for the current modeling workflow, rerun commands, artifact map, and result interpretation.

This `pipelines/` directory only lists the active executable scripts:

1. `01_in_sample_pipeline.py`
   - Baseline build, feature-upgrade, strict-v2, reports, figures.
2. `02_cross_event_pipeline.py`
   - Cross-event V3 build and stabilization.
3. `03_exploration_pipeline.py`
   - Exploration V2 bundles, appendix-only quality/hazard analyses, and event-expansion utilities.

Compatibility shims remain in `project/modeling/` for `run_pipeline.py` and `15-19`.

Archived pre-fix snapshots live under `project/modeling/legacy/archive_model_fix_20260309/`.
