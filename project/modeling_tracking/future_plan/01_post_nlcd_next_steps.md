# Post-NLCD Next Steps

## Priority-1 (Paper-grade robustness)
- Add spatial dependence checks (Moran's I) and cluster-robust SE by spatial blocks.
- Add leave-one-event-out (LOEO) validation for generalization.
- Add side-by-side table for facility-type filtering strategy:
  - full types
  - excluding `clinic,townhall,public_building`

## Priority-2 (Data quality closure)
- Replace Puerto Rico NLCD source with year-consistent land-cover layer (target: 2016) when data access is available.
- Rebuild recovery panel from raw post-event daily tif stacks (remove cached fallback dependency).

## Priority-3 (Model refinement)
- MixedLM: test simplified random structure / GEE comparison.
- Logit: test regularized logistic regression under land-use controls to mitigate separation risk.
- Cox: evaluate alternative recovery thresholds after raw stack recovery.

## Deliverable Update Rule
- Keep `project/modeling/output/model_summary_for_report.csv` as single source of truth.
- Any report number change must be regenerated from pipeline outputs, not manual edits.
