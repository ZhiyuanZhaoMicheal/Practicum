# Modeling README

This is the single active modeling README for the project.

Older root-level summaries were archived on 2026-03-10 under:

- `project/modeling/legacy/archive_readmes_20260310/`

## Project Goal

The project uses Black Marble nightlight data to study whether pixels near critical infrastructure behave as if backup power is present after disasters.

There are now two explicit modeling goals:

1. Explanatory modeling
   - ask whether buffer pixels are systematically more resilient than comparable non-buffer pixels
2. Cross-event transport
   - ask whether the model can generalize to an unseen event rather than only fit the observed events

Those two goals are related but not interchangeable. The strongest explanatory model is not automatically the strongest predictive transport model.

## Active Entry Points

- `project/modeling/pipelines/01_in_sample_pipeline.py`
  - baseline build, NLCD attach, feature upgrade, strict-v2
- `project/modeling/pipelines/02_cross_event_pipeline.py`
  - cross-event V3 build and stabilization
- `project/modeling/pipelines/03_exploration_pipeline.py`
  - exploration bundles, quality-adjusted appendix lines, hazard appendix, BUG experiments, readiness reruns, and BUG2 pilot scaffolding

Use the modeling environment:

```bash
source .venv_modeling/bin/activate
```

## Data And Design

The core panel is pixel-level and centered on:

- `pixel_id`
- `event_id`
- `lon`, `lat`
- `pre_mean_ntl`
- `post_mean_ntl`
- `delta_ntl`
- `in_buffer`
- `distance_to_nearest`
- `n_facilities_in_buffer`
- `facility_type_std`
- `is_damaged`

Important later additions include:

- `land_use_group`
- OSM local context features
- cloud and observation-quality proxies
- local land-use shares
- event profile features
- repaired international population and urbanization covariates

The most important engineering guardrail is now the locked cohort:

- `sample_lock_flag` is treated as a hard gate
- fold-level fill and scaling are fit on the training fold only
- held-out events do not get to redefine preprocessing

This means later model comparisons are much more trustworthy than earlier versions that could drift in sample definition.

## Modeling Timeline

### 1. Baseline 6-event stage

The first stage established a coherent six-event baseline and asked the simplest question:

> does a buffer signal exist at all?

Main result:

- buffer pixels showed an early resilience signal
- `MixedLM` quickly became the strongest explanatory line
- the first transport baseline looked plausible before event heterogeneity was stress-tested

### 2. Land-use and feature-upgrade stage

This stage addressed three main risks:

- land-use confounding
- missing local infrastructure context
- silent sample drift across specifications

Key additions:

- NLCD land-use controls
- OSM local context features
- cloud-validity proxies
- sample locking

This stage improved interpretability and led to the later strict mainline.

### 3. Strict-v2 stage

`strict-v2` is the cleanest explanatory line, not the most ambitious predictive one.

Its role is:

> holding the main controls fixed, does the buffer signal remain stable?

Current strict full-panel reading:

- OLS `coef_in_buffer = 0.0269`, `p = 0.0544`
- MixedLM `coef_in_buffer = 0.0269`, `p = 0.00943`
- Logit `odds_ratio_in_buffer = 0.7503`, `AUC = 0.7299`
- Cox `hazard_ratio_in_buffer = 1.3319`, `p = 6.15e-15`

Interpretation:

- the explanatory signal is stable
- `strict-v2`, especially `MixedLM`, remains the main explanatory anchor

### 4. Cross-event V3 and stabilization

The transport line asks a harder question:

> if one event is held out entirely, can the model still rank damage or recovery meaningfully?

Direct event expansion exposed real generalization failure rather than fixing it.

Current V3 and stabilization reading:

- V3 Logit `AUC = 0.4897`
- V3 AFT `c_index = 0.5354`
- stabilized final Logit `AUC = 0.4814`
- stabilized final survival-best `c_index = 0.5213`

Interpretation:

- the six-event setup was learnable
- direct expansion to ten events revealed real domain shift
- cross-event transport is still weaker than the explanatory line

### 5. Quality-adjusted and hazard appendix lines

These lines remain useful, but they are not headline evidence by default.

Quality-adjusted appendix:

- transport Logit `AUC = 0.4973`
- spatial-block Logit `AUC = 0.5949`
- facility-matched Logit `odds_ratio_in_buffer = 0.7308`, `p = 0.00741`

Hazard / exposure appendix:

- `HZ1` Logit `AUC = 0.6025`
- `HZ1` Logit `Brier = 0.4424`
- `HZ1` Cox `c_index = 0.5341`

Interpretation:

- hazard and exposure features clearly improve damage ranking
- but this line is still treated as appendix support rather than the main ex-ante claim

### 6. Event readiness and selection

One major lesson from event expansion is that more events do not automatically improve the main model.

The project now uses readiness scoring instead of assuming every event belongs in the same mainline pool.

Current readiness-based recommendation:

Mainline candidates:

- `ian_charlotteharbor`
- `earthquake_sanjuan`
- `ida_neworleans`
- `irma_miami`
- `laura_lakecharles`

Sensitivity only:

- `ian_fortmyers`

Repair first:

- `dorian_freeport`
- `earthquake_hatay`
- `maria_sanjuan`
- `michael_panamacity`

### 7. Readiness-filtered HZ1 rerun

We tested whether the hazard/exposure line should move from the full event pool to the cleaner readiness subset.

Result:

- full-event `HZ1` Logit `AUC = 0.6025`
- readiness-filtered `HZ1_READY` Logit `AUC = 0.5630`
- full-event `HZ1` Logit `Brier = 0.4424`
- readiness-filtered `HZ1_READY` Logit `Brier = 0.3554`

Interpretation:

- the cleaner subset improves calibration
- but it loses ranking power relative to the full-event `HZ1`
- therefore `HZ1_READY` should be treated as a robustness subset, not as the new predictive anchor

### 8. BUG1 proxy-refinement test

We added a parallel `BUG-aware` family to test whether a smarter generator proxy could improve cross-event damage transport without touching the mainline.

The first version only used prior-weighted facility-type proxies, not official generator inventory.

Main result:

- `BUG0` Logit `AUC = 0.4973`
- `BUG1A` Logit `AUC = 0.4965`
- `BUG1A` vs `BUG0` AUC delta = `-0.0008`
- `BUG1A` Logit Brier improved slightly from `0.2409` to `0.2392`
- replacing the legacy spatial context with BUG-prior structure performed worse

Interpretation:

- prior-weighted BUG proxy features did not improve cross-event transport in a meaningful way
- this was a useful negative result
- the project should stop tuning BUG priors and stop treating smarter POI proxies as a likely breakthrough

BUG1 is now a proxy-refinement test, not a candidate mainline.

### 9. BUG2 Puerto Rico pilot

The next BUG step is no longer proxy construction. It is mechanism validation using official inventory or permit data.

The first pilot is Puerto Rico:

- pilot events: `earthquake_sanjuan`, `maria_sanjuan`
- current status: `awaiting_inventory`

What has already been implemented:

- canonical schema definition
- canonical template file
- QA gate
- acquisition backlog
- official feature-attachment interface
- local pilot report output

Current reading:

- the pilot line is ready
- but it cannot move into model comparison until a canonical Puerto Rico inventory file is added

## Current Main Conclusions

If the goal is explanation:

- the project is already in a good state
- `strict-v2`, especially `MixedLM`, is the strongest explanatory line
- matched Logit and quality-controlled appendix analyses support the same direction

If the goal is cross-event prediction:

- the project is only partially successful
- the strongest predictive ranking result still comes from the full-event hazard/exposure appendix line
- the readiness-filtered rerun is cleaner but weaker in ranking
- BUG1 proxy refinement failed to add transport value
- BUG2 should be treated as a mechanism-validation pilot, not as a new main modeling branch

## Active Commands

Mainline and appendix reruns:

```bash
python project/modeling/pipelines/01_in_sample_pipeline.py build-baseline
python project/modeling/pipelines/01_in_sample_pipeline.py attach-nlcd
python project/modeling/pipelines/01_in_sample_pipeline.py feature-upgrade
python project/modeling/pipelines/01_in_sample_pipeline.py strict-v2
python project/modeling/pipelines/02_cross_event_pipeline.py build-v3
python project/modeling/pipelines/02_cross_event_pipeline.py stabilize-v3
python project/modeling/pipelines/03_exploration_pipeline.py run-v2
python project/modeling/pipelines/03_exploration_pipeline.py quality-matched-v1
python project/modeling/pipelines/03_exploration_pipeline.py hazard-mainline-v1
python project/modeling/pipelines/03_exploration_pipeline.py hazard-readiness-v1
python project/modeling/pipelines/03_exploration_pipeline.py bug-transport-v1
python project/modeling/pipelines/03_exploration_pipeline.py bug2-pr-pilot-v1
```

## Key Artifacts

Headline / active reference artifacts:

- `project/modeling/output/model_summary_feature_upgrade_v2_strict.csv`
- `project/modeling/output/logo_aggregate_metrics_v2_strict.csv`
- `project/modeling/output/model_summary_cross_event_v3.csv`
- `project/modeling/output/cross_event_round_comparison_v3x.csv`
- `project/modeling/output/cross_event_stop_decision_v3x.json`
- `project/modeling/output/hazard_transport_aggregate_metrics_v1.csv`
- `project/modeling/output/hazard_transport_readiness_aggregate_metrics_v1.csv`
- `project/modeling/output/bug_transport_aggregate_metrics_v1.csv`
- `project/modeling/output/bug2_pr_pilot_qa_v1.csv`

Supporting reports:

- `project/modeling_report/12_hazard_exposure_transport_report.md`
- `project/modeling_report/hazard_transport_readiness_report_v1.md`
- `project/modeling_report/bug_transport_report.md`
- `project/modeling_report/bug2_pr_pilot_report.md`

## Archive

Historical pre-fix code and docs are stored in:

- `project/modeling/legacy/archive_model_fix_20260309/`
- `project/modeling/legacy/archive_premerge/`
- `project/modeling/legacy/archive_readmes_20260310/`

The canonical report index remains:

- `project/modeling_report/index.md`
