# Modeling Issue Log

All issues here correspond to `project/modeling/output/model_issue_log.csv`.

## Update 2026-02-23 16:26:46 UTC
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [all] maria_sanjuan | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] michael_panamacity | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] earthquake_sanjuan | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] ida_neworleans | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] laura_lakecharles | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] irma_miami | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] all | insufficient_nlcd_coverage | symptom=land_use coverage <= 20% | fix=skip with_nlcd model reruns | impact=only no_nlcd model variant available | status=open

## Update 2026-02-23 16:28:39 UTC
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [all] maria_sanjuan | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] michael_panamacity | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] earthquake_sanjuan | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] ida_neworleans | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] laura_lakecharles | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] irma_miami | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] all | insufficient_nlcd_coverage | symptom=land_use coverage <= 20% | fix=skip with_nlcd model reruns | impact=only no_nlcd model variant available | status=open

## Update 2026-02-23 16:29:32 UTC
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [all] maria_sanjuan | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] michael_panamacity | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] earthquake_sanjuan | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] ida_neworleans | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] laura_lakecharles | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] irma_miami | missing_nlcd | symptom=No NLCD raster found for event | fix=keep land_use as NaN and continue | impact=NLCD-enhanced model skipped for this event | status=open
- [all] all | insufficient_nlcd_coverage | symptom=land_use coverage <= 20% | fix=skip with_nlcd model reruns | impact=only no_nlcd model variant available | status=open

## Update 2026-02-23 17:51:57 UTC
- [all] maria_sanjuan | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] michael_panamacity | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] earthquake_sanjuan | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] ida_neworleans | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] laura_lakecharles | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] irma_miami | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] all | panel_build_failed_use_cached | symptom=No valid event data available to build pixel panel. | fix=load existing all_events_pixel_panel_v1.parquet | impact=pipeline continues without raw pre/post rebuild | status=resolved
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_build_failed_use_cached | symptom=No recovery rows built. | fix=load existing recovery_daily_panel_v1.parquet | impact=cox baseline runs on cached recovery panel | status=resolved
- [all] maria_sanjuan | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] michael_panamacity | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] earthquake_sanjuan | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] ida_neworleans | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] laura_lakecharles | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] irma_miami | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_build_failed_use_cached_with_land_use | symptom=No recovery rows built. | fix=merge cached recovery panel with panel_nlcd land_use by pixel_id | impact=cox with_nlcd runs without rebuilding post-event stacks | status=resolved
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_panel_unavailable | symptom=recovery_threshold=0.8: No recovery rows built. | fix=skip this recovery-threshold scenario | impact=partial robustness table for Cox recovery thresholds | status=open
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_panel_unavailable | symptom=recovery_threshold=0.9: No recovery rows built. | fix=skip this recovery-threshold scenario | impact=partial robustness table for Cox recovery thresholds | status=open
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_panel_unavailable | symptom=recovery_threshold=0.95: No recovery rows built. | fix=skip this recovery-threshold scenario | impact=partial robustness table for Cox recovery thresholds | status=open

## Update 2026-02-23 17:53:26 UTC
- [all] maria_sanjuan | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] michael_panamacity | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] earthquake_sanjuan | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] ida_neworleans | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] laura_lakecharles | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] irma_miami | missing_tif | symptom=pre=0, post=0 | fix=skip this event for panel build | impact=event dropped from model sample | status=open
- [all] all | panel_build_failed_use_cached | symptom=No valid event data available to build pixel panel. | fix=load existing all_events_pixel_panel_v1.parquet | impact=pipeline continues without raw pre/post rebuild | status=resolved
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_build_failed_use_cached | symptom=No recovery rows built. | fix=load existing recovery_daily_panel_v1.parquet | impact=cox baseline runs on cached recovery panel | status=resolved
- [all] maria_sanjuan | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] michael_panamacity | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] earthquake_sanjuan | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] ida_neworleans | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] laura_lakecharles | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [all] irma_miami | missing_pre_tif | symptom=cannot infer target raster grid for NLCD align | fix=fallback to lon/lat sampling from panel coordinates | impact=land_use attached without event-grid reprojection | status=resolved
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_build_failed_use_cached_with_land_use | symptom=No recovery rows built. | fix=merge cached recovery panel with panel_nlcd land_use by pixel_id | impact=cox with_nlcd runs without rebuilding post-event stacks | status=resolved
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_panel_unavailable | symptom=recovery_threshold=0.8: No recovery rows built. | fix=skip this recovery-threshold scenario | impact=partial robustness table for Cox recovery thresholds | status=open
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_panel_unavailable | symptom=recovery_threshold=0.9: No recovery rows built. | fix=skip this recovery-threshold scenario | impact=partial robustness table for Cox recovery thresholds | status=open
- [Cox] maria_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] michael_panamacity | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] earthquake_sanjuan | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] ida_neworleans | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] laura_lakecharles | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] irma_miami | missing_post_tif | symptom=No post-event daily tifs | fix=skip event in cox panel | impact=reduced sample | status=open
- [Cox] all | recovery_panel_unavailable | symptom=recovery_threshold=0.95: No recovery rows built. | fix=skip this recovery-threshold scenario | impact=partial robustness table for Cox recovery thresholds | status=open

## Update 2026-02-23 23:02:08 UTC
- [all] maria_sanjuan | missing_cloud_screening | symptom=No cloud screening CSV found under project/script | fix=set missing_weather_flag=1 and continue | impact=weather controls unavailable for this event | status=monitor
- [all] michael_panamacity | missing_cloud_screening | symptom=No cloud screening CSV found under project/script | fix=set missing_weather_flag=1 and continue | impact=weather controls unavailable for this event | status=monitor
- [all] earthquake_sanjuan | missing_cloud_screening | symptom=No cloud screening CSV found under project/script | fix=set missing_weather_flag=1 and continue | impact=weather controls unavailable for this event | status=monitor
- [all] ida_neworleans | missing_cloud_screening | symptom=No cloud screening CSV found under project/script | fix=set missing_weather_flag=1 and continue | impact=weather controls unavailable for this event | status=monitor
- [all] laura_lakecharles | missing_cloud_screening | symptom=No cloud screening CSV found under project/script | fix=set missing_weather_flag=1 and continue | impact=weather controls unavailable for this event | status=monitor
- [all] irma_miami | missing_cloud_screening | symptom=No cloud screening CSV found under project/script | fix=set missing_weather_flag=1 and continue | impact=weather controls unavailable for this event | status=monitor
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor

## Update 2026-02-23 23:03:16 UTC
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions; skip random-intercept export | impact=random-effect chart unavailable for this variant | status=monitor

## Update 2026-02-24 00:16:56 UTC
- [MixedLM] all | model_fit_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=skip mixedlm for this variant | impact=mixed-effect result unavailable | status=open
- [MixedLM] all | model_fit_failed | symptom=Singular matrix | fix=skip mixedlm for this variant | impact=mixed-effect result unavailable | status=open
- [Logit] all | model_fit_failed | symptom=Singular matrix | fix=skip logit for this variant | impact=logit result unavailable | status=open
- [Cox] all | base_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=try event strata model | impact=fallback sequence triggered | status=monitor
- [Cox] all | strata_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=try time-interaction model | impact=fallback sequence triggered | status=monitor
- [Cox] all | time_interaction_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=use best available cox candidate | impact=partial PH repair | status=monitor
- [MixedLM] earthquake_sanjuan | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for MixedLM | status=monitor
- [Logit] earthquake_sanjuan | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Cox] earthquake_sanjuan | fold_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=record NaN metrics | impact=partial LOEO for Cox | status=monitor
- [MixedLM] earthquake_sanjuan | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for MixedLM | status=monitor
- [Logit] earthquake_sanjuan | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Cox] earthquake_sanjuan | fold_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=record NaN metrics | impact=partial LOEO for Cox | status=monitor
- [MixedLM] ida_neworleans | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for MixedLM | status=monitor
- [Logit] ida_neworleans | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Cox] ida_neworleans | fold_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=record NaN metrics | impact=partial LOEO for Cox | status=monitor
- [MixedLM] ida_neworleans | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for MixedLM | status=monitor
- [Logit] ida_neworleans | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Cox] ida_neworleans | fold_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=record NaN metrics | impact=partial LOEO for Cox | status=monitor
- [MixedLM] irma_miami | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for MixedLM | status=monitor
- [Logit] irma_miami | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Cox] irma_miami | fold_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=record NaN metrics | impact=partial LOEO for Cox | status=monitor
- [MixedLM] irma_miami | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for MixedLM | status=monitor
- [Logit] irma_miami | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Cox] irma_miami | fold_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=record NaN metrics | impact=partial LOEO for Cox | status=monitor
- [MixedLM] laura_lakecharles | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for MixedLM | status=monitor
- [Logit] laura_lakecharles | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Cox] laura_lakecharles | fold_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=record NaN metrics | impact=partial LOEO for Cox | status=monitor
- [MixedLM] laura_lakecharles | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for MixedLM | status=monitor
- [Logit] laura_lakecharles | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Cox] laura_lakecharles | fold_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=record NaN metrics | impact=partial LOEO for Cox | status=monitor
- [MixedLM] maria_sanjuan | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for MixedLM | status=monitor
- [Logit] maria_sanjuan | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Cox] maria_sanjuan | fold_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=record NaN metrics | impact=partial LOEO for Cox | status=monitor
- [MixedLM] maria_sanjuan | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for MixedLM | status=monitor
- [Logit] maria_sanjuan | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Cox] maria_sanjuan | fold_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=record NaN metrics | impact=partial LOEO for Cox | status=monitor
- [MixedLM] michael_panamacity | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for MixedLM | status=monitor
- [Logit] michael_panamacity | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Cox] michael_panamacity | fold_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=record NaN metrics | impact=partial LOEO for Cox | status=monitor
- [MixedLM] michael_panamacity | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for MixedLM | status=monitor
- [Logit] michael_panamacity | fold_fit_failed | symptom=Singular matrix | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Cox] michael_panamacity | fold_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=record NaN metrics | impact=partial LOEO for Cox | status=monitor

## Update 2026-02-24 00:20:09 UTC
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions | impact=random-effect table unavailable | status=monitor
- [MixedLM] all | formula_fallback | symptom=failed full formula for full_locked | fix=use reduced mixed formula: delta_ntl ~ in_buffer * pre_mean_ntl + C(land_use_group) | impact=mixedlm kept with reduced controls | status=resolved
- [Logit] all | formula_fallback | symptom=failed full formula for full_locked | fix=use reduced formula: is_damaged ~ in_buffer * pre_mean_ntl + C(event_id) + C(land_use_group) | impact=logit kept with reduced controls | status=resolved
- [Cox] all | base_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=try event strata model | impact=fallback sequence triggered | status=monitor
- [Cox] all | strata_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=try time-interaction model | impact=fallback sequence triggered | status=monitor
- [Cox] all | time_interaction_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=use best available cox candidate | impact=partial PH repair | status=monitor
- [Logit] earthquake_sanjuan | fold_fit_failed | symptom=predict requires that you use a DataFrame when predicting from a model
that was created using the formula api.

The original error message returned by patsy is:
Error converting data to categorical: observation with value 'earthquake_sanjuan' does not match any of the expected levels (expected: ['ida_neworleans', 'irma_miami', ..., 'maria_sanjuan', 'michael_panamacity'])
    is_damaged ~ in_buffer * pre_mean_ntl + C(event_id) + C(land_use_group)
                                            ^^^^^^^^^^^ | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Logit] ida_neworleans | fold_fit_failed | symptom=predict requires that you use a DataFrame when predicting from a model
that was created using the formula api.

The original error message returned by patsy is:
Error converting data to categorical: observation with value 'ida_neworleans' does not match any of the expected levels (expected: ['earthquake_sanjuan', 'irma_miami', ..., 'maria_sanjuan', 'michael_panamacity'])
    is_damaged ~ in_buffer * pre_mean_ntl + C(event_id) + C(land_use_group)
                                            ^^^^^^^^^^^ | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Logit] irma_miami | fold_fit_failed | symptom=predict requires that you use a DataFrame when predicting from a model
that was created using the formula api.

The original error message returned by patsy is:
Error converting data to categorical: observation with value 'irma_miami' does not match any of the expected levels (expected: ['earthquake_sanjuan', 'ida_neworleans', ..., 'maria_sanjuan', 'michael_panamacity'])
    is_damaged ~ in_buffer * pre_mean_ntl + C(event_id) + C(land_use_group)
                                            ^^^^^^^^^^^ | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Logit] laura_lakecharles | fold_fit_failed | symptom=predict requires that you use a DataFrame when predicting from a model
that was created using the formula api.

The original error message returned by patsy is:
Error converting data to categorical: observation with value 'laura_lakecharles' does not match any of the expected levels (expected: ['earthquake_sanjuan', 'ida_neworleans', ..., 'maria_sanjuan', 'michael_panamacity'])
    is_damaged ~ in_buffer * pre_mean_ntl + C(event_id) + C(land_use_group)
                                            ^^^^^^^^^^^ | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Logit] maria_sanjuan | fold_fit_failed | symptom=predict requires that you use a DataFrame when predicting from a model
that was created using the formula api.

The original error message returned by patsy is:
Error converting data to categorical: observation with value 'maria_sanjuan' does not match any of the expected levels (expected: ['earthquake_sanjuan', 'ida_neworleans', ..., 'laura_lakecharles', 'michael_panamacity'])
    is_damaged ~ in_buffer * pre_mean_ntl + C(event_id) + C(land_use_group)
                                            ^^^^^^^^^^^ | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor
- [Logit] michael_panamacity | fold_fit_failed | symptom=predict requires that you use a DataFrame when predicting from a model
that was created using the formula api.

The original error message returned by patsy is:
Error converting data to categorical: observation with value 'michael_panamacity' does not match any of the expected levels (expected: ['earthquake_sanjuan', 'ida_neworleans', ..., 'laura_lakecharles', 'maria_sanjuan'])
    is_damaged ~ in_buffer * pre_mean_ntl + C(event_id) + C(land_use_group)
                                            ^^^^^^^^^^^ | fix=record NaN metrics | impact=partial LOEO for Logit | status=monitor

## Update 2026-02-24 00:21:22 UTC
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions | impact=random-effect table unavailable | status=monitor
- [MixedLM] all | formula_fallback | symptom=failed full formula for full_locked | fix=use reduced mixed formula: delta_ntl ~ in_buffer * pre_mean_ntl + C(land_use_group) | impact=mixedlm kept with reduced controls | status=resolved
- [Logit] all | formula_fallback | symptom=failed full formula for full_locked | fix=use reduced formula: is_damaged ~ in_buffer * pre_mean_ntl + C(event_id) + C(land_use_group) | impact=logit kept with reduced controls | status=resolved
- [Cox] all | base_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=try event strata model | impact=fallback sequence triggered | status=monitor
- [Cox] all | strata_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=try time-interaction model | impact=fallback sequence triggered | status=monitor
- [Cox] all | time_interaction_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=use best available cox candidate | impact=partial PH repair | status=monitor
- [Cox] all | formula_fallback | symptom=full Cox failed for full_locked | fix=fallback to nlcd-level cox design | impact=full_locked uses reduced Cox controls | status=resolved

## Update 2026-02-24 00:48:24 UTC
- No critical issue observed in this run.

## Update 2026-02-27 04:00:23 UTC
- No critical issue observed in this run.

## Update 2026-02-27 04:03:33 UTC
- No critical issue observed in this run.

## Update 2026-02-27 04:09:51 UTC
- No critical issue observed in this run.

## Update 2026-02-27 04:12:39 UTC
- No critical issue observed in this run.

## Update 2026-02-27 13:55:14 UTC
- No critical issue observed in this run.

## Update 2026-02-27 14:23:10 UTC
- No critical issue observed in this run.

## Update 2026-02-27 14:29:09 UTC
- No critical issue observed in this run.

## Update 2026-02-27 23:54:32 UTC
- No critical issue observed in this run.

## Update 2026-02-28 00:25:20 UTC
- No critical issue observed in this run.

## Update 2026-02-28 00:26:29 UTC
- No critical issue observed in this run.

## Update 2026-02-28 00:27:37 UTC
- No critical issue observed in this run.

## Update 2026-03-10 01:36:21 UTC
- No critical issue observed in this run.

## Update 2026-03-10 01:36:48 UTC
- No critical issue observed in this run.

## Update 2026-03-10 01:37:10 UTC
- No critical issue observed in this run.

## Update 2026-03-10 01:40:01 UTC
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions | impact=random-effect table unavailable | status=monitor
- [MixedLM] all | formula_fallback | symptom=failed full formula for full_locked | fix=use reduced mixed formula: delta_ntl ~ in_buffer * pre_mean_ntl + C(land_use_group) | impact=mixedlm kept with reduced controls | status=resolved
- [Logit] all | formula_fallback | symptom=failed full formula for full_locked | fix=use reduced formula: is_damaged ~ in_buffer * pre_mean_ntl + C(event_id) + C(land_use_group) | impact=logit kept with reduced controls | status=resolved
- [Cox] all | base_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=try event strata model | impact=fallback sequence triggered | status=monitor
- [Cox] all | strata_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=try time-interaction model | impact=fallback sequence triggered | status=monitor
- [Cox] all | time_interaction_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=use best available cox candidate | impact=partial PH repair | status=monitor
- [Cox] all | formula_fallback | symptom=full Cox failed for full_locked | fix=fallback to nlcd-level cox design | impact=full_locked uses reduced Cox controls | status=resolved

## Update 2026-03-10 01:40:23 UTC
- [MixedLM] all | random_effect_extraction_failed | symptom=Cannot predict random effects from singular covariance structure. | fix=keep fixed effects and predictions | impact=random-effect table unavailable | status=monitor
- [MixedLM] all | formula_fallback | symptom=failed full formula for full_locked | fix=use reduced mixed formula: delta_ntl ~ in_buffer * pre_mean_ntl + C(land_use_group) | impact=mixedlm kept with reduced controls | status=resolved
- [Logit] all | formula_fallback | symptom=failed full formula for full_locked | fix=use reduced formula: is_damaged ~ in_buffer * pre_mean_ntl + C(event_id) + C(land_use_group) | impact=logit kept with reduced controls | status=resolved
- [Cox] all | base_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=try event strata model | impact=fallback sequence triggered | status=monitor
- [Cox] all | strata_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=try time-interaction model | impact=fallback sequence triggered | status=monitor
- [Cox] all | time_interaction_fit_failed | symptom=delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model | fix=use best available cox candidate | impact=partial PH repair | status=monitor
- [Cox] all | formula_fallback | symptom=full Cox failed for full_locked | fix=fallback to nlcd-level cox design | impact=full_locked uses reduced Cox controls | status=resolved

## Update 2026-03-10 01:42:07 UTC
- No critical issue observed in this run.

## Update 2026-03-10 01:43:48 UTC
- No critical issue observed in this run.

## Update 2026-03-10 01:51:57 UTC
- No critical issue observed in this run.

## Update 2026-03-10 01:52:55 UTC
- No critical issue observed in this run.

## Update 2026-03-10 01:53:28 UTC
- No critical issue observed in this run.

## Update 2026-03-10 14:35:32 UTC
- No critical issue observed in this run.

## Update 2026-03-10 14:48:38 UTC
- No critical issue observed in this run.

## Update 2026-03-10 14:49:16 UTC
- No critical issue observed in this run.

## Update 2026-03-10 14:50:02 UTC
- No critical issue observed in this run.

## Update 2026-03-10 14:50:06 UTC
- No critical issue observed in this run.

## Update 2026-03-10 14:51:15 UTC
- No critical issue observed in this run.

## Update 2026-03-10 14:52:23 UTC
- No critical issue observed in this run.

## Update 2026-03-10 14:52:28 UTC
- No critical issue observed in this run.
