# Exploration Master Plan (V2)

- Anchor: v3r1 outputs as round0
- Sample lock: `sample_lock_flag=1`
- Validation: LOEO by event (6 folds)
- Stop rule: if two consecutive lines are marginal (<+0.02 AUC and <+0.01 survival best), stop
- This round adds six lines: cloud, mask, urban+pop, spatial, contribution, extreme-drop sensitivity
