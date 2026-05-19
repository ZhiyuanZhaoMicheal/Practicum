# Extreme Event Sensitivity Plan

Candidate rule (double threshold):
- High shift: smd_mean or psi_mean in top 33%
- Poor prediction: >=2 task metrics in worst 33%
- Candidate if both true

Sensitivity only:
- drop-1 for each candidate
- drop-2 for top2 candidates
- control-drop for one non-extreme event
- Keep main conclusion unchanged
