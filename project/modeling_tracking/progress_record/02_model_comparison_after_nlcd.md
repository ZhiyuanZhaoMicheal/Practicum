# Model Comparison After NLCD Control

## Key Metrics (`in_buffer`)
- OLS: `no_nlcd` coef = 0.0283 (p=0.0701) -> `with_nlcd` coef = -0.0233 (p=0.1292)
- MixedLM: `no_nlcd` coef = 0.0283 (p=0.0196) -> `with_nlcd` coef = -0.0233 (p=0.0496)
- Logit: `no_nlcd` OR = 0.6831 (p=1.02e-07) -> `with_nlcd` OR = 1.1676 (p=0.1243)
- Cox: `no_nlcd` HR = 1.1261 (p=4.53e-07) -> `with_nlcd` HR = 1.1075 (p=2.00e-05)

## Interpretation
- Land-use controlled specification materially changes OLS/MixedLM/Logit direction and significance.
- Cox remains directionally consistent (HR>1) and statistically strong.
- Cross-model conflict indicates sensitivity to land-use control design and sample composition.

## Practical Conclusion (Current Snapshot)
- If prioritizing event-level hierarchical explanation in the baseline setup: MixedLM (`no_nlcd`) remains strongest.
- If prioritizing consistency under current NLCD-controlled setup: Cox provides the most stable positive signal.
