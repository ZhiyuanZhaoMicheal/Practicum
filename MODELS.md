# NightLight · Project Methods & Results

> Detecting backup power deployment from nighttime satellite imagery during disasters.
> 25 disaster events · 2016–2023 · 17 U.S. states + Turkey

---

## 1 · Research Question

**Can we detect backup-generator deployment from satellite nighttime light (NTL) during major outages?**

Three sub-questions:
- **RQ1 (Detection)** — Do facility-adjacent pixels behave differently during outages?
- **RQ2 (Prediction)** — Can NTL temporal patterns alone predict backup probability?
- **RQ3 (Equity)** — Do communities with more critical facilities experience less severe outages?

The challenge: no public registry of generators exists, so we use **critical infrastructure** (hospitals, airports, fire stations, police, power plants) as **proxy labels** — these facilities are legally required to maintain backup power.

---

## 2 · Data Sources

| Dataset | Source | Role |
|---|---|---|
| **Nighttime light** | NASA VIIRS Black Marble VNP46A2 (daily, 500 m) via Google Earth Engine | Primary signal |
| **Critical facilities (POI)** | OpenStreetMap (Overpass API) | Proxy labels + Stage 3 facility density |
| **Outage records** | DOE EAGLE-I (2014–2023) | Stage 3 dependent variable |
| **Demographics** | US Census ACS 2022 | Stage 3 control variables |
| **Hurricane tracks** | IBTrACS v4 | Stage 3 wind exposure |
| **ZIP boundaries** | US Census ZCTA 2020 | Stage 3 spatial unit |
| **Generator permits** | Miami-Dade County Open Data (Hurricane Irma) | Independent ground-truth check |

### NTL pre-processing

```
band       : Gap_Filled_DNB_BRDF_Corrected_NTL  (atmosphere/BRDF/lunar-corrected)
scale      : raw DN × 0.1  =  nW/cm²/sr
cloud mask : extract bits 6–7 (cloud), 8 (shadow), 9 (cirrus) from QF_Cloud_Mask
filter     : usable if cloud_fraction < 30%  in study-area ROI
export     : 500 m GeoTIFF
```

**25 events** selected for diversity in disaster type (hurricane / earthquake / winter storm / derecho / severe storm / ice storm) and city size (large / medium / small).

---

## 3 · Stage 1 · Exploratory Analysis

### Buffer and resilience definitions

```
buffer radius : aerodrome 1250 m, all other facility types 750 m
```

For each facility type $f$, baseline NTL is the pre-disaster mean within the buffer:

$$
\text{pre\_buf}_f = \frac{1}{|T_{\text{pre}}|}\sum_{t \in T_{\text{pre}}} \text{NTL}_{\text{buf}, f}(t)
$$

**Resilience Ratio** $R(t)$:

$$
R_{\text{buf}, f}(t) = \frac{\text{NTL}_{\text{buf}, f}(t)}{\text{pre\_buf}_f}, \qquad
R_{\text{nobuf}, f}(t) = \frac{\text{NTL}_{\text{nobuf}, f}(t)}{\text{pre\_nobuf}_f}
$$

**Resilience Advantage** = $R_{\text{buf}} - R_{\text{nobuf}}$ (cross-event comparison metric).

### Key findings

- Buffer pixels consistently retain more brightness than non-buffer pixels across all 25 events.
- **Floor effect**: in dim baseline areas, R cannot drop much further; signal weaker.
- Hospital and airport buffers show the strongest contrast; police and fire stations weaker.
- Cross-city heterogeneity: large cities (Miami, New Orleans) show clearer signal than small cities (Lake Charles, Panama City).

These observations motivated formal statistical testing in Stage 1.5 and feature engineering in Stage 2.

---

## 4 · Stage 1.5 · Interpretive Modeling — Triangulation

To confirm the resilience signal is statistically real and to identify confounds, four independent statistical models are applied to the same panel (n ≈ 10,306 pixels, 6 events):

### Shared specification

$$
Y_i = \beta_0 + \beta_1 \cdot \text{in\_buffer}_i + \beta_2 \cdot \text{pre\_mean\_ntl}_i + \beta_3 (\text{in\_buffer} \times \text{pre\_mean\_ntl})_i + \gamma \cdot \mathbb{1}\{\text{event}_i\} \;[+\; \delta \cdot \text{NLCD}_i\,] + \varepsilon_i
$$

Two variants per model: with and without NLCD land-use controls.

### Model A · OLS — average effect size

$$
\Delta\text{ntl}_i = \beta_0 + \beta_1 \cdot \text{in\_buffer}_i + \cdots + \varepsilon_i, \qquad \text{HC1 SE}
$$

Buffer pixels show **+2.8 % less NTL decline** (p = 0.070); the interaction term `in_buffer × pre_mean_ntl` is highly significant with NLCD controls (β = +0.010, **p = 0.0002**) — the floor effect made statistical.

### Model B · MixedLM — clustering correction

$$
\Delta\text{ntl}_{ij} = \beta_0 + \beta_1 \cdot \text{in\_buffer}_{ij} + \cdots + u_j + \varepsilon_{ij}, \qquad u_j \sim \mathcal{N}(0,\sigma_u^2)
$$

Event-level random intercepts. Same coefficient as OLS but tighter inference: **p improves from 0.070 to 0.020**.

### Model C · Logistic — damage probability

$$
\text{damaged}_i = \mathbb{1}[\Delta\text{ntl}_i < -0.10], \quad
\log\frac{P(\text{damaged}_i)}{1-P(\text{damaged}_i)} = \beta_0 + \beta_1 \cdot \text{in\_buffer}_i + \cdots
$$

Buffer pixels show **OR = 0.68 (p < 0.001)** — 32 % lower damage odds. LOEO sign consistency: **6/6 events**.

### Model D · Cox PH — recovery speed

$$
h(t \mid x_i) = h_0(t) \cdot \exp(\beta_1 \cdot \text{in\_buffer}_i + \cdots), \qquad \text{HR} = e^{\beta_1}
$$

Buffer pixels recover **~13 % faster** (HR = 1.13, p < 0.001), stable across 80/90/95 % thresholds.

### Triangulation summary

| Model | Effect direction | Significance |
|---|---|---|
| OLS | + | p = 0.020 (MixedLM) |
| Logistic | OR = 0.68 | p < 0.001 |
| Cox PH | HR = 1.13 | p < 0.001 |

All four converge on the same conclusion: **the proxy-label signal is real, robust, and modest in magnitude**. The signal is strongest in brighter areas (interaction term), weakest in dim small cities (floor effect). This motivates tree-based predictive models in Stage 2 with city-level normalization features.

---

## 5 · Stage 2 · Predictive Model

### Pixel panel

```
panel size    : ~33,700 pixels × 25 events
features      : pure NTL behavior (no spatial proximity to facilities)
label         : in_buffer_strict ∈ {0, 1}  (within 750 m of HIGH/MEDIUM facility)
                HIGH  = hospital, aerodrome, power_plant
                MEDIUM = fire_station, police
filter        : pre_mean_ntl > 0.5  (exclude pixels below noise floor)
```

### Feature set (10 features, pure NTL)

```
NTL behavior              : drop_magnitude, delta_ntl, log_pre_ntl, log_post_ntl,
                            log_city_pre_mean, ntl_relative
Floor effect indicator    : below_city_median
City / disaster controls  : city_size_code, is_hurricane, is_earthquake
```

The model deliberately excludes facility-proximity features so its predictions reflect what is genuinely visible from satellite NTL temporal pattern alone — not memorization of where facilities are located.

### Algorithms

```
RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=20,
                       max_features='sqrt', class_weight='balanced')

XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
              scale_pos_weight=5, early_stopping_rounds=50)

Ensemble  P = 0.7 · P_RF + 0.3 · P_XGB
```

### Validation — Leave-One-Event-Out (LOEO)

```
for held_out in 25 events:
    train = pixels from all events except held_out
    test  = pixels from held_out
    fit, predict, compute AUC
mean ± std over 25 folds
```

### Result

$$
\boxed{\text{LOEO AUC} = \mathbf{0.704}}
$$

A real but modest signal — significantly above random (0.5), confirming that nighttime light temporal patterns alone carry detectable information about commercial-scale backup power activity. The 500 m / 25-hectare pixel size sets the floor on what is detectable.

### Outputs

```
data/result/stage2/
├── pixel_panel.parquet                     # 33.7K × 25 events
├── rf_modelD.pkl, xgb_modelD.pkl           # Final ensemble
├── feature_importance_modelD.csv
└── {Event}_prob_map_modelD.tif × 25        # Per-event pixel-level probability maps
```

---

## 6 · Stage 3 · Zip-Code Spatial Regression

### Panel construction

Stage 2 prob_map for each event → spatial join to ZCTA → aggregate per ZIP:

```
panel size : 1,002 ZIP-event observations  ·  22 events  ·  19 U.S. states

per row:
  ZCTA5CE20, event_id, state, disaster_type, area_km2,
  mean_prob, max_prob, n_pixels,        # Stage 2 prob aggregated
  fac_count, fac_density,                # = OSM POI count / area_km2
  ntl_drop_pct
```

(Puerto Rico and Turkey excluded — no US ZCTA boundaries.)

### Controls

```
log_pop_density = log(population / area)         # Census ACS 2022
log_income      = log(median household income)   # Census ACS 2022
wind_exposure   = exp(-(d / R34)²)               # IBTrACS Holland decay (hurricanes only)
event FE γᵢ                                       # event fixed effects
```

### Outage severity (for closed-loop validation)

$$
\text{severity}_{\text{county}} = \log(1 + \text{total\_customers}) \times \text{mean\_duration}
$$

Joined to ZIPs via county centroid.

### Results

#### Model 1 · Baseline OLS

$$
\text{mean\_prob} = \beta_0 + \beta_1 \cdot \text{fac\_density} + \gamma_i + \varepsilon
$$

| Variable | β | p | R² | n |
|---|---|---|---|---|
| fac_density | **+0.058** | < 1 × 10⁻⁴⁰ | **0.556** | 1,002 |

#### Model 2 · OLS with Census controls

$$
\text{mean\_prob} = \beta_0 + \beta_1 \cdot \text{fac\_density} + \beta_2 \cdot \log(\text{pop}) + \beta_3 \cdot \log(\text{income}) + \gamma_i + \varepsilon
$$

| Variable | β | p | R² | n |
|---|---|---|---|---|
| fac_density | +0.049 | < 1 × 10⁻⁴⁰ | **0.747** | 977 |
| log_pop_density | +0.064 | < 1 × 10⁻⁴⁰ | | |
| log_income | −0.080 | < 1 × 10⁻⁴⁰ | | |

> Facility density remains highly significant after controlling for population density and income — the effect is not purely a proxy for urbanization.

#### Model 3 · Spatial Error Model (robustness check)

Using `spreg.GM_Error_Het` with KNN k=5 spatial weights:

$$
y = X\beta + u, \qquad u = \lambda W u + \varepsilon
$$

Coefficient direction and significance are preserved under spatial-error control — the result is robust to spatial autocorrelation.

#### Model 4 · Hurricane subsample with wind exposure

$$
\text{mean\_prob} = \beta_0 + \beta_1 \cdot \text{fac\_density} + \beta_2 \cdot \text{wind\_exposure} + \beta_3 \cdot \log(\text{pop}) + \beta_4 \cdot \log(\text{income}) + \gamma_i + \varepsilon
$$

| Variable | β | p |
|---|---|---|
| fac_density | +0.054 | 6 × 10⁻⁶ |
| wind_exposure | −0.20 | 0.094 (n.s.) |
| log_pop_density | +0.056 | < 1 × 10⁻⁴⁰ |
| log_income | −0.092 | < 1 × 10⁻⁴⁰ |

n = 536, R² = **0.775**. After controlling for population density, wind exposure is no longer a significant predictor — distance to the storm track was largely a proxy for urbanization.

#### Model 5 · Closed-loop validation against EAGLE-I outage severity

$$
\text{outage\_severity} = \beta_0 + \beta_1 \cdot \text{mean\_prob} + \beta_2 \cdot \log(\text{pop}) + \beta_3 \cdot \log(\text{income}) + \gamma_i + \varepsilon
$$

| Variable | β | p |
|---|---|---|
| **mean_prob** | **+123.8** | **0.012** |
| log_pop_density | −5.87 | 0.45 |
| log_income | +11.67 | 0.36 |

n = 911, R² = 0.760.

> Predicted backup probability is **positively** correlated with actual outage severity, and the relationship strengthens after controlling for Census demographics. This is consistent with the natural deployment direction:
>
> **Generators are installed where outages strike most.** Areas with frequent severe blackouts have higher rates of backup deployment, so during disasters their NTL signature resembles the institutional "lights stay on" pattern Model D was trained to recognize.

### Equity finding

ZIPs grouped by outage-severity tertile:

| Outage tier | n | Facility density (per km²) |
|---|---|---|
| Low (Q1) | 366 | **0.490** |
| Mid (Q2) | 265 | 0.311 |
| High (Q3) | 304 | **0.309** |

t-test (Q1 vs Q3): t = 2.56, **p = 0.011**

> ZIPs that suffer the most severe outages have approximately **63 %** of the critical-facility density of the least-affected ZIPs. Income-quartile comparison (Q1 vs Q4) is not significant (p = 0.233) — the disparity follows urban spatial structure (core vs periphery), not income lines.

---

## 7 · Ground-Truth Validation · Miami-Dade Generator Permits

To validate Stage 2 predictions against actual generator records, we use **Miami-Dade County permit data** for Hurricane Irma (Sept 2017): 592 standalone-generator permits, with explicit residential / commercial flags (`RESCOMM`).

```
Total permits        : 592 (499 residential + 93 commercial)
Within Irma study area : 169
With NTL coverage     : 136 (106 residential + 30 commercial)
```

### Sanity check: probability distribution at known generator locations

| Cohort | n | Median prob | Above event median (0.672) | Above 0.5 |
|---|---|---|---|---|
| **Commercial** | 30 | **0.722** | **83 %** | **100 %** |
| Residential | 106 | 0.606 | 14 % | 88 % |

> **Commercial generators are detected** — 83 % of permitted commercial-installation locations sit above the event-wide median probability, with all 30 sampled sites scoring > 0.5. This is despite Model D having no spatial-proximity features.
>
> **Residential generators remain below the noise floor** — at residential locations the model's distribution is no different from random pixels. This empirically confirms the 500 m resolution boundary: only commercial / institutional backup behavior is distinguishable; household-scale generators are too small to register at 25-hectare pixel scale.

This split is the cleanest evidence in the project: the model succeeds where the physics permits and fails where the physics forbids.

---

## 8 · Visualization

| Component | Stack | Purpose |
|---|---|---|
| Per-pixel probability maps | GeoTIFF + GeoJSON | 25 events, exported per-event |
| Interactive dashboard | Vue 3 + Vite + MapLibre GL JS | Per-pixel heatmap + facility overlay + recovery curves |
| Documentation site | Vue Router (hash mode) | 12 sections of methodology + results |
| Deployment | GitHub Actions → GitHub Pages | Static, free hosting |

The dashboard supports five basemaps (Dark Matter, Positron, Voyager, Satellite, Dark No-Labels), per-event quantile color scales (so each event's distribution displays with full contrast), and click-through facility popups with predicted probability.

---

## 9 · Conclusions

### What the satellite signal tells us

1. **Pixel-level (Stage 2)** — Pure NTL temporal patterns predict backup-power proxy labels with **AUC 0.704** across 25 unseen events. A real but modest signal at 500 m resolution.
2. **Aggregate level (Stage 3)** — Facility density predicts mean predicted probability with **R² = 0.747** after Census controls, and predicted probability is positively associated with real outage severity (β = +123.8, p = 0.012) — consistent with backup deployment occurring where outages happen most.
3. **Ground truth (Miami-Dade)** — 83 % of commercial-permit locations score above the event-wide median probability; residential permits do not. The detection capability matches the physics: institutional yes, household no.

### What it tells us about infrastructure equity

The **63 % gap** in facility density between the most outage-vulnerable and least-vulnerable ZIPs is not aligned with disaster risk — facilities are sited by daily service demand, not resilience need. The disparity follows urban spatial structure, not income.

### Limitations

| Limitation | Implication |
|---|---|
| 500 m / 25-hectare pixel | Cannot resolve sub-block geography; residential generators below noise floor |
| Daily revisit + 30 % cloud loss | Some events have only a handful of usable observations |
| Proxy labels (facility buffers, not generator records) | Validation must rely on independent ground truth (Miami-Dade) where available |

### Future work

- **Higher-resolution sensors** (≤ 100 m, e.g. Luojia-1, future SDGSAT) could push detection from commercial down to residential scale.
- **Generator permit databases** (where they exist, like Miami-Dade) provide cleaner training signal than facility buffers.
- **Real-time outage response** — extending the dashboard to ingest live VIIRS data could surface aggregate backup deployment within 24 h of a disaster.
- **Equity-aware infrastructure planning** — the 63 % gap finding suggests resilience investment should consciously offset urban-periphery disparity.

---

## Pipeline reference

```
Stage 0  Data download  (NASA GEE + OSM + EAGLE-I + Census + IBTrACS)
   ↓
Stage 1  EDA           (Resilience Ratio R = NTL_post / NTL_pre)
   ↓
Stage 1.5 Interpretive (OLS · MixedLM · Logistic · Cox PH triangulation)
   ↓
Stage 2  Predictive    (RF + XGBoost ensemble, pure-NTL features, LOEO AUC = 0.704)
   ↓
Stage 3  ZIP regression (Census-controlled R² = 0.747, closed-loop β = +123.8 p = 0.012)
   ↓
Stage 4  Visualization (Vue + MapLibre dashboard, 25 events, interactive)
```
