"""
stage3_modelD_extra_regressions.py
==================================
Run the remaining Stage 3 models for Model D, completing parity with the
original Model-A regression suite:

  · M2  · Moran's I on Model D residuals (single event: Isaias_Newark)
  · M4  · City-size subgroup (large / medium / small)
  · M5  · Hurricane + wind_exposure  (single covariate)
  · M5+ · M5 + Census
  · M6  · M1 + ntl_drop
  · M7  · Hurricane + wind + ntl_drop
  · M9  · M8 + wind  (severity ~ mean_prob + wind, hurricane subset)
  · A'  · fac_density ~ pop + income (reverse regression)
  · B'  · A' + outage_severity
  · C'  · A' + outage_severity + wind  (hurricane subset)

Inputs:
  - data/result/stage3/zipcode_panel_modelD.parquet  (Model D mean_prob)
  - data/raw/acs_zcta_2022.csv
  - data/raw/ibtracs_NA.csv
  - data/raw/Outage_Dataset_R1/eaglei_outages_with_events_*.csv
  - data/raw/zcta520/*.shp, data/raw/counties/*.shp

Output:
  - data/result/stage3/regression_results_modelD_extra.json
"""

import os, sys, json, glob, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from scipy import stats
from libpysal.weights import KNN
from esda.moran import Moran

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR      = os.path.join(DATA_DIR, 'raw')
STAGE3_DIR   = os.path.join(DATA_DIR, 'result', 'stage3')

# ─── Load Model D panel ───────────────────────────────────────────────
panel = pd.read_parquet(os.path.join(STAGE3_DIR, 'zipcode_panel_modelD.parquet'))
panel['ZCTA5CE20'] = panel['ZCTA5CE20'].astype(str).str.zfill(5)
print(f"Panel: {len(panel)} ZIP-event obs, {panel['event_id'].nunique()} events")

# ─── Merge ACS Census ────────────────────────────────────────────────
acs = pd.read_csv(os.path.join(RAW_DIR, 'acs_zcta_2022.csv'))
acs['ZCTA5CE20'] = acs['ZCTA5CE20'].astype(str).str.zfill(5)
panel = panel.merge(acs[['ZCTA5CE20', 'total_pop', 'median_income']], on='ZCTA5CE20', how='left')
panel['pop_density']     = panel['total_pop'] / panel['area_km2'].clip(lower=0.01)
panel['log_pop_density'] = np.log(panel['pop_density'].clip(lower=1))
panel['log_income']      = np.log(panel['median_income'].clip(lower=1000))

# ─── Add city_size from event_id (mirror Stage 2 mapping) ────────────
CITY_SIZE = {
    'Maria_SanJuan':'large','Irma_Miami':'large','Ida_NewOrleans':'large',
    'Laura_LakeCharles':'small','Michael_PanamaCity':'small','Earthquake_SanJuan':'large',
    'Ian_CharlotteHarbor':'small','Ian_FortMyers':'medium','Earthquake_Hatay':'medium',
    'Florence_Wilmington':'medium','Irma_Savannah':'medium','Isaias_Newark':'large',
    'Matthew_Jacksonville':'large','Zeta_Atlanta':'large','Zeta_Birmingham':'medium',
    'Matthew_Fayetteville':'medium','Florence_MyrtleBeach':'small','Isaias_Westchester':'large',
    'Uri_Houston':'large','Derecho_Chicago':'large','Severe_Detroit':'large',
    'Noreaster_Boston':'large','IceStorm_OKC':'medium','Severe_Nashville':'large',
    'Atmos_Seattle':'large',
}
panel['city_size'] = panel['event_id'].map(CITY_SIZE)

# ─── Compute wind_exposure for hurricane events ──────────────────────
print("\nComputing wind_exposure from IBTrACS ...")
ibt = pd.read_csv(os.path.join(RAW_DIR, 'ibtracs_NA.csv'), skiprows=[1], low_memory=False)
ibt['SEASON'] = pd.to_numeric(ibt['SEASON'], errors='coerce')
ibt['LAT'] = pd.to_numeric(ibt['LAT'], errors='coerce')
ibt['LON'] = pd.to_numeric(ibt['LON'], errors='coerce')
for c in ['USA_R34_NE','USA_R34_SE','USA_R34_SW','USA_R34_NW']:
    ibt[c] = pd.to_numeric(ibt[c], errors='coerce')
ibt['R34_mean'] = ibt[['USA_R34_NE','USA_R34_SE','USA_R34_SW','USA_R34_NW']].mean(axis=1, skipna=True)

HURRICANE_NAME_YEAR = {
    'Irma_Miami':           ('IRMA',     2017),
    'Ida_NewOrleans':       ('IDA',      2021),
    'Laura_LakeCharles':    ('LAURA',    2020),
    'Michael_PanamaCity':   ('MICHAEL',  2018),
    'Ian_CharlotteHarbor':  ('IAN',      2022),
    'Ian_FortMyers':        ('IAN',      2022),
    'Florence_Wilmington':  ('FLORENCE', 2018),
    'Irma_Savannah':        ('IRMA',     2017),
    'Isaias_Newark':        ('ISAIAS',   2020),
    'Matthew_Jacksonville': ('MATTHEW',  2016),
    'Zeta_Atlanta':         ('ZETA',     2020),
    'Zeta_Birmingham':      ('ZETA',     2020),
    'Matthew_Fayetteville': ('MATTHEW',  2016),
    'Florence_MyrtleBeach': ('FLORENCE', 2018),
    'Isaias_Westchester':   ('ISAIAS',   2020),
}

# Load ZCTA centroids in WGS84 (for distance to track)
zcta_dir = os.path.join(RAW_DIR, 'zcta520')
shp = glob.glob(os.path.join(zcta_dir, '*.shp'))[0]
zcta_gdf = gpd.read_file(shp).to_crs('EPSG:4326')
zcta_gdf['ZCTA5CE20'] = zcta_gdf['ZCTA5CE20'].astype(str).str.zfill(5)
zcta_proj = zcta_gdf.to_crs('EPSG:5070')
zcta_gdf['cx'] = zcta_proj.geometry.centroid.to_crs('EPSG:4326').x
zcta_gdf['cy'] = zcta_proj.geometry.centroid.to_crs('EPSG:4326').y
zip_ctr = zcta_gdf[['ZCTA5CE20','cx','cy']]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2-lat1)
    dlam = np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlam/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def wind_exposure_for_event(eid, panel_subset, ibt_track):
    """Return dict ZCTA → wind_exposure = exp(-(d / R34)^2)."""
    if ibt_track.empty: return {}
    ctr = zip_ctr.set_index('ZCTA5CE20')
    out = {}
    for zip_id in panel_subset['ZCTA5CE20'].unique():
        if zip_id not in ctr.index: continue
        z = ctr.loc[zip_id]
        # min distance to any track point
        d = haversine_km(z['cy'], z['cx'], ibt_track['LAT'].values, ibt_track['LON'].values)
        idx = np.nanargmin(d)
        d_min = d[idx]
        r34 = ibt_track['R34_mean'].iloc[idx]
        if pd.isna(r34) or r34 <= 0:
            r34 = ibt_track['R34_mean'].dropna().mean()
            if pd.isna(r34) or r34 <= 0:
                continue
        # R34 is in nautical miles → convert to km
        r34_km = r34 * 1.852
        out[zip_id] = float(np.exp(-(d_min / r34_km)**2))
    return out

panel['wind_exposure'] = np.nan
for eid, (name, yr) in HURRICANE_NAME_YEAR.items():
    track = ibt[(ibt['SEASON']==yr) & (ibt['NAME'].str.upper()==name)]
    if track.empty:
        print(f"  [WARN] No IBTrACS track for {eid} ({name} {yr})")
        continue
    wexp = wind_exposure_for_event(eid, panel[panel['event_id']==eid], track)
    mask = panel['event_id']==eid
    panel.loc[mask, 'wind_exposure'] = panel.loc[mask, 'ZCTA5CE20'].map(wexp)
    n_set = panel.loc[mask, 'wind_exposure'].notna().sum()
    print(f"  {eid}: wind_exposure assigned to {n_set}/{mask.sum()} ZIPs")

# ─── Compute outage severity per ZIP (county-level, same as full script) ─
print("\nComputing EAGLE-I severity per ZIP ...")
EVENTS_META = {
    'Irma_Miami':            ('Florida','2017-09-10'),
    'Ida_NewOrleans':        ('Louisiana','2021-08-29'),
    'Laura_LakeCharles':     ('Louisiana','2020-08-27'),
    'Michael_PanamaCity':    ('Florida','2018-10-10'),
    'Ian_CharlotteHarbor':   ('Florida','2022-09-28'),
    'Ian_FortMyers':         ('Florida','2022-09-28'),
    'Florence_Wilmington':   ('North Carolina','2018-09-14'),
    'Irma_Savannah':         ('Georgia','2017-09-11'),
    'Isaias_Newark':         ('New Jersey','2020-08-04'),
    'Matthew_Jacksonville':  ('Florida','2016-10-07'),
    'Zeta_Atlanta':          ('Georgia','2020-10-29'),
    'Zeta_Birmingham':       ('Alabama','2020-10-29'),
    'Matthew_Fayetteville':  ('North Carolina','2016-10-08'),
    'Florence_MyrtleBeach':  ('South Carolina','2018-09-14'),
    'Isaias_Westchester':    ('New York','2020-08-04'),
    'Uri_Houston':           ('Texas','2021-02-15'),
    'Derecho_Chicago':       ('Illinois','2020-08-10'),
    'Severe_Detroit':        ('Michigan','2019-07-20'),
    'Noreaster_Boston':      ('Massachusetts','2021-10-27'),
    'IceStorm_OKC':          ('Oklahoma','2020-10-27'),
    'Severe_Nashville':      ('Tennessee','2023-07-18'),
    'Atmos_Seattle':         ('Washington','2022-11-04'),
}
eagle_dir = os.path.join(RAW_DIR, 'Outage_Dataset_R1')
eagle_dfs = []
for yr in range(2014, 2024):
    p = os.path.join(eagle_dir, f'eaglei_outages_with_events_{yr}.csv')
    if os.path.exists(p): eagle_dfs.append(pd.read_csv(p, low_memory=False))
eagle = pd.concat(eagle_dfs, ignore_index=True)
eagle['event_began'] = pd.to_datetime(eagle['Datetime Event Began'], errors='coerce')
sevs = []
for eid, (state, lf) in EVENTS_META.items():
    target = pd.Timestamp(lf)
    mask = (eagle['state_event'].str.contains(state, case=False, na=False) &
            ((eagle['event_began'] - target).abs().dt.days <= 14))
    s = eagle[mask].groupby('fips').agg(
        tot=('max_customers','sum'), mdur=('duration','mean')).reset_index()
    s['severity_county'] = np.log1p(s['tot']) * s['mdur']
    s['event_id'] = eid
    sevs.append(s[['event_id','fips','severity_county']])
sev = pd.concat(sevs, ignore_index=True)

# ZIP→county
counties_shp = glob.glob(os.path.join(RAW_DIR, 'counties', '*.shp'))[0]
cnty = gpd.read_file(counties_shp).to_crs('EPSG:4326')
cnty['fips'] = (cnty['STATEFP'].astype(str) + cnty['COUNTYFP'].astype(str)).astype(int)
zcta_proj = zcta_gdf.to_crs('EPSG:5070')
zcta_gdf['centroid'] = zcta_proj.geometry.centroid.to_crs('EPSG:4326')
ctr_geo = zcta_gdf.set_geometry('centroid')[['ZCTA5CE20','centroid']].rename(
    columns={'centroid':'geometry'}).set_geometry('geometry')
z2c = gpd.sjoin(ctr_geo, cnty[['fips','geometry']], how='left', predicate='within')[
    ['ZCTA5CE20','fips']].drop_duplicates(subset='ZCTA5CE20')
panel = panel.merge(z2c, on='ZCTA5CE20', how='left').merge(sev, on=['event_id','fips'], how='left')
print(f"  ZIPs with severity: {panel['severity_county'].notna().sum()}/{len(panel)}")

# ─── Helpers ──────────────────────────────────────────────────────────
def event_dummies(df):
    return pd.get_dummies(pd.Categorical(df['event_id']).codes, prefix='ev', drop_first=True).astype(float)

def fit_ols(df, y_col, x_cols, name, use_fe=True):
    df = df.copy()
    cols = [y_col] + x_cols
    mask = df[cols].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    parts = [df[x_cols].astype(float).reset_index(drop=True)]
    if use_fe and df['event_id'].nunique() > 1:
        parts.append(event_dummies(df).reset_index(drop=True))
    X = pd.concat(parts, axis=1)
    X = sm.add_constant(X)
    y = df[y_col].astype(float).values
    m = sm.OLS(y, X).fit(cov_type='HC1')
    out = {
        'name': name, 'n': int(len(df)),
        'r_squared': round(float(m.rsquared), 4),
        'adj_r_squared': round(float(m.rsquared_adj), 4),
        'coefs': {c: [round(float(m.params[c]), 4),
                      round(float(m.pvalues[c]), 6)] for c in x_cols},
    }
    print(f"\n[{name}]  n={out['n']}  R²={out['r_squared']}")
    for c in x_cols:
        print(f"   {c:<22s}  β={out['coefs'][c][0]:+.4f}  p={out['coefs'][c][1]:.4g}")
    return out

results = {}

# ═══ M2 · Moran's I on Model D residuals (Isaias_Newark) ═══
print("\n" + "="*60)
print("M2 · Moran's I on residuals (Isaias_Newark)")
print("="*60)
ev = panel[panel['event_id']=='Isaias_Newark'].dropna(subset=['mean_prob','fac_density']).copy()
if len(ev) > 10:
    # No controls
    X = sm.add_constant(ev[['fac_density']].astype(float))
    res_no = sm.OLS(ev['mean_prob'].astype(float).values, X).fit().resid
    ev = ev.merge(zip_ctr, on='ZCTA5CE20', how='left').dropna(subset=['cx','cy']).reset_index(drop=True)
    coords = ev[['cx','cy']].values
    w = KNN.from_array(coords, k=5); w.transform='r'
    mi_no = Moran(res_no[:len(ev)], w)
    # With Census
    ev2 = ev.dropna(subset=['log_pop_density','log_income']).reset_index(drop=True)
    X2 = sm.add_constant(ev2[['fac_density','log_pop_density','log_income']].astype(float))
    res_yes = sm.OLS(ev2['mean_prob'].astype(float).values, X2).fit().resid
    coords2 = ev2[['cx','cy']].values
    w2 = KNN.from_array(coords2, k=5); w2.transform='r'
    mi_yes = Moran(res_yes, w2)
    results['m2_morans_i'] = {
        'event': 'Isaias_Newark',
        'no_controls': {'I': round(float(mi_no.I),4), 'p': round(float(mi_no.p_sim),4), 'n': int(len(ev))},
        'with_controls': {'I': round(float(mi_yes.I),4), 'p': round(float(mi_yes.p_sim),4), 'n': int(len(ev2))},
    }
    print(f"  no_controls:    I={mi_no.I:.4f}  p={mi_no.p_sim}")
    print(f"  with_controls:  I={mi_yes.I:.4f}  p={mi_yes.p_sim}")

# ═══ M4 · City-size subgroup ═══
print("\n" + "="*60)
print("M4 · City-size subgroup")
print("="*60)
results['m4_subgroup'] = {}
for size in ['large','medium','small']:
    sub = panel[panel['city_size']==size].copy()
    if len(sub) < 10: continue
    r = fit_ols(sub, 'mean_prob', ['fac_density'], f'M4 · subgroup={size}')
    has = sub[sub['fac_count']>0]['mean_prob']
    no  = sub[sub['fac_count']==0]['mean_prob']
    r['with_fac_mean'] = round(float(has.mean()),3)
    r['without_fac_mean'] = round(float(no.mean()),3)
    results['m4_subgroup'][size] = r

panel['ntl_drop'] = panel['ntl_drop_pct']  # alias for clarity

# ═══ M5 · Hurricane + wind_exposure ═══
print("\n" + "="*60)
print("M5 · Hurricane subset · mean_prob ~ fac + wind + FE")
print("="*60)
hur = panel[panel['disaster_type']=='hurricane'].copy()
results['m5_wind'] = fit_ols(hur, 'mean_prob', ['fac_density','wind_exposure'],
                             'M5 · Hurricane + wind')

results['m5_plus_wind_census'] = fit_ols(hur, 'mean_prob',
    ['fac_density','wind_exposure','log_pop_density','log_income'],
    'M5+ · Hurricane + wind + Census')

# ═══ M6 · M1 + ntl_drop ═══
print("\n" + "="*60)
print("M6 · M1 + ntl_drop")
print("="*60)
results['m6_ntldrop'] = fit_ols(panel, 'mean_prob', ['fac_density','ntl_drop'],
                                'M6 · mean_prob ~ fac + ntl_drop + FE')

# ═══ M7 · Hurricane + wind + ntl_drop ═══
print("\n" + "="*60)
print("M7 · Hurricane full controls (wind + ntl_drop)")
print("="*60)
results['m7_full_hurricane'] = fit_ols(hur, 'mean_prob',
    ['fac_density','wind_exposure','ntl_drop'],
    'M7 · Hurricane + wind + ntl_drop + FE')

# ═══ M9 · M8 + wind ═══
print("\n" + "="*60)
print("M9 · severity ~ mean_prob + wind  (hurricane subset)")
print("="*60)
hur_sev = hur.dropna(subset=['severity_county','wind_exposure']).copy()
results['m9_severity_wind'] = fit_ols(hur_sev, 'severity_county',
    ['mean_prob','wind_exposure','fac_density'],
    'M9 · severity ~ mean_prob + wind + fac + FE')

# ═══ Reverse regressions: A' / B' / C' ═══
print("\n" + "="*60)
print("Reverse regressions: fac_density as DV (fairness analysis)")
print("="*60)
results['fairness_A'] = fit_ols(panel, 'fac_density',
    ['log_pop_density','log_income'],
    "A' · fac_density ~ pop + income + FE")

p_with_sev = panel.dropna(subset=['severity_county']).copy()
results['fairness_B'] = fit_ols(p_with_sev, 'fac_density',
    ['log_pop_density','log_income','severity_county'],
    "B' · A' + severity")

hur_full = hur_sev.dropna(subset=['log_pop_density','log_income']).copy()
results['fairness_C'] = fit_ols(hur_full, 'fac_density',
    ['log_pop_density','log_income','severity_county','wind_exposure'],
    "C' · B' + wind  (hurricane subset)")

# ═══ Equity gap on Model D severity tertiles ═══
print("\n" + "="*60)
print("Equity gap · Model D severity tertiles")
print("="*60)
ps = panel.dropna(subset=['severity_county']).copy()
ps['tertile'] = pd.qcut(ps['severity_county'], 3, labels=['Q1_low','Q2_mid','Q3_high'])
agg = ps.groupby('tertile', observed=True).agg(
    n=('mean_prob','size'),
    fac_density=('fac_density','mean'),
    mean_prob=('mean_prob','mean'),
    pop_density=('pop_density','mean'),
).round(3)
print(agg)
q1 = ps[ps['tertile']=='Q1_low']['fac_density']
q3 = ps[ps['tertile']=='Q3_high']['fac_density']
t, p = stats.ttest_ind(q1, q3)
results['equity_gap_severity'] = {
    'tertiles': agg.reset_index().to_dict('records'),
    'q1_vs_q3_t': round(float(t),3),
    'q1_vs_q3_p': float(p),
    'ratio_q3_q1': round(float(q3.mean()/q1.mean()),3),
}
print(f"  Q1 vs Q3 t={t:.2f}  p={p:.2e}  ratio Q3/Q1 = {q3.mean()/q1.mean():.3f}")

# Income quartile fairness
print("\n[Income quartile fairness]")
panel['income_q'] = pd.qcut(panel['median_income'].dropna(), 4, labels=['Q1','Q2','Q3','Q4'])
agg_i = panel.dropna(subset=['median_income']).groupby('income_q', observed=True).agg(
    n=('fac_density','size'), fac_density=('fac_density','mean'),
    pop_density=('pop_density','mean')).round(3)
print(agg_i)
qi1 = panel[panel['income_q']=='Q1']['fac_density']
qi4 = panel[panel['income_q']=='Q4']['fac_density']
ti, pi = stats.ttest_ind(qi1.dropna(), qi4.dropna())
results['equity_gap_income'] = {
    'quartiles': agg_i.reset_index().to_dict('records'),
    'q1_vs_q4_t': round(float(ti),3),
    'q1_vs_q4_p': float(pi),
}
print(f"  Q1 vs Q4 t={ti:.2f}  p={pi:.4f}")

# ─── Save ─────────────────────────────────────────────────────────────
out = os.path.join(STAGE3_DIR, 'regression_results_modelD_extra.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
print(f"\nSaved: {out}")
