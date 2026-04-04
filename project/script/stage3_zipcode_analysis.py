"""
stage3_zipcode_analysis.py
==========================
Stage 3: Zip-code-level spatial regression.
Tests whether facility density and predicted backup power probability
correlate with historical outage severity.

Usage:
    python stage3_zipcode_analysis.py

Inputs:
    - ZCTA shapefile (Census 2020)
    - EAGLE-I outage data (2014-2023)
    - Stage 2 probability maps (.tif)
    - OSM facility locations (poi_cache/)
    - IBTrACS hurricane tracks
    - Census ACS (downloaded via API)

Output:
    - data/result/stage3/zipcode_panel.parquet
    - data/result/stage3/regression_results.json
"""

import os, sys, glob, json, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import xy as tif_xy
from pyproj import Transformer
from scipy import stats
from shapely.geometry import Point, box

warnings.filterwarnings('ignore')

# ─── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR      = os.path.join(DATA_DIR, 'raw')
PROC_DIR     = os.path.join(DATA_DIR, 'processed')
STAGE2_DIR   = os.path.join(DATA_DIR, 'result', 'stage2')
STAGE3_DIR   = os.path.join(DATA_DIR, 'result', 'stage3')
POI_DIR      = os.path.join(STAGE2_DIR, 'poi_cache')
os.makedirs(STAGE3_DIR, exist_ok=True)

# ─── Event Configuration ────────────────────────────────────────────
EVENTS = {
    'Maria_SanJuan':       {'dash': 'maria',          'drive': 'Maria',                 'state': 'Puerto Rico', 'landfall': '2017-09-20', 'type': 'hurricane'},
    'Irma_Miami':          {'dash': 'irma',           'drive': 'Irma_Miami',            'state': 'Florida',     'landfall': '2017-09-10', 'type': 'hurricane'},
    'Ida_NewOrleans':      {'dash': 'ida',            'drive': 'Ida_NewOrleans',        'state': 'Louisiana',   'landfall': '2021-08-29', 'type': 'hurricane'},
    'Laura_LakeCharles':   {'dash': 'laura',          'drive': 'Laura_LakeCharles',     'state': 'Louisiana',   'landfall': '2020-08-27', 'type': 'hurricane'},
    'Michael_PanamaCity':  {'dash': 'michael',        'drive': 'Michael',               'state': 'Florida',     'landfall': '2018-10-10', 'type': 'hurricane'},
    'Earthquake_SanJuan':  {'dash': 'eq-pr',          'drive': 'Earthquake',            'state': 'Puerto Rico', 'landfall': '2020-01-07', 'type': 'earthquake'},
    'Ian_CharlotteHarbor': {'dash': 'ian-charlotte',  'drive': 'Ian_CharlotteHarbor',   'state': 'Florida',     'landfall': '2022-09-28', 'type': 'hurricane'},
    'Ian_FortMyers':       {'dash': 'ian-fortmyers',  'drive': 'Ian_FortMyers',         'state': 'Florida',     'landfall': '2022-09-28', 'type': 'hurricane'},
    'Earthquake_Hatay':    {'dash': 'eq-hatay',       'drive': 'Earthquake_Hatay',      'state': 'Turkey',      'landfall': '2023-02-06', 'type': 'earthquake'},
    'Florence_Wilmington': {'dash': 'florence-wilm',  'drive': 'Florence_Wilmington',   'state': 'North Carolina','landfall': '2018-09-14','type': 'hurricane'},
    'Irma_Savannah':       {'dash': 'irma-savannah',  'drive': 'Irma_Savannah',         'state': 'Georgia',     'landfall': '2017-09-11', 'type': 'hurricane'},
    'Isaias_Newark':       {'dash': 'isaias-nj',      'drive': 'Isaias_Newark',         'state': 'New Jersey',  'landfall': '2020-08-04', 'type': 'hurricane'},
    'Matthew_Jacksonville':{'dash': 'matthew-jax',    'drive': 'Matthew_Jacksonville',  'state': 'Florida',     'landfall': '2016-10-07', 'type': 'hurricane'},
    'Zeta_Atlanta':        {'dash': 'zeta-atlanta',   'drive': 'Zeta_Atlanta',          'state': 'Georgia',     'landfall': '2020-10-29', 'type': 'hurricane'},
    'Zeta_Birmingham':     {'dash': 'zeta-birmingham','drive': 'Zeta_Birmingham',       'state': 'Alabama',     'landfall': '2020-10-29', 'type': 'hurricane'},
    'Matthew_Fayetteville':{'dash': 'matthew-nc',     'drive': 'Matthew_Fayetteville',  'state': 'North Carolina','landfall': '2016-10-08','type': 'hurricane'},
    'Florence_MyrtleBeach':{'dash': 'florence-sc',    'drive': 'Florence_MyrtleBeach',  'state': 'South Carolina','landfall': '2018-09-14','type': 'hurricane'},
    'Isaias_Westchester':  {'dash': 'isaias-ny',      'drive': 'Isaias_Westchester',    'state': 'New York',    'landfall': '2020-08-04', 'type': 'hurricane'},
    'Uri_Houston':         {'dash': 'uri-houston',    'drive': 'Uri_Houston',           'state': 'Texas',       'landfall': '2021-02-15', 'type': 'winter_storm'},
    'Derecho_Chicago':     {'dash': 'derecho-chicago','drive': 'Derecho_Chicago',       'state': 'Illinois',    'landfall': '2020-08-10', 'type': 'derecho'},
    'Severe_Detroit':      {'dash': 'severe-detroit', 'drive': 'Severe_Detroit',        'state': 'Michigan',    'landfall': '2019-07-20', 'type': 'severe_storm'},
    'Noreaster_Boston':    {'dash': 'noreaster-boston','drive': 'Noreaster_Boston',      'state': 'Massachusetts','landfall': '2021-10-27','type': 'winter_storm'},
    'IceStorm_OKC':        {'dash': 'icestorm-okc',   'drive': 'IceStorm_OKC',          'state': 'Oklahoma',    'landfall': '2020-10-27', 'type': 'ice_storm'},
    'Severe_Nashville':    {'dash': 'severe-nashville','drive': 'Severe_Nashville',     'state': 'Tennessee',   'landfall': '2023-07-18', 'type': 'severe_storm'},
    'Atmos_Seattle':       {'dash': 'atmos-seattle',  'drive': 'Atmos_Seattle',         'state': 'Washington',  'landfall': '2022-11-04', 'type': 'winter_storm'},
}

FAC_TYPE_MAP = {
    'hospital': 'hospital', 'aerodrome': 'airport', 'fire_station': 'fire',
    'power_plant': 'power', 'police': 'police', 'government': 'government',
    'substation': 'substation', 'water_works': 'water',
}


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Load ZCTA boundaries and intersect with study areas
# ══════════════════════════════════════════════════════════════════════

def load_zcta():
    """Load Census ZCTA shapefile."""
    zcta_dir = os.path.join(RAW_DIR, 'zcta520')
    shp = glob.glob(os.path.join(zcta_dir, '*.shp'))
    if not shp:
        print("ERROR: ZCTA shapefile not found. Run download first.")
        sys.exit(1)
    gdf = gpd.read_file(shp[0])
    gdf = gdf.to_crs('EPSG:4326')
    print(f"ZCTA loaded: {len(gdf)} zip codes")
    return gdf


def get_study_area_bounds(event_id):
    """Get WGS84 bounding box from the event's probability map TIF."""
    cfg = EVENTS[event_id]
    tif_path = os.path.join(STAGE2_DIR, f'{event_id}_prob_map.tif')
    if not os.path.exists(tif_path):
        return None

    with rasterio.open(tif_path) as src:
        b = src.bounds
        crs = src.crs

    if str(crs).upper() != 'EPSG:4326':
        t = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        lon_min, lat_min = t.transform(b.left, b.bottom)
        lon_max, lat_max = t.transform(b.right, b.top)
    else:
        lon_min, lat_min = b.left, b.bottom
        lon_max, lat_max = b.right, b.top

    return box(lon_min, lat_min, lon_max, lat_max)


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Per-event zip code features
# ══════════════════════════════════════════════════════════════════════

def compute_zipcode_features(event_id, zcta_gdf):
    """
    For one event, compute per-zip-code features:
    - mean_backup_prob: mean predicted probability from Stage 2
    - facility_density: POIs per km²
    - ntl_drop_pct: mean NTL decline
    """
    cfg = EVENTS[event_id]

    # Get study area bounds
    study_box = get_study_area_bounds(event_id)
    if study_box is None:
        return None

    # Filter ZCTAs to study area
    zips = zcta_gdf[zcta_gdf.intersects(study_box)].copy()
    if zips.empty:
        return None

    # Load prob map
    tif_path = os.path.join(STAGE2_DIR, f'{event_id}_prob_map.tif')
    with rasterio.open(tif_path) as src:
        prob = src.read(1).astype(np.float32)
        transform = src.transform
        tif_crs = src.crs
        nrows, ncols = prob.shape

    # Pixel coordinates → WGS84
    row_idx, col_idx = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing='ij')
    xs, ys = tif_xy(transform, row_idx.flatten(), col_idx.flatten())
    xs, ys = np.array(xs), np.array(ys)

    if str(tif_crs).upper() != 'EPSG:4326':
        t = Transformer.from_crs(tif_crs, 'EPSG:4326', always_xy=True)
        lons, lats = t.transform(xs, ys)
    else:
        lons, lats = xs, ys

    prob_flat = prob.flatten()
    valid = (prob_flat > 0) & np.isfinite(prob_flat)

    # Create pixel GeoDataFrame
    pixels = gpd.GeoDataFrame({
        'prob': prob_flat[valid],
        'geometry': [Point(lon, lat) for lon, lat in zip(lons[valid], lats[valid])],
    }, crs='EPSG:4326')

    # Load POIs
    poi_path = os.path.join(POI_DIR, f'{event_id}_poi.csv')
    pois = pd.DataFrame()
    if os.path.exists(poi_path):
        pois = pd.read_csv(poi_path)
        if not pois.empty and 'lon' in pois.columns:
            pois = gpd.GeoDataFrame(
                pois, geometry=[Point(r.lon, r.lat) for _, r in pois.iterrows()],
                crs='EPSG:4326'
            )

    # Load pixel panel for NTL drop
    panel_path = os.path.join(STAGE2_DIR, 'pixel_panel.parquet')
    panel = pd.read_parquet(panel_path)
    ev_panel = panel[panel['event_id'] == event_id]

    # Spatial join: pixels → ZCTAs
    pixel_in_zip = gpd.sjoin(pixels, zips[['ZCTA5CE20', 'geometry']], how='left', predicate='within')

    # Per-zip aggregation
    zip_stats = pixel_in_zip.groupby('ZCTA5CE20').agg(
        mean_prob=('prob', 'mean'),
        max_prob=('prob', 'max'),
        n_pixels=('prob', 'count'),
    ).reset_index()

    # POI count per zip
    if not pois.empty and isinstance(pois, gpd.GeoDataFrame):
        poi_in_zip = gpd.sjoin(pois, zips[['ZCTA5CE20', 'geometry']], how='left', predicate='within')
        poi_counts = poi_in_zip.groupby('ZCTA5CE20').size().reset_index(name='fac_count')
        zip_stats = zip_stats.merge(poi_counts, on='ZCTA5CE20', how='left')
    zip_stats['fac_count'] = zip_stats.get('fac_count', 0).fillna(0).astype(int)

    # Compute zip area for density
    zips_proj = zips.to_crs('EPSG:3857')  # for area in m²
    zips['area_km2'] = zips_proj.geometry.area / 1e6
    zip_stats = zip_stats.merge(zips[['ZCTA5CE20', 'area_km2']], on='ZCTA5CE20', how='left')
    zip_stats['fac_density'] = zip_stats['fac_count'] / zip_stats['area_km2'].clip(lower=0.01)

    # NTL drop (event-level average as proxy)
    zip_stats['ntl_drop_pct'] = -ev_panel['delta_ntl'].mean() if len(ev_panel) > 0 else 0

    # Metadata
    zip_stats['event_id'] = event_id
    zip_stats['disaster_type'] = cfg['type']
    zip_stats['state'] = cfg['state']

    return zip_stats


# ══════════════════════════════════════════════════════════════════════
# STEP 3: EAGLE-I outage severity per county
# ══════════════════════════════════════════════════════════════════════

def load_eagle_i():
    """Load and aggregate EAGLE-I outage data."""
    eagle_dir = os.path.join(RAW_DIR, 'Outage_Dataset_R1')
    dfs = []
    for yr in range(2014, 2024):
        path = os.path.join(eagle_dir, f'eaglei_outages_with_events_{yr}.csv')
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)
    df['event_began'] = pd.to_datetime(df['Datetime Event Began'], errors='coerce')
    print(f"EAGLE-I loaded: {len(df):,} records")
    return df


def get_county_outage_severity(eagle_df, state, landfall_date, window_days=14):
    """
    Compute outage severity for counties in a state around the landfall date.
    Returns DataFrame with fips, total_customers, max_duration, severity_score.
    """
    target = pd.Timestamp(landfall_date)
    mask = (
        (eagle_df['state_event'].str.contains(state, case=False, na=False)) &
        ((eagle_df['event_began'] - target).abs().dt.days <= window_days)
    )
    matched = eagle_df[mask].copy()
    if matched.empty:
        return pd.DataFrame()

    county_stats = matched.groupby('fips').agg(
        total_customers=('max_customers', 'sum'),
        peak_customers=('max_customers', 'max'),
        max_duration=('duration', 'max'),
        mean_duration=('duration', 'mean'),
        n_outages=('fips', 'count'),
    ).reset_index()

    # Severity composite: log(customers) × duration
    county_stats['severity_score'] = (
        np.log1p(county_stats['total_customers']) *
        county_stats['mean_duration']
    )

    return county_stats


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Regression
# ══════════════════════════════════════════════════════════════════════

def run_regressions(zip_panel):
    """Run OLS and spatial regressions on the zip-code panel."""
    import statsmodels.api as sm

    results = {}

    # ── OLS: mean_prob ~ fac_density + event FE ──
    print("\n[OLS] mean_prob ~ fac_density + event_FE")
    zip_panel['event_code'] = pd.Categorical(zip_panel['event_id']).codes
    event_dummies = pd.get_dummies(zip_panel['event_code'], prefix='ev', drop_first=True).astype(float)
    X = pd.concat([zip_panel[['fac_density']].astype(float), event_dummies], axis=1)
    X = sm.add_constant(X)
    y = zip_panel['mean_prob'].astype(float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    model = sm.OLS(y[mask], X[mask]).fit()
    print(f"  R² = {model.rsquared:.4f}")
    print(f"  fac_density: coef = {model.params['fac_density']:.4f}, p = {model.pvalues['fac_density']:.2e}")

    results['ols_prob'] = {
        'r_squared': round(model.rsquared, 4),
        'fac_density_coef': round(model.params['fac_density'], 4),
        'fac_density_p': float(model.pvalues['fac_density']),
        'n': int(mask.sum()),
    }

    # ── T-test: with vs without facilities ──
    has_fac = zip_panel[zip_panel['fac_count'] > 0]['mean_prob']
    no_fac = zip_panel[zip_panel['fac_count'] == 0]['mean_prob']
    t, p = stats.ttest_ind(has_fac, no_fac)
    mw_u, mw_p = stats.mannwhitneyu(has_fac, no_fac, alternative='greater')

    print(f"\n[T-test] Zips with facilities vs without")
    print(f"  With:    mean P = {has_fac.mean():.3f} (n={len(has_fac)})")
    print(f"  Without: mean P = {no_fac.mean():.3f} (n={len(no_fac)})")
    print(f"  t={t:.2f}, p={p:.2e}")
    print(f"  Mann-Whitney: U={mw_u:.0f}, p={mw_p:.2e}")

    results['ttest'] = {
        'with_mean': round(has_fac.mean(), 3),
        'without_mean': round(no_fac.mean(), 3),
        'difference': round(has_fac.mean() - no_fac.mean(), 3),
        't_stat': round(t, 2),
        'p_value': float(p),
        'n_with': int(len(has_fac)),
        'n_without': int(len(no_fac)),
    }

    # ── Per-event consistency ──
    consistent = 0
    total_ev = 0
    per_event = []
    for eid in sorted(zip_panel['event_id'].unique()):
        ev = zip_panel[zip_panel['event_id'] == eid]
        w = ev[ev['fac_count'] > 0]['mean_prob']
        n = ev[ev['fac_count'] == 0]['mean_prob']
        if len(w) >= 3 and len(n) >= 3:
            d = w.mean() - n.mean()
            if d > 0: consistent += 1
            total_ev += 1
            per_event.append({'event': eid, 'with': round(w.mean(), 3),
                              'without': round(n.mean(), 3), 'delta': round(d, 3)})

    print(f"\n[Consistency] {consistent}/{total_ev} events ({100*consistent/max(total_ev,1):.0f}%)")
    results['consistency'] = {
        'consistent': consistent,
        'total': total_ev,
        'pct': round(100 * consistent / max(total_ev, 1), 1),
        'per_event': per_event,
    }

    return results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("STAGE 3: Zip-Code Level Analysis")
    print("=" * 60)

    # Step 1: Load ZCTA
    print("\n[1/4] Loading ZCTA boundaries...")
    zcta = load_zcta()

    # Step 2: Compute per-zip features for each event
    print("\n[2/4] Computing zip-code features per event...")
    all_zips = []
    for event_id, cfg in EVENTS.items():
        if cfg['state'] in ('Turkey', 'Puerto Rico'):
            print(f"  {event_id}: SKIP (no US ZCTA)")
            continue
        print(f"  {event_id}...", end=" ", flush=True)
        zf = compute_zipcode_features(event_id, zcta)
        if zf is not None and len(zf) > 0:
            all_zips.append(zf)
            print(f"{len(zf)} zip codes, {int(zf['fac_count'].sum())} facilities")
        else:
            print("no data")

    if not all_zips:
        print("ERROR: No zip code data generated")
        return

    zip_panel = pd.concat(all_zips, ignore_index=True)
    print(f"\nTotal: {len(zip_panel)} zip-event observations, {zip_panel['event_id'].nunique()} events")

    # Save panel
    panel_path = os.path.join(STAGE3_DIR, 'zipcode_panel.parquet')
    zip_panel.to_parquet(panel_path)
    print(f"Saved: {panel_path}")

    # Step 3: Regressions
    print("\n[3/4] Running regressions...")
    results = run_regressions(zip_panel)

    # Save results
    results_path = os.path.join(STAGE3_DIR, 'regression_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\nSaved: {results_path}")

    # Step 4: Summary
    print(f"\n{'='*60}")
    print("STAGE 3 SUMMARY")
    print(f"{'='*60}")
    print(f"  Zip codes: {len(zip_panel)}")
    print(f"  Events:    {zip_panel['event_id'].nunique()}")
    print(f"  OLS R²:    {results['ols_prob']['r_squared']}")
    print(f"  T-test:    p = {results['ttest']['p_value']:.2e}")
    print(f"  Consistency: {results['consistency']['consistent']}/{results['consistency']['total']}")


if __name__ == '__main__':
    main()
