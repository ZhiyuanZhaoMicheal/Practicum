import rasterio, json, numpy as np, pandas as pd
from pyproj import Transformer

BASE = '/Users/zhaozhiyuan/Documents/GitHub/Practicum/project'
OUT  = f'{BASE}/nightlight-dashboard/public/data'

# All 25 events: EventName -> dash_id
ALL_EVENTS = {
    'Maria_SanJuan': 'maria',
    'Irma_Miami': 'irma',
    'Ida_NewOrleans': 'ida',
    'Laura_LakeCharles': 'laura',
    'Michael_PanamaCity': 'michael',
    'Earthquake_SanJuan': 'eq-pr',
    'Ian_CharlotteHarbor': 'ian-charlotte',
    'Ian_FortMyers': 'ian-fortmyers',
    'Earthquake_Hatay': 'eq-hatay',
    'Florence_Wilmington': 'florence-wilm',
    'Irma_Savannah': 'irma-savannah',
    'Isaias_Newark': 'isaias-nj',
    'Matthew_Jacksonville': 'matthew-jax',
    'Zeta_Atlanta': 'zeta-atlanta',
    'Zeta_Birmingham': 'zeta-birmingham',
    # 10 new
    'Matthew_Fayetteville': 'matthew-nc',
    'Florence_MyrtleBeach': 'florence-sc',
    'Isaias_Westchester': 'isaias-ny',
    'Uri_Houston': 'uri-houston',
    'Derecho_Chicago': 'derecho-chicago',
    'Severe_Detroit': 'severe-detroit',
    'Noreaster_Boston': 'noreaster-boston',
    'IceStorm_OKC': 'icestorm-okc',
    'Severe_Nashville': 'severe-nashville',
    'Atmos_Seattle': 'atmos-seattle',
}

TYPE_MAP = {
    'hospital': 'hospital',
    'aerodrome': 'airport',
    'fire_station': 'fire',
    'power_plant': 'power',
    'police': 'police',
    'government': 'government',
    'substation': 'substation',
    'water_works': 'water',
}

BUFFER_RADII = {
    'airport': 1250,
    'hospital': 750,
    'fire': 750,
    'power': 750,
    'police': 750,
    'government': 750,
    'substation': 750,
    'water': 750,
}

prob_stats_all = {}

for event_name, dash_id in ALL_EVENTS.items():
    tif_path = f'{BASE}/data/result/stage2/{event_name}_prob_map.tif'
    poi_path = f'{BASE}/data/result/stage2/poi_cache/{event_name}_poi.csv'

    print(f'Processing {event_name} -> {dash_id}')

    # Read TIF
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        rows, cols = data.shape

    # Get pixel coordinates for prob > 0.05
    mask = (data > 0.05) & ~np.isnan(data)
    ys, xs = np.where(mask)
    probs = data[mask]

    # Convert pixel coords to geographic coords
    geo_xs = transform.c + xs * transform.a + ys * transform.b
    geo_ys = transform.f + xs * transform.d + ys * transform.e

    # Transform to WGS84 if needed
    if crs and str(crs) != 'EPSG:4326':
        transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        lons, lats = transformer.transform(geo_xs, geo_ys)
    else:
        lons, lats = geo_xs, geo_ys

    # Build GeoJSON
    features = []
    for i in range(len(probs)):
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [round(float(lons[i]), 6), round(float(lats[i]), 6)]
            },
            'properties': {
                'probability': round(float(probs[i]), 3)
            }
        })

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    geojson_path = f'{OUT}/prob_{dash_id}.geojson'
    with open(geojson_path, 'w') as f:
        json.dump(geojson, f, separators=(',', ':'))

    print(f'  GeoJSON: {len(features)} features')

    # Compute prob stats
    prob_vals = np.array([f['properties']['probability'] for f in features])
    prob_stats_all[dash_id] = {
        'n': int(len(prob_vals)),
        'min': round(float(np.min(prob_vals)), 3),
        'max': round(float(np.max(prob_vals)), 3),
        'mean': round(float(np.mean(prob_vals)), 3),
        'p25': round(float(np.percentile(prob_vals, 25)), 3),
        'p50': round(float(np.percentile(prob_vals, 50)), 3),
        'p75': round(float(np.percentile(prob_vals, 75)), 3),
        'above_50': int(np.sum(prob_vals >= 0.5)),
        'above_80': int(np.sum(prob_vals >= 0.8)),
    }

    # Compute heatmap bounds
    lon_arr = np.array([f['geometry']['coordinates'][0] for f in features])
    lat_arr = np.array([f['geometry']['coordinates'][1] for f in features])
    padding = 0.005
    min_lon = float(np.min(lon_arr)) - padding
    max_lon = float(np.max(lon_arr)) + padding
    min_lat = float(np.min(lat_arr)) - padding
    max_lat = float(np.max(lat_arr)) + padding

    # Filter POIs to heatmap bounds
    try:
        poi_df = pd.read_csv(poi_path)
        filtered = poi_df[
            (poi_df['lon'] >= min_lon) & (poi_df['lon'] <= max_lon) &
            (poi_df['lat'] >= min_lat) & (poi_df['lat'] <= max_lat)
        ].copy()

        facilities = []
        for _, row in filtered.iterrows():
            ftype = TYPE_MAP.get(row['facility_type'], row['facility_type'])
            radius = BUFFER_RADII.get(ftype, 750)
            facilities.append({
                'name': row['name'],
                'type': ftype,
                'coords': [round(row['lon'], 5), round(row['lat'], 5)],
                'probability': 0.5,
                'radiusM': radius,
            })

        fac_path = f'{OUT}/facilities_{dash_id}.json'
        with open(fac_path, 'w') as f:
            json.dump(facilities, f, indent=2)
        print(f'  Facilities: {len(facilities)}')
    except Exception as e:
        print(f'  POI error: {e}')

# Save prob_stats for use later
with open('/tmp/prob_stats_25.json', 'w') as f:
    json.dump(prob_stats_all, f, indent=2)

print('\nDone! prob_stats saved to /tmp/prob_stats_25.json')
