"""
Re-fetch POI data from Overpass with name tags, update poi_cache CSVs.
"""
import os, time, requests, pandas as pd

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OSM_QUERIES = {
    "hospital":     'node["amenity"="hospital"]({bbox});way["amenity"="hospital"]({bbox});',
    "aerodrome":    'node["aeroway"="aerodrome"]({bbox});way["aeroway"="aerodrome"]({bbox});',
    "power_plant":  'node["power"="plant"]({bbox});way["power"="plant"]({bbox});',
    "fire_station": 'node["amenity"="fire_station"]({bbox});way["amenity"="fire_station"]({bbox});',
    "police":       'node["amenity"="police"]({bbox});way["amenity"="police"]({bbox});',
    "government":   'node["amenity"="townhall"]({bbox});way["amenity"="townhall"]({bbox});'
                    'node["office"="government"]({bbox});way["office"="government"]({bbox});',
    "substation":   'node["power"="substation"]({bbox});way["power"="substation"]({bbox});',
    "water_works":  'node["man_made"="water_works"]({bbox});way["man_made"="water_works"]({bbox});',
}

POI_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "result", "stage2", "poi_cache"
)


def query_with_names(bbox, ftype, template, retries=3):
    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"
    body = template.replace("{bbox}", bbox_str)
    ql = f'[out:json][timeout:60];({body});out center tags;'

    for attempt in range(retries):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": ql}, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for el in data.get("elements", []):
                if el["type"] == "node":
                    lon, lat = el["lon"], el["lat"]
                elif "center" in el:
                    lon, lat = el["center"]["lon"], el["center"]["lat"]
                else:
                    continue
                name = el.get("tags", {}).get("name", "")
                results.append({"facility_type": ftype, "lon": lon, "lat": lat, "name": name})
            return results
        except Exception as e:
            print(f"  [retry {attempt+1}/{retries}] {ftype}: {e}")
            time.sleep(15 * (attempt + 1))
    return []


def main():
    cache_files = sorted(f for f in os.listdir(POI_CACHE_DIR) if f.endswith("_poi.csv"))

    for csv_file in cache_files:
        event_id = csv_file.replace("_poi.csv", "")
        csv_path = os.path.join(POI_CACHE_DIR, csv_file)
        df = pd.read_csv(csv_path)

        if "name" in df.columns and df["name"].notna().any() and (df["name"] != "").any():
            named = (df["name"] != "").sum()
            print(f"[SKIP] {event_id} already has {named}/{len(df)} names")
            continue

        # Derive bbox from existing POI coordinates (with padding)
        pad = 0.05
        south, north = df["lat"].min() - pad, df["lat"].max() + pad
        west, east = df["lon"].min() - pad, df["lon"].max() + pad
        bbox = (south, west, north, east)

        print(f"\n[{event_id}] bbox=({south:.2f},{west:.2f},{north:.2f},{east:.2f})")

        all_rows = []
        for ftype, template in OSM_QUERIES.items():
            rows = query_with_names(bbox, ftype, template)
            named = sum(1 for r in rows if r["name"])
            print(f"  {ftype:15s}: {len(rows):4d} found, {named} with name")
            all_rows.extend(rows)
            time.sleep(15)  # 15s between queries to avoid rate limiting

        new_df = pd.DataFrame(all_rows)
        new_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path} ({len(new_df)} rows)")


if __name__ == "__main__":
    main()
