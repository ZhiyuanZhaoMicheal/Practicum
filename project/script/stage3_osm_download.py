"""
stage3_osm_download.py
======================
Download critical facility POIs from OpenStreetMap for Stage 3 events.
Uses Overpass API (free, no key needed).

Usage:
    python stage3_osm_download.py                # all events
    python stage3_osm_download.py --events uri_houston zeta_atlanta

Output:
    data/result/stage3/poi_cache/<event_id>_poi.csv
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd

# Import event configs (no ee dependency)
from stage3_events import EVENTS

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM tags → facility types (same as Stage 2)
FACILITY_QUERIES = {
    "hospital": [
        'nwr["amenity"="hospital"]',
    ],
    "aerodrome": [
        'nwr["aeroway"="aerodrome"]',
    ],
    "fire_station": [
        'nwr["amenity"="fire_station"]',
    ],
    "police": [
        'nwr["amenity"="police"]',
    ],
    "power_plant": [
        'nwr["power"="plant"]',
    ],
    "government": [
        'nwr["office"="government"]',
        'nwr["amenity"="townhall"]',
        'nwr["amenity"="courthouse"]',
    ],
    "substation": [
        'nwr["power"="substation"]',
    ],
    "water_works": [
        'nwr["man_made"="water_works"]',
        'nwr["man_made"="wastewater_plant"]',
    ],
}


def query_overpass(bbox, facility_type, tags):
    """Query Overpass API for a facility type within a bounding box."""
    south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]
    bbox_str = f"{south},{west},{north},{east}"

    union_parts = "".join(f"{tag}({bbox_str});" for tag in tags)
    query = f"""
    [out:json][timeout:60];
    ({union_parts});
    out center;
    """

    data = {"elements": []}
    for attempt in range(5):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=120)
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = 15 * (attempt + 1)
                print(f"    Server error {resp.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt < 4:
                wait = 15 * (attempt + 1)
                print(f"    Error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [ERROR] Overpass query failed after 5 attempts: {e}")
                return []

    results = []
    for el in data.get("elements", []):
        tags_data = el.get("tags", {})
        name = tags_data.get("name", "")

        if el["type"] == "node":
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue

        results.append({
            "name": name if name else f"{facility_type} facility",
            "facility_type": facility_type,
            "lat": lat,
            "lon": lon,
            "osm_id": el["id"],
            "osm_type": el["type"],
        })

    return results


def download_pois_for_event(event_id):
    """Download all facility POIs for one event."""
    cfg = EVENTS[event_id]
    bbox = cfg["bounds"]
    print(f"\n  {event_id} — {cfg['name']}")
    print(f"    BBox: {bbox}")

    all_pois = []
    for fac_type, tags in FACILITY_QUERIES.items():
        pois = query_overpass(bbox, fac_type, tags)
        print(f"    {fac_type:<15} {len(pois):>3} POIs")
        all_pois.extend(pois)
        time.sleep(3)  # be polite to Overpass

    return pd.DataFrame(all_pois)


def main():
    parser = argparse.ArgumentParser(description="Download OSM POIs for Stage 3")
    parser.add_argument("--events", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    event_ids = args.events or list(EVENTS.keys())

    if args.output_dir:
        out_dir = args.output_dir
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        out_dir = os.path.join(project_root, "data", "result", "stage3", "poi_cache")
    os.makedirs(out_dir, exist_ok=True)

    print(f"OSM POI Download — {len(event_ids)} events")
    print(f"Output: {out_dir}")

    summary = []
    for eid in event_ids:
        if eid not in EVENTS:
            print(f"  [WARN] Unknown event: {eid}")
            continue

        poi_df = download_pois_for_event(eid)

        # De-duplicate by osm_id
        before = len(poi_df)
        if not poi_df.empty:
            poi_df = poi_df.drop_duplicates(subset=["osm_id"])

        out_path = os.path.join(out_dir, f"{EVENTS[eid]['drive_root']}_poi.csv")
        poi_df.to_csv(out_path, index=False)
        print(f"    Saved {len(poi_df)} POIs → {out_path}")
        summary.append({"event": eid, "pois": len(poi_df)})

    print(f"\n{'='*40}")
    print(f"{'Event':<22} {'POIs':>6}")
    print("-" * 30)
    for s in summary:
        print(f"  {s['event']:<22} {s['pois']:>4}")
    total = sum(s["pois"] for s in summary)
    print(f"  {'TOTAL':<22} {total:>4}")


if __name__ == "__main__":
    main()
