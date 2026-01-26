"""
Texas Critical Infrastructure POI Extractor
============================================
Extracts historical POI data for critical infrastructure in Houston and Dallas, Texas.
Uses OSMnx to query Overpass API with temporal targeting (2022-06-01).

Output: CSV file with Point coordinates (Lat/Lon) and metadata attributes.
"""

import os
import sys
import warnings
from datetime import datetime

import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Temporal target - snapshot from 2022
HISTORICAL_DATE = "2022-06-01T00:00:00Z"

# Spatial scope
LOCATIONS = [
    "Houston, Texas, USA",
    "Dallas, Texas, USA"
]

# Target infrastructure tags (OSM key-value pairs)
INFRASTRUCTURE_TAGS = {
    "power": ["plant", "generator", "substation"],
    "amenity": ["hospital", "fire_station", "police"],
    "telecom": ["data_center"],
    "man_made": ["water_works", "wastewater_plant"],
    "aeroway": ["aerodrome"]
}

# Output file
OUTPUT_FILE = "texas_critical_infra_points_2022.csv"
OUTPUT_GEOJSON = "texas_critical_infra_points_2022.geojson"

# Texas coordinate bounds for sanity check
TEXAS_LAT_BOUNDS = (25.8, 36.5)  # Latitude range
TEXAS_LON_BOUNDS = (-106.6, -93.5)  # Longitude range

# More specific bounds for Houston/Dallas area
HOUSTON_DALLAS_LAT_BOUNDS = (29.0, 33.5)
HOUSTON_DALLAS_LON_BOUNDS = (-97.5, -94.5)

# Columns to keep in final output
KEEP_COLUMNS = [
    'name', 'power', 'amenity', 'man_made', 'telecom', 'aeroway',
    'generator:source', 'addr:full', 'addr:street', 'addr:city',
    'operator', 'capacity', 'voltage', 'lat', 'lon', 'city_source'
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def configure_osmnx_historical():
    """Configure OSMnx for historical data extraction."""
    ox.settings.overpass_settings = f'[out:json][timeout:180][date:"{HISTORICAL_DATE}"]'
    ox.settings.timeout = 180
    ox.settings.log_console = False
    print(f"[CONFIG] Overpass API configured for historical date: {HISTORICAL_DATE}")


def configure_osmnx_current():
    """Configure OSMnx for current data (fallback)."""
    ox.settings.overpass_settings = '[out:json][timeout:180]'
    ox.settings.timeout = 180
    ox.settings.log_console = False
    print("[CONFIG] Overpass API configured for CURRENT data (fallback mode)")


def build_tags_dict():
    """Build the tags dictionary for OSMnx query."""
    tags = {}
    for key, values in INFRASTRUCTURE_TAGS.items():
        if isinstance(values, list):
            tags[key] = values
        else:
            tags[key] = [values]
    return tags


def fetch_infrastructure_data(location, tags, use_historical=True):
    """
    Fetch infrastructure data from OSM for a given location.

    Args:
        location: Place name string
        tags: Dictionary of OSM tags to query
        use_historical: Whether to use historical date setting

    Returns:
        GeoDataFrame with fetched features, or None if failed
    """
    try:
        print(f"[FETCH] Querying {location}...")
        gdf = ox.features_from_place(location, tags=tags)

        if gdf is None or len(gdf) == 0:
            print(f"[WARN] No data returned for {location}")
            return None

        print(f"[FETCH] Retrieved {len(gdf)} features from {location}")
        return gdf

    except Exception as e:
        print(f"[ERROR] Failed to fetch {location}: {str(e)}")
        return None


def convert_to_centroids(gdf):
    """
    Convert all geometries to Point centroids.

    Args:
        gdf: GeoDataFrame with mixed geometries

    Returns:
        GeoDataFrame with Point geometries only
    """
    if gdf is None or len(gdf) == 0:
        return None

    # Ensure CRS is set (default to WGS84)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Convert to centroids
    gdf = gdf.copy()
    gdf['geometry'] = gdf['geometry'].centroid

    # Extract lat/lon from centroid points
    gdf['lon'] = gdf['geometry'].x
    gdf['lat'] = gdf['geometry'].y

    return gdf


def clean_and_filter(gdf, city_name):
    """
    Clean attributes and filter out rows without meaningful tags.

    Args:
        gdf: GeoDataFrame to clean
        city_name: Source city name for tracking

    Returns:
        Cleaned GeoDataFrame
    """
    if gdf is None or len(gdf) == 0:
        return None

    gdf = gdf.copy()

    # Add source city
    gdf['city_source'] = city_name

    # Reset index to get osmid as a column
    gdf = gdf.reset_index()

    # Define the infrastructure tag columns
    tag_columns = ['power', 'amenity', 'man_made', 'telecom', 'aeroway']

    # Filter: keep only rows that have at least one non-null tag from our categories
    existing_tag_cols = [col for col in tag_columns if col in gdf.columns]

    if existing_tag_cols:
        # Create mask for rows with at least one valid tag
        mask = gdf[existing_tag_cols].notna().any(axis=1)
        gdf = gdf[mask]

    # Select and reorder columns
    available_cols = [col for col in KEEP_COLUMNS if col in gdf.columns]

    # Add osmid if available
    if 'osmid' in gdf.columns:
        available_cols = ['osmid'] + available_cols
    if 'element_type' in gdf.columns:
        available_cols = ['element_type'] + available_cols

    gdf = gdf[available_cols]

    print(f"[CLEAN] {city_name}: {len(gdf)} records after filtering")
    return gdf


def verify_data(df):
    """
    Self-verification protocol with multiple checks.

    Args:
        df: Final DataFrame to verify

    Returns:
        Tuple of (success: bool, report: dict)
    """
    report = {
        "check_volume": False,
        "check_coordinates": False,
        "check_location_sanity": False,
        "check_file": False,
        "total_records": 0,
        "errors": []
    }

    if df is None or len(df) == 0:
        report["errors"].append("DataFrame is empty or None")
        return False, report

    report["total_records"] = len(df)

    # Check 1: Volume check (expect > 100 records)
    if len(df) > 100:
        report["check_volume"] = True
        print(f"[CHECK 1] PASS: Volume check - {len(df)} records (> 100)")
    else:
        report["errors"].append(f"Volume check failed: only {len(df)} records")
        print(f"[CHECK 1] FAIL: Volume check - only {len(df)} records")

    # Check 2: Coordinate columns exist and contain valid floats
    if 'lat' in df.columns and 'lon' in df.columns:
        lat_valid = df['lat'].notna().sum()
        lon_valid = df['lon'].notna().sum()

        if lat_valid > 0 and lon_valid > 0:
            report["check_coordinates"] = True
            print(f"[CHECK 2] PASS: Coordinate check - {lat_valid} valid lat, {lon_valid} valid lon")
        else:
            report["errors"].append("No valid coordinates found")
            print("[CHECK 2] FAIL: No valid coordinates")
    else:
        report["errors"].append("lat/lon columns missing")
        print("[CHECK 2] FAIL: lat/lon columns missing")

    # Check 3: Location sanity check (coordinates within Texas)
    if 'lat' in df.columns and 'lon' in df.columns:
        mean_lat = df['lat'].mean()
        mean_lon = df['lon'].mean()

        lat_ok = HOUSTON_DALLAS_LAT_BOUNDS[0] <= mean_lat <= HOUSTON_DALLAS_LAT_BOUNDS[1]
        lon_ok = HOUSTON_DALLAS_LON_BOUNDS[0] <= mean_lon <= HOUSTON_DALLAS_LON_BOUNDS[1]

        if lat_ok and lon_ok:
            report["check_location_sanity"] = True
            print(f"[CHECK 3] PASS: Location sanity - Mean coords ({mean_lat:.4f}, {mean_lon:.4f}) within Texas bounds")
        else:
            report["errors"].append(f"Mean coords ({mean_lat:.4f}, {mean_lon:.4f}) outside expected bounds")
            print(f"[CHECK 3] FAIL: Mean coords ({mean_lat:.4f}, {mean_lon:.4f}) outside expected bounds")

    # Determine overall success
    success = report["check_volume"] and report["check_coordinates"] and report["check_location_sanity"]

    return success, report


def save_output(df, csv_path, geojson_path=None):
    """
    Save DataFrame to CSV and optionally GeoJSON.

    Returns:
        bool: Success status
    """
    try:
        # Save CSV
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"[SAVE] CSV saved to: {csv_path}")

        # Verify file exists
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path)
            print(f"[SAVE] File verified: {file_size:,} bytes")

            # Optionally save GeoJSON
            if geojson_path and 'lat' in df.columns and 'lon' in df.columns:
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df['lon'], df['lat']),
                    crs="EPSG:4326"
                )
                gdf.to_file(geojson_path, driver='GeoJSON')
                print(f"[SAVE] GeoJSON saved to: {geojson_path}")

            return True
        else:
            print(f"[ERROR] File not found after save: {csv_path}")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to save: {str(e)}")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("TEXAS CRITICAL INFRASTRUCTURE POI EXTRACTOR")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Build tags dictionary
    tags = build_tags_dict()
    print(f"[INFO] Target tags: {list(tags.keys())}")
    print()

    # Attempt historical data extraction first
    use_historical = True
    all_data = []

    # Configure for historical query
    configure_osmnx_historical()
    print()

    for location in LOCATIONS:
        gdf = fetch_infrastructure_data(location, tags, use_historical=True)

        if gdf is not None:
            # Convert to centroids
            gdf = convert_to_centroids(gdf)

            # Clean and filter
            city_name = location.split(",")[0]
            gdf = clean_and_filter(gdf, city_name)

            if gdf is not None and len(gdf) > 0:
                all_data.append(gdf)

    # If historical query returned too little data, try current data as fallback
    total_historical = sum(len(d) for d in all_data) if all_data else 0

    if total_historical < 50:
        print()
        print("[FALLBACK] Historical query returned limited data. Trying current data...")
        configure_osmnx_current()
        print()

        all_data = []  # Reset
        use_historical = False

        for location in LOCATIONS:
            gdf = fetch_infrastructure_data(location, tags, use_historical=False)

            if gdf is not None:
                gdf = convert_to_centroids(gdf)
                city_name = location.split(",")[0]
                gdf = clean_and_filter(gdf, city_name)

                if gdf is not None and len(gdf) > 0:
                    all_data.append(gdf)

    # Consolidate all data
    print()
    print("-" * 70)

    if not all_data:
        print("[FAILURE REPORT]")
        print("No data could be extracted from any location.")
        print("Possible causes:")
        print("  - Network connectivity issues")
        print("  - Overpass API rate limiting")
        print("  - Invalid location names")
        return False

    # Merge all datasets
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"[MERGE] Combined dataset: {len(final_df)} total records")

    # Remove duplicates based on coordinates
    before_dedup = len(final_df)
    final_df = final_df.drop_duplicates(subset=['lat', 'lon'], keep='first')
    print(f"[DEDUP] Removed {before_dedup - len(final_df)} duplicate coordinates")
    print(f"[FINAL] Unique records: {len(final_df)}")
    print()

    # Self-verification
    print("-" * 70)
    print("SELF-VERIFICATION PROTOCOL")
    print("-" * 70)

    success, report = verify_data(final_df)

    # Check 4: Save and verify file creation
    print()
    output_csv = os.path.join(os.getcwd(), OUTPUT_FILE)
    output_geojson = os.path.join(os.getcwd(), OUTPUT_GEOJSON)

    file_saved = save_output(final_df, output_csv, output_geojson)
    report["check_file"] = file_saved

    if file_saved:
        print(f"[CHECK 4] PASS: File creation verified")
    else:
        print(f"[CHECK 4] FAIL: File creation failed")
        report["errors"].append("File creation failed")

    # Final status
    print()
    print("=" * 70)

    overall_success = success and file_saved

    if overall_success:
        print("SUCCESS: Data extracted successfully!")
        print(f"  Total Records: {len(final_df)}")
        print(f"  Data Source: {'Historical (2022-06-01)' if use_historical else 'Current (fallback)'}")
        print(f"  CSV File: {OUTPUT_FILE}")
        print(f"  GeoJSON File: {OUTPUT_GEOJSON}")

        # Print summary by category
        print()
        print("CATEGORY BREAKDOWN:")
        for col in ['power', 'amenity', 'man_made', 'telecom', 'aeroway']:
            if col in final_df.columns:
                count = final_df[col].notna().sum()
                if count > 0:
                    values = final_df[col].dropna().value_counts().head(5).to_dict()
                    print(f"  {col}: {count} records - {values}")
    else:
        print("FAILURE REPORT")
        print(f"  Checks passed: {sum([report['check_volume'], report['check_coordinates'], report['check_location_sanity'], report['check_file']])}/4")
        print(f"  Errors:")
        for error in report["errors"]:
            print(f"    - {error}")

    print("=" * 70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return overall_success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[ABORT] Script interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
