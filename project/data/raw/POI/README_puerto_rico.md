# Puerto Rico POI Dataset Guide

This document explains the Puerto Rico infrastructure outputs in `POI/` and how teammates should use them.

## 1) Canonical Dataset

- Main file: `POI/puerto_rico_grid_infra.csv`
- Row count (current): `23007`
- Columns:
  - `id`
  - `category`
  - `type`
  - `lat`
  - `lon`
  - `name`
  - `city`
  - `tags` (JSON string)
  - `source` (`osm_grid`, `osm_retry`, `hifld`)

Use this file as the default input for analysis/modeling.

## 2) Related Files and Purpose

- `POI/puerto_rico_grid_infra_merged.csv`
  - Same merged result snapshot as main file.
- `POI/puerto_rico_critical_infra_2022.csv`
  - Compatibility copy for earlier naming expectations.
- `POI/puerto_rico_grid_retry_infra.csv`
  - Incremental records recovered from retrying failed chunks.
- `POI/puerto_rico_hifld_supplement.csv`
  - Authoritative supplement from HIFLD layers (power/hospitals/police/fire/wastewater).
- `POI/puerto_rico_grid_failed_chunks.csv`
  - Grid chunks that still failed after retries.
  - Columns: `chunk_id,south,west,north,east`
- `POI/puerto_rico_infra_map.html`
  - Interactive fast-cluster map for visual QA.

## 3) Source Layer Meaning

- `osm_grid`: baseline Overpass grid extraction.
- `osm_retry`: recovered records from failed-chunk retry/subdivision.
- `hifld`: ArcGIS HIFLD supplement records.

Always keep `source` for provenance-aware filtering.

## 4) Quick Start for Teammates

### A. Visual browse

Open:

- `POI/puerto_rico_infra_map.html`

This map is clustered and colored by `category`.

### B. Quick profiling in Python

```bash
python3 - <<'PY'
import pandas as pd, json
df = pd.read_csv("POI/puerto_rico_grid_infra.csv")
print("rows:", len(df))
print("source counts:")
print(df["source"].value_counts())
print("category counts:")
print(df["category"].value_counts())
PY
```

### C. Filter by source/category

```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_csv("POI/puerto_rico_grid_infra.csv")
sub = df[(df["source"]=="hifld") & (df["category"]=="healthcare")]
print(sub[["id","type","name","city","lat","lon"]].head(20).to_string(index=False))
PY
```

## 5) Regeneration / Update Flow

### Step 1: Retry + merge extraction

```bash
python3 POI/extract_puerto_rico_infrastructure.py
```

Expected outputs:

- `POI/puerto_rico_grid_retry_infra.csv`
- `POI/puerto_rico_hifld_supplement.csv`
- updated `POI/puerto_rico_grid_infra.csv`
- updated `POI/puerto_rico_grid_infra_merged.csv`
- updated `POI/puerto_rico_critical_infra_2022.csv`

### Step 2: Rebuild interactive map + generator profiling

```bash
python3 POI/visualize_puerto_rico_infra.py
```

Expected output:

- `POI/puerto_rico_infra_map.html`
- console breakdown of `generator:source` and `generator:method`.

## 6) Notes / Caveats

- `tags` is a JSON string, not expanded columns. Parse it when you need detailed attributes.
- Some OSM chunks can still fail due to API throttling (`429/504`) and may require reruns.
- `hifld` rows are retained together with OSM rows; do not drop `source` unless you intentionally de-duplicate across providers.
