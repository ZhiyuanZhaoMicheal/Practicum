#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import json


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "project" / "modeling" / "output"
CONFIG_EVENTS = ROOT / "project" / "modeling" / "config" / "events_6.json"

REMOTE_REF = "teammate/main"
MANIFEST_CSV = OUTPUT_DIR / "teammate_reuse_manifest.csv"
GAP_CSV = OUTPUT_DIR / "teammate_reuse_gap.csv"
SYNC_LOG_CSV = OUTPUT_DIR / "teammate_reuse_sync_log.csv"
GATE_CSV = OUTPUT_DIR / "input_data_gate_report.csv"
DOWNLOAD_TRIGGER_CSV = OUTPUT_DIR / "download_trigger_plan.csv"

TS_FMT = "%Y-%m-%d %H:%M:%S UTC"


@dataclass
class ManifestRow:
    path: str
    category: str
    event_id: str
    source_remote: str


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(TS_FMT)


def run_git(*args: str) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return res.stdout.decode("utf-8", errors="replace")


def event_from_processed_path(path: str) -> str:
    key_map = {
        "Maria-VNP46A2": "maria_sanjuan",
        "Michael-VNP46A2": "michael_panamacity",
        "Earthquake-VNP46A2": "earthquake_sanjuan",
        "Ida_NewOrleans-VNP46A2": "ida_neworleans",
        "Laura_LakeCharles-VNP46A2": "laura_lakecharles",
        "Irma_Miami-VNP46A2": "irma_miami",
    }
    for k, eid in key_map.items():
        if f"/{k}-pre/" in path or f"/{k}-post/" in path:
            return eid
    return "unknown"


def event_from_poi_path(path: str) -> str:
    m = re.search(r"/([a-z_]+)_critical_infra_poi\.csv$", path)
    return m.group(1) if m else "unknown"


def event_from_cloud_path(path: str) -> str:
    name = Path(path).name.lower()
    token_map = {
        "maria_sanjuan": ["maria_sanjuan", "hurricane_maria_sanjuan"],
        "michael_panamacity": ["michael_fl", "michael_panamacity"],
        "earthquake_sanjuan": ["earthquake_sanjuan"],
        "ida_neworleans": ["ida_neworleans", "hurricane_ida_neworleans"],
        "laura_lakecharles": ["laura_lakecharles"],
        "irma_miami": ["irma_miami"],
    }
    for eid, tokens in token_map.items():
        if any(tok in name for tok in tokens):
            return eid
    return "unknown"


def collect_manifest_rows() -> List[ManifestRow]:
    all_files = run_git("ls-tree", "-r", "--name-only", REMOTE_REF).splitlines()
    rows: List[ManifestRow] = []

    tif_re = re.compile(
        r"^project/data/processed/"
        r"(Maria|Michael|Earthquake|Ida_NewOrleans|Laura_LakeCharles|Irma_Miami)"
        r"-VNP46A2-(pre|post)/.+\.tif$"
    )
    poi_re = re.compile(r"^project/result/[a-z_]+_critical_infra_poi\.csv$")
    cloud_re = re.compile(r"^project/script/.*cloud_screening.*\.csv$", re.IGNORECASE)

    for path in all_files:
        if tif_re.match(path):
            if "composite" in path.lower():
                continue
            rows.append(
                ManifestRow(
                    path=path,
                    category="ntl_tif",
                    event_id=event_from_processed_path(path),
                    source_remote=REMOTE_REF,
                )
            )
            continue
        if poi_re.match(path):
            rows.append(
                ManifestRow(
                    path=path,
                    category="poi",
                    event_id=event_from_poi_path(path),
                    source_remote=REMOTE_REF,
                )
            )
            continue
        if cloud_re.match(path):
            rows.append(
                ManifestRow(
                    path=path,
                    category="cloud_screening",
                    event_id=event_from_cloud_path(path),
                    source_remote=REMOTE_REF,
                )
            )
            continue
        if path == "project/script/multi_event_ntl_download.ipynb":
            rows.append(
                ManifestRow(
                    path=path,
                    category="download_script",
                    event_id="all",
                    source_remote=REMOTE_REF,
                )
            )

    rows.sort(key=lambda r: (r.category, r.event_id, r.path))
    return rows


def write_manifest(rows: List[ManifestRow]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "category", "event_id", "source_remote"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "path": r.path,
                    "category": r.category,
                    "event_id": r.event_id,
                    "source_remote": r.source_remote,
                }
            )


def write_gap(rows: List[ManifestRow]) -> List[Dict[str, object]]:
    gap_rows: List[Dict[str, object]] = []
    for r in rows:
        target = ROOT / r.path
        exists = target.exists()
        gap_rows.append(
            {
                "path": r.path,
                "category": r.category,
                "event_id": r.event_id,
                "exists_local": int(exists),
                "source_remote": r.source_remote,
                "action": "keep_local" if exists else "reuse_from_teammate",
            }
        )

    with GAP_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["path", "category", "event_id", "exists_local", "source_remote", "action"],
        )
        w.writeheader()
        w.writerows(gap_rows)
    return gap_rows


def export_blob_to_path(remote_path: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        proc = subprocess.run(
            ["git", "show", f"{REMOTE_REF}:{remote_path}"],
            cwd=ROOT,
            check=True,
            stdout=fh,
            stderr=subprocess.PIPE,
        )
    return proc.returncode


def sync_missing(gap_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    missing_rows = [r for r in gap_rows if int(r["exists_local"]) == 0]
    logs: List[Dict[str, object]] = []
    ts = utc_now()

    if not missing_rows:
        logs.append(
            {
                "timestamp": ts,
                "path": "",
                "category": "summary",
                "event_id": "all",
                "status": "no_missing_files",
                "size_bytes": 0,
                "source_remote": REMOTE_REF,
            }
        )
        return logs

    with tempfile.TemporaryDirectory(prefix="teammate_sync_stage_") as td:
        stage_root = Path(td)
        for row in missing_rows:
            rel = row["path"]
            staged_path = stage_root / rel
            target = ROOT / rel
            try:
                export_blob_to_path(rel, staged_path)
                size = staged_path.stat().st_size if staged_path.exists() else 0
                if target.exists():
                    status = "skipped_exists"
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(staged_path, target)
                    status = "copied"
                logs.append(
                    {
                        "timestamp": ts,
                        "path": rel,
                        "category": row["category"],
                        "event_id": row["event_id"],
                        "status": status,
                        "size_bytes": size,
                        "source_remote": REMOTE_REF,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logs.append(
                    {
                        "timestamp": ts,
                        "path": rel,
                        "category": row["category"],
                        "event_id": row["event_id"],
                        "status": f"failed:{exc}",
                        "size_bytes": 0,
                        "source_remote": REMOTE_REF,
                    }
                )

    logs.sort(key=lambda r: (r["status"], r["category"], r["event_id"], r["path"]))
    return logs


def write_sync_logs(log_rows: List[Dict[str, object]]) -> None:
    with SYNC_LOG_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "path",
                "category",
                "event_id",
                "status",
                "size_bytes",
                "source_remote",
            ],
        )
        w.writeheader()
        w.writerows(log_rows)


def list_daily_tifs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*.tif") if "composite" not in p.name.lower()])


def find_cloud_csv(event_id: str) -> Optional[Path]:
    script_dir = ROOT / "project" / "script"
    if not script_dir.exists():
        return None

    candidates_map = {
        "maria_sanjuan": [
            "maria_sanjuan_cloud_screening.csv",
            "hurricane_maria_sanjuan_cloud_screening.csv",
        ],
        "michael_panamacity": [
            "michael_panamacity_cloud_screening.csv",
            "Michael_FL_cloud_screening.csv",
        ],
        "earthquake_sanjuan": [
            "earthquake_sanjuan_cloud_screening.csv",
            "Earthquake_sanjuan_cloud_screening.csv",
        ],
        "ida_neworleans": [
            "ida_neworleans_cloud_screening.csv",
            "hurricane_ida_neworleans_cloud_screening.csv",
        ],
        "laura_lakecharles": [
            "laura_lakecharles_cloud_screening.csv",
        ],
        "irma_miami": [
            "irma_miami_cloud_screening.csv",
        ],
    }
    for name in candidates_map.get(event_id, []):
        p = script_dir / name
        if p.exists():
            return p
    return None


def write_data_gate() -> List[Dict[str, object]]:
    cfg = json.loads(CONFIG_EVENTS.read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    missing_events: List[str] = []

    for event_id, event_cfg in cfg.items():
        pre_dir = ROOT / event_cfg["pre_dir"]
        post_dir = ROOT / event_cfg["post_dir"]
        poi_csv = ROOT / event_cfg["poi_csv"]
        cloud_csv = find_cloud_csv(event_id)

        pre_count = len(list_daily_tifs(pre_dir))
        post_count = len(list_daily_tifs(post_dir))
        poi_ok = poi_csv.exists()
        cloud_ok = cloud_csv is not None and cloud_csv.exists()

        pre_ok = pre_count >= 7
        post_ok = post_count >= 14
        gate_pass = pre_ok and post_ok and poi_ok

        missing = []
        if not pre_ok:
            missing.append("pre_tif")
        if not post_ok:
            missing.append("post_tif")
        if not poi_ok:
            missing.append("poi_csv")
        if not gate_pass:
            missing_events.append(event_id)

        rows.append(
            {
                "event_id": event_id,
                "pre_count": pre_count,
                "post_count": post_count,
                "poi_exists": int(poi_ok),
                "cloud_exists": int(cloud_ok),
                "pre_threshold": 7,
                "post_threshold": 14,
                "pre_ok": int(pre_ok),
                "post_ok": int(post_ok),
                "poi_ok": int(poi_ok),
                "gate_pass": int(gate_pass),
                "missing_items": ",".join(missing),
            }
        )

    with GATE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "pre_count",
                "post_count",
                "poi_exists",
                "cloud_exists",
                "pre_threshold",
                "post_threshold",
                "pre_ok",
                "post_ok",
                "poi_ok",
                "gate_pass",
                "missing_items",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    with DOWNLOAD_TRIGGER_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["event_id", "action", "reason", "script_path", "notes"],
        )
        w.writeheader()
        if missing_events:
            for eid in missing_events:
                w.writerow(
                    {
                        "event_id": eid,
                        "action": "trigger_download",
                        "reason": "data_gate_failed",
                        "script_path": "project/script/multi_event_ntl_download.ipynb",
                        "notes": "Run only for missing events in SELECTED_EVENTS; keep reuse-first policy.",
                    }
                )
        else:
            w.writerow(
                {
                    "event_id": "all",
                    "action": "no_action",
                    "reason": "all_events_passed_gate",
                    "script_path": "",
                    "notes": "No missing event data after teammate reuse sync.",
                }
            )

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = collect_manifest_rows()
    write_manifest(manifest_rows)
    gap_rows = write_gap(manifest_rows)
    sync_rows = sync_missing(gap_rows)
    write_sync_logs(sync_rows)
    gate_rows = write_data_gate()

    n_manifest = len(manifest_rows)
    n_missing = sum(1 for r in gap_rows if int(r["exists_local"]) == 0)
    n_copied = sum(1 for r in sync_rows if r.get("status") == "copied")
    n_gate_ok = sum(1 for r in gate_rows if int(r["gate_pass"]) == 1)

    print(f"Manifest rows: {n_manifest}")
    print(f"Missing before sync: {n_missing}")
    print(f"Copied from teammate/main: {n_copied}")
    print(f"Data gate pass events: {n_gate_ok}/{len(gate_rows)}")
    print(f"Wrote: {MANIFEST_CSV.relative_to(ROOT)}")
    print(f"Wrote: {GAP_CSV.relative_to(ROOT)}")
    print(f"Wrote: {SYNC_LOG_CSV.relative_to(ROOT)}")
    print(f"Wrote: {GATE_CSV.relative_to(ROOT)}")
    print(f"Wrote: {DOWNLOAD_TRIGGER_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
