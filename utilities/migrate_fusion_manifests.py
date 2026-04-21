#!/usr/bin/env python3
"""Backfill fusion manifest files for existing CSV folders.

This script builds/updates fusion manifests for folders that contain fusion CSVs,
without requiring DB access. It is strict by default: unknown CSV naming patterns
raise an error so bad data does not silently pass through.
"""

import argparse
import csv
import json
import os
import re
from typing import Dict, List, Tuple

MANIFEST_FILE = "fusion_manifest.json"
CONTRACT_VERSION = 1

FILENAME_PATTERNS = [
    (re.compile(r"^Keywords_(\d+)\.csv$"), "keyword", "keyword_id"),
    (re.compile(r"^Detections_(\d+)\.csv$"), "detection", "class_id"),
    (re.compile(r"^Clusters_(\d+)\.csv$"), "cluster", "cluster_id"),
    (re.compile(r"^topic_(\d+)\.csv$"), "topic", "topic_id"),
]


def infer_entity_from_filename(file_name: str) -> Tuple[str, int, str]:
    for pattern, entity_type, mode_id in FILENAME_PATTERNS:
        match = pattern.match(file_name)
        if match:
            return entity_type, int(match.group(1)), mode_id
    raise ValueError(f"Unsupported fusion CSV filename pattern: {file_name}")


def is_fusion_csv_filename(file_name: str) -> bool:
    for pattern, _, _ in FILENAME_PATTERNS:
        if pattern.match(file_name):
            return True
    return False


def infer_available_hsv_bins(columns: List[str]) -> List[int]:
    bins: List[int] = []
    for col in columns:
        if not col.startswith("hsv_"):
            continue
        suffix = col[4:]
        if suffix.isdigit():
            bins.append(int(suffix))
    return sorted(set(bins))


def infer_mode(folder_files: List[str]) -> str:
    if any(name.startswith("Keywords_") for name in folder_files):
        return "Keywords"
    if any(name.startswith("Detections_") for name in folder_files):
        return "Detections"
    if any(name.startswith("Clusters_") for name in folder_files):
        return "Clusters"
    if any(name.startswith("topic_") for name in folder_files):
        return "Topics"
    return "Unknown"


def build_file_entry(csv_path: str, hsv_preset_name: str) -> Dict:
    file_name = os.path.basename(csv_path)
    entity_type, entity_id, _ = infer_entity_from_filename(file_name)
    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        columns = next(reader, [])

    return {
        "entity_type": entity_type,
        "entity_id": int(entity_id),
        "csv_schema_version": 1,
        "has_hsv_summary_rows": "hsv_3_to_22_sum" in columns,
        "available_hsv_bins": infer_available_hsv_bins(columns),
        "hsv_preset_name": hsv_preset_name,
    }


def migrate_folder(folder_path: str, manifest_file: str, hsv_preset_name: str, strict: bool) -> bool:
    all_files = sorted(os.listdir(folder_path))
    csv_files = [f for f in all_files if f.lower().endswith(".csv")]
    fusion_csv_files = [f for f in csv_files if is_fusion_csv_filename(f)]
    if not fusion_csv_files:
        return False

    manifest_path = os.path.join(folder_path, manifest_file)
    mode = infer_mode(fusion_csv_files)

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "generator": "utilities/migrate_fusion_manifests.py",
        "mode": mode,
        "mode_id": None,
        "cluster_type": None,
        "files": {},
    }

    discovered_mode_ids = set()
    for csv_name in fusion_csv_files:
        csv_path = os.path.join(folder_path, csv_name)
        try:
            entry = build_file_entry(csv_path, hsv_preset_name)
            _, _, mode_id = infer_entity_from_filename(csv_name)
            discovered_mode_ids.add(mode_id)
            manifest["files"][csv_name] = entry
        except Exception as exc:
            if strict:
                raise
            print(f"Skipping {csv_path}: {exc}")

    if not manifest["files"]:
        if strict:
            raise ValueError(f"No manifest entries produced for folder: {folder_path}")
        return False

    if len(discovered_mode_ids) == 1:
        manifest["mode_id"] = list(discovered_mode_ids)[0]

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Wrote {manifest_path} with {len(manifest['files'])} file entries")
    return True


def walk_and_migrate(root: str, manifest_file: str, hsv_preset_name: str, strict: bool) -> None:
    migrated = 0
    for current_root, _, _ in os.walk(root):
        changed = migrate_folder(current_root, manifest_file, hsv_preset_name, strict)
        if changed:
            migrated += 1

    if migrated == 0:
        print(f"No CSV folders found under {root}")
    else:
        print(f"Migration complete: {migrated} folder(s) updated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill fusion manifests for existing CSV folders")
    parser.add_argument("--root", required=True, help="Root folder to scan (e.g., utilities/data)")
    parser.add_argument("--manifest-file", default=MANIFEST_FILE, help="Manifest filename")
    parser.add_argument("--hsv-preset", default="background_default", help="Default HSV preset name")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail on unknown CSV pattern or malformed files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    walk_and_migrate(
        root=args.root,
        manifest_file=args.manifest_file,
        hsv_preset_name=args.hsv_preset,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
