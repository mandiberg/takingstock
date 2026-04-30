#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict


DEFAULT_ROOT = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/output_folder"
MARKERS = ("_cropped_", "_inpaint_")


def is_float_token(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def format_multiplier_for_cache_path(multiplier_values, places=4):
    def format_one(value):
        rounded = round(float(value), places)
        text_value = f"{rounded:.{places}f}".rstrip("0").rstrip(".")
        if "." not in text_value:
            text_value += ".0"
        return text_value

    return "_".join(format_one(value) for value in multiplier_values)


def split_suffix(name):
    marker_index = -1
    marker = None
    for candidate in MARKERS:
        candidate_index = name.rfind(candidate)
        if candidate_index > marker_index:
            marker_index = candidate_index
            marker = candidate

    if marker_index < 0:
        return None

    prefix = name[: marker_index + len(marker)]
    remainder = name[marker_index + len(marker) :]
    if not remainder:
        return None

    tokens = remainder.split("_")
    numeric_start = len(tokens)
    for index in range(len(tokens) - 1, -1, -1):
        if is_float_token(tokens[index]):
            numeric_start = index
            continue
        break

    if numeric_start == len(tokens):
        return None

    leading_tokens = tokens[:numeric_start]
    numeric_tokens = tokens[numeric_start:]
    if not numeric_tokens or not all(is_float_token(token) for token in numeric_tokens):
        return None

    return prefix, leading_tokens, numeric_tokens


def build_renames(root_dir):
    rename_map = {}
    skipped = []
    with os.scandir(root_dir) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue

            dir_name = entry.name
            suffix_parts = split_suffix(dir_name)
            if suffix_parts is None:
                continue

            prefix, leading_tokens, numeric_tokens = suffix_parts
            rounded_suffix = format_multiplier_for_cache_path(numeric_tokens)
            new_name = prefix
            if leading_tokens:
                new_name += "_".join(leading_tokens) + "_"
            new_name += rounded_suffix

            if new_name == dir_name:
                continue

            old_path = entry.path
            new_path = os.path.join(root_dir, new_name)
            rename_map[old_path] = new_path

            if os.path.exists(new_path) and old_path != new_path:
                skipped.append((old_path, new_path, "target already exists"))

    return rename_map, skipped


def detect_collisions(rename_map):
    reverse_map = defaultdict(list)
    for old_path, new_path in rename_map.items():
        reverse_map[new_path].append(old_path)

    collisions = []
    for new_path, old_paths in reverse_map.items():
        if len(old_paths) > 1:
            collisions.append((new_path, sorted(old_paths)))
    return collisions


def apply_renames(rename_map, blocked_targets):
    renamed = []
    skipped = []
    blocked_targets = set(blocked_targets)
    rename_items = sorted(rename_map.items(), key=lambda item: item[0].count(os.sep), reverse=True)

    for old_path, new_path in rename_items:
        if new_path in blocked_targets:
            skipped.append((old_path, new_path, "blocked by collision or existing target"))
            continue
        if not os.path.exists(old_path):
            skipped.append((old_path, new_path, "source no longer exists"))
            continue
        if os.path.exists(new_path):
            skipped.append((old_path, new_path, "target already exists"))
            continue

        os.rename(old_path, new_path)
        renamed.append((old_path, new_path))

    return renamed, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Rename cropped/inpaint cache directories to the rounded multiplier suffix format."
    )
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true", help="Perform the renames. Default is dry-run.")
    parser.add_argument("--dry-run", action="store_true", help="Compatibility flag; dry-run is already default.")
    parser.add_argument("--limit", type=int, default=20, help="Max examples to print for each section.")
    args = parser.parse_args()

    rename_map, existing_target_skips = build_renames(args.root)
    collisions = detect_collisions(rename_map)
    blocked_targets = {new_path for new_path, _old_paths in collisions}
    blocked_targets.update(new_path for _old_path, new_path, _reason in existing_target_skips)

    print(f"root: {args.root}")
    print(f"candidate_renames: {len(rename_map)}")
    print(f"existing_target_conflicts: {len(existing_target_skips)}")
    print(f"multi_source_collisions: {len(collisions)}")

    if rename_map:
        print("sample_renames:")
        for old_path, new_path in list(sorted(rename_map.items()))[: args.limit]:
            print(f"  {old_path} -> {new_path}")

    if existing_target_skips:
        print("sample_existing_target_conflicts:")
        for old_path, new_path, reason in existing_target_skips[: args.limit]:
            print(f"  {reason}: {old_path} -> {new_path}")

    if collisions:
        print("sample_multi_source_collisions:")
        for new_path, old_paths in collisions[: args.limit]:
            print(f"  target: {new_path}")
            for old_path in old_paths:
                print(f"    source: {old_path}")

    if not args.apply:
        print("dry_run_only: rerun with --apply to perform non-blocked renames")
        return

    renamed, skipped = apply_renames(rename_map, blocked_targets)
    print(f"renamed: {len(renamed)}")
    print(f"skipped: {len(skipped)}")

    if renamed:
        print("sample_applied_renames:")
        for old_path, new_path in renamed[: args.limit]:
            print(f"  {old_path} -> {new_path}")

    if skipped:
        print("sample_skipped:")
        for old_path, new_path, reason in skipped[: args.limit]:
            print(f"  {reason}: {old_path} -> {new_path}")


if __name__ == "__main__":
    main()