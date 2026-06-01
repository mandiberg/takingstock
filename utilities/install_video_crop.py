#!/usr/bin/env python3
"""
Normalize video dimensions in an installation folder by cropping videos that
are exactly 2px wider or taller than a paired dimension down to match the
smaller size, then updating installation.csv accordingly.

Usage:
    python install_video_crop.py <folder> [--dry-run]
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


def load_csv(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    return rows, fieldnames


def find_2px_pairs(rows):
    """
    Scan unique (width, height) pairs in the CSV and return a mapping of
    larger_dim -> smaller_dim for every pair that differs by exactly 2px on
    one axis while the other axis is identical.
    """
    dims = set()
    for row in rows:
        dims.add((int(row["width"]), int(row["height"])))

    pairs = {}  # (larger_w, larger_h) -> (target_w, target_h)
    dims_list = sorted(dims)
    for i, (w1, h1) in enumerate(dims_list):
        for w2, h2 in dims_list[i + 1 :]:
            if w1 == w2 and abs(h1 - h2) == 2:
                larger = (w1, h1) if h1 > h2 else (w2, h2)
                smaller = (w1, h1) if h1 < h2 else (w2, h2)
                pairs[larger] = smaller
            elif h1 == h2 and abs(w1 - w2) == 2:
                larger = (w1, h1) if w1 > w2 else (w2, h2)
                smaller = (w1, h1) if w1 < w2 else (w2, h2)
                pairs[larger] = smaller
    return pairs


def crop_video(input_path, output_path, src_w, src_h, target_w, target_h):
    """
    Crop input_path to target dimensions, centering the crop window, and
    write the result to output_path.  Audio is stream-copied unchanged.
    Returns (success: bool, stderr: str).
    """
    x_offset = (src_w - target_w) // 2
    y_offset = (src_h - target_h) // 2

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"crop={target_w}:{target_h}:{x_offset}:{y_offset}",
        "-c:a", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


def main():
    parser = argparse.ArgumentParser(
        description="Crop videos with 2px dimension mismatches to normalize them."
    )
    parser.add_argument(
        "folder",
        help="Folder containing installation.csv and video files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying any files",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    csv_path = folder / "installation.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    rows, fieldnames = load_csv(csv_path)

    pairs = find_2px_pairs(rows)
    if not pairs:
        print("No 2px dimension pairs found. Nothing to do.")
        return

    pair_counts = {}
    for row in rows:
        w, h = int(row["width"]), int(row["height"])
        if (w, h) in pairs:
            pair_counts[(w, h)] = pair_counts.get((w, h), 0) + 1
    total_to_crop = sum(pair_counts.values())

    print(f"Found {total_to_crop} file(s) to crop across {len(pairs)} dimension pair(s):")
    for larger, smaller in sorted(pairs.items()):
        count = pair_counts.get(larger, 0)
        print(f"  {larger[0]}x{larger[1]}  →  {smaller[0]}x{smaller[1]}  ({count} file(s))")
    print()

    updated_rows = []
    errors = []
    crop_index = 0
    success_count = 0
    start_time = time.time()

    for row in rows:
        w, h = int(row["width"]), int(row["height"])

        if (w, h) not in pairs:
            updated_rows.append(row)
            continue

        crop_index += 1
        target_w, target_h = pairs[(w, h)]
        file_name = row["file_name"]
        video_path = folder / file_name

        if not video_path.exists():
            print(f"  WARNING: {file_name} not found in folder — skipping")
            updated_rows.append(row)
            continue

        print(f"  [{crop_index}/{total_to_crop}] {'[dry-run] ' if args.dry_run else ''}Cropping {file_name}")
        print(f"    {w}x{h}  →  {target_w}x{target_h}")

        if not args.dry_run:
            tmp_path = video_path.with_suffix(".tmp.mp4")
            success, stderr = crop_video(video_path, tmp_path, w, h, target_w, target_h)

            if success:
                success_count += 1
                os.replace(tmp_path, video_path)
                new_ratio = round(target_w / target_h, 3)
                row = dict(row)
                row["width"] = target_w
                row["height"] = target_h
                row["ratio"] = new_ratio
                print(f"    Done — new ratio {new_ratio}")
            else:
                print(f"    ERROR: ffmpeg failed:\n{stderr[-400:]}", file=sys.stderr)
                errors.append(file_name)
                if tmp_path.exists():
                    tmp_path.unlink()

            if crop_index % 10 == 0 or crop_index == total_to_crop:
                elapsed = time.time() - start_time
                rate = crop_index / elapsed if elapsed > 0 else 0
                print(
                    f"\n  --- Progress: {crop_index}/{total_to_crop} processed, "
                    f"{success_count} succeeded, {len(errors)} error(s) — "
                    f"{elapsed:.1f}s elapsed ({rate:.2f} files/s) ---\n"
                )

        updated_rows.append(row)

    if args.dry_run:
        print("\nDry run complete — no files modified.")
        return

    if errors:
        print(
            f"\nFinished with {len(errors)} error(s). "
            "CSV has not been updated to avoid partial state.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    affected = len([r for r in rows if (int(r["width"]), int(r["height"])) in pairs])
    print(f"\nDone. {affected} video(s) cropped, installation.csv updated.")


if __name__ == "__main__":
    main()
