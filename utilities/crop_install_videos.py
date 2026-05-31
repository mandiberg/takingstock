#!/usr/bin/env python3
"""
Normalize video dimensions in an installation folder by cropping videos that
are exactly 2px wider or taller than a paired dimension down to match the
smaller size, then updating installation.csv accordingly.

Usage:
    python crop_install_videos.py <folder> [--dry-run]
"""

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def crop_video(input_path, output_path, src_w, src_h, target_w, target_h,
               threads_per_job=4):
    """
    Crop input_path to target dimensions, centering the crop window, and
    write the result to output_path.  Audio is stream-copied unchanged.
    Returns (success: bool, stderr: str).
    """
    x_offset = (src_w - target_w) // 2
    y_offset = (src_h - target_h) // 2

    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-loglevel", "error",
        "-i", str(input_path),
        "-vf", f"crop={target_w}:{target_h}:{x_offset}:{y_offset}",
        "-c:a", "copy",
        "-threads", str(threads_per_job),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return result.returncode == 0, result.stderr


def crop_job(row, folder, pairs, dry_run, threads_per_job):
    """
    Process one CSV row. Returns (updated_row, error_filename_or_None, out, err).
    Thread-safe: each call operates only on its own per-row file paths.
    """
    out, err = [], []
    w, h = int(row["width"]), int(row["height"])

    if (w, h) not in pairs:
        return row, None, out, err

    target_w, target_h = pairs[(w, h)]
    file_name = row["file_name"]
    video_path = folder / file_name

    if not video_path.exists():
        out.append(f"  WARNING: {file_name} not found in folder — skipping")
        return row, None, out, err

    prefix = "[dry-run] " if dry_run else ""
    out.append(f"  {prefix}Cropping {file_name}  ({w}x{h} \u2192 {target_w}x{target_h})")

    if dry_run:
        return row, None, out, err

    tmp_path = video_path.with_suffix(".tmp.mp4")
    success, stderr = crop_video(
        video_path, tmp_path, w, h, target_w, target_h, threads_per_job
    )

    if success:
        os.replace(tmp_path, video_path)
        new_ratio = round(target_w / target_h, 3)
        row = dict(row)
        row["width"] = target_w
        row["height"] = target_h
        row["ratio"] = new_ratio
        out.append(f"    Done — new ratio {new_ratio}")
        return row, None, out, err

    err.append(f"    ERROR: ffmpeg failed for {file_name}:\n{stderr[-400:]}")
    if tmp_path.exists():
        tmp_path.unlink()
    return row, file_name, out, err


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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel ffmpeg jobs (default: 4)",
    )
    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=4,
        metavar="N",
        dest="threads_per_job",
        help="ffmpeg -threads value per job (default: 4)",
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

    print("2px dimension pairs that will be normalized (larger → smaller):")
    for larger, smaller in sorted(pairs.items()):
        print(f"  {larger[0]}x{larger[1]}  →  {smaller[0]}x{smaller[1]}")
    print()

    print(f"Workers: {args.workers}  |  threads/job: {args.threads_per_job}\n")

    results = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(
                crop_job, row, folder, pairs, args.dry_run, args.threads_per_job
            ): i
            for i, row in enumerate(rows)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            updated_row, error, out_lines, err_lines = future.result()
            for line in out_lines:
                print(line)
            for line in err_lines:
                print(line, file=sys.stderr)
            results[idx] = (updated_row, error)

    updated_rows = [r for r, _ in results]
    errors = [e for _, e in results if e is not None]

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
