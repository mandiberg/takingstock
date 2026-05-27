#!/usr/bin/env python3
"""
analyze_mode0_to_mode1_throughput.py

Analyzes image loss between MODE 0 (df_sorted CSVs) and MODE 1 (cluster_files.csv).

Matches input CSVs to output cluster folders by their shared key: c{N}_p{N}_t{N}.
Reports per-cluster counts, retention rates, and flags heavy-loss clusters.

Optionally parses a stdout log file to extract per-cluster intermediate stage
counts (post-dedupe, skip_face, cropfail, etc.) if MODE 1 was run with logging.

Usage:
    python analyze_mode0_to_mode1_throughput.py \
        --csv-folder /path/to/make_video_CSVs/SegmentHelper_TheOffice/projector_test \
        --output-folder /Volumes/LaCie/output_folder/_dress_rehearsal_2plus \
        [--log-file /path/to/stdout.txt] \
        [--out throughput_report.csv] \
        [--flag-threshold 3]

Stage-by-stage loss map (MODE 1 pipeline):
  df_sorted CSV (ct{N} rows)
    → _mode1_filter_excluded_images   [Exclude table filter]
    → _mode1_recheck_is_dupe_of       [DB is_dupe_of flag — NOTE: currently buggy, see bug note]
    → remove_duplicates (SSIM pass)   [Highest-impact reduction]
    → linear_test_df assembly:
        → this_dist > MAXD            [Distance outlier trim — correct]
        → generate_cropped_image fail [Crop failure]
        → is_face pair test fail      [Sequential pair blending test — suspected over-aggressive]
        → inpaint bailout
    → cluster_files.csv rows          [Final output]
"""

import os
import re
import argparse
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# Default paths — override via CLI or edit here
# ---------------------------------------------------------------------------
DEFAULT_CSV_FOLDER = (
    "/Users/michaelmandiberg/Documents/projects-active/"
    "facemap_production/make_video_CSVs/SegmentHelper_TheOffice/projector_test"
)
DEFAULT_OUTPUT_FOLDER = (
    "/Volumes/LaCie/output_folder/_dress_rehearsal_2plus"
)
DEFAULT_OUT = "throughput_report.csv"
DEFAULT_FLAG_THRESHOLD = 3  # flag clusters with <= N output rows

# ---------------------------------------------------------------------------
# Filename / folder name patterns
#   Input CSV:       df_sorted_c572_p734_t0_ct101.csv
#   Output folder:   clustercc572_p734_t0_1779821951.1780741
#   Shared key:      c572_p734_t0
# ---------------------------------------------------------------------------
CSV_RE = re.compile(r"^df_sorted_(c\d+_p\d+_t\d+)_ct(\d+)\.csv$")
FOLDER_RE = re.compile(r"^clusterc(c\d+_p\d+_t\d+)_([\d.]+)$")
KEY_PARTS_RE = re.compile(r"^(c\d+)_(p\d+)_(t\d+)$")

# Log line patterns (from MODE 1 stdout)
# "[MODE1 ASSEMBLY] linear_test_df summary rows_total=N rows_saved=M ..."
LOG_ASSEMBLY_RE = re.compile(
    r"\[MODE1 ASSEMBLY\] linear_test_df summary "
    r"rows_total=(\d+) rows_saved=(\d+) rows_write_failed=(\d+) "
    r"cache_hits=(\d+) cache_misses=(\d+) skip_face=(\d+)"
)
# "assembling cluster {cluster_no}, topic {topic_no}, pose {pose_no}, ..."
LOG_ASSEMBLING_RE = re.compile(
    r"assembling cluster (c\d+), topic (t\d+), pose (p\d+).*csv file: (df_sorted_\S+)"
)
# "[MODE1 DEDUPE][exclude] ... rows_before=N ... rows_after=M dropped=K"
LOG_EXCLUDE_RE = re.compile(
    r"\[MODE1 DEDUPE\]\[exclude\] .*rows_before=(\d+).*dropped=(\d+) rows_after=(\d+)"
)
# remove_duplicates summary line
LOG_DEDUPE_SUMMARY_RE = re.compile(
    r"look_closer=(\d+).*notdupe.*=(\d+).*pass2.*=(\d+).*ssim_pairs=(\d+).*confirmed_dupes=(\d+)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_csv_folder(csv_folder: str) -> dict:
    """
    Walk csv_folder; for each df_sorted_*.csv return a record keyed on
    the c{N}_p{N}_t{N} cluster key.

    Returns:
        dict[key] = {
            "csv_file": filename,
            "ct": int,          # count in filename (from _ct{N})
            "csv_rows": int,    # actual row count in file
        }
    """
    result = {}
    for fname in os.listdir(csv_folder):
        m = CSV_RE.match(fname)
        if not m:
            continue
        key = m.group(1)
        ct = int(m.group(2))
        fpath = os.path.join(csv_folder, fname)
        try:
            # read only image_id column for speed
            df = pd.read_csv(fpath, usecols=["image_id"])
            csv_rows = len(df)
        except Exception as exc:
            print(f"  [warn] could not read {fname}: {exc}")
            csv_rows = None
        result[key] = {
            "csv_file": fname,
            "csv_path": fpath,
            "ct": ct,
            "csv_rows": csv_rows,
        }
    return result


def _parse_output_folder(output_folder: str) -> dict:
    """
    Walk output_folder; for each clusterc* folder find cluster_files.csv
    and read its row count.

    Returns:
        dict[key] -> list of {
            "folder": dirname,
            "timestamp": float,
            "cluster_files_rows": int or None,
            "output_image_ids": list[int],
        }
    Multiple entries per key are possible (re-runs produce new timestamped folders).
    """
    result = defaultdict(list)
    for dname in os.listdir(output_folder):
        m = FOLDER_RE.match(dname)
        if not m:
            continue
        key = m.group(1)
        timestamp = float(m.group(2)) if m.group(2) else None
        dpath = os.path.join(output_folder, dname)
        cf_path = os.path.join(dpath, "cluster_files.csv")
        if os.path.exists(cf_path):
            try:
                cf = pd.read_csv(cf_path)
                rows = len(cf)
                ids = cf["image_id"].tolist() if "image_id" in cf.columns else []
            except Exception as exc:
                print(f"  [warn] could not read {cf_path}: {exc}")
                rows = None
                ids = []
        else:
            rows = 0
            ids = []
        result[key].append({
            "folder": dname,
            "timestamp": timestamp,
            "cluster_files_rows": rows,
            "output_image_ids": ids,
        })
    return dict(result)


def _parse_log_file(log_path: str) -> dict:
    """
    Parse a MODE 1 stdout log and extract per-cluster assembly stats.

    Returns:
        dict[csv_filename] = {
            "rows_total": int,
            "rows_saved": int,
            "rows_write_failed": int,
            "cache_hits": int,
            "cache_misses": int,
            "skip_face": int,
            "exclude_rows_before": int,
            "exclude_dropped": int,
            "dedupe_confirmed_dupes": int,
        }
    """
    if not log_path or not os.path.exists(log_path):
        return {}

    log_data = {}
    current_csv = None
    current_exclude = {}
    current_dedupe = {}

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()

            m = LOG_ASSEMBLING_RE.search(line)
            if m:
                current_csv = m.group(4)
                current_exclude = {}
                current_dedupe = {}
                continue

            m = LOG_EXCLUDE_RE.search(line)
            if m and current_csv:
                current_exclude = {
                    "exclude_rows_before": int(m.group(1)),
                    "exclude_dropped": int(m.group(2)),
                }
                continue

            m = LOG_DEDUPE_SUMMARY_RE.search(line)
            if m and current_csv:
                current_dedupe = {"dedupe_confirmed_dupes": int(m.group(5))}
                continue

            m = LOG_ASSEMBLY_RE.search(line)
            if m and current_csv:
                log_data[current_csv] = {
                    "rows_total": int(m.group(1)),
                    "rows_saved": int(m.group(2)),
                    "rows_write_failed": int(m.group(3)),
                    "cache_hits": int(m.group(4)),
                    "cache_misses": int(m.group(5)),
                    "skip_face": int(m.group(6)),
                    **current_exclude,
                    **current_dedupe,
                }
                current_csv = None
                current_exclude = {}
                current_dedupe = {}

    return log_data


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    csv_data: dict,
    output_data: dict,
    log_data: dict,
    flag_threshold: int,
) -> pd.DataFrame:
    all_keys = sorted(set(csv_data.keys()) | set(output_data.keys()))
    rows = []
    for key in all_keys:
        inp = csv_data.get(key, {})
        out_list = output_data.get(key, [])

        # If multiple timestamped folders exist for same key, take the most recent
        if out_list:
            out_list_sorted = sorted(
                out_list, key=lambda x: x.get("timestamp") or 0, reverse=True
            )
            latest_out = out_list_sorted[0]
        else:
            latest_out = {}

        ct = inp.get("ct")
        csv_rows = inp.get("csv_rows")
        out_rows = latest_out.get("cluster_files_rows", 0) if latest_out else 0

        # Log data for this CSV (if available)
        csv_file = inp.get("csv_file", "")
        ld = log_data.get(csv_file, {})

        # Compute derived metrics
        retention_pct = round(out_rows / csv_rows * 100, 1) if csv_rows else None
        ct_vs_csv_match = (ct == csv_rows) if (ct is not None and csv_rows is not None) else None
        multiple_runs = len(out_list) > 1

        # Extract pose/cluster/topic from key
        kp = KEY_PARTS_RE.match(key)
        cluster_id = kp.group(1) if kp else ""
        pose_id = kp.group(2) if kp else ""
        topic_id = kp.group(3) if kp else ""

        row = {
            # Identification
            "cluster_key": key,
            "cluster_id": cluster_id,
            "pose_id": pose_id,
            "topic_id": topic_id,
            "input_csv": csv_file,
            "output_folder": latest_out.get("folder", ""),
            # Input counts
            "input_ct_from_filename": ct,
            "input_csv_rows": csv_rows,
            "ct_matches_csv": ct_vs_csv_match,
            # Output counts
            "output_rows": out_rows,
            "loss_count": (csv_rows - out_rows) if csv_rows is not None else None,
            "retention_pct": retention_pct,
            # Flags
            "flagged_low_output": out_rows <= flag_threshold,
            "anchor_stuck": out_rows == 1,  # exactly 1 output = first-run pass + all pair tests failed
            "no_output_folder": len(out_list) == 0,
            "multiple_runs": multiple_runs,
            "run_count": len(out_list),
            # Log-derived intermediate counts (empty if no log)
            "log_rows_total": ld.get("rows_total"),
            "log_rows_saved": ld.get("rows_saved"),
            "log_skip_face": ld.get("skip_face"),
            "log_rows_write_failed": ld.get("rows_write_failed"),
            "log_cache_hits": ld.get("cache_hits"),
            "log_cache_misses": ld.get("cache_misses"),
            "log_exclude_rows_before": ld.get("exclude_rows_before"),
            "log_exclude_dropped": ld.get("exclude_dropped"),
            "log_dedupe_confirmed_dupes": ld.get("dedupe_confirmed_dupes"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, flag_threshold: int):
    n_total = len(df)
    n_matched = int(((df["input_csv"] != "") & (df["output_rows"] > 0)).sum())
    n_no_output = int(df["no_output_folder"].sum())
    n_zero_output = int(((~df["no_output_folder"]) & (df["output_rows"] == 0)).sum())
    n_flagged = int(df["flagged_low_output"].sum())

    has_retention = df["retention_pct"].notna()
    r = df.loc[has_retention, "retention_pct"]

    print("\n" + "=" * 60)
    print("MODE 0 → MODE 1 THROUGHPUT REPORT")
    print("=" * 60)
    print(f"  Input CSVs found:          {df['input_csv'].ne('').sum()}")
    print(f"  Output cluster keys found: {df['output_folder'].ne('').sum()}")
    print(f"  Matched (input + output):  {n_matched}")
    print(f"  No output folder:          {n_no_output}")
    print(f"  Output folder exists, 0 rows: {n_zero_output}")
    print(f"  Flagged (output <= {flag_threshold} rows): {n_flagged}")
    print(f"  Multiple run folders:      {int(df['multiple_runs'].sum())}")

    if len(r):
        print(f"\n  Retention rate distribution (matched clusters):")
        buckets = [
            ("  0%  (complete loss)", 0, 0),
            ("  1–10%",               1, 10),
            ("  11–25%",             11, 25),
            ("  26–50%",             26, 50),
            ("  51–100%",            51, 100),
        ]
        for label, lo, hi in buckets:
            if lo == 0 and hi == 0:
                n = (r == 0).sum()
            else:
                n = ((r >= lo) & (r <= hi)).sum()
            bar = "#" * min(int(n / max(len(r), 1) * 40), 40)
            print(f"    {label:25s}: {int(n):5d}  {bar}")
        print(f"\n  Median retention: {r.median():.1f}%")
        print(f"  Mean retention:   {r.mean():.1f}%")
        print(f"  Min retention:    {r.min():.1f}%  →  {df.loc[r.idxmin(), 'cluster_key']}")

    # Flagged examples
    flagged = df[df["flagged_low_output"] & df["input_csv"].ne("")].sort_values("retention_pct")
    if not flagged.empty:
        print(f"\n  FLAGGED CLUSTERS (output ≤ {flag_threshold} rows) — first 30:")
        cols = ["cluster_key", "input_ct_from_filename", "output_rows", "retention_pct"]
        log_cols = ["log_rows_total", "log_skip_face", "log_dedupe_confirmed_dupes"]
        display_cols = cols + [c for c in log_cols if flagged[c].notna().any()]
        print(flagged[display_cols].head(30).to_string(index=False))

    # anchor-stuck diagnostic
    n_anchor_stuck = int(df["anchor_stuck"].sum())
    print(f"\n  Anchor-stuck (output == 1): {n_anchor_stuck} ({100*n_anchor_stuck/max(len(df),1):.1f}%)")
    print("  (First image always saves via first_run=True; anchor-stuck means every")
    print("   subsequent image failed the is_face pair test against the anchor.)")

    # Pose-level breakdown
    if "pose_id" in df.columns:
        pose_stats = (
            df[df["input_csv"] != ""]
            .groupby("pose_id")
            .agg(
                clusters=("cluster_key", "count"),
                anchor_stuck=("anchor_stuck", "sum"),
                flagged=("flagged_low_output", "sum"),
                mean_retention=("retention_pct", "mean"),
                median_retention=("retention_pct", "median"),
                mean_input_ct=("input_ct_from_filename", "mean"),
            )
            .round(1)
            .sort_values("anchor_stuck", ascending=False)
        )
        pose_stats["anchor_stuck_pct"] = (
            (pose_stats["anchor_stuck"] / pose_stats["clusters"] * 100).round(1)
        )
        print(f"\n  Per-pose breakdown (sorted by anchor-stuck count):")
        print(pose_stats.to_string())

    # Clusters in output with no matching input CSV (orphaned runs)
    orphaned = df[(df["input_csv"] == "") & (df["output_folder"] != "")]
    if not orphaned.empty:
        print(f"\n  ORPHANED output folders (no matching input CSV): {len(orphaned)}")
        print(orphaned[["cluster_key", "output_folder", "output_rows"]].head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze MODE 0 → MODE 1 image throughput.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--csv-folder",
        default=DEFAULT_CSV_FOLDER,
        help="Folder containing df_sorted_*.csv files (MODE 0 output).",
    )
    parser.add_argument(
        "--output-folder",
        default=DEFAULT_OUTPUT_FOLDER,
        help="Folder containing clusterc* subdirectories (MODE 1 output).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional: path to MODE 1 stdout log for intermediate stage counts.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output CSV path for the full report (default: throughput_report.csv).",
    )
    parser.add_argument(
        "--flag-threshold",
        type=int,
        default=DEFAULT_FLAG_THRESHOLD,
        help="Flag clusters with <= N output rows (default: 3).",
    )
    parser.add_argument(
        "--no-read-csv",
        action="store_true",
        help="Skip reading input CSV row counts (faster; uses ct{N} from filename).",
    )
    args = parser.parse_args()

    print(f"Scanning CSV folder:    {args.csv_folder}")
    csv_data = _parse_csv_folder(args.csv_folder)
    if args.no_read_csv:
        for v in csv_data.values():
            v["csv_rows"] = v["ct"]
    print(f"  Found {len(csv_data)} input CSVs")

    print(f"Scanning output folder: {args.output_folder}")
    output_data = _parse_output_folder(args.output_folder)
    print(f"  Found {len(output_data)} cluster keys in output")

    log_data = {}
    if args.log_file:
        print(f"Parsing log file:       {args.log_file}")
        log_data = _parse_log_file(args.log_file)
        print(f"  Found log entries for {len(log_data)} CSVs")

    df = build_report(csv_data, output_data, log_data, args.flag_threshold)
    print_summary(df, args.flag_threshold)

    df.to_csv(args.out, index=False)
    print(f"\nFull report written to: {args.out}")

    # Quick spot-check of the example from the user
    example_key = "c572_p734_t0"
    if example_key in df["cluster_key"].values:
        ex = df[df["cluster_key"] == example_key].iloc[0]
        print(f"\nExample cluster {example_key}:")
        print(f"  Input ct:    {ex['input_ct_from_filename']}")
        print(f"  Output rows: {ex['output_rows']}")
        print(f"  Retention:   {ex['retention_pct']}%")


if __name__ == "__main__":
    main()
