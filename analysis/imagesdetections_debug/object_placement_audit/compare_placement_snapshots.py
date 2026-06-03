#!/usr/bin/env python3
"""Compare two ImagesDetections placement audit snapshots.

Loads the CSVs produced by analyze_imagesdetections_placement.py from two
snapshot directories (before/after a logic change) and produces:

  global_slot_shares_diff.csv   — overall slot balance shift
  placement_rate_diff.csv       — per-class placed_rate change, sorted by |Δ|
  slot_share_diff.csv           — per-class per-slot share delta, long format
  outlier_flag_diff.csv         — classes that gained or lost outlier flags
  tracked_classes_detail.csv    — full before/after for Phase 1 target classes
  summary.txt                   — human-readable highlights (also printed)

Usage:
    python compare_placement_snapshots.py \\
        --before SegmentHelper_TheOffice_existing_state \\
        --after  SegmentHelper_TheOffice_new_state \\
        [--output-dir placement_diff_phase1] \\
        [--tracked-classes 24,26,27,56,57,59,60,63,90]


python analysis/imagesdetections_debug/object_placement_audit/compare_placement_snapshots.py \
  --before analysis/imagesdetections_debug/object_placement_audit/SegmentHelper_TheOffice_existing_state \
  --after  analysis/imagesdetections_debug/object_placement_audit/SegmentHelper_TheOffice_new_state \
  --output-dir analysis/imagesdetections_debug/object_placement_audit/placement_diff_phase1
  
  """

import argparse
import json
import os
from datetime import datetime

import pandas as pd

# Phase 1 target class_ids and their human names
DEFAULT_TRACKED = [24, 26, 27, 56, 57, 59, 60, 63, 67, 82, 89, 90, 94, 95,96]
CLASS_NAMES = {
    1: "person",
    13: "stop sign",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    56: "chair",
    57: "couch",
    59: "bed",
    60: "dining table",
    63: "laptop",
    83: "bag",
    87: "unknown_87",
    90: "stethoscope",
}

SHARE_COLS = [
    "share_left_hand", "share_right_hand", "share_top_face",
    "share_left_eye", "share_right_eye", "share_mouth",
    "share_shoulder", "share_waist", "share_feet",
]

FLAG_COLS = [
    "flag_fixed_low_placement", "flag_fixed_slot_dominance",
    "flag_fixed_waist_absence", "flag_fixed_feet_absence",
    "flag_iqr_low_placement", "flag_iqr_slot_dominance",
    "flag_mad_low_placement", "flag_mad_slot_dominance",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_snapshot(folder):
    """Load all CSVs from a snapshot folder. Returns dict of DataFrames + metadata."""
    files = {
        "class_rates": "class_rates.csv",
        "class_slot_matrix": "class_slot_matrix_all_conf.csv",
        "global_slot_shares": "global_slot_shares.csv",
        "class_slot_counts_long": "class_slot_counts_long.csv",
        "outlier_summary": "outlier_summary.csv",
    }
    result = {}
    for key, fname in files.items():
        path = os.path.join(folder, fname)
        if os.path.exists(path):
            result[key] = pd.read_csv(path)
        else:
            print(f"  Warning: {fname} not found in {folder} — skipping.")
            result[key] = None

    meta_path = os.path.join(folder, "run_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            result["metadata"] = json.load(f)
    else:
        result["metadata"] = {}

    return result


# ---------------------------------------------------------------------------
# Diff functions
# ---------------------------------------------------------------------------

def diff_global_slot_shares(before_df, after_df):
    """Compare global slot share per slot and conf_band."""
    merged = before_df.merge(
        after_df, on=["conf_band", "slot"], suffixes=("_before", "_after")
    )
    merged["delta"] = merged["slot_share_after"] - merged["slot_share_before"]
    merged["pct_change"] = (
        merged["delta"] / merged["slot_share_before"].replace(0, float("nan"))
    ) * 100
    return merged.sort_values("delta", key=abs, ascending=False).reset_index(drop=True)


def diff_placement_rates(before_df, after_df):
    """Per-class placed_rate change, sorted by |Δplaced_rate_detection|."""
    rate_cols = ["placed_rate_detection", "placed_rate_image"]
    id_cols = ["class_id", "conf_band"]
    keep = id_cols + rate_cols + ["total_detections", "total_images"]

    b = before_df[keep].rename(
        columns={c: f"{c}_before" for c in rate_cols + ["total_detections", "total_images"]}
    )
    a = after_df[keep].rename(
        columns={c: f"{c}_after" for c in rate_cols + ["total_detections", "total_images"]}
    )
    merged = b.merge(a, on=id_cols)
    merged["delta_placed_rate_detection"] = (
        merged["placed_rate_detection_after"] - merged["placed_rate_detection_before"]
    )
    merged["delta_placed_rate_image"] = (
        merged["placed_rate_image_after"] - merged["placed_rate_image_before"]
    )
    # Flag if total_detections changed (should be identical for same-scope reruns)
    merged["detection_count_changed"] = (
        merged["total_detections_before"] != merged["total_detections_after"]
    )
    return merged.sort_values(
        "delta_placed_rate_detection", key=abs, ascending=False
    ).reset_index(drop=True)


def diff_slot_shares(before_df, after_df):
    """Per-class per-slot share delta, long format sorted by |Δ|."""
    id_cols = ["class_id", "conf_band"]
    b = before_df[id_cols + SHARE_COLS].copy()
    a = after_df[id_cols + SHARE_COLS].copy()
    merged = b.merge(a, on=id_cols, suffixes=("_before", "_after"))

    rows = []
    for _, row in merged.iterrows():
        for sc in SHARE_COLS:
            slot = sc.replace("share_", "")
            b_val = float(row[f"{sc}_before"])
            a_val = float(row[f"{sc}_after"])
            delta = a_val - b_val
            rows.append({
                "class_id": int(row["class_id"]),
                "conf_band": row["conf_band"],
                "slot": slot,
                "share_before": round(b_val, 6),
                "share_after": round(a_val, 6),
                "delta": round(delta, 6),
                "abs_delta": abs(delta),
            })

    result = (
        pd.DataFrame(rows)
        .sort_values("abs_delta", ascending=False)
        .drop(columns=["abs_delta"])
        .reset_index(drop=True)
    )
    return result


def diff_outlier_flags(before_df, after_df):
    """Which classes gained or lost outlier flags."""
    id_cols = ["class_id", "conf_band"]
    keep = id_cols + FLAG_COLS + ["outlier_flag_count", "outlier_reasons"]

    b = before_df[keep].copy()
    a = after_df[keep].copy()
    merged = b.merge(a, on=id_cols, suffixes=("_before", "_after"))
    merged["flag_count_delta"] = (
        merged["outlier_flag_count_after"] - merged["outlier_flag_count_before"]
    )

    gained_list, lost_list = [], []
    for _, row in merged.iterrows():
        gained, lost = [], []
        for f in FLAG_COLS:
            was_set = bool(row[f"{f}_before"])
            now_set = bool(row[f"{f}_after"])
            if not was_set and now_set:
                gained.append(f.replace("flag_", ""))
            elif was_set and not now_set:
                lost.append(f.replace("flag_", ""))
        gained_list.append(";".join(gained))
        lost_list.append(";".join(lost))

    merged["gained_flags"] = gained_list
    merged["lost_flags"] = lost_list

    changed = merged[merged["flag_count_delta"] != 0].copy()
    out_cols = id_cols + [
        "outlier_flag_count_before", "outlier_flag_count_after", "flag_count_delta",
        "gained_flags", "lost_flags",
        "outlier_reasons_before", "outlier_reasons_after",
    ]
    return changed[out_cols].sort_values("flag_count_delta").reset_index(drop=True)


def build_tracked_detail(before_df, after_df, tracked_classes):
    """Full before/after comparison for the tracked Phase 1 target classes."""
    id_cols = ["class_id", "conf_band"]
    key_cols = (
        id_cols
        + ["total_detections", "total_images", "placed_rate_detection", "placed_rate_image"]
        + SHARE_COLS
        + ["outlier_flag_count"]
    )
    b = before_df[before_df["class_id"].isin(tracked_classes)][key_cols]
    a = after_df[after_df["class_id"].isin(tracked_classes)][key_cols]
    merged = b.merge(a, on=id_cols, suffixes=("_before", "_after"))

    delta_cols = (
        ["total_detections", "total_images", "placed_rate_detection", "placed_rate_image"]
        + SHARE_COLS
        + ["outlier_flag_count"]
    )
    for col in delta_cols:
        bc = f"{col}_before"
        ac = f"{col}_after"
        if bc in merged.columns and ac in merged.columns:
            merged[f"delta_{col}"] = merged[ac] - merged[bc]

    return merged.sort_values(["class_id", "conf_band"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def build_summary_text(
    before_meta, after_meta,
    global_diff, rate_diff, slot_diff, outlier_diff, tracked_detail,
    tracked_classes,
):
    lines = []
    sep = "=" * 70

    lines += [
        sep,
        "PLACEMENT SNAPSHOT DIFF — PHASE 1 AUDIT",
        f"Generated : {datetime.utcnow().isoformat()}Z",
        f"Before    : {before_meta.get('helper_table', '?')}  ({before_meta.get('generated_at', '?')})",
        f"After     : {after_meta.get('helper_table', '?')}  ({after_meta.get('generated_at', '?')})",
        sep,
    ]

    # --- Global slot shares ---
    lines.append("\n--- GLOBAL SLOT SHARE CHANGES (all_conf) ---")
    g = global_diff[global_diff["conf_band"] == "all_conf"].copy()
    if g.empty:
        lines.append("  (no data)")
    else:
        for _, row in g.iterrows():
            arrow = f"{row['slot_share_before']:.4f} → {row['slot_share_after']:.4f}"
            lines.append(
                f"  {row['slot']:20s}  {arrow}  Δ{row['delta']:+.4f}"
                f"  ({row['pct_change']:+.1f}%)"
            )

    # --- Top placement rate movers ---
    lines.append("\n--- TOP 15 PLACEMENT RATE CHANGES (all_conf, by |Δ|) ---")
    r = rate_diff[rate_diff["conf_band"] == "all_conf"].head(15)
    for _, row in r.iterrows():
        cid = int(row["class_id"])
        name = CLASS_NAMES.get(cid, f"cls{cid}")
        arrow = f"{row['placed_rate_detection_before']:.4f} → {row['placed_rate_detection_after']:.4f}"
        lines.append(
            f"  [{cid:4d}] {name:20s}  placed_rate: {arrow}  Δ{row['delta_placed_rate_detection']:+.5f}"
        )

    # --- Tracked class detail ---
    lines.append("\n--- PHASE 1 TARGET CLASSES — DETAIL (all_conf) ---")
    td = tracked_detail[tracked_detail["conf_band"] == "all_conf"].copy()
    if td.empty:
        lines.append("  (none found in snapshots)")
    for _, row in td.iterrows():
        cid = int(row["class_id"])
        name = CLASS_NAMES.get(cid, f"cls{cid}")
        lines.append(f"\n  [{cid}] {name}")
        pr_arrow = (
            f"{row['placed_rate_detection_before']:.4f} → "
            f"{row['placed_rate_detection_after']:.4f}"
        )
        lines.append(
            f"    placed_rate  : {pr_arrow}  Δ{row['delta_placed_rate_detection']:+.5f}"
        )
        for sc in SHARE_COLS:
            slot = sc.replace("share_", "")
            b_val = float(row[f"{sc}_before"])
            a_val = float(row[f"{sc}_after"])
            delta = float(row[f"delta_{sc}"])
            # Show slot if it's active or meaningfully changed
            if abs(delta) > 0.005 or b_val > 0.01 or a_val > 0.01:
                marker = " ←" if abs(delta) > 0.02 else ""
                lines.append(
                    f"    {slot:15s} : {b_val:.4f} → {a_val:.4f}  Δ{delta:+.4f}{marker}"
                )

    # --- Outlier flag changes ---
    lines.append("\n--- OUTLIER FLAG CHANGES ---")
    if outlier_diff.empty:
        lines.append("  No changes in outlier flag counts.")
    else:
        for _, row in outlier_diff.iterrows():
            cid = int(row["class_id"])
            name = CLASS_NAMES.get(cid, f"cls{cid}")
            lines.append(
                f"  [{cid:4d}] {name:20s}  flags: "
                f"{int(row['outlier_flag_count_before'])} → "
                f"{int(row['outlier_flag_count_after'])}  "
                f"(Δ{int(row['flag_count_delta']):+d})"
            )
            if row["lost_flags"]:
                lines.append(f"         lost   : {row['lost_flags']}")
            if row["gained_flags"]:
                lines.append(f"         gained : {row['gained_flags']}")

    lines.append(f"\n{sep}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Diff two ImagesDetections placement audit snapshots."
    )
    parser.add_argument(
        "--before", required=True,
        help="Path to the before-snapshot directory."
    )
    parser.add_argument(
        "--after", required=True,
        help="Path to the after-snapshot directory."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Directory to write diff output files. "
            "Defaults to <after_dir>/../placement_diff_<timestamp>."
        ),
    )
    parser.add_argument(
        "--tracked-classes",
        default=",".join(str(x) for x in DEFAULT_TRACKED),
        help="Comma-separated class_ids for detailed Phase 1 analysis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tracked_classes = [int(x) for x in args.tracked_classes.split(",")]

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.after)),
        f"placement_diff_{ts}",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading before snapshot: {args.before}")
    before = load_snapshot(args.before)
    print(f"Loading after  snapshot: {args.after}")
    after = load_snapshot(args.after)
    print()

    results = {}

    if before["global_slot_shares"] is not None and after["global_slot_shares"] is not None:
        results["global_slot_shares_diff"] = diff_global_slot_shares(
            before["global_slot_shares"], after["global_slot_shares"]
        )

    if before["class_rates"] is not None and after["class_rates"] is not None:
        results["placement_rate_diff"] = diff_placement_rates(
            before["class_rates"], after["class_rates"]
        )
        results["slot_share_diff"] = diff_slot_shares(
            before["class_rates"], after["class_rates"]
        )
        results["outlier_flag_diff"] = diff_outlier_flags(
            before["class_rates"], after["class_rates"]
        )
        results["tracked_classes_detail"] = build_tracked_detail(
            before["class_rates"], after["class_rates"], tracked_classes
        )

    # Write CSVs
    for name, df in results.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"Written: {path}")

    # Summary
    summary = build_summary_text(
        before["metadata"],
        after["metadata"],
        results.get("global_slot_shares_diff", pd.DataFrame()),
        results.get("placement_rate_diff", pd.DataFrame()),
        results.get("slot_share_diff", pd.DataFrame()),
        results.get("outlier_flag_diff", pd.DataFrame()),
        results.get("tracked_classes_detail", pd.DataFrame()),
        tracked_classes,
    )

    print()
    print(summary)

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary written : {summary_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
