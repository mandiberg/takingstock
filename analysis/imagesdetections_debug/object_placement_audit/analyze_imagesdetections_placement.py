#!/usr/bin/env python3
"""Analyze ImagesDetections placement rates for targeted classes.

Outputs reusable CSV artifacts under analysis/imagesdetections_debug/object_placement_audit.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

if sys.platform == "darwin":
    sys.path.insert(1, "/Users/michaelmandiberg/Documents/GitHub/takingstock/")

from mp_db_io import DataIO  # noqa: E402

# =========================
# Analysis Scope Constants
# =========================
COCO_TARGET_CLASS_IDS = [
    1,
    26,
    27,
    56,
    57,
    59,
    60,
    63,
    67,
    73,
    74,
] + list(range(39, 51))
CUSTOM_CLASS_MIN_ID = 80

HELPER_TABLE_PROGRESSION = [
    "SegmentHelper_oct2025_every40_even",
    "SegmentHelper_oct2025_every40",
    "SegmentHelper_T11_Oct20_COCO_Custom_every40",
    "SegmentHelper_T11_Oct20_COCO_Custom_evens_quarters",
    "SegmentHelper_T11_Oct20_COCO_Custom",
]

# Unified testing-set helper target requested by user.
DEFAULT_HELPER_TABLE = "SegmentHelper_T11_Oct20_COCO_Custom_every40"
TESTING_SET_INTERSECT_TABLE = None

SLOTS = [
    "left_hand",
    "right_hand",
    "top_face",
    "left_eye",
    "right_eye",
    "mouth",
    "shoulder",
    "waist",
    "feet",
]

# Output slot order requested in UI exports.
SLOT_OUTPUT_ORDER = [
    "feet",
    "left_eye",
    "left_hand",
    "mouth",
    "right_eye",
    "right_hand",
    "shoulder",
    "top_face",
    "waist",
]

# Alternative body-first order for side-by-side inspection.
SLOT_BODY_ORDER = [
    "top_face",
    "left_eye",
    "right_eye",
    "mouth",
    "shoulder",
    "left_hand",
    "right_hand",
    "waist",
    "feet",
]

# =========================
# Threshold Constants
# =========================
HIGH_CONF_THRESHOLD = 0.8
PLACED_RATE_DETECTION_MIN = 0.05
PLACED_RATE_IMAGE_MIN = 0.05
SLOT_DOMINANCE_MAX = 0.85
WAIST_FEET_MIN_SHARE = 0.005
IQR_MULTIPLIER = 1.5
MAD_Z_THRESHOLD = 3.5


def get_engine():
    io = DataIO()
    db = io.db
    engine = create_engine(
        "mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
            user=db["user"],
            pw=db["pass"],
            db=db["name"],
            socket=db["unix_socket"],
        ),
        poolclass=NullPool,
    )
    return engine


def class_filter_sql():
    cls = ",".join(str(x) for x in sorted(set(COCO_TARGET_CLASS_IDS)))
    return f"(d.class_id IN ({cls}) OR d.class_id >= {CUSTOM_CLASS_MIN_ID})"


def helper_exists(conn, helper_table):
    q = text("SHOW TABLES LIKE :tbl")
    return conn.execute(q, {"tbl": helper_table}).fetchone() is not None


def get_class_names(conn):
    class_df = pd.read_sql(
        text("SELECT class_id, class_name FROM YoloClasses"),
        conn,
    )
    if "class_name" not in class_df.columns:
        class_df["class_name"] = None
    class_df["class_name"] = class_df["class_name"].fillna("")
    return class_df


def build_temp_tables(conn, helper_table, intersect_table=None):
    filter_sql = class_filter_sql()

    intersect_join_sql = ""
    if intersect_table:
        intersect_join_sql = f"INNER JOIN {intersect_table} ix ON ix.image_id = d.image_id"

    conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_det_base"))
    conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_assign_base"))

    create_det = text(
        f"""
        CREATE TEMPORARY TABLE tmp_det_base AS
        SELECT
            d.detection_id,
            d.image_id,
            d.class_id,
            d.conf,
            CASE WHEN d.conf >= :high_conf THEN 'high_conf' ELSE 'low_conf' END AS conf_band
        FROM Detections d
        INNER JOIN {helper_table} sh ON sh.image_id = d.image_id
                {intersect_join_sql}
        WHERE d.conf > 0
          AND {filter_sql}
        """
    )
    conn.execute(create_det, {"high_conf": HIGH_CONF_THRESHOLD})

    conn.execute(text("CREATE INDEX idx_tmp_det_detection ON tmp_det_base (detection_id)"))
    conn.execute(text("CREATE INDEX idx_tmp_det_class_band ON tmp_det_base (class_id, conf_band)"))

    conn.execute(
        text(
            """
            CREATE TEMPORARY TABLE tmp_assign_base (
                class_id INT,
                conf_band VARCHAR(16),
                detection_id BIGINT,
                image_id BIGINT,
                slot VARCHAR(32)
            )
            """
        )
    )

    slot_cols = {
        "left_hand": "left_hand_object_id",
        "right_hand": "right_hand_object_id",
        "top_face": "top_face_object_id",
        "left_eye": "left_eye_object_id",
        "right_eye": "right_eye_object_id",
        "mouth": "mouth_object_id",
        "shoulder": "shoulder_object_id",
        "waist": "waist_object_id",
        "feet": "feet_object_id",
    }

    for slot_name, col_name in slot_cols.items():
        conn.execute(
            text(
                f"""
                INSERT INTO tmp_assign_base (class_id, conf_band, detection_id, image_id, slot)
                SELECT db.class_id, db.conf_band, db.detection_id, db.image_id, :slot_name
                FROM tmp_det_base db
                INNER JOIN ImagesDetections idet ON idet.{col_name} = db.detection_id
                """
            ),
            {"slot_name": slot_name},
        )

    conn.execute(text("CREATE INDEX idx_tmp_assign_class_band ON tmp_assign_base (class_id, conf_band)"))
    conn.execute(text("CREATE INDEX idx_tmp_assign_slot ON tmp_assign_base (slot)"))


def fetch_frames(conn):
    denom = pd.read_sql(
        text(
            """
            SELECT
                class_id,
                conf_band,
                COUNT(*) AS total_detections,
                COUNT(DISTINCT image_id) AS total_images
            FROM tmp_det_base
            GROUP BY class_id, conf_band
            """
        ),
        conn,
    )

    assigned = pd.read_sql(
        text(
            """
            SELECT
                class_id,
                conf_band,
                COUNT(DISTINCT detection_id) AS assigned_detections,
                COUNT(DISTINCT image_id) AS assigned_images,
                COUNT(*) AS total_slot_assignments
            FROM tmp_assign_base
            GROUP BY class_id, conf_band
            """
        ),
        conn,
    )

    slot_counts = pd.read_sql(
        text(
            """
            SELECT
                class_id,
                conf_band,
                slot,
                COUNT(*) AS slot_assignment_count,
                COUNT(DISTINCT detection_id) AS slot_unique_detections,
                COUNT(DISTINCT image_id) AS slot_images
            FROM tmp_assign_base
            GROUP BY class_id, conf_band, slot
            """
        ),
        conn,
    )

    return denom, assigned, slot_counts


def add_all_conf_rows(df, value_cols):
    all_conf = df.groupby("class_id", as_index=False)[value_cols].sum()
    all_conf["conf_band"] = "all_conf"
    return pd.concat([df, all_conf], ignore_index=True)


def build_rates(denom, assigned, slot_counts):
    denom = add_all_conf_rows(denom, ["total_detections", "total_images"])
    assigned = add_all_conf_rows(
        assigned,
        ["assigned_detections", "assigned_images", "total_slot_assignments"],
    )

    slot_counts_all = (
        slot_counts.groupby(["class_id", "slot"], as_index=False)["slot_assignment_count"].sum()
    )
    slot_counts_all["conf_band"] = "all_conf"
    slot_counts = pd.concat([slot_counts, slot_counts_all], ignore_index=True)

    rates = denom.merge(assigned, on=["class_id", "conf_band"], how="left")
    for col in ["assigned_detections", "assigned_images", "total_slot_assignments"]:
        rates[col] = rates[col].fillna(0)

    rates["placed_rate_detection"] = np.where(
        rates["total_detections"] > 0,
        rates["assigned_detections"] / rates["total_detections"],
        np.nan,
    )
    rates["unplaced_rate_detection"] = 1.0 - rates["placed_rate_detection"]
    rates["placed_rate_image"] = np.where(
        rates["total_images"] > 0,
        rates["assigned_images"] / rates["total_images"],
        np.nan,
    )
    rates["unplaced_rate_image"] = 1.0 - rates["placed_rate_image"]

    slot_pivot = slot_counts.pivot_table(
        index=["class_id", "conf_band"],
        columns="slot",
        values="slot_assignment_count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    for slot in SLOTS:
        if slot not in slot_pivot.columns:
            slot_pivot[slot] = 0

    rates = rates.merge(slot_pivot, on=["class_id", "conf_band"], how="left")
    for slot in SLOTS:
        rates[slot] = rates[slot].fillna(0)
        rates[f"share_{slot}"] = np.where(
            rates["total_slot_assignments"] > 0,
            rates[slot] / rates["total_slot_assignments"],
            0.0,
        )

    share_cols = [f"share_{slot}" for slot in SLOTS]
    rates["max_slot_share"] = rates[share_cols].max(axis=1)
    rates["max_slot_name"] = rates[share_cols].idxmax(axis=1).str.replace("share_", "", regex=False)

    return rates, slot_pivot


def _iqr_flags(series, high=False):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    if high:
        cutoff = q3 + IQR_MULTIPLIER * iqr
        return series > cutoff
    cutoff = q1 - IQR_MULTIPLIER * iqr
    return series < cutoff


def _mad_z(series):
    med = series.median()
    mad = np.median(np.abs(series - med))
    if mad == 0:
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (series - med) / mad


def add_outlier_flags(rates):
    rates = rates.copy()

    rates["flag_fixed_low_placement"] = (
        (rates["placed_rate_detection"] < PLACED_RATE_DETECTION_MIN)
        | (rates["placed_rate_image"] < PLACED_RATE_IMAGE_MIN)
    )
    rates["flag_fixed_slot_dominance"] = rates["max_slot_share"] > SLOT_DOMINANCE_MAX
    rates["flag_fixed_waist_absence"] = rates["share_waist"] < WAIST_FEET_MIN_SHARE
    rates["flag_fixed_feet_absence"] = rates["share_feet"] < WAIST_FEET_MIN_SHARE

    rates["flag_iqr_low_placement"] = False
    rates["flag_iqr_slot_dominance"] = False
    rates["flag_mad_low_placement"] = False
    rates["flag_mad_slot_dominance"] = False

    for band, subset in rates.groupby("conf_band"):
        idx = subset.index
        rates.loc[idx, "flag_iqr_low_placement"] = _iqr_flags(subset["placed_rate_detection"], high=False).values
        rates.loc[idx, "flag_iqr_slot_dominance"] = _iqr_flags(subset["max_slot_share"], high=True).values

        z_place = _mad_z(subset["placed_rate_detection"])
        z_skew = _mad_z(subset["max_slot_share"])

        rates.loc[idx, "flag_mad_low_placement"] = (z_place < -MAD_Z_THRESHOLD).values
        rates.loc[idx, "flag_mad_slot_dominance"] = (z_skew > MAD_Z_THRESHOLD).values

    flag_cols = [c for c in rates.columns if c.startswith("flag_")]
    rates["outlier_flag_count"] = rates[flag_cols].sum(axis=1)

    def reason_row(row):
        reasons = []
        if row["flag_fixed_low_placement"]:
            reasons.append("fixed_low_placement")
        if row["flag_fixed_slot_dominance"]:
            reasons.append("fixed_slot_dominance")
        if row["flag_fixed_waist_absence"]:
            reasons.append("fixed_waist_absence")
        if row["flag_fixed_feet_absence"]:
            reasons.append("fixed_feet_absence")
        if row["flag_iqr_low_placement"]:
            reasons.append("iqr_low_placement")
        if row["flag_iqr_slot_dominance"]:
            reasons.append("iqr_slot_dominance")
        if row["flag_mad_low_placement"]:
            reasons.append("mad_low_placement")
        if row["flag_mad_slot_dominance"]:
            reasons.append("mad_slot_dominance")
        return ";".join(reasons)

    rates["outlier_reasons"] = rates.apply(reason_row, axis=1)

    return rates


def summarize_global_slots(rates):
    rows = []
    for band, subset in rates.groupby("conf_band"):
        total = subset["total_slot_assignments"].sum()
        for slot in SLOTS:
            slot_total = subset[slot].sum()
            share = (slot_total / total) if total else 0.0
            rows.append(
                {
                    "conf_band": band,
                    "slot": slot,
                    "slot_assignment_count": int(slot_total),
                    "slot_share": float(share),
                }
            )
    return pd.DataFrame(rows)


def round_rate_columns(df):
    rate_cols = [
        "placed_rate_detection",
        "unplaced_rate_detection",
        "placed_rate_image",
        "unplaced_rate_image",
        "max_slot_share",
    ] + [f"share_{slot}" for slot in SLOTS]
    existing = [c for c in rate_cols if c in df.columns]
    df[existing] = df[existing].round(2)
    return df


def reorder_columns(df, slot_order):
    front = ["class_id", "class_name", "conf_band"]
    metric_cols = [
        "total_detections",
        "assigned_detections",
        "placed_rate_detection",
        "unplaced_rate_detection",
        "total_images",
        "assigned_images",
        "placed_rate_image",
        "unplaced_rate_image",
        "total_slot_assignments",
    ]
    slot_cols = [s for s in slot_order if s in df.columns]
    share_cols = [f"share_{s}" for s in slot_order if f"share_{s}" in df.columns]
    tail = [
        "max_slot_name",
        "max_slot_share",
        "outlier_flag_count",
        "outlier_reasons",
    ] + [c for c in df.columns if c.startswith("flag_")]

    ordered = [c for c in front + metric_cols + slot_cols + share_cols + tail if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


def write_outputs(output_dir, helper_table, intersect_table, rates, slot_counts, global_slots):
    os.makedirs(output_dir, exist_ok=True)

    rates = round_rate_columns(rates)
    rates = reorder_columns(rates, SLOT_OUTPUT_ORDER)

    slot_counts = slot_counts.copy()
    slot_order_map = {slot: i for i, slot in enumerate(SLOT_OUTPUT_ORDER)}
    slot_counts["slot_order"] = slot_counts["slot"].map(slot_order_map).fillna(999).astype(int)

    slot_matrix = (
        rates[rates["conf_band"] == "all_conf"]
        [["class_id", "class_name"] + SLOT_OUTPUT_ORDER + ["total_slot_assignments", "total_images"]]
        .sort_values("class_id")
    )

    slot_matrix_body_order = (
        rates[rates["conf_band"] == "all_conf"]
        [["class_id", "class_name"] + SLOT_BODY_ORDER + ["total_slot_assignments", "total_images"]]
        .sort_values("class_id")
    )

    outliers = rates[rates["outlier_flag_count"] > 0].copy()
    outliers = outliers.sort_values(["conf_band", "outlier_flag_count", "placed_rate_detection"], ascending=[True, False, True])

    rates.sort_values(["conf_band", "class_id"]).to_csv(
        os.path.join(output_dir, "class_rates.csv"), index=False
    )
    slot_counts.sort_values(["conf_band", "class_id", "slot_order", "slot"]).drop(columns=["slot_order"]).to_csv(
        os.path.join(output_dir, "class_slot_counts_long.csv"), index=False
    )
    slot_matrix.to_csv(os.path.join(output_dir, "class_slot_matrix_all_conf.csv"), index=False)
    slot_matrix_body_order.to_csv(
        os.path.join(output_dir, "class_slot_matrix_body_order_all_conf.csv"),
        index=False,
    )
    outliers.to_csv(os.path.join(output_dir, "outlier_summary.csv"), index=False)
    global_slots["slot_order"] = global_slots["slot"].map(slot_order_map).fillna(999).astype(int)
    global_slots.sort_values(["conf_band", "slot_order", "slot"]).drop(columns=["slot_order"]).to_csv(
        os.path.join(output_dir, "global_slot_shares.csv"), index=False
    )

    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "helper_table": helper_table,
        "intersect_table": intersect_table,
        "high_conf_threshold": HIGH_CONF_THRESHOLD,
        "placed_rate_detection_min": PLACED_RATE_DETECTION_MIN,
        "placed_rate_image_min": PLACED_RATE_IMAGE_MIN,
        "slot_dominance_max": SLOT_DOMINANCE_MAX,
        "waist_feet_min_share": WAIST_FEET_MIN_SHARE,
        "iqr_multiplier": IQR_MULTIPLIER,
        "mad_z_threshold": MAD_Z_THRESHOLD,
        "class_scope": {
            "coco_target_class_ids": sorted(set(COCO_TARGET_CLASS_IDS)),
            "custom_class_min_id": CUSTOM_CLASS_MIN_ID,
        },
        "slot_output_order": SLOT_OUTPUT_ORDER,
        "slot_body_order": SLOT_BODY_ORDER,
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run_one_stage(engine, helper_table, output_root, intersect_table=None):
    print(f"\n=== Running helper table: {helper_table} ===")
    with engine.connect() as conn:
        if not helper_exists(conn, helper_table):
            print(f"Skipping missing helper table: {helper_table}")
            return False
        if intersect_table and not helper_exists(conn, intersect_table):
            print(f"Skipping missing intersect table: {intersect_table}")
            return False

        build_temp_tables(conn, helper_table, intersect_table=intersect_table)
        denom, assigned, slot_counts = fetch_frames(conn)
        class_names = get_class_names(conn)

    rates, _ = build_rates(denom, assigned, slot_counts)
    rates = rates.merge(class_names, on="class_id", how="left")
    rates["class_name"] = rates["class_name"].fillna("unknown")

    slot_counts = slot_counts.merge(class_names, on="class_id", how="left")
    slot_counts["class_name"] = slot_counts["class_name"].fillna("unknown")

    rates = add_outlier_flags(rates)
    global_slots = summarize_global_slots(rates)

    output_dir = os.path.join(output_root, helper_table)
    write_outputs(output_dir, helper_table, intersect_table, rates, slot_counts, global_slots)

    all_conf = rates[rates["conf_band"] == "all_conf"]
    flagged = (all_conf["outlier_flag_count"] > 0).sum()
    print(
        f"Wrote outputs to {output_dir} | classes={len(all_conf)} | "
        f"flagged_classes={flagged}"
    )
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ImagesDetections object placement rates.")
    parser.add_argument(
        "--helper-table",
        type=str,
        default=DEFAULT_HELPER_TABLE,
        help="Single helper table to analyze.",
    )
    parser.add_argument(
        "--run-progression",
        action="store_true",
        help="Run full staged helper-table progression.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/Users/michaelmandiberg/Documents/GitHub/takingstock/analysis/imagesdetections_debug/object_placement_audit",
        help="Directory root for output artifacts.",
    )
    parser.add_argument(
        "--intersect-table",
        type=str,
        default=TESTING_SET_INTERSECT_TABLE,
        help="Optional second helper table to intersect on image_id.",
    )
    parser.add_argument(
        "--no-intersect",
        action="store_true",
        help="Disable image_id intersection with --intersect-table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    engine = get_engine()
    intersect_table = None if args.no_intersect else args.intersect_table

    if args.run_progression:
        for helper in HELPER_TABLE_PROGRESSION:
            # Progression mode analyzes each helper directly; intersection is for testing-set mode.
            run_one_stage(engine, helper, args.output_root, intersect_table=None)
    else:
        run_one_stage(engine, args.helper_table, args.output_root, intersect_table=intersect_table)


if __name__ == "__main__":
    main()
