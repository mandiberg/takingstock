#!/usr/bin/env python3
"""A/B audit for pre-placement suppression order on a fixed image set.

This script compares:
  A) suppression-first: suppress -> overlap
  B) current pipeline: overlap -> suppress

It writes CSV artifacts to support golden-set review.

Example:
python analysis/imagesdetections_debug/object_placement_audit/preplacement_ab_audit.py \
  --image-ids 396589,125734939 \
  --output-root /Users/michaelmandiberg/Documents/GitHub/facemap/analysis/imagesdetections_debug/object_placement_audit
"""

import argparse
import copy
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.pool import NullPool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(1, REPO_ROOT)

from mp_db_io import DataIO  # noqa: E402
from tools_clustering import ToolsClustering  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    "/Users/michaelmandiberg/Documents/GitHub/facemap/"
    "analysis/imagesdetections_debug/object_placement_audit"
)


def validate_unix_socket(db_cfg):
    socket_path = db_cfg.get("unix_socket")
    if not socket_path:
        raise RuntimeError(
            "MySQL unix_socket is empty in DataIO db config. "
            "Check [local_settings].unix_socket in config.ini."
        )
    if not os.path.exists(socket_path):
        raise RuntimeError(
            "MySQL unix_socket path does not exist: "
            f"{socket_path}. "
            "Check [local_settings].unix_socket in config.ini and that MySQL is running."
        )


def get_engine():
    io = DataIO()
    db = io.db
    validate_unix_socket(db)
    return create_engine(
        "mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
            user=db["user"],
            pw=db["pass"],
            db=db["name"],
            socket=db["unix_socket"],
        ),
        poolclass=NullPool,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="A/B audit for pre-placement suppression ordering.")
    parser.add_argument("--image-ids", default=None, type=str, help="Comma-separated image_ids.")
    parser.add_argument("--image-ids-file", default=None, type=str, help="Path to text/csv with one image_id per line.")
    parser.add_argument("--helper-table", default=None, type=str, help="Optional helper table to pull image_ids from.")
    parser.add_argument("--limit", default=None, type=int, help="Optional cap for helper-table image_ids.")
    parser.add_argument("--offset", default=0, type=int, help="Optional helper-table offset.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=str)
    return parser.parse_args()


def parse_image_ids_arg(value):
    if not value:
        return []
    ids = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        ids.append(int(token))
    return ids


def load_image_ids_file(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            # allow "123,optional_note" input
            token = raw.split(",", 1)[0].strip()
            ids.append(int(token))
    return ids


def fetch_helper_image_ids(conn, helper_table, limit=None, offset=0):
    sql = f"SELECT DISTINCT image_id FROM {helper_table} ORDER BY image_id"
    if limit is not None:
        sql += " LIMIT :limit OFFSET :offset"
        rows = conn.execute(text(sql), {"limit": int(limit), "offset": int(offset)}).fetchall()
    else:
        rows = conn.execute(text(sql)).fetchall()
    return [int(r[0]) for r in rows]


def dedupe_keep_order(ids):
    out = []
    seen = set()
    for value in ids:
        x = int(value)
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def build_detection_rows(conn, image_ids):
    if not image_ids:
        return []

    sql = text(
        """
        SELECT detection_id, image_id, class_id, conf, bbox_norm
        FROM Detections
        WHERE image_id IN :image_ids
        ORDER BY image_id, detection_id
        """
    ).bindparams(bindparam("image_ids", expanding=True))

    rows = conn.execute(sql, {"image_ids": list(image_ids)}).fetchall()
    return rows


def parse_detections_for_image(raw_rows, cl):
    detections = []
    skipped_bbox_parse = 0
    below_conf = 0

    for row in raw_rows:
        conf = float(row[3])
        if conf <= cl.MIN_DETECTION_CONFIDENCE:
            below_conf += 1
            continue

        bbox = cl.parse_bbox_norm(row[4])
        if bbox is None:
            skipped_bbox_parse += 1
            continue

        detections.append(
            {
                "detection_id": int(row[0]),
                "image_id": int(row[1]),
                "class_id": int(row[2]),
                "class_id_raw": int(row[2]),
                "conf": conf,
                "bbox": bbox,
                "top": bbox["top"],
                "left": bbox["left"],
                "right": bbox["right"],
                "bottom": bbox["bottom"],
            }
        )

    return detections, skipped_bbox_parse, below_conf


def stats_by_rule(stats_dict):
    by_rule = stats_dict.get("by_rule", {}) if stats_dict else {}
    out = {}
    for rule_id, payload in by_rule.items():
        out[str(rule_id)] = {
            "candidates": int(payload.get("candidates", 0)),
            "suppressed": int(payload.get("suppressed", 0)),
            "dual_keep": int(payload.get("dual_keep", 0)),
            "new_bonus_win": int(payload.get("new_bonus_win", 0)),
            "custom_tie_win": int(payload.get("custom_tie_win", 0)),
        }
    return out


def run_ab_for_image(image_id, detections, cl):
    # A: suppression-first, then overlap.
    cl.reset_preplacement_suppression_stats()
    a_after_suppress = cl.apply_preplacement_suppression(copy.deepcopy(detections), image_id=image_id)
    a_stats = cl.get_preplacement_suppression_stats() or {}
    a_final = cl.resolve_overlapping_detections(copy.deepcopy(a_after_suppress))

    # B: overlap-first (current), then suppression.
    cl.reset_preplacement_suppression_stats()
    b_after_overlap = cl.resolve_overlapping_detections(copy.deepcopy(detections))
    b_final = cl.apply_preplacement_suppression(copy.deepcopy(b_after_overlap), image_id=image_id)
    b_stats = cl.get_preplacement_suppression_stats() or {}

    return {
        "a_after_suppress": a_after_suppress,
        "a_final": a_final,
        "a_stats": a_stats,
        "b_after_overlap": b_after_overlap,
        "b_final": b_final,
        "b_stats": b_stats,
    }


def flatten_events(events, scenario_label):
    rows = []
    for event in events or []:
        row = dict(event)
        row["scenario"] = scenario_label
        rows.append(row)
    return rows


def main():
    args = parse_args()

    input_ids = []
    if args.image_ids:
        input_ids.extend(parse_image_ids_arg(args.image_ids))
    if args.image_ids_file:
        input_ids.extend(load_image_ids_file(args.image_ids_file))

    engine = get_engine()
    with engine.connect() as conn:
        if args.helper_table:
            input_ids.extend(fetch_helper_image_ids(conn, args.helper_table, limit=args.limit, offset=args.offset))

    image_ids = dedupe_keep_order(input_ids)
    if not image_ids:
        raise RuntimeError("No image_ids provided. Use --image-ids, --image-ids-file, or --helper-table.")

    run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(args.output_root, "ab_audits", f"ab_audit_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)

    cl = ToolsClustering("ObjectFusion", VERBOSE=False, session=None)

    with engine.connect() as conn:
        raw_rows = build_detection_rows(conn, image_ids)

    rows_by_image = defaultdict(list)
    for row in raw_rows:
        rows_by_image[int(row[1])].append(row)

    summary_rows = []
    events_rows = []
    rule_rows = []
    final_det_rows = []

    for image_id in image_ids:
        raw_for_image = rows_by_image.get(int(image_id), [])
        detections, skipped_bbox_parse, below_conf = parse_detections_for_image(raw_for_image, cl)
        ab = run_ab_for_image(int(image_id), detections, cl)

        a_stats = ab["a_stats"]
        b_stats = ab["b_stats"]

        summary_rows.append(
            {
                "image_id": int(image_id),
                "raw_detections_total": len(raw_for_image),
                "below_conf_filtered": int(below_conf),
                "bbox_parse_fail_filtered": int(skipped_bbox_parse),
                "parsed_detections_total": len(detections),
                "a_after_suppress_count": len(ab["a_after_suppress"]),
                "a_final_count": len(ab["a_final"]),
                "b_after_overlap_count": len(ab["b_after_overlap"]),
                "b_final_count": len(ab["b_final"]),
                "a_candidates_total": int(a_stats.get("pair_candidates_total", 0)) + int(a_stats.get("family_candidates_total", 0)),
                "b_candidates_total": int(b_stats.get("pair_candidates_total", 0)) + int(b_stats.get("family_candidates_total", 0)),
                "a_suppressed_total": int(a_stats.get("suppressed_total", 0)),
                "b_suppressed_total": int(b_stats.get("suppressed_total", 0)),
                "suppressed_delta_a_minus_b": int(a_stats.get("suppressed_total", 0)) - int(b_stats.get("suppressed_total", 0)),
                "a_dual_keep_total": int(a_stats.get("dual_keep_total", 0)),
                "b_dual_keep_total": int(b_stats.get("dual_keep_total", 0)),
            }
        )

        events_rows.extend(flatten_events(a_stats.get("events", []), "A_suppress_then_overlap"))
        events_rows.extend(flatten_events(b_stats.get("events", []), "B_overlap_then_suppress"))

        a_by_rule = stats_by_rule(a_stats)
        b_by_rule = stats_by_rule(b_stats)
        all_rules = sorted(set(a_by_rule.keys()) | set(b_by_rule.keys()))
        for rule_id in all_rules:
            a_rule = a_by_rule.get(rule_id, {})
            b_rule = b_by_rule.get(rule_id, {})
            rule_rows.append(
                {
                    "image_id": int(image_id),
                    "rule_id": rule_id,
                    "a_candidates": int(a_rule.get("candidates", 0)),
                    "b_candidates": int(b_rule.get("candidates", 0)),
                    "candidate_delta_a_minus_b": int(a_rule.get("candidates", 0)) - int(b_rule.get("candidates", 0)),
                    "a_suppressed": int(a_rule.get("suppressed", 0)),
                    "b_suppressed": int(b_rule.get("suppressed", 0)),
                    "suppressed_delta_a_minus_b": int(a_rule.get("suppressed", 0)) - int(b_rule.get("suppressed", 0)),
                    "a_dual_keep": int(a_rule.get("dual_keep", 0)),
                    "b_dual_keep": int(b_rule.get("dual_keep", 0)),
                    "dual_keep_delta_a_minus_b": int(a_rule.get("dual_keep", 0)) - int(b_rule.get("dual_keep", 0)),
                }
            )

        for scenario, dets in (
            ("A_final", ab["a_final"]),
            ("B_final", ab["b_final"]),
        ):
            for det in dets:
                bbox = det.get("bbox", {}) or {}
                final_det_rows.append(
                    {
                        "scenario": scenario,
                        "image_id": int(image_id),
                        "detection_id": int(det.get("detection_id")),
                        "class_id": int(det.get("class_id_raw", det.get("class_id"))),
                        "conf": float(det.get("conf", 0.0)),
                        "top": bbox.get("top"),
                        "left": bbox.get("left"),
                        "right": bbox.get("right"),
                        "bottom": bbox.get("bottom"),
                    }
                )

    summary_path = os.path.join(out_dir, "ab_summary_by_image.csv")
    rules_path = os.path.join(out_dir, "ab_rule_comparison.csv")
    events_path = os.path.join(out_dir, "ab_events.csv")
    finals_path = os.path.join(out_dir, "ab_final_detections.csv")
    ids_path = os.path.join(out_dir, "input_image_ids.csv")

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(rule_rows).to_csv(rules_path, index=False)
    pd.DataFrame(events_rows).to_csv(events_path, index=False)
    pd.DataFrame(final_det_rows).to_csv(finals_path, index=False)

    with open(ids_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id"])
        for image_id in image_ids:
            writer.writerow([int(image_id)])

    print("A/B audit complete")
    print(f"image_count={len(image_ids)}")
    print(f"summary={summary_path}")
    print(f"rule_comparison={rules_path}")
    print(f"events={events_path}")
    print(f"final_detections={finals_path}")
    print(f"input_ids={ids_path}")


if __name__ == "__main__":
    main()
