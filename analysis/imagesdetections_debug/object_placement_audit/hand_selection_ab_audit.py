#!/usr/bin/env python3
"""A/B audit for hand winner selection on a fixed golden image set.

Compares:
  A) baseline hand ranking (distance-only)
  B) current hand ranking (distance + directional bias + conf + compat tier)

Outputs CSV files for quick review on another machine.

Example:
python analysis/imagesdetections_debug/object_placement_audit/hand_selection_ab_audit.py \
  --image-ids 396589,125734939 \
  --output-root /Users/michaelmandiberg/Documents/GitHub/facemap/analysis/imagesdetections_debug/object_placement_audit

python analysis/imagesdetections_debug/object_placement_audit/hand_selection_ab_audit.py \
  --image-ids-file /path/to/golden_image_ids.txt \
  --output-root /Users/michaelmandiberg/Documents/GitHub/facemap/analysis/imagesdetections_debug/object_placement_audit

python analysis/imagesdetections_debug/object_placement_audit/hand_selection_ab_audit.py \
  --helper-table SegmentHelper_TheOffice \
  --limit 500 \
  --offset 0 \
  --output-root /Users/michaelmandiberg/Documents/GitHub/facemap/analysis/imagesdetections_debug/object_placement_audit  

  """

import argparse
import csv
import os
import sys
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(1, REPO_ROOT)

from mp_db_io import DataIO  # noqa: E402
from mp_sort_pose import SortPose  # noqa: E402
from tools_clustering import ToolsClustering  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    "/Users/michaelmandiberg/Documents/GitHub/facemap/"
    "analysis/imagesdetections_debug/object_placement_audit"
)


def parse_args():
    parser = argparse.ArgumentParser(description="A/B audit for hand winner selection.")
    parser.add_argument("--image-ids", default=None, type=str, help="Comma-separated image_ids.")
    parser.add_argument("--image-ids-file", default=None, type=str, help="Path to text/csv with one image_id per line.")
    parser.add_argument("--helper-table", default=None, type=str, help="Optional helper table to pull image_ids from.")
    parser.add_argument("--limit", default=None, type=int, help="Optional cap for helper-table image_ids.")
    parser.add_argument("--offset", default=0, type=int, help="Optional helper-table offset.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=str)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose classification logging.")
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
            token = raw.split(",", 1)[0].strip()
            ids.append(int(token))
    return ids


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
    engine = create_engine(
        "mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
            user=db["user"],
            pw=db["pass"],
            db=db["name"],
            socket=db["unix_socket"],
        ),
        poolclass=NullPool,
    )
    return io, engine


def fetch_helper_image_ids(conn, helper_table, limit=None, offset=0):
    sql = f"SELECT DISTINCT image_id FROM {helper_table} ORDER BY image_id"
    if limit is not None:
        sql += " LIMIT :limit OFFSET :offset"
        rows = conn.execute(text(sql), {"limit": int(limit), "offset": int(offset)}).fetchall()
    else:
        rows = conn.execute(text(sql)).fetchall()
    return [int(r[0]) for r in rows]


def build_sortpose_for_knuckles():
    cfg = {
        "motion": {
            "side_to_side": False,
            "forward_smile": True,
            "laugh": False,
            "forward_nosmile": False,
            "static_pose": False,
            "simple": False,
        },
        "face_height_output": 500,
        "image_edge_multiplier": [2.2, 2.2, 2.6, 2.2],
        "EXPAND": False,
        "ONE_SHOT": False,
        "JUMP_SHOT": False,
        "HSV_CONTROL": None,
        "VERBOSE": False,
        "INPAINT": False,
        "SORT_TYPE": "ObjectFusion",
        "OBJ_CLS_ID": None,
        "UPSCALE_MODEL_PATH": None,
        "LMS_DIMENSIONS": 3,
    }
    return SortPose(config=cfg)


def fetch_pose_context(io_obj, image_id, sort_pose, default_hand_pos):
    try:
        series = io_obj.get_encodings_mongo(int(image_id))
    except Exception:
        return {
            "left_knuckle": list(default_hand_pos),
            "right_knuckle": list(default_hand_pos),
            "left_shoulder": None,
            "right_shoulder": None,
            "body_landmarks_normalized": None,
        }

    body_landmarks_normalized = None
    hand_results = None
    try:
        body_landmarks_normalized = io_obj.unpickle_array(series[3]) if series[3] is not None else None
    except Exception:
        body_landmarks_normalized = None

    try:
        hand_results = series[5]
    except Exception:
        hand_results = None

    left_knuckle = list(default_hand_pos)
    right_knuckle = list(default_hand_pos)
    try:
        lk, rk, _lsrc, _rsrc = sort_pose.prep_knuckle_landmarks(hand_results, body_landmarks_normalized)
        if lk is not None:
            left_knuckle = lk
        if rk is not None:
            right_knuckle = rk
    except Exception:
        pass

    # Use a non-verbose instance for geometry helper extraction.
    geom_tc = ToolsClustering(CLUSTER_TYPE="ObjectFusion", VERBOSE=False, session=None)
    left_shoulder, right_shoulder = geom_tc.extract_shoulder_points(body_landmarks_normalized)

    return {
        "left_knuckle": left_knuckle,
        "right_knuckle": right_knuckle,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "body_landmarks_normalized": body_landmarks_normalized,
    }


def detection_payload(det):
    if det is None:
        return {"detection_id": None, "class_id": None, "conf": None}
    return {
        "detection_id": int(det.get("detection_id")) if det.get("detection_id") is not None else None,
        "class_id": int(float(det.get("class_id"))) if det.get("class_id") is not None else None,
        "conf": float(det.get("conf")) if det.get("conf") is not None else None,
    }


def run_variant(tc, image_id, ctx):
    return tc.query_and_classify_detections(
        image_id=int(image_id),
        left_knuckle=ctx["left_knuckle"],
        right_knuckle=ctx["right_knuckle"],
        left_shoulder=ctx["left_shoulder"],
        right_shoulder=ctx["right_shoulder"],
        body_landmarks_normalized=ctx["body_landmarks_normalized"],
    )


def build_row(image_id, side, baseline_det, new_det):
    b = detection_payload(baseline_det)
    n = detection_payload(new_det)
    changed = b["detection_id"] != n["detection_id"]
    return {
        "image_id": int(image_id),
        "hand_side": side,
        "changed": int(changed),
        "baseline_detection_id": b["detection_id"],
        "baseline_class_id": b["class_id"],
        "baseline_conf": b["conf"],
        "new_detection_id": n["detection_id"],
        "new_class_id": n["class_id"],
        "new_conf": n["conf"],
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    io_obj, engine = get_engine()

    image_ids = []
    image_ids.extend(parse_image_ids_arg(args.image_ids))
    if args.image_ids_file:
        image_ids.extend(load_image_ids_file(args.image_ids_file))

    with engine.connect() as conn:
        if args.helper_table:
            image_ids.extend(
                fetch_helper_image_ids(conn, args.helper_table, limit=args.limit, offset=args.offset)
            )

    image_ids = dedupe_keep_order(image_ids)
    if not image_ids:
        raise SystemExit("No image_ids supplied. Use --image-ids, --image-ids-file, or --helper-table.")

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(args.output_root, f"hand_ab_audit_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # metadata = MetaData(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # A: baseline (distance-only) by zeroing non-distance hand weights.
    tc_baseline = ToolsClustering(CLUSTER_TYPE="ObjectFusion", VERBOSE=bool(args.verbose), session=None)
    tc_baseline.session = session
    tc_baseline.HAND_DIRECTIONAL_BIAS_WEIGHT = 0.0
    tc_baseline.HAND_CONFIDENCE_WEIGHT = 0.0
    tc_baseline.HAND_COMPAT_LEVEL_WEIGHT = 0.0

    # B: current behavior (directional + confidence + compat level weights enabled).
    tc_new = ToolsClustering(CLUSTER_TYPE="ObjectFusion", VERBOSE=bool(args.verbose), session=None)
    tc_new.session = session

    sort_pose = build_sortpose_for_knuckles()
    default_hand_pos = list(tc_new.DEFAULT_HAND_POSITION)

    per_hand_rows = []
    per_image_rows = []

    for idx, image_id in enumerate(image_ids, start=1):
        ctx = fetch_pose_context(io_obj, image_id, sort_pose, default_hand_pos)

        baseline = run_variant(tc_baseline, image_id, ctx)
        new = run_variant(tc_new, image_id, ctx)

        left_row = build_row(
            image_id,
            "left",
            baseline.get("left_hand_object"),
            new.get("left_hand_object"),
        )
        right_row = build_row(
            image_id,
            "right",
            baseline.get("right_hand_object"),
            new.get("right_hand_object"),
        )
        per_hand_rows.append(left_row)
        per_hand_rows.append(right_row)

        per_image_rows.append(
            {
                "image_id": int(image_id),
                "left_changed": int(left_row["changed"]),
                "right_changed": int(right_row["changed"]),
                "any_changed": int(bool(left_row["changed"] or right_row["changed"])),
            }
        )

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(image_ids)} images")

    changed_rows = [r for r in per_hand_rows if int(r["changed"]) == 1]
    total_slots = len(per_hand_rows)
    changed_slots = len(changed_rows)
    total_images = len(per_image_rows)
    changed_images = sum(int(r["any_changed"]) for r in per_image_rows)

    per_hand_path = os.path.join(out_dir, "hand_ab_per_hand.csv")
    changed_path = os.path.join(out_dir, "hand_ab_changed_only.csv")
    summary_path = os.path.join(out_dir, "hand_ab_summary_by_image.csv")
    ids_path = os.path.join(out_dir, "input_image_ids.csv")

    write_csv(
        per_hand_path,
        per_hand_rows,
        [
            "image_id",
            "hand_side",
            "changed",
            "baseline_detection_id",
            "baseline_class_id",
            "baseline_conf",
            "new_detection_id",
            "new_class_id",
            "new_conf",
        ],
    )

    write_csv(
        changed_path,
        changed_rows,
        [
            "image_id",
            "hand_side",
            "changed",
            "baseline_detection_id",
            "baseline_class_id",
            "baseline_conf",
            "new_detection_id",
            "new_class_id",
            "new_conf",
        ],
    )

    write_csv(
        summary_path,
        per_image_rows,
        ["image_id", "left_changed", "right_changed", "any_changed"],
    )

    write_csv(ids_path, [{"image_id": int(i)} for i in image_ids], ["image_id"])

    print("Hand A/B audit complete")
    print(f"Output directory: {out_dir}")
    print(f"Images evaluated: {total_images}")
    print(f"Slots changed: {changed_slots}/{total_slots}")
    print(f"Images with any hand change: {changed_images}/{total_images}")
    print(f"CSV: {per_hand_path}")
    print(f"CSV: {changed_path}")
    print(f"CSV: {summary_path}")
    print(f"CSV: {ids_path}")


if __name__ == "__main__":
    main()
