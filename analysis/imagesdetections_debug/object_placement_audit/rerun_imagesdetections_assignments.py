#!/usr/bin/env python3
"""Assignment-only rerun for ImagesDetections on a helper-table subset.

This script avoids the interactive Clustering_SQL flow and does not run KMeans.
It recomputes object slot assignments using existing assignment logic in
ToolsClustering.process_detections_for_df, then rewrites ImagesDetections rows
for only the requested helper-table image_ids.

python analysis/imagesdetections_debug/object_placement_audit/rerun_imagesdetections_assignments.py \
  --helper-table SegmentHelper_TheOffice \
  --output-root /Users/michaelmandiberg/Documents/GitHub/facemap/analysis/imagesdetections_debug/object_placement_audit \
  --skip-backup \
  --dry-run

--delete-chunk-size 2000
  --find-checkpoint

"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial

import pandas as pd
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(1, REPO_ROOT)

from mp_db_io import DataIO  # noqa: E402
from mp_sort_pose import SortPose  # noqa: E402
from tools_clustering import ToolsClustering  # noqa: E402
DEFAULT_HELPER_TABLE = "SegmentHelper_may26_deleteme_missingArms3D"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DELETE_CHUNK_SIZE = 5000
DEFAULT_OUTPUT_ROOT = (
    "/Users/michaelmandiberg/Documents/GitHub/takingstock/"
    "analysis/imagesdetections_debug/object_placement_audit"
)

OBJECT_COLS = [
    "left_hand_object",
    "right_hand_object",
    "top_face_object",
    "left_eye_object",
    "right_eye_object",
    "mouth_object",
    "shoulder_object",
    "waist_object",
    "feet_object",
]

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


def helper_exists(conn, helper_table):
    q = text("SHOW TABLES LIKE :tbl")
    return conn.execute(q, {"tbl": helper_table}).fetchone() is not None


def get_imagesdetections_columns(conn):
    cols = conn.execute(text("SHOW COLUMNS FROM ImagesDetections")).fetchall()
    return [row[0] for row in cols]


def fetch_helper_image_ids(conn, helper_table, limit=None, offset=0):
    sql = f"SELECT DISTINCT image_id FROM {helper_table} ORDER BY image_id"
    if limit is not None:
        sql += " LIMIT :limit OFFSET :offset"
        rows = conn.execute(text(sql), {"limit": int(limit), "offset": int(offset)}).fetchall()
    else:
        rows = conn.execute(text(sql)).fetchall()
    return [int(r[0]) for r in rows]


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_checkpoint(checkpoint_path):
    """Load checkpoint state if it exists. Returns dict or None."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def save_checkpoint(checkpoint_path, state):
    """Save checkpoint state to file."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def find_existing_checkpoint(helper_table, output_dir):
    """Find the most recent checkpoint for the given helper table (any date)."""
    assignment_reruns_dir = os.path.join(output_dir, "assignment_reruns")
    if not os.path.exists(assignment_reruns_dir):
        return None

    checkpoint_dir = os.path.join(assignment_reruns_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None

    prefix = f"checkpoint_{helper_table}_"
    matches = [
        os.path.join(checkpoint_dir, fname)
        for fname in os.listdir(checkpoint_dir)
        if fname.startswith(prefix) and fname.endswith(".json")
    ]
    if not matches:
        return None
    # Return most recently modified checkpoint
    return max(matches, key=os.path.getmtime)


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


def _fetch_one(io, image_id):
    """Fetch encodings for a single image_id. Designed to be called from threads."""
    try:
        series = io.get_encodings_mongo(int(image_id))
        return {
            "image_id": int(image_id),
            "body_landmarks_normalized": io.unpickle_array(series[3]),
            "hand_results": series[5],
        }
    except Exception:
        return {
            "image_id": int(image_id),
            "body_landmarks_normalized": None,
            "hand_results": None,
        }


def fetch_mongo_batch(io, image_ids, num_workers=8):
    """Fetch a batch of images from MongoDB concurrently using a thread pool.

    MongoClient is thread-safe; threads share the single io instance safely.
    executor.map preserves input order.
    """
    fetch_fn = partial(_fetch_one, io)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        rows = list(executor.map(fetch_fn, image_ids))
    return pd.DataFrame(rows)


def add_knuckle_columns(df, sort_pose):
    knuckles = df.apply(
        lambda row: sort_pose.prep_knuckle_landmarks(
            row.get("hand_results"), row.get("body_landmarks_normalized")
        ),
        axis=1,
    ).tolist()
    df[["left_pointer_knuckle_norm", "right_pointer_knuckle_norm", "left_source", "right_source"]] = pd.DataFrame(
        knuckles,
        index=df.index,
    )
    return df


def max_detection_id_by_image(conn, image_ids):
    if not image_ids:
        return {}
    sql = text(
        """
        SELECT image_id, MAX(detection_id) AS max_detection_id
        FROM Detections
        WHERE image_id IN :image_ids
        GROUP BY image_id
        """
    ).bindparams(bindparam("image_ids", expanding=True))
    rows = conn.execute(sql, {"image_ids": list(image_ids)}).fetchall()
    return {int(r[0]): int(r[1]) for r in rows if r[1] is not None}


def extract_detection_id(value):
    if value is None:
        return None
    if isinstance(value, dict):
        det = value.get("detection_id")
        return int(det) if det is not None else None
    try:
        return int(value)
    except Exception:
        return None


def make_insert_frame(df, include_last_reprocessed):
    records = []
    for _, row in df.iterrows():
        left_knuckle = row.get("left_pointer_knuckle_norm") or [0.0, 8.0, 0.0]
        right_knuckle = row.get("right_pointer_knuckle_norm") or [0.0, 8.0, 0.0]

        rec = {
            "image_id": int(row["image_id"]),
            "left_pointer_x": float(left_knuckle[0]) if len(left_knuckle) > 0 else None,
            "left_pointer_y": float(left_knuckle[1]) if len(left_knuckle) > 1 else None,
            "left_source": row.get("left_source") or "default",
            "right_pointer_x": float(right_knuckle[0]) if len(right_knuckle) > 0 else None,
            "right_pointer_y": float(right_knuckle[1]) if len(right_knuckle) > 1 else None,
            "right_source": row.get("right_source") or "default",
            "left_hand_object_id": extract_detection_id(row.get("left_hand_object")),
            "right_hand_object_id": extract_detection_id(row.get("right_hand_object")),
            "top_face_object_id": extract_detection_id(row.get("top_face_object")),
            "left_eye_object_id": extract_detection_id(row.get("left_eye_object")),
            "right_eye_object_id": extract_detection_id(row.get("right_eye_object")),
            "mouth_object_id": extract_detection_id(row.get("mouth_object")),
            "shoulder_object_id": extract_detection_id(row.get("shoulder_object")),
            "waist_object_id": extract_detection_id(row.get("waist_object")),
            "feet_object_id": extract_detection_id(row.get("feet_object")),
        }
        if include_last_reprocessed:
            rec["last_reprocessed_detection_id"] = row.get("last_reprocessed_detection_id")
        records.append(rec)
    return pd.DataFrame(records)


def ensure_subset_backup(conn, helper_table, backup_table):
    conn.execute(text(f"CREATE TABLE IF NOT EXISTS {backup_table} LIKE ImagesDetections"))
    conn.execute(
        text(
            f"""
            INSERT IGNORE INTO {backup_table}
            SELECT idt.*
            FROM ImagesDetections idt
            INNER JOIN {helper_table} sh ON sh.image_id = idt.image_id
            """
        )
    )


def delete_subset_rows_batched(engine, image_ids, delete_chunk_size=DEFAULT_DELETE_CHUNK_SIZE):
    """Delete existing ImagesDetections rows for target image_ids in small committed chunks.

    This avoids one giant transaction that can hit lock-wait timeouts.
    """
    if not image_ids:
        return 0

    sql = text("DELETE FROM ImagesDetections WHERE image_id IN :image_ids").bindparams(
        bindparam("image_ids", expanding=True)
    )

    total_deleted = 0
    batches = list(chunked(list(image_ids), int(delete_chunk_size)))
    total_batches = len(batches)

    for idx, batch_ids in enumerate(batches, start=1):
        with engine.begin() as conn:
            result = conn.execute(sql, {"image_ids": [int(x) for x in batch_ids]})
            total_deleted += int(result.rowcount or 0)
        if idx % 20 == 0 or idx == total_batches:
            print(
                f"[initial-delete {idx}/{total_batches}] "
                f"deleted_rows_so_far={total_deleted}"
            )

    return total_deleted


def parse_args():
    parser = argparse.ArgumentParser(description="Rerun ImagesDetections assignment-only for helper subset.")
    parser.add_argument("--helper-table", default=DEFAULT_HELPER_TABLE, type=str)
    parser.add_argument("--chunk-size", default=DEFAULT_CHUNK_SIZE, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Optional limit for testing.")
    parser.add_argument("--offset", default=0, type=int, help="Optional offset for testing.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=str)
    parser.add_argument("--backup-table", default=None, type=str)
    parser.add_argument("--skip-backup", action="store_true")
    parser.add_argument("--skip-initial-delete", action="store_true")
    parser.add_argument("--checkpoint-file", default=None, type=str, help="Checkpoint file to resume from.")
    parser.add_argument("--find-checkpoint", action="store_true", help="Auto-find checkpoint for this helper table.")
    parser.add_argument("--cleanup-checkpoint", action="store_true", help="Delete checkpoint file on successful completion.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mongo-workers", default=8, type=int, help="Number of threads for parallel MongoDB fetching (default: 8).")
    parser.add_argument(
        "--delete-chunk-size",
        default=DEFAULT_DELETE_CHUNK_SIZE,
        type=int,
        help="Chunk size for initial delete phase (default: 5000).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    engine = get_engine()
    Session = sessionmaker(bind=engine)

    # Determine run_id and checkpoint path
    run_date = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    checkpoint_file = args.checkpoint_file
    
    # Auto-find checkpoint if requested
    if args.find_checkpoint and not checkpoint_file:
        checkpoint_file = find_existing_checkpoint(args.helper_table, args.output_root)
        if checkpoint_file:
            print(f"Found checkpoint: {checkpoint_file}")
    
    # Load checkpoint state if available
    checkpoint = load_checkpoint(checkpoint_file) if checkpoint_file else None
    if checkpoint:
        print(f"Resuming from checkpoint: completed batch {checkpoint.get('last_completed_batch_num')}")
        # Auto-enable skip_backup on resume
        if not args.skip_backup:
            args.skip_backup = True
            print("Auto-enabling --skip-backup for resumed run")

    run_id = checkpoint.get("run_id") if checkpoint else datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    suffix = args.helper_table.split("_")[-1] if "_" in args.helper_table else args.helper_table
    backup_table = args.backup_table or f"ImagesDetections_backup_{suffix}_{run_id}".replace("-", "_")

    output_dir = os.path.join(args.output_root, "assignment_reruns")
    os.makedirs(output_dir, exist_ok=True)
    
    # Checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Generate checkpoint path if not resuming
    if not checkpoint_file:
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{args.helper_table}_{run_date}.json")
    
    run_meta_path = os.path.join(output_dir, f"run_{args.helper_table}_{run_id}.json")

    io = DataIO()
    io.query_face = False
    io.query_body = True
    io.query_hands = True

    sort_pose = build_sortpose_for_knuckles()

    with engine.begin() as conn:
        if not helper_exists(conn, args.helper_table):
            raise RuntimeError(f"Helper table not found: {args.helper_table}")
        table_cols = get_imagesdetections_columns(conn)

    include_last_reprocessed = "last_reprocessed_detection_id" in table_cols

    with engine.connect() as conn:
        image_ids = fetch_helper_image_ids(conn, args.helper_table, limit=args.limit, offset=args.offset)

    if not image_ids:
        raise RuntimeError("No image_ids found for helper table scope.")

    # Convert to list of batches upfront for checkpoint resuming
    all_batches = list(enumerate(chunked(image_ids, args.chunk_size), start=1))
    total_batches = len(all_batches)
    
    # Determine starting batch
    start_batch = 1
    if checkpoint:
        start_batch = checkpoint.get("last_completed_batch_num", 0) + 1
        print(f"Resuming from batch {start_batch} of {total_batches}")

    print(f"Helper table: {args.helper_table}")
    print(f"Image scope size: {len(image_ids)}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Total batches: {total_batches}")
    print(f"Dry run: {args.dry_run}")
    print(f"Starting batch: {start_batch}")
    print(f"Mongo workers: {args.mongo_workers}")

    if start_batch == 1 and not args.skip_backup and not args.dry_run:
        with engine.begin() as conn:
            print(f"Creating/updating subset backup table: {backup_table}")
            ensure_subset_backup(conn, args.helper_table, backup_table)

    if start_batch == 1 and not args.skip_initial_delete and not args.dry_run:
        print(
            "Initial subset delete starting in batches: "
            f"chunk_size={int(args.delete_chunk_size)}"
        )
        deleted = delete_subset_rows_batched(
            engine,
            image_ids,
            delete_chunk_size=int(args.delete_chunk_size),
        )
        print(f"Initial subset delete complete. Rows deleted: {deleted}")

    total_written = checkpoint.get("total_written", 0) if checkpoint else 0
    total_processed = checkpoint.get("total_processed", 0) if checkpoint else 0
    suppression_totals = {
        "enabled": None,
        "iou_threshold": None,
        "new_bonus": None,
        "pair_candidates_total": 0,
        "family_candidates_total": 0,
        "suppressed_total": 0,
        "suppressed_pair_total": 0,
        "suppressed_family_total": 0,
        "dual_keep_total": 0,
        "new_bonus_win_total": 0,
        "custom_tie_win_total": 0,
        "by_rule": {},
        "events": [],
    }

    # Create ToolsClustering once — avoids re-reading the compatibility matrix CSV
    # on every batch. Session is swapped per-batch below.
    cl = ToolsClustering("ObjectFusion", VERBOSE=False, session=None)

    print(f"Starting batch processing from batch {start_batch} of {total_batches}...")
    for batch_num, batch_ids in all_batches:
        if batch_num < start_batch:
            continue
        print(f"Processing batch {batch_num}/{total_batches} with {len(batch_ids)} images...")
        batch_ids = list(batch_ids)
        batch_df = fetch_mongo_batch(io, batch_ids, num_workers=args.mongo_workers)
        batch_df = add_knuckle_columns(batch_df, sort_pose)

        session = Session()
        cl.session = session
        batch_suppression = None

        try:
            batch_df = cl.process_detections_for_df(batch_df)
            print(f"Batch {batch_num} processing complete. Fetching pre-placement suppression stats...")
            batch_suppression = cl.get_preplacement_suppression_stats()
            print(f"Batch {batch_num} suppression stats: {batch_suppression}")
        finally:
            session.close()
            cl.session = None

        if batch_suppression:
            suppression_totals["enabled"] = batch_suppression.get("enabled")
            suppression_totals["iou_threshold"] = batch_suppression.get("iou_threshold")
            suppression_totals["new_bonus"] = batch_suppression.get("new_bonus")
            for key in (
                "pair_candidates_total",
                "family_candidates_total",
                "suppressed_total",
                "suppressed_pair_total",
                "suppressed_family_total",
                "dual_keep_total",
                "new_bonus_win_total",
                "custom_tie_win_total",
            ):
                suppression_totals[key] += int(batch_suppression.get(key, 0))

            for rule_id, counts in batch_suppression.get("by_rule", {}).items():
                if rule_id not in suppression_totals["by_rule"]:
                    suppression_totals["by_rule"][rule_id] = {
                        "candidates": 0,
                        "suppressed": 0,
                        "dual_keep": 0,
                        "new_bonus_win": 0,
                        "custom_tie_win": 0,
                    }
                for key in ("candidates", "suppressed", "dual_keep", "new_bonus_win", "custom_tie_win"):
                    suppression_totals["by_rule"][rule_id][key] += int(counts.get(key, 0))

            if len(suppression_totals["events"]) < 200:
                needed = 200 - len(suppression_totals["events"])
                suppression_totals["events"].extend(batch_suppression.get("events", [])[:needed])

        with engine.connect() as conn:
            max_det_map = max_detection_id_by_image(conn, batch_ids)

        if include_last_reprocessed:
            batch_df["last_reprocessed_detection_id"] = batch_df["image_id"].map(max_det_map)

        insert_df = make_insert_frame(batch_df, include_last_reprocessed)

        if not args.dry_run:
            with engine.begin() as conn:
                # Keep chunk-level idempotency and allow restarts.
                ids_sql = ",".join(str(int(x)) for x in batch_ids)
                conn.execute(text(f"DELETE FROM ImagesDetections WHERE image_id IN ({ids_sql})"))
                if len(insert_df) > 0:
                    insert_df.to_sql("ImagesDetections", con=conn, if_exists="append", index=False, method="multi")

        total_processed += len(batch_ids)
        total_written += len(insert_df)

        # Save checkpoint after successful batch write
        if not args.dry_run:
            checkpoint_state = {
                "run_id": run_id,
                "helper_table": args.helper_table,
                "last_completed_batch_num": batch_num,
                "total_batches": total_batches,
                "total_processed": int(total_processed),
                "total_written": int(total_written),
                "checkpoint_saved_at": datetime.utcnow().isoformat() + "Z",
            }
            save_checkpoint(checkpoint_file, checkpoint_state)

        if batch_num % 10 == 0 or total_processed == len(image_ids):
            print(
                f"[batch {batch_num}/{total_batches}] processed={total_processed}/{len(image_ids)} "
                f"written={total_written} suppressed={suppression_totals['suppressed_total']}"
            )

    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "helper_table": args.helper_table,
        "backup_table": None if args.skip_backup else backup_table,
        "skip_backup": bool(args.skip_backup),
        "skip_initial_delete": bool(args.skip_initial_delete),
        "dry_run": bool(args.dry_run),
        "chunk_size": int(args.chunk_size),
        "offset": int(args.offset),
        "limit": args.limit,
        "image_count": int(len(image_ids)),
        "total_batches": total_batches,
        "total_processed": int(total_processed),
        "total_written": int(total_written),
        "include_last_reprocessed_detection_id": bool(include_last_reprocessed),
        "resumed_from_checkpoint": bool(checkpoint),
        "checkpoint_file": checkpoint_file if checkpoint else None,
        "preplacement_suppression": suppression_totals,
    }

    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Run metadata written: {run_meta_path}")
    print(f"Checkpoint file: {checkpoint_file}")
    
    # Clean up checkpoint only if requested
    if args.cleanup_checkpoint and os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print(f"Checkpoint removed (--cleanup-checkpoint flag): {checkpoint_file}")
        except Exception as e:
            print(f"Warning: Could not remove checkpoint: {e}")
    else:
        if os.path.exists(checkpoint_file):
            print(f"Checkpoint retained for inspection (use --cleanup-checkpoint to remove on next run)")


if __name__ == "__main__":
    main()
