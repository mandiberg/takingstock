import os
import sys
import argparse
import time
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import pymongo

# importing from main github folder (derive from this script location)
ROOT_GITHUB = str(Path(__file__).resolve().parents[1])
sys.path.insert(1, ROOT_GITHUB)

from mp_db_io import DataIO
from my_declarative_base import Encodings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit Encodings mongo_* flags vs Mongo docs and set missing_mongo_* fields."
    )
    parser.add_argument("--batch-size", type=int, default=5000, help="SQL batch size for image_id scanning")
    parser.add_argument("--limit", type=int, default=None, help="Optional max image_ids per target")
    parser.add_argument("--dry-run", action="store_true", help="Do not write SQL updates")
    parser.add_argument(
        "--recheck-all",
        action="store_true",
        help="Re-check all rows where mongo_* = 1, not only rows with missing_* IS NULL",
    )
    parser.add_argument(
        "--set-present-zero",
        action="store_true",
        help="When doc exists in Mongo, set missing_* = 0 (default leaves present rows unchanged)",
    )
    parser.add_argument(
        "--only",
        choices=["body_norm", "hand", "hand_norm", "body_3d", "all"],
        default="all",
        help="Run one specific audit target or all",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--is-ssd", action="store_true", default=True, help="Pass through to DataIO")
    return parser.parse_args()


def create_sql_engine(db):
    if db.get("unix_socket"):
        return create_engine(
            "mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
                user=db["user"],
                pw=db["pass"],
                db=db["name"],
                socket=db["unix_socket"],
            ),
            pool_pre_ping=True,
            pool_recycle=600,
            poolclass=NullPool,
        )

    return create_engine(
        "mysql+pymysql://{user}:{pw}@{host}/{db}".format(
            host=db["host"],
            db=db["name"],
            user=db["user"],
            pw=db["pass"],
        ),
        pool_pre_ping=True,
        pool_recycle=600,
        poolclass=NullPool,
    )


def has_body_landmarks_norm(doc):
    if not doc:
        return False
    return doc.get("nlms") is not None


def has_hand_landmarks(doc):
    return doc is not None


def has_hand_landmarks_norm(doc):
    if not doc:
        return False

    if doc.get("nlms") is not None:
        return True

    left_hand = doc.get("left_hand")
    right_hand = doc.get("right_hand")

    left_norm = isinstance(left_hand, dict) and left_hand.get("hand_landmarks_norm") is not None
    right_norm = isinstance(right_hand, dict) and right_hand.get("hand_landmarks_norm") is not None
    return left_norm or right_norm


def has_body_landmarks_3d(doc):
    if not doc:
        return False
    return doc.get("body_landmarks_3D") is not None


def fetch_ids_batch(session, source_col, missing_col, last_image_id, batch_size, recheck_all):
    source_attr = getattr(Encodings, source_col)
    missing_attr = getattr(Encodings, missing_col)

    query = (
        select(Encodings.image_id)
        .where(source_attr == 1)
        .where(Encodings.image_id > last_image_id)
        .order_by(Encodings.image_id.asc())
        .limit(batch_size)
    )

    if not recheck_all:
        query = query.where(missing_attr.is_(None))

    return [row[0] for row in session.execute(query).fetchall()]


def fetch_mongo_docs_by_image_id(collection, image_ids, projection=None):
    if not image_ids:
        return {}

    query = {"image_id": {"$in": image_ids}}
    docs = collection.find(query, projection)
    return {doc["image_id"]: doc for doc in docs if "image_id" in doc}


def update_missing_flags(session, missing_col, missing_ids, present_ids, dry_run=False, set_present_zero=False):
    writes = 0

    if missing_ids:
        writes += len(missing_ids)
        if not dry_run:
            (
                session.query(Encodings)
                .filter(Encodings.image_id.in_(missing_ids))
                .update({getattr(Encodings, missing_col): 1}, synchronize_session=False)
            )

    if set_present_zero and present_ids:
        writes += len(present_ids)
        if not dry_run:
            (
                session.query(Encodings)
                .filter(Encodings.image_id.in_(present_ids))
                .update({getattr(Encodings, missing_col): 0}, synchronize_session=False)
            )

    return writes


def run_target_audit(session, target, args):
    name = target["name"]
    source_col = target["source_col"]
    missing_col = target["missing_col"]
    collection = target["collection"]
    exists_fn = target["exists_fn"]
    projection = target["projection"]

    print(f"\n=== Auditing target: {name} ===")
    print(f"SQL source column: {source_col}")
    print(f"SQL missing flag:  {missing_col}")
    print(f"Mongo collection:  {collection.name}")

    total_checked = 0
    total_missing = 0
    total_present = 0
    total_written = 0
    last_image_id = -1

    start = time.time()

    while True:
        ids_batch = fetch_ids_batch(
            session,
            source_col=source_col,
            missing_col=missing_col,
            last_image_id=last_image_id,
            batch_size=args.batch_size,
            recheck_all=args.recheck_all,
        )

        if not ids_batch:
            break

        if args.limit is not None:
            remaining = args.limit - total_checked
            if remaining <= 0:
                break
            ids_batch = ids_batch[:remaining]
            if not ids_batch:
                break

        docs_map = fetch_mongo_docs_by_image_id(collection, ids_batch, projection=projection)

        missing_ids = []
        present_ids = []
        for image_id in ids_batch:
            doc = docs_map.get(image_id)
            if exists_fn(doc):
                present_ids.append(image_id)
            else:
                missing_ids.append(image_id)

        writes = update_missing_flags(
            session,
            missing_col=missing_col,
            missing_ids=missing_ids,
            present_ids=present_ids,
            dry_run=args.dry_run,
            set_present_zero=args.set_present_zero,
        )

        # if not args.dry_run:
        #     session.commit()

        total_checked += len(ids_batch)
        total_missing += len(missing_ids)
        total_present += len(present_ids)
        total_written += writes
        last_image_id = ids_batch[-1]

        print(
            f"Checked {total_checked:,} | missing {total_missing:,} | present {total_present:,} | "
            f"written {total_written:,} | last_image_id {last_image_id}"
        )

        if args.verbose and missing_ids:
            preview = missing_ids[:10]
            print(f"  sample missing ids ({len(preview)}): {preview}")

    elapsed = time.time() - start
    print(
        f"Done {name}: checked={total_checked:,}, missing={total_missing:,}, "
        f"present={total_present:,}, written={total_written:,}, elapsed={elapsed:.1f}s"
    )


def main():
    args = parse_args()

    io = DataIO(args.is_ssd)
    db = io.db

    engine = create_sql_engine(db)
    Session = sessionmaker(bind=engine)
    session = Session()

    mongo_client = pymongo.MongoClient(io.dbmongo["host"])
    mongo_db = mongo_client[io.dbmongo["name"]]

    mongo_collection = mongo_db[io.dbmongo["collection"]]
    bboxnormed_collection = mongo_db["body_landmarks_norm"]
    mongo_hand_collection = mongo_db["hand_landmarks"]

    targets = [
        {
            "name": "body_norm",
            "source_col": "mongo_body_landmarks_norm",
            "missing_col": "missing_mongo_body_landmarks_norm",
            "collection": bboxnormed_collection,
            "exists_fn": has_body_landmarks_norm,
            "projection": {"_id": 0, "image_id": 1, "nlms": 1},
        },
        {
            "name": "hand",
            "source_col": "mongo_hand_landmarks",
            "missing_col": "missing_mongo_hand_landmarks",
            "collection": mongo_hand_collection,
            "exists_fn": has_hand_landmarks,
            "projection": {"_id": 0, "image_id": 1},
        },
        {
            "name": "hand_norm",
            "source_col": "mongo_hand_landmarks_norm",
            "missing_col": "missing_mongo_hand_landmarks_norm",
            "collection": mongo_hand_collection,
            "exists_fn": has_hand_landmarks_norm,
            "projection": {
                "_id": 0,
                "image_id": 1,
                "nlms": 1,
                "left_hand.hand_landmarks_norm": 1,
                "right_hand.hand_landmarks_norm": 1,
            },
        },
        {
            "name": "body_3d",
            "source_col": "mongo_body_landmarks_3D",
            "missing_col": "missing_mongo_body_landmarks_3D",
            "collection": mongo_collection,
            "exists_fn": has_body_landmarks_3d,
            "projection": {"_id": 0, "image_id": 1, "body_landmarks_3D": 1},
        },
    ]

    selected_targets = targets
    if args.only != "all":
        selected_targets = [target for target in targets if target["name"] == args.only]

    mode = "DRY RUN" if args.dry_run else "WRITE"
    print("=" * 90)
    print("audit_missing_mongo.py")
    print(f"Mode: {mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Limit per target: {args.limit}")
    print(f"Recheck all: {args.recheck_all}")
    print(f"Set present to 0: {args.set_present_zero}")
    print(f"Targets: {[target['name'] for target in selected_targets]}")
    print("=" * 90)

    try:
        for target in selected_targets:
            run_target_audit(session, target, args)
    finally:
        session.close()
        mongo_client.close()
        engine.dispose()


if __name__ == "__main__":
    main()
