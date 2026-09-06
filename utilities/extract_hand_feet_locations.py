import os
import pickle
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# importing project-specific models
import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/takingstock/')
from my_declarative_base import SegmentTable, Encodings, Base
from sqlalchemy.ext.declarative import declarative_base
Base2 = declarative_base()

# Define LocationHandsFeet model if not already defined
# Schema v2: adds hip/knee points, ankle->heel->toe fallback tracking, and
# derived sided leg-shape features (mid-hip relative) for cluster separability testing.
from sqlalchemy import Column, Integer, Float, Boolean, String
from sqlalchemy.ext.declarative import declarative_base
Base2 = declarative_base()

HelperTable_name = "SegmentHelper_TheGym" # if you set to None, comment out the helpertable join in the query
class HelperTable(Base2):
    __tablename__ = HelperTable_name
    seg_image_id=Column(Integer,primary_key=True, autoincrement=True)
    image_id = Column(Integer, primary_key=True, autoincrement=True)


class LocationHandsFeet(Base2):
    __tablename__ = 'LocationHandsFeet'
    image_id = Column(Integer, primary_key=True)

    left_hand_x = Column(Float)
    left_hand_y = Column(Float)
    left_hand_vis = Column(Boolean)
    right_hand_x = Column(Float)
    right_hand_y = Column(Float)
    right_hand_vis = Column(Boolean)

    hip_left_x = Column(Float)
    hip_left_y = Column(Float)
    hip_left_vis = Column(Boolean)
    hip_right_x = Column(Float)
    hip_right_y = Column(Float)
    hip_right_vis = Column(Boolean)
    # average of whichever hip(s) are visible; the shared vertical reference for leg deltas
    mid_hip_x = Column(Float)
    mid_hip_y = Column(Float)

    knee_left_x = Column(Float)
    knee_left_y = Column(Float)
    knee_left_vis = Column(Boolean)
    knee_right_x = Column(Float)
    knee_right_y = Column(Float)
    knee_right_vis = Column(Boolean)

    # foot point resolved via ankle -> heel -> toe fallback; source records which was used
    foot_left_x = Column(Float)
    foot_left_y = Column(Float)
    foot_left_vis = Column(Boolean)
    foot_left_source = Column(String(5))  # 'ankle' | 'heel' | 'toe' | None
    foot_right_x = Column(Float)
    foot_right_y = Column(Float)
    foot_right_vis = Column(Boolean)
    foot_right_source = Column(String(5))

    # derived, sided leg-shape features (mid-hip relative, face-height units)
    ankle_rel_y_left = Column(Float)
    ankle_rel_y_right = Column(Float)
    knee_rel_y_left = Column(Float)
    knee_rel_y_right = Column(Float)
    leg_extension_max = Column(Float)   # max(ankle_rel_y_left, ankle_rel_y_right) over visible sides
    leg_extension_min = Column(Float)
    leg_asymmetry = Column(Float)       # leg_extension_max - leg_extension_min
    visible_leg_count = Column(Integer)  # 0, 1, or 2

# MongoDB setup
import pymongo
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["stock"]
mongo_collection = mongo_db["encodings"]  # original body_landmarks
mongo_collection_norm = mongo_db["body_landmarks_norm"]  # normalized landmarks

# MySQL setup (preserving credentials framework)
from mp_db_io import DataIO
io = DataIO()
db = io.db
engine = create_engine(
    f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
    pool_pre_ping=True,
    pool_recycle=600,
    poolclass=NullPool
)
Session = sessionmaker(bind=engine)
session = Session()

# Batch processing parameters
batch_size = 1000
num_threads = 16
# sept 6 2026 -- processd full encodings table up to 45014557
# switching to the segment helper
start_encoding_id = 61591855
last_id = start_encoding_id
VIS_THRESHOLD = 0.5  # matches existing body-landmark visibility convention elsewhere in the pipeline

# MediaPipe pose landmark indices
LM_LEFT_HAND, LM_RIGHT_HAND = 15, 16
LM_HIP_LEFT, LM_HIP_RIGHT = 23, 24
LM_KNEE_LEFT, LM_KNEE_RIGHT = 25, 26
LM_ANKLE_LEFT, LM_ANKLE_RIGHT = 27, 28
LM_HEEL_LEFT, LM_HEEL_RIGHT = 29, 30
LM_TOE_LEFT, LM_TOE_RIGHT = 31, 32


def get_lm(nlms, idx):
    try:
        return nlms.landmark[idx]
    except (IndexError, AttributeError):
        return None


def is_visible(lm, threshold=VIS_THRESHOLD):
    return lm is not None and lm.visibility is not None and lm.visibility > threshold


def resolve_foot_point(nlms, ankle_idx, heel_idx, toe_idx):
    """Ankle -> heel -> toe fallback; returns (x, y, vis, source)."""
    for idx, source in ((ankle_idx, "ankle"), (heel_idx, "heel"), (toe_idx, "toe")):
        lm = get_lm(nlms, idx)
        if is_visible(lm):
            return lm.x, lm.y, True, source
    return None, None, False, None


def resolve_mid_hip(hip_left, hip_right, hip_left_vis, hip_right_vis):
    if hip_left_vis and hip_right_vis:
        return (hip_left.x + hip_right.x) / 2.0, (hip_left.y + hip_right.y) / 2.0
    if hip_left_vis:
        return hip_left.x, hip_left.y
    if hip_right_vis:
        return hip_right.x, hip_right.y
    return None, None


def build_location_row(image_id, nlms):
    lh, rh = get_lm(nlms, LM_LEFT_HAND), get_lm(nlms, LM_RIGHT_HAND)
    lh_vis, rh_vis = is_visible(lh), is_visible(rh)

    hip_l, hip_r = get_lm(nlms, LM_HIP_LEFT), get_lm(nlms, LM_HIP_RIGHT)
    hip_l_vis, hip_r_vis = is_visible(hip_l), is_visible(hip_r)
    mid_hip_x, mid_hip_y = resolve_mid_hip(hip_l, hip_r, hip_l_vis, hip_r_vis)

    knee_l, knee_r = get_lm(nlms, LM_KNEE_LEFT), get_lm(nlms, LM_KNEE_RIGHT)
    knee_l_vis, knee_r_vis = is_visible(knee_l), is_visible(knee_r)

    foot_l_x, foot_l_y, foot_l_vis, foot_l_source = resolve_foot_point(
        nlms, LM_ANKLE_LEFT, LM_HEEL_LEFT, LM_TOE_LEFT
    )
    foot_r_x, foot_r_y, foot_r_vis, foot_r_source = resolve_foot_point(
        nlms, LM_ANKLE_RIGHT, LM_HEEL_RIGHT, LM_TOE_RIGHT
    )

    ankle_rel_y_left = (foot_l_y - mid_hip_y) if (foot_l_vis and mid_hip_y is not None) else None
    ankle_rel_y_right = (foot_r_y - mid_hip_y) if (foot_r_vis and mid_hip_y is not None) else None
    knee_rel_y_left = (knee_l.y - mid_hip_y) if (knee_l_vis and mid_hip_y is not None) else None
    knee_rel_y_right = (knee_r.y - mid_hip_y) if (knee_r_vis and mid_hip_y is not None) else None

    visible_rel_y = [v for v in (ankle_rel_y_left, ankle_rel_y_right) if v is not None]
    leg_extension_max = max(visible_rel_y) if visible_rel_y else None
    leg_extension_min = min(visible_rel_y) if visible_rel_y else None
    leg_asymmetry = (
        leg_extension_max - leg_extension_min
        if (ankle_rel_y_left is not None and ankle_rel_y_right is not None)
        else None
    )
    visible_leg_count = sum(1 for v in (foot_l_vis, foot_r_vis) if v)

    return LocationHandsFeet(
        image_id=image_id,
        left_hand_x=lh.x if lh else None, left_hand_y=lh.y if lh else None, left_hand_vis=lh_vis,
        right_hand_x=rh.x if rh else None, right_hand_y=rh.y if rh else None, right_hand_vis=rh_vis,

        hip_left_x=hip_l.x if hip_l else None, hip_left_y=hip_l.y if hip_l else None, hip_left_vis=hip_l_vis,
        hip_right_x=hip_r.x if hip_r else None, hip_right_y=hip_r.y if hip_r else None, hip_right_vis=hip_r_vis,
        mid_hip_x=mid_hip_x, mid_hip_y=mid_hip_y,

        knee_left_x=knee_l.x if knee_l else None, knee_left_y=knee_l.y if knee_l else None, knee_left_vis=knee_l_vis,
        knee_right_x=knee_r.x if knee_r else None, knee_right_y=knee_r.y if knee_r else None, knee_right_vis=knee_r_vis,

        foot_left_x=foot_l_x, foot_left_y=foot_l_y, foot_left_vis=foot_l_vis, foot_left_source=foot_l_source,
        foot_right_x=foot_r_x, foot_right_y=foot_r_y, foot_right_vis=foot_r_vis, foot_right_source=foot_r_source,

        ankle_rel_y_left=ankle_rel_y_left, ankle_rel_y_right=ankle_rel_y_right,
        knee_rel_y_left=knee_rel_y_left, knee_rel_y_right=knee_rel_y_right,
        leg_extension_max=leg_extension_max, leg_extension_min=leg_extension_min,
        leg_asymmetry=leg_asymmetry, visible_leg_count=visible_leg_count,
    )


def process_batch(rows):
    """Each thread gets its own MySQL session and Mongo client."""
    thread_engine = create_engine(
        f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
        pool_pre_ping=True,
        pool_recycle=600,
        poolclass=NullPool,
    )
    ThreadSession = sessionmaker(bind=thread_engine)
    thread_session = ThreadSession()
    thread_mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    thread_mongo_db = thread_mongo_client["stock"]

    try:
        inserted_or_updated = 0
        seen_image_ids = set()
        for encoding_id, image_id in rows:
            if image_id in seen_image_ids:
                continue
            seen_image_ids.add(image_id)

            mongo_doc_norm = thread_mongo_db["body_landmarks_norm"].find_one({"image_id": image_id}, {"nlms": 1})
            if not (mongo_doc_norm and mongo_doc_norm.get("nlms")):
                continue

            try:
                nlms = pickle.loads(mongo_doc_norm["nlms"])
                row = build_location_row(image_id, nlms)
                thread_session.merge(row)
                inserted_or_updated += 1
            except Exception as exc:
                print(f"Error extracting landmarks for image_id {image_id}: {exc}")

        if inserted_or_updated:
            thread_session.commit()

        return inserted_or_updated
    finally:
        thread_session.close()
        thread_mongo_client.close()
        thread_engine.dispose()


from concurrent.futures import ThreadPoolExecutor, as_completed

while True:
    # producer: fetch only the next batch of rows that are both in the helper table
    # and beyond the configured start point.
    query = (
        session.query(Encodings.encoding_id, Encodings.image_id)
        .join(HelperTable, HelperTable.image_id == Encodings.image_id)
        .filter(
            Encodings.mongo_body_landmarks_norm.is_(True),
            Encodings.is_face.is_(True),
            Encodings.encoding_id > last_id,
        )
        .order_by(Encodings.encoding_id)
        .limit(batch_size * num_threads)
    )
    results = query.all()

    if not results:
        print("No more rows to process. Exiting.")
        break

    batches = [results[i:i + batch_size] for i in range(0, len(results), batch_size)]
    processed_total = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        for future in as_completed(futures):
            processed_total += future.result()

    last_id = results[-1][0]
    print(f"Processed up to encoding_id = {last_id}, rows written = {processed_total}")

session.close()
mongo_client.close()
engine.dispose()