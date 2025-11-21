import os
import pickle
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# importing project-specific models
import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from my_declarative_base import SegmentTable, Encodings, Base

# MongoDB setup
import pymongo
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["stock"]
mongo_collection = mongo_db["encodings"]  # adjust collection name if needed

# MySQL setup (preserving credentials framework)
from mp_db_io import DataIO
io = DataIO()
db = io.db
engine = create_engine(
    f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
    poolclass=NullPool
)
Session = sessionmaker(bind=engine)
session = Session()

# Batch processing parameters
batch_size = 1000
last_id = 0

while True:
    # # 1. Fetch next batch of SegmentTable rows where is_body is True and is_feet is NULL
    # results = (
    #     session.query(SegmentTable.seg_image_id, SegmentTable.image_id)
    #     .filter(
    #         SegmentTable.mongo_body_landmarks.is_(True),
    #         SegmentTable.is_feet.is_(None),
    #         SegmentTable.seg_image_id > last_id
    #     )
    #     .order_by(SegmentTable.seg_image_id)
    #     .limit(batch_size)
    #     .all()
    # )

    # 1. switching to encoding table directly
    results = (
        session.query(Encodings.encoding_id, Encodings.image_id)
        .filter(
            Encodings.mongo_body_landmarks.is_(True),
            Encodings.is_feet.is_(None),
            Encodings.encoding_id > last_id
        )
        .order_by(Encodings.encoding_id)
        .limit(batch_size)
        .all()
    )
            # Encodings.is_feet.is_(None),

    if not results:
        print("No more rows to process. Exiting.")
        break

    for encoding_id, image_id in results:
        # 2. Fetch pickled body_landmarks from Mongo
        mongo_doc = mongo_collection.find_one({"image_id": image_id}, {"body_landmarks": 1})
        is_feet = False

        if mongo_doc and mongo_doc.get("body_landmarks"):
            # 3. Unpickle to MediaPipe object
            body_landmarks = pickle.loads(mongo_doc["body_landmarks"])

            # 4. Evaluate visibility for feet landmarks (27-32)
            foot_lms = body_landmarks.landmark[27:33]
            # print(f"Foot landmarks: {foot_lms}")
            visible_count = sum(1 for lm in foot_lms if lm.visibility > 0.85)
            # is_feet = (visible_count >= (len(foot_lms) / 2))
            is_feet = (visible_count >= 1) # if any foot landmark is visible, we consider it as feet
            # if is_feet:
            # is_feet = (visible_count >= (len(foot_lms) / 2))
            # print(f"Image ID: {image_id}, Visible foot landmarks: {visible_count}, is_feet: {is_feet}")

        # 5. Update MySQL tables
        # skipping segment table bc doing it directly in encodings
        # session.query(SegmentTable).\
        #     filter(SegmentTable.seg_image_id == seg_image_id).\
        #     update({"is_feet": is_feet})
        session.query(Encodings).\
            filter(Encodings.image_id == image_id).\
            update({"is_feet": is_feet})

    session.commit()
    last_id = results[-1][0]
    print(f"Processed up to seg_image_id = {last_id}")

session.close()
mongo_client.close()
engine.dispose()
# Note: Ensure that the MongoDB and MySQL connections are properly closed after processing.