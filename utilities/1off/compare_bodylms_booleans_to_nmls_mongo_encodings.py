import os
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
from pathlib import Path
import pandas as pd
import concurrent.futures

# this was a one-ff script to compare image_ids with mysql mongo_hand_lanmark = 1 booleans
# and see if they existed in mongo hand_landmarks collection

# importing from another folder
import sys
ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/facemap/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)
from mp_db_io import DataIO
from my_declarative_base import Images, SegmentTable, Encodings, SegmentBig, Base, Clusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON
import pymongo
from pymongo.errors import DuplicateKeyError

######## Michael's Credentials ########
# platform specific credentials
io = DataIO()
db = io.db
# overriding DB for testing
io.db["name"] = "stock"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#######################################


engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Connect to MongoDB
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["stock"]
# mongo_collection = mongo_db["encodings"]
mongo_collection = mongo_db["body_landmarks_norm"]

# Define the batch size
batch_size = 1000
num_threads = 8  # Adjust based on your CPU and IO
MODE = 0 #0 for overall compare, 1 to recheck against entry and bson dump
# MODE 0 ignores is_body and is_face, MODE 1 filters on those
IS_FACE = 0
IS_BODY = 1

last_id = 2200  # starting from this id 
print(f"Starting from last_id: {last_id}")
# IMPORT_DIR = "/users/michaelmandiberg"
EXPORT_DIR = os.path.join(io.ROOT_PROD,"is_body_no_nlms_Nov30")  # Directory to save BSON files
# # EXPORT_DIR = os.path.join("/Volumes/OWC4/segment_images_deshardJSON_aug2_toArchive/mongo_exports_fromlist_adobeE")  # Directory to save BSON files
# # touch the directory if it does not exist
# os.makedirs(EXPORT_DIR, exist_ok=True)
# print(f"Export directory: {EXPORT_DIR}")


# select max encoding_id to start from
Base = declarative_base()


# define a missing landmarks table to store results
table_name = "SegmentHelperMissing_nov2025"
class SegmentHelperMissing_nov2025(Base):
    __tablename__ = table_name
    id = Column(Integer, primary_key=True, autoincrement=True)
    encoding_id = Column(Integer, unique=False, nullable=False)
    image_id = Column(Integer, unique=False, nullable=False)
    body_landmarks = Column(Boolean, nullable=True)
    body_landmarks_norm = Column(Boolean, nullable=True)
    face_landmarks = Column(Boolean, nullable=True)
    face_encodings = Column(Boolean, nullable=True)
    hand_landmarks = Column(Boolean, nullable=True)
    body_world_landmarks = Column(Boolean, nullable=True)

Base.metadata.create_all(engine)






def process_batch(batch_start, batch_end):
    batch_results_rows = []
    # global image_ids_set_global
    # Each thread needs its own session and mongo client
    thread_engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
    ThreadSession = sessionmaker(bind=thread_engine)
    thread_session = ThreadSession()
    thread_mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    thread_mongo_db = thread_mongo_client["stock"]

    # use thread_session to query mysql for image_ids in the batch range where mongo_body_landmarks is True
    results = thread_session.query(Encodings.image_id, Encodings.encoding_id, Encodings.mongo_body_landmarks, Encodings.mongo_body_landmarks_norm).filter(
        Encodings.is_body.is_(True),
        Encodings.is_face.is_(True),
        Encodings.is_face_no_lms.is_(None),
        # Encodings.is_face.is_(True),
        # Encodings.is_body.is_(True),
        Encodings.encoding_id >= batch_start,
        Encodings.encoding_id < batch_end
    ).all()

    # results = get_mysql_results(batch_start, batch_end, thread_session)
    print(f"Thread processing batch {batch_start} to {batch_end}, got {len(results)} results")
    # print(f"First 5 results: {results[:5]}")

    for result in results:
        image_id = result[0]
        encoding_id = result[1]
        # print(f"Processing encoding_id: {encoding_id}, image_id: {image_id},")
        if image_id is None:
            continue
        missing_landmarks = {
                "encoding_id": encoding_id,
                "image_id": image_id,
                }
        collection_norm = thread_mongo_db["body_landmarks_norm"]
        doc = collection_norm.find_one({"image_id": image_id})
        if doc is None:
            missing_landmarks["body_landmarks_norm"] = None

            # check body
            collection_encodings = thread_mongo_db["encodings"]
            doc = collection_encodings.find_one({"image_id": image_id})
            if doc is None:
                missing_landmarks["body_landmarks"] = None
                missing_landmarks["face_landmarks"] = None
                missing_landmarks["face_encodings"] = None
                # print(f"   --  no body_landmarks for image_id {image_id}")
            else:
                missing_landmarks["body_landmarks"] = bool(doc.get("body_landmarks", None))
                missing_landmarks["face_landmarks"] = bool(doc.get("face_landmarks", None))
                missing_landmarks["face_encodings"] = bool(doc.get("face_encodings", None))
                # print(f"  ++  has body_landmarks for image_id {image_id}")

            # check hands
            collection_hands = thread_mongo_db["hand_landmarks"]
            doc = collection_hands.find_one({"image_id": image_id})
            if doc is None:
                missing_landmarks["hand_landmarks"] = None
            else:
                missing_landmarks["hand_landmarks"] = bool(doc.get("hand_landmarks", None))

            # check 3D
            collection_world = thread_mongo_db["body_world_landmarks"]
            doc = collection_world.find_one({"image_id": image_id})
            if doc is None:
                missing_landmarks["body_world_landmarks"] = None
            else:
                missing_landmarks["body_world_landmarks"] = bool(doc.get("body_world_landmarks", None))

            print(f" >< No MongoDB doc: {missing_landmarks}")
            batch_results_rows.append(missing_landmarks)
        else:
            # for now, doing nothing if we find the nlms
            pass
            # missing_landmarks["body_landmarks_norm"] = True
            # print(f"  ==  already has body_landmarks_norm for image_id {image_id}")

    # store batch_results_rows into mysql table
    if batch_results_rows:
        # Normalize rows to have all expected columns
        cols = [
            "encoding_id",
            "image_id",
            "body_landmarks",
            "body_landmarks_norm",
            "face_landmarks",
            "face_encodings",
            "hand_landmarks",
            "body_world_landmarks",
        ]
        values = []
        for row in batch_results_rows:
            values.append(tuple(row.get(c, None) for c in cols))

        # Use INSERT IGNORE to avoid duplicates (requires a unique key on one or more columns)
        placeholders = ", ".join(["%s"] * len(cols))
        col_list = ", ".join(cols)
        sql = f"INSERT IGNORE INTO {table_name} ({col_list}) VALUES ({placeholders})"

        conn = thread_engine.raw_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany(sql, values)
            conn.commit()
            print(f"Inserted {cursor.rowcount} rows into {table_name} from batch {batch_start} to {batch_end} (duplicates ignored)")
        finally:
            cursor.close()
            conn.close()

    thread_session.close()
    thread_engine.dispose()
    thread_mongo_client.close()
    return batch_results_rows




# Get min and max encoding_id for batching
# min_id = session.query(sqlalchemy.func.min(Encodings.encoding_id)).scalar() or 0
min_id = last_id + 1
# max_id = session.query(sqlalchemy.func.max(Encodings.encoding_id)).scalar() or 0
# max_id = max(image_ids_set_global)
max_id = session.query(sqlalchemy.func.max(Encodings.encoding_id)).scalar()
results_rows = []

for batch_start in range(min_id, max_id + 1, batch_size * num_threads):
    batch_ranges = [
        (start, min(start + batch_size, max_id + 1))
        for start in range(batch_start, min(batch_start + batch_size * num_threads, max_id + 1), batch_size)
    ]
    results_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_batch, start, end) for start, end in batch_ranges]
        for future in concurrent.futures.as_completed(futures):
            print("A thread has completed its batch:.")
            # results_rows.extend(future.result())
        # break  # temporary for testing

    # batch_df = pd.DataFrame(results_rows)
    print(f"Processed batch {batch_start} to {min(batch_start + batch_size * num_threads, max_id + 1)}, total results rows: {len(results_rows)}")

    # save csv of images where hand_landmarks is True
    # if not batch_df.empty:
    #     print(len(batch_df))

        # hand_landmarks_true_df = batch_df[batch_df["hand_landmarks"] == True]
        # hand_landmarks_true_csv_path = os.path.join(EXPORT_DIR, f"hand_landmarks_true_batch_{batch_start}_{min(batch_start + batch_size * num_threads, max_id + 1)}.csv")
        # hand_landmarks_true_df.to_csv(hand_landmarks_true_csv_path, index=False)
        # print(f"Saved hand_landmarks True entries to {hand_landmarks_true_csv_path}")   

        # # save csv of images where hand_landmarks is False
        # hand_landmarks_false_df = batch_df[batch_df["hand_landmarks"] == False]
        # hand_landmarks_false_csv_path = os.path.join(EXPORT_DIR, f"hand_landmarks_false_batch_{batch_start}_{min(batch_start + batch_size * num_threads, max_id +   1)}.csv")
        # hand_landmarks_false_df.to_csv(hand_landmarks_false_csv_path, index=False)
        # print(f"Saved hand_landmarks False entries to {hand_landmarks_false_csv_path}")

    # if not batch_df.empty:
    #     batch_df.to_sql(
    #         name=table_name,
    #         con=engine,
    #         if_exists="append",
    #         index=False,
    #         method="multi"
    #     )
    # results_rows.clear()
    # break  # temporary for testing

session.close()