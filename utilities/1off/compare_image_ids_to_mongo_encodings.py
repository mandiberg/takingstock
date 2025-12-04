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
batch_size = 10000
num_threads = 8  # Adjust based on your CPU and IO
MODE = 0 #0 for overall compare, 1 to recheck against entry and bson dump
# MODE 0 ignores is_body and is_face, MODE 1 filters on those
IS_FACE = 0
IS_BODY = 1

last_id = 54640001  # starting from this id 
print(f"Starting from last_id: {last_id}")
IMPORT_DIR = "/users/michaelmandiberg"
EXPORT_DIR = os.path.join(io.ROOT_PROD,"mongo_exports_oct19_sets")  # Directory to save BSON files
# EXPORT_DIR = os.path.join("/Volumes/OWC4/segment_images_deshardJSON_aug2_toArchive/mongo_exports_fromlist_adobeE")  # Directory to save BSON files
# touch the directory if it does not exist
os.makedirs(EXPORT_DIR, exist_ok=True)
print(f"Export directory: {EXPORT_DIR}")


# select max encoding_id to start from
Base = declarative_base()

# table_name = 'compare_sql_mongo_results3'
# class CompareSqlMongoResults(Base):
#     __tablename__ = table_name
#     encoding_id = Column(Integer, primary_key=True)
#     image_id = Column(Integer)
#     is_body = Column(Boolean)
#     is_face = Column(Boolean)
#     face_landmarks = Column(Integer)
#     body_landmarks = Column(Integer)
#     face_encodings68 = Column(Integer)
#     nlms = Column(Integer)
#     left_hand = Column(Integer)
#     right_hand = Column(Integer)
#     body_world_landmarks = Column(Integer)
# Base.metadata.create_all(engine)
# last_id = session.query(sqlalchemy.func.max(CompareSqlMongoResults.encoding_id)).scalar()
# if last_id is None:
#     last_id = 0
# last_id = 0
# print(f"Starting from last_id: {last_id}")

# # variables to filter encodings on
# migrated_SQL = 1
# migrated = None
# migrated_Mongo = None
# is_body = 1
# is_face = 1
# mongo_body_landmarks_3D = 1

# collection_names = ['encodings', 'body_landmarks_norm', 'hand_landmarks', 'body_landmarks_3D']
# document_names_dict = {
#     # "encodings": ["encoding_id", "face_landmarks", "body_landmarks", "face_encodings68"],
#     "encodings": ["face_landmarks", "body_landmarks", "face_encodings68"],
#     "body_landmarks_norm": ["nlms"],
#     "hand_landmarks": ["left_hand", "right_hand"],
#     "body_landmarks_3D": ["body_world_landmarks"]
# }

# sql_field_names_dict = {
#     "face_landmarks": "mongo_face_landmarks",
#     "body_landmarks": "mongo_body_landmarks",
#     "face_encodings68": "mongo_encodings",
#     "nlms": "mongo_body_landmarks_norm",
#     "left_hand": "mongo_hand_landmarks",
#     "right_hand": "mongo_hand_landmarks",
#     "body_world_landmarks": "mongo_body_landmarks_3D"
# }


is_face_fields = ["face_landmarks", "face_encodings68"]
is_body_fields = ["body_landmarks", "body_landmarks_norm", "hand_landmarks", "body_landmarks_3D"]

# # Initialize counters outside the loop (at the top of your script, before the while loop)
# if 'counts' not in locals():
#     counts = {}
#     for collection_name in collection_names:
#         for document_name in document_names_dict[collection_name]:
#             key = f"{collection_name}.{document_name}"
#             counts[key] = {"sql_only": 0, "mongo_only": 0, "both_match": 0}

results_rows = []


'''

'''

def process_batch(batch_start, batch_end):
    global image_ids_set_global
    # Each thread needs its own session and mongo client
    # thread_engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    #     user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    # ), poolclass=NullPool)
    # ThreadSession = sessionmaker(bind=thread_engine)
    # thread_session = ThreadSession()
    thread_mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    thread_mongo_db = thread_mongo_client["stock"]

    # slice image_ids_set_global to only those in the batch range
    image_ids_set = set()
    for image_id in image_ids_set_global:
        try:
            int_image_id = int(image_id)
            if batch_start <= int_image_id < batch_end:
                image_ids_set.add(int_image_id)
        except ValueError:
            print(f"Skipping non-integer image_id: {image_id}")

    results = list(image_ids_set)
    results_rows = []

    # results = get_mysql_results(batch_start, batch_end, thread_session)
    print(f"Thread processing batch {batch_start} to {batch_end}, got {len(results)} results")
    # print(f"First 5 results: {results[:5]}")

    for image_id in results:
        if image_id is None:
            continue
        this_row = {
                    "image_id": image_id,
                }
        collection = thread_mongo_db["hand_landmarks"]
        doc = collection.find_one({"image_id": image_id})
        if doc is None:
            this_row["hand_landmarks"] = False
            print(f"No MongoDB document found in hand_landmarks for image_id {image_id}")
        else:
            this_row["hand_landmarks"] = True
    thread_mongo_client.close()
    return results_rows

# LOAD IMAGE IDS SET FROM FILE
global_collection_file = "hand_landmarks" # this will probably need to be updated mannually
set_name = "sql_only"
image_ids_set_global = set()
bson_image_ids_set = set()

if global_collection_file == "body_landmarks_norm":
    file_name = "nlms"
elif "hand_landmarks" in global_collection_file:
    file_name = "hand_landmarks_right_hand"
else:
    file_name = global_collection_file
#open the set file in the set_name folder, and load it into a set
with open(os.path.join(EXPORT_DIR, set_name, f"{set_name}_{file_name}.txt"), "r") as f:
    first_line = f.readline()  # read header
    if first_line.startswith("Entries"):
        print(f"skipping header line: {first_line.strip()}")
        first_id = None
    elif first_line.strip().isdigit():
        first_id = first_line.strip()
        print(f"first id: {first_id}")
    else:
        print("Unexpected file format, first line:", first_line)

    remaining_lines = f.readlines()  # read the rest of the lines
    image_ids_set_global = set(int(line.strip()) for line in remaining_lines)
    if first_id is not None:
        image_ids_set_global.add(int(first_id))
print(f"Loaded set {set_name} for collection {global_collection_file} with {len(image_ids_set_global)} entries")



# Get min and max encoding_id for batching
# min_id = session.query(sqlalchemy.func.min(Encodings.encoding_id)).scalar() or 0
min_id = last_id + 1
# max_id = session.query(sqlalchemy.func.max(Encodings.encoding_id)).scalar() or 0
max_id = max(image_ids_set_global)

for batch_start in range(min_id, max_id + 1, batch_size * num_threads):
    batch_ranges = [
        (start, min(start + batch_size, max_id + 1))
        for start in range(batch_start, min(batch_start + batch_size * num_threads, max_id + 1), batch_size)
    ]
    results_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_batch, start, end) for start, end in batch_ranges]
        for future in concurrent.futures.as_completed(futures):
            results_rows.extend(future.result())

    batch_df = pd.DataFrame(results_rows)
    print(f"Processed batch {batch_start} to {min(batch_start + batch_size * num_threads, max_id + 1)}, total results rows: {len(results_rows)}")

    # save csv of images where hand_landmarks is True
    if not batch_df.empty:
        print(batch_df)
        hand_landmarks_true_df = batch_df[batch_df["hand_landmarks"] == True]
        hand_landmarks_true_csv_path = os.path.join(EXPORT_DIR, f"hand_landmarks_true_batch_{batch_start}_{min(batch_start + batch_size * num_threads, max_id + 1)}.csv")
        hand_landmarks_true_df.to_csv(hand_landmarks_true_csv_path, index=False)
        print(f"Saved hand_landmarks True entries to {hand_landmarks_true_csv_path}")   

        # save csv of images where hand_landmarks is False
        hand_landmarks_false_df = batch_df[batch_df["hand_landmarks"] == False]
        hand_landmarks_false_csv_path = os.path.join(EXPORT_DIR, f"hand_landmarks_false_batch_{batch_start}_{min(batch_start + batch_size * num_threads, max_id +   1)}.csv")
        hand_landmarks_false_df.to_csv(hand_landmarks_false_csv_path, index=False)
        print(f"Saved hand_landmarks False entries to {hand_landmarks_false_csv_path}")

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