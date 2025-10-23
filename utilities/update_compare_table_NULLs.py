import os
import bson
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
from pathlib import Path
import pandas as pd
import concurrent.futures


# importing from another folder
import sys
ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/facemap/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)
from mp_db_io import DataIO
from mp_bson import MongoBSONExporter
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
exporter = MongoBSONExporter(mongo_db)

# Define the batch size
batch_size = 5000
last_id = 45540000  # starting from encoding_id 45540000
print(f"Starting from last_id: {last_id}")
EXPORT_DIR = os.path.join(io.ROOT_PROD,"mongo_exports_oct19_sets")  # Directory to save BSON files
# EXPORT_DIR = os.path.join("/Volumes/OWC4/segment_images_deshardJSON_aug2_toArchive/mongo_exports_fromlist_adobeE")  # Directory to save BSON files
# touch the directory if it does not exist
os.makedirs(EXPORT_DIR, exist_ok=True)
print(f"Export directory: {EXPORT_DIR}")
# select max encoding_id to start from
Base = declarative_base()

table_name = 'compare_sql_mongo_results_ultradone'
class CompareSqlMongoResults(Base):
    __tablename__ = table_name
    encoding_id = Column(Integer, primary_key=True)
    image_id = Column(Integer)
    is_body = Column(Boolean)
    is_face = Column(Boolean)
    face_landmarks = Column(Integer)
    body_landmarks = Column(Integer)
    face_encodings68 = Column(Integer)
    nlms = Column(Integer)
    left_hand = Column(Integer)
    right_hand = Column(Integer)
    body_world_landmarks = Column(Integer)

HelperTable_name = "SegmentHelper_oct2025_missing_face_encodings"
HelperTable_name = None

if HelperTable_name is not None:
    class HelperTable(Base):
        __tablename__ = HelperTable_name
        seg_image_id=Column(Integer,primary_key=True, autoincrement=True)
        image_id = Column(Integer, primary_key=True, autoincrement=True)

Base.metadata.create_all(engine)
# last_id = session.query(sqlalchemy.func.max(CompareSqlMongoResults.encoding_id)).scalar()
# if last_id is None:
#     last_id = 0

# variables to filter encodings on
migrated_SQL = 1
migrated = None
migrated_Mongo = None
is_body = 1
is_face = 1
mongo_body_landmarks_3D = 1

collection_names = ['encodings', 'body_landmarks_norm', 'hand_landmarks', 'body_world_landmarks']
document_names_dict = {
    # "encodings": ["encoding_id", "face_landmarks", "body_landmarks", "face_encodings68"],
    "encodings": ["face_landmarks", "body_landmarks", "face_encodings68"],
    "body_landmarks_norm": ["nlms"],
    "hand_landmarks": ["left_hand", "right_hand"],
    "body_world_landmarks": ["body_world_landmarks"]
}

sql_field_names_dict = {
    "face_landmarks": "mongo_face_landmarks",
    "body_landmarks": "mongo_body_landmarks",
    "face_encodings68": "mongo_encodings",
    "nlms": "mongo_body_landmarks_norm",
    "left_hand": "mongo_hand_landmarks",
    "right_hand": "mongo_hand_landmarks",
    "body_world_landmarks": "mongo_body_landmarks_3D"
}


# Initialize counters outside the loop (at the top of your script, before the while loop)
if 'counts' not in locals():
    counts = {}
    for collection_name in collection_names:
        for document_name in document_names_dict[collection_name]:
            key = f"{collection_name}.{document_name}"
            counts[key] = {"sql_only": 0, "mongo_only": 0, "both_match": 0}

results_rows = []


'''
Go through the Encodings table in batches, using the variables above to filter.
There will be about 7 million results in total
For each encoding, get the image_id and use it to compare the SQL data to the Mongo data
For each collection in collection_names, compare the SQL booleans to the actual presence of data in Mongo
If there is a mismatch, print it out
If there is data in SQL but not in Mongo, print it out
If there is data in Mongo but not in SQL, print it out

'''


def query_encodings_for_encoding_ids(engine, encodings_field_name, image_ids_list):
    # print(f"query_encodings_for_encoding_ids for field {encodings_field_name} and {len(image_ids_list)} image_ids")
    # chunk the image_ids_list into smaller lists of size 1000 to avoid query size limits
    chunk_size = 1000
    chunks = [image_ids_list[i:i + chunk_size] for i in range(0, len(image_ids_list), chunk_size)]
    df_list = []
    for chunk in chunks:
        image_ids_str = ",".join([str(image_id) for image_id in chunk])
        query = f"SELECT encoding_id, image_id, {encodings_field_name} FROM encodings WHERE image_id IN ({image_ids_str})"
        # print(f"Executing query: {query}")
        try:
            df_chunk = pd.read_sql(
                query,
                engine
            )
            df_list.append(df_chunk)
        except Exception as e:
            print(f"Error reading from SQL table encodings: {e}")
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.DataFrame()
    # print(f"Retrieved {len(df)} rows from encodings for given image_ids")
    # print(df.head())
    return df

def record_mysql_NULL_booleans_from_set(engine, mongo_db, batch_start, batch_size = 1000):
    # access the global set of image_id pairs
    global image_ids_list_global
    global global_collection_file
    global table_name
    global list_name
    # print(f"record_mysql_NULL_booleans_from_set with image_ids_list_global len: {len(image_ids_list_global)}")
    exporter = MongoBSONExporter(mongo_db)

    # print(f"recording MYSQL NULL booleans based on queried set for batch starting at {batch_start} with size {batch_size}")
    # Flatten all document names and map to their collection
    # collections_found = set()
    # col_to_collection = exporter.col_to_collection
    # print(f"table name before loop: {table_name}")

    encodings_table_name = "encodings"
    if "hand_landmarks" in global_collection_file:
        encodings_field_name = "mongo_hand_landmarks"
        compare_field_name = ["right_hand", "left_hand"]
    else:
        print("unknown global_collection_file:", global_collection_file)
        return
    
    if list_name == "in_both":
        cell_value = "NULL"
    elif list_name == "only_in_sql":
        cell_value = 1
    else:
        print("unknown list_name:", list_name)
        return
    # to get the mysql table from the global_collection_file
    # go through the document_names_dict to find which collection it is

    thread_image_ids_list = image_ids_list_global[batch_start-1:batch_start + batch_size]
    print(f"Processing {len(thread_image_ids_list)} image_ids in this batch from index {batch_start} to {batch_start + batch_size}")

    if len(thread_image_ids_list) == 0:
        print("No image_ids to process in this batch, returning")
        return

    # query the encodings table for these image_ids to return encoding_id, image_id, and the relevant mongo field
    df = query_encodings_for_encoding_ids(engine, encodings_field_name, thread_image_ids_list)
    
    # set the 

    # df = query_sql(engine, encodings_field_name, batch_start, batch_size, encodings_table_name)
    # print(f"Retrieved {len(df)} rows from {encodings_table_name} for validation")

    def set_values_for_compare_fields(engine, table_name, df, field, cell_value="NULL"):
        # prepare an SQL update that takes the whole batch of 5000 and sets the compare_field_name to NULL where the encodings_field_name is not NULL
        for idx, row in df.iterrows():
            encoding_id = int(row['encoding_id'])
            image_id = int(row['image_id'])
            encodings_value = row[encodings_field_name]
            if pd.notnull(encodings_value):
                # print(f"Setting {field} to NULL for encoding_id={encoding_id}, image_id={image_id} because {field} is present")
                # set the compare_field_name to NULL in the table_name for this encoding_id
                exporter.write_MySQL_value(engine, table_name, "encoding_id", encoding_id, field, cell_value)
                if image_id % 10000 == 0:
                    print(f"Processed encoding_id={encoding_id}, image_id={image_id} for field {field}")
    if isinstance(compare_field_name, list):
        for field in compare_field_name:
            set_values_for_compare_fields(engine, table_name, df, field, cell_value)
    else:
        set_values_for_compare_fields(engine, table_name, df, compare_field_name, cell_value)


def process_batch(batch_start, batch_end, function):
    # Each thread needs its own session and mongo client
    thread_engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
    ThreadSession = sessionmaker(bind=thread_engine)
    thread_session = ThreadSession()
    thread_mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    thread_mongo_db = thread_mongo_client["stock"]
    exporter = MongoBSONExporter(thread_mongo_db)

    calculated_batch_size = (batch_end - batch_start) # adding one to make sure there is not gap even if it creates an overlap of one 
    if function == "record_mysql_NULL_booleans_from_set":
        print(f"Processing batch from {batch_start} to {batch_end} with size {calculated_batch_size} using {function}")
        record_mysql_NULL_booleans_from_set(thread_engine, thread_mongo_db, batch_start, calculated_batch_size)

    thread_session.close()
    thread_mongo_client.close()
    return results_rows

def process_in_batches(min_id, max_id, batch_size, num_threads, this_process, this_function, break_after_first=False):
    for batch_start in range(min_id, max_id + 1, batch_size * num_threads):
        batch_ranges = [
            (start, min(start + batch_size, max_id + 1))
            for start in range(batch_start, min(batch_start + batch_size * num_threads, max_id + 1), batch_size)
        ]
        print(f"Processing batch ranges: {batch_ranges}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(this_process, start, end, this_function) for start, end in batch_ranges]
            if break_after_first:
                return  

if __name__ == "__main__":

    # Get min and max encoding_id for batching
    min_id = last_id + 1
    max_id = session.query(sqlalchemy.func.max(Encodings.encoding_id)).scalar() or 0
    num_threads = 8  # Adjust based on your CPU and IO

    global global_collection_file
    global_collection_file = "hand_landmarks_right_hand" # this will probably need to be updated mannually

    global list_name
    list_name = "in_both"
    # list_name = "mongo_only"

    with open(os.path.join(EXPORT_DIR, f"{list_name}_{global_collection_file}.txt"), "r") as f:
        lines = f.readlines()[1:]  # skip header
        this_list = [line.strip() for line in lines]
    print(f"Loaded set {list_name} for collection {global_collection_file} with {len(this_list)} entries")
    print(f"Sample entries: {list(this_list)[:10]}")

    # make this_list a global variable to be used in process_batch
    global image_ids_list_global
    image_ids_list_global = this_list

    break_after_first = False  # for testing
    function = "record_mysql_NULL_booleans_from_set"
    process_in_batches(min_id, max_id, batch_size, num_threads, process_batch, function, break_after_first)


    session.close()

