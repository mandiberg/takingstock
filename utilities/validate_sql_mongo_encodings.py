import os
import bson
import sqlalchemy
from sqlalchemy import create_engine
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

# Define the batch size
batch_size = 100000
MODE = 1 #0 for overall compare, 1 to recheck against entry and bson dump
IS_FACE = 0
IS_BODY = 1
MODE = 0 
EXPORT_DIR = os.path.join(io.ROOT_PROD,"mongo_exports_sept28")  # Directory to save BSON files
# touch the directory if it does not exist
os.makedirs(EXPORT_DIR, exist_ok=True)
print(f"Export directory: {EXPORT_DIR}")
# select max encoding_id to start from
Base = declarative_base()

table_name = 'compare_sql_mongo_results'
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
Base.metadata.create_all(engine)
# last_id = session.query(sqlalchemy.func.max(CompareSqlMongoResults.encoding_id)).scalar()
# if last_id is None:
#     last_id = 0
last_id = 59000000
print(f"Starting from last_id: {last_id}")

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

def process_batch(batch_start, batch_end):
    # Each thread needs its own session and mongo client
    thread_engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
    ThreadSession = sessionmaker(bind=thread_engine)
    thread_session = ThreadSession()
    thread_mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    thread_mongo_db = thread_mongo_client["stock"]

    calculated_batch_size = (batch_end - batch_start) # adding one to make sure there is not gap even if it creates an overlap of one 
    # validate_zero_columns_against_mongo(thread_engine, thread_mongo_db, document_names_dict, batch_start, calculated_batch_size) # for checking first round validation

    validate_zero_columns_against_mongo_prereshard(thread_engine, thread_mongo_db, document_names_dict, batch_start, calculated_batch_size)
    # results_rows = []

    # results = get_mysql_results(batch_start, batch_end, thread_session)
    # print(f"Thread processing batch {batch_start} to {batch_end}, got {len(results)} results")
    # print(f"First 5 results: {results[:5]}")

    # for encoding_id, image_id, mongo_encodings, mongo_body_landmarks, mongo_face_landmarks, mongo_body_landmarks_norm, mongo_hand_landmarks, mongo_body_landmarks_3D, is_body, is_face in results:
    #     if encoding_id is None or image_id is None:
    #         continue
    #     mongo_docs = {}
    #     for collection_name in collection_names:
    #         collection = thread_mongo_db[collection_name]
    #         doc = collection.find_one({"image_id": image_id})
    #         mongo_docs[collection_name] = doc

    #     this_row = {
    #         "encoding_id": encoding_id,
    #         "image_id": image_id,
    #         "is_body": is_body,
    #         "is_face": is_face
    #     }
    #     for collection_name in collection_names:
    #         doc = mongo_docs[collection_name]
    #         document_names = document_names_dict[collection_name]
    #         for document_name in document_names:
    #             sql_field_name = sql_field_names_dict[document_name]
    #             sql_boolean = locals().get(sql_field_name)
    #             mongo_data_present = doc is not None and document_name in doc and doc[document_name] is not None
    #             if sql_boolean and not mongo_data_present:
    #                 value = 0
    #             elif not sql_boolean and mongo_data_present:
    #                 value = 1
    #             else:
    #                 value = None
    #             this_row[document_name] = value

    #     if any(v is not None for k, v in this_row.items() if k not in ["encoding_id", "image_id", "is_body", "is_face"]):
    #         results_rows.append(this_row)

    thread_session.close()
    thread_mongo_client.close()
    return results_rows

def get_mysql_results(batch_start, batch_end, thread_session):
    if MODE == 0:
        results = (
            thread_session.query(
                Encodings.encoding_id, Encodings.image_id, Encodings.mongo_encodings, Encodings.mongo_body_landmarks,
                Encodings.mongo_face_landmarks, Encodings.mongo_body_landmarks_norm, Encodings.mongo_hand_landmarks,
                Encodings.mongo_body_landmarks_3D, Encodings.is_body, Encodings.is_face
            )
            .filter(
                Encodings.encoding_id >= batch_start,
                Encodings.encoding_id < batch_end
            )
            .order_by(Encodings.encoding_id)
            .all()
        )
    elif MODE == 1:
        results = (
            thread_session.query(
            Encodings.encoding_id, Encodings.image_id, Encodings.mongo_encodings, Encodings.mongo_body_landmarks,
            Encodings.mongo_face_landmarks, Encodings.mongo_body_landmarks_norm, Encodings.mongo_hand_landmarks,
            Encodings.mongo_body_landmarks_3D, Encodings.is_body, Encodings.is_face
            )
            .join(
            CompareSqlMongoResults,
            Encodings.encoding_id == CompareSqlMongoResults.encoding_id
            )
            .filter(
            Encodings.encoding_id >= batch_start,
            Encodings.encoding_id < batch_end,
            CompareSqlMongoResults.is_body == IS_BODY,
            CompareSqlMongoResults.is_face == IS_FACE,
            CompareSqlMongoResults.body_world_landmarks.is_(True),
            )
            .order_by(Encodings.encoding_id)
            .all()
        )
    
    return results

def validate_zero_columns_against_mongo(engine, mongo_db, document_names_dict, batch_start, batch_size = 1000, table_name="compare_sql_mongo_results"):
    import pandas as pd

    # Flatten all document names and map to their collection
    col_to_collection = {}
    for collection, docnames in document_names_dict.items():
        for docname in docnames:
            col_to_collection[docname] = collection

    face_cols = ["face_landmarks", "face_encodings68"]
    body_cols = ["body_landmarks", "nlms", "body_world_landmarks"]

    # Get total rows
    # total_rows = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", engine).iloc[0,0]
    # offset = 0

    # while offset < total_rows:
    docname_cols = [col for docnames in document_names_dict.values() for col in docnames]
    where_clause = "(" + " OR ".join([f"{col} IS NOT NULL" for col in docname_cols]) + f") AND encoding_id >= {batch_start}"    
    # print(f"Querying {table_name} with WHERE {where_clause} LIMIT {batch_size}")
    # print(f"Querying {table_name} with WHERE {where_clause} LIMIT {batch_size}")
    df = pd.read_sql(
        f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT {batch_size}",
        engine
    )
    # offset += batch_size

    for idx, row in df.iterrows():
        encoding_id = row['encoding_id']
        image_id = row['image_id']
        # print(f"encoding_id={encoding_id}, image_id={image_id}")
        for col in col_to_collection:
            if col in row and row[col] == 0:
                if (col in face_cols and row['is_face'] == 0) or (col in body_cols and row['is_body'] == 0):
                    # set the value in SQL to NULL
                    session.execute(
                        sqlalchemy.text(f"UPDATE {table_name} SET {col} = NULL WHERE encoding_id = :encoding_id"),
                        {"encoding_id": encoding_id}
                    )
                    session.commit()
                else:
                    # check MongoDB
                        # print(f" -- just set to NULL {col} for encoding_id={encoding_id} as is_face=0")
                    # if col in body_cols and row['is_body'] == 0:
                    #     # print(f" -- set to NULL {col} for encoding_id={encoding_id} as is_body=0")
                    #     continue
                    # print(f"Checking encoding_id={encoding_id}, image_id={image_id}, column={col} against MongoDB collection {col_to_collection[col]}")
                    collection = mongo_db[col_to_collection[col]]
                    doc = collection.find_one({"image_id": image_id})
                    # print(doc)
                    if doc and col in doc and doc[col] is not None:
                        if col == "body_world_landmarks":
                            session.execute(
                                sqlalchemy.text(f"UPDATE {table_name} SET {col} = NULL WHERE encoding_id = :encoding_id"),
                                {"encoding_id": encoding_id}
                            )
                            session.commit()
                            # print(f" -- just set to NULL {col} for encoding_id={encoding_id} because doc exists and is len({len(doc[col])}) > 0")

                        else:
                            print(f"Discrepancy: encoding_id={encoding_id}, image_id={image_id}, column={col}")
                    else:
                        if "hand" in col:
                            # print(f"Validated zero: encoding_id={encoding_id}, image_id={image_id}, column={col} (hand landmarks often missing)")
                            continue
                        elif col == "body_landmarks" and row['is_body'] == 1:
                            pass
                            # print(f"SQL is_body but MongoDB is NULL: encoding_id={encoding_id}, image_id={image_id}, column={col}")
                        elif col in face_cols and row['is_face'] == 1:
                            pass
                            # print(f"SQL is_face but MongoDB is NULL: encoding_id={encoding_id}, image_id={image_id}, column={col}")
                        else:
                            pass
                            # print(f"Validated zero: encoding_id={encoding_id}, image_id={image_id}, column={col} and is_body is {row['is_body']}, is_face is {row['is_face']}")
        print(f"Checked encoding_id {encoding_id}")
        if encoding_id % 1000 == 0:
            print(f"Checked encoding_id {encoding_id}")

def validate_zero_columns_against_mongo_prereshard(engine, mongo_db, document_names_dict, batch_start, batch_size = 1000, table_name="compare_sql_mongo_results"):
    import pandas as pd
    # define collection names from collection_names list
    for collection_name in collection_names:
        globals()[f"{collection_name}_collection"] = mongo_db[collection_name]

    exporter = MongoBSONExporter(encodings_collection, body_landmarks_norm_collection, hand_landmarks_collection, body_world_landmarks_collection)

    print(f"Validating missing columns against MongoDB for batch starting at {batch_start} with size {batch_size}")
    
    # Flatten all document names and map to their collection
    collections_found = set()
    col_to_collection = {}
    for collection, docnames in document_names_dict.items():
        for docname in docnames:
            col_to_collection[docname] = collection
    col_to_collection = exporter.col_to_collection

    face_cols = ["face_landmarks", "face_encodings68"]
    body_cols = ["body_landmarks", "nlms", "body_world_landmarks"]

    # Get total rows
    # total_rows = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", engine).iloc[0,0]
    # offset = 0

    # while offset < total_rows:
    docname_cols = [col for docnames in document_names_dict.values() for col in docnames]
    where_clause = "(" + " OR ".join([f"{col} IS NOT NULL" for col in docname_cols]) + f") AND encoding_id >= {batch_start} AND encoding_id < {batch_start + batch_size}"
    # print(f"Querying {table_name} with WHERE {where_clause} LIMIT {batch_size}")
    # print(f"Querying {table_name} with WHERE {where_clause} LIMIT {batch_size}")
    try:
        df = pd.read_sql(
            f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT {batch_size}",
            engine
        )
    except Exception as e:
        print(f"Error reading from SQL table {table_name}: {e}")
        df = pd.DataFrame()


    this_batch_dict = {}
    # offset += batch_size
    # print(f"Retrieved {len(df)} rows from {table_name} for validation")
    df_existing_documents = pd.DataFrame()
    for idx, row in df.iterrows():
        this_result_dict = {}
        encoding_id = row['encoding_id']
        image_id = row['image_id']
        # print(f"encoding_id={encoding_id}, image_id={image_id}")
        for col in col_to_collection:
            if col in row and row[col] == 0:
                collection = mongo_db[col_to_collection[col]]
                doc = collection.find_one({"image_id": image_id})
                if doc and col in doc and doc[col] is not None:
                    if col_to_collection[col] == "encodings":
                        this_result_dict["encoding_id"] = encoding_id
                        # print(f" collection is encodings for {col} ")
                    # if there is a result, save it in a dict to write out later
                    print(f"Discrepancy: encoding_id={encoding_id}, image_id={image_id}, column={col}")
                    this_result_dict[col] = (doc[col])
                    collections_found.add(col_to_collection[col])
        if this_result_dict.get("body_world_landmarks", None) is not None and "body_landmarks" not in (this_result_dict, col_to_collection):
            # print("body_landmarks missing but body_world_landmarks present, going to investigate further")
            # check to see if a document exists in encodings collection that has body_landmarks
            enc_doc = encodings_collection.find_one({"image_id": image_id})
            if enc_doc and "body_landmarks" in enc_doc and enc_doc["body_landmarks"] is not None:
                # print(f" >> enc_doc = {enc_doc['body_landmarks']}")
                this_result_dict["body_landmarks"] = enc_doc["body_landmarks"]
                # print(f" ++ found body_landmarks in encodings collection for image_id {image_id}")
                collections_found.add("encodings")
                # print(f"collections_found now: {collections_found}")
                # print(f"this_result_dict now: {this_result_dict}")
            else:
                print(f" ~~ did not find body_landmarks in encodings collection for image_id {image_id}")

        if this_result_dict:
            # print(this_result_dict)
            this_batch_dict[image_id] = this_result_dict
            # for key in this_result_dict:
            #     print(f"++ found data for {encoding_id} column {key} with length {len(this_result_dict[key])}")
        # print(f"Checked encoding_id {encoding_id}")
        if encoding_id % 100000 == 0:
            print(f"Checked encoding_id {encoding_id}")
    # print(len(this_batch_dict))

    # store the results in text file as BSONn using exporter.write_bson_batches which requires batch_bson, offset, export_dir   
    if this_batch_dict:
        print(f"going to write {len(this_batch_dict)} documents to BSON files for batch starting at {batch_start}")
        exporter.write_bson_batches(this_batch_dict, batch_start, EXPORT_DIR, collections_found)
    # print(f"Wrote missing columns to missing_columns_batch_{batch_start}.bson")

def read_and_store_bson(engine, mongo_db, document_names_dict, table_name="compare_sql_mongo_results"):
    import pandas as pd
    # define collection names from collection_names list
    for collection_name in collection_names:
        globals()[f"{collection_name}_collection"] = mongo_db[collection_name]

    exporter = MongoBSONExporter(encodings_collection, body_landmarks_norm_collection, hand_landmarks_collection, body_world_landmarks_collection)
    print(f"reading and writing bson data from EXPORT_DIR {EXPORT_DIR} to SQL table {table_name}")
    
    # Flatten all document names and map to their collection
    batch_size = 2
    col_to_collection = exporter.col_to_collection

    list_of_bson_files = [f for f in os.listdir(EXPORT_DIR) if f.endswith('.bson')]
    print(f"Found {len(list_of_bson_files)} BSON files in {EXPORT_DIR}")

    this_collection_docs = []

    # get full filepaths for each file in list_of_bson_files
    collection_files_dict = exporter.build_batch_list(EXPORT_DIR, batch_size)
    for collection_name, batches_list in collection_files_dict.items():
        for batch in batches_list:
            all_docs = exporter.read_batch(batch)
            this_collection_docs.extend(all_docs)
        print(f"Processing batch of {len(this_collection_docs)} documents for collection {collection_name}")

        for doc in this_collection_docs:
            print(doc)

if __name__ == "__main__":

    # Get min and max encoding_id for batching
    # min_id = session.query(sqlalchemy.func.min(Encodings.encoding_id)).scalar() or 0
    min_id = last_id + 1
    max_id = session.query(sqlalchemy.func.max(Encodings.encoding_id)).scalar() or 0

    num_threads = 8  # Adjust based on your CPU and IO

    # for batch_start in range(min_id, max_id + 1, batch_size):
    #     # print(f"Processing batch starting at {batch_start}")
    #     validate_zero_columns_against_mongo(engine, mongo_db, document_names_dict, batch_start, batch_size)

    if MODE ==0:
        # export to BSON in multithreaded way from pre-reshard mongo db
        for batch_start in range(min_id, max_id + 1, batch_size * num_threads):
            batch_ranges = [
                (start, min(start + batch_size, max_id + 1))
                for start in range(batch_start, min(batch_start + batch_size * num_threads, max_id + 1), batch_size)
            ]
            results_rows = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_batch, start, end) for start, end in batch_ranges]
            # break  # temporary for testing

    elif MODE == 1:
        # read from BSON and write to mongo and sql
        table_name = 'compare_sql_mongo_results'
        read_and_store_bson(engine, mongo_db, document_names_dict, table_name)

    session.close()

