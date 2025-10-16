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
IS_FACE = 0
IS_BODY = 1
MODE = 1  # 0 validate_zero_columns_against_mongo_prereshard (outputs bson)s
#0 validate_zero_columns_against_mongo_prereshard (outputs bson) 
# 1 read_and_store_bson
if MODE == 0: FOLDER_MODE = 0 # 0 is the first way, 1 is by filepath, limit 1
else: FOLDER_MODE = 1 # 0 is the first way, 1 is by filepath, limit 1

## OVERRIDE for NML PC doover ##
# FOLDER_MODE = 0

# 2 find entries present in mongo, but not recorded in sql table
# 57607900
last_id = 0
print(f"Starting from last_id: {last_id}")
EXPORT_DIR = os.path.join(io.ROOT_PROD,"mongo_exports_oct14_SSD4green/body_world_landmarks")  # Directory to save BSON files
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


def get_mysql_results(batch_start, batch_end, thread_session):
    if MODE == 0:
        if HelperTable_name is not None:
            results = (
                thread_session.query(
                    Encodings.encoding_id, Encodings.image_id, Encodings.mongo_encodings, Encodings.mongo_body_landmarks,
                    Encodings.mongo_face_landmarks, Encodings.mongo_body_landmarks_norm, Encodings.mongo_hand_landmarks,
                    Encodings.mongo_body_landmarks_3D, Encodings.is_body, Encodings.is_face
                )
                .join(
                    HelperTable,
                    Encodings.image_id == HelperTable.image_id
                )
                .filter(
                    Encodings.encoding_id >= batch_start,
                    Encodings.encoding_id < batch_end
                )
                .order_by(Encodings.encoding_id)
                .all()
            )
        else:
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

def validate_zero_columns_against_mongo(engine, mongo_db, document_names_dict, batch_start, table_name, batch_size = 1000):
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
                            pass
                            # print(f"Discrepancy: encoding_id={encoding_id}, image_id={image_id}, column={col}")
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

def validate_zero_columns_against_mongo_prereshard(engine, mongo_db, document_names_dict, batch_start, table_name, batch_size = 1000):
    print(f"validate_zero_columns_against_mongo_prereshard batch starting at {batch_start} with size {batch_size}")
    # define collection names from collection_names list
    for collection_name in collection_names:
        globals()[f"{collection_name}_collection"] = mongo_db[collection_name]

    exporter = MongoBSONExporter(mongo_db)

    # print(f"Validating missing columns against MongoDB for batch starting at {batch_start} with size {batch_size}")
    
    # Flatten all document names and map to their collection
    collections_found = set()
    col_to_collection = {}
    for collection, docnames in document_names_dict.items():
        for docname in docnames:
            col_to_collection[docname] = collection
    col_to_collection = exporter.col_to_collection

    # Get total rows
    # total_rows = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", engine).iloc[0,0]
    # offset = 0

    # while offset < total_rows:
    df = query_sql(engine, document_names_dict, batch_start, batch_size, table_name)
    
    # print("57624173 row is", df.loc[df['encoding_id'] == 57624173])
    # print(df)
    # check to see if encoding_id 57624173 is in the df
    # if 57624173 in df['encoding_id'].values:
    #     # print(df.dtypes)
    #     # print("Found row with encoding_id 57624173 :", df.loc[df['encoding_id'] == 57624173])
    #     df['body_world_landmarks'] = df['body_world_landmarks'].astype('Int64')
    #     df['nlms'] = df['nlms'].astype('Int64')
    #     # print("Found row with encoding_id 57624173 :", df.loc[df['encoding_id'] == 57624173])
    #     # print(df.loc[df['encoding_id'] == 57624173, ['body_world_landmarks', 'nlms']])
    #     # print(df['body_world_landmarks'].dtype)
    #     # print(df['nlms'].dtype)        
    # else:
    #     print("Did not find encoding_id 57624173 in the dataframe")
        
    this_batch_dict = {}
    # offset += batch_size
    # print(f"Retrieved {len(df)} rows from {table_name} for validation")
    for idx, row in df.iterrows():
        this_result_dict = {}
        encoding_id = int(row['encoding_id'])
        image_id = int(row['image_id'])
        for col in col_to_collection:
            col_value = row.get(col, None)
            # print(f"looking for {col}: {col_value} for encoding_id={encoding_id}")
            if col_value == 0:
                # print(f"found encoding_id={encoding_id}, image_id={image_id}, column={col} is zero in SQL, checking MongoDB")
                collection = mongo_db[col_to_collection[col]]
                doc = collection.find_one({"image_id": image_id})
                # print(doc)
                # if encoding_id == 849111:
                #     print("debug breakpoint")
                #     print(doc)
                #     print(row)
                #     break
                # print(f"encoding id {encoding_id} looking for {col} in {doc.keys()}")
                if doc and col in doc and doc[col] is not None:
                    # print(f" -- found data in mongo collection {col_to_collection[col]} for {col} for encoding_id={encoding_id}")
                    if col_to_collection[col] == "encodings":
                        # this is a special case to just search for encoding_id in encodings collection,
                        # because it has a different schema that includes encoding_id

                        ### commented out for production run
                        # print(f" -- found data in encodings collection for {col} for encoding_id={encoding_id}")
                        this_result_dict["encoding_id"] = encoding_id
                        # print(f" collection is encodings for {col} ")
                    # this is the general case where it saves all of them??
                    # if there is a result, save it in a dict to write out later
                    ### commented out for production run
                    # print(f"Discrepancy: encoding_id={encoding_id}, image_id={image_id}, column={col}")
                    # set the column value for this result
                    this_result_dict[col] = (doc[col])
                    # note that we found this collection has data, to write out later
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

def query_sql(engine, document_names_dict, batch_start, batch_size, table_name):
    docname_cols = [col for docnames in document_names_dict.values() for col in docnames]
    where_clause = "(" + " OR ".join([f"{col} IS NOT NULL" for col in docname_cols]) + f") AND encoding_id >= {batch_start} AND encoding_id < {batch_start + batch_size}"
    query = f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT {batch_size}"
    
    # print(f"Querying {table_name} with WHERE {where_clause} LIMIT {batch_size}")
    # print(f"Querying {table_name} with WHERE {where_clause} LIMIT {batch_size}")
    # try:
    #     with engine.connect() as conn:
    #         result = conn.execute(text(query))
    #         rows = result.fetchall()
    #         columns = result.keys()
    #         # print(f"Fetched {len(rows)} rows from SQL")
    #         # print(f"rows: {rows[:2]}")  # Print first 2 rows for inspection
    #         # find the row with encoding_id 57624173
    #         for row in rows:
    #             # row is a tuple, convert to dict for easier access
    #             row = dict(zip(columns, row))
    #             # print(row)
    #             if row['encoding_id'] == 57624173:
    #                 print(f"Found SQL row with encoding_id 57624173: {row}")
    #                 break
    #         # Convert to DataFrame
    #         df = pd.DataFrame(rows, columns=columns)
    #         # Explicitly set dtypes for integer columns that may have NULLs
    #         int_cols = [
    #             'encoding_id', 'image_id', 'face_landmarks', 'body_landmarks',
    #             'face_encodings68', 'nlms', 'left_hand', 'right_hand',
    #             'body_world_landmarks', 'is_face', 'is_body'
    #         ]
    #         for col in int_cols:
    #             if col in df.columns:
    #                 df[col] = df[col].astype('Int64')

    try:
        df = pd.read_sql(
            f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT {batch_size}",
            engine
        )
    except Exception as e:
        print(f"Error reading from SQL table {table_name}: {e}")
        df = pd.DataFrame()
    return df
    # print(f"Wrote missing columns to missing_columns_batch_{batch_start}.bson")


def record_mysql_NULL_booleans_present_in_mongo(engine, mongo_db, document_names_dict, batch_start, table_name, batch_size = 1000):
    for collection_name in collection_names:
        globals()[f"{collection_name}_collection"] = mongo_db[collection_name]

    exporter = MongoBSONExporter(mongo_db)

    print(f"recording MYSQL NULL booleans for data present in MongoDB for batch starting at {batch_start} with size {batch_size}")
    # Flatten all document names and map to their collection
    collections_found = set()
    # col_to_collection = exporter.col_to_collection

    df = query_sql(engine, document_names_dict, batch_start, batch_size, table_name)
    this_batch_dict = {}
    # offset += batch_size
    print(f"Retrieved {len(df)} rows from {table_name} for validation")
    for idx, row in df.iterrows():
        this_result_dict = {}
        encoding_id = int(row['encoding_id'])
        image_id = int(row['image_id'])
        # print(f"encoding_id={encoding_id}, image_id={image_id}")
        for col in exporter.col_to_collection:
            if col in row and row[col] == 1:
                collection = mongo_db[exporter.col_to_collection[col]]
                doc = collection.find_one({"image_id": image_id})
                if doc and col in doc and doc[col] is not None:
                    print(f"Discrepancy: mongo data found encoding_id={encoding_id}, image_id={image_id}, column={col}")

                    # update booleans in mysql tables
                    exporter.write_MySQL_value(engine, "encodings", "image_id", image_id, exporter.sql_field_names_dict[col], cell_value=1)
                    # exporter.write_MySQL_value(engine, "segmentBig_isface", "image_id", image_id, exporter.sql_field_names_dict[col], cell_value=1)

                    # # set table_name value to NULL
                    exporter.write_MySQL_value(engine, table_name, "encoding_id", encoding_id, col, cell_value="NULL")

def process_batch(batch_start, batch_end, function):
    global table_name
    # Each thread needs its own session and mongo client
    thread_engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
    ThreadSession = sessionmaker(bind=thread_engine)
    thread_session = ThreadSession()
    thread_mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    thread_mongo_db = thread_mongo_client["stock"]
    exporter = MongoBSONExporter(thread_mongo_db)

    print(f"Thread processing batch from {batch_start} of type {type(batch_start)} to {batch_end} of type {type(batch_end)} using {function}")
    # this catches whether this function is called with encoding_id/image_id or a file list
    if isinstance(batch_start, int) and isinstance(batch_end, int) and FOLDER_MODE == 0:
        # print(f"this is a start end of encoding_id range: {batch_start} to {batch_end}")
        calculated_batch_size = (batch_end - batch_start) # adding one to make sure there is not gap even if it creates an overlap of one 
    # validate_zero_columns_against_mongo(thread_engine, thread_mongo_db, document_names_dict, batch_start, calculated_batch_size) # for checking first round validation
        # print(f"Processing batch from {batch_start} to {batch_end} with size {calculated_batch_size} using {function}")
        if function == "validate_zero_columns_against_mongo_prereshard":
            validate_zero_columns_against_mongo_prereshard(thread_engine, thread_mongo_db, document_names_dict, batch_start, table_name,calculated_batch_size)
        elif function == "record_mysql_NULL_booleans_present_in_mongo":
            # print(f"Processing batch from {batch_start} to {batch_end} with size {calculated_batch_size} using {function}")
            record_mysql_NULL_booleans_present_in_mongo(thread_engine, thread_mongo_db, document_names_dict, batch_start, table_name, calculated_batch_size)
        else:
            print("if isinstance, Unknown function:", function)
    elif function == "read_and_store_bson_batch":
        # print(f"this is a list of files or batches of files: {batch_start} to {batch_end}")
        # handle list vs single file
        if FOLDER_MODE == 0:
            batch_list = batch_start
            collection = batch_end
        elif FOLDER_MODE == 1:
            batch_list = batch_start
            collection = None
        # calculated_batch_size = len(batch_start) # list of tuples
        exporter.read_and_store_bson_batch(thread_engine, thread_mongo_db, batch_list, collection, table_name)
    else:
        print("unknown function:", function)

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

def process_batch_list_in_batches(batch_list, num_threads, this_process, this_function, break_after_first=False):
    collections = batch_list.keys()
    for collection in collections:
        print(f"Collection: {collection} has {len(batch_list[collection])} batches to process")
        for batch in batch_list[collection]:
            print(f"This Batch Collection: {collection} has batch len: {len(batch)}")
    # batch_list = [(collection, batch) for collection in collections for batch in batch_list[collection]]
    # print(f"Processing {len(batch_list)} batches in batch_list with {num_threads} threads")
    # print(batch_list)

    # for collection, batch in batch_list:
        # print(f"Processing {collection} batch : {batch_list[collection]}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                # passing in a list of files as first argument, None as second argument triggers read_and_store_bson
                executor.submit(this_process, batch, collection, this_function)
                if break_after_first:
                    return  


def process_file_list_in_batches(batch_list, num_threads, this_process, this_function, break_after_first=False):
    print(f"Processing {len(batch_list)} files with {num_threads} threads")
    # print(batch_list)
    for file in batch_list:
        print(f"This is the file to process : {file}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # passing in a list of files as first argument, None as second argument triggers read_and_store_bson
            executor.submit(this_process, file, None, this_function)
            if break_after_first:
                return  


if __name__ == "__main__":

    # Get min and max encoding_id for batching
    min_id = last_id + 1
    max_id = session.query(sqlalchemy.func.max(Encodings.encoding_id)).scalar() or 0
    num_threads = 8  # Adjust based on your CPU and IO

    if MODE == 0:
        # export to BSON in multithreaded way from pre-reshard mongo db
        function = "validate_zero_columns_against_mongo_prereshard"
        print(f"fProcessing encoding_id rom {min_id} to {max_id} in batches of {batch_size} with function {function}")
        for batch_start in range(min_id, max_id + 1, batch_size * num_threads):
            batch_ranges = [
                (start, min(start + batch_size, max_id + 1))
                for start in range(batch_start, min(batch_start + batch_size * num_threads, max_id + 1), batch_size)
            ]
            results_rows = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_batch, start, end, function) for start, end in batch_ranges]
            # break  # temporary for testing

    elif MODE == 1:
        # build batch dict + list of files - dict.keys() are collection names
        # each value is a list of lists of files. The sublists are the batchs, of len=batch_size
        # read from BSON and write to mongo and sql
        # table_name = 'compare_sql_mongo_results'
        function = "read_and_store_bson_batch"
        if FOLDER_MODE == 0:
            # use batch list builder -- where each doc is its own bson file
            collection_files_dict = exporter.build_batch_list(session, EXPORT_DIR, batch_size)
            process_batch_list_in_batches(collection_files_dict, num_threads, process_batch, function)
        else:
            # use file list builder
            list_of_bson_files = exporter.build_folder_bson_file_list_full_paths(session, EXPORT_DIR, batch_size=num_threads)
            # select all completed_bson_file from BsonFileLog to skip those
            # list_of_bson_files = exporter.remove_already_completed_files(session, list_of_bson_files)
            print("list_of_bson_files number of collections: ", len(list_of_bson_files))
            process_file_list_in_batches(list_of_bson_files, num_threads, process_batch, function, break_after_first=False)
        # read_and_store_bson(engine, mongo_db, document_names_dict, table_name)

    elif MODE == 2:
        # find entries present in mongo, but not recorded in sql table
        break_after_first = False  # for testing
        function = "record_mysql_NULL_booleans_present_in_mongo"
        process_in_batches(min_id, max_id, batch_size, num_threads, process_batch, function, break_after_first)

    session.close()

