import csv
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
# mongo_collection = mongo_db["body_landmarks_norm"]
exporter = MongoBSONExporter(mongo_db)

# Define the batch size
batch_size = 5000
IS_FACE = 0
IS_BODY = 1
MODE = 1  # 0 validate_zero_columns_against_mongo_prereshard (outputs bson)s
#0 validate_zero_columns_against_mongo_prereshard (outputs bson) 
# 1 read_and_store_bson
# 4 checks output from #3 against actual mysql table and updates
# 5 takes helper table csvs with missing mongo data and checks to see if data exists in mongo, if so outputs bson files
# 6 compares sql_only image_ids from previous step to mongo bson files

if MODE in [0,4,5]: FOLDER_MODE = 0 # 0 is the first way, 1 is by filepath, limit 1
else: FOLDER_MODE = 1 # 0 is the first way, 1 is by filepath, limit 1

## OVERRIDE for NML PC doover ##
# FOLDER_MODE = 0

# 2 find entries present in mongo, but not recorded in sql table
# 57607900

last_id = 0  # starting from encoding_id 32920000
print(f"Starting from last_id: {last_id}")
IMPORT_DIR = "/users/michaelmandiberg"
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

# redefining Table Name for final ingest Nov 30 2025
# this is only for updating it to note when values stored
table_name = "SegmentHelperMissing_nov2025"

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

def query_sql(engine, document_names_dict_or_string, batch_start, batch_size, table_name):
    print(f"query_sql batch for {document_names_dict_or_string} starting at {batch_start} with size {batch_size}")
    if isinstance(document_names_dict_or_string, str):
        # this is only when I am looking for encodings
        document_names_dict = {document_names_dict_or_string: [document_names_dict_or_string]}
        select = f"image_id, encoding_id, {document_names_dict_or_string}"
    else:
        document_names_dict = document_names_dict_or_string
        select = "*"
    
    if batch_start == 0 and batch_size == 0:
        image_ids_list = list(image_ids_set_global)
        image_ids_set_global_str = ", ".join([str(i) for i in image_ids_list])
        image_ids_set_global_sql = f"({image_ids_set_global_str})"
        limit = " "
        where_clause = f" image_id IN {image_ids_set_global_sql} "
    else:
        limit = f" LIMIT {batch_size} "
        where_clause = "(" + " OR ".join([f"{col} IS NOT NULL" for col in docname_cols]) + f" AND encoding_id >= {batch_start} AND encoding_id < {batch_start + batch_size} "  + ")"

    docname_cols = [col for docnames in document_names_dict.values() for col in docnames]
    query = f"SELECT {select} FROM {table_name} WHERE {where_clause} {limit}"
    print(f"Executing query: {query[:200]}")
    try:
        df = pd.read_sql(
            query,
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
                    exporter.update_MySQL_value(engine, "encodings", "image_id", image_id, exporter.sql_field_names_dict[col], cell_value=1)
                    # exporter.update_MySQL_value(engine, "segmentBig_isface", "image_id", image_id, exporter.sql_field_names_dict[col], cell_value=1)

                    # # set table_name value to NULL
                    exporter.update_MySQL_value(engine, table_name, "encoding_id", encoding_id, col, cell_value="NULL")


def record_mysql_NULL_booleans_from_set(engine, mongo_db, document_names_dict, batch_start, table_name, batch_size = 1000):
    # access the global set of image_id pairs
    global image_ids_set_global
    global global_collection_file

    exporter = MongoBSONExporter(mongo_db)

    print(f"recording MYSQL NULL booleans based on image_ids_set_global with size {len(image_ids_set_global)}")
    # Flatten all document names and map to their collection
    # collections_found = set()
    col_to_collection = exporter.col_to_collection
    # print(f"table name before loop: {table_name}")
    encodings_table_name = "encodings"

    # kludge to handle body_landmarks_norm case where collection name differs from key
    if "body_landmarks_norm" in global_collection_file: col = "nlms"
    else: col = global_collection_file

    collection_name = col_to_collection.get(col, None)
    print(f"collection_name is: {collection_name} for global_collection_file {global_collection_file}")
    # print(f"global_collection_file is: {global_collection_file}", "document_names_dict is:", document_names_dict)

    # encodings_field_name = None
    encodings_field_name = sql_field_names_dict.get(col, None)
    # # try to set encodings_field_name based on global_collection_file
    # for key, value in document_names_dict.items():
    #     if global_collection_file in value:
    #         for v in value:
    #             if global_collection_file == v:
    #                 encodings_field_name = v
    #         break
    # print(f"encodings_field_name set to {encodings_field_name} for global_collection_file {global_collection_file}")


    if "hand_landmarks" in global_collection_file:
        encodings_field_name = "mongo_hand_landmarks"
    elif encodings_field_name is None:
        print("unknown global_collection_file:", global_collection_file)
        return
    # to get the mysql table from the global_collection_file
    # go through the document_names_dict to find which collection it is
    df = query_sql(engine, encodings_field_name, batch_start, batch_size, encodings_table_name)
    print(f"Retrieved {len(df)} rows from {encodings_table_name} for validation")
    this_batch_dict = {}
    # offset += batch_size
    print(f"Retrieved {len(df)} rows from {table_name} for validation")
    print(df)

    # make sure all image_ids_set_global are integers
    image_ids_set_global_int = set([int(i) for i in image_ids_set_global])

    # image_ids_set_global that are not in df
    df_image_ids_set = set(df['image_id'].tolist())
    print(f"df_image_ids_set has {len(df_image_ids_set)} image_ids")
    missing_image_ids = image_ids_set_global_int - df_image_ids_set
    print(f"Found {len(missing_image_ids)} image_ids in global set not in df")

    for image_id in image_ids_set_global_int:
        if "body" in encodings_field_name: is_boolean = "is_body"
        elif "face" in encodings_field_name: is_boolean = "is_face"
        elif "hand" in encodings_field_name: is_boolean = "mongo_hand_landmarks"
        else: is_boolean = None

        if image_id in df_image_ids_set:
            existing_value = df.loc[df['image_id'] == image_id, encodings_field_name].values[0]
            print(f"  +  image_id {image_id} from global set found in df with existing value {existing_value} for field {encodings_field_name}")
            # do stuff here using image_id as index
            exporter.update_MySQL_value(engine, encodings_table_name, "image_id", image_id, encodings_field_name, cell_value=1)
            if is_boolean is not None:
                # also write is_boolean to 1
                exporter.update_MySQL_value(engine, encodings_table_name, "image_id", image_id, is_boolean, cell_value=1)
            # print(f"collection_name is: {collection_name}")
            if collection_name == "encodings":
                # check for agreement between encoding_id in mongo and sql
                # get the encoding_id of this image_id from sql
                sql_encoding_id = session.query(Encodings).filter_by(image_id=image_id).first().encoding_id
                # get the encoding_id of this image_id from mongo
                collection = mongo_db[collection_name]
                mongo_doc = collection.find_one({"image_id": image_id})
                if mongo_doc is not None:
                    mongo_encoding_id = mongo_doc.get("encoding_id", None)
                    # print(f"image_id {image_id} has sql_encoding_id {sql_encoding_id} and mongo_encoding_id {mongo_encoding_id}")
                    if mongo_encoding_id != sql_encoding_id:
                        # print(f"Mismatch encoding_id for image_id {image_id}: SQL has {sql_encoding_id}, Mongo has {mongo_encoding_id}. Updating Mongo.")
                        # update the mongo document to set encoding_id to sql_encoding_id
                        result = collection.update_one(
                            {"image_id": image_id},
                            {"$set": {"encoding_id": sql_encoding_id}}
                        )
                        if result.modified_count > 0:
                            print(f"Updated MongoDB document for image_id {image_id} with encoding_id {sql_encoding_id}")

        else:
            print(f" --- image_id {image_id} from global set not found in df >>> inserting new row in encodings table")
            # insert into encodings table with image_id and set the encodings_field_name to 1
            new_encoding = Encodings(
                image_id=image_id,
                **{encodings_field_name: 1}
            )
            session.add(new_encoding)
            session.commit()
            if is_boolean is not None:
                # also write is_boolean to 1
                exporter.update_MySQL_value(engine, encodings_table_name, "image_id", image_id, is_boolean, cell_value=1)
            print(f" >>> Inserted new encoding for image_id {image_id} with {encodings_field_name} set to 1")
            if collection_name == "encodings":
                # update the encoding_id, only for the encodings collection
                # get the encoding_id of the newly inserted row
                new_encoding_id = session.query(Encodings).filter_by(image_id=image_id).first().encoding_id
                print(f"Inserted new encoding for image_id {image_id} with encoding_id {new_encoding_id}")

                # update the mongo document to set encoding_id to new_encoding_id
                collection = mongo_db[collection_name]
                result = collection.update_one(
                    {"image_id": image_id},
                    {"$set": {"encoding_id": new_encoding_id}}
                )
                if result.modified_count > 0:
                    print(f"Updated MongoDB document for image_id {image_id} with encoding_id {new_encoding_id}")

    # for idx, row in df.iterrows():

    #     image_id = int(row['image_id'])
    #     # print(f"encoding_id={encoding_id}, image_id={image_id}")
    #     if image_id in image_ids_set_global_int:
    #         print(f"  +  image_id {image_id} found in global set, checking columns")
    #     else:
    #         print(f" --- image_id {image_id} NOT found in global set, skipping")
    #         continue
        # for col in exporter.col_to_collection:
        #     if col in row and row[col] == 1:
        #         collection = mongo_db[exporter.col_to_collection[col]]
        #         doc = collection.find_one({"image_id": image_id})
        #         if doc and col in doc and doc[col] is not None:
        #             print(f"Discrepancy: mongo data found encoding_id={encoding_id}, image_id={image_id}, column={col}")

        #             # update booleans in mysql tables
        #             exporter.update_MySQL_value(engine, "encodings", "image_id", image_id, exporter.sql_field_names_dict[col], cell_value=1)
        #             # exporter.update_MySQL_value(engine, "segmentBig_isface", "image_id", image_id, exporter.sql_field_names_dict[col], cell_value=1)

        #             # # set table_name value to NULL
        #             exporter.update_MySQL_value(engine, table_name, "encoding_id", encoding_id, col, cell_value="NULL")

def check_NMLmongo_for_mysql_only_image_ids(engine, mongo_db, document_names_dict, batch_start, table_name, batch_size = 1000):
    global image_ids_set_global # all missing image_ids 
    global global_collection_file # the mongo document and/or collection to check
    global missing_mongo_set # the missing image_ids in mongo for this collection
    
    if "hand" in global_collection_file:
        col = ["left_hand", "right_hand"]
    elif "body_landmarks_norm" in global_collection_file:
        col = "nlms"
    else:
        col = global_collection_file
    exporter = MongoBSONExporter(mongo_db)

    intersecting_image_ids = image_ids_set_global.intersection(missing_mongo_set)
    this_missing_mongo_set = intersecting_image_ids
    print(f"check_NMLmongo_for_mysql_only_image_ids found {len(this_missing_mongo_set)} missing image_ids for collection {col}")

    def process_image_id(image_id, doc, col, this_result_dict):
        this_doc_values = doc.get(col, None)
        if this_doc_values is not None:
            print(f" ++  found {col} values for image_id={image_id} with length {len(this_doc_values)}")
            this_result_dict[col] = this_doc_values
            # print(f" -- found data in mongo collection {col} for image_id={image_id}")
            this_result_dict["image_id"] = image_id
            this_result_dict["encoding_id"] = doc.get("encoding_id", None)
            # note that we found this collection has data, to write out later
            # collections_found.add(exporter.col_to_collection.get(col, None))
        return this_result_dict

    # construct mongo connection via mongo_db and col
    if isinstance(col, list):
        collection_name = exporter.col_to_collection.get(col[0], None)
    else:
        collection_name = exporter.col_to_collection.get(col, None)
    collection = mongo_db[collection_name]

    # collections_found = set()
    this_batch_dict = {}
    for image_id in this_missing_mongo_set:
        print(f"checking image_id {image_id} in mongo collection {collection_name}")
        doc = collection.find_one({"image_id": image_id})
        this_result_dict = {}
        if doc is not None:
            if isinstance(col, list):
                for this_col in col:
                    this_result_dict = process_image_id(image_id, doc, this_col, this_result_dict)
            else:
                this_result_dict = process_image_id(image_id, doc, col, this_result_dict)
            # print(f"  +  {image_id}, found document in MongoDB collection {col}, doc: {len(doc)} keys")
            # this_doc_values = doc.get(col, None)
            # if this_doc_values is not None:
            #     print(f" ++  found {col} values for image_id={image_id} with length {len(this_doc_values)}")
            #     this_result_dict[col] = this_doc_values
            #     # print(f" -- found data in mongo collection {col} for image_id={image_id}")
            #     this_result_dict["image_id"] = image_id
            #     this_result_dict["encoding_id"] = doc.get("encoding_id", None)
            #     # note that we found this collection has data, to write out later
            #     # collections_found.add(exporter.col_to_collection.get(col, None))

            if this_result_dict:
                # print(this_result_dict)
                this_batch_dict[image_id] = this_result_dict
                # for key in this_result_dict:
                #     print(f"++ found data for {encoding_id} column {key} with length {len(this_result_dict[key])}")
            # print(f"Checked encoding_id {encoding_id}")


        else:
            print(f" --- image_id {image_id} still missing in MongoDB collection {collection_name}")
    if this_batch_dict:
        print(f"going to write {len(this_batch_dict)} documents to BSON files for batch ")
        exporter.write_bson_batches(this_batch_dict, image_id, EXPORT_DIR, [col])
        

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
        elif function == "record_mysql_NULL_booleans_from_set":
            # print(f"Processing batch from {batch_start} to {batch_end} with size {calculated_batch_size} using {function}")
            record_mysql_NULL_booleans_from_set(thread_engine, thread_mongo_db, document_names_dict, batch_start, table_name, calculated_batch_size)
        elif function == "check_NMLmongo_for_mysql_only_image_ids":
            check_NMLmongo_for_mysql_only_image_ids(thread_engine, thread_mongo_db, document_names_dict, batch_start, table_name, calculated_batch_size)
            # only_in_mongo, only_in_sql, in_both = get_both_indexes(thread_engine, thread_mongo_db, exporter, "body_landmarks_norm", "nlms", "mongo_body_landmarks_norm")
            # save_sets("body_landmarks_norm", only_in_mongo, only_in_sql, in_both)
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


def get_both_indexes(engine, mongo_db, exporter, collection_name, doc_name, sql_field_name):
    print(f"Checking collection: {collection_name}, document: {doc_name}, SQL field: {sql_field_name}")

    mongo_index = exporter.get_mongo_index(mongo_db, collection_name, doc_name)
    print(f"MongoDB index for collection {collection_name} contains len: {len(mongo_index)} entries")
    print(mongo_index[:5])  # print first 5 entries
    # mongo_index_simple = [doc['image_id'] for doc in mongo_index]
    mongo_index_simple = mongo_index
    print(f"MongoDB index as simple list: {mongo_index_simple[:5]}")
    mongo_index_set = set(mongo_index_simple)        # compare with SQL index

    where = f"{sql_field_name} = 1"
    sql_index = exporter.get_sql_index(engine, "Encodings", where)
    print(f"SQL index for collection {collection_name} contains len: {len(sql_index)} entries")
    print(sql_index[:5])  # print first 5 entries
    sql_index_set = set(sql_index)
    # find discrepancies
    only_in_mongo = mongo_index_set - sql_index_set
    only_in_sql = sql_index_set - mongo_index_set
    in_both = mongo_index_set & sql_index_set


    return only_in_mongo, only_in_sql, in_both

def save_sets(doc_name, only_in_mongo, only_in_sql, in_both):
    with open(os.path.join(EXPORT_DIR, f"mongo_only_{doc_name}.txt"), "w") as f:
        f.write(f"Entries only in MongoDB ({len(only_in_mongo)}):\n")
        for entry in only_in_mongo:
            f.write(f"{entry}\n")

        # Save only_in_sql
    with open(os.path.join(EXPORT_DIR, f"sql_only_{doc_name}.txt"), "w") as f:
        f.write(f"Entries only in SQL ({len(only_in_sql)}):\n")
        for entry in only_in_sql:
            f.write(f"{entry}\n")

        # Save in_both
    with open(os.path.join(EXPORT_DIR, f"in_both_{doc_name}.txt"), "w") as f:
        f.write(f"Entries in both ({len(in_both)}):\n")
        for entry in in_both:
            f.write(f"{entry}\n")

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
    
    elif MODE == 3:
        # compare mongo index to sql index
        collection_name = "encodings"

        document_names= document_names_dict[collection_name]
        for doc_name in document_names:
            sql_field_name = sql_field_names_dict[doc_name]
            print(f"Checking collection: {collection_name}, document: {doc_name}, SQL field: {sql_field_name}")
            if sql_field_name == "mongo_hand_landmarks": continue

            mongo_only, sql_only, in_both = get_both_indexes(engine, mongo_db, exporter, collection_name, doc_name, sql_field_name)

            print(f"Entries only in MongoDB ({len(mongo_only)}): {list(mongo_only)[:10]}")
            print(f"Entries only in SQL ({len(sql_only)}): {list(sql_only)[:10]}")
            print(f"Entries in both ({len(in_both)}): {list(in_both)[:10]}")

            # save the results for each mongo_only only_insql in_both to their own text file
            # Save mongo_only
            save_sets(doc_name, mongo_only, sql_only, in_both)

    elif MODE == 4:
        # open the sets saved in MODE 3 to update MYSQL table
        global_collection_file = "hand_landmarks_left_hand" # this will probably need to be updated mannually

        # set_name = "in_both"
        set_name = "mongo_only"
        with open(os.path.join(EXPORT_DIR, f"{set_name}_{global_collection_file}.txt"), "r") as f:
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
            this_set = set(line.strip() for line in remaining_lines)
            if first_id is not None:
                this_set.add(first_id)

        print(f"Loaded set {set_name} for collection {global_collection_file} with {len(this_set)} entries")
        print(f"Sample entries: {list(this_set)[:10]}")

        # OLD WAY to batch, running through whole thing
        # # make this_set a global variable to be used in process_batch
        # global image_ids_set_global
        # image_ids_set_global = this_set

        # break_after_first = False  # for testing
        # function = "record_mysql_NULL_booleans_from_set"
        # process_in_batches(min_id, max_id, batch_size, num_threads, process_batch, function, break_after_first)

        # go through this_set in batches
        image_ids_list = list(this_set)
        for i in range(0, len(image_ids_list), batch_size):
            batch_image_ids = set(image_ids_list[i:i + batch_size])
            print(f"Processing batch of size {len(batch_image_ids)}")
            # make this_set a global variable to be used in process_batch
            global image_ids_set_global
            image_ids_set_global = batch_image_ids

            function = "record_mysql_NULL_booleans_from_set"
            process_batch(0, 0, function)  # batch_start and batch_end are not used in this function

    elif MODE == 5:
        # opens all csvs in IMPORT_DIR.
        # These will be helper table output which is a csv. image_id are column index 2
        # which contains all unique image_ids with missing data in mongo
        #
        # global_collection_file = "face_landmarks" 
        # global_collection_file = "face_encodings68" 
        # global_collection_file = "body_landmarks" 
        # global_collection_file = "body_landmarks_norm" 
        global_collection_file = "body_world_landmarks" 
        # global_collection_file = "hand_landmarks" 
        if "hand_landmarks" in global_collection_file:
            global_collection_file = "hand_landmarks"
            file_name = "hand_landmarks_right_hand"
        elif "body_landmarks_norm" in global_collection_file:
            file_name = global_collection_file
        else:
            file_name = global_collection_file
        set_name = "final_sql_only"
        # set_name = "sql_only"
        with open(os.path.join(EXPORT_DIR, set_name, f"{set_name}_{file_name}.txt"), "r") as f:
            first_line = f.readline()  # read header
            if isinstance(first_line, str):
                print(f"skipping header line: {first_line.strip()}")
                first_id = None
            elif first_line.strip().isdigit():
                first_id = int(first_line.strip())
                print(f"first id: {first_id}")
            else:
                print("Unexpected file format, first line:", first_line)

            remaining_lines = f.readlines()  # read the rest of the lines
            missing_mongo_set = set(int(line.strip()) for line in remaining_lines)
            if first_id is not None:
                missing_mongo_set.add(first_id)
            print(f"Loaded set {set_name} for collection {global_collection_file} with {len(missing_mongo_set)} entries")
            print(f"Sample entries: {list(missing_mongo_set)[:10]}")
        # # build a set of all image_ids from all csv files
        # image_ids_list = []
        # csv_files = [f for f in os.listdir(IMPORT_DIR) if f.endswith('.csv')]
        # for csv_file in csv_files:
        #     csv_path = os.path.join(IMPORT_DIR, csv_file)
        #     print(f"Processing CSV file: {csv_path}")
        #     with open(csv_path, 'r') as f:
        #         reader = csv.reader(f)
        #         header = next(reader)  # skip header
        #         for row in reader:
        #             if len(row) > 1 and row[1].isdigit():
        #                 image_id = int(row[1])
        #                 image_ids_list.append(image_id)
        #             elif len(row) == 1 and row[0].isdigit():
        #                 image_id = int(row[0])
        #                 image_ids_list.append(image_id)

        # print(f"Total unique image_ids collected from CSV files: {len(image_ids_list)}")

        # missing_image_ids_by_collection_folder = "/Volumes/OWC4/segment_images/mongo_exports_oct19_sets/sql_only"
        
        # nov 27 haack
        image_ids_list = list(missing_mongo_set)

        for i in range(0, len(image_ids_list), batch_size):
            batch_image_ids = set(image_ids_list[i:i + batch_size])
            print(f"Processing batch of size {len(batch_image_ids)}")
            # make this_set a global variable to be used in process_batch
            image_ids_set_global = batch_image_ids

            function = "check_NMLmongo_for_mysql_only_image_ids"
            process_batch(0, 0, function)  # batch_start and batch_end are not used in this function
    elif MODE == 6:
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
            image_ids_set_global = set(line.strip() for line in remaining_lines)
            if first_id is not None:
                image_ids_set_global.add(first_id)
        print(f"Loaded set {set_name} for collection {global_collection_file} with {len(image_ids_set_global)} entries")

        # the bson files live in the EXPORT_DIR / collection_name folder
        bson_folder = os.path.join(EXPORT_DIR, global_collection_file)
        bson_files = [os.path.join(bson_folder, f) for f in os.listdir(bson_folder) if f.endswith('.bson')]
        print(f"Found {len(bson_files)} bson files in folder {bson_folder}")
        
        # open each bson file, and read the image_id and store that in a set

        for bson_file in bson_files:
            # open the bson_file and access the data
            bson_file_batch_list = exporter.read_bson(bson_file)
            for bson_file_data in bson_file_batch_list:       
                bson_image_id = bson_file_data.get("image_id", None)
                if bson_image_id is not None:
                    bson_image_ids_set.add(str(bson_image_id))
        print(f"Collected {len(bson_image_ids_set)} unique image_ids from BSON files")

        # find the difference between image_ids_set_global and bson_image_ids_set
        missing_in_bson = image_ids_set_global - bson_image_ids_set
        print(f"Found {len(missing_in_bson)} image_ids in global set not in BSON files")

        # write the missing_in_bson to a text file
        still_missing_filename = os.path.join(EXPORT_DIR, f"still_{set_name}_{global_collection_file}.txt")
        with open(still_missing_filename, "w") as f:
            for image_id in missing_in_bson:
                f.write(f"{image_id}\n")
        print(f"Wrote still missing image_ids to {still_missing_filename}")

    # after all potential processing, close session
    session.close()
