'''


'''

import os
import bson
import gc
import pymongo
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from my_declarative_base import Encodings, Base, Images
from mp_db_io import DataIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import time
import re
import json

IS_SSD = False
VERBOSE = False
QUIET = True
CHECK_FIRST = True
BATCH_SIZE = 10000  # Adjust as needed
START_BATCH = 0  # #65 for E. For resuming interrupted processes
MODE = 4 # 1 for making image_id list, 2 for exporting bson, 3 for importing bson, 4 for importing from batch files, 5 for importing MySQL
site_name_id = 3
# 3 is adobe, 4 is istock.

io = DataIO(IS_SSD)
db = io.db
if MODE != 5:
    # EXPORT_DIR = os.path.join(io.ROOT_PROD,"mongo_exports_fromlist")  # Directory to save BSON files
    EXPORT_DIR = os.path.join("/Volumes/OWC4/segment_images","mongo_exports_fromlist_istockAB")  # Directory to save BSON files
    # EXPORT_DIR = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/mongo_exports_fromlist_adobeF"  # Directory to save BSON files
    IMAGE_ID_FILE = os.path.join("/Volumes/OWC4/salvage","image_ids.txt")  # File containing image IDs to process
    BATCHES_FOLDER = os.path.join("/Volumes/OWC5/segment_images","mongo_exports")  # Folder containing batch files
    print(f"Export directory: {EXPORT_DIR}")

collection_names = ['encodings', 'body_landmarks_norm', 'hand_landmarks', 'body_landmarks_3D']
document_names_dict = {
    "encodings": ["encoding_id", "face_landmarks", "body_landmarks", "face_encodings68"],
    "body_landmarks_norm": ["nlms"],
    "hand_landmarks": ["left_hand", "right_hand"],
    "body_landmarks_3D": ["body_world_landmarks"]
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

def init_session():
    global engine, session
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
    Session = sessionmaker(bind=engine)
    session = Session()

def close_session():
    session.close()
    engine.dispose()

def init_mongo():
    global mongo_client, mongo_db, mongo_collection, bboxnormed_collection, body_world_collection, mongo_hand_collection
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    mongo_db = mongo_client["stock"]
    mongo_collection = mongo_db["encodings"]
    bboxnormed_collection = mongo_db["body_landmarks_norm"]
    body_world_collection = mongo_db["body_world_landmarks"]
    mongo_hand_collection = mongo_db["hand_landmarks"]

def close_mongo():
    mongo_client.close()

def ensure_export_dir():
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)

def export_bson(collection, query, filename):
    doc = collection.find_one(query)
    if doc:
        with open(filename, "wb") as f:
            f.write(bson.BSON.encode(doc))
        if VERBOSE:
            print(f"Exported {filename}")
    else:
        if VERBOSE:
            print(f"No document found for {query}")

def open_bson(filename):
    if not os.path.exists(filename):
        # if VERBOSE:
        #     print(f"File {filename} does not exist.")
        return None
    with open(filename, "rb") as f:
        try:
            data = list(bson.decode_file_iter(f))
        except Exception as e:
            if VERBOSE:
                print(f"Trying backup bson decode after Error decoding BSON file {filename}: {e}")
            data = bson.BSON.decode(f.read())
    return data

def import_bson(image_id):
    # all_mongo_data = {}
    sql_booleans = {}

    for collection_name, collection in zip(collection_names, collections):
        collection_mongo_data = {}
        collection_do_upsert = False
        data = open_bson(f"{EXPORT_DIR}/{image_id}_{collection_name}.bson")
        if VERBOSE: print(f"Processing {collection_name} for image_id {image_id}")
        # print(data.keys())
        # if collection_name == 'hand_landmarks': print(data)
        for fieldname in document_names_dict[collection_name]:
            if VERBOSE: print(f"Processing {fieldname} for image_id {image_id}")
            if data is not None and fieldname in data.keys():
                if data[fieldname] in (b'\x80\x04N', b'\x80\x04N.'):
                    if VERBOSE: print(f"Found pickled None for {fieldname} in {collection_name} for image_id {image_id}")
                    collection_mongo_data[fieldname] = None
                    if fieldname != "encoding_id":  # encoding_id should not be None
                        sql_booleans[sql_field_names_dict[fieldname]] = False
                else:
                    # if collection_name == 'hand_landmarks': print(data[fieldname])
                    # print(pickle.loads(data[fieldname]))
                    collection_mongo_data[fieldname] = data[fieldname]
                    collection_do_upsert = True
                    if fieldname != "encoding_id":  # encoding_id should not be None
                        sql_booleans[sql_field_names_dict[fieldname]] = True
                if VERBOSE:
                    print(f"Found {fieldname} for image_id {image_id}")
            elif fieldname != "encoding_id":  # skip encoding_id
                # data is None or fieldname not in data
                if VERBOSE: print(f"NO DATA or {fieldname} not found for image_id {image_id}")
                sql_booleans[sql_field_names_dict[fieldname]] = False
                if VERBOSE:
                    print(f"No {fieldname} found for image_id {image_id}")
        # if any(sql_booleans.values()) is True and collection_name == 'hand_landmarks':
        if collection_do_upsert:
            # print(f"Going to upsert {collection_name} with image_id {image_id} mongo_data: {collection_mongo_data.keys()}")
            # upsert collection_mongo_data in collection
            collection.update_one({"image_id": image_id}, {"$set": collection_mongo_data}, upsert=True)

        # handle is_face and is_body booleans
        sql_booleans["is_face"] = sql_booleans.get("mongo_face_landmarks", False)
        sql_booleans["is_body"] = sql_booleans.get("mongo_body_landmarks", False)
        sql_booleans["migrated"] = True
        # originaly thought I would store them, but not using.
        # all_mongo_data[collection_name] = collection_mongo_data

    # print(f"Processed {collection_name} for image_id {image_id}: {sql_booleans}")
    # print(f"Processing {collection_name} for image_id {image_id} mongo_data: {all_mongo_data}")
    return sql_booleans

def build_collection_dict(collection_data):
    """
    Given a list of BSON objects (dicts), return a dict mapping image_id to the object.
    """
    return {item["image_id"]: item for item in collection_data if "image_id" in item}


def load_bson_from_batches(counter):
    # print(f"Loading BSON data from batch {counter}")
    batch_data = {}
    image_ids = set()
    for collection_name in collection_names:
        bson_path = f"{BATCHES_FOLDER}/{collection_name}_batch_{counter}.bson"
        # print(f"Looking for BSON file: {bson_path}")
        data = open_bson(bson_path)
        if data is not None:
            collection_data = build_collection_dict(data)
            batch_data[collection_name] = collection_data
            print(f"Loaded {len(data)} records from {collection_name}_batch_{counter}.bson")
            for image_id in collection_data.keys():
                # print(f"Found image_id {image_id} in {collection_name}")
                image_ids.add(image_id)
        else:
            if VERBOSE: print(f"No data found for {collection_name} in batch {counter}")
    # get all image_ids in this batch
    # for collection_name, data in batch_data.items():
    #     for item in data:
    #         if "image_id" in item:
    #             print(f"Found image_id {item['image_id']} in {collection_name}")
    #             image_ids.add(item["image_id"])

    # print(f"Built list of {len(image_ids)} image IDs from {BATCHES_FOLDER}")
    return image_ids, batch_data

def import_bson_batch(image_ids, batch_data, session):
    # all_mongo_data = {}
    sql_booleans = {}

    for image_id in image_ids:
        # if image_id != 60751883: continue
        this_success = False
        this_sql_booleans = {}
        for collection_name, collection in zip(collection_names, collections):
            collection_mongo_data = {}
            collection_do_upsert = False
            collection_data = batch_data.get(collection_name, [])
            # print(type(collection_data))
            # print(f"collection_data for {collection_name}: {collection_data[0]}")
            data = collection_data.get(image_id)
            # print(f"Importing {collection_name} for image_id {image_id} with data: {data is not None}")
            
            if VERBOSE: print(f"Processing {collection_name} for image_id {image_id}")
            # print(data.keys())
            # if collection_name == 'hand_landmarks': print(data)
            for fieldname in document_names_dict[collection_name]:
                if VERBOSE: print(f"Processing {fieldname} for image_id {image_id}")
                if data is not None and fieldname in data.keys():
                    if data[fieldname] in (b'\x80\x04N', b'\x80\x04N.'):
                        if VERBOSE: print(f"Found pickled None for {fieldname} in {collection_name} for image_id {image_id}")
                        collection_mongo_data[fieldname] = None
                        if fieldname != "encoding_id":  # encoding_id should not be None
                            this_sql_booleans[sql_field_names_dict[fieldname]] = False
                    else:
                        # if collection_name == 'hand_landmarks': print(data[fieldname])
                        # print(pickle.loads(data[fieldname]))
                        collection_mongo_data[fieldname] = data[fieldname]
                        collection_do_upsert = True
                        if fieldname != "encoding_id":  # encoding_id should not be None
                            this_sql_booleans[sql_field_names_dict[fieldname]] = True
                    if VERBOSE:
                        print(f"Found {fieldname} for image_id {image_id}")
                elif fieldname != "encoding_id":  # skip encoding_id
                    # data is None or fieldname not in data
                    if VERBOSE: print(f"NO DATA or {fieldname} not found for image_id {image_id}")
                    this_sql_booleans[sql_field_names_dict[fieldname]] = False
                    if VERBOSE:
                        print(f"No {fieldname} found for image_id {image_id}")
            # if any(this_sql_booleans.values()) is True and collection_name == 'hand_landmarks':
            if collection_do_upsert:
                # print(f"Going to upsert {collection_name} with image_id {image_id} mongo_data: {collection_mongo_data.keys()}")
                # upsert collection_mongo_data in collection
                # print(f"  -----  Before Upserting this_success is {this_success} for image_id {image_id}")
                try:
                    collection.update_one({"image_id": image_id}, {"$set": collection_mongo_data}, upsert=True)
                    if VERBOSE: print(f"  +++++  Upserted {collection_name} for image_id {image_id}:{collection_mongo_data['encoding_id']}")
                    this_success = True
                except Exception as e:
                    if "E11000" not in str(e):
                        # if the error is E11000 duplicate key error collection:
                        print(f" ~~~ unknown error ~~~ {e}")
                    # print(f"  -----  E11000 duplicate key error upserting {collection_name} for image_id {image_id}:{collection_mongo_data['encoding_id']}")

                    # check to see if it exists in the collection and delete. named check_ to avoid overwriting collection in loop
                    for check_collection_name, check_collection in zip(collection_names, collections):
                        # print(f"  -----  Checking image/enc combo in Collection: {collection_name}")
                        # Check if the image_id and encoding_id combo exists in the collection
                        if check_collection.find_one({"image_id": image_id, "encoding_id": collection_mongo_data['encoding_id']}):
                            print(f"  /////  Found matching combo in Collection: {check_collection_name}")
                            # delete the document to test re-upsert
                            check_collection.delete_one({"image_id": image_id, "encoding_id": collection_mongo_data['encoding_id']})
                            print(f"  \\\\\  Deleted matching combo in Collection: {check_collection_name} for re-upsert test")

                    # check to see if image_id exists in mysql encodings table
                    
                    existing_encoding = session.query(Encodings.encoding_id).filter(
                        Encodings.image_id == image_id
                    ).one_or_none()

                    # existing_encoding = session.execute(text("SELECT * FROM encodings WHERE image_id = :image_id")).params(image_id=image_id).fetchone()
                    if existing_encoding:
                        new_encoding_id = existing_encoding[0]
                        # print(f"  >>>>>  Existing encoding_id for image_id {image_id} is {new_encoding_id}")
                    else:
                        # insert image_id and collection_mongo_data['encoding_id'] into mysql encodings table
                        try:
                            enc = Encodings(image_id=image_id)
                            session.add(enc)
                            session.commit()
                            new_encoding_id = enc.encoding_id
                            # print(f"  >>>>>  Inserted image_id {image_id} and NO encoding_id into MySQL, and it now has encoding_id {new_encoding_id}")
                        except Exception as e:
                            print(f"  !!!!!  Error inserting image_id {image_id} and encoding_id {collection_mongo_data['encoding_id']} into MySQL: {e}")
                    try:
                        # print(f"  -----  Trying Re-Upsert of {collection_name} for image_id {image_id} with new encoding_id {new_encoding_id}")
                        collection_mongo_data['encoding_id'] = new_encoding_id
                        # print(f"  -----  collection_mongo_data now has encoding_id {collection_mongo_data['encoding_id']}")
                        collection.update_one({"image_id": image_id}, {"$set": collection_mongo_data},upsert=True)
                        print(f"  +++++  Re-Upserted {collection_name} for image_id {image_id}:{collection_mongo_data['encoding_id']}")
                        this_success = True
                    except Exception as e:
                        print(f"   XXX    Second Error upserting {collection_name} for image_id {image_id}:{collection_mongo_data['encoding_id']}: {e}")
                        this_success = False

                    # print(f"Mongo data: {collection_mongo_data}")

            # handle is_face and is_body booleans
            this_sql_booleans["is_face"] = this_sql_booleans.get("mongo_face_landmarks", False)
            this_sql_booleans["is_body"] = this_sql_booleans.get("mongo_body_landmarks", False)
            this_sql_booleans["migrated_Mongo"] = True
            # originaly thought I would store them, but not using.
            # all_mongo_data[collection_name] = collection_mongo_data
        # print(f"Finished processing image_id {image_id} with success: {this_success}")
        if this_success is True: sql_booleans[image_id] = this_sql_booleans
        # print(len(sql_booleans))
    # print(f"Processed {collection_name} for image_id {image_id}: {sql_booleans}")
    # print(f"Processing {collection_name} for image_id {image_id} mongo_data: {all_mongo_data}")
    return sql_booleans

def export_task(image_id, EXPORT_DIR,):
    export_bson(mongo_collection, {"image_id": image_id}, f"{EXPORT_DIR}/{image_id}_encodings.bson")
    export_bson(bboxnormed_collection, {"image_id": image_id}, f"{EXPORT_DIR}/{image_id}_body_landmarks_norm.bson")
    export_bson(mongo_hand_collection, {"image_id": image_id}, f"{EXPORT_DIR}/{image_id}_hand_landmarks.bson")
    export_bson(body_world_collection, {"image_id": image_id}, f"{EXPORT_DIR}/{image_id}_body_landmarks_3D.bson")
    return image_id



def upsert_sql_data(session, results):
    if not results:
        return
    # print(f"Upserting {len(results)} image_ids in the database.")
    # print(f"Results keys: {results.keys()}")
    # print(f"Results values: {results.values()}")

    # get the column names from results
    all_columns = set()
    for sql_booleans in results.values():
        all_columns.update(sql_booleans.keys())
    all_columns = sorted(all_columns)
    # columns = ['image_id'] + all_columns
    print(f"Columns to upsert: {all_columns}")

    # build the list of dicts for upsert, with image_id included as pkey
    values = []
    for image_id, sql_booleans in results.items():
        row = {'image_id': image_id}
        for col in all_columns:
            row[col] = sql_booleans.get(col, None)
        values.append(row)
    print(f"Example of Values to upsert: {values[0]}...")  # Print first 5 for verification

    stmt = insert(Encodings).values(values)
    update_dict = {col: stmt.inserted[col] for col in all_columns}
    upsert_stmt = stmt.on_duplicate_key_update(**update_dict)
    print(f"Upsert statement prepared for {len(values)} records")
    # print(f"Example upsert statement: {str(upsert_stmt)[:500]}...")  # Print first 500 chars for verification
    session.execute(upsert_stmt)
    session.commit()

def main():
    init_mongo()
    global collections
    collections = [mongo_collection, bboxnormed_collection, mongo_hand_collection, body_world_collection]

    if MODE != 5: ensure_export_dir()
    # Encodings_Migration = io.create_class_from_reflection(engine, 'encodings', 'encodings_migration')

    if MODE == 1:
        init_session()
        # list dir for files in /Volumes/OWC4/salvage/
        salvage_dir = "/Volumes/OWC4/salvage/"
        json_files = [f for f in os.listdir(salvage_dir) if f.endswith(".json")]
        print(f"Found {len(json_files)} JSON files in {salvage_dir}")
        site_image_ids = []
        for json_file in json_files:
            # open each json file
            with open(os.path.join(salvage_dir, json_file), "r") as f:
                site_image_ids_string = f.read()
                # site_image_ids_string is a string that looks like this: ["1000004106.jpg", "1000007876.jpg", "1000025018.jpg", etc
                # Convert the string to a list
                these_site_image_ids = eval(site_image_ids_string)
                site_image_ids.extend(these_site_image_ids)
        # Remove the .jpg from each filename
        site_image_ids = [site_image_id.split('.')[0] for site_image_id in site_image_ids]
        # print(site_image_ids[:10])  # Print first 10 for verification
        # print(f"Processing {(site_image_ids)} image IDs from the JSON file.")
        # query Images in batches of 1000
        results = []
        for i in range(0, len(site_image_ids), 1000):
            print(f"Processing batch {i // 1000 + 1} of {len(site_image_ids) // 1000 + 1}")
            batch = site_image_ids[i:i + 1000]
            results.extend(session.query(Images).filter(
                Images.site_image_id.in_(batch),
                Images.site_name_id == site_name_id
            ).all())
        print(f"Found {len(results)} images in the database matching the site_image_ids.")
        # write results to file
        with open("/Volumes/OWC4/salvage/image_ids.txt", "w") as f:
            for image in results:
                f.write(f"{image.image_id}\n")
        print(f"Exported {len(results)} image IDs to /Volumes/OWC4/salvage/image_ids.txt")
        close_session()

    elif MODE == 2:
        # load image_ids from file
        with open("/Volumes/OWC4/salvage/image_ids-istockAB.txt", "r") as f:
            image_ids = [int(line.strip()) for line in f.readlines()]
        print(f"Processing {len(image_ids)} image IDs from the file.")
        offset = 0
        total = len(image_ids)
        while offset < total:
            batch = image_ids[offset:offset+BATCH_SIZE]
            image_ids_exported = []
            with ThreadPoolExecutor(max_workers=io.NUMBER_OF_PROCESSES) as executor:
                futures = {executor.submit(export_task, image_id, EXPORT_DIR): image_id for image_id in batch}
                for future in as_completed(futures):
                    image_id = future.result()
                    if image_id is not None:
                        image_ids_exported.append(image_id)

            offset += BATCH_SIZE
            gc.collect()
            print(f"Processed batch up to offset {offset}")

    elif MODE == 3:
        init_session()
        # load /Volumes/OWC4/salvage/image_ids-adobe1DONE.txt as list of image IDs
        image_ids = []
        with open("/Volumes/OWC4/salvage/image_ids-istockAB.txt", "r") as f:
            image_ids = [int(line.strip()) for line in f.readlines()]
        print(f"Processing {len(image_ids)} image IDs from the file.")
        if CHECK_FIRST:
            already_migrated = session.query(Encodings.image_id).filter(
                Encodings.migrated.is_(True),
                Encodings.image_id.in_(image_ids)
            ).all()
            already_migrated = {image_id[0] for image_id in already_migrated}
            print(f"Already migrated {len(already_migrated)} image IDs.")
            image_ids = [image_id for image_id in image_ids if image_id not in already_migrated]
            print(f"After filtering, {len(image_ids)} image IDs to process.")

        results = {}

        # results = []
        for i in range(0, len(image_ids), 1000):
            if i < (START_BATCH-1) * 1000:
                continue
            print(f"Processing batch {i // 1000 + 1} of {len(image_ids) // 1000 + 1}")
            batch = image_ids[i:i + 1000]
            start_time = time.time()
            for image_id in batch:
                # query Images in batches of 1000
                this_result = import_bson(image_id)
                if this_result:
                    # print(f"Processed image_id {image_id}: {this_result}")
                    results[image_id] = this_result
                else:
                    print(f"No data found for image_id {image_id}")
                if not QUIET: print(f"Processed image_id {image_id}: mongo_face_landmarks': {this_result['mongo_face_landmarks']}, 'mongo_body_landmarks': {this_result['mongo_body_landmarks']}, 'mongo_encodings': {this_result['mongo_encodings']},  'mongo_body_landmarks_norm': {this_result['mongo_body_landmarks_norm']}, 'mongo_hand_landmarks': {this_result['mongo_hand_landmarks']}, 'mongo_body_landmarks_3D': {this_result['mongo_body_landmarks_3D']}")
            # store booleans in sql
            print(f"Upserting {len(results)} image_ids in the database.")
            upsert_sql_data(session, results)
            results = {}
            end_time = time.time()
            print(f"Batch {i // 1000 + 1} processed in {end_time - start_time:.2f} seconds.")

            
            # results.extend(session.query(Images).filter(
            #     Images.site_image_id.in_(batch),
            #     Images.site_name_id == site_name_id
            # ).all())
        # print(f"Found {len(results)} images in the database matching the image_ids.")
        # write results to file
        # with open("/Volumes/OWC4/salvage/image_ids.txt", "w") as f:
        #     for image in results:
        #         f.write(f"{image.image_id}\n")
        # print(f"Exported {len(results)} image IDs to /Volumes/OWC4/salvage/image_ids.txt")
        close_session()

    elif MODE == 4:
        # import from batch files
        init_session()
        counter = START_BATCH
        while True:
            start_time = time.time()

            # load batch file
            image_ids, batch_data = load_bson_from_batches(counter)
            print(f"Loaded {len(image_ids)} image IDs from batch {counter}")
            for image_id in image_ids:
                for collection_name, collection in zip(collection_names, collections):
                    collection_mongo_data = {}
                    collection_do_upsert = False
                    collection_data = batch_data.get(collection_name, [])
                    # print(type(collection_data))
                    # print(f"collection_data for {collection_name}: {collection_data[0]}")
                    data = collection_data.get(image_id)
                    print(f"Importing {collection_name} for image_id {image_id} with data: {data}")
                break

            # if CHECK_FIRST:
            #     already_migrated = session.query(Encodings.image_id).filter(
            #         Encodings.migrated_Mongo.is_(True),
            #         Encodings.image_id.in_(image_ids)
            #     ).all()
            #     already_migrated = {image_id[0] for image_id in already_migrated}
            #     print(f"Already migrated {len(already_migrated)} image IDs.")
            #     image_ids = [image_id for image_id in image_ids if image_id not in already_migrated]
            #     print(f"After filtering, {len(image_ids)} image IDs to process.")

            # sql_booleans = import_bson_batch(image_ids, batch_data, session)
            # print(f"Processed batch {counter} with {len(sql_booleans)} sql_booleans, a dict of dicts.")
            # # print(sql_booleans)


            # # store booleans in sql
            # print(f"Upserting {len(sql_booleans)} image_ids in the database.")
            # upsert_sql_data(session, sql_booleans)
            end_time = time.time()
            print(f"Batch {counter} processed in {end_time - start_time:.2f} seconds.")
            print("- ")
            print("- ")

            break # temporary for testing
            counter += BATCH_SIZE
    elif MODE == 5:
        # import SQL data
        init_session()
        counter = 0
        start_time = time.time()
        sql_table_dump_file = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/ultra_sql/Encodings_Migration_Ultra.sql"
        '''
        # this is the SQL to create the Encodings_Migration table:        
        CREATE TABLE `Encodings_Migration` (
        `migration_id` int NOT NULL AUTO_INCREMENT,
        `encoding_id` int DEFAULT NULL,
        `image_id` int DEFAULT NULL,
        `is_face` tinyint(1) DEFAULT NULL,
        `is_body` tinyint(1) DEFAULT NULL,
        `is_face_distant` tinyint(1) DEFAULT NULL,
        `face_x` decimal(6,3) DEFAULT NULL,
        `face_y` decimal(6,3) DEFAULT NULL,
        `face_z` decimal(6,3) DEFAULT NULL,
        `mouth_gap` decimal(6,3) DEFAULT NULL,
        `face_landmarks` blob,
        `bbox` json DEFAULT NULL,
        `face_encodings68` blob,
        `body_landmarks` blob,
        `mongo_encodings` tinyint(1) DEFAULT NULL,
        `mongo_body_landmarks` tinyint(1) DEFAULT NULL,
        `mongo_face_landmarks` tinyint(1) DEFAULT NULL,
        `is_small` tinyint(1) DEFAULT NULL,
        `mongo_body_landmarks_norm` tinyint(1) DEFAULT NULL,
        `two_noses` tinyint(1) DEFAULT NULL,
        `is_dupe_of` int DEFAULT NULL,
        `mongo_hand_landmarks` tinyint(1) DEFAULT NULL,
        `mongo_hand_landmarks_norm` tinyint(1) DEFAULT NULL,
        `is_face_no_lms` tinyint(1) DEFAULT NULL,
        `is_feet` tinyint(1) DEFAULT NULL,
        `mongo_body_landmarks_3D` tinyint(1) DEFAULT NULL,
        `is_hand_left` tinyint(1) DEFAULT NULL,
        `is_hand_right` tinyint(1) DEFAULT NULL,
        `migrated_SQL` tinyint(1) DEFAULT NULL,
        `migrated_Mongo` tinyint(1) DEFAULT NULL,
        PRIMARY KEY (`migration_id`),
        UNIQUE KEY `image_id` (`image_id`),
        UNIQUE KEY `encoding_id` (`encoding_id`),
        KEY `is_dupe_of` (`is_dupe_of`),
        CONSTRAINT `encodings_migration_ibfk_1` FOREIGN KEY (`encoding_id`) REFERENCES `Encodings` (`encoding_id`),
        CONSTRAINT `encodings_migration_ibfk_2` FOREIGN KEY (`image_id`) REFERENCES `Images` (`image_id`),
        CONSTRAINT `encodings_migration_ibfk_3` FOREIGN KEY (`is_dupe_of`) REFERENCES `Images` (`image_id`)
        ) ENGINE=InnoDB AUTO_INCREMENT=22988877 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
        
        
        '''

        def extract_json_from_string(bbox_match):
            # Look for JSON pattern: starts with '{' and ends with '}'
            bbox_json_string = bbox_match.group(1)
            
            # Method 2: Using eval (be careful with this approach)
            try:
                # This works because the outer structure is a Python string literal
                unescaped = eval(bbox_json_string)
                json_obj2 = json.loads(unescaped)
                # print(f"Parsed JSON object: {json_obj2}")
                return json_obj2
            except Exception as e:
                print(f"Error with eval method: {e}")
                return None


        with open(sql_table_dump_file, "r") as f:
            # only read lines that start with INSERT INTO `Encodings_Migration` VALUES
            lines = [line for line in f.readlines() if line.startswith("INSERT INTO")]
            for line in lines:
                # imaged_ids holds the ids for each line (about 5000)
                image_ids = []
                row_dict = {}
                item_counter = 0
                # split line into individual value tuples
                tuples = line.split("VALUES")[1].strip().rstrip(";").split("),(")

                tuples[0] = tuples[0].lstrip("(")
                tuples[-1] = tuples[-1].rstrip(")")
                # print first tuple for verification
                # print(f"First tuple: {tuples[0]}")

                # for each tuple/row, build a dict with image_id as key
                for t in tuples:

                    # this is what a line tuple looks like:
                    # First tuple: 1,77108103,68354647,1,1,NULL,-26.314,-31.326,12.182,7.932,NULL,'"{\n    \"left\": 957,\n    \"right\": 1239,\n    \"top\": 424,\n    \"bottom\": 706\n}"',NULL,NULL,1,1,1,NULL,1,NULL,NULL,NULL,NULL,NULL,0,1,NULL,NULL,NULL,0
                    bbox_json = None
                    # Look for bbox JSON pattern: starts with '"{ and ends with }"'
                    # bbox_match = re.search(r"(\'\\\"{.*?}\\\"\')", t)
                    bbox_match = re.search(r"(\'\\\"{.*?}\\\"\')", t)
                    # print(f"bbox_match: {bbox_match}")
                    if bbox_match:
                        # find the bad one and get rid of it
                        bbox_json = bbox_match.group(0)
                        # print(f"Found bbox_json: {bbox_json}")
                        # Replace bbox_json in t with a placeholder
                        t_clean = t.replace(bbox_json, '__BBOX__')
                        # print(f"t_clean: {t_clean}")
                        values = t_clean.split(",")

                        # this is the good one
                        loaded_bbox_json = extract_json_from_string(bbox_match)
                        # print(f"Loaded bbox_json: {loaded_bbox_json}")

                        # Place bbox_json back into the correct position (index 11)
                        # values[11] = loaded_bbox_json

                        # print(f"Values with bbox_json restored: {values} and values[11]: {values[11]}")
                    else:
                        values = t.split(",")


                    # First tuple: 1,77108103,68354647,1,1,NULL,-26.314,-31.326,12.182,7.932,NULL,'\"{\\n    \\\"left\\\": 957,\\n    \\\"right\\\": 1239,\\n    \\\"top\\\": 424,\\n    \\\"bottom\\\": 706\\n}\"',NULL,NULL,1,1,1,NULL,1,NULL,NULL,NULL,NULL,NULL,0,1,NULL,NULL,NULL,0
                    # values = t.split(",")
                    # print(f"Values before building: {values}")
                    if len(values) > 1:
                        try:
                            def parse_int(val):
                                # print(f"Parsing int from value: {val}")
                                return int(val) if val != 'NULL' else None
                            def parse_float(val):
                                # print(f"Parsing float from value: {val}")
                                return float(val) if val != 'NULL' else None
                            def parse_json(val):
                                print(f"Parsing json from value: {val}")
                                if val == 'NULL':
                                    return None
                                # Remove outer single and double quotes if present
                                if val.startswith("'") and val.endswith("'"):
                                    val = val[1:-1]
                                # Unescape all \" to "
                                val = val.replace('\\"', '"')
                                val = val.replace('\"', '"')
                                val = val.replace('\"', '"')
                                # Unescape
                                val = val.encode('utf-8').decode('unicode_escape')
                                print(f"unloaded json string: {val}")
                                jsonval = json.loads(val)
                                print(f"Parsed json: {jsonval}")
                                return jsonval                            
                            
                            # if any of values 5,26,27 is 1 then print all of those values with their lables
                            if values[5] == '1':
                                print(f" ~~ 'is_face_distant': {parse_int(values[5])}")

                            image_id = parse_int(values[2])
                            # print(f"Processing image_id: {image_id}")
                            row_dict[image_id] = {
                                # 'migration_id': parse_int(values[0]),
                                # 'encoding_id': parse_int(values[1]),
                                'image_id': parse_int(values[2]),
                                'is_face': parse_int(values[3]),
                                'is_body': parse_int(values[4]),
                                'is_face_distant': parse_int(values[5]),
                                'face_x': parse_float(values[6]),
                                'face_y': parse_float(values[7]),
                                'face_z': parse_float(values[8]),
                                'mouth_gap': parse_float(values[9]),
                                'bbox': loaded_bbox_json,
                                'mongo_encodings': parse_int(values[14]),
                                'mongo_body_landmarks': parse_int(values[15]),
                                'mongo_face_landmarks': parse_int(values[16]),
                                'is_small': parse_int(values[17]),
                                'mongo_body_landmarks_norm': parse_int(values[18]),
                                'two_noses': parse_int(values[19]),
                                'is_dupe_of': parse_int(values[20]),
                                'mongo_hand_landmarks': parse_int(values[21]),
                                'mongo_hand_landmarks_norm': parse_int(values[22]),
                                'is_face_no_lms': parse_int(values[23]),
                                'is_feet': parse_int(values[24]),
                                'mongo_body_landmarks_3D': parse_int(values[25]),
                                'is_hand_left': parse_int(values[26]),
                                'is_hand_right': parse_int(values[27]),
                                'migrated_SQL': 1,
                            }
                        except ValueError:
                            print(f"Could not build dict for image_id: {values[1]}")
                    item_counter += 1
                    # if item_counter % 100 == 0:
                    #     print(f"  Processed {item_counter} tuples so far...")
                    # if item_counter >= 4:
                    #     break # temporary for testing
                # image_ids.extend(list(row_dict.keys()))
                # print(row_dict)
                image_ids = list(row_dict.keys())
                print(f"Processed {len(tuples)} tuples, to make a {len(row_dict)} item row dict which matches total image_ids: {len(image_ids)}")

                def remove_list_from_dict(to_remove, row_dict):
                    for to_remove_id in to_remove:
                        if to_remove_id in row_dict:
                            del row_dict[to_remove_id]
                    return row_dict
                if CHECK_FIRST:
                    # remove already migrated from row_dict
                    already_migrated = session.query(Encodings.image_id).filter(
                        Encodings.migrated_SQL.is_(True),
                        Encodings.image_id.in_(image_ids)
                    ).all()
                    already_migrated = {image_id[0] for image_id in already_migrated}
                    row_dict = remove_list_from_dict(already_migrated, row_dict)
                    print(f"Removed {len(already_migrated)} Already migrated, to make a {len(row_dict)} item row dict")

                    # only site 3,4 in rowdict
                    site_image_ids_in_dict = session.query(Images.image_id).filter(
                        Images.site_name_id.in_([3,4]),
                        Images.image_id.in_(list(row_dict.keys()))
                    ).all()
                    site_image_ids_in_dict = {image_id[0] for image_id in site_image_ids_in_dict}
                    # print(f"site_image_ids_in_dict has {len(site_image_ids_in_dict)} image IDs.")
                    # removce any image_ids not in site 3,4 from row_dict (passing through set difference)
                    row_dict = remove_list_from_dict(set(row_dict.keys()) - site_image_ids_in_dict, row_dict)
                    print(f"After filtering to site 3,4, row_dict has {len(row_dict)} image IDs.")

                # store booleans in sql
                # print(f"Upserting {len(sql_booleans)} image_ids in the database.")
                upsert_sql_data(session, row_dict)
                end_time = time.time()
                print(f"Batch {counter} processed in {end_time - start_time:.2f} seconds.")
                print("- ")
                print("- ")

                # break # temporary for testing
                counter += 1
            
    close_session()
    close_mongo()
    print("Export complete.")

if __name__ == "__main__":
    main()