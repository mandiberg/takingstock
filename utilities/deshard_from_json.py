'''


'''

import os
import bson
import gc
import pymongo
from sqlalchemy import create_engine, MetaData, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from my_declarative_base import Encodings, Base, Images
from mp_db_io import DataIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

IS_SSD = False
VERBOSE = True
BATCH_SIZE = 1000  # Adjust as needed
MODE = 2 # 1 for making image_id list, 2 for exporting bson, 3 for importing bson
site_name_id = 3

io = DataIO(IS_SSD)
db = io.db
# EXPORT_DIR = os.path.join(io.ROOT_PROD,"mongo_exports_fromlist")  # Directory to save BSON files
EXPORT_DIR = os.path.join("/Volumes/OWC4/segment_images","mongo_exports_fromlist_istockAB")  # Directory to save BSON files
print(f"Export directory: {EXPORT_DIR}")

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
        data = bson.BSON.decode(f.read())
    return data

def import_bson(image_id):
    all_mongo_data = {}
    sql_booleans = {}
    collections = [mongo_collection, bboxnormed_collection, mongo_hand_collection, body_world_collection]
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


    for collection_name, collection in zip(collection_names, collections):
        collection_mongo_data = {}
        data = open_bson(f"{EXPORT_DIR}/{image_id}_{collection_name}.bson")
        # print(f"Processing {collection_name} for image_id {image_id}")
        # print(data.keys())
        # if collection_name == 'hand_landmarks': print(data)
        for fieldname in document_names_dict[collection_name]:
            # print(f"Processing {fieldname} for image_id {image_id}")
            if data is not None and fieldname in data.keys():
                if data[fieldname] in (b'\x80\x04N', b'\x80\x04N.'):
                    # print(f"Found pickled None for {fieldname} in {collection_name} for image_id {image_id}")
                    collection_mongo_data[fieldname] = None
                    if fieldname != "encoding_id":  # encoding_id should not be None
                        sql_booleans[sql_field_names_dict[fieldname]] = False
                else:
                    # if collection_name == 'hand_landmarks': print(data[fieldname])
                    # print(pickle.loads(data[fieldname]))
                    collection_mongo_data[fieldname] = data[fieldname]
                    if fieldname != "encoding_id":  # encoding_id should not be None
                        sql_booleans[sql_field_names_dict[fieldname]] = True
                # if VERBOSE:
                #     print(f"Found {fieldname} for image_id {image_id}")
            else:
                sql_booleans[sql_field_names_dict[fieldname]] = False
                # if VERBOSE:
                #     print(f"No {fieldname} found for image_id {image_id}")
        if any(sql_booleans.values()) is True and collection_name == 'hand_landmarks':
            print(f"Going to upsert {collection_name} with image_id {image_id} mongo_data: {collection_mongo_data}")
            # upsert collection_mongo_data in collection
            collection.update_one({"image_id": image_id}, {"$set": collection_mongo_data}, upsert=True)

        all_mongo_data[collection_name] = collection_mongo_data

    print(f"Processed {collection_name} for image_id {image_id}: {sql_booleans}")
    # print(f"Processing {collection_name} for image_id {image_id} mongo_data: {all_mongo_data}")
    return sql_booleans

def export_task(image_id, EXPORT_DIR,):
    export_bson(mongo_collection, {"image_id": image_id}, f"{EXPORT_DIR}/{image_id}_encodings.bson")
    export_bson(bboxnormed_collection, {"image_id": image_id}, f"{EXPORT_DIR}/{image_id}_body_landmarks_norm.bson")
    export_bson(mongo_hand_collection, {"image_id": image_id}, f"{EXPORT_DIR}/{image_id}_hand_landmarks.bson")
    export_bson(body_world_collection, {"image_id": image_id}, f"{EXPORT_DIR}/{image_id}_body_landmarks_3D.bson")
    return image_id

def main():
    init_mongo()
    ensure_export_dir()
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
        with open("/Volumes/OWC4/salvage/image_ids-adobe1DONEhead.txt", "r") as f:
            image_ids = [int(line.strip()) for line in f.readlines()]
        print(f"Processing {len(image_ids)} image IDs from the file.")


        # results = []
        for i in range(0, len(image_ids), 1000):
            print(f"Processing batch {i // 1000 + 1} of {len(image_ids) // 1000 + 1}")
            batch = image_ids[i:i + 1000]
            for image_id in batch:
                # query Images in batches of 1000
                results = import_bson(image_id)
                # print(f"Processed image_id {image_id}: {results}")
            # store booleans in sql

            
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



    close_mongo()
    print("Export complete.")

if __name__ == "__main__":
    main()