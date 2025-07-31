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

IS_SSD = False
VERBOSE = True
BATCH_SIZE = 1000  # Adjust as needed
MODE = 1 # 1 for making image_id list, 2 for exportin bson

io = DataIO(IS_SSD)
db = io.db
EXPORT_DIR = os.path.join(io.ROOT_PROD,"mongo_exports_fromlist")  # Directory to save BSON files
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
        site_name_id = 3
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
        with open("/Volumes/OWC4/salvage/image_ids.txt", "r") as f:
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

    close_mongo()
    print("Export complete.")

if __name__ == "__main__":
    main()