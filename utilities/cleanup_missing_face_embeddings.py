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
SITE_NAME_ID = 15  # Adjust as needed, e.g., 16 for nappy

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


def do_task(image_id):
    # Create a new session for this thread
    Session = sessionmaker(bind=engine)
    thread_session = Session()
    print(f"Processing image_id: {image_id}")
    query = {"image_id": image_id}
    doc = mongo_collection.find_one(query)
    if doc:
        if "face_encodings68" in doc.keys():
            face = doc["face_encodings68"]
            print(f"{image_id}: {len(face)}")
        else:
            print(f"setting is_face, mongo_encodings, mongo_face_landmarks, to None in Encodings for image_id {image_id}")
            # this is commented out for testing. uncomment to actually reset the fields
            encodings = thread_session.query(Encodings).filter(Encodings.image_id == image_id).first()
            if encodings:
                encodings.is_face = None
                encodings.mongo_encodings = None
                encodings.mongo_face_landmarks = None
                thread_session.commit()
                print(f"Updated Encodings for image_id {image_id}")
            else:
                print(f"No Encodings found for image_id {image_id}")
    thread_session.close()



def main():
    init_mongo()
    # Encodings_Migration = io.create_class_from_reflection(engine, 'encodings', 'encodings_migration')
    init_session()

    # query mysql for image_ids where encodings.is_face = 1 and encodings.is_body = 1 and images.site_name_id = SITE_NAME_ID
    results = (
        session.query(Encodings.image_id)
        .join(Images, Encodings.image_id == Images.image_id)
        .filter(
            Encodings.is_face.is_(True),
            Encodings.is_body.is_(True),
            Images.site_name_id == SITE_NAME_ID
        )
        .order_by(Encodings.image_id)
    )
    image_ids = [row.image_id for row in results]
    total = len(image_ids)
    offset = 0
    # load image_ids from file
    while offset < total:
        batch = image_ids[offset:offset+BATCH_SIZE]
        image_ids_exported = []
        with ThreadPoolExecutor(max_workers=io.NUMBER_OF_PROCESSES) as executor:
            futures = {executor.submit(do_task, image_id): image_id for image_id in batch}
            for future in as_completed(futures):
                image_id = future.result()
                if image_id is not None:
                    image_ids_exported.append(image_id)

        offset += BATCH_SIZE
        gc.collect()
        print(f"Processed batch up to offset {offset}")
    close_session()

    close_mongo()
    print(" complete.")

if __name__ == "__main__":
    main()