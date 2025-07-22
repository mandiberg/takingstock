'''
This script exports mongo documents to bson files, so they can be imported into the other shard

There are four collections that need to be exported.
The documents include some integers (image_id, encoding_id) alongside pickled data.
The start of the picked data looks like this: Binary.createFromBase64('gASVNwMAAAAAAACMKG1lZGlhcGlwZS5mcmFtZXdvcmsuZm9ybWF0cy5sYW5kbWFya19wYjKUjBZOb3JtYWxpemVkTGFuZG1hcmtM…', 0)
The hand_landmarks collection has a different structure -- it uses proper nested JSON instead of pickled data.

Here are the steps to Export Mongo:
    select image_id and all of the booleans listed below from Encodings_Migration where ANY (is_body, is_face, mongo_hand_landmarks) is true and migrated_Mongo is None
    export the bson data from each collection for those image_ids if the mongo booleans are true. These are the booleans:
        mongo_encodings, mongo_body_landmarks, mongo_face_landmarks correspond to mongo_collection 
        mongo_body_landmarks_norm correspond to bboxnormed_collection 
        mongo_hand_landmarks, mongo_hand_landmarks_norm correspond to mongo_hand_collection
        mongo_body_landmarks_3D correspond to body_world_collection
    set "migrated_Mongo" boolean == 0 after exporting the bson data (this avoids re-exporting the same data, and tells us what needs to be reimported in the new shard)

The number of results will be in the 10Ms, so this will be done in batches.
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
from my_declarative_base import Encodings, Base
from mp_db_io import DataIO

IS_SSD = False
VERBOSE = True
BATCH_SIZE = 1000  # Adjust as needed

io = DataIO(IS_SSD)
db = io.db
EXPORT_DIR = os.path.join(io.ROOT_PROD,"mongo_exports")  # Directory to save BSON files
print(f"Export directory: {EXPORT_DIR}")

def init_session():
    global engine, Session, session
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

def main():
    init_session()
    init_mongo()
    ensure_export_dir()
    Encodings_Migration = io.create_class_from_reflection(engine, 'encodings', 'encodings_migration')

    offset = 0
    while True:
        # Step 1: Query for migration candidates
        results = session.query(
            Encodings_Migration.image_id,
            Encodings_Migration.encoding_id,
            Encodings_Migration.mongo_encodings,
            Encodings_Migration.mongo_body_landmarks,
            Encodings_Migration.mongo_face_landmarks,
            Encodings_Migration.mongo_body_landmarks_norm,
            Encodings_Migration.mongo_hand_landmarks,
            Encodings_Migration.mongo_hand_landmarks_norm,
            Encodings_Migration.mongo_body_landmarks_3D,
            Encodings_Migration.migrated_Mongo
        ).filter(
            (Encodings_Migration.is_body == True) |
            (Encodings_Migration.is_face == True) |
            (Encodings_Migration.mongo_hand_landmarks == True),
            Encodings_Migration.migrated_Mongo.is_(None)
        ).offset(offset).limit(BATCH_SIZE).all()

        if not results:
            print("No (more) records to process.")
            break

        for row in results:
            # print(row)
            image_id = row.image_id
            encoding_id = row.encoding_id

            # Step 2: Export BSON from MongoDB collections
            # encodings, body_landmarks, face_landmarks
            if any([row.mongo_encodings, row.mongo_body_landmarks, row.mongo_face_landmarks]):
                export_bson(mongo_collection, {"image_id": image_id}, f"{EXPORT_DIR}/{image_id}_encodings.bson")
            if row.mongo_body_landmarks_norm:
                export_bson(bboxnormed_collection, {"image_id": image_id}, f"{EXPORT_DIR}/{image_id}_body_landmarks_norm.bson")
            if any([row.mongo_hand_landmarks, row.mongo_hand_landmarks_norm]):
                export_bson(mongo_hand_collection, {"image_id": image_id}, f"{EXPORT_DIR}/{image_id}_hand_landmarks.bson")
            if row.mongo_body_landmarks_3D:
                export_bson(body_world_collection, {"image_id": image_id}, f"{EXPORT_DIR}/{image_id}_body_landmarks_3D.bson")

            # Step 3: Update migrated_Mongo status
            session.query(Encodings_Migration).filter(
                Encodings_Migration.image_id == image_id
            ).update({"migrated_Mongo": 0})

        session.commit()
        offset += BATCH_SIZE
        gc.collect()
        if VERBOSE:
            print(f"Processed batch up to offset {offset}")

    close_session()
    close_mongo()
    print("Export complete.")

if __name__ == "__main__":
    main()