from pymongo import MongoClient
import pymongo
from bson.objectid import ObjectId
import pickle
from mediapipe.framework.formats import landmark_pb2
import math

import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from mp_db_io import DataIO

IS_SSD = True
io = DataIO(IS_SSD)
db = io.db
# io.db["name"] = "stock"
# io.db["name"] = "ministock"
mongo_client = pymongo.MongoClient(io.dbmongo['host'])
mongo_db = mongo_client[io.dbmongo['name']]
# mongo_collection = mongo_db[io.dbmongo['collection']]

world_body_collection = mongo_db["body_world_landmarks"]

MONGO_ONLY_IMAGE_ID_FILE = "/Volumes/OWC4/segment_images/mongo_exports_oct19_sets/mongo_only_body_world_landmarks.txt"


def check_mongo_only_file():
    """Read image_id values from MONGO_ONLY_IMAGE_ID_FILE and check world_body_collection
    Print whether each image_id is empty (contains an empty pickle) or not null.
    """
    import os
    if not os.path.exists(MONGO_ONLY_IMAGE_ID_FILE):
        print(f"MONGO_ONLY_IMAGE_ID_FILE not found: {MONGO_ONLY_IMAGE_ID_FILE}")
        return

    not_null_dir = os.path.dirname(MONGO_ONLY_IMAGE_ID_FILE)
    base, ext = os.path.splitext(os.path.basename(MONGO_ONLY_IMAGE_ID_FILE))
    not_null_filename = f"{base}_not_null{ext or '.txt'}"
    not_null_path = os.path.join(not_null_dir, not_null_filename)

    # Open the not-null output file in append mode so we can record valid ids as we go
    with open(not_null_path, "a") as not_null_fh, open(MONGO_ONLY_IMAGE_ID_FILE, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                image_id = int(line)
            except ValueError:
                print(f"Skipping invalid image_id line: {line}")
                continue

            doc = world_body_collection.find_one({"image_id": image_id})
            if not doc:
                # No document found — nothing to keep, ensure we delete any stray docs
                del_result = world_body_collection.delete_many({"image_id": image_id})
                print(f"{image_id} is empty (no doc) — deleted {del_result.deleted_count} documents")
                continue

            data = doc.get("body_world_landmarks")
            if not data:
                # missing or falsy field
                del_result = world_body_collection.delete_many({"image_id": image_id})
                print(f"{image_id} is empty (no field) — deleted {del_result.deleted_count} documents")
                continue

            # attempt to unpickle and inspect
            try:
                obj = pickle.loads(data)
            except Exception as e:
                # If unpickle fails, treat as not null but report the error
                print(f"{image_id} pickle load error: {e}; treating as not null")
                # record as not null
                not_null_fh.write(f"{image_id}\n")
                not_null_fh.flush()
                continue

            # Decide emptiness based on common structures
            is_empty = False
            # protobuf-style object with .landmark
            if obj is None:
                is_empty = True
            elif hasattr(obj, "landmark"):
                try:
                    if len(obj.landmark) == 0:
                        is_empty = True
                except Exception:
                    pass
            elif isinstance(obj, (list, tuple)):
                if len(obj) == 0:
                    is_empty = True
            elif isinstance(obj, bytes):
                if len(obj) == 0:
                    is_empty = True

            if is_empty:
                
                del_result = world_body_collection.delete_many({"image_id": image_id})
                # print(f"{image_id} is empty (no lms) — deleted {del_result.deleted_count} documents")
            else:
                # record non-empty ids to the not-null file
                not_null_fh.write(f"{image_id}\n")
                not_null_fh.flush()
                print(f"{image_id} is not null")


if __name__ == "__main__":
    check_mongo_only_file()
