from pymongo import MongoClient
import pymongo
from bson.objectid import ObjectId

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

bboxnormed_collection = mongo_db["body_landmarks_norm"]
# n_phonebbox_collection= mongo_db["bboxnormed_phone"]


# # Find duplicates and remove them, keeping the first occurrence
# pipeline = [
#     {"$group": {"_id": "$image_id", "uniqueIds": {"$addToSet": "$_id"}, "count": {"$sum": 1}}},
#     {"$match": {"count": {"$gt": 1}}}
# ]

# duplicates = list(bboxnormed_collection.aggregate(pipeline))

# for doc in duplicates:
#     # Skip the first document and delete the rest
#     ids_to_delete = doc["uniqueIds"][1:]
#     print("Deleting", len(ids_to_delete), "duplicate entries for image", doc["_id"])
#     # bboxnormed_collection.delete_many({"_id": {"$in": ids_to_delete}})

# print("Duplicate entries removed.")



def remove_duplicates(collection):
    pipeline = [
        # Group by image_id and keep the most recent document
        {"$sort": {"_id": -1}},  # Sort by _id descending (most recent first)
        {"$group": {
            "_id": "$image_id",
            "most_recent_id": {"$first": "$_id"},
            "count": {"$sum": 1}
        }},
        # Filter only groups with more than one document
        {"$match": {"count": {"$gt": 1}}}
    ]

    duplicates = list(collection.aggregate(pipeline))

    for doc in duplicates:
        # Keep the most recent document
        most_recent_id = doc['most_recent_id']
        
        # Remove all other documents with the same image_id
        result = collection.delete_many({
            "image_id": doc['_id'],
            "_id": {"$ne": ObjectId(most_recent_id)}
        })
        
        print(f"Removed {result.deleted_count} duplicate(s) for image_id: {doc['_id']}")

    print("Duplicate removal complete.")

# Usage
remove_duplicates(bboxnormed_collection)
