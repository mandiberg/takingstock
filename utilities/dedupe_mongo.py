from pymongo import MongoClient
import pymongo

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
mongo_collection = mongo_db[io.dbmongo['collection']]

bboxnormed_collection = mongo_db["bboxnormed_lms"]
# n_phonebbox_collection= mongo_db["bboxnormed_phone"]


# Find duplicates and remove them, keeping the first occurrence
pipeline = [
    {"$group": {"_id": "$image_id", "uniqueIds": {"$addToSet": "$_id"}, "count": {"$sum": 1}}},
    {"$match": {"count": {"$gt": 1}}}
]

duplicates = list(bboxnormed_collection.aggregate(pipeline))

for doc in duplicates:
    # Skip the first document and delete the rest
    ids_to_delete = doc["uniqueIds"][1:]
    print("Deleting", len(ids_to_delete), "duplicate entries for image", doc["_id"])
    # bboxnormed_collection.delete_many({"_id": {"$in": ids_to_delete}})

print("Duplicate entries removed.")
