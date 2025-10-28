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

# bboxnormed_collection = mongo_db["body_landmarks_norm"]
# n_phonebbox_collection= mongo_db["bboxnormed_phone"]

SUBSET_LANDMARKS = [i for i in range(13,22)]

image_id = 125829255
#
#
# 10152053

image_id2 = 10145859
# 10145859
# 10146238
# 10576048

# 4618383

# X-33--27_Y-2-2_Z-2-2_ct3200_0013_4618383
def get_landmarks(image_id):

    # for normed bbox version
    # cursor = bboxnormed_collection.find({"image_id": image_id})
    # for doc in cursor: 
    #     nlms = pickle.loads(doc["nlms"])
    #     return nlms

    # for world landmarks version
    cursor = world_body_collection.find({"image_id": image_id})
    for doc in cursor: 
        lms3D = pickle.loads(doc["body_world_landmarks"])
        return lms3D


lms1 = get_landmarks(image_id)
lms2 = get_landmarks(image_id2)


distances = []
print(type(lms1))
print(type(lms2))

    # # Ensure both landmark lists have the same number of landmarks
# if len(lms1.landmark) != len(lms2.landmark):
#     raise ValueError("Both landmark sets must have the same number of landmarks.")

for idx, lm1 in enumerate(lms1.landmark):
    print(f"{image_id} Landmark {idx}: ({lm1.x}, {lm1.y}, {lm1.z}, {lm1.visibility})")
    if idx in SUBSET_LANDMARKS:
        lm2 = lms2.landmark[idx]
        # Calculate the Euclidean distance between the x and y coordinates
        distance = math.sqrt((lm2.x - lm1.x) ** 2 + (lm2.y - lm1.y) ** 2)
        distances.append(distance)
    

print(distances)


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



# def remove_duplicates(collection):
#     pipeline = [
#         # Group by image_id and keep the most recent document
#         {"$sort": {"_id": -1}},  # Sort by _id descending (most recent first)
#         {"$group": {
#             "_id": "$image_id",
#             "most_recent_id": {"$first": "$_id"},
#             "count": {"$sum": 1}
#         }},
#         # Filter only groups with more than one document
#         {"$match": {"count": {"$gt": 1}}}
#     ]

#     duplicates = list(collection.aggregate(pipeline))

#     for doc in duplicates:
#         # Keep the most recent document
#         most_recent_id = doc['most_recent_id']
        
#         # Remove all other documents with the same image_id
#         result = collection.delete_many({
#             "image_id": doc['_id'],
#             "_id": {"$ne": ObjectId(most_recent_id)}
#         })
        
#         print(f"Removed {result.deleted_count} duplicate(s) for image_id: {doc['_id']}")

#     print("Duplicate removal complete.")

# # Usage
# remove_duplicates(bboxnormed_collection)
