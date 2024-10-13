import os
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
import pymongo
from pymongo.errors import DuplicateKeyError

# importing from another folder
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from mp_db_io import DataIO
from my_declarative_base import Images, Base, SegmentTable, Encodings, Clusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

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
mongo_collection = mongo_db["encodings"]
mongo_collection_destination = mongo_db["encodings_segment"]

# Define the batch size
batch_size = 10
try:
    start_enc_id = mongo_collection_destination.find_one(sort=[("image_id", -1)])["image_id"]
except Exception as e:
    print(f"An error occurred: {e}")
    start_enc_id = 0

# last_id = 4262137
# final_id = 4287381
# this last configuration is for a set of getty where I had body but not face. 
# i did this after doign a full run of face, so I had to cludge some limits in if/else start_enc_id
print("start_enc_id: ", start_enc_id)
while True:
    try:
        if last_id == 0:
            start_enc_id = 0
        else:
            start_enc_id = last_id
    except Exception as e:
        print(f"An error occurred: {e}")
    print("start_enc_id: ", start_enc_id)
    
    # # this was for moving everything over
    # results = session.query(Encodings.encoding_id, Encodings.image_id, Encodings.face_landmarks, Encodings.face_encodings68, Encodings.body_landmarks).\
    #     filter(Encodings.encoding_id > start_enc_id, Encodings.encoding_id <= final_id, Encodings.is_face == 0, Encodings.is_body == 1).\
    #     limit(batch_size).all()

    # this is for making a segment
    results = session.query(SegmentTable.image_id).\
        filter(SegmentTable.image_id > start_enc_id).\
        limit(batch_size).all()
    print("results: ", results)

    if len(results) == 0:
        break

    for result in results:
        image_id = result[0]
        last_id = image_id

        results = mongo_collection.find_one({"image_id": image_id})
        if results:
            encoding_id = results.get('encoding_id', None)
            face_encodings68 = results.get('face_encodings68', None)
            face_landmarks = results.get('face_landmarks', None)
            body_landmarks = results.get('body_landmarks', None)


        # print(encoding_id, image_id, face_landmarks, face_encodings68, body_landmarks)
        # Store data in MongoDB
        if encoding_id and image_id and face_encodings68:
            try:
                mongo_collection_destination.insert_one({
                    "encoding_id": encoding_id,
                    "image_id": image_id,
                    "face_landmarks": face_landmarks,
                    "body_landmarks": body_landmarks,
                    "face_encodings68": face_encodings68
                })
            except DuplicateKeyError as e:
                print(f"Duplicate key error for encoding_id: {encoding_id}, image_id: {image_id}")
                print(f"Error details: {e}")
                continue  # Move to the next iteration of the loop
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue  # Move to the next iteration of the loop
    print(f"Completed encoding_id {encoding_id} - image_id {image_id}.")
    
print("All changes committed.")
mongo_client.close()    

session.close()
