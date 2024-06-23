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
from my_declarative_base import Images, Base, Encodings, Clusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

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
mongo_db = mongo_client["test2"]
mongo_collection = mongo_db["encodings3"]

# Define the batch size
batch_size = 100000
start_enc_id = 0

while True:
    try:
        start_enc_id = mongo_collection.find_one(sort=[("encoding_id", -1)])["encoding_id"]
    except Exception as e:
        print(f"An error occurred: {e}")
    print("start_enc_id: ", start_enc_id)
    # Query the Images table for image_id and contentUrl where site_name_id is 1
    results = session.query(Encodings.encoding_id, Encodings.image_id, Encodings.face_landmarks, Encodings.face_encodings68, Encodings.body_landmarks).filter(Encodings.encoding_id > start_enc_id, Encodings.is_face == True).limit(batch_size).all()
    if len(results) == 0:
        break

    for encoding_id, image_id, face_landmarks, face_encodings68, body_landmarks in results:
        # Store data in MongoDB
        if encoding_id and image_id:
            try:
                mongo_collection.insert_one({
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
