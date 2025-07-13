import os
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
from pathlib import Path


# importing from another folder
import sys
ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/facemap/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)
from mp_db_io import DataIO
from my_declarative_base import Images, SegmentTable, Encodings, SegmentBig, Base, Clusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON
import pymongo
from pymongo.errors import DuplicateKeyError

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
# mongo_collection = mongo_db["encodings"]
mongo_collection = mongo_db["body_landmarks_norm"]

# Define the batch size
batch_size = 1000
last_id = 0
# TARGET = "tokens"
# TARGET = "encodings"
# TARGET = "segment"
TARGET = "mongo_body_landmarks_norm"
# currently set up for SegmentTable. need to change SegmentTable to Images if you want to use on main table

if TARGET == "mongo_body_landmarks_norm":
    # select all the image_ids from the mongo collection
    mongo_image_ids = [x["image_id"] for x in mongo_collection.find()]
    # assign mongo_body_landmarks_norm to 1 for all the image_ids in the mysql database
    batch_size = 1000
    for i in range(0, len(mongo_image_ids), batch_size):
        batch_ids = mongo_image_ids[i:i+batch_size]
        session.query(Encodings).filter(Encodings.image_id.in_(batch_ids)).update({"mongo_body_landmarks_norm": 1})
        session.query(SegmentTable).filter(SegmentTable.image_id.in_(batch_ids)).update({"mongo_body_landmarks_norm": 1})
        session.commit()
        print(f"Processed {i} of {len(mongo_image_ids)}")
    session.close()
    exit()

while True:
    # try:
    #     start_enc_id = mongo_collection.find_one(sort=[("encoding_id", -1)])["encoding_id"]
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    print("start_id: ", last_id)
    # Query the Images table for image_id and contentUrl where site_name_id is 1
    # results = session.query(Encodings.encoding_id, Encodings.image_id, Encodings.face_landmarks, Encodings.face_encodings68, Encodings.body_landmarks).filter(Encodings.encoding_id > start_enc_id, Encodings.is_face == True).limit(batch_size).all()


    try:
        # Query the Images table for image_id and contentUrl where site_name_id is 1
        # results = session.query(SegmentBig.seg_image_id).filter(SegmentBig.tokenized_keyword_list is not None, SegmentBig.seg_image_id > last_id).limit(batch_size).all()
        # SegmentBig.mongo_tokens is None,
        if TARGET == "tokens":
            results = session.query(SegmentBig.seg_image_id, SegmentBig.image_id).\
                filter(SegmentBig.tokenized_keyword_list.isnot(None), SegmentBig.mongo_tokens.is_(None), SegmentBig.seg_image_id > last_id).\
                limit(batch_size).all()
        elif TARGET == "encodings":
            # results = session.query(Encodings.encoding_id, Encodings.image_id).\
            #     filter(
            #         sqlalchemy.or_(
            #             Encodings.face_landmarks.isnot(None),
            #             Encodings.body_landmarks.isnot(None),
            #             Encodings.face_encodings68.isnot(None)
            #         ),
            #         Encodings.mongo_encodings.is_(None),
            #         Encodings.encoding_id > last_id
            #     ).\
            #     limit(batch_size).all()
            results = session.query(Encodings.encoding_id, Encodings.image_id).\
                filter(
                    Encodings.mongo_encodings == 1,
                    Encodings.mongo_body_landmarks.is_(None),
                    Encodings.mongo_face_landmarks.is_(None),
                    Encodings.encoding_id > last_id
                ).\
                limit(batch_size).all()

        elif TARGET == "segment":
            # results = session.query(SegmentTable.seg_image_id, SegmentTable.image_id).\
            #     filter(
            #         sqlalchemy.or_(
            #             SegmentTable.face_landmarks.isnot(None),
            #             SegmentTable.body_landmarks.isnot(None),
            #             SegmentTable.face_encodings68.isnot(None),
            #             SegmentTable.keyword_list.isnot(None),
            #             SegmentTable.tokenized_keyword_list.isnot(None)
            #         ),
            #         SegmentTable.mongo_tokens.is_(None),
            #         SegmentTable.seg_image_id > last_id
            #     ).\
            #     limit(batch_size).all()
            results = session.query(SegmentTable.seg_image_id, SegmentTable.image_id).\
                filter(
                    SegmentTable.mongo_tokens == 1,
                    SegmentTable.mongo_body_landmarks.is_(None),
                    SegmentTable.mongo_face_landmarks.is_(None),
                    SegmentTable.seg_image_id > last_id
                ).\
                limit(batch_size).all()

        if len(results) == 0:
            print("No more results found.")
            break

        # Initialize counters
        # total_processed = 0
        # current_batch = []

        # for result in results:
        #     print("result", result)



        if TARGET == "tokens":
            session.bulk_update_mappings(SegmentBig, [{"seg_image_id": seg_image_id, "image_id": image_id, "tokenized_keyword_list": None, "mongo_tokens": True} for seg_image_id, image_id in results])
        elif TARGET == "encodings":
            # session.bulk_update_mappings(Encodings, [{"encoding_id": encoding_id, "image_id": image_id, "face_landmarks": None, "body_landmarks": None,"face_encodings68": None,  "mongo_encodings": True} for encoding_id, image_id in results])
            image_ids = [image_id for encoding_id, image_id in results]
            mongo_results = mongo_collection.find({"image_id": {"$in": image_ids}})

            # can't do a bulk update because I don't have the seg_image_id
            for mongo_result in mongo_results:
                # for each mongo_result, update the corresponding row in the mysql database
                session.query(Encodings).filter(Encodings.image_id == mongo_result["image_id"]).update({
                    "mongo_face_landmarks": 1 if mongo_result["face_landmarks"] is not None else Encodings.mongo_face_landmarks,
                    "mongo_body_landmarks": 1 if mongo_result["body_landmarks"] is not None else Encodings.mongo_body_landmarks
                })
                # print the values inserted


        elif TARGET == "segment":
            # session.bulk_update_mappings(SegmentTable, [{"seg_image_id": seg_image_id, "image_id": image_id, "face_landmarks": None, "body_landmarks": None,"face_encodings68": None, "tokenized_keyword_list": None, "keyword_list": None, "mongo_tokens": True} for seg_image_id, image_id in results])
            image_ids = [image_id for seg_image_id, image_id in results]
            mongo_results = mongo_collection.find({"image_id": {"$in": image_ids}})

            # can't do a bulk update because I don't have the seg_image_id
            for mongo_result in mongo_results:
                # for each mongo_result, update the corresponding row in the mysql database
                session.query(SegmentTable).filter(SegmentTable.image_id == mongo_result["image_id"]).update({
                    "mongo_face_landmarks": 1 if mongo_result["face_landmarks"] is not None else SegmentTable.mongo_face_landmarks,
                    "mongo_body_landmarks": 1 if mongo_result["body_landmarks"] is not None else SegmentTable.mongo_body_landmarks
                })
                # print the values inserted


                # print("mongo_result", mongo_result)
        session.commit()
        # current_batch = []
        last_id = results[-1][0]
        print("last_id: ", last_id)
    except Exception as e:
        print(f"An error occurred: {e}")

# Close the session
session.close()