import os
import pickle
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# importing project-specific models
import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from my_declarative_base import SegmentTable, Encodings, Base

# Define LocationHandsFeet model if not already defined
from sqlalchemy import Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
Base2 = declarative_base()

class LocationHandsFeet(Base2):
    __tablename__ = 'LocationHandsFeet'
    image_id = Column(Integer, primary_key=True)
    left_hand_x = Column(Float)
    left_hand_y = Column(Float)
    right_hand_x = Column(Float)
    right_hand_y = Column(Float)
    left_foot_x = Column(Float)
    left_foot_y = Column(Float)
    right_foot_x = Column(Float)
    right_foot_y = Column(Float)

# MongoDB setup
import pymongo
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["stock"]
mongo_collection = mongo_db["encodings"]  # original body_landmarks
mongo_collection_norm = mongo_db["body_landmarks_norm"]  # normalized landmarks

# MySQL setup (preserving credentials framework)
from mp_db_io import DataIO
io = DataIO()
db = io.db
engine = create_engine(
    f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
    poolclass=NullPool
)
Session = sessionmaker(bind=engine)
session = Session()

# Batch processing parameters
batch_size = 1000
last_id = 0

while True:

    # 1. switching to encoding table directly
    results = (
        session.query(Encodings.encoding_id, Encodings.image_id)
        .filter(
            Encodings.mongo_body_landmarks.is_(True),
            Encodings.is_feet.is_(True),
            Encodings.is_face.is_(True),
            Encodings.encoding_id > last_id
        )
        .order_by(Encodings.encoding_id)
        .limit(batch_size)
        .all()
    )
            # Encodings.is_feet.is_(None),

    if not results:
        print("No more rows to process. Exiting.")
        break


    for encoding_id, image_id in results:
        # Fetch normalized body landmarks from Mongo
        mongo_doc_norm = mongo_collection_norm.find_one({"image_id": image_id}, {"nlms": 1})
        if mongo_doc_norm and mongo_doc_norm.get("nlms"):
            nlms = pickle.loads(mongo_doc_norm["nlms"])
            # Extract xy for hands (15, 16) and feet (29, 30)
            # Assume nlms.landmark is a list of objects with .x and .y
            try:
                lh = nlms.landmark[15]
                rh = nlms.landmark[16]
                lf = nlms.landmark[29]
                rf = nlms.landmark[30]
                # Insert or update LocationHandsFeet
                session.merge(LocationHandsFeet(
                    image_id=image_id,
                    left_hand_x=lh.x, left_hand_y=lh.y,
                    right_hand_x=rh.x, right_hand_y=rh.y,
                    left_foot_x=lf.x, left_foot_y=lf.y,
                    right_foot_x=rf.x, right_foot_y=rf.y
                ))
            except Exception as e:
                print(f"Error extracting landmarks for image_id {image_id}: {e}")
        else:
            print(f"No normalized landmarks for image_id {image_id}")


    session.commit()
    last_id = results[-1][0]
    print(f"Processed up to seg_image_id = {last_id}")

session.close()
mongo_client.close()
engine.dispose()
# Note: Ensure that the MongoDB and MySQL connections are properly closed after processing.