import os
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
from pathlib import Path
import pandas as pd


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
batch_size = 10000

# select max encoding_id to start from
Base = declarative_base()

class CompareSqlMongoResults(Base):
    __tablename__ = 'compare_sql_mongo_results'
    encoding_id = Column(Integer, primary_key=True)
    image_id = Column(Integer)
    is_body = Column(Boolean)
    is_face = Column(Boolean)
    face_landmarks = Column(Integer)
    body_landmarks = Column(Integer)
    face_encodings68 = Column(Integer)
    nlms = Column(Integer)
    left_hand = Column(Integer)
    right_hand = Column(Integer)
    body_world_landmarks = Column(Integer)
Base.metadata.create_all(engine)
last_id = session.query(sqlalchemy.func.max(CompareSqlMongoResults.encoding_id)).scalar()
if last_id is None:
    last_id = 0
print(f"Starting from last_id: {last_id}")

# variables to filter encodings on
migrated_SQL = 1
migrated = None
migrated_Mongo = None
is_body = 1
is_face = 1
mongo_body_landmarks_3D = 1

collection_names = ['encodings', 'body_landmarks_norm', 'hand_landmarks', 'body_landmarks_3D']
document_names_dict = {
    # "encodings": ["encoding_id", "face_landmarks", "body_landmarks", "face_encodings68"],
    "encodings": ["face_landmarks", "body_landmarks", "face_encodings68"],
    "body_landmarks_norm": ["nlms"],
    "hand_landmarks": ["left_hand", "right_hand"],
    "body_landmarks_3D": ["body_world_landmarks"]
}

sql_field_names_dict = {
    "face_landmarks": "mongo_face_landmarks",
    "body_landmarks": "mongo_body_landmarks",
    "face_encodings68": "mongo_encodings",
    "nlms": "mongo_body_landmarks_norm",
    "left_hand": "mongo_hand_landmarks",
    "right_hand": "mongo_hand_landmarks",
    "body_world_landmarks": "mongo_body_landmarks_3D"
}


# Initialize counters outside the loop (at the top of your script, before the while loop)
if 'counts' not in locals():
    counts = {}
    for collection_name in collection_names:
        for document_name in document_names_dict[collection_name]:
            key = f"{collection_name}.{document_name}"
            counts[key] = {"sql_only": 0, "mongo_only": 0, "both_match": 0}

results_rows = []


'''
Go through the Encodings table in batches, using the variables above to filter.
There will be about 7 million results in total
For each encoding, get the image_id and use it to compare the SQL data to the Mongo data
For each collection in collection_names, compare the SQL booleans to the actual presence of data in Mongo
If there is a mismatch, print it out
If there is data in SQL but not in Mongo, print it out
If there is data in Mongo but not in SQL, print it out

'''

while True:
    results = (
        session.query(Encodings.encoding_id, Encodings.image_id, Encodings.mongo_encodings, Encodings.mongo_body_landmarks, Encodings.mongo_face_landmarks, Encodings.mongo_body_landmarks_norm, Encodings.mongo_hand_landmarks, Encodings.mongo_body_landmarks_3D, Encodings.is_body, Encodings.is_face)
        .filter(
            # Encodings.migrated_SQL == migrated_SQL,
            # Encodings.migrated == migrated,
            # Encodings.migrated_Mongo == migrated_Mongo,
            # Encodings.is_body == is_body,
            # Encodings.is_face == is_face,
            # Encodings.mongo_body_landmarks_3D == mongo_body_landmarks_3D,
            Encodings.encoding_id > last_id
        )
        .order_by(Encodings.encoding_id)
        .limit(batch_size)
        .all()
    )

    if not results:
        print("No more rows to process. Exiting.")
        break
    for encoding_id, image_id, mongo_encodings, mongo_body_landmarks, mongo_face_landmarks, mongo_body_landmarks_norm, mongo_hand_landmarks, mongo_body_landmarks_3D, is_body, is_face in results:
        last_id = encoding_id  # Update last_id for the next batch
        # print(f"Processing encoding_id: {encoding_id}, image_id: {image_id}")

        # Fetch documents from MongoDB
        mongo_docs = {}
        for collection_name in collection_names:
            collection = mongo_db[collection_name]
            doc = collection.find_one({"image_id": image_id})
            mongo_docs[collection_name] = doc
        # print all the boolean values
        # print(f"SQL booleans for encoding_id {encoding_id}, image_id {image_id}:")
        # for field_name in sql_field_names_dict.values():
        #     print(f"  {field_name}: {locals().get(field_name)}")
        #     print(f"MongoDB documents for image_id {image_id}:")
        # Compare SQL booleans to Mongo data presence
        for collection_name in collection_names:
            doc = mongo_docs[collection_name]
            document_names = document_names_dict[collection_name]
            for document_name in document_names:
                sql_field_name = sql_field_names_dict[document_name]
                sql_boolean = locals().get(sql_field_name)  # Get the SQL boolean value dynamically
                mongo_data_present = doc is not None and document_name in doc and doc[document_name] is not None
                # print(f"  {collection_name}.{document_name}: Mongo data present: {mongo_data_present}")


                # # Inside the loop, increment counters instead of printing
                # if sql_boolean and not mongo_data_present:
                #     counts[f"{collection_name}.{document_name}"]["sql_only"] += 1
                # elif not sql_boolean and mongo_data_present:
                #     counts[f"{collection_name}.{document_name}"]["mongo_only"] += 1
                # else:
                #     counts[f"{collection_name}.{document_name}"]["both_match"] += 1



    # print(f"Processed batch up to encoding_id {last_id}:")
    # for key, value in counts.items():
    #     print(f"  {key}: {value}")
        if encoding_id is None or image_id is None:
            print(f" -- skip encoding_id {encoding_id} no image_id")
            continue
        # initialize the result row
        this_row = {
            "encoding_id": encoding_id,
            "image_id": image_id,
            "is_body": is_body,
            "is_face": is_face
        }

        # Compare SQL booleans to Mongo data presence
        for collection_name in collection_names:
            doc = mongo_docs[collection_name]
            document_names = document_names_dict[collection_name]
            for document_name in document_names:
                sql_field_name = sql_field_names_dict[document_name]
                sql_boolean = locals().get(sql_field_name)
                mongo_data_present = doc is not None and document_name in doc and doc[document_name] is not None

                # Determine value to store
                if sql_boolean and not mongo_data_present:
                    value = 0
                elif not sql_boolean and mongo_data_present:
                    value = 1
                else:
                    value = None

                # for each document, create a new row in results_rows
                this_row[document_name] = value

        if any(v is not None for k, v in this_row.items() if k not in ["encoding_id", "image_id", "is_body", "is_face"]):
            results_rows.append(this_row)

    # At the end of each batch, create a DataFrame and (optionally) save or process it
    batch_df = pd.DataFrame(results_rows)
    print(f"Processed batch up to encoding_id {last_id}: discrepancies found this batch: {len(batch_df)}")
    # print(batch_df)
    # Insert into MySQL table
    batch_df.to_sql(
        name="compare_sql_mongo_results",
        con=engine,
        if_exists="append",
        index=False,
        method="multi"
    )
    results_rows.clear()
    # break  # temporary for testing


# Close the session
session.close()