import os
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool


# importing from another folder
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from mp_db_io import DataIO
from my_declarative_base import Images, Encodings, SegmentBig, Base, Clusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

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

# Define the batch size
batch_size = 5
last_id = 0
# currently set up for SegmentTable. need to change SegmentTable to Images if you want to use on main table

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
        results = session.query(SegmentBig.seg_image_id, SegmentBig.image_id).\
            filter(SegmentBig.tokenized_keyword_list.isnot(None), SegmentBig.mongo_tokens.is_(None), SegmentBig.seg_image_id > last_id).\
            limit(batch_size).all()
        if len(results) == 0:
            print("No more results found.")
            break

        # Initialize counters
        total_processed = 0
        current_batch = []

        for result in results:
            print("seg_image_id: ", result[0], "image_id: ", result[1])
        #     # for unhashpath
        #     # new_imagename, contentUrl = generate_local_unhashed_image_filepath(contentUrl)
        #     seg_image_id, image_id = result
        #     # for getty SNAFU
        #     print("seg_image_id: ", seg_image_id, "image_id: ", image_id)
        #     current_batch.append(image_id)
        #     total_processed += 1
        #     last_id = seg_image_id

        session.bulk_update_mappings(SegmentBig, [{"seg_image_id": seg_image_id, "image_id": image_id, "tokenized_keyword_list": None, "mongo_tokens": True} for seg_image_id, image_id in results])
        # session.commit()
        print(session.query(SegmentBig).filter(SegmentBig.seg_image_id.in_(current_batch)).update({"tokenized_keyword_list": None, "mongo_tokens": True}, synchronize_session=False))
        # print(f"{total_processed} Changes committed for {batch_size} rows.")
        current_batch = []
        last_id = results[-1][0]
        print("last_id: ", last_id)
        break
    except Exception as e:
        print(f"An error occurred: {e}")

# Close the session
session.close()