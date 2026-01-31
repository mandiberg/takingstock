import os
import pickle
import cv2
import numpy as np
import sqlalchemy
from sqlalchemy import Float, create_engine, Column, Integer, Boolean, String
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy.pool import NullPool
from pathlib import Path
from pymediainfo import MediaInfo

# importing project-specific models
import sys
ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/takingstock/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)

from my_declarative_base import SegmentBig, SegmentTable, Encodings, Base, Images

# MongoDB setup
import pymongo
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["stock"]
mongo_collection = mongo_db["encodings"]  # adjust collection name if needed

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
batch_size = 1
last_id = 0 # this is now image_id, not encoding_id
VERBOSE = False

HelperTable_name = "SegmentHelper_oct2025_every40"
class HelperTable(Base):
    __tablename__ = HelperTable_name
    seg_image_id=Column(Integer,primary_key=True, autoincrement=True)
    image_id = Column(Integer, primary_key=True, autoincrement=True)

def get_shape(site_name_id,imagename):
    ## get the image somehow
    # select_image_ids_query = (
    #     select(SegmentTable.site_name_id,SegmentTable.imagename)
    #     .filter(SegmentTable.image_id == target_image_id)
    # )

    # result = session.execute(select_image_ids_query).fetchall()
    # site_name_id,imagename=result[0]
    site_specific_root_folder = io.folder_list[site_name_id]
    file=site_specific_root_folder+"/"+imagename  ###os.path.join was acting wierd so had to do this

    try:
        if io.platform == "darwin":
            media_info = MediaInfo.parse(file, library_file="/opt/homebrew/Cellar/libmediainfo/24.06/lib/libmediainfo.dylib")
        else:
            media_info = MediaInfo.parse(file)
    except Exception as e:
        print("Error getting media info, file not found", file)
        # traceback.print_exc() 
        return None,None

    for track in media_info.tracks:
        if track.track_type == 'Image':
            return track.height,track.width

    return None,None 


while True:
    # s = aliased(HelperTable, name='s')
    # ihp = aliased(ImagesArmsPoses3D, name='ihp')
    results = (
        session.query(
            Images.image_id,
            Images.site_name_id,
            Images.imagename
        )
        # .join(s, s.image_id == Encodings.image_id)
        # .join(ihp, s.image_id == ihp.image_id)
        .filter(
            Images.h.is_(None),
            Images.w.is_(None),
        )
        .limit(batch_size)
        .all()
    )
    # .order_by(Encodings.image_id)
        # Encodings.is_feet.is_(None),

    if not results:
        print("No more rows to process. Exiting.")
        break
    
    print(f"Processing {len(results)} records...")

    # Process in batches
    for i in range(0, len(results), batch_size):
        batch = results[i:i + batch_size]
        batch_success = []
        for image_id, site_name_id, imagename in batch:
            # # 1. build path to image file
            # image_path = io.get_image_path(site_name_id, imagename)
            # if not os.path.isfile(image_path):
            #     print(f"Image file not found: {image_path}. Skipping.")
            #     continue
            # print(f"Processing image_id: {image_id}, image_path: {image_path}")

            # 2. get h,w from pymediainfo
            h,w = get_shape(site_name_id,imagename)
            if h is None or w is None:
                print(f" ❌ Could not retrieve dimensions for image_id {image_id}. Skipping.")
                continue
            else:
                if VERBOSE: print(f"{image_id} - Retrieved dimensions: h={h}, w={w}")
                result_to_save = {

                    'image_id': int(image_id),
                    'h': int(h),
                    'w': int(w),
                }
                batch_success.append(result_to_save)
                print(f"✅ {image_id} - Successfully retrieved dimensions")
                    
            

        # 4. Save batch_success to MySQL in one batch based on dictionary
        if batch_success:
            # print(f"Saving {len(batch_success)} pose results to MySQL...")
            # print(f"these are the image_ids: {[item['image_id'] for item in batch_success]}")
            session.bulk_update_mappings(Images, batch_success)
            # session.bulk_update_mappings(SegmentBig, batch_success)
            # session.commit()
        print(f"Saved {len(batch_success)} pose results to MySQL.")

    # session.commit()
    last_id = results[-1][1]
    print(f"Processed up to image_id = {last_id}")

    # for testing, break after one batch
    break

session.close()
mongo_client.close()
engine.dispose()
# Note: Ensure that the MongoDB and MySQL connections are properly closed after processing.