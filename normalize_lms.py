#################################

from sqlalchemy import create_engine, text,func, select, delete, and_, or_
from sqlalchemy.orm import sessionmaker,scoped_session, declarative_base
from sqlalchemy.pool import NullPool
# from my_declarative_base import Images,ImagesBackground, SegmentTable, Site 
from mp_db_io import DataIO
import pickle
import numpy as np
from pick import pick
import threading
import queue
import csv
import os
import cv2
import mediapipe as mp
import shutil
import pandas as pd
import json
from my_declarative_base import Base, Clusters, Encodings, Images,PhoneBbox, SegmentTable, Images
#from sqlalchemy.ext.declarative import declarative_base
from mp_sort_pose import SortPose
import pymongo
from mediapipe.framework.formats import landmark_pb2
from pymediainfo import MediaInfo
import traceback 
import time

NOSE_ID=0


Base = declarative_base()
VERBOSE = False
SKIP_EXISTING = False # Skips images with a normed bbox but that have Images.h
IS_SSD = True

io = DataIO(IS_SSD)
db = io.db
# io.db["name"] = "stock"
# io.db["name"] = "ministock"
mongo_client = pymongo.MongoClient(io.dbmongo['host'])
mongo_db = mongo_client[io.dbmongo['name']]
mongo_collection = mongo_db[io.dbmongo['collection']]

bboxnormed_collection = mongo_db["body_landmarks_norm"]
# n_phonebbox_collection= mongo_db["bboxnormed_phone"]

def ensure_unique_index(collection, field_name):
    # List existing indexes on the collection
    indexes = list(collection.list_indexes())

    # Check if the unique index already exists
    for index in indexes:
        if index['key'] == {field_name: 1}:
            if index.get('unique', False):
                print(f"Unique index on '{field_name}' already exists.")
                return
            else:
                # Drop the non-unique index if it exists
                collection.drop_index(index['name'])
                print(f"Non-unique index on '{field_name}' dropped.")

                # Create a unique index on the specified field
                collection.create_index([(field_name, 1)], unique=True)
                print(f"Unique index on '{field_name}' created successfully.")

# Ensure unique index on the image_id field
ensure_unique_index(bboxnormed_collection, 'image_id')

# Create a database engine
if db['unix_socket']:
    # for MM's MAMP config
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)


image_edge_multiplier = [1.5,1.5,2,1.5] # bigger portrait
image_edge_multiplier_sm = [1.2, 1.2, 1.6, 1.2] # standard portrait
face_height_output = 500
motion = {"side_to_side": False, "forward_smile": True, "laugh": False, "forward_nosmile": False, "static_pose": False, "simple": False}

EXPAND = False
ONE_SHOT = False # take all files, based off the very first sort order.
JUMP_SHOT = False # jump to random file if can't find a run

sort = SortPose(motion, face_height_output, image_edge_multiplier_sm,EXPAND, ONE_SHOT, JUMP_SHOT, None, VERBOSE, False, None, 0)
# sort = SortPose(motion, face_height_output, image_edge_multiplier,EXPAND, ONE_SHOT, JUMP_SHOT, HSV_BOUNDS, VERBOSE,INPAINT, SORT_TYPE, OBJ_CLS_ID)

# Create a session
session = scoped_session(sessionmaker(bind=engine))

LIMIT= 10
# Initialize the counter
counter = 0
USE_OBJ = 0

# Number of threads
#num_threads = io.NUMBER_OF_PROCESSES
num_threads = 1

if VERBOSE: print("objects created")


def get_shape(target_image_id):
    ## get the image somehow
    select_image_ids_query = (
        select(SegmentTable.site_name_id,SegmentTable.imagename)
        .filter(SegmentTable.image_id == target_image_id)
    )

    result = session.execute(select_image_ids_query).fetchall()
    site_name_id,imagename=result[0]
    site_specific_root_folder = io.folder_list[site_name_id]
    file=site_specific_root_folder+"/"+imagename  ###os.path.join was acting wierd so had to do this

    try:
        if io.platform == "darwin":
            media_info = MediaInfo.parse(file, library_file="/opt/homebrew/Cellar/libmediainfo/24.06/lib/libmediainfo.dylib")
        else:
            media_info = MediaInfo.parse(file)
    except Exception as e:
        print("Error getting media info, file not found", target_image_id)
        # traceback.print_exc() 
        return None,None

    for track in media_info.tracks:
        if track.track_type == 'Image':
            return track.height,track.width

    return None,None 


# def normalize_phone_bbox(phone_bbox,nose_pos,face_height,shape):
#     height,width = shape[:2]
#     # print("phone_bbox",phone_bbox)
#     n_phone_bbox=phone_bbox
#     n_phone_bbox["right"]=(n_phone_bbox["right"] -nose_pos["x"])/face_height
#     n_phone_bbox["left"]=(n_phone_bbox["left"] -nose_pos["x"])/face_height
#     n_phone_bbox["top"]=(n_phone_bbox["top"] -nose_pos["y"])/face_height
#     n_phone_bbox["bottom"]=(n_phone_bbox["bottom"] -nose_pos["y"])/face_height
#     # n_phone_bbox["right"]=(n_phone_bbox["right"]*width -nose_pos["x"])/face_height
#     # n_phone_bbox["left"]=(n_phone_bbox["left"]*width -nose_pos["x"])/face_height
#     # n_phone_bbox["top"]=(n_phone_bbox["top"]*height -nose_pos["y"])/face_height
#     # n_phone_bbox["bottom"]=(n_phone_bbox["bottom"]*height -nose_pos["y"])/face_height
#     # print("n_phone_bbox",n_phone_bbox)

#     return n_phone_bbox

def get_landmarks_mongo(image_id):
    if image_id:
        results = mongo_collection.find_one({"image_id": image_id})
        if results:
            body_landmarks = results['body_landmarks']
            # print("got encodings from mongo, types are: ", type(face_encodings68), type(face_landmarks), type(body_landmarks))
            return body_landmarks
        else:
            return None
    else:
        return None
    
# def insert_n_landmarks(image_id,n_landmarks):
#     nlms_dict = { "image_id": image_id, "nlms": pickle.dumps(n_landmarks) }
#     x = bboxnormed_collection.insert_one(nlms_dict)
#     print("inserted id",x.inserted_id)
#     return


# def insert_n_landmarks(image_id, n_landmarks):
#     start = time.time()
#     nlms_dict = {"image_id": image_id, "nlms": pickle.dumps(n_landmarks)}
#     result = bboxnormed_collection.update_one(
#         {"image_id": image_id},  # filter
#         {"$set": nlms_dict},     # update
#         upsert=True              # insert if not exists
#     )
#     if result.upserted_id:
#         print("Inserted new document with id:", result.upserted_id)
#     else:
#         print("Updated existing document")
#     print("Time to insert:", time.time()-start)
#     return


def insert_n_phone_bbox(image_id,n_phone_bbox):
    # nlms_dict = { "image_id": image_id, "n_phone_bbox": n_phone_bbox }
    # x = n_phonebbox_collection.insert_one(nlms_dict)
    # print("inserted id",x.inserted_id)
    # return
    phone_bbox_norm_entry = (
        session.query(PhoneBbox)
        .filter(PhoneBbox.image_id == image_id)
        .first()
    )    
    if phone_bbox_norm_entry:
        phone_bbox_norm_entry.bbox_27_norm = json.dumps(n_phone_bbox)
        if VERBOSE:
            print("image_id:", PhoneBbox.image_id,"bbox_27_norm:", phone_bbox_norm_entry.bbox_27_norm)

            
    session.commit()

def unpickle_array(pickled_array):
    if pickled_array:
        try:
            # Attempt to unpickle using Protocol 3 in v3.7
            return pickle.loads(pickled_array, encoding='latin1')
        except TypeError:
            # If TypeError occurs, unpickle using specific protocl 3 in v3.11
            # return pickle.loads(pickled_array, encoding='latin1', fix_imports=True)
            try:
                # Set the encoding argument to 'latin1' and protocol argument to 3
                obj = pickle.loads(pickled_array, encoding='latin1', fix_imports=True, errors='strict', protocol=3)
                return obj
            except pickle.UnpicklingError as e:
                print(f"Error loading pickle data: {e}")
                return None
    else:
        return None

def get_face_height(target_image_id):
    select_image_ids_query = (
        select(SegmentTable.bbox)
        .filter(SegmentTable.image_id == target_image_id)
    )
    result = session.execute(select_image_ids_query).fetchall()
    bbox=result[0][0]

    face_height = sort.convert_bbox_to_face_height(bbox)
    return face_height

def insert_shape(target_image_id,shape):
    Images_entry = (
        session.query(Images)
        .filter(Images.image_id == target_image_id)
        # .first() #### MICHAEL I suspect this is a database dependent problem, doesnt work for me
    )    
    if Images_entry:
        Images_entry.w=shape[0]
        Images_entry.h=shape[1]
        if VERBOSE:
            print("image_id:", Images_entry.image_id,"height:", Images_entry.h,"width:", Images_entry.w)

            
    session.commit()
    return
def get_phone_bbox(target_image_id):
    select_image_ids_query = (
        select(PhoneBbox.bbox_27)
        .filter(PhoneBbox.image_id == target_image_id)
    )
    result = session.execute(select_image_ids_query).fetchall()
    phone_bbox=result[0][0]
    if type(phone_bbox)==str:
        phone_bbox=json.loads(phone_bbox)
        if VERBOSE: print("bbox type", type(phone_bbox))  

    return phone_bbox

def calc_nlm(image_id_to_shape, lock, session):
    # start a timer
    start = time.time()
    target_image_id = list(image_id_to_shape.keys())[0]
    height,width = image_id_to_shape[target_image_id]
    if VERBOSE: print("height,width from DB:",height,width)
    if VERBOSE: print("target_image_id",target_image_id)
    face_height=get_face_height(target_image_id)
    # print timer
    if VERBOSE: print("Time to get face height:", time.time()-start)
    start = time.time()

    if height and width:
        print(target_image_id, "have height,width already",height,width)
    else:
        height,width=get_shape(target_image_id)
        if not height or not width: 
            print(">> IMAGE NOT FOUND,", target_image_id)
            return
        insert_shape(target_image_id,[height,width])

    if VERBOSE: print("Time to get h w:", time.time()-start)
    start = time.time()

    body_landmarks=unpickle_array(get_landmarks_mongo(target_image_id))
    if VERBOSE: print("Time to get mongo lms:", time.time()-start)
    start = time.time()

    if body_landmarks:
        if VERBOSE: print("has body_landmarks")

        ### NORMALIZE LANDMARKS ###
        nose_pixel_pos = sort.set_nose_pixel_pos(body_landmarks,[height,width])
        print("nose_pixel_pos",nose_pixel_pos)
        n_landmarks=sort.normalize_landmarks(body_landmarks,nose_pixel_pos,face_height,[height,width])
        
        if VERBOSE: print("Time to get norm lms:", time.time()-start)
        start = time.time()

        sort.insert_n_landmarks(bboxnormed_collection, target_image_id,n_landmarks)
        if VERBOSE: print("did insert_n_landmarks, going to get phone bbox")
        if VERBOSE: print("Time to get insert:", time.time()-start)
        start = time.time()
        
        if USE_OBJ > 0: 
            phone_bbox=get_phone_bbox(target_image_id)
            if phone_bbox:
                n_phone_bbox=sort.normalize_phone_bbox(phone_bbox,nose_pixel_pos,face_height,[height,width])
                insert_n_phone_bbox(target_image_id,n_phone_bbox)
            else:
                print("PHONE BBOX NOT FOUND 404", target_image_id)
    else:
        print("BODY LANDMARK NOT FOUND 404", target_image_id)

    


    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter -= 1
        session.commit()
    if counter % 100 == 0:
        print(f"This many left: {counter}")
    return

#######MULTI THREADING##################
# Create a lock for thread synchronization
lock = threading.Lock()
threads_completed = threading.Event()

# Create a queue for distributing work among threads
work_queue = queue.Queue()
        
function=calc_nlm


if USE_OBJ == 27:
    distinct_image_ids_query = select(Images.image_id.distinct()).\
        outerjoin(SegmentTable,Images.image_id == SegmentTable.image_id).\
        outerjoin(PhoneBbox,PhoneBbox.image_id == SegmentTable.image_id).\
        filter(SegmentTable.bbox != None).\
        filter(SegmentTable.mongo_body_landmarks == 1).\
        filter(PhoneBbox.bbox_27 != None).\
        filter(PhoneBbox.bbox_27_norm == None).\
        filter(PhoneBbox.conf_27 != -1).\
        limit(LIMIT)

else:
    distinct_image_ids_query = select(Images.image_id.distinct(), Images.h, Images.w).\
        outerjoin(SegmentTable,Images.image_id == SegmentTable.image_id).\
        filter(SegmentTable.bbox != None).\
        filter(SegmentTable.mongo_body_landmarks == 1).\
        filter(SegmentTable.image_id >= 9942966).\
        limit(LIMIT)

# put this back in at future date if needed
        # filter(Images.h == None).\

if SKIP_EXISTING:
    # skips the ones that have obj bbox which have already been done
    normed_image_ids_query = select(SegmentTable.image_id.distinct(), Images.h, Images.w).\
        outerjoin(Images, Images.image_id == SegmentTable.image_id).\
        outerjoin(PhoneBbox,PhoneBbox.image_id == SegmentTable.image_id).\
        filter(
            or_(
                PhoneBbox.bbox_27_norm != None,
                PhoneBbox.bbox_67_norm != None,
                PhoneBbox.bbox_26_norm != None,
                PhoneBbox.bbox_27_norm != None,
                PhoneBbox.bbox_32_norm != None
            )
        ).\
        filter(SegmentTable.image_id >= 9942966).\
        limit(LIMIT)

    distinct_image_ids_query = distinct_image_ids_query.except_(normed_image_ids_query)


if VERBOSE: print("about to execute query")
results = session.execute(distinct_image_ids_query).fetchall()

# make a dictionary of image_id to shape
for result in results:
    image_id_to_shape = {}
    image_id, height, width = result
    image_id_to_shape[image_id] = (height, width)
    work_queue.put(image_id_to_shape)        

# distinct_image_ids = [row[0] for row in session.execute(distinct_image_ids_query).fetchall()]
# if VERBOSE: print("query length",len(distinct_image_ids))
# if VERBOSE: print("distinct_image_ids",(distinct_image_ids))
# for counter,target_image_id in enumerate(distinct_image_ids):
#     if counter%1000==0:print("###########"+str(counter)+"images processed ##########")
    # work_queue.put(target_image_id)        

if VERBOSE: print("queue filled")
        
def threaded_fetching():
    while not work_queue.empty():
        param = work_queue.get()
        function(param, lock, session)
        work_queue.task_done()

def threaded_processing():
    thread_list = []
    for _ in range(num_threads):
        thread = threading.Thread(target=threaded_fetching)
        thread_list.append(thread)
        thread.start()
    # Wait for all threads to complete
    for thread in thread_list:
        thread.join()
    # Set the event to signal that threads are completed
    threads_completed.set()

threaded_processing()
# Commit the changes to the database
threads_completed.wait()

print("done")
# Close the session
session.commit()
session.close()
