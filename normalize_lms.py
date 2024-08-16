#################################

from sqlalchemy import create_engine, text,func, select, delete, and_
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

NOSE_ID=0


Base = declarative_base()
VERBOSE = False

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

LIMIT= 10000
# Initialize the counter
counter = 0
USE_OBJ = 67

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
        traceback.print_exc() 
        return None,None

    for track in media_info.tracks:
        if track.track_type == 'Image':
            return track.height,track.width

    return None,None 

def normalize_landmarks(landmarks,nose_pos,face_height,shape):
    height,width = shape[:2]
    translated_landmarks = landmark_pb2.NormalizedLandmarkList()
    i=0
    for landmark in landmarks.landmark:
        # print(landmark)
        translated_landmark = landmark_pb2.NormalizedLandmark()
        translated_landmark.x = (landmark.x*width -nose_pos["x"])/face_height
        translated_landmark.y = (landmark.y*height-nose_pos["y"])/face_height
        translated_landmark.visibility = landmark.visibility
        translated_landmarks.landmark.append(translated_landmark)

    return translated_landmarks

def normalize_phone_bbox(phone_bbox,nose_pos,face_height,shape):
    height,width = shape[:2]
    # print("phone_bbox",phone_bbox)
    n_phone_bbox=phone_bbox
    n_phone_bbox["right"]=(n_phone_bbox["right"] -nose_pos["x"])/face_height
    n_phone_bbox["left"]=(n_phone_bbox["left"] -nose_pos["x"])/face_height
    n_phone_bbox["top"]=(n_phone_bbox["top"] -nose_pos["y"])/face_height
    n_phone_bbox["bottom"]=(n_phone_bbox["bottom"] -nose_pos["y"])/face_height
    # n_phone_bbox["right"]=(n_phone_bbox["right"]*width -nose_pos["x"])/face_height
    # n_phone_bbox["left"]=(n_phone_bbox["left"]*width -nose_pos["x"])/face_height
    # n_phone_bbox["top"]=(n_phone_bbox["top"]*height -nose_pos["y"])/face_height
    # n_phone_bbox["bottom"]=(n_phone_bbox["bottom"]*height -nose_pos["y"])/face_height
    # print("n_phone_bbox",n_phone_bbox)

    return n_phone_bbox

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
    
def insert_n_landmarks(image_id,n_landmarks):
    # bboxnormed_collection
    # print(image_id,n_landmarks)
    nlms_dict = { "image_id": image_id, "nlms": pickle.dumps(n_landmarks) }
    x = bboxnormed_collection.insert_one(nlms_dict)
    print("inserted id",x.inserted_id)
    return

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
        phone_bbox_norm_entry.bbox_67_norm = json.dumps(n_phone_bbox)
        if VERBOSE:
            print("image_id:", PhoneBbox.image_id,"bbox_67_norm:", phone_bbox_norm_entry.bbox_67_norm)

            
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
    if type(bbox)==str:
        bbox=json.loads(bbox)
    # if VERBOSE:
    #     print("bbox",bbox)
    face_height=bbox["top"]-bbox["bottom"]
    return face_height

def insert_shape(target_image_id,shape):
    Images_entry = (
        session.query(Images)
        .filter(Images.image_id == target_image_id)
        .first()
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
        select(PhoneBbox.bbox_67)
        .filter(PhoneBbox.image_id == target_image_id)
    )
    result = session.execute(select_image_ids_query).fetchall()
    phone_bbox=result[0][0]
    if type(phone_bbox)==str:
        phone_bbox=json.loads(phone_bbox)
        if VERBOSE: print("bbox type", type(phone_bbox))  

    return phone_bbox

def calc_nlm(target_image_id, lock, session):
    face_height=get_face_height(target_image_id)
    height,width=get_shape(target_image_id)
    if not height or not width: 
        print(">> IMAGE NOT FOUND,", target_image_id)
        return
    insert_shape(target_image_id,[height,width])

    body_landmarks=unpickle_array(get_landmarks_mongo(target_image_id))
    if body_landmarks:
        nose_pixel_pos ={
            "x":0,
            "y":0,
            "visibility":0
        }
        # nose_pixel_pos <- 864, 442 (stay as a separate variable)
        # nose_normalized_pos 0,0
        # nose_pos=body_landmarks.landmark[NOSE_ID]
        nose_pixel_pos["x"]+=body_landmarks.landmark[NOSE_ID].x*width
        nose_pixel_pos["y"]+=body_landmarks.landmark[NOSE_ID].y*height
        nose_pixel_pos["visibility"]+=body_landmarks.landmark[NOSE_ID].visibility
        n_landmarks=normalize_landmarks (body_landmarks,nose_pixel_pos,face_height,[height,width])
        insert_n_landmarks(target_image_id,n_landmarks)
        
        phone_bbox=get_phone_bbox(target_image_id)
        if phone_bbox:
            n_phone_bbox=normalize_phone_bbox(phone_bbox,nose_pixel_pos,face_height,[height,width])
            insert_n_phone_bbox(target_image_id,n_phone_bbox)
        else:
            print("PHONE BBOX NOT FOUND 404")
    else:
        print("BODY LANDMARK NOT FOUND 404")

    


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


if USE_OBJ == 67:
    distinct_image_ids_query = select(Images.image_id.distinct()).\
        outerjoin(SegmentTable,Images.image_id == SegmentTable.image_id).\
        outerjoin(PhoneBbox,PhoneBbox.image_id == SegmentTable.image_id).\
        filter(SegmentTable.bbox != None).\
        filter(SegmentTable.mongo_body_landmarks == 1).\
        filter(PhoneBbox.bbox_67 != None).\
        filter(PhoneBbox.bbox_67_norm == None).\
        filter(PhoneBbox.conf_67 != -1).\
        limit(LIMIT)

else:
    distinct_image_ids_query = select(Images.image_id.distinct()).\
        outerjoin(SegmentTable,Images.image_id == SegmentTable.image_id).\
        filter(SegmentTable.bbox != None).\
        filter(SegmentTable.mongo_body_landmarks == 1).\
        limit(LIMIT)

    # filter(Images.h == None).\



if VERBOSE: print("about to execute query")


distinct_image_ids = [row[0] for row in session.execute(distinct_image_ids_query).fetchall()]
if VERBOSE: print("query length",len(distinct_image_ids))
for counter,target_image_id in enumerate(distinct_image_ids):
    if counter%1000==0:print("###########"+str(counter)+"images processed ##########")
    work_queue.put(target_image_id)        

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
