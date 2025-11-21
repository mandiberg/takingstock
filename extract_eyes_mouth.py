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
import math
from mp_pose_est import SelectPose

NOSE_ID=0


Base = declarative_base()
VERBOSE = False
IS_SSD = True

USE_OBJ = 0 # select the bbox to work with
SKIP_BODY = False # skip body landmarks. mostly you want to skip when doing obj bbox

io = DataIO(IS_SSD)
db = io.db
# io.db["name"] = "stock"
# io.db["name"] = "ministock"
mongo_client = pymongo.MongoClient(io.dbmongo['host'])
mongo_db = mongo_client[io.dbmongo['name']]
mongo_collection = mongo_db[io.dbmongo['collection']]

face_landmarks_collection = mongo_db["encodings"]
bboxnormed_collection = mongo_db["body_landmarks_norm"]
# n_phonebbox_collection= mongo_db["bboxnormed_phone"]

# start a timer
start = time.time()

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

# construct SortPose with config dict
cfg = {
    'motion': motion,
    'face_height_output': face_height_output,
    'image_edge_multiplier': image_edge_multiplier_sm,
    'EXPAND': EXPAND,
    'ONE_SHOT': ONE_SHOT,
    'JUMP_SHOT': JUMP_SHOT,
    'HSV_CONTROL': None,
    'VERBOSE': VERBOSE,
    'INPAINT': False,
    'SORT_TYPE': None,
    'OBJ_CLS_ID': 0
}
sort = SortPose(config=cfg)
# sort = SortPose(motion, face_height_output, image_edge_multiplier,EXPAND, ONE_SHOT, JUMP_SHOT, HSV_BOUNDS, VERBOSE,INPAINT, SORT_TYPE, OBJ_CLS_ID)

placeholder_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
pose = SelectPose(placeholder_image)

# Create a session
session = scoped_session(sessionmaker(bind=engine))

LIMIT= 25
# Initialize the counter
counter = 0

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
            if VERBOSE: print ("track.height,track.width", track.height,track.width)
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
            return unpickle_array(body_landmarks)
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
        phone_bbox_norm_entry.bbox_26_norm = json.dumps(n_phone_bbox)
        if VERBOSE:
            print("image_id:", PhoneBbox.image_id,"bbox_26_norm:", phone_bbox_norm_entry.bbox_26_norm)

            
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


def get_face_height_bbox(target_image_id):
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
        .first() #### MICHAEL I suspect this is a database dependent problem, doesnt work for me
    )    
    if Images_entry:
        Images_entry.h=shape[0]
        Images_entry.w=shape[1]
        if VERBOSE:
            print("image_id:", Images_entry.image_id,"height:", Images_entry.h,"width:", Images_entry.w)

            
    session.commit()
    return
def get_phone_bbox(target_image_id):
    select_image_ids_query = (
        select(PhoneBbox.bbox_26)
        .filter(PhoneBbox.image_id == target_image_id)
    )
    result = session.execute(select_image_ids_query).fetchall()
    phone_bbox=result[0][0]
    if type(phone_bbox)==str:
        phone_bbox=json.loads(phone_bbox)
        if VERBOSE: print("bbox type", type(phone_bbox))  

    return phone_bbox

def get_eyes_mouth(target_image_id):
    # select target image from mongo mongo_collection
    results = face_landmarks_collection.find_one({"image_id": target_image_id})
    if results:
        # set the face height input properties
        sort.faceLms = pickle.loads(results['face_landmarks'])
        # set the face height
        # sort.get_faceheight_data()
        # return sort.face_height
        dist_mouth = pose.get_dist_btwn_landmarks(sort.faceLms,13,14)
        dist_eye_l = pose.get_dist_btwn_landmarks(sort.faceLms,159,145)
        dist_eye_r = pose.get_dist_btwn_landmarks(sort.faceLms,386,374)
        eye_pitch = pose.get_eye_pitch(sort.faceLms)
        print("dist_mouth, dist_eye_l, dist_eye_r, eye_pitch",dist_mouth, dist_eye_l, dist_eye_r, eye_pitch)
        return dist_mouth,dist_eye_l,dist_eye_r,eye_pitch
    else:
        return None, None, None, None

def extract_eyes_mouth(image_id_to_shape, lock, session):
    if VERBOSE: print("extract_eyes_mouth target_image_id",target_image_id)
    target_image_id = list(image_id_to_shape.keys())[0]
    bbox = io.unstring_json(image_id_to_shape[target_image_id])
    dist_mouth, dist_eye_l, dist_eye_r, eye_pitch = get_eyes_mouth(target_image_id)
    print("dist_mouth, dist_eye_l, dist_eye_r, eye_pitch",dist_mouth, dist_eye_l, dist_eye_r, eye_pitch)
    

    ## begin testing stuff

    ## FOR TESTING
    def visualize_landmarks(target_image_id, nose_pixel_pos_face, nose_pixel_pos_body, body_landmarks):
        all_body_landmarks_dict = sort.get_landmarks_2d(body_landmarks, list(range(33)), "dict")
        # query mysql SegmentTable for imagename
        select_image_ids_query = (
            select(SegmentTable.imagename, SegmentTable.site_name_id)
            .filter(SegmentTable.image_id == target_image_id)
        )
        result = session.execute(select_image_ids_query).fetchall()
        imagename=result[0][0]
        site_name_id=result[0][1]

        # open the image with cv2
        image_path = os.path.join(io.ROOT,io.folder_list[site_name_id],imagename)
        print("image_path",image_path)
        image = cv2.imread(image_path)
        # print("image shape",image.shape)
        if nose_pixel_pos_face and nose_pixel_pos_body:
            # Draw the circle with the converted integer coordinates
            cv2.circle(image, (int(nose_pixel_pos_face[0]), int(nose_pixel_pos_face[1])), 5, (0, 255, 0), -1)
            cv2.circle(image, (int(nose_pixel_pos_body['x']), int(nose_pixel_pos_body['y'])), 10, (255, 0, 0), -1)

            # iterate through all_body_landmarks_dict and draw the points
            for key, value in all_body_landmarks_dict.items():
                print("value",value)
                x, y = value
                if x < 10 or y < 10:
                    cv2.circle(image, (int(x*sort.w), int(y*sort.h)), 5, (0, 0, 255), -1)
                else:
                    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        if image is not None and image.size > 0:
            image = pose.draw_face_landmarks(image, sort.faceLms, bbox)
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    # set the nose pixel position in the expected dictionary format

    visualize_landmarks(target_image_id, None, None, sort.faceLms)
    
    ## end testing stuff

    # gets ride of visiblity

    return
    session.query(Encodings).filter(Encodings.image_id == target_image_id).update({
            Encodings.dist_mouth: dist_mouth,
            Encodings.dist_eye_l: dist_eye_l,
            Encodings.dist_eye_r: dist_eye_r,
            Encodings.eye_pitch: eye_pitch
        }, synchronize_session=False)
    
    session.query(SegmentTable).filter(SegmentTable.image_id == target_image_id).update({
            SegmentTable.dist_mouth: dist_mouth,
            SegmentTable.dist_eye_l: dist_eye_l,
            SegmentTable.dist_eye_r: dist_eye_r,
            SegmentTable.eye_pitch: eye_pitch
        }, synchronize_session=False)
    
    # print timer

    ## FOR TESTING
    # projected_landmarks = sort.project_normalized_landmarks(n_landmarks, nose_pixel_pos_body, face_height, [height, width])
    # visualize_landmarks(target_image_id, nose_pixel_pos_face, nose_pixel_pos_body, projected_landmarks)



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

function=extract_eyes_mouth



distinct_image_ids_query = select(Images.image_id.distinct(), SegmentTable.bbox).\
    outerjoin(SegmentTable,Images.image_id == SegmentTable.image_id).\
    filter(SegmentTable.bbox != None).\
    filter(SegmentTable.two_noses.is_(None)).\
    filter(SegmentTable.mongo_body_landmarks == 1).\
    filter(SegmentTable.mongo_body_landmarks_norm.is_(None)).\
    limit(LIMIT).\
    offset(300)

# put this back in at future date if needed
        # filter(Images.h == None).\
        # filter(SegmentTable.image_id >= 9942966).\



if VERBOSE: print("about to execute query")
results = session.execute(distinct_image_ids_query).fetchall()

# make a dictionary of image_id to shape
for result in results:
    image_id_to_shape = {}
    image_id, bbox = result
    image_id_to_shape[image_id] = (bbox)

    ### temp single thread for debugging
    extract_eyes_mouth(image_id_to_shape, lock=None, session=session)
    print("done with single thread")
    print(" ")
    print(" ")
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

# Print the time taken
print("Time taken:", time.time()-start)