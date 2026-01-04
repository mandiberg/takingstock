#################################

import sys
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
from my_declarative_base import Base, Clusters, Encodings, Detections, Images,PhoneBbox, SegmentTable, SegmentBig, Images
#from sqlalchemy.ext.declarative import declarative_base
from mp_sort_pose import SortPose
import pymongo
from mediapipe.framework.formats import landmark_pb2
from pymediainfo import MediaInfo
import traceback 
import time
import math
import cv2
from sqlalchemy import Column, Integer, ForeignKey

NOSE_ID=0


Base = declarative_base()
VERBOSE = True
IS_SSD = True

SKIP_EXISTING = False # Skips images with a normed bbox but that have Images.h - I think only applies to phone bbox
USE_OBJ = True # do objet detections?
SKIP_BODY = True # skip body landmarks. mostly you want to skip when doing obj bbox
                # or are just redoing hands
REPROCESS_HANDS = False # do hands
IS_SEGMENT_BIG = False # use SegmentBig table. IF False, and IS_SSD is false, it will use Encodings table
SegmentHelper_name = 'SegmentHelperObject_74_clock'
THIS_CLASS_ID = 74 # for object bbox normalization
SegmentFolder = "/Volumes/OWC5/segment_images"
io = DataIO(IS_SSD)
db = io.db
# io.db["name"] = "stock"
# io.db["name"] = "ministock"
mongo_client = pymongo.MongoClient(io.dbmongo['host'])
mongo_db = mongo_client[io.dbmongo['name']]
mongo_collection = mongo_db[io.dbmongo['collection']]

face_landmarks_collection = mongo_db["encodings"]
bboxnormed_collection = mongo_db["body_landmarks_norm"]
mongo_hand_collection = mongo_db["hand_landmarks"]

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
    'SORT_TYPE': 'planar_hands',
    'OBJ_CLS_ID': 0
}
sort = SortPose(config=cfg)
# sort = SortPose(motion, face_height_output, image_edge_multiplier,EXPAND, ONE_SHOT, JUMP_SHOT, HSV_BOUNDS, VERBOSE,INPAINT, SORT_TYPE, OBJ_CLS_ID)

# Create a session
session = scoped_session(sessionmaker(bind=engine))

LIMIT= 2000000
# Initialize the counter
counter = 2000

# Number of threads
#num_threads = io.NUMBER_OF_PROCESSES
num_threads = 1

# define SegmentHelper 

class SegmentHelper(Base):
    __tablename__ = SegmentHelper_name
    seg_image_id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('Images.image_id'))


if VERBOSE: print("objects created")



def get_shape(target_image_id):
    ## get the image somehow
    if VERBOSE: print("get_shape target_image_id", target_image_id)
    
    if IS_SEGMENT_BIG:
        select_image_ids_query = (
            select(SegmentBig.site_name_id, SegmentBig.imagename)
            .filter(SegmentBig.image_id == target_image_id)
        )
    elif (not IS_SEGMENT_BIG and not IS_SSD) or (SegmentHelper_name is not None):
        # use images table
        select_image_ids_query = (
            select(Images.site_name_id, Images.imagename)
            .filter(Images.image_id == target_image_id)
        )
    else:
        select_image_ids_query = (
            select(SegmentTable.site_name_id, SegmentTable.imagename)
            .filter(SegmentTable.image_id == target_image_id)
        )
    
    result = session.execute(select_image_ids_query).fetchall()
    print("result", result)
    site_name_id, imagename = result[0]
    site_specific_root_folder = io.folder_list[site_name_id]
    if SegmentHelper_name is not None and IS_SSD and SegmentFolder is not None:
        file = os.path.join(SegmentFolder, os.path.basename(site_specific_root_folder), imagename)
        print("using SegmentHelper_name for path with SegmentFolder", file)
    else:
        file = site_specific_root_folder + "/" + imagename  # os.path.join acting weird, so avoided
    if VERBOSE: print("get_shape file", file)

    try:
        if io.platform == "darwin":
            media_info = MediaInfo.parse(file, library_file="/opt/homebrew/opt/libmediainfo/lib/libmediainfo.dylib")
            if VERBOSE: print("darwin got media_info")
        else:
            media_info = MediaInfo.parse(file)
    except Exception as e:
        print("Error getting media info, file not found", target_image_id)
        return None, None

    # Try to get image dimensions from MediaInfo
    for track in media_info.tracks:
        # print("track.track_type", track.track_type)
        
        # Check if it's an image track
        if track.track_type == 'Image':
            if VERBOSE: print("track.height, track.width", track.height, track.width)
            if track.height is not None and track.width is not None:
                return track.height, track.width
        
    # If MediaInfo fails, try loading the image with OpenCV as a fallback
    try:
        image = cv2.imread(file)
        if image is not None:
            height, width = image.shape[:2]  # Extract height and width
            if VERBOSE: print("cv2 found dimensions", height, width)
            return height, width
        else:
            print(f"Could not read the image using cv2, file: {file}")
            return None, None
    except Exception as e:
        print(f"Error loading image with cv2 for file: {file}, error: {e}")
        return None, None

    return None, None


def normalize_obj_bbox(obj_bbox,nose_pos,face_height,shape):
    height,width = shape[:2]
    print("obj_bbox type",type(obj_bbox))
    n_obj_bbox=io.unstring_json(obj_bbox)
    n_obj_bbox["right"]=(n_obj_bbox["right"] -nose_pos["x"])/face_height
    n_obj_bbox["left"]=(n_obj_bbox["left"] -nose_pos["x"])/face_height
    n_obj_bbox["top"]=(n_obj_bbox["top"] -nose_pos["y"])/face_height
    n_obj_bbox["bottom"]=(n_obj_bbox["bottom"] -nose_pos["y"])/face_height
    # n_obj_bbox["right"]=(n_obj_bbox["right"]*width -nose_pos["x"])/face_height
    # n_obj_bbox["left"]=(n_obj_bbox["left"]*width -nose_pos["x"])/face_height
    # n_obj_bbox["top"]=(n_obj_bbox["top"]*height -nose_pos["y"])/face_height
    # n_obj_bbox["bottom"]=(n_obj_bbox["bottom"]*height -nose_pos["y"])/face_height
    print("n_obj_bbox",n_obj_bbox)

    return n_obj_bbox

def get_landmarks_mongo(image_id):
    if image_id:
        results = mongo_collection.find_one({"image_id": image_id})
        if results:
            try:
                body_landmarks = results['body_landmarks']
            # print("got encodings from mongo, types are: ", type(face_encodings68), type(face_landmarks), type(body_landmarks))
                return unpickle_array(body_landmarks)
            except KeyError:
                print("KeyError, body_landmarks not found for", image_id)
                return None
        else:
            return None
    else:
        return None
    
def get_hand_landmarks_mongo(image_id):
    if image_id:
        results = mongo_hand_collection.find_one({"image_id": image_id})
        if results:
            hand_landmarks = results['nlms']
            # print("got encodings from mongo, types are: ", type(face_encodings68), type(face_landmarks), type(body_landmarks))
            return unpickle_array(hand_landmarks)
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
    from sqlalchemy import cast, JSON as JSON_TYPE
    phone_bbox_norm_entry = (
        session.query(PhoneBbox)
        .filter(PhoneBbox.image_id == image_id)
        .first()
    )    
    if phone_bbox_norm_entry:
        phone_bbox_norm_entry.bbox_26_norm = cast(json.dumps(n_phone_bbox), JSON_TYPE)
        if VERBOSE:
            print("image_id:", PhoneBbox.image_id,"bbox_26_norm:", phone_bbox_norm_entry.bbox_26_norm)
    session.commit()

def insert_detections_norm_bbox(detection_id,n_bbox):
    from sqlalchemy import cast, JSON as JSON_TYPE
    detections_bbox_norm_entry = (
        session.query(Detections)
        .filter(Detections.detection_id == detection_id)
        .first()
    )    
    if detections_bbox_norm_entry:
        detections_bbox_norm_entry.bbox_norm = cast(json.dumps(n_bbox), JSON_TYPE)
        if VERBOSE:
            print("detection_id:", Detections.detection_id,"bbox_norm:", detections_bbox_norm_entry.bbox_norm)
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

def get_face_height_face_lms(target_image_id,bbox, face_landmarks=None):
    # select target image from mongo mongo_collection if not passed through
    if not face_landmarks:
        results = face_landmarks
    else:
        results = face_landmarks_collection.find_one({"image_id": target_image_id})
    if results:
        # set the face height input properties
        if type(bbox)==str: bbox = io.unstring_json(bbox)
        sort.bbox = bbox
        sort.faceLms = pickle.loads(results['face_landmarks'])
        # set the face height
        sort.get_faceheight_data()
        return sort.face_height
    else:
        return None

def get_face_height_bbox(target_image_id):
    if IS_SEGMENT_BIG:
        select_image_ids_query = (
            select(SegmentBig.bbox)
            .filter(SegmentBig.image_id == target_image_id)
        )
    elif not IS_SEGMENT_BIG and not IS_SSD:
        # use Encodings table
        select_image_ids_query = (
            select(Encodings.bbox)
            .filter(Encodings.image_id == target_image_id)
        )
    else:
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

def get_obj_bbox(target_image_id):
    predicate = text("""
    (
    bbox_norm IS NULL
    OR NOT (
        JSON_EXTRACT(bbox_norm, '$.left') IS NOT NULL
        OR (
        JSON_TYPE(bbox_norm) = 'STRING'
        AND JSON_VALID(CAST(JSON_UNQUOTE(bbox_norm) AS JSON)) = 1
        AND JSON_EXTRACT(CAST(JSON_UNQUOTE(bbox_norm) AS JSON), '$.left') IS NOT NULL
        )
    )
    )
    """)
    select_image_ids_query = (
        select(Detections.detection_id, Detections.bbox, Detections.class_id, Detections.conf)
        .filter(and_(Detections.image_id == target_image_id))
        .filter(predicate)
    )
    
    result = session.execute(select_image_ids_query).fetchall()
    return result

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

def calc_nlm(image_id_to_shape, lock, session):
    if VERBOSE: print("calc_nlm image_id_to_shape",image_id_to_shape)
    target_image_id = list(image_id_to_shape.keys())[0]
    body_landmarks = None

    # TK this needs to be ported to calc body code
    height,width, bbox = image_id_to_shape[target_image_id]

    # get the shape of the image if no height in db
    if height and width:
        if VERBOSE: print(target_image_id, "have height,width already",height,width)
    else:
        height,width=get_shape(target_image_id)
        if not height or not width: 
            print(">> IMAGE NOT FOUND,", target_image_id)
            return
        insert_shape(target_image_id,[height,width])


    sort.h = height
    sort.w = width
    if VERBOSE: print("height,width from DB:",height,width)
    if VERBOSE: print("target_image_id",target_image_id)


    # get all the landmarks
    face_encodings68, face_landmarks, body_landmarks, body_landmarks_normalized, body_landmarks_3D, hand_results = io.get_encodings_mongo(target_image_id)

    if face_landmarks is None:
        print("FACE LANDMARK NOT FOUND 404, bailing for this one ", target_image_id)
        return
    face_height=get_face_height_face_lms(target_image_id,bbox, face_landmarks)
    
    if sort.VERBOSE: print("face_height from lms",face_height)
    nose_pixel_pos_face = sort.get_face_2d_point(1)
    if sort.VERBOSE: print("nose_pixel_pos from face",nose_pixel_pos_face)
    # only do this if the io.get_encodings_mongo didn't return the body landmarks
    if not body_landmarks: body_landmarks=get_landmarks_mongo(target_image_id)
    print("body_landmarks",type(body_landmarks), " first few lms:", body_landmarks[:5] if body_landmarks else "None")
    if body_landmarks and type(body_landmarks) == bytes:
        nose_pixel_pos_body_withviz = sort.set_nose_pixel_pos(body_landmarks,[height,width])
        # exit the whole script
        print(" ✅ ✅ ✅ Converted and saved lms, with nose pixel", nose_pixel_pos_body_withviz)
        # sys.exit(0)

    else:
        print("BODY LANDMARK NOT FOUND 404, bailing for this one ", target_image_id)
        return
        nose_pixel_pos_body_withviz = nose_pixel_pos_face
    if sort.VERBOSE: print("nose_pixel_pos from body",nose_pixel_pos_body_withviz)
    # hand_results=get_hand_landmarks_mongo(target_image_id)
    # print("hand_results",hand_results)

    #     # for drawing landmarks on test image
    # landmarks_2d = sort.get_landmarks_2d(row['face_landmarks'], list(range(33)), "list")
    # print("landmarks_2d before drawing", landmarks_2d)
    # cropped_image = sort.draw_point(cropped_image, landmarks_2d, index = 0)                    

    # landmarks_2d = sort.get_landmarks_2d(row['face_landmarks'], list(range(420)), "list")
    # cropped_image = sort.draw_point(cropped_image, landmarks_2d, index = 0)                    

    # Extract x and y coordinates
    xF, yF = int(nose_pixel_pos_face[0]), int(nose_pixel_pos_face[1])
    xB, yB = int(nose_pixel_pos_body_withviz['x']), int(nose_pixel_pos_body_withviz['y'])
    

        # gets ride of visiblity
    nose_pixel_pos_body = {'x': xB, 'y': yB}


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

        # Draw the circle with the converted integer coordinates
        image = cv2.imread(image_path)
        print("image shape",image.shape)
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

        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # set the nose pixel position in the expected dictionary format

    # visualize_landmarks(target_image_id, nose_pixel_pos_face, nose_pixel_pos_body, body_landmarks)
    
    ## end testing stuff

    # gets ride of visiblity
    nose_pixel_pos_body = {'x': xB, 'y': yB}

    # Calculate Euclidean distance
    distance = math.sqrt((xB - xF)**2 + (yB - yF)**2)

    if distance > 300:
        print(f" >>> TWO NOSES - {target_image_id}. Dist between nose_pixel_pos and nose_pixel_pos_body:", distance)

        session.query(Encodings).filter(Encodings.image_id == target_image_id).update({
                Encodings.two_noses: 1
            }, synchronize_session=False)
        if not IS_SEGMENT_BIG:
            # skip this if using SegmentBig, no two_noses column
            session.query(SegmentTable).filter(SegmentTable.image_id == target_image_id).update({
                    SegmentTable.two_noses: 1
                }, synchronize_session=False)
        session.commit()
        return

    else:
        # DO THIS LATER
        # TK store face_height and nose_pixel_pos_body in encodings table
        pass


    if hand_results:
        if VERBOSE: print("has hand_landmarks")
        hand_landmarks_norm=sort.normalize_hand_landmarks(hand_results,nose_pixel_pos_body,face_height,[height,width])
        # print("hand_landmarks_norm",hand_landmarks_norm)
        sort.update_hand_landmarks_in_mongo(mongo_hand_collection, target_image_id,hand_landmarks_norm)
        session.query(Encodings).filter(Encodings.image_id == target_image_id).update({
                Encodings.mongo_hand_landmarks_norm: 1
            }, synchronize_session=False)
        if not IS_SEGMENT_BIG:
            # skip this if using SegmentBig, no mongo_hand_landmarks_norm column
            session.query(SegmentTable).filter(SegmentTable.image_id == target_image_id).update({
                    SegmentTable.mongo_hand_landmarks_norm: 1
                }, synchronize_session=False)
        # session.commit()    
        
    if body_landmarks:
        if VERBOSE: print("has body_landmarks")

        ### NORMALIZE LANDMARKS ###
        if VERBOSE: print("nose_pixel_pos",nose_pixel_pos_body)

        if not SKIP_BODY:
            # if type body_landmarks is bytes, unpickle it
            if type(body_landmarks) == bytes:
                body_landmarks = pickle.loads(body_landmarks)
            n_landmarks=sort.normalize_landmarks(body_landmarks,nose_pixel_pos_body,face_height,[height,width])

            # if VERBOSE: print("Time to get norm lms:", time.time()-start)
            # start = time.time()

            if VERBOSE: print("about to insert n_landmarks",n_landmarks)
            
            # store the normalized landmarks in mongo
            sort.insert_n_landmarks(bboxnormed_collection, target_image_id,n_landmarks)
            # update encodings and segment tables to reflect that the landmarks have been normalized
            session.query(Encodings).filter(Encodings.image_id == target_image_id).update({
                    Encodings.mongo_body_landmarks_norm: 1
                }, synchronize_session=False)
            if not IS_SEGMENT_BIG:
                # skip this if using SegmentBig, no mongo_body_landmarks_norm column
                session.query(SegmentTable).filter(SegmentTable.image_id == target_image_id).update({
                        SegmentTable.mongo_body_landmarks_norm: 1
                    }, synchronize_session=False)
            # session.commit()

            if VERBOSE: print("did insert_n_landmarks, going to get phone bbox")
            # if VERBOSE: print("Time to get insert:", time.time()-start)
            # start = time.time()
        
        elif USE_OBJ: 
            obj_results = get_obj_bbox(target_image_id)
            # itterate through the obj_results
            print("obj_results",obj_results)
            for detection_id, obj_bbox, class_id, conf in obj_results:
                if obj_bbox and (obj_bbox != 'null' or conf > 0):
                    print("going to normalize obj_bbox",obj_bbox)
                    n_obj_bbox=normalize_obj_bbox(obj_bbox,nose_pixel_pos_body,face_height,[height,width])
                    # temp comment
                    insert_detections_norm_bbox(detection_id,n_obj_bbox)
                else:
                    print("PHONE BBOX NOT FOUND 404", target_image_id)

            # phone_bbox=get_phone_bbox(target_image_id)
            # if phone_bbox:
            #     n_phone_bbox=normalize_obj_bbox(phone_bbox,nose_pixel_pos_body,face_height,[height,width])
            #     insert_detections_norm_bbox(target_image_id,n_phone_bbox)
            # else:
            #     print("PHONE BBOX NOT FOUND 404", target_image_id)
    else:
        print("BODY LANDMARK NOT FOUND 404", target_image_id)

    ## FOR TESTING
    # projected_landmarks = sort.project_normalized_landmarks(n_landmarks, nose_pixel_pos_body, face_height, [height, width])
    # visualize_landmarks(target_image_id, nose_pixel_pos_face, nose_pixel_pos_body, projected_landmarks)



    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter -= 1
        # temp comment
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


# old way, with phone bbox
# if USE_OBJ == 26:
#     distinct_image_ids_query = select(Images.image_id.distinct(), Images.h, Images.w, SegmentTable.bbox).\
#         outerjoin(SegmentTable,Images.image_id == SegmentTable.image_id).\
#         outerjoin(PhoneBbox,PhoneBbox.image_id == SegmentTable.image_id).\
#         filter(SegmentTable.bbox != None).\
#         filter(SegmentTable.two_noses.is_(None)).\
#         filter(SegmentTable.mongo_body_landmarks == 1).\
#         filter(PhoneBbox.bbox_26 != None).\
#         filter(PhoneBbox.bbox_26_norm == None).\
#         filter(PhoneBbox.conf_26 != -1).\
#         limit(LIMIT)

# new way, with detections
if USE_OBJ:
    print("doing OBJ using Detections")
    predicate_text = """
    (
    bbox_norm IS NULL
    OR NOT (
        JSON_EXTRACT(bbox_norm, '$.left') IS NOT NULL
        OR (
        JSON_TYPE(bbox_norm) = 'STRING'
        AND JSON_VALID(CAST(JSON_UNQUOTE(bbox_norm) AS JSON)) = 1
        AND JSON_EXTRACT(CAST(JSON_UNQUOTE(bbox_norm) AS JSON), '$.left') IS NOT NULL
        )
    )
    )
    """

    # distinct_image_ids_query = select(Detections.detection_id, Detections.bbox, Detections.class_id, Detections.conf).filter(text(predicate_text))

    distinct_image_ids_query = select(Images.image_id.distinct(), Images.h, Images.w, Encodings.bbox).\
        outerjoin(Encodings,Images.image_id == Encodings.image_id).\
        outerjoin(Detections,Detections.image_id == Encodings.image_id).\
        join(SegmentHelper,SegmentHelper.image_id == Encodings.image_id).\
        filter(Encodings.bbox != None).\
        filter(Encodings.two_noses.is_(None)).\
        filter(Encodings.mongo_body_landmarks == 1).\
        filter(Detections.bbox != None).\
        filter(text(predicate_text)).\
        filter(Detections.class_id == THIS_CLASS_ID).\
        filter(Detections.conf != -1).\
        limit(LIMIT)
    
    # distinct_image_ids_query = select(Images.image_id.distinct(), Images.h, Images.w, Encodings.bbox).\
    #     outerjoin(SegmentTable,Images.image_id == SegmentTable.image_id).\
    #     outerjoin(Detections,Detections.image_id == SegmentTable.image_id).\
    #     filter(SegmentTable.bbox != None).\
    #     filter(SegmentTable.two_noses.is_(None)).\
    #     filter(SegmentTable.mongo_body_landmarks == 1).\
    #     filter(Detections.bbox != None).\
    #     filter(Detections.bbox_norm == None).\
    #     filter(Detections.conf != -1).\
    #     limit(LIMIT)

elif REPROCESS_HANDS == True and IS_SEGMENT_BIG == True:
    print("doing HANDS using SegmentTable")
    distinct_image_ids_query = select(Images.image_id.distinct(), Images.h, Images.w, SegmentTable.bbox).\
        outerjoin(SegmentTable,Images.image_id == SegmentTable.image_id).\
        filter(SegmentTable.bbox != None).\
        filter(SegmentTable.two_noses.is_(None)).\
        filter(SegmentTable.mongo_hand_landmarks == 1).\
        filter(SegmentTable.mongo_hand_landmarks_norm.is_(None)).\
        limit(LIMIT)
elif REPROCESS_HANDS == True and IS_SEGMENT_BIG == False:
    print("doing HANDS using Encodings")
    distinct_image_ids_query = select(Images.image_id.distinct(), Images.h, Images.w, Encodings.bbox).\
        outerjoin(Images, Images.image_id == Encodings.image_id).\
        filter(Encodings.bbox != None).\
        filter(Encodings.two_noses.is_(None)).\
        filter(Encodings.mongo_hand_landmarks == 1).\
        filter(Encodings.mongo_hand_landmarks_norm.is_(None)).\
        limit(LIMIT)
elif not SKIP_BODY and IS_SEGMENT_BIG == True:
    print("doing BODY using SegmentBig")
    distinct_image_ids_query = select(Images.image_id.distinct(), Images.h, Images.w, SegmentTable.bbox).\
        outerjoin(SegmentTable,Images.image_id == SegmentTable.image_id).\
        filter(SegmentTable.bbox != None).\
        filter(SegmentTable.two_noses.is_(None)).\
        filter(SegmentTable.mongo_body_landmarks == 1).\
        filter(SegmentTable.mongo_body_landmarks_norm.is_(None)).\
        limit(LIMIT)
elif not SKIP_BODY and IS_SEGMENT_BIG == False:
    # use Encodings table
    print("doing BODY using Encodings")
    distinct_image_ids_query = select(Images.image_id.distinct(), Images.h, Images.w, Encodings.bbox).\
        outerjoin(Images, Images.image_id == Encodings.image_id).\
        filter(Encodings.bbox != None).\
        filter(Encodings.two_noses.is_(None)).\
        filter(Encodings.mongo_body_landmarks == 1).\
        filter(Encodings.mongo_body_landmarks_norm.is_(None)).\
        limit(LIMIT)
else:
    print(f"doing something else that wasn't caught because SKIP_BODY is {SKIP_BODY}, REPROCESS_HANDS is {REPROCESS_HANDS} and IS_SEGMENT_BIG is {IS_SEGMENT_BIG}")


# # TESTING OVERRIDE seghelper is for testing
# distinct_image_ids_query = select(Images.image_id.distinct(), Images.h, Images.w, SegmentBig.bbox).\
#     outerjoin(SegmentBig,Images.image_id == SegmentBig.image_id).\
#     outerjoin(SegmentHelper,Images.image_id == SegmentHelper.image_id).\
#     outerjoin(Encodings, Encodings.image_id == SegmentBig.image_id).\
#     outerjoin(NMLImages, NMLImages.image_id == SegmentBig.image_id).\
#     filter(Encodings.bbox != None).\
#     filter(Encodings.two_noses.is_(None)).\
#     filter(Encodings.mongo_hand_landmarks == 1).\
#     filter(Encodings.mongo_hand_landmarks_norm.is_(None)).\
#     filter(NMLImages.nml_id > 4191363).\
#     limit(LIMIT)

# put this back in at future date if needed
        # filter(Images.h == None).\
        # filter(SegmentTable.image_id >= 9942966).\

if SKIP_EXISTING:
    # skips the ones that have obj bbox which have already been done
    normed_image_ids_query = select(SegmentTable.image_id.distinct(), Images.h, Images.w).\
        outerjoin(Images, Images.image_id == SegmentTable.image_id).\
        outerjoin(PhoneBbox,PhoneBbox.image_id == SegmentTable.image_id).\
        filter(
            or_(
                PhoneBbox.bbox_63_norm != None,
                PhoneBbox.bbox_67_norm != None,
                PhoneBbox.bbox_26_norm != None,
                PhoneBbox.bbox_26_norm != None,
                PhoneBbox.bbox_32_norm != None
            )
        ).\
        filter(SegmentTable.image_id >= 9942966).\
        limit(LIMIT)

    distinct_image_ids_query = distinct_image_ids_query.except_(normed_image_ids_query)


if VERBOSE: print("about to execute query")
results = session.execute(distinct_image_ids_query).fetchall()
if VERBOSE: print("query executed, results length", len(results))
# make a dictionary of image_id to shape
for result in results:
    if VERBOSE: print("result", result)
    image_id_to_shape = {}
    image_id, height, width, bbox = result
    image_id_to_shape[image_id] = (height, width, bbox)

    ### temp single thread for debugging
    # calc_nlm(image_id_to_shape, lock=None, session=session)
    if VERBOSE: print("done with single thread", image_id_to_shape)
    # print(" ")
    # print(" ")
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
# temp comment
# session.commit()
session.close()

# Print the time taken
print("Time taken:", time.time()-start)