from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import csv
import os
import hashlib
import cv2
import math
import pickle
import sys # can delete for production
from sys import platform
import json
import base64
import gc
import traceback
import threading
import psutil

import sys
import re
import types # Import types for SimpleNamespace
        
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2 # Needed for NormalizedLandmarkList and LandmarkList
from mediapipe.framework.formats import classification_pb2 # Needed for ClassificationList and Classification
import pandas as pd
from ultralytics import YOLO

from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, Boolean, DECIMAL, BLOB, JSON, String, Date, ForeignKey, update, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
# my ORM
from my_declarative_base import Base, Images, WanderingImages, NMLImages, Keywords, Counters, SegmentTable, SegmentBig_isnotface, ImagesKeywords, ImagesBackground, Encodings, PhoneBbox, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects import mysql
import pymongo
from pymongo.errors import DuplicateKeyError

from mp_pose_est import SelectPose
from mp_db_io import DataIO
from mp_sort_pose import SortPose

#####new imports #####
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

import dlib
import face_recognition_models


# outputfolder = os.path.join(ROOT,folder+"_output_febmulti")
SAVE_ORIG = False
DRAW_BOX = False
MINSIZE = 500
SLEEP_TIME=0
VERBOSE = False

# only for triage
sortfolder ="getty_test"

#use in some clean up for getty
http="https://media.gettyimages.com/photos/"

# am I looking on RAID/SSD for a folder? If not, will pull directly from SQL
# if so, also change the site_name_id etc around line 930
IS_FOLDER = True

# these only matter if SQL (not folder)
DO_OVER = True
FIND_NO_IMAGE = True
# OVERRIDE_PATH = False
OVERRIDE_PATH = "/Volumes/SSD4/images_getty"
OVERRIDE_TOPIC = False
# OVERRIDE_TOPIC = [16, 17, 18, 23, 24, 45, 53]
SHUTTER_SSD_OVERRIDE = True
if SHUTTER_SSD_OVERRIDE: 
    OVERRIDE_PATH = "/Volumes/SSD4green/images_shutterstock"
    SHUTTERFOLDER = "C/C"

'''
Oct 13, got up to 109217155
switching to topic targeted
'''

'''
1   getty
2   shutterstock - IP
3   adobe
4   istock
5   pexels - all wandering?
6   unsplash
7   pond5 - all wandering?
8   123rf
9   alamy
10  visualchinagroup
11	picxy
12	pixerf
13	imagesbazaar
14	indiapicturebudget
15	iwaria
16	nappy 
17	picha
18	afripics
'''
# I think this only matters for IS_FOLDER mode, and the old SQL way
SITE_NAME_ID = 2
# 2, shutter. 4, istock
# 7 pond5, 8 123rf
POSE_ID = 0

# folder doesn't matter if IS_FOLDER is False. Declared FAR below. 
# MAIN_FOLDER = "/Volumes/RAID54/images_shutterstock"
# MAIN_FOLDER = "/Volumes/OWC5/images_adobe"
# MAIN_FOLDER = "/Volumes/ExFAT_SSD4_/images_adobe"
MAIN_FOLDER = "/Volumes/OWC5/images_shutterstock"
# MAIN_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/afripics_v2/images"

# MAIN_FOLDER = "/Volumes/SSD4/images_getty_reDL"
BATCH_SIZE = 1000 # Define how many from each folder in each batch
LIMIT = 1000

#temp hack to go 1 subfolder at a time
THESE_FOLDER_PATHS = ["9/9C", "9/9D", "9/9E", "9/9F", "9/90", "9/91", "9/92", "9/93", "9/94", "9/95", "9/96", "9/97", "9/98", "9/99"]

# MAIN_FOLDER = "/Volumes/SSD4/adobeStockScraper_v3/images"
# MAIN_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/gettyimages/newimages"
CSV_FOLDERCOUNT_PATH = os.path.join(MAIN_FOLDER, "folder_countout.csv")

IS_SSD=True

# set BODY to true, set SSD to false, set TOPIC_ID
# for silence, start at 103893643
# for HDD topic, start at 28714744
BODYLMS = True
HANDLMS = True
REDO_BODYLMS_3D = True # this makes it skip hands and YOLO
if REDO_BODYLMS_3D: HANDLMS = False # if doing 3D redo, don't do hands
TOPIC_ID = None
# TOPIC_ID = [24, 29] # adding a TOPIC_ID forces it to work from SegmentBig_isface, currently at 7412083
DO_INVERSE = True
SEGMENT = 0 # topic_id set to 0 or False if using HelperTable or not using a segment
HelperTable_name = "SegmentHelper_nov23_T37_forwardish" # set to False if not using a HelperTable
# HelperTable_name = False    
# SegmentTable_name = 'SegmentOct20'
SegmentTable_name = 'SegmentBig_isnotface'
# if HelperTable_name, set start point
START_IMAGE_ID = 0

if BODYLMS is True or HANDLMS is True:
    # THIS NEEDS TO BE REFACTORED FOR GPU
    get_background_mp = mp.solutions.selfie_segmentation
    get_bg_segment = get_background_mp.SelfieSegmentation()

    ############# Reencodings #############

    FROM =f"{SegmentTable_name} seg1"
    if SegmentTable_name == 'SegmentOct20':
        SELECT = "DISTINCT seg1.image_id, seg1.site_name_id, seg1.contentUrl, seg1.imagename, seg1.site_image_id, seg1.mongo_body_landmarks, seg1.mongo_face_landmarks, seg1.bbox"
    elif SegmentTable_name == 'SegmentBig_isface' or SegmentTable_name == 'SegmentBig_isnotface':
        # segmentbig does not have mongo booleans
        SELECT = "DISTINCT seg1.image_id, seg1.site_name_id, seg1.contentUrl, seg1.imagename, seg1.site_image_id, e.mongo_body_landmarks, e.mongo_face_landmarks, e.bbox"
        FROM += " JOIN Encodings e ON seg1.image_id = e.image_id"
    

    # FROM ="Encodings e"
    if BODYLMS or (BODYLMS and HANDLMS):
        QUERY = " "
        if SegmentTable_name == 'SegmentOct20':
            if REDO_BODYLMS_3D:
                QUERY += " seg1.mongo_body_landmarks IS NOT NULL and seg1.no_image IS NULL and seg1.mongo_body_landmarks_3D IS NULL"
            else:
                QUERY = " seg1.mongo_body_landmarks IS NULL and seg1.no_image IS NULL"
        elif SegmentTable_name == 'SegmentBig_isface':
            if REDO_BODYLMS_3D:
                QUERY += " e.mongo_body_landmarks IS NOT NULL and seg1.no_image IS NULL and e.mongo_body_landmarks_3D IS NULL"
            else:
                QUERY = " e.mongo_body_landmarks IS NULL "

        # if doing both BODYLMS and HANDLMS, query as if BODY, and also do HAND on those image_ids
        if TOPIC_ID:
            # FROM = " SegmentBig_isface seg1 "
            FROM += " LEFT JOIN ImagesTopics it ON seg1.image_id = it.image_id"
            SUBQUERY = f" AND it.topic_id IN {tuple(TOPIC_ID)} "
        else:
            SUBQUERY = " "
        if HelperTable_name:
            FROM += f" INNER JOIN {HelperTable_name} ht ON seg1.image_id = ht.image_id "
            QUERY += f" AND seg1.image_id > {START_IMAGE_ID}"


    elif HANDLMS:
        QUERY = " seg1.mongo_hand_landmarks IS NULL and seg1.no_image IS NULL"
        # SUBQUERY = " "
        # temp for testing one pose at a time
        if POSE_ID:
            SUBQUERY = f" AND seg1.image_id IN (SELECT ip.image_id FROM ImagesPoses128 ip WHERE ip.cluster_id = {POSE_ID})"
        if DO_INVERSE:
            SUBQUERY = f" AND seg1.image_id NOT IN (SELECT ip.image_id FROM ImagesPoses128 ip)"
        else:
            SUBQUERY = f" AND seg1.image_id IN (SELECT ip.image_id FROM ImagesPoses128 ip)"
            

    # SUBQUERY = f"(SELECT seg1.image_id FROM {SegmentTable_name} seg1 WHERE face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2)"
    # SUBQUERY = f"(SELECT seg1.image_id FROM {SegmentTable_name} seg1 WHERE face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2)"
    elif SEGMENT:
        QUERY = " "
        FROM = f"{SegmentTable_name} seg1 LEFT JOIN ImagesTopics it ON seg1.image_id = it.image_id"
        # SUBQUERY = f" seg1.mongo_body_landmarks IS NULL AND face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2 AND it.topic_id = {SEGMENT}"
        SUBQUERY = f" seg1.mongo_body_landmarks IS NULL AND it.topic_id = {SEGMENT}"

    elif HelperTable_name:
        FROM += f" INNER JOIN {HelperTable_name} ht ON seg1.image_id = ht.image_id LEFT JOIN ImagesTopics it ON seg1.image_id = it.image_id"
        QUERY = "e.body_landmarks IS NULL AND seg1.site_name_id NOT IN (1,4)"
        SUBQUERY = ""
    WHERE = f"{QUERY} {SUBQUERY}"

else:
    
    ############ KEYWORD SELECT #############
    SELECT = "DISTINCT i.image_id, i.site_name_id, i.contentUrl, i.imagename, e.encoding_id, i.site_image_id, e.face_landmarks, e.bbox"
    # FROM ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings e ON i.image_id = e.image_id"
    FROM ="Images i LEFT JOIN Encodings e ON i.image_id = e.image_id"
    # gettytest3
    # WHERE = "e.face_encodings68 IS NULL AND e.face_encodings IS NOT NULL"
    # production
    # WHERE = "e.is_face IS TRUE AND e.face_encodings68 IS NULL"
    if DO_OVER and FIND_NO_IMAGE:
        # find images with missing files
        # find all images that have not been processed, and have not been declared no image
        WHERE = f"e.encoding_id IS NULL AND i.no_image IS NULL AND e.two_noses is NULL AND i.site_name_id = {SITE_NAME_ID}"
    elif DO_OVER and not FIND_NO_IMAGE:
        # find all images that have been processed, but have no face found, and aren't no_image or two_noses
        # WHERE = f"e.encoding_id IS NOT NULL AND e.is_face = 0 AND e.mongo_encodings is NULL AND e.two_noses is NULL AND i.no_image IS NULL AND i.site_name_id = {SITE_NAME_ID}"
        # find all images that have been processed, have encodings, but no bbox to reprocess
        WHERE = f"e.encoding_id IS NOT NULL AND e.bbox IS NULL AND e.mongo_encodings =1 AND e.is_body IS NULL AND e.two_noses is NULL AND i.no_image IS NULL AND i.site_name_id = {SITE_NAME_ID}"
    else:
        WHERE = f"e.encoding_id IS NULL AND i.site_name_id = {SITE_NAME_ID}"
    if OVERRIDE_TOPIC: 
        FROM += " LEFT JOIN ImagesTopics_isnotface it ON i.image_id = it.image_id"
        WHERE += f" AND it.topic_id IN {tuple(OVERRIDE_TOPIC)} "
        # WHERE += f" AND i.topic_id = {OVERRIDE_TOPIC}"
    WHERE += f" AND i.image_id >  {START_IMAGE_ID}" if START_IMAGE_ID else ""
    WHERE += f" AND i.no_image IS NULL"
    QUERY = WHERE
    SUBQUERY = ""
    # AND i.age_id NOT IN (1,2,3,4)
    IS_SSD= False
    #########################################





## Gettytest3
# WHERE = "e.face_encodings IS NULL AND e.bbox IS NOT NULL"


##########################################


############# FROM A SEGMENT #############
# SegmentTable_name = 'June20segment123straight'
# FROM ="Images i LEFT JOIN Encodings e ON i.image_id = e.image_id"
# QUERY = "e.face_encodings68 IS NULL AND e.bbox IS NOT NULL AND e.image_id IN"
# # QUERY = "e.image_id IN"
# SUBQUERY = f"(SELECT seg1.image_id FROM {SegmentTable_name} seg1 )"
# WHERE = f"{QUERY} {SUBQUERY}"
# IS_SSD=True
##########################################


# platform specific credentials
io = DataIO(IS_SSD)
db = io.db
ROOT = io.ROOT 
# GPU OVERRIDE = 
io.NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES_GPU
# overriding DB for testing
# io.db["name"] = "gettytest3"

# --- Initialize MediaPipe objects with GPU delegate ---
FACE_DETECTOR_MODEL_PATH = '/Users/michaelmandiberg/Documents/GitHub/facemap/models/blaze_face_short_range.tflite'
FACE_LANDMARKER_MODEL_PATH = '/Users/michaelmandiberg/Documents/GitHub/facemap/models/face_landmarker.task'
HAND_LANDMARKER_MODEL_PATH = '/Users/michaelmandiberg/Documents/GitHub/facemap/models/hand_landmarker.task'
POSE_LANDMARKER_MODEL_PATH = '/Users/michaelmandiberg/Documents/GitHub/facemap/models/pose_landmarker_full.task'

# Base options for GPU
base_options_detector_gpu = python.BaseOptions(
    delegate=python.BaseOptions.Delegate.GPU,
    model_asset_path=FACE_DETECTOR_MODEL_PATH
)
# Face Detector options
face_detector_options = vision.FaceDetectorOptions(
    base_options=base_options_detector_gpu,
    running_mode=vision.RunningMode.IMAGE, # Specifies the input data type (IMAGE, VIDEO, or LIVE_STREAM)
    min_detection_confidence=0.7 # Minimum confidence score for a face to be considered detected
)

# # Face Landmarker options (formerly Face Mesh)
base_options_landmarker_gpu = python.BaseOptions(
    delegate=python.BaseOptions.Delegate.GPU,
    model_asset_path=FACE_LANDMARKER_MODEL_PATH
)
face_landmarker_options = vision.FaceLandmarkerOptions(
    base_options=base_options_landmarker_gpu,
    running_mode=vision.RunningMode.IMAGE, # Or .VIDEO, .LIVE_STREAM
)

# Hand Landmarker options
base_options_hand_gpu = python.BaseOptions(
    delegate=python.BaseOptions.Delegate.GPU,
    model_asset_path=HAND_LANDMARKER_MODEL_PATH
)
hand_landmarker_options = vision.HandLandmarkerOptions(
    base_options=base_options_hand_gpu,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.5, # Corresponds to min_detection_confidence in old API
    min_tracking_confidence=0.5
)

# Base options for GPU delegate for PoseLandmarker
base_options_pose_gpu = python.BaseOptions(
    delegate=python.BaseOptions.Delegate.GPU,
    model_asset_path=POSE_LANDMARKER_MODEL_PATH
)
pose_landmarker_options = vision.PoseLandmarkerOptions(
    base_options=base_options_pose_gpu,
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False # Set to True if you need segmentation masks
)

# Create the detector and landmarker objects outside the loop for efficiency
face_detector = vision.FaceDetector.create_from_options(face_detector_options)
face_landmarker = vision.FaceLandmarker.create_from_options(face_landmarker_options)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_landmarker_options)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_landmarker_options)

#not currently in use, so commented out
# mp_drawing = mp.solutions.drawing_utils
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

####### new imports and models ########
face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
YOLO_MODEL = YOLO("yolov8m.pt")   #MEDIUM

SMALL_MODEL = False
NUM_JITTERS= 1
###############

OBJ_CLS_LIST=[67,63,26,27,32] ## 
OBJ_CLS_NAME={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat'\
   , 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat'\
    , 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe'\
    , 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard'\
    , 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard'\
    , 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl'\
    , 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza'\
    , 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet'\
    , 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster'\
    , 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier'\
    , 79: 'toothbrush'}

## CREATING POSE OBJECT FOR SELFIE SEGMENTATION
## none of these are used in this script ##
## just to initialize the object ## 
# image_edge_multiplier = [1.5,1.5,2,1.5] # bigger portrait
# image_edge_multiplier_sm = [1.2, 1.2, 1.6, 1.2] # standard portrait
image_edge_multiplier_sm = [2.2, 2.2, 2.6, 2.2] # standard portrait
face_height_output = 500
motion = {"side_to_side": False, "forward_smile": True, "laugh": False, "forward_nosmile":  False, "static_pose":  False, "simple": False}
EXPAND = False
ONE_SHOT = True # take all files, based off the very first sort order.
JUMP_SHOT = False # jump to random file if can't find a run

sort = SortPose(motion, face_height_output, image_edge_multiplier_sm,EXPAND, ONE_SHOT, JUMP_SHOT, None, VERBOSE)


start = time.time()

def init_session():
    # init session
    global engine, Session, session
    # engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
    #                                 .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
    
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)

    # metadata = MetaData(engine)
    metadata = MetaData() # apparently don't pass engine
    Session = sessionmaker(bind=engine)
    session = Session()
    Base = declarative_base()

def close_session():
    session.close()
    engine.dispose()

def collect_the_garbage():
    if 'image' in locals():
        del image
    gc.collect()
    print("garbage collected")

def init_mongo():
    # init session
    # global engine, Session, session
    global mongo_client, mongo_db, mongo_collection, bboxnormed_collection, body_world_collection, mongo_hand_collection

    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    mongo_db = mongo_client["stock"]
    mongo_collection = mongo_db["encodings"]
    bboxnormed_collection = mongo_db["body_landmarks_norm"]
    body_world_collection = mongo_db["body_world_landmarks"]
    mongo_hand_collection = mongo_db["hand_landmarks"]

def close_mongo():
    mongo_client.close()    


# not sure if I'm using this
class Object:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

def get_hash_folders(filename):
    m = hashlib.md5()
    m.update(filename.encode('utf-8'))
    d = m.hexdigest()
    # csvWriter1.writerow(["https://upload.wikimedia.org/wikipedia/commons/"+d[0]+'/'+d[0:2]+'/'+filename])
    return d[0], d[0:2]

def read_csv(csv_file):
    with open(csv_file, encoding="utf-8", newline="") as in_file:
        reader = csv.reader(in_file, delimiter=",")
        next(reader)  # Header row

        for row in reader:
            yield row

def print_get_split(split):
    now = time.time()
    duration = now - split
    print(duration)
    return now

def ensure_image_cv2(image):
    # convert image back to numpy array if it's a mediapipe image
    if isinstance(image, mp.Image):
        image = image.numpy_view()
    # Ensure image is 3-channel (RGB) and uint8 for dlib
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = image.astype(np.uint8)
    return image

def ensure_image_mp(image):
    # convert image back to mediapipe image if it's a numpy array
    if isinstance(image, np.ndarray):
        image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))
    return image
def save_image_elsewhere(image, path):
    #saves a CV2 image elsewhere -- used in setting up test segment of images
    oldfolder = "newimages"
    newfolder = "testimages"
    outpath = path.replace(oldfolder, newfolder)
    try:
        print(outpath)
        cv2.imwrite(outpath, image)
        print("wrote")

    except:
        print("save_image_elsewhere couldn't write")

def save_image_by_path(image, sort, name):
    global sortfolder
    def mkExist(outfolder):
        isExist = os.path.exists(outfolder)
        if not isExist: 
            os.mkdir(outfolder)

    sortfolder_path = os.path.join(ROOT,sortfolder)
    outfolder = os.path.join(sortfolder_path,sort)
    outpath = os.path.join(outfolder, name)
    mkExist(sortfolder)
    mkExist(outfolder)

    try:
        print(outpath)

        cv2.imwrite(outpath, image)

    except:
        print("save_image_by_path couldn't write")




def insertignore(dataframe,table):

     # creating column list for insertion
     cols = "`,`".join([str(i) for i in dataframe.columns.tolist()])

     # Insert DataFrame recrds one by one.
     for i,row in dataframe.iterrows():
         sql = "INSERT IGNORE INTO `"+table+"` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
         engine.connect().execute(sql, tuple(row))


def insertignore_df(dataframe,table_name, engine):

     # Convert the DataFrame to a SQL table using pandas' to_sql method
     with engine.connect() as connection:
         dataframe.to_sql(name=table_name, con=connection, if_exists='append', index=False)


def insertignore_dict(dict_data,table_name):

     # # creating column list for insertion
     # # cols = "`,`".join([str(i) for i in dataframe.columns.tolist()])
     # cols = "`,`".join([str(i) for i in list(dict.keys())])
     # tup = tuple(list(dict.values()))

     # sql = "INSERT IGNORE INTO `"+table+"` (`" +cols + "`) VALUES (" + "%s,"*(len(tup)-1) + "%s)"
     # engine.connect().execute(sql, tup)

     # Create a SQLAlchemy Table object representing the target table
     target_table = Table(table_name, metadata, extend_existing=True, autoload_with=engine)

     # Insert the dictionary data into the table using SQLAlchemy's insert method
     with engine.connect() as connection:
         connection.execute(target_table.insert(), dict_data)

def selectORM(session, FILTER, LIMIT):
    query = session.query(Images.image_id, Images.site_name_id, Images.contentUrl, Images.imagename,
                          Encodings.encoding_id, Images.site_image_id, Encodings.face_landmarks, Encodings.bbox)\
        .join(ImagesKeywords, Images.image_id == ImagesKeywords.image_id)\
        .join(Keywords, ImagesKeywords.keyword_id == Keywords.keyword_id)\
        .outerjoin(Encodings, Images.image_id == Encodings.image_id)\
        .filter(*FILTER)\
        .limit(LIMIT)

    results = query.all()
    results_dict = [dict(row) for row in results]
    return results_dict

def selectSQL(start_id):
    init_session()
    if start_id:
        # if FROM contains "seg1" or "segment", then assign SegmentTable_name to image_id
        if "seg1" in FROM or "segment" in FROM:
            image_id_table = SegmentTable_name
        else:
            image_id_table = "i"
        selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {QUERY} AND {image_id_table}.image_id > {start_id} {SUBQUERY} LIMIT {str(LIMIT)};"
    else:
        selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)};"

    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    close_session()
    return(resultsjson)

def slice_mp_image(image, bbox):
    slice_np = image.numpy_view()[bbox["top"]:bbox["bottom"], bbox["left"]:bbox["right"]]
    # Create a new mediapipe.Image from the cropped numpy array
    slice_np_uint8 = slice_np.astype(np.uint8)
    slice_mp_image = mp.Image(image_format=image.image_format, data=slice_np_uint8)
    return slice_mp_image

def get_bbox(faceDet, height, width):
    bbox = {}
    bbox_obj = faceDet.location_data.relative_bounding_box
    xy_min = _normalized_to_pixel_coordinates(bbox_obj.xmin, bbox_obj.ymin, width,height)
    xy_max = _normalized_to_pixel_coordinates(bbox_obj.xmin + bbox_obj.width, bbox_obj.ymin + bbox_obj.height,width,height)
    if xy_min and xy_max:
        # TOP AND BOTTOM WERE FLIPPED 
        # both in xy_min assign, and in face_mesh.process(image[np crop])
        left,top =xy_min
        right,bottom = xy_max
        bbox={"left":left,"right":right,"top":top,"bottom":bottom}
    else:
        print("no results???")
    return(bbox)


def convert_landmarker_to_facemesh(landmarker_result: vision.FaceLandmarkerResult):
    """
    Converts a mediapipe.tasks.python.vision.FaceLandmarkerResult object
    to mimic the structure of the results object from the older
    mp.solutions.face_mesh.FaceMesh().process() method.

    Args:
        landmarker_result (vision.FaceLandmarkerResult): The result object
                                                        from face_landmarker.detect().

    Returns:
        types.SimpleNamespace: A mock results object with 'multi_face_landmarks',
                            'multi_face_blendshapes', and 'multi_face_transformations'
                            attributes, structured like the old API.
    """
    # Create a mock results object
    results = types.SimpleNamespace()
    results.multi_face_landmarks = []
    results.multi_face_blendshapes = []
    results.multi_face_transformations = []

    if landmarker_result.face_landmarks:
        for face_lms_list in landmarker_result.face_landmarks:
            # Create a NormalizedLandmarkList for each face
            normalized_landmark_list = landmark_pb2.NormalizedLandmarkList()
            # Convert each item to a NormalizedLandmark protobuf message
            for lm in face_lms_list:
                normalized_landmark = landmark_pb2.NormalizedLandmark()
                normalized_landmark.x = lm.x
                normalized_landmark.y = lm.y
                normalized_landmark.z = lm.z
                if hasattr(lm, "visibility"):
                    normalized_landmark.visibility = lm.visibility
                if hasattr(lm, "presence"):
                    normalized_landmark.presence = lm.presence
                normalized_landmark_list.landmark.append(normalized_landmark)
            results.multi_face_landmarks.append(normalized_landmark_list)

    if landmarker_result.face_blendshapes:
        for blendshapes_list in landmarker_result.face_blendshapes:
            # Create a ClassificationList (which is how blendshapes were structured in old API)
            classification_list = classification_pb2.ClassificationList()
            for category in blendshapes_list:
                # Create a Classification object for each blendshape category
                classification = classification_pb2.Classification(
                    index=category.index,
                    score=category.score,
                    label=category.category_name # Use category_name as label
                )
                classification_list.classification.append(classification)
            # The old API's multi_face_blendshapes was a list of ClassificationList objects
            results.multi_face_blendshapes.append(classification_list)

    if landmarker_result.facial_transformation_matrixes:
        # The transformation matrices are already numpy arrays in the new API,
        # and the old API also expected a list of numpy arrays.
        results.multi_face_transformations = landmarker_result.facial_transformation_matrixes

    return results

def find_face(image, df):
    # image is SRGBA mp.Image (for mp task GPU implementation)
    image = ensure_image_mp(image)
    # find_face_start = time.time()
    number_of_detections = 0
    is_face = False
    is_face_no_lms = None

    # Perform face detection
    detection_result = face_detector.detect(image)

    if detection_result.detections:
        number_of_detections = len(detection_result.detections)
        print("---------------- >>>>>>>>>>>>>>>>> number_of_detections", number_of_detections)

        # Assuming you take the first detected face for simplicity
        faceDet = detection_result.detections[0]
        # The bounding box format from FaceDetector is different.
        # You'll need to convert it to your bbox format if `get_bbox` expects something specific.
        bbox_mp = faceDet.bounding_box
        bbox = {
            "left": bbox_mp.origin_x,
            "top": bbox_mp.origin_y,
            "right": bbox_mp.origin_x + bbox_mp.width,
            "bottom": bbox_mp.origin_y + bbox_mp.height
        }
        if bbox:
            # take just the bbox slice of the mp.Image and detect on that slice
            mp_image_face = slice_mp_image(image, bbox)
            landmarker_result = face_landmarker.detect(mp_image_face)
 
            #read any image containing a face
            if landmarker_result.face_landmarks:
                
                #construct pose object to solve pose
                is_face = True
                pose = SelectPose(image)

                results = convert_landmarker_to_facemesh(landmarker_result)

                #get landmarks
                faceLms = pose.get_face_landmarks(results,bbox)

                #calculate base data from landmarks
                pose.calc_face_data(faceLms)

                # get angles, using r_vec property stored in class
                # angles are meta. there are other meta --- size and resize or something.
                angles = pose.rotationMatrixToEulerAnglesToDegrees()
                mouth_gap = pose.get_mouth_data(faceLms)

                if is_face:
                    encodings = calc_encodings(image, faceLms,bbox) ## changed parameters
                    print(">> find_face SPLIT >> calc_encodings")

                #df.at['1', 'is_face_distant'] = is_face_distant
                bbox_json = json.dumps(bbox, indent = 4) 

                df.at['1', 'face_x'] = angles[0]
                df.at['1', 'face_y'] = angles[1]
                df.at['1', 'face_z'] = angles[2]
                df.at['1', 'mouth_gap'] = mouth_gap
                df.at['1', 'face_landmarks'] = pickle.dumps(faceLms)
                df.at['1', 'bbox'] = bbox_json
                if SMALL_MODEL is True:
                    df.at['1', 'face_encodings'] = pickle.dumps(encodings)
                else:
                    df.at['1', 'face_encodings68'] = pickle.dumps(encodings)
            else:
                print("+++++++++++++++++  YES FACE but NO FACE LANDMARKS +++++++++++++++++++++")
                image_id = df.at['1', 'image_id']
                is_face_no_lms = True
        else:
            print("+++++++++++++++++  NO BBOX DETECTED +++++++++++++++++++++")
        
    else:
        print("+++++++++++++++++  NO FACE DETECTED +++++++++++++++++++++")
        number_of_detections = 0
        image_id = df.at['1', 'image_id']
        no_image_name = f"no_face_landmarks_{image_id}.jpg"
        is_face_no_lms = False
    df.at['1', 'is_face'] = is_face
    df.at['1', 'is_face_no_lms'] = is_face_no_lms

    return df, number_of_detections



def calc_encodings(image, faceLms,bbox):## changed parameters and rebuilt
    image = ensure_image_cv2(image)
    def get_dlib_all_points(landmark_points):
        raw_landmark_set = []
        for index in landmark_points:                       ######### CORRECTION: landmark_points_5_3 is the correct one for sure
            # print(faceLms[index].x)

            # second attempt, tries to project faceLms from bbox origin
            x = int(faceLms.landmark[index].x * width + bbox["left"])
            y = int(faceLms.landmark[index].y * height + bbox["top"])

            landmark_point=dlib.point([x,y])
            raw_landmark_set.append(landmark_point)
        dlib_all_points=dlib.points(raw_landmark_set)
        return dlib_all_points
        # print("all_points", all_points)
        # print(bbox)


    # second attempt, tries to project faceLms from bbox origin
    width = (bbox["right"]-bbox["left"])
    height = (bbox["bottom"]-bbox["top"])

    landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,
                      288,323,454,389,71,63,105,66,107,336,296,334,293,
                      301,168,197,5,4,75,97,2,326,305,33,160,158,133,
                      153,144,362,385,387,263,373,380,61,39,37,0,267,
                      269,291,405,314,17,84,181,78,82,13,312,308,317,
                      14,87]
                      
    landmark_points_5 = [ 263, #left eye away from centre
                       362, #left eye towards centre
                       33,  #right eye away from centre
                       133, #right eye towards centre
                        2 #bottom of nose tip 
                    ]
                    
    if SMALL_MODEL is True:landmark_points=landmark_points_5
    else:landmark_points=landmark_points_68
    dlib_all_points68 = get_dlib_all_points(landmark_points_68)

    # ymin ("top") would be y value for top left point.
    bbox_rect= dlib.rectangle(left=bbox["left"], top=bbox["top"], right=bbox["right"], bottom=bbox["bottom"])

    if (dlib_all_points68 is None) or (bbox is None):return 
    
    full_object_detection68=dlib.full_object_detection(bbox_rect,dlib_all_points68)
    encodings68=face_encoder.compute_face_descriptor(image, full_object_detection68, num_jitters=NUM_JITTERS)

    encodings = encodings68
    return np.array(encodings).tolist()

def convert_landmarker_to_bodyLms(detection_result: vision.PoseLandmarkerResult):
    """
    Converts a mediapipe.tasks.python.vision.PoseLandmarkerResult object
    to mimic the structure of the results object (bodyLms) from the older
    mp.solutions.pose.Pose().process() method.

    Args:
        detection_result (vision.PoseLandmarkerResult): The result object
                                                        from pose_landmarker.detect().

    Returns:
        types.SimpleNamespace: A mock results object with 'pose_landmarks' and
                            'pose_world_landmarks' attributes, structured like the old API.
    """
    bodyLms = types.SimpleNamespace()
    bodyLms.pose_landmarks = [] # Initialize as a list
    bodyLms.pose_world_landmarks = [] # Initialize as a list
    # segmentation_mask is not included as enable_segmentation was False in old code

    if detection_result.pose_landmarks:
        normalized_landmark_list = landmark_pb2.NormalizedLandmarkList()
        # Iterate and append each landmark individually to avoid TypeError
        for lm in detection_result.pose_landmarks[0]:
            new_lm = landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility, presence=lm.presence)
            normalized_landmark_list.landmark.append(new_lm)
        bodyLms.pose_landmarks.append(normalized_landmark_list) # Append the protobuf object to the list

    if detection_result.pose_world_landmarks:
        world_landmark_list = landmark_pb2.LandmarkList()
        # Iterate and append each landmark individually to avoid TypeError
        for lm in detection_result.pose_world_landmarks[0]:
            new_lm = landmark_pb2.Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility, presence=lm.presence)
            world_landmark_list.landmark.append(new_lm)
        bodyLms.pose_world_landmarks.append(world_landmark_list) # Append the protobuf object to the list

    return bodyLms

def convert_landmarker_to_handLms(detection_result: vision.HandLandmarkerResult):
    """
    Converts a mediapipe.tasks.python.vision.HandLandmarkerResult object
    to mimic the structure of the results object from the older
    mp.solutions.hands.Hands().process() method.

    Args:
        detection_result (vision.HandLandmarkerResult): The result object
                                                        from hand_landmarker.detect().

    Returns:
        types.SimpleNamespace: A mock results object with 'multi_hand_landmarks',
                            'multi_hand_world_landmarks', and 'multi_handedness'
                            attributes, structured like the old API.
    """
    results = types.SimpleNamespace()
    results.multi_hand_landmarks = []
    results.multi_hand_world_landmarks = []
    results.multi_handedness = []

    if detection_result.hand_landmarks:
        for idx, hand_lms_list in enumerate(detection_result.hand_landmarks):
            # Convert hand_landmarks (List[NormalizedLandmark]) to NormalizedLandmarkList
            normalized_landmark_list = landmark_pb2.NormalizedLandmarkList()
            # Iterate and append each landmark individually to avoid TypeError
            for lm in hand_lms_list:
                new_lm = landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility, presence=lm.presence)
                normalized_landmark_list.landmark.append(new_lm)
            results.multi_hand_landmarks.append(normalized_landmark_list)

            # Convert hand_world_landmarks (List[Landmark]) to LandmarkList
            if detection_result.hand_world_landmarks and idx < len(detection_result.hand_world_landmarks):
                world_landmark_list = landmark_pb2.LandmarkList()
                # Iterate and append each landmark individually to avoid TypeError
                for lm in detection_result.hand_world_landmarks[idx]:
                    new_lm = landmark_pb2.Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility, presence=lm.presence)
                    world_landmark_list.landmark.append(new_lm)
                results.multi_hand_world_landmarks.append(world_landmark_list)
            else:
                results.multi_hand_world_landmarks.append(landmark_pb2.LandmarkList())


            # Convert handedness (List[Category]) to ClassificationList
            if detection_result.handedness and idx < len(detection_result.handedness):
                classification_list = classification_pb2.ClassificationList()
                # Iterate and append each category individually to avoid TypeError
                for category in detection_result.handedness[idx]:
                    classification = classification_pb2.Classification(
                        index=category.index,
                        score=category.score,
                        label=category.category_name
                    )
                    classification_list.classification.append(classification)
                results.multi_handedness.append(classification_list)
            else:
                results.multi_handedness.append(classification_pb2.ClassificationList())

    return results

def find_body(image):



    if VERBOSE: print("find_body")
    mp_image = ensure_image_mp(image)  # Ensure image is in the correct format for MediaPipe
    is_body = body_landmarks = body_world_landmarks = None # Initialize world_landmarks
    try:
        # Process the image to detect pose landmarks
        detection_result = pose_landmarker.detect(mp_image)

        # PoseLandmarkerResult has 'pose_landmarks' and 'pose_world_landmarks' directly
        # These are List[NormalizedLandmark] and List[Landmark] respectively,
        # where each list contains the 33 landmarks for the detected pose.
        # If no pose is detected, these lists will be empty.
        if detection_result.pose_landmarks:

            # print("got bodyLms", detection_result )
            is_body = True
            # The pose_landmarks and pose_world_landmarks are already lists of landmarks
            # for the *single* detected pose (or the first one if multiple were allowed).
            # If you configured num_poses > 1, you'd iterate detection_result.pose_landmarks
            # and detection_result.pose_world_landmarks as lists of lists.
            # With default num_poses=1, they are directly the list of 33 landmarks.
            body_landmarks = detection_result.pose_landmarks[0] # Access the first (and likely only) pose
            body_world_landmarks = detection_result.pose_world_landmarks[0] # Access the first (and likely only) pose
        else:
            print("No body detected.")

    except Exception as e:
        print(f"[find_body] An error occurred: {e}")

    return is_body, body_landmarks, body_world_landmarks

def find_hands(image, pose):    
    def extract_hand_landmarks_new_api(detection_result):
        """
        Extracts hand landmarks and related data from the new MediaPipe HandLandmarkerResult.
        This function produces a data structure similar to the old API's output.

        Args:
            detection_result (mediapipe.tasks.python.vision.HandLandmarkerResult):
                The result object from hand_landmarker.detect().

        Returns:
            list: A list of dictionaries, where each dictionary contains:
                - "image_landmarks": List of (x, y, z) tuples for image coordinates.
                - "world_landmarks": List of (x, y, z) tuples for world coordinates.
                - "handedness": String label ("Left" or "Right").
                - "confidence_score": Float confidence score for handedness.
        """
        hands_data = []

        if detection_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                image_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                world_landmarks = [(lm.x, lm.y, lm.z) for lm in detection_result.hand_world_landmarks[idx]]

                handedness_category = detection_result.handedness[idx][0]
                confidence_score = handedness_category.score
                hand_label = handedness_category.category_name

                hand_data = {
                    "image_landmarks": image_landmarks,
                    "world_landmarks": world_landmarks,
                    "handedness": hand_label,
                    "confidence_score": confidence_score
                }
                hands_data.append(hand_data)

        return hands_data

    mp_image = ensure_image_mp(image)  # Ensure image is in the correct format for MediaPipe

    try:
        detection_result = hand_landmarker.detect(mp_image)
        if not detection_result.hand_landmarks:
            return None, None
        else:
            hand_landmarks_list = extract_hand_landmarks_new_api(detection_result)
            return True, hand_landmarks_list

    except Exception as e:
        print(f"[find_hands] An error occurred: {e}")
        print(f"[find_hands] this item failed: {mp_image}")
        return None, None

def capitalize_directory(path):
    dirname, filename = os.path.split(path)
    parts = dirname.split('/')
    capitalized_parts = [part if i < len(parts) - 2 else part.upper() for i, part in enumerate(parts)]
    capitalized_dirname = '/'.join(capitalized_parts)
    return os.path.join(capitalized_dirname, filename)


def process_image_enc_only(task):
    # print("process_image_enc_only")

    encoding_id = task[0]
    faceLms = task[2]
    bbox = io.unstring_json(task[3])
    cap_path = capitalize_directory(task[1])
    print(cap_path)
    try:
        image = cv2.imread(cap_path)  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
      
    except:
        print(f"[process_image]this item failed: {task}")

    if image is not None and image.shape[0]>MINSIZE and image.shape[1]>MINSIZE:
        face_encodings = calc_encodings(image, faceLms,bbox)

    else:
        print('toooooo smallllll')

    pickled_encodings = pickle.dumps(face_encodings)
    df = pd.DataFrame(columns=['encoding_id'])
    df.at['1', 'encoding_id'] = encoding_id

    if SMALL_MODEL is True and NUM_JITTERS == 1:
        df.at['1', 'face_encodings'] = pickled_encodings
        sql = """
        UPDATE Encodings SET face_encodings = :face_encodings
        WHERE encoding_id = :encoding_id
        """
    elif SMALL_MODEL is False and NUM_JITTERS == 1:
        print("updating face_encodings68")
        df.at['1', 'face_encodings68'] = pickled_encodings
        sql = """
        UPDATE Encodings SET face_encodings68 = :face_encodings68
        WHERE encoding_id = :encoding_id
        """
    elif SMALL_MODEL is True and NUM_JITTERS == 3:
        df.at['1', 'face_encodings_J3'] = pickled_encodings
        sql = """
        UPDATE Encodings SET face_encodings_J3 = :face_encodings_J3
        WHERE encoding_id = :encoding_id
        """
    elif SMALL_MODEL is False and NUM_JITTERS == 3:
        df.at['1', 'face_encodings68_J3'] = pickled_encodings
        sql = """
        UPDATE Encodings SET face_encodings68_J3 = :face_encodings68_J3
        WHERE encoding_id = :encoding_id
        """
    elif SMALL_MODEL is True and NUM_JITTERS == 5:
        df.at['1', 'face_encodings_J5'] = pickled_encodings
        sql = """
        UPDATE Encodings SET face_encodings_J5 = :face_encodings_J5
        WHERE encoding_id = :encoding_id
        """
    elif SMALL_MODEL is False and NUM_JITTERS == 5:
        df.at['1', 'face_encodings68_J5'] = pickled_encodings
        sql = """
        UPDATE Encodings SET face_encodings68_J5 = :face_encodings68_J5
        WHERE encoding_id = :encoding_id
        """


    try:
        with engine.begin() as conn:
            params = df.to_dict("records")
            conn.execute(text(sql), params)

        print("updated:",str(encoding_id))
    except OperationalError as e:
        print(e)

def process_image_find_body_subroutine(image_id, image, bbox):
    is_body = body_landmarks = body_world_landmarks = n_landmarks = face_height = nose_pixel_pos = None
    # Check if body landmarks are already in the normalized collection
    existing_norm = bboxnormed_collection.find_one({"image_id": image_id})
    existing_worldbody = body_world_collection.find_one({"image_id": image_id})
    existing_body = mongo_collection.find_one({"image_id": image_id})
    if existing_norm is not None: n_landmarks = pickle.loads(existing_norm["nlms"])
    if existing_body is not None and "body_landmarks" in existing_body: 
        try:
            # this is to check to see if there are body lms, or if it is just face lms
            body_landmarks = pickle.loads(existing_body["body_landmarks"])
        except:
            print("no body_landmarks", image_id)
    if existing_worldbody is not None and "body_world_landmarks" in existing_worldbody: 
        body_world_landmarks = pickle.loads(existing_worldbody["body_world_landmarks"])
    if VERBOSE:
        print("n_landmarks", image_id, n_landmarks)
        print("body_landmarks", image_id,  body_landmarks)
        print("body_world_landmarks", image_id, body_world_landmarks)
    if None not in (body_landmarks, body_world_landmarks, n_landmarks, bbox):
        print(f"body_landmarks, body_world_landmarks, n_landmarks already exist for image_id: {image_id}")
        is_body = True
    elif None not in (body_landmarks, body_world_landmarks) and None in (n_landmarks, bbox):
        print(f"body_landmarks, body_world_landmarks BUT NO BBOX SO NO NLMS already exist for image_id: {image_id}")
        is_body = True
    else:
        # existing_worldbody = body_world_collection.find_one({"image_id": image_id})
        # print("existing_worldbody", existing_worldbody)
        is_body, body_lms_new_api, body_world_lms_new_api = find_body(image)

        
        # CONVERT HERE
        if is_body is not None:
            pose_detection_result_conversion_object = types.SimpleNamespace(
                pose_landmarks=[body_lms_new_api],
                pose_world_landmarks=[body_world_lms_new_api]
            )
            converted_body_landmarks = convert_landmarker_to_bodyLms(pose_detection_result_conversion_object)
            if converted_body_landmarks.pose_landmarks:
                body_landmarks = converted_body_landmarks.pose_landmarks[0]
            else:
                body_landmarks = None
            if converted_body_landmarks.pose_world_landmarks:
                body_world_landmarks = converted_body_landmarks.pose_world_landmarks[0]
            else:
                body_world_landmarks = None
        else:
            if VERBOSE: print("No body landmarks found, setting body_landmarks and body_world_landmarks to None")
            body_landmarks = None
            body_world_landmarks = None

        image = ensure_image_cv2(image)  # sort expects cv2 image
        if is_body and bbox and body_landmarks is not None:
            ### NORMALIZE LANDMARKS ###
            nose_pixel_pos = sort.set_nose_pixel_pos(body_landmarks, image.shape)
            if VERBOSE: print("nose_pixel_pos", nose_pixel_pos)
            face_height = sort.convert_bbox_to_face_height(bbox)
            n_landmarks = sort.normalize_landmarks(body_landmarks, nose_pixel_pos, face_height, image.shape)
            # print("n_landmarks", n_landmarks)
            # sort.insert_n_landmarks(bboxnormed_collection, target_image_id, n_landmarks)
            ### Save normalized landmarks to MongoDB
            bboxnormed_collection.update_one(
                {"image_id": image_id},
                {"$set": {"nlms": pickle.dumps(n_landmarks)}},
                upsert=True
            )
            if VERBOSE: print(f"Normalized landmarks stored for image_id: {image_id}")
        elif is_body and not bbox:
            if VERBOSE: print("Body landmarks found but no bbox, no normalization")
            n_landmarks = None
        else:
            print("No body landmarks found, though is_body is " + str(is_body))
            n_landmarks = None
    if VERBOSE:
        print("n_landmarks", image_id, n_landmarks)
        print("body_landmarks", image_id,  body_landmarks)
        print("body_world_landmarks", image_id, body_world_landmarks)
    return is_body, n_landmarks, body_landmarks, body_world_landmarks, face_height, nose_pixel_pos

def process_image_normalize_object_bbox(bbox_dict, nose_pixel_pos, face_height, image_shape):
    ### normed object bbox
    for OBJ_CLS_ID in OBJ_CLS_LIST:
        if VERBOSE: print("OBJ_CLS_ID to norm", OBJ_CLS_ID)
        bbox_key = "bbox"
        # conf_key = "conf_{0}".format(OBJ_CLS_ID)
        bbox_n_key = "bbox_{0}_norm".format(OBJ_CLS_ID)
        if VERBOSE: print("OBJ_CLS_ID", OBJ_CLS_ID)
        try: 
            if VERBOSE: print("trying to get bbox", OBJ_CLS_ID)
            bbox_dict_value = bbox_dict[OBJ_CLS_ID]["bbox"]
            bbox_dict_value = io.unstring_json(bbox_dict_value)
        except: 
            if VERBOSE: print("no bbox", OBJ_CLS_ID)
            bbox_dict_value = None
        if bbox_dict_value and not nose_pixel_pos:
            print("normalized bbox but no nose_pixel_pos for ")
        elif bbox_dict_value:
            if VERBOSE: print("setting normed bbox for OBJ_CLS_ID", OBJ_CLS_ID)
            if VERBOSE: print("bbox_dict_value", bbox_dict_value)
            if VERBOSE: print("bbox_n_key", bbox_n_key)

            n_phone_bbox=sort.normalize_phone_bbox(bbox_dict_value,nose_pixel_pos,face_height,image_shape)
            bbox_dict[bbox_n_key]=n_phone_bbox
            print("normed bbox", bbox_dict[bbox_n_key])
        else:
            pass
            if VERBOSE: print(f"NO {bbox_key} for",)
    return bbox_dict

def process_image_hands_subroutine(image_id, image):

    existing_hand = mongo_hand_collection.find_one({"image_id": image_id})
    if existing_hand:
        if VERBOSE:print(f"hand landmarks already exist for image_id: {image_id}")
        update_hand = False
        is_hands = True
        pose = hand_landmarks = None
    else:
        # do hand stuff
        pose = SelectPose(image)
        is_hands, hand_landmarks = find_hands(image, pose)
        if not is_hands:
            if VERBOSE:print(" ------ >>>>>  NO HANDS for ", image_id)
            update_hand = False
        else:
            if VERBOSE:print(" ------ >>>>>  YES HANDS for ", image_id)
            update_hand = True
    return pose, is_hands, hand_landmarks, update_hand

def save_body_hands_mysql_and_mongo(session, image_id, image, bbox_dict, body_landmarks, body_world_landmarks, n_landmarks, hue_bb, lum_bb, sat_bb, val_bb, lum_torso_bb, hue, lum, sat, val, lum_torso, is_left_shoulder, is_right_shoulder, selfie_bbox, is_body, mongo_body_landmarks, is_hands, update_hand, hand_landmarks, pose, is_feet):
    # if REDO_BODYLMS_3D, it will only add body_world_landmarks to mongo and updated SQL mongo_body_landmarks_3D and is_feet
    # to add in 0's so these don't reprocesses repeatedly
    
    if body_world_landmarks is not None:
        mongo_body_landmarks_3D = True
    else:
        if VERBOSE: print("No body_world_landmarks detected.", body_world_landmarks)
        mongo_body_landmarks_3D = False

    if is_body is None or is_body is False: 
        is_body = False 
        mongo_body_landmarks = False
    else: 
        is_body = True
        mongo_body_landmarks = True
    if is_hands is None or is_hands is False: is_hands = False
    else: is_hands = True
    if is_feet is None or is_feet is False: is_feet = False
    else: is_feet = True

    print("going to save", image_id, "is_body", is_body, "is_hands", is_hands, "is_feet", is_feet, "mongo_body_landmarks", mongo_body_landmarks, "mongo_body_landmarks_3D", mongo_body_landmarks_3D)
    if mongo_body_landmarks is not None and not REDO_BODYLMS_3D:
        # skip this if REDO_BODYLMS_3D
        # get encoding_id for mongo insert_one
        encoding_id_results = session.query(Encodings.encoding_id).filter(Encodings.image_id == image_id).first()
        encoding_id = encoding_id_results[0]

        # assigning mongo boolean based on presence of n_landmarks
        if n_landmarks is not None: mongo_body_landmarks_norm = 1
        else: mongo_body_landmarks_norm = 0
        
        if VERBOSE: print("starting to save body landmarks for ", image_id)
        session.query(Encodings).filter(Encodings.image_id == image_id).update({
            Encodings.is_body: is_body,
            # Encodings.body_landmarks: body_landmarks
            Encodings.mongo_body_landmarks: is_body,
            Encodings.mongo_body_landmarks_norm: mongo_body_landmarks_norm # is it saving all of that norm lms data to sql???
        }, synchronize_session=False)

        session.query(SegmentTable).filter(SegmentTable.image_id == image_id).update({
            SegmentTable.mongo_body_landmarks: is_body,
            SegmentTable.mongo_body_landmarks_norm: mongo_body_landmarks_norm
        }, synchronize_session=False)

        # MySQL
        ### save image.shape to Images.h and Images.w
        session.query(Images).filter(Images.image_id == image_id).update({
            Images.h: image.shape[0],
            Images.w: image.shape[1]
        }, synchronize_session=False)
        if VERBOSE: print("body landmarks added to MySQL session for", image_id)
        if VERBOSE: print("bbox_dict", bbox_dict, "hue", hue, "for image_id", image_id)
        if bbox_dict is not None:
            ### save bbox_dict (including normed bbox) to PhoneBbox
            for OBJ_CLS_ID in OBJ_CLS_LIST:
                bbox_n_key = f"bbox_{OBJ_CLS_ID}_norm"
                # print(bbox_dict)
                if bbox_dict[OBJ_CLS_ID]["bbox"]:
                    # Create a new PhoneBbox entry
                    new_entry_phonebbox = PhoneBbox(image_id=image_id)

                    if VERBOSE: print(f"bbox_dict[OBJ_CLS_ID][bbox]: {bbox_dict[OBJ_CLS_ID]['bbox']}")
                    if VERBOSE: print(f"bbox_dict[OBJ_CLS_ID][conf]: {bbox_dict[OBJ_CLS_ID]['conf']}")
                    if VERBOSE: print("bbox_n_key:", bbox_n_key)
                    if VERBOSE: print(f"bbox_dict[bbox_n_key]: {bbox_dict[bbox_n_key]}")

                    # Set attributes
                    setattr(new_entry_phonebbox, f"bbox_{OBJ_CLS_ID}", bbox_dict[OBJ_CLS_ID]["bbox"])
                    setattr(new_entry_phonebbox, f"conf_{OBJ_CLS_ID}", bbox_dict[OBJ_CLS_ID]["conf"])
                    try:
                        setattr(new_entry_phonebbox, bbox_n_key, bbox_dict[bbox_n_key])
                    except:
                        print(f"Error setting {bbox_n_key} for {image_id}")
                    # Add the new entry to the session
                    session.merge(new_entry_phonebbox)
                    
                    print(f"New Bbox {OBJ_CLS_ID} session entry for image_id {image_id} created successfully.")
                else:
                    pass
                    if VERBOSE: print(f"No bbox for {OBJ_CLS_ID} in image_id {image_id}")

        if hue is not None:
            ### ImageBackground

            # Create a new ImagesBackground entry
            new_entry_ib = ImagesBackground(image_id=image_id)

            # bbox
            new_entry_ib.hue_bb = hue_bb
            new_entry_ib.lum_bb = lum_bb
            new_entry_ib.sat_bb = sat_bb
            new_entry_ib.val_bb = val_bb
            new_entry_ib.lum_torso_bb = lum_torso_bb
            # no bbox
            new_entry_ib.hue = hue
            new_entry_ib.lum = lum
            new_entry_ib.sat = sat
            new_entry_ib.val = val
            new_entry_ib.lum_torso = lum_torso   
            # selfie stuff
            new_entry_ib.is_left_shoulder = is_left_shoulder
            new_entry_ib.is_right_shoulder = is_right_shoulder
            new_entry_ib.selfie_bbox = selfie_bbox

            # Add the new entry to the session
            session.merge(new_entry_ib)

        if VERBOSE: print("made it through the bbox ib gauntlet for", image_id)

        ### save regular landmarks to mongo, only if lms exist         
        if body_landmarks:
            if VERBOSE: print("storing body_landmarks for image_id", image_id, body_landmarks)
            existing_entry = mongo_collection.find_one({"image_id": image_id})
            # if VERBOSE: print("encoding_id", encoding_id, "image_id", image_id)
            if existing_entry:
                mongo_collection.update_one(
                    {"image_id": image_id},
                    {"$set": {"body_landmarks": pickle.dumps(body_landmarks)}}
                )
                print("----------- >>>>>>>>   mongo body_landmarks updated:", image_id)
            else:
                # get encoding_id for mongo insert_one
                encoding_id_results = session.query(Encodings.encoding_id).filter(Encodings.image_id == image_id).first()
                encoding_id = encoding_id_results[0]
                if VERBOSE: print("encoding_id", encoding_id, "image_id", image_id)
                mongo_collection.insert_one(
                    {"image_id": image_id, "encoding_id":encoding_id, "body_landmarks": pickle.dumps(body_landmarks)}
                )
                print("----------- >>>>>>>>   mongo body_landmarks inserted:", image_id)

        ### save normalized landmarks, will always be None if reprocessing, because no nose_pixel_pos?        
        if n_landmarks:
            if VERBOSE: print("because n_landmarks, storing them", n_landmarks)
            existing_entry = bboxnormed_collection.find_one({"image_id": image_id})
            if existing_entry:
                bboxnormed_collection.update_one(
                    {"image_id": image_id},
                    {"$set": {"nlms": pickle.dumps(n_landmarks)}},
                    upsert=True
                )
                print("----------- >>>>>>>>   mongo n_landmarks updated:", image_id)
            else:
                bboxnormed_collection.insert_one(
                    {"image_id": image_id, "nlms": pickle.dumps(n_landmarks)}
                )
                print("----------- >>>>>>>>   mongo n_landmarks inserted:", image_id)

    # save hand landmarks
    if VERBOSE: print("storing hand_landmarks for image_id", image_id)

    if not REDO_BODYLMS_3D:
        session.query(Encodings).filter(Encodings.image_id == image_id).update({
            # Encodings.body_landmarks: body_landmarks
            Encodings.mongo_hand_landmarks: is_hands,
        }, synchronize_session=False)

        session.query(SegmentTable).filter(SegmentTable.image_id == image_id).update({
            SegmentTable.mongo_hand_landmarks: is_hands,
        }, synchronize_session=False)

        if update_hand:
            try:
                pose.store_hand_landmarks(image_id, hand_landmarks, mongo_hand_collection)
                print("----------- >>>>>>>>   mongo hand_landmarks updated:", image_id)
            except:
                print("----------- XXXXXXXX   mongo hand_landmarks FAILED TO UPDATE:", image_id)

    # save is_feet and mongo_body_landmarks_3D regardless of REDO_BODYLMS_3D
    ### Save normalized landmarks to MongoDB
    body_world_collection.update_one(
        {"image_id": image_id},
        {"$set": {"body_world_landmarks": pickle.dumps(body_world_landmarks)}},
        upsert=True
    )
    if VERBOSE: print(f"body_world_landmarks stored for image_id: {image_id}")
    session.query(Encodings).filter(Encodings.image_id == image_id).update({
        # Encodings.body_landmarks: body_landmarks
        Encodings.is_feet: is_feet,
        Encodings.mongo_body_landmarks_3D: mongo_body_landmarks_3D,
    }, synchronize_session=False)

    session.query(SegmentTable).filter(SegmentTable.image_id == image_id).update({
        SegmentTable.is_feet: is_feet,
        SegmentTable.mongo_body_landmarks_3D: mongo_body_landmarks_3D,
    }, synchronize_session=False)


    # # store image_id in NMLImages table - is_nml_db is boolean. 0 means mac, 1 means nml pc db
    if io.platform == "darwin": is_nml_db = 0
    else: is_nml_db = 1
    existing_NML_entry = session.query(NMLImages).filter_by(image_id=image_id).first()
    if not existing_NML_entry:
        if VERBOSE: print("   #########    new image, not in NML results_dict: ", image_id)
        try:
            new_NML_entry = NMLImages(image_id=image_id, is_nml_db=is_nml_db)
            session.add(new_NML_entry)
            session.commit()
        except:
            print("failed to store wandering image,", image_id)
    else:
        if VERBOSE: print(f"Entry already exists for image_id: {image_id}")
        pass

    
    # Check if the current batch is ready for commit
    # if total_processed % BATCH_SIZE == 0:

    # for testing, comment out the commit
    session.commit()


def check_is_feet(body_landmarks):
    is_feet = None
    # 4. Evaluate visibility for feet landmarks (27-32)
    foot_lms = body_landmarks.landmark[27:33]
    # print(f"Foot landmarks: {foot_lms}")
    visible_count = sum(1 for lm in foot_lms if lm.visibility > 0.85)
    is_feet = (visible_count >= (len(foot_lms) / 2))
    return is_feet

def find_and_save_body(image_id, image, bbox, mongo_body_landmarks, hand_landmarks):
    if VERBOSE: print("find_and_save_body", mongo_body_landmarks)
    hue = sat = val = lum = lum_torso = hue_bb = sat_bb = val_bb = lum_bb = lum_torso_bb = selfie_bbox = bbox_dict = None
    is_left_shoulder=is_right_shoulder = is_feet = pose = is_hands = hand_landmarks = update_hand = None

    if BODYLMS and mongo_body_landmarks is None or REDO_BODYLMS_3D:
        if VERBOSE: print("doing body, mongo_body_landmarks is None")
        # find body landmarks and normalize them using function
        is_body, n_landmarks, body_landmarks, body_world_landmarks, face_height, nose_pixel_pos = process_image_find_body_subroutine(image_id, image, bbox)
        if body_landmarks is not None:
            is_feet = check_is_feet(body_landmarks)
        if VERBOSE: print("is_feet", image_id, is_feet)

        if face_height and not REDO_BODYLMS_3D:
            # only do this when there is a face. skip for no face -body reprocessing
            ### detect object info, 
            if VERBOSE:print("detecting objects")
            bbox_dict=sort.return_bbox(YOLO_MODEL,image, OBJ_CLS_LIST)
            if VERBOSE: print("detected objects")

            ### normed object bbox
            bbox_dict = process_image_normalize_object_bbox(bbox_dict, nose_pixel_pos, face_height, image.shape)

            ### do imagebackground calcs

            segmentation_mask=sort.get_segmentation_mask(get_bg_segment,image,None,None)
            is_left_shoulder,is_right_shoulder=sort.test_shoulders(segmentation_mask)
            if VERBOSE: print("shoulders",is_left_shoulder,is_right_shoulder)
            hue,sat,val,lum, lum_torso=sort.get_bg_hue_lum(image,segmentation_mask,bbox)  
            if VERBOSE: print("sat values before insert", hue,sat,val,lum,lum_torso)

            if bbox:
                #will do a second round for bbox with same cv2 image
                hue_bb,sat_bb, val_bb, lum_bb, lum_torso_bb =sort.get_bg_hue_lum(image,segmentation_mask,bbox)  
                if VERBOSE: print("sat values before insert", hue_bb,sat_bb, val_bb, lum_bb, lum_torso_bb)
            else:
                hue_bb = sat_bb = val_bb = lum_bb = lum_torso_bb = None
            selfie_bbox=sort.get_selfie_bbox(segmentation_mask)
            if VERBOSE: print("selfie_bbox",selfie_bbox)
        elif REDO_BODYLMS_3D:
            if VERBOSE: print("got 3D bodylms in a DOOVER, skipped the rest", image_id)
        else:
            if VERBOSE: print("no face, skipping object detection, just did the body, all values already are set to None", image_id)
    elif BODYLMS and mongo_body_landmarks:
        # this was for some error handling. to handle exisitin mongo_body_landmarks will require refactoring
        print("doing body, mongo_body_landmarks is not None", mongo_body_landmarks)
    ### save object bbox info
    # session = sort.parse_bbox_dict(session, image_id, PhoneBbox, OBJ_CLS_LIST, bbox_dict)
    if not REDO_BODYLMS_3D:
        is_hands = None
        if HANDLMS:
            # I need to check carefully to see make sure i do not update hand to none
            pose, is_hands, hand_landmarks, update_hand = process_image_hands_subroutine(image_id, image)
        else: 
            update_hand = False
    else:
        update_hand = False

    for _ in range(io.max_retries):
        try:
            # try to save using function:
            if VERBOSE: print("saving body and hands", image_id, "is_body", is_body, "is_hands", is_hands, "is_feet", is_feet)
            save_body_hands_mysql_and_mongo(session, image_id, image, bbox_dict, body_landmarks, body_world_landmarks, n_landmarks, hue_bb, lum_bb, sat_bb, val_bb, lum_torso_bb, hue, lum, sat, val, lum_torso, is_left_shoulder, is_right_shoulder, selfie_bbox, is_body, mongo_body_landmarks, is_hands, update_hand, hand_landmarks, pose, is_feet)
            if VERBOSE: print("saved body and hands")
            # if sql hiccups it will try again, if not, it will hit break on next line
            break  # Transaction succeeded, exit the loop
        except OperationalError as e:
            print(e)
            print(f"[process_image]session.query failed: {image_id}")
            time.sleep(io.retry_delay)

def check_open_files():
    process = psutil.Process(os.getpid())
    open_files = process.open_files()
    print(f"Currently open files: {len(open_files)}")
    if len(open_files) > 100:  # arbitrary threshold
        for f in open_files[-10:]:  # show last 10
            print(f"  {f.path}")

def process_image_bodylms(task):
    # this is the main show April 2025

    if VERBOSE: print("task is: ",task)
    image_id = task[0] ### is it enc or image_id
    if task[4] is not None:
        if type(task[4]) == str:
            bbox = io.unstring_json(task[4])
        elif type(task[4]) == dict:
            bbox = task[4]
        else:
            print("no bbox for this task", task)
            bbox = None
    else:
        print("no bbox for this task", task)
        bbox = None
    cap_path = capitalize_directory(task[1])
    # mongo_face_landmarks = task[2]
    mongo_body_landmarks = task[3]
    init_session()
    init_mongo()
    hand_landmarks = None

    thread_id = threading.get_ident()
    try:
        # print(f"Thread {thread_id}: Processing {cap_path}")
        image = cv2.imread(cap_path)
        
        if image is None:
            # print(f"Thread {thread_id}: Failed to load image: {cap_path}")
            return
            
        # Your processing here
        
        if image is not None and image.shape[0]>MINSIZE and image.shape[1]>MINSIZE:
            # Do findbody

            find_and_save_body(image_id, image, bbox, mongo_body_landmarks, hand_landmarks)
            
        else:
            no_image = True
            # store no_image in Images table
            session.query(Images).filter(Images.image_id == image_id).update({
                Images.no_image: no_image
            }, synchronize_session=False)
            session.query(SegmentTable).filter(SegmentTable.image_id == image_id).update({
                SegmentTable.no_image: no_image
            }, synchronize_session=False)
            session.commit()
            print('no image or toooooo smallllll, stored in Images table for image_id', image_id)


    except OSError as e:
        if e.errno == 24:  # Too many open files
            print(f"Thread {thread_id}: Too many open files error!")
            check_open_files()  # From code above
        traceback.print_exc()
    except Exception as e:
        print(f'Thread {thread_id}: Error: {str(e)}')
        traceback.print_exc()
    finally:
        # Cleanup
        # Close the session and dispose of the engine before the worker process exits
        close_mongo()
        close_session()
        # collect_the_garbage()
        if 'image' in locals():
            del image
        gc.collect()

    # store data



def process_image(task):
    print("process_image this is where the action is")
    # print("processing task:", task)
    pr_split = time.time()
    def save_image_triage(image,df):
        #saves a CV2 image elsewhere -- used in setting up test segment of images
        if df.at['1', 'is_face']:
            sort = "face"
        elif df.at['1', 'is_body']:
            sort = "body"
        else:
            sort = "none"
        name = str(df.at['1', 'image_id'])+".jpg"
        save_image_by_path(image, sort, name)

    init_session()
    init_mongo()

    no_image = False

    df = pd.DataFrame(columns=['image_id','is_face','is_body','is_face_distant','face_x','face_y','face_z','mouth_gap','face_landmarks','bbox','face_encodings','face_encodings68_J','body_landmarks'])
    df.at['1', 'image_id'] = task[0]
    image_test = cv2.imread(task[1])
    print(">> SPLIT >> image_test shape", image_test.shape)
    cap_path = capitalize_directory(task[1])
    # print(">> SPLIT >> made DF, about to imread")
    # pr_split = print_get_split(pr_split)

    try:
        print(">> SPLIT >> trying to read image:", cap_path)
        # i think i'm doing this twice. I should just do it here. 
        image = cv2.imread(task[1])
        
        h,w,_ = image.shape
        print(">> SPLIT >> image shape", h, w, image.shape)
        # h, w, _ = image.shape
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
        mp_image = ensure_image_mp(image)
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))
        # mp_image is RGB now 
        # this is for when you need to move images into a testing folder structure
        # save_image_elsewhere(image, task)
    except Exception as e:
        print('Error:', str(e))
        print(f"[process_image] this item failed: {task}")

    # print(">> SPLIT >> done imread, about to find face")
    # pr_split = print_get_split(pr_split)

    if mp_image is not None and h > MINSIZE and w > MINSIZE:
        # Do FaceMesh

        print(">> SPLIT >> about to find_face")
        df, number_of_detections = find_face(mp_image, df)
        is_small = 0
        # pr_split = print_get_split(pr_split)
        print(number_of_detections, df)
        print(">> SPLIT >> done find_face")

        # Do Body Pose
        # temporarily commenting this out
        # to reactivate, will have to accept return of is_body, body_landmarks
        # df = find_body(mp_image, df)

        # print(">> SPLIT >> done find_body")
        # pr_split = print_get_split(pr_split)

        # for testing: this will save images into folders for is_face, is_body, and none. 
        # only save images that aren't too smallllll
        # save_image_triage(image,df)
    elif mp_image is not None and h>0 and w>0 :
        print('smallllll but still processing')
        # print(task[0], "shape of image", image.shape)
        df, number_of_detections = find_face(mp_image, df)
        # print(df)
        is_small = 1
    else:
        print(">> no image", task)
        # assign no_image = 1 in SegmentBig_isnoface table
        no_image = True

        try:
            # store no_image in Images table
            session.query(Images).filter(Images.image_id == task[0]).update({
                Images.no_image: no_image
            }, synchronize_session=False)
            session.commit()
            print("stored no image in Images")
        except:
            print("failed to store no_image in Images")


        # check SegmentTable.image_id == task[0] to see if entry exists in SegmentTable


        existing_entry = session.query(SegmentBig_isnotface).filter(SegmentBig_isnotface.image_id == task[0]).all()
        # query existing entry
        # print("existing_entry to store no_image:", existing_entry)
        # print(type(existing_entry))
        if existing_entry:
            print("existing_entry to store no_image")
            # if segment table entry exists, update it
            # otherwise, will continue on, and store no_image in SegmentBig_isnotface new entry
            session.query(SegmentBig_isnotface).filter(SegmentBig_isnotface.image_id == task[0]).update({
                SegmentBig_isnotface.no_image: no_image
            }, synchronize_session=False)
            print("stored no image in existing SegmentBig_isnotface entry")
            return
        else:
            print("creating new no_image entry in SegmentBig_isnotface")
            # create new entry in SegmentBig_isnotface
            new_segment_entry = SegmentBig_isnotface(image_id=task[0], no_image=no_image)
            session.add(new_segment_entry)
            session.commit()
            print("stored no_image in SegmentBig_isnotface for ", task[0])
            return
            # # store no_image in Images table
            # session.query(Images).filter(Images.image_id == image_id).update({
            #     Images.no_image: no_image
            # }, synchronize_session=False)
            # session.query(SegmentTable).filter(SegmentTable.image_id == image_id).update({
            #     SegmentTable.no_image: no_image
            # }, synchronize_session=False)
            # session.commit()
            # print('no image or toooooo smallllll, stored in Images table')
             

    # testing, so quitting before I store data
    # return
    print(f"number_of_detections: {number_of_detections}, for image_id: {task[0]}")
    try:
        
        # DEBUGGING --> need to change this back to "encodings"
        # print(df)  ### made it all lower case to avoid discrepancy
        # print(df.at['1', 'face_encodings'])
        dict_df = df.to_dict('index')
        insert_dict = dict_df["1"]
        insert_dict['is_small'] = is_small

        # remove all nan/none/null values
        keys_to_remove = []
        for key, value in insert_dict.items():
            # print("about to try", key)
            try:
                if np.isnan(value):
                    # print("is NaN")
                    # print(key, value)
                    keys_to_remove.append(key)
                else:
                    pass 
                    #is non NaN
                    # print("slips through")
                    # print(key, value)
            except TypeError:
                # print("is not NaN")
                # print(key, value)
                pass 
                #is non NaN

        for key in keys_to_remove:
            del insert_dict[key]


        # print(">> SPLIT >> done insert_dict stuff")
        # pr_split = print_get_split(pr_split)


        # print("dict_df", insert_dict)
        # quit()
        image_id = insert_dict['image_id']

        # if IS_FOLDER is not True:
        # Check if the entry exists in the Encodings table
        image_id = insert_dict['image_id']
        # can I filter this by site_id? would that make it faster or slower? 
        existing_entry = session.query(Encodings).filter_by(image_id=image_id).first()

        two_noses = False
        if number_of_detections > 1:
            # there are more than one face detected
            two_noses = True
            print("too many faces detected")
            # store too_many_faces in Encodings table
            session.query(Encodings).filter(Encodings.image_id == image_id).update({
                Encodings.two_noses: two_noses
            }, synchronize_session=False)
            # print("not committing yet")
            session.commit()

        else:    
            # if there is one face detected
            face_encodings68 = insert_dict.pop('face_encodings68', None)
            face_landmarks = insert_dict.pop('face_landmarks', None)
            if face_encodings68: is_encodings = 1
            else: is_encodings = 0

            if existing_entry is not None and existing_entry.mongo_encodings is None:
                is_face_no_lms = insert_dict["is_face_no_lms"]
                if VERBOSE: print(f"existing_entry for image_id: {image_id} with no existing mongo_encodings. is_face_no_lms is: {is_face_no_lms} Going to add these encodings: {is_encodings}")
                # this is a new face to update an existing encodings entry
                # update the existing entry with the insert_dict values
                for key, value in insert_dict.items():
                    if key not in ['image_id', 'encoding_id']:
                        setattr(existing_entry, key, value)
                existing_entry.mongo_encodings = is_encodings
                existing_entry.mongo_face_landmarks = is_encodings
                existing_entry.is_face_no_lms = is_face_no_lms
                encoding_id = existing_entry.encoding_id  # Assuming 'encoding_id' is the name of your primary key column
                commit_this = True

            elif existing_entry is not None and existing_entry.mongo_encodings == 1 and existing_entry.bbox is None:
                is_face_no_lms = insert_dict["is_face_no_lms"]
                if VERBOSE: print(f"existing_entry for image_id: {image_id} with no existing mongo_encodings. is_face_no_lms is: {is_face_no_lms} Going to add these encodings: {is_encodings}")
                # this is a new face to update an existing encodings entry
                # update the existing entry with the insert_dict values
                for key, value in insert_dict.items():
                    if key not in ['image_id', 'encoding_id']:
                        setattr(existing_entry, key, value)
                existing_entry.mongo_encodings = is_encodings
                existing_entry.mongo_face_landmarks = is_encodings
                encoding_id = existing_entry.encoding_id  # Assuming 'encoding_id' is the name of your primary key column
                commit_this = True


            # elif IS_FOLDER is True or existing_entry is None:
            elif existing_entry is None:
                print(f"new entry for image_id: {image_id}")
                new_entry = Encodings(**insert_dict)
                new_entry.mongo_encodings = is_encodings
                new_entry.mongo_face_landmarks = is_encodings
                encoding_id = new_entry.encoding_id  # Assuming 'encoding_id' is the name of your primary key column
                session.add(new_entry)
                commit_this = True

            else:
                print("already exists, nothing to update")
                commit_this = False

            if commit_this:
                for _ in range(io.max_retries):
                    try:

                        if VERBOSE: print("about to commit")
                        # print the values for the session
                        for key, value in insert_dict.items():
                            print(f"{key}: {value}")


                        session.commit()
                        print(f"Newly updated/inserted row has encoding_id: {encoding_id} for image_id {image_id} with is_encodings: {is_encodings}")

                        if is_encodings:
                            try:
                                mongo_collection.insert_one({
                                    "encoding_id": encoding_id,
                                    "image_id": image_id,
                                    "face_landmarks": face_landmarks,
                                    "face_encodings68": face_encodings68
                                })
                            except DuplicateKeyError as e:
                                print(f"Duplicate key error for encoding_id: {encoding_id}, image_id: {image_id}")
                                print(f"Error details: {e}")
                                continue  # Move to the next iteration of the loop
                            except Exception as e:
                                print(f"An unexpected error occurred: {e}")
                                continue  # Move to the next iteration of the loop

                        break  # Transaction succeeded, exit the loop
                    except OperationalError as e:
                        print("exception on new_entry session.commit")
                        print(e)
                        time.sleep(io.retry_delay)

    except OperationalError as e:
        print("process_image", image_id, e)

    # autoincrement the counter_value for counter_name == number_of_detections in Counters table
    session.query(Counters).filter(Counters.counter_name == str(number_of_detections)).update({
        Counters.counter_value: Counters.counter_value + 1
    }, synchronize_session=False)
    print(f"autoincremented counter_value for counter_name: {number_of_detections}")
    session.commit()

    # Close the session and dispose of the engine before the worker process exits
    close_mongo()
    close_session()
    collect_the_garbage()
    # print(f"finished session {image_id}")

        # save image based on is_face
def do_job(tasks_to_accomplish, tasks_that_are_done):
    #print("do_job")
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:
            # print("queue.Empty")
            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            if len(task) > 2:
                
                if BODYLMS is True or HANDLMS is True or REDO_BODYLMS_3D is True:
                    if VERBOSE: print("do_job via process_image_bodylms:")
                    process_image_bodylms(task)
                else:
                    # landmarks and bbox, so this is an encodings only
                    process_image_enc_only(task)
                    if VERBOSE: print("process_image_enc_only")


            else:
                if VERBOSE: print("do_job via regular process_image:", task)
                process_image(task)
                # print(f"done process_image for {task}")
            # tasks_that_are_done.put(task + ' is done by ' + current_process().name)
            time.sleep(SLEEP_TIME)
    return True


def main():
    # print("main")


    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    count = 0
    this_count = 0
    folder_count = 0
    last_round = False
    jsonsplit = time.time()

    if IS_FOLDER is True:
        print("in IS_FOLDER")
        folder_paths = io.make_hash_folders(MAIN_FOLDER, as_list=True)
        # print(len(folder_paths))
        completed_folders = io.get_csv_aslist(CSV_FOLDERCOUNT_PATH)
        # print(len(completed_folders))
        for folder_path in folder_paths:
            
            # if folder_path in THESE_FOLDER_PATHS:
            if folder_path not in completed_folders:

                folder = os.path.join(MAIN_FOLDER,folder_path)
                folder_count += 1
                if not os.path.exists(folder):
                    # print(str(folder_count), "no folder here:",folder)
                    continue
                else:
                    print(str(folder_count), folder)

                img_list = io.get_img_list(folder)
                # print("len(img_list)", len(img_list))


                # Initialize an empty list to store all the results
                all_results = []

                # Split the img_list into smaller batches and process them one by one
                for i in range(0, len(img_list), BATCH_SIZE):

                    batch_img_list = img_list[i : i + BATCH_SIZE]

                    print(f"total img_list: {len(img_list)} no. processed: {i} no. left: {len(img_list)-i}")
                    if len(img_list)-i<BATCH_SIZE: print("last_round for img_list")
                    # CHANGE FOR EACH SITE
                    # ALSO site_image_id DOWN BELOW 
                    # Collect site_image_id values from the image filenames
                    if SITE_NAME_ID == 8:
                    # 123rf
                        batch_site_image_ids = [img.split("-")[0] for img in batch_img_list]
                    elif SITE_NAME_ID == 5:
                        batch_site_image_ids = [img.split("-")[-1].replace(".jpg","") for img in batch_img_list]
                    elif SITE_NAME_ID == 1:
                    # gettyimages
                        batch_site_image_ids = [img.split("-id")[-1].replace(".jpg", "") for img in batch_img_list]
                    # site_name_id = 1
                    else:
                    # # Adobe and pexels and shutterstock and istock
                        batch_site_image_ids = [img.split(".")[0] for img in batch_img_list]
                    site_name_id = SITE_NAME_ID

                    if VERBOSE: print("batch_site_image_ids", len(batch_site_image_ids))
                    if VERBOSE: print("batch_site_image_ids", batch_site_image_ids[:5])


                    # query the database for the current batch and return image_id and encoding_id
                    for _ in range(io.max_retries):

                        try:
                            if VERBOSE: print(f"Processing batch {i//BATCH_SIZE + 1}...")
                            init_session()
                            # task = (image_id,imagepath,row["mongo_face_landmarks"], row["mongo_body_landmarks"],row["bbox"])
                            # batch_query = session.query(Images.image_id, Images.site_image_id, Encodings.encoding_id) \

                            # get Images and Encodings values for each site_image_id in the batch
                            # adding in mongo stuff. should return NULL if not there
                            batch_query = session.query(Images.image_id, Images.site_image_id, Images.imagename, Encodings.encoding_id, Encodings.mongo_face_landmarks, Encodings.mongo_body_landmarks, Encodings.bbox, Encodings.mongo_body_landmarks_3D) \
                                                .outerjoin(Encodings, Images.image_id == Encodings.image_id) \
                                                .filter(Images.site_image_id.in_(batch_site_image_ids), Images.site_name_id == site_name_id, Images.no_image.isnot(True))
                            batch_results = batch_query.all()

                            all_results.extend(batch_results)
                            if VERBOSE: print("about to close_session()")
                            # Close the session and dispose of the engine before the worker process exits
                            close_session()

                        except OperationalError as e:
                            print("error getting batch results")
                            print(e)
                            time.sleep(io.retry_delay)
                    if VERBOSE: print(f"no. all_results: {len(all_results)}")

                    # print("results:", all_results)
                    results_dict = {result.site_image_id: result for result in batch_results}

                    # going back through the img_list, to use as key for the results_dict

                    images_left_to_process = len(batch_img_list)
                    for img in batch_img_list:

                        # CHANGE FOR EACH SITE
                        if SITE_NAME_ID == 8:
                            # extract site_image_id for 213rf
                            site_image_id = img.split("-")[0]

                        # # extract site_image_id for getty images
                        # elif SITE_NAME_ID == 1:
                            # site_image_id = img.split("-id")[-1].replace(".jpg", "")

                        else:
                        # # # extract site_image_id for adobe and pexels and shutterstock and istock and pond5
                            site_image_id = img.split(".")[0]

                        # print("site_image_id", site_image_id)
                        if site_image_id in results_dict:
                            result = results_dict[site_image_id]
                            # if VERBOSE: print("is in results", result)
                            # print("in results encoding_id", result.encoding_id)
                            imagepath = os.path.join(folder, img)
                                                        
                            if not result.encoding_id:
                                # if it hasn't been encoded yet, add it to the tasks
                                task = (result.image_id, imagepath)
                                print(">> adding to face queue:", result.image_id, "site_image_id", site_image_id)
                            elif result.mongo_body_landmarks and result.mongo_body_landmarks_3D is None and REDO_BODYLMS_3D is True:
                                # if body has been found but not 3D, add it to the tasks
                                print(">>>> adding to 3D BODY queue:", result.image_id, "site_image_id", site_image_id)
                                task = (result.image_id, imagepath, result.mongo_face_landmarks, result.mongo_body_landmarks, result.bbox)
                            elif result.mongo_face_landmarks and result.mongo_body_landmarks is None:
                                # if face has been encoded but not body, add it to the tasks
                                print(">>>> adding to BODY queue:", result.image_id, "site_image_id", site_image_id)
                                task = (result.image_id, imagepath, result.mongo_face_landmarks, result.mongo_body_landmarks, result.bbox)
                            elif result.mongo_face_landmarks and result.mongo_body_landmarks is not None:
                                if VERBOSE: print("     xx ALREADY FULLY DONE:", result.image_id)
                                task = None
                            elif result.mongo_face_landmarks == 0 and result.mongo_body_landmarks == 1:
                                if VERBOSE: print("     xxxx ALREADY FULLY DONE:", result.image_id)
                                task = None
                            elif result.mongo_face_landmarks == 0 and result.mongo_body_landmarks == 0:
                                if VERBOSE: print("     xxxx ALREADY FULLY DONE - nobody here:", result.image_id)
                                task = None
                            elif result.mongo_face_landmarks is None and result.mongo_body_landmarks == 1 and result.bbox is not None:
                                print("~?~ WEIRD no face, but bbox:", result.image_id, imagepath, result.mongo_face_landmarks, result.mongo_body_landmarks, result.bbox)
                                # for the integrated version, this will do both
                                task = (result.image_id, imagepath)
                            elif result.mongo_face_landmarks is None and result.mongo_body_landmarks is None:
                                print("~~ no face, adding to face queue for is_face_no_lms do_over:", result.image_id, imagepath, result.mongo_face_landmarks, result.mongo_body_landmarks, result.bbox)
                                # for the integrated version, this will do both
                                task = (result.image_id, imagepath)
                            elif result.mongo_face_landmarks == 0 and result.mongo_body_landmarks is None:
                                print("~~~~ no face, do body without bbox:", result.image_id, imagepath, result.mongo_face_landmarks, result.mongo_body_landmarks, result.bbox)
                                # for the integrated version, this will do both
                                task = (result.image_id, imagepath, result.mongo_face_landmarks, result.mongo_body_landmarks, None)
                            else:
                                # skips BOTH 1) face and body have been encoded 2) no face was detected -- these should be rerun to find the body only
                                print(" ?  something is wacky, skipping:", result.image_id, imagepath, result.mongo_face_landmarks, result.mongo_body_landmarks, result.bbox)
                                task = None
                            # print(task)
                            if task is not None:
                                tasks_to_accomplish.put(task)
                                this_count += 1
                        else: 
                            # store site_image_id and SITE_NAME_ID in WanderingImages table
                            wandering_name_site_id = site_image_id+"."+str(SITE_NAME_ID)
                            existing_entry = session.query(WanderingImages).filter_by(wandering_name_site_id=wandering_name_site_id).first()
                            if not existing_entry and not ("-id" in wandering_name_site_id):
                                print("wandering image, not in results_dict: ", site_image_id)
                                try:
                                    new_wandering_entry = WanderingImages(wandering_name_site_id=wandering_name_site_id, site_image_id=site_image_id, site_name_id=SITE_NAME_ID)
                                    session.add(new_wandering_entry)
                                    session.commit()
                                except:
                                    print("failed to store wandering image,", wandering_name_site_id)
                            else:
                                if VERBOSE: print(f"Entry already exists for wandering_name_site_id: {wandering_name_site_id}")
                                pass

                        images_left_to_process = images_left_to_process -1 
                        if VERBOSE: 
                            if images_left_to_process < 500: print(f"no. images_left_to_process: {images_left_to_process}")

                    # print total count for this batch
                    print(f"######### total task count for this batch: {str(this_count)}")
                    print("just stored wandering images")

                    for w in range(io.NUMBER_OF_PROCESSES):
                        p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done))
                        processes.append(p)
                        p.start()

                    # completing process
                    for p in processes:
                        # print("completing process")
                        p.join()
                    print(">> SPLIT >> p.join,  this folder")
                    split = print_get_split(jsonsplit)


                    print(f"total count for {folder_path}: {str(this_count)}")
                    # if this_count > 500:
                    #     quit()
                    # else:
                    #     this_count = 0
                    # quit()
                count += this_count
                print(f"completed {str(this_count)} of {str(len(img_list))}")
                print(f"total count for {folder_path}: {str(count)}")
                this_count = 0

                # save success to CSV_FOLDERCOUNT_PATH
                io.write_csv(CSV_FOLDERCOUNT_PATH, [folder_path])

    else:
        print("old school SQL")
        start_id = 0
        while True:
            init_session()

            # print("about to SQL: ",SELECT,FROM,WHERE,LIMIT)
            resultsjson = selectSQL(start_id)  
            print("got results, count is: ",len(resultsjson))
            if resultsjson:
                last_result = resultsjson[-1]
                print("last_result", last_result)
                start_id = last_result["image_id"]
            print(">> SPLIT >> jsonsplit")
            split = print_get_split(jsonsplit)
            #catches the last round, where it returns less than full results
            if last_round == True:
                print("last_round caught, should break")
                break
            elif len(resultsjson) != LIMIT:
                last_round = True
                print("last_round just assigned")
            # process resultsjson
            for row in resultsjson:
                # encoding_id = row["encoding_id"]
                image_id = row["image_id"]
                item = row["contentUrl"]
                hashed_path = row["imagename"]
                site_id = row["site_name_id"]

                if SHUTTER_SSD_OVERRIDE:
                    # if SHUTTERPATH is in hashed_path, then pass, else, continue
                    if SHUTTERFOLDER not in hashed_path:
                        print(f"skipping {hashed_path}")
                        continue
                # if site_id == 1:
                #     # print("fixing gettyimages hash")
                #     orig_filename = item.replace(http, "")+".jpg"
                #     orig_filename = orig_filename.replace(".jpg.jpg", ".jpg")
                #     d0, d02 = get_hash_folders(orig_filename)
                #     hashed_path = os.path.join(d0, d02, orig_filename)
                if not image_id or not site_id or not hashed_path: 
                    print("missing image_id or site_id or hashed_path", row)
                # gets folder via the folder list, keyed with site_id integer
                else:
                    try:
                        if OVERRIDE_PATH:
                            imagepath = os.path.join(OVERRIDE_PATH, hashed_path)
                        else:
                            imagepath=os.path.join(io.folder_list[site_id], hashed_path)
                    except:
                        print("missing folder for site_id", site_id)
                        continue
                    if row.get("mongo_face_landmarks") is not None:
                        # this is a reprocessing, so don't need to test isExist
                        if VERBOSE: print("reprocessing")
                        task = (image_id, imagepath, row.get("mongo_face_landmarks"), row.get("mongo_body_landmarks"), row.get("bbox"))
                    else:
                        task = (image_id, imagepath)

                        # getting rid of isExist test for now
                        # isExist = os.path.exists(imagepath)
                        # print(">> SPLIT >> isExist")
                        # split = print_get_split(split)
                        # if isExist: 
                        #     task = (image_id,imagepath)
                        # else:
                        #     print("this file is missssssssssing --------> ",imagepath)
                    tasks_to_accomplish.put(task)
                    # print("tasks_to_accomplish.put(task) ",imagepath)

            # creating processes
            for w in range(io.NUMBER_OF_PROCESSES):
                p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done))
                processes.append(p)
                p.start()

            # completing process
            for p in processes:
                # print("completing process")
                p.join()
            print(">> SPLIT >> p.join, done with this query")
            split = print_get_split(split)
            # Close the session and dispose of the engine before the worker process exits
            close_session()

            count += len(resultsjson)
            print("completed round, total results processed is: ",count)


    end = time.time()
    print (end - start)
    print ("total processed ",count)
    return True

if __name__ == '__main__':
    main()


    if hand_landmarker:
        hand_landmarker.close()
        print("HandLandmarker model closed.")
    if pose_landmarker:
        pose_landmarker.close()
        print("PoseLandmarker model closed.")
    if face_landmarker:
        face_landmarker.close()
        print("FaceLandmarker model closed.")

