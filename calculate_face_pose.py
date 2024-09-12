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

import numpy as np
import mediapipe as mp
import pandas as pd
from ultralytics import YOLO

from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, Boolean, DECIMAL, BLOB, JSON, String, Date, ForeignKey, update, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
# my ORM
from my_declarative_base import Base, Images, Keywords, SegmentTable, ImagesKeywords, ImagesBackground, Encodings, PhoneBbox, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

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
IS_FOLDER = False


'''
1   getty
2   shutterstock
3   adobe
4   istock
5   pexels
6   unsplash
7   pond5
8   123rf
9   alamy - WIP
10  visualchinagroup - done
11	picxy - done
12	pixerf - done
13	imagesbazaar - done
14	indiapicturebudget - done
15	iwaria - done
16	nappy  - done
17	picha - done
18	afripics
'''

SITE_NAME_ID = 1
# 2, shutter. 4, istock
# 7 pond5
POSE_ID = 0

# MAIN_FOLDER = "/Volumes/RAID18/images_pond5"
# MAIN_FOLDER = "/Volumes/SSD4green/images_pixerf"
MAIN_FOLDER = "/Volumes/SSD4green/images_alamy"
# MAIN_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/images_picha"
# MAIN_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/afripics_v2/images"

# MAIN_FOLDER = "/Volumes/SSD4/images_getty_reDL"
BATCH_SIZE = 1000 # Define how many from each folder in each batch
LIMIT = 10000

#temp hack to go 1 subfolder at a time
# THESE_FOLDER_PATHS = ["8/8A", "8/8B","8/8C", "8/8D", "8/8E", "8/8F", "8/80", "8/81", "8/82", "8/83", "8/84", "8/85", "8/86", "8/87", "8/88", "8/89"]
THESE_FOLDER_PATHS = ["9/9C", "9/9D", "9/9E", "9/9F", "9/90", "9/91", "9/92", "9/93", "9/94", "9/95", "9/96", "9/97", "9/98", "9/99"]

# MAIN_FOLDER = "/Volumes/SSD4/adobeStockScraper_v3/images"
# MAIN_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/gettyimages/newimages"
CSV_FOLDERCOUNT_PATH = os.path.join(MAIN_FOLDER, "folder_countout.csv")

IS_SSD=True
BODYLMS = True # only matters if IS_FOLDER is False
HANDLMS = True
DO_INVERSE = True
SEGMENT = 0 # topic_id set to 0 or False if using HelperTable or not using a segment
HelperTable_name = False #"SegmentHelperMay7_fingerpoint" # set to False if not using a HelperTable


if BODYLMS is True or HANDLMS is True:

    # prep for image background object
    get_background_mp = mp.solutions.selfie_segmentation
    get_bg_segment = get_background_mp.SelfieSegmentation()


    ############# Reencodings #############
    SELECT = "DISTINCT seg1.image_id, seg1.site_name_id, seg1.contentUrl, seg1.imagename, seg1.site_image_id, seg1.mongo_body_landmarks, seg1.mongo_face_landmarks, seg1.bbox"

    SegmentTable_name = 'SegmentOct20'
    FROM =f"{SegmentTable_name} seg1"
    # FROM ="Encodings e"
    if BODYLMS or (BODYLMS and HANDLMS):
        QUERY = " seg1.mongo_body_landmarks IS NULL and seg1.no_image IS NULL"
        SUBQUERY = " "
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
    if SEGMENT:
        QUERY = " "
        FROM = f"{SegmentTable_name} seg1 LEFT JOIN ImagesTopics it ON seg1.image_id = it.image_id"
        # SUBQUERY = f" seg1.mongo_body_landmarks IS NULL AND face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2 AND it.topic_id = {SEGMENT}"
        SUBQUERY = f" seg1.mongo_body_landmarks IS NULL AND it.topic_id = {SEGMENT}"

    if HelperTable_name:
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
    WHERE = f"e.encoding_id IS NULL AND i.site_name_id = {SITE_NAME_ID}"
    WHERE += " AND i.image_id > 88123000 "
    # AND i.age_id NOT IN (1,2,3,4)
    IS_SSD=False
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
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
# overriding DB for testing
# io.db["name"] = "gettytest3"


#creating my objects
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=1, static_image_mode=True)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mp_hands = mp.solutions.hands

####### new imports and models ########
mp_face_detection = mp.solutions.face_detection #### added face detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# predictor_path = "shape_predictor_68_face_landmarks.dat"
# sp = dlib.shape_predictor(predictor_path)

# # dlib hack
# face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
# facerec = dlib.face_recognition_model_v1(face_rec_model_path)
# detector = dlib.get_frontal_face_detector()

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
ONE_SHOT = False # take all files, based off the very first sort order.
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

def init_mongo():
    # init session
    # global engine, Session, session
    global mongo_client, mongo_db, mongo_collection, bboxnormed_collection, mongo_hand_collection
    # engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
    #                                 .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
    
    # engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    #     user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    # ), poolclass=NullPool)

    # # metadata = MetaData(engine)
    # metadata = MetaData() # apparently don't pass engine
    # Session = sessionmaker(bind=engine)
    # session = Session()
    # Base = declarative_base()

    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    mongo_db = mongo_client["stock"]
    mongo_collection = mongo_db["encodings"]
    bboxnormed_collection = mongo_db["body_landmarks_norm"]
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
        selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {QUERY} AND seg1.image_id > {start_id} {SUBQUERY} LIMIT {str(LIMIT)};"
    else:
        selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)};"

    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    close_session()
    return(resultsjson)

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

def retro_bbox(image):
    height, width, _ = image.shape
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_det: 
        results_det=face_det.process(image)  ## [:,:,::-1] is the shortcut for converting BGR to RGB
        
    # is_face = False
    bbox_json = None
    if results_det.detections:
        faceDet=results_det.detections[0]
        bbox = get_bbox(faceDet, height, width)
        if bbox:
            bbox_json = json.dumps(bbox, indent = 4)
    else:
        print("no results???")
    return bbox_json


def find_face(image, df):
    # image is RGB
    find_face_start = time.time()
    height, width, _ = image.shape
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_det: 
        # print(">> find_face SPLIT >> with mp.solutions constructed")
        # ff_split = print_get_split(find_face_start)

        results_det=face_det.process(image)  ## [:,:,::-1] is the shortcut for converting BGR to RGB

        # print(">> find_face SPLIT >> face_det.process(image)")
        # ff_split = print_get_split(ff_split)
        
    '''
    0 type model: When we will select the 0 type model then our face detection model will be able to detect the 
    faces within the range of 2 meters from the camera.
    1 type model: When we will select the 1 type model then our face detection model will be able to detect the 
    faces within the range of 5 meters. Though the default value is 0.
    '''
    is_face = False

    if results_det.detections:
        faceDet=results_det.detections[0]
        bbox = get_bbox(faceDet, height, width)
        # print(">> find_face SPLIT >> get_bbox()")
        # ff_split = print_get_split(ff_split)

        if bbox:

            with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                             refine_landmarks=False,
                                             max_num_faces=1,
                                             min_detection_confidence=0.5
                                             ) as face_mesh:
            # Convert the BGR image to RGB and cropping it around face boundary and process it with MediaPipe Face Mesh.
                                # crop_img = img[y:y+h, x:x+w]
                # print(">> find_face SPLIT >> const face_mesh")
                # ff_split = print_get_split(ff_split)

                results = face_mesh.process(image[bbox["top"]:bbox["bottom"],bbox["left"]:bbox["right"]])   
                # print(">> find_face SPLIT >> face_mesh.process")
                # ff_split = print_get_split(ff_split)
 
            #read any image containing a face
            if results.multi_face_landmarks:
                
                #construct pose object to solve pose
                is_face = True
                pose = SelectPose(image)

                #get landmarks
                faceLms = pose.get_face_landmarks(results, image,bbox)

                # print(">> find_face SPLIT >> got lms")
                # ff_split = print_get_split(ff_split)


                #calculate base data from landmarks
                pose.calc_face_data(faceLms)

                # get angles, using r_vec property stored in class
                # angles are meta. there are other meta --- size and resize or something.
                angles = pose.rotationMatrixToEulerAnglesToDegrees()
                mouth_gap = pose.get_mouth_data(faceLms)
                             ##### calculated face detection results
                # old version, encodes everything

                # print(">> find_face SPLIT >> done face calcs")
                # ff_split = print_get_split(ff_split)


                if is_face:

                # # new version, attempting to filter the amount that get encoded
                # if is_face  and -20 < angles[0] < 10 and np.abs(angles[1]) < 4 and np.abs(angles[2]) < 3 :
                    # Calculate Face Encodings if is_face = True
                    # print("in encodings conditional")
                    # turning off to debug
                    encodings = calc_encodings(image, faceLms,bbox) ## changed parameters
                    print(">> find_face SPLIT >> calc_encodings")
                    # ff_split = print_get_split(ff_split)

                    # # image is currently BGR so converting back to RGB
                    # image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
                    
                    # # gets bg hue and lum without bbox
                    # hue,sat,val,lum, lum_torso=sort.get_bg_hue_lum(image_rgb)
                    # print("HSV", hue, sat, val)

                    # gets bg hue and lum with bbox
                    # hue_bb,sat_bb, val_bb, lum_bb, lum_torso_bb =sort.get_bg_hue_lum(image,bbox,faceLms)
                    # print("HSV", hue_bb,sat_bb, val_bb)
                    # quit()
                #     print(encodings)
                #     exit()
                # #df.at['1', 'is_face'] = is_face

                # # debug
                # else:
                #     print("bad angles")
                #     print(angles[0])
                #     print(angles[1])
                #     print(angles[2])

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
    df.at['1', 'is_face'] = is_face
    # print(">> find_face SPLIT >> prepped dataframe")
    # ff_split = print_get_split(ff_split)

    return df

def calc_encodings(image, faceLms,bbox):## changed parameters and rebuilt
    def get_dlib_all_points(landmark_points):
        raw_landmark_set = []
        for index in landmark_points:                       ######### CORRECTION: landmark_points_5_3 is the correct one for sure
            # print(faceLms.landmark[index].x)

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
    
    # dlib_all_points = get_dlib_all_points(landmark_points)

    # temp test hack
    # dlib_all_points5 = get_dlib_all_points(landmark_points_5)
    dlib_all_points68 = get_dlib_all_points(landmark_points_68)

    # ymin ("top") would be y value for top left point.
    bbox_rect= dlib.rectangle(left=bbox["left"], top=bbox["top"], right=bbox["right"], bottom=bbox["bottom"])


    # if (dlib_all_points is None) or (bbox is None):return 
    # full_object_detection=dlib.full_object_detection(bbox_rect,dlib_all_points)
    # encodings=face_encoder.compute_face_descriptor(image, full_object_detection, num_jitters=NUM_JITTERS)

    if (dlib_all_points68 is None) or (bbox is None):return 
    
    # full_object_detection5=dlib.full_object_detection(bbox_rect,dlib_all_points5)
    # encodings5=face_encoder.compute_face_descriptor(image, full_object_detection5, num_jitters=NUM_JITTERS)
    # encodings5j=face_encoder.compute_face_descriptor(image, full_object_detection5, num_jitters=3)
    # encodings5v2=facerec.compute_face_descriptor(image, full_object_detection5, num_jitters=NUM_JITTERS)

    full_object_detection68=dlib.full_object_detection(bbox_rect,dlib_all_points68)
    encodings68=face_encoder.compute_face_descriptor(image, full_object_detection68, num_jitters=NUM_JITTERS)
    # encodings68j=face_encoder.compute_face_descriptor(image, full_object_detection68, num_jitters=3)
    # encodings68v2=facerec.compute_face_descriptor(image, full_object_detection68, num_jitters=NUM_JITTERS)

    # # hack of full dlib
    # dets = detector(image, 1)
    # for k, d in enumerate(dets):
    #     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    #         k, d.left(), d.top(), d.right(), d.bottom()))
    #     # Get the landmarks/parts for the face in box d.
    #     shape = sp(image, d)
    #     # print("shape")
    #     # print(shape.pop())
    #     face_descriptor = facerec.compute_face_descriptor(image, shape)
    #     # print(face_descriptor)
    #     encD=np.array(face_descriptor)


    encodings = encodings68

    # enc1=np.array(encodings5)
    # enc2=np.array(encodings68)
    # d=np.linalg.norm(enc1 - enc2, axis=0)

    # # distance = pose.get_d(encodings5, encodings68)
    # print("distance between 5 and 68 ")    
    # print(d)    


    # d=np.linalg.norm(encD - enc2, axis=0)

    # # distance = pose.get_d(encodings5, encodings68)
    # print("distance between dlib and mp hack - 68 ")    
    # print(d)    


    # # enc12=np.array(encodings5v2)
    # # enc22=np.array(encodings68v2)
    # # d=np.linalg.norm(enc12 - enc22, axis=0)

    # # # distance = pose.get_d(encodings5, encodings68)
    # # print("distance between 5v2 and 68v2 ")    
    # # print(d)    


    # enc1j=np.array(encodings5j)
    # enc2j=np.array(encodings68j)
    # d=np.linalg.norm(enc1j - enc2j, axis=0)

    # # distance = pose.get_d(encodings5, encodings68)
    # print("distance between 5j and 68j ")    
    # print(d)    

    # d=np.linalg.norm(enc1j - enc1, axis=0)
    # # distance = pose.get_d(encodings5, encodings68)
    # print("distance between 5 and 5j ")    
    # print(d)    


    # d=np.linalg.norm(enc2j - enc2, axis=0)
    # # distance = pose.get_d(encodings5, encodings68)
    # print("distance between 68 and 68j ")    
    # print(d)    


    # # d=np.linalg.norm(enc2 - enc22, axis=0)
    # # # distance = pose.get_d(encodings5, encodings68)
    # # print("distance between 68v and 68v2 ")    
    # # print(d)    


    # print(len(encodings))
    return np.array(encodings).tolist()

def find_body(image):
    #print("find_body")
    with mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5) as pose:
      # for idx, file in enumerate(file_list):
        try:
            # image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            bodyLms = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # print("bodyLms, ", bodyLms)
            # bodyLms = results.pose_landmarks.landmark
            if not bodyLms.pose_landmarks:
                is_body = False
                body_landmarks = None
            else: 
                is_body = True
                body_landmarks = bodyLms.pose_landmarks

            
        except:
            print(f"[find_body]this item failed: {image}")
        return is_body, body_landmarks

def find_hands(image, pose):
    #print("find_body")

    with mp_hands.Hands(
        static_image_mode=True,          # If True, hand detection will be performed every frame.
        max_num_hands=2,                 # Detect a maximum of 2 hands.
        min_detection_confidence=0.4,    # Minimum confidence to detect hands.
        min_tracking_confidence=0.5      # Minimum confidence for hand landmarks tracking.
    ) as hands_detector:

        try:

            # Assuming image is in BGR format, as typically used in OpenCV.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image to detect hand landmarks.
            detection_result = hands_detector.process(image_rgb)
            if not detection_result.multi_handedness:
                # print("   >>>>>   No hands detected:", )
                return None, None
            else:
                hand_landmarks_list = pose.extract_hand_landmarks(detection_result)
                # print(f"Detected hands: {hand_landmarks_list}")
                return True, hand_landmarks_list

        except:
            print(f"[find_hands] this item failed: {image}")
            return None, None
        # # Extract the hand landmarks and handedness.
        # hand_landmarks_list = detection_result.multi_hand_landmarks
        # handedness_list = detection_result.multi_handedness

        # # If hands are detected
        # if hand_landmarks_list:
        #     for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
        #         print(f"Handedness: {handedness.classification[0].label}")
        #         for idx, landmark in enumerate(hand_landmarks.landmark):
        #             print(f"Landmark {idx}: (x={landmark.x}, y={landmark.y}, z={landmark.z})")



def capitalize_directory(path):
    dirname, filename = os.path.split(path)
    parts = dirname.split('/')
    capitalized_parts = [part if i < len(parts) - 2 else part.upper() for i, part in enumerate(parts)]
    capitalized_dirname = '/'.join(capitalized_parts)
    return os.path.join(capitalized_dirname, filename)

# this was for reprocessing the missing bbox
def process_image_bbox(task):
    # df = pd.DataFrame(columns=['image_id','bbox'])
    # print("task is: ",task)
    encoding_id = task[0]
    cap_path = capitalize_directory(task[1])
    try:
        image = cv2.imread(cap_path)        
        # this is for when you need to move images into a testing folder structure
        # save_image_elsewhere(image, task)
    except:
        print(f"[process_image]this item failed, even after uppercasing: {task}")
    # print("processing: ")
    # print(encoding_id)
    if image is not None and image.shape[0]>MINSIZE and image.shape[1]>MINSIZE:
        # Do FaceMesh
        bbox_json = retro_bbox(image)
        # print(bbox_json)
        if bbox_json: 
            for _ in range(io.max_retries):
                try:
                    update_sql = f"UPDATE Encodings SET bbox = '{bbox_json}' WHERE encoding_id = {encoding_id};"
                    engine.connect().execute(text(update_sql))
                    print("bboxxin:")
                    print(encoding_id)
                    break  # Transaction succeeded, exit the loop
                except OperationalError as e:
                    print(e)
                    time.sleep(io.retry_delay)
        else:
            print("no bbox")

    else:
        print('toooooo smallllll')
        # I should probably assign no_good here...?

    # store data


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
    # df.at['1', 'face_encodings'] = pickled_encodings

    # set name of df and table column, based on model and jitters
    # df_table_column = "face_encodings"
    # if SMALL_MODEL is not True:
    #     df_table_column = df_table_column+"68"
    # if NUM_JITTERS > 1:
    #     df_table_column = df_table_column+"_J"+str(NUM_JITTERS)

    # df.at['1', df_table_column] = pickled_encodings
    # sql = """
    # UPDATE Encodings SET df_table_column = :df_table_column
    # WHERE encoding_id = :encoding_id
    # """

    # else:
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

def process_image_bodylms(task):
    # df = pd.DataFrame(columns=['image_id','bbox'])
    if VERBOSE: print("task is: ",task)
    image_id = task[0] ### is it enc or image_id
    bbox = io.unstring_json(task[4])
    cap_path = capitalize_directory(task[1])
    mongo_face_landmarks = task[2]
    mongo_body_landmarks = task[3]
    init_session()
    init_mongo()
    nose_pixel_pos = None
    n_landmarks = None
    hand_landmarks = None
    is_body = False

    # df = pd.DataFrame(columns=['image_id','is_face','is_body','is_face_distant','face_x','face_y','face_z','mouth_gap','face_landmarks','bbox','face_encodings','face_encodings68_J','body_landmarks'])
    # df.at['1', 'image_id'] = image_id

    try:
        image = cv2.imread(cap_path)        
        # this is for when you need to move images into a testing folder structure
        # save_image_elsewhere(image, task)
    except Exception as e:
        print('Error:', str(e))
        print(f"[process_image]this imread failed, even after uppercasing: {task}")
    # print("processing: ")
    # print(image_id)
    if image is not None and image.shape[0]>MINSIZE and image.shape[1]>MINSIZE:
        # Do findbody


        
        if BODYLMS and mongo_body_landmarks is None:
            print("doing body, mongo_body_landmarks is None")
            is_body, body_landmarks = find_body(image)
            if is_body:
                ### NORMALIZE LANDMARKS ###
                nose_pixel_pos = sort.set_nose_pixel_pos(body_landmarks,image.shape)
                if VERBOSE: print("nose_pixel_pos",nose_pixel_pos)
                face_height = sort.convert_bbox_to_face_height(bbox)
                n_landmarks=sort.normalize_landmarks(body_landmarks,nose_pixel_pos,face_height,image.shape)
                # print("n_landmarks",n_landmarks)
                # sort.insert_n_landmarks(bboxnormed_collection, target_image_id,n_landmarks)
            else:
                print("no body")
                n_landmarks = None
            
            ### detect object info, 
            print("detecting objects")
            bbox_dict=sort.return_bbox(YOLO_MODEL,image, OBJ_CLS_LIST)
            if VERBOSE: print("detected objects")

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
                    print("normalized bbox but no nose_pixel_pos for ", image_id)
                elif bbox_dict_value:
                    if VERBOSE: print("setting normed bbox for OBJ_CLS_ID", OBJ_CLS_ID)
                    if VERBOSE: print("bbox_dict_value", bbox_dict_value)
                    if VERBOSE: print("bbox_n_key", bbox_n_key)

                    n_phone_bbox=sort.normalize_phone_bbox(bbox_dict_value,nose_pixel_pos,face_height,image.shape)
                    bbox_dict[bbox_n_key]=n_phone_bbox
                    print("normed bbox", bbox_dict[bbox_n_key])
                else:
                    pass
                    if VERBOSE: print(f"NO {bbox_key} for", image_id)
    

            ### do imagebackground calcs

            segmentation_mask=sort.get_segmentation_mask(get_bg_segment,image,None,None)
            is_left_shoulder,is_right_shoulder=sort.test_shoulders(segmentation_mask)
            if VERBOSE: print("shoulders",is_left_shoulder,is_right_shoulder)
            hue,sat,val,lum, lum_torso=sort.get_bg_hue_lum(image,segmentation_mask,bbox)  
            if VERBOSE: print("sat values before insert", hue,sat,val,lum,lum_torso)

            if bbox:
                #will do a second round for bbox with same cv2 image
                # bbox,face_landmarks=get_bbox(target_image_id)
                hue_bb,sat_bb, val_bb, lum_bb, lum_torso_bb =sort.get_bg_hue_lum(image,segmentation_mask,bbox)  
                if VERBOSE: print("sat values before insert", hue_bb,sat_bb, val_bb, lum_bb, lum_torso_bb)
                # hue_bb,sat_bb, val_bb, lum_bb, lum_torso_bb =get_bg_hue_lum(img,bbox,facelandmark)
            else:
                hue_bb = sat_bb = val_bb = lum_bb = lum_torso_bb = None

            selfie_bbox=sort.get_selfie_bbox(segmentation_mask)
            if VERBOSE: print("selfie_bbox",selfie_bbox)

        ### save object bbox info
        # session = sort.parse_bbox_dict(session, image_id, PhoneBbox, OBJ_CLS_LIST, bbox_dict)

        if HANDLMS:
            # do hand stuff
            # print("doing HANDLMS, bc is ", HANDLMS)
            pose = SelectPose(image)
            is_hands, hand_landmarks = find_hands(image, pose)
            # print("is_hands", is_hands)
            if not is_hands:
                print(" ------ >>>>>  NO HANDS for ", image_id)
                # print("hand_landmarks", hand_landmarks)

        for _ in range(io.max_retries):
            try:
                # new_entry = Encodings(**insert_dict)
                # session.add(new_entry)
                # session.commit()

                # this is wrong. i need to change back to before, I think.
                if is_body and not mongo_body_landmarks:
                    print("storing n_landmarks")
                    session.query(Encodings).filter(Encodings.image_id == image_id).update({
                        Encodings.is_body: is_body,
                        # Encodings.body_landmarks: body_landmarks
                        Encodings.mongo_body_landmarks: is_body,
                        Encodings.mongo_body_landmarks_norm: 1
                    }, synchronize_session=False)

                    session.query(SegmentTable).filter(SegmentTable.image_id == image_id).update({
                        SegmentTable.mongo_body_landmarks: is_body,
                        SegmentTable.mongo_body_landmarks_norm: 1
                    }, synchronize_session=False)

                    # MySQL
                    ### save image.shape to Images.h and Images.w
                    session.query(Images).filter(Images.image_id == image_id).update({
                        Images.h: image.shape[0],
                        Images.w: image.shape[1]
                    }, synchronize_session=False)

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


                    ### save regular landmarks                
                    if body_landmarks:
                        mongo_collection.update_one(
                            {"image_id": image_id},
                            {"$set": {"body_landmarks": pickle.dumps(body_landmarks)}},
                            upsert=True
                        )
                        print("----------- >>>>>>>>   mongo body_landmarks updated:", image_id)

                    ### save normalized landmarks                
                    if n_landmarks:
                        bboxnormed_collection.update_one(
                            {"image_id": image_id},
                            {"$set": {"nlms": pickle.dumps(n_landmarks)}},
                            upsert=True
                        )
                        print("----------- >>>>>>>>   mongo n_landmarks updated:", image_id)

                if is_hands:
                    # save hand landmarks
                    # print("storing hand_landmarks")


                    session.query(Encodings).filter(Encodings.image_id == image_id).update({
                        # Encodings.body_landmarks: body_landmarks
                        Encodings.mongo_hand_landmarks: is_hands,
                    }, synchronize_session=False)

                    session.query(SegmentTable).filter(SegmentTable.image_id == image_id).update({
                        SegmentTable.mongo_hand_landmarks: is_hands,
                    }, synchronize_session=False)

                    # mongo_hand_collection.update_one(
                    #     {"image_id": image_id},
                    #     {"$set": {"hand_landmarks": pickle.dumps(hand_landmarks)}},
                    #     upsert=True
                    # )
                    pose.store_hand_landmarks(image_id, hand_landmarks, mongo_hand_collection)
                    print("----------- >>>>>>>>   mongo hand_landmarks updated:", image_id)

                    pass

                # Check if the current batch is ready for commit
                # if total_processed % BATCH_SIZE == 0:

                # for testing, comment out the commit
                session.commit()
                
                # print("------ ++++++ MySQL bbbbody/hand_lms updated:", image_id)
                # print("sleeepy temp time")
                # time.sleep(1)
                break  # Transaction succeeded, exit the loop
            except OperationalError as e:
                print(e)
                print(f"[process_image]session.query failed: {task}")
                time.sleep(io.retry_delay)
        else:
            print("no bbox")

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
        print('no image or toooooo smallllll, stored in Images table')

        # I should probably assign no_good here...?
    # Close the session and dispose of the engine before the worker process exits
    close_mongo()
    close_session()

    # store data



def process_image(task):
    #print("process_image this is where the action is")
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


    df = pd.DataFrame(columns=['image_id','is_face','is_body','is_face_distant','face_x','face_y','face_z','mouth_gap','face_landmarks','bbox','face_encodings','face_encodings68_J','body_landmarks'])
    # print(task)
    df.at['1', 'image_id'] = task[0]
    cap_path = capitalize_directory(task[1])
    # print(">> SPLIT >> made DF, about to imread")
    # pr_split = print_get_split(pr_split)

    try:
        # i think i'm doing this twice. I should just do it here. 
        image = cv2.imread(cap_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
        # image is RGB now 
        # this is for when you need to move images into a testing folder structure
        # save_image_elsewhere(image, task)
    except:
        print(f"[process_image]this item failed: {task}")

    # print(">> SPLIT >> done imread, about to find face")
    # pr_split = print_get_split(pr_split)

    if image is not None and image.shape[0]>MINSIZE and image.shape[1]>MINSIZE:
        # Do FaceMesh

        print(">> SPLIT >> about to find_face")
        df = find_face(image, df)
        is_small = 0
        # pr_split = print_get_split(pr_split)

        # Do Body Pose
        # temporarily commenting this out
        # to reactivate, will have to accept return of is_body, body_landmarks
        # df = find_body(image, df)

        # print(">> SPLIT >> done find_body")
        # pr_split = print_get_split(pr_split)

        # for testing: this will save images into folders for is_face, is_body, and none. 
        # only save images that aren't too smallllll
        # save_image_triage(image,df)
    elif image is not None and image.shape[0]>0 and image.shape[1]>0 :
        print('smallllll but still processing')
        # print(task[0], "shape of image", image.shape)
        df = find_face(image, df)
        # print(df)
        is_small = 1
    else:
        print(">> no image", task)
        # I should probably assign no_good here...?
        return

    # store data
    # return
    
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

        if IS_FOLDER is not True:
            # Check if the entry exists in the Encodings table
            image_id = insert_dict['image_id']
            # can I filter this by site_id? would that make it faster or slower? 
            existing_entry = session.query(Encodings).filter_by(image_id=image_id).first()
            print("DB: existing_entry", existing_entry)

        # print(">> SPLIT >> done query for existing_entry")
        # pr_split = print_get_split(pr_split)

        if IS_FOLDER is True or existing_entry is None:
            for _ in range(io.max_retries):
                try:
                    face_encodings68 = insert_dict.pop('face_encodings68', None)
                    face_landmarks = insert_dict.pop('face_landmarks', None)
                    # print(f"trying to store {image_id}")
                    # update_sql = f"UPDATE Encodings SET bbox = '{bbox_json}' WHERE encoding_id = {encoding_id};"
                    # engine.connect().execute(text(update_sql))
                    # Entry does not exist, insert insert_dict into the table
                    new_entry = Encodings(**insert_dict)
                    if face_encodings68: is_encodings = 1
                    else: is_encodings = 0
                    new_entry.mongo_encodings = is_encodings
                    new_entry.mongo_face_landmarks = is_encodings
                    session.add(new_entry)
                    session.commit()
                    # print(f"just added to db")

                    encoding_id = new_entry.encoding_id  # Assuming 'encoding_id' is the name of your primary key column

                    print(f"Newly inserted row has encoding_id: {encoding_id}")
                    # Get the last row ID
                    print("Last Row ID:", encoding_id, "image_id", image_id)


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


                # session.query(Encodings).filter(Encodings.image_id == image_id).update({
                #     Encodings.is_body: is_body,
                #     # Encodings.body_landmarks: body_landmarks
                #     Encodings.mongo_body_landmarks: is_body
                # })

                # session.query(SegmentTable).filter(SegmentTable.image_id == image_id).update({
                #     SegmentTable.mongo_body_landmarks: is_body
                # })

                # # total_processed += 1

                # if body_landmarks:
                #     mongo_collection.update_one(
                #         {"image_id": image_id},
                #         {"$set": {"body_landmarks": body_landmarks}}
                #     )
                #     print("----------- >>>>>>>>   mongo body_landmarks updated:", image_id)


                    break  # Transaction succeeded, exit the loop
                except OperationalError as e:
                    print("exception on new_entry session.commit")
                    print(e)
                    time.sleep(io.retry_delay)

        else:
            print("already exists, not adding")



        # print(">> SPLIT >> done commit new_entry")
        # pr_split = print_get_split(pr_split)



        # insertignore_df(df,"encodings", engine)  ### made it all lower case to avoid discrepancy
    except OperationalError as e:
        print(e)

    # Close the session and dispose of the engine before the worker process exits
    close_mongo()
    close_session()
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
                
                if BODYLMS is True or HANDLMS is True:
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
    print("main")


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
        print(len(folder_paths))
        completed_folders = io.get_csv_aslist(CSV_FOLDERCOUNT_PATH)
        print(len(completed_folders))
        for folder_path in folder_paths:
            
            # if folder_path in THESE_FOLDER_PATHS:
            if folder_path not in completed_folders:

                folder = os.path.join(MAIN_FOLDER,folder_path)
                folder_count += 1
                if not os.path.exists(folder):
                    print(str(folder_count), "no folder here:",folder)
                    continue
                else:
                    print(str(folder_count), folder)

                img_list = io.get_img_list(folder)
                print("len(img_list)")
                print(len(img_list))


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

                    print("batch_site_image_ids", len(batch_site_image_ids))
                    print("batch_site_image_ids", batch_site_image_ids[:5])


                    # query the database for the current batch and return image_id and encoding_id
                    for _ in range(io.max_retries):

                        try:
                            print(f"Processing batch {i//BATCH_SIZE + 1}...")
                            init_session()
                            batch_query = session.query(Images.image_id, Images.site_image_id, Encodings.encoding_id) \
                                                .outerjoin(Encodings, Images.image_id == Encodings.image_id) \
                                                .filter(Images.site_image_id.in_(batch_site_image_ids), Images.site_name_id == site_name_id)
                            batch_results = batch_query.all()

                            all_results.extend(batch_results)
                            print("about to close_session()")
                            # Close the session and dispose of the engine before the worker process exits
                            close_session()

                        except OperationalError as e:
                            print("error getting batch results")
                            print(e)
                            time.sleep(io.retry_delay)
                    print(f"no. all_results: {len(all_results)}")

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

                        print("site_image_id", site_image_id)
                        if site_image_id in results_dict:
                            result = results_dict[site_image_id]
                            # print("in results", result)
                            # print("in results encoding_id", result.encoding_id)
                            if not result.encoding_id:
                                # if it hasn't been encoded yet, add it to the tasks
                                imagepath = os.path.join(folder, img)
                                task = (result.image_id, imagepath)
                                print(task)
                                tasks_to_accomplish.put(task)
                                this_count += 1

                        else: 
                            print("not in results_dict, will process: ", site_image_id)
                        images_left_to_process = images_left_to_process -1 
                        if images_left_to_process < 500: print(f"no. images_left_to_process: {images_left_to_process}")



                    for w in range(NUMBER_OF_PROCESSES):
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
                        imagepath=os.path.join(io.folder_list[site_id], hashed_path)
                    except:
                        print("missing folder for site_id", site_id)
                        continue
                    if row["mongo_face_landmarks"] is not None:
                        # this is a reprocessing, so don't need to test isExist
                        if VERBOSE: print("reprocessing")
                        task = (image_id,imagepath,row["mongo_face_landmarks"], row["mongo_body_landmarks"],row["bbox"])
                    else:
                        task = (image_id,imagepath)

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
            for w in range(NUMBER_OF_PROCESSES):
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


        # # print the output
        # while not tasks_that_are_done.empty():
        #     print("tasks are done")
        #     print(tasks_that_are_done.get())
            count += len(resultsjson)
            print("completed round, total results processed is: ",count)


    end = time.time()
    print (end - start)
    print ("total processed ",count)
    return True

if __name__ == '__main__':
    main()



