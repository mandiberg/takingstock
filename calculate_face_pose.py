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

from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, Boolean, DECIMAL, BLOB, JSON, String, Date, ForeignKey, update, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
# my ORM
from my_declarative_base import Base, Images, Keywords, SegmentTable, ImagesKeywords, Encodings, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

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

# only for triage
sortfolder ="getty_test"

#use in some clean up for getty
http="https://media.gettyimages.com/photos/"

# am I looking on RAID/SSD for a folder? If not, will pull directly from SQL
# if so, also change the site_name_id etc around line 930
IS_FOLDER = True
SITE_NAME_ID = 13
# 2, shutter. 4, istock
# 7 pond5


'''
1   getty
2   shutterstock
3   adobe
4   istock
5   pexels
6   unsplash
7   pond5
8   123rf
9   alamy
10  visualchinagroup
11	picxy
12	pixerf
13	imagesbazaar - doing
14	indiapicturebudget
15	iwaria
16	nappy
17	picha
18	afripics
'''


# MAIN_FOLDER = "/Volumes/RAID54/images_shutterstock"
MAIN_FOLDER = "/Volumes/RAID18/images_bazzar"
# MAIN_FOLDER = "/Volumes/SSD4/images_getty_reDL"
BATCH_SIZE = 25000 # Define how many from each folder in each batch

#temp hack to go 1 subfolder at a time
# THESE_FOLDER_PATHS = ["8/8A", "8/8B","8/8C", "8/8D", "8/8E", "8/8F", "8/80", "8/81", "8/82", "8/83", "8/84", "8/85", "8/86", "8/87", "8/88", "8/89"]
THESE_FOLDER_PATHS = ["9/9C", "9/9D", "9/9E", "9/9F", "9/90", "9/91", "9/92", "9/93", "9/94", "9/95", "9/96", "9/97", "9/98", "9/99"]

# MAIN_FOLDER = "/Volumes/SSD4/adobeStockScraper_v3/images"
# MAIN_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/gettyimages/newimages"
CSV_FOLDERCOUNT_PATH = os.path.join(MAIN_FOLDER, "folder_countout.csv")

IS_SSD=False
BODYLMS = False # only matters if IS_FOLDER is False
SEGMENT = 0 # topic_id set to 0 or False if using HelperTable or not using a segment
HelperTable_name = False #"SegmentHelperMay7_fingerpoint" # set to False if not using a HelperTable


if BODYLMS is True:

    ############# Reencodings #############
    SELECT = "DISTINCT seg1.image_id, seg1.site_name_id, seg1.contentUrl, seg1.imagename, seg1.site_image_id, seg1.mongo_body_landmarks, seg1.mongo_face_landmarks, seg1.bbox"

    SegmentTable_name = 'SegmentOct20'
    FROM =f"{SegmentTable_name} seg1"
    # FROM ="Encodings e"
    QUERY = "seg1.mongo_body_landmarks IS NULL AND seg1.site_name_id > 1"
    SUBQUERY = " "
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

LIMIT = 2000

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

SMALL_MODEL = False
NUM_JITTERS = 1
###############

## CREATING POSE OBJECT FOR SELFIE SEGMENTATION
## none of these are used in this script ##
## just to initialize the object ## 
image_edge_multiplier = [1.5,1.5,2,1.5] # bigger portrait
image_edge_multiplier_sm = [1.2, 1.2, 1.6, 1.2] # standard portrait
face_height_output = 500
motion = {"side_to_side": False, "forward_smile": True, "laugh": False, "forward_nosmile":  False, "static_pose":  False, "simple": False}
EXPAND = False
ONE_SHOT = False # take all files, based off the very first sort order.
JUMP_SHOT = False # jump to random file if can't find a run

sort = SortPose(motion, face_height_output, image_edge_multiplier_sm,EXPAND, ONE_SHOT, JUMP_SHOT)


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
    global mongo_client, mongo_db, mongo_collection
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

def selectSQL():
    init_session()
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
                body_landmarks = pickle.dumps(bodyLms.pose_landmarks)

            
        except:
            print(f"[find_body]this item failed: {image}")
        return is_body, body_landmarks

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
    # print("task is: ",task)
    image_id = task[0] ### is it enc or image_id
    cap_path = capitalize_directory(task[1])
    init_session()
    init_mongo()


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
        is_body, body_landmarks = find_body(image)

        # bbox_json = retro_bbox(image)
        # print(bbox_json)
        # if bbox_json: 

        print("is_body", is_body)
        for _ in range(io.max_retries):
            try:
                # new_entry = Encodings(**insert_dict)
                # session.add(new_entry)
                # session.commit()

                session.query(Encodings).filter(Encodings.image_id == image_id).update({
                    Encodings.is_body: is_body,
                    # Encodings.body_landmarks: body_landmarks
                    Encodings.mongo_body_landmarks: is_body
                })

                session.query(SegmentTable).filter(SegmentTable.image_id == image_id).update({
                    SegmentTable.mongo_body_landmarks: is_body
                })

                # total_processed += 1

                if body_landmarks:
                    mongo_collection.update_one(
                        {"image_id": image_id},
                        {"$set": {"body_landmarks": body_landmarks}}
                    )
                    print("----------- >>>>>>>>   mongo body_landmarks updated:", image_id)
                
                # Check if the current batch is ready for commit
                # if total_processed % BATCH_SIZE == 0:

                session.commit()
                
                print("bbbbody:")
                print(image_id)
                break  # Transaction succeeded, exit the loop
            except OperationalError as e:
                print(e)
                print(f"[process_image]session.query failed: {task}")
                time.sleep(io.retry_delay)
        else:
            print("no bbox")

    else:
        print('no image or toooooo smallllll')
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
        df = find_face(image, df)
        # print(">> SPLIT >> done find_face")
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
    else:
        print('toooooo smallllll')
        # I should probably assign no_good here...?

    # store data
    # print(df)
    try:
        # DEBUGGING --> need to change this back to "encodings"
        # print(df)  ### made it all lower case to avoid discrepancy
        # print(df.at['1', 'face_encodings'])
        dict_df = df.to_dict('index')
        insert_dict = dict_df["1"]

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
                
                if BODYLMS is True:
                    print("do_job via process_image_bodylms:")
                    process_image_bodylms(task)
                else:
                    # landmarks and bbox, so this is an encodings only
                    process_image_enc_only(task)
                    print("process_image_enc_only")


            else:
                print("do_job via regular process_image:", task)
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

        while True:
            init_session()

            # print("about to SQL: ",SELECT,FROM,WHERE,LIMIT)
            resultsjson = selectSQL()    
            print("got results, count is: ",len(resultsjson))
            # print(resultsjson)
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
                # print(row)
                # encoding_id = row["encoding_id"]
                image_id = row["image_id"]
                item = row["contentUrl"]
                hashed_path = row["imagename"]
                site_id = row["site_name_id"]
                if site_id == 1:
                    # print("fixing gettyimages hash")
                    orig_filename = item.replace(http, "")+".jpg"
                    orig_filename = orig_filename.replace(".jpg.jpg", ".jpg")
                    d0, d02 = get_hash_folders(orig_filename)
                    hashed_path = os.path.join(d0, d02, orig_filename)
                
                # gets folder via the folder list, keyed with site_id integer
                imagepath=os.path.join(io.folder_list[site_id], hashed_path)

                if row["mongo_face_landmarks"] is not None:
                    # this is a reprocessing, so don't need to test isExist
                    print("reprocessing")
                    task = (image_id,imagepath,row["mongo_face_landmarks"],row["bbox"])
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



