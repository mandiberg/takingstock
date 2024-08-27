import cv2
import pandas as pd
import os
import time
import sys
import pickle
import base64
import json
import ast
import traceback
import numpy as np
from mediapipe.framework.formats import landmark_pb2

from sqlalchemy import create_engine, text,select, MetaData, Table, Column, Numeric, Integer, VARCHAR, update, Float
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
# my ORM
from my_declarative_base import Base, SegmentTable,ImagesBackground, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Images
import pymongo

#mine
from mp_sort_pose import SortPose
from mp_db_io import DataIO

VIDEO = False
CYCLECOUNT = 1

# keep this live, even if not SSD
SegmentTable_name = 'SegmentOct20'

# SATYAM, this is MM specific
# for when I'm using files on my SSD vs RAID
IS_SSD = True
#IS_MOVE is in move_toSSD_files.py

# I/O utils
io = DataIO(IS_SSD)
db = io.db
# overriding DB for testing
# io.db["name"] = "stock"
# io.db["name"] = "ministock"

# This is for when you only have the segment table. RW SQL query
IS_SEGONLY= True

HSV_CONTROL = False # defining so it doesn't break below, if commented out
# This tells it to pull luminosity. Comment out if not using
if HSV_CONTROL: HSV_BOUNDS = {"LUM_MIN": 0, "LUM_MAX": 40, "SAT_MIN": 0, "SAT_MAX": 1000, "HUE_MIN": 0, "HUE_MAX": 360}
else: HSV_BOUNDS = {"LUM_MIN": 0, "LUM_MAX": 100, "SAT_MIN": 0, "SAT_MAX": 1000, "HUE_MIN": 0, "HUE_MAX": 360}
HSV_BOUNDS["d128_WEIGHT"] = 1
HSV_BOUNDS["HSV_WEIGHT"] = 1
HSV_BOUNDS["LUM_WEIGHT"] = 1
# converts everything to a 0-1 scale
HSV_NORMS = {"LUM": .01, "SAT": 1,  "HUE": 0.002777777778, "VAL": 1}

# this is for controlling if it is using
# all clusters, 
IS_CLUSTER = False
CLUSTER_TYPE = "Poses"
DROP_LOW_VIS = False
# CLUSTER_TYPE = "Clusters"
# number of clusters to analyze -- this is also declared in Clustering_SQL. Move to IO?
N_CLUSTERS = 20
# this is for IS_ONE_CLUSTER to only run on a specific CLUSTER_NO
IS_ONE_CLUSTER = False
CLUSTER_NO = 2

# cut the kids
NO_KIDS = True
USE_PAINTED = True
OUTPAINT = False
INPAINT= True
INPAINT_MAX = {"top":.4,"right":.4,"bottom":.2,"left":.4}
INPAINT_MAX_SHOULDERS = {"top":.4,"right":.15,"bottom":.2,"left":.15}
OUTPAINT_MAX = {"top":.7,"right":.7,"bottom":.2,"left":.7}

BLUR_THRESH_MAX={"top":50,"right":100,"bottom":10,"left":100}
BLUR_THRESH_MIN={"top":0,"right":20,"bottom":10,"left":20}

# BLUR_RADIUS = 200
# SIGMAX=1000
BLUR_RADIUS = 1  ##computationally more expensive
# SIGMAX=10

BLUR_RADIUS = io.oddify(BLUR_RADIUS)
MASK_OFFSET = [50,50,50,50]
if OUTPAINT: from outpainting_modular import outpaint, image_resize
VERBOSE = True
V_VERBOSE= False
SAVE_IMG_PROCESS = False
# this controls whether it is using the linear or angle process
IS_ANGLE_SORT = False

# this control whether sorting by topics
IS_TOPICS = False
N_TOPICS = 30

IS_ONE_TOPIC = False
TOPIC_NO = [15]

#  is isolated,  is business,  babies, 17 pointing
#  is doctor <<  covid
#  is hands
#  phone = 15
#  feeling frustrated
#  is hands to face
#  shout
# 7 is surprise
#  is yoga << planar,  planar,  fingers crossed

# SORT_TYPE = "128d"
# SORT_TYPE ="planar"
SORT_TYPE = "planar_body"
NORMED_BODY_LMS = True

# if planar_body set OBJ_CLS_ID for each object type
# 67 is phone, 63 is laptop, 26: 'handbag', 27: 'tie', 32: 'sports ball'
if SORT_TYPE == "planar_body": OBJ_CLS_ID = 0
else: OBJ_CLS_ID = 0 # if sort is 128d, this gets picked up. borks if starting from image_id

ONE_SHOT = True # take all files, based off the very first sort order.
JUMP_SHOT = True # jump to random file if can't find a run (I don't think this applies to planar?)




METAS_FILE = "metas.csv"

NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES

# if IS_SSD:
#     io.ROOT = io.ROOT_PROD 
# else:
#     io.ROOT = io.ROOT36


if IS_SEGONLY is not True:
    print("production run. IS_SSD is", IS_SSD)

    # # # # # # # # # # # #
    # # for production  # #
    # # # # # # # # # # # #

    # SAVE_SEGMENT controls whether the result will be saved to the db as a new table
    SAVE_SEGMENT = False
    SELECT = "DISTINCT(i.image_id), i.site_name_id, i.contentUrl, i.imagename, i.description e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings68, i.site_image_id"

    # don't need keywords if SegmentTable
    # this is for MM segment table
    # FROM =f"Images i LEFT JOIN Encodings e ON i.image_id = e.image_id INNER JOIN {SegmentTable_name} seg ON i.site_image_id = seg.site_image_id"
    # WHERE = "e.is_face IS TRUE AND e.face_encodings IS NOT NULL AND e.bbox IS NOT NULL AND i.site_name_id = 8 AND i.age_id NOT IN (1,2,3,4)"

    # this is for gettytest3 table
    FROM ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings e ON i.image_id = e.image_id JOIN ImagesClusters ic ON i.image_id = ic.image_id"
    WHERE = "e.is_face IS TRUE AND e.bbox IS NOT NULL AND i.site_name_id = 1 AND k.keyword_text LIKE 'smil%'"
    LIMIT = 500

##################MICHAEL#####################
elif IS_SEGONLY and io.platform == "darwin":

    SAVE_SEGMENT = False
    # no JOIN just Segment table
    SELECT = "DISTINCT(s.image_id), s.site_name_id, s.contentUrl, s.imagename, s.description, s.face_x, s.face_y, s.face_z, s.mouth_gap, s.bbox, s.site_image_id"

    FROM =f"{SegmentTable_name} s "

    # this is the standard segment topics/clusters query for June 2024
    WHERE = "  s.face_x > -33 AND s.face_x < -27 AND s.face_y > -2 AND s.face_y < 2 AND s.face_z > -2 AND s.face_z < 2"

    # HIGHER
    # WHERE = "s.site_name_id != 1 AND face_encodings68 IS NOT NULL AND face_x > -27 AND face_x < -23 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2"

    # WHERE += " AND mouth_gap > 15 "
    # WHERE += " AND s.age_id NOT IN (1,2,3,4) "
    # WHERE += " AND s.age_id > 4 "

    ## To add keywords to search
    # FROM += " JOIN ImagesKeywords ik ON s.image_id = ik.image_id JOIN Keywords k ON ik.keyword_id = k.keyword_id "
    # WHERE += " AND k.keyword_text LIKE 'shout%' "

    if IS_CLUSTER or IS_ONE_CLUSTER:
        FROM += f" JOIN Images{CLUSTER_TYPE} ic ON s.image_id = ic.image_id "
    if IS_TOPICS or IS_ONE_TOPIC:
        FROM += " JOIN ImagesTopics it ON s.image_id = it.image_id "
        WHERE += " AND it.topic_score > .3"
        SELECT += ", it.topic_score" # add description here, after resegmenting
    if NO_KIDS:
        WHERE += " AND s.age_id NOT IN (1,2,3) "
    if HSV_BOUNDS:
        FROM += " JOIN ImagesBackground ibg ON s.image_id = ibg.image_id "
        # WHERE += " AND ibg.lum > .3"
        SELECT += ", ibg.lum, ibg.lum_bb, ibg.hue, ibg.hue_bb, ibg.sat, ibg.sat_bb, ibg.val, ibg.val_bb, ibg.lum_torso, ibg.lum_torso_bb " # add description here, after resegmenting
    ###############
    if OBJ_CLS_ID > 0:
        FROM += " JOIN PhoneBbox pb ON s.image_id = pb.image_id "
        # SELECT += ", pb.bbox_67, pb.conf_67, pb.bbox_63, pb.conf_63, pb.bbox_26, pb.conf_26, pb.bbox_27, pb.conf_27, pb.bbox_32, pb.conf_32 "
        SELECT += ", pb.bbox_"+str(OBJ_CLS_ID)+", pb.conf_"+str(OBJ_CLS_ID)
        WHERE += " AND pb.bbox_"+str(OBJ_CLS_ID)+" IS NOT NULL "
    if SORT_TYPE == "planar_body":
        WHERE += " AND s.mongo_body_landmarks = 1  "
    ###############
    # # join to keywords
    # FROM += " JOIN ImagesKeywords ik ON s.image_id = ik.image_id JOIN Keywords k ON ik.keyword_id = k.keyword_id "
    # WHERE += " AND k.keyword_text LIKE 'surpris%' "

    # # testing mongo
    # FROM += " JOIN Encodings e ON s.image_id = e.image_id "
    # WHERE += " AND e.encoding_id > 2612275"

    # WHERE = "s.site_name_id != 1"
    LIMIT = 6000000

    # TEMP TK TESTING
    # WHERE += " AND s.site_name_id = 8"
######################################
############SATYAM##################
elif IS_SEGONLY and io.platform == "win32":

    SAVE_SEGMENT = False
    # no JOIN just Segment table
    SELECT = "DISTINCT(s.image_id), s.site_name_id, s.contentUrl, s.imagename, s.description, s.face_x, s.face_y, s.face_z, s.mouth_gap, s.bbox, s.site_image_id"
    # SELECT = "DISTINCT(s.image_id), s.site_name_id, s.contentUrl, s.imagename,s.face_x, s.face_y, s.face_z, s.mouth_gap, s.face_landmarks, s.bbox, s.face_encodings68, s.site_image_id, s.body_landmarks"

    FROM =f"{SegmentTable_name} s "

    # this is the standard segment topics/clusters query for April 2024
    # WHERE = " face_encodings68 IS NOT NULL AND face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2"
    WHERE = "  s.face_x > -33 AND s.face_x < -27 AND s.face_y > -2 AND s.face_y < 2 AND s.face_z > -2 AND s.face_z < 2"
    # HIGHER
    # WHERE = "s.site_name_id != 1 AND face_encodings68 IS NOT NULL AND face_x > -27 AND face_x < -23 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2"

    # WHERE += " AND mouth_gap > 15 "
    # WHERE += " AND s.age_id NOT IN (1,2,3,4) "
    # WHERE += " AND s.age_id > 4 "

    ## To add keywords to search
    # FROM += " JOIN ImagesKeywords ik ON s.image_id = ik.image_id JOIN Keywords k ON ik.keyword_id = k.keyword_id "
    # WHERE += " AND k.keyword_text LIKE 'shout%' "

    if IS_CLUSTER or IS_ONE_CLUSTER:
        FROM += f" JOIN Images{CLUSTER_TYPE} ic ON s.image_id = ic.image_id "
    if IS_TOPICS or IS_ONE_TOPIC:
        FROM += " JOIN ImagesTopics it ON s.image_id = it.image_id "
        WHERE += " AND it.topic_score > .3"
        SELECT += ", it.topic_score" # add description here, after resegmenting
    # if NO_KIDS:
    #     WHERE += " AND s.age_id NOT IN (1,2,3) "
    if HSV_BOUNDS:
        FROM += " JOIN ImagesBackground ibg ON s.image_id = ibg.image_id "
        # WHERE += " AND ibg.lum > .3"
        WHERE +=" AND ibg.selfie_bbox IS NOT NULL "
        SELECT += ", ibg.lum, ibg.lum_bb, ibg.hue, ibg.hue_bb, ibg.sat, ibg.sat_bb, ibg.val, ibg.val_bb, ibg.lum_torso, ibg.lum_torso_bb, ibg.selfie_bbox " # add description here, after resegmenting
    ###############
    if OBJ_CLS_ID:
        FROM += " JOIN PhoneBbox pb ON s.image_id = pb.image_id "
        SELECT += ", pb.bbox_67, pb.conf_67, pb.bbox_63, pb.conf_63, pb.bbox_26, pb.conf_26, pb.bbox_27, pb.conf_27, pb.bbox_32, pb.conf_32 "
    if SORT_TYPE == "planar_body":
        WHERE += " AND s.mongo_body_landmarks = 1 "
    ###############
    # # join to keywords
    # FROM += " JOIN ImagesKeywords ik ON s.image_id = ik.image_id JOIN Keywords k ON ik.keyword_id = k.keyword_id "
    # WHERE += " AND k.keyword_text LIKE 'surpris%' "

    # WHERE = "s.site_name_id != 1"
    LIMIT = 1000

    # TEMP TK TESTING
    # WHERE += " AND s.site_name_id = 8"
######################################

motion = {
    "side_to_side": False,
    "forward_smile": True,
    "laugh": False,
    "forward_nosmile":  False,
    "static_pose":  False,
    "simple": False,
}

EXPAND = False
# face_height_output is how large each face will be. default is 750
face_height_output = 1000
# face_height_output = 256

# define ratios, in relationship to nose
# units are ratio of faceheight
# top, right, bottom, left
# image_edge_multiplier = [1, 1, 1, 1] # just face
image_edge_multiplier = [1.5,1.5,2,1.5] # bigger portrait
# image_edge_multiplier = [1.5,1.33, 2.5,1.33] # bigger 2x3 portrait
# image_edge_multiplier = [1.4,2.6,1.9,2.6] # wider for hands
# image_edge_multiplier = [3,5,3,5] # megawide for testing
# image_edge_multiplier = [1.4,3.3,3,3.3] # widerest 16:10 for hands -- actual 2:3
# image_edge_multiplier = [1.3,3.4,2.9,3.4] # slightly less wide 16:10 for hands < Aug 27
# image_edge_multiplier = [1.6,3.84,3.2,3.84] # wiiiiiiiidest 16:10 for hands
# image_edge_multiplier = [1.45,3.84,2.87,3.84] # wiiiiiiiidest 16:9 for hands
# image_edge_multiplier = [1.2,2.3,1.7,2.3] # medium for hands
# image_edge_multiplier = [1.2, 1.2, 1.6, 1.2] # standard portrait
# sort.max_image_edge_multiplier is the maximum of the elements
UPSCALE_MODEL_PATH=os.path.join(os.getcwd(), "models", "FSRCNN_x4.pb")
# construct my own objects
sort = SortPose(motion, face_height_output, image_edge_multiplier,EXPAND, ONE_SHOT, JUMP_SHOT, HSV_BOUNDS, VERBOSE,INPAINT, SORT_TYPE, OBJ_CLS_ID,UPSCALE_MODEL_PATH=UPSCALE_MODEL_PATH)

# start_img_name = "median"
# start_site_image_id = None

start_img_name = "start_image_id"
start_site_image_id = 3279908

# 3279908 hands to face excited
# 6050372 hands to mouth in shock
# 87210848 43008591 11099323 phone held up

# start_site_image_id is not working Aug 2024
# start_img_name = "start_site_image_id"
# start_site_image_id = "F/F4/538577009.jpg"
# start_site_image_id = "0/02/159079944-hopeful-happy-young-woman-looking-amazed-winning-prize-standing-white-background.jpg"
# start_site_image_id = "0/08/158083627-man-in-white-t-shirt-gesturing-with-his-hands-studio-cropped.jpg"

# for PFP
# start_img_name = "start_face_encodings"
# start_site_image_id = [-0.13242901861667633, 0.09738104045391083, 0.003530653193593025, -0.04780442640185356, -0.13073976337909698, 0.07189705967903137, -0.006513072177767754, -0.051335446536540985, 0.1768932193517685, -0.03729865700006485, 0.1137416809797287, 0.13994133472442627, -0.23849385976791382, -0.08209677785634995, 0.06067033112049103, 0.07974598556756973, -0.1882513463497162, -0.24926315248012543, -0.011344537138938904, -0.10508193075656891, 0.010317208245396614, 0.06348179280757904, 0.02852417528629303, 0.06981766223907471, -0.14760875701904297, -0.34729471802711487, -0.014949701726436615, -0.09429284185171127, 0.08592978119850159, -0.11939340829849243, 0.04517041891813278, 0.06180906295776367, -0.1773814857006073, 0.011621855199337006, 0.010536111891269684, 0.12963438034057617, -0.07557092607021332, 0.0027374476194381714, 0.2890719771385193, 0.0692337155342102, -0.17323020100593567, 0.0724603682756424, 0.021229337900877, 0.361629843711853, 0.250482439994812, 0.021974680945277214, 0.018878426402807236, -0.022722169756889343, 0.09668144583702087, -0.29601603746414185, 0.11375367641448975, 0.2568872570991516, 0.11404240131378174, 0.04999732971191406, 0.02831254154443741, -0.15830034017562866, -0.031099170446395874, 0.028748074546456337, -0.180643692612648, 0.13169123232364655, 0.058790236711502075, -0.0858338251709938, 0.029470380395650864, -0.002784252166748047, 0.2532877027988434, 0.07375448942184448, -0.11085735261440277, -0.12285713106393814, 0.11346398293972015, -0.19246435165405273, -0.1447266787290573, 0.054258447140455246, -0.1335202157497406, -0.1264294683933258, -0.23741140961647034, 0.07753928005695343, 0.3753989636898041, 0.08984167128801346, -0.18434450030326843, 0.042485352605581284, -0.08978638052940369, -0.03871896490454674, 0.06451354175806046, 0.08044029772281647, -0.11364202201366425, -0.1158837378025055, -0.10755209624767303, 0.044953495264053345, 0.2573489546775818, 0.049939051270484924, -0.07680445909500122, 0.20810386538505554, 0.09711501002311707, 0.05330953001976013, 0.08986716717481613, 0.0984266921877861, -0.036112621426582336, -0.011795245110988617, -0.15438663959503174, -0.027118921279907227, -0.012514196336269379, -0.11667540669441223, 0.04242435097694397, 0.13383115828037262, -0.18503828346729279, 0.19057676196098328, 0.017584845423698425, -0.005235005170106888, 0.010936722159385681, 0.08952657878398895, -0.1809171438217163, -0.07223983108997345, 0.16210225224494934, -0.264881432056427, 0.3121953308582306, 0.21528613567352295, 0.02137373574078083, 0.12006716430187225, 0.08322857320308685, 0.0802738219499588, -0.013485163450241089, 0.005497157573699951, -0.0893208310008049, -0.06330209970474243, 0.017513029277324677, -0.007281661033630371, 0.06451432406902313, 0.10179871320724487]

# start_img_name = "start_bbox"
# start_site_image_id = [94, 428, 428,0]

# no gap
# start_site_image_id = "5/58/95516714-happy-well-dressed-man-holding-a-gift-on-white-background.jpg"


d = None


# override io.db for testing mode
# db['name'] = "123test"

if db['unix_socket']:
    # for MM's MAMP config
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

mongo_client = pymongo.MongoClient(io.dbmongo['host'])
mongo_db = mongo_client[io.dbmongo['name']]
io.mongo_db = mongo_db


# handle the table objects based on CLUSTER_TYPE
ClustersTable_name = CLUSTER_TYPE
ImagesClustersTable_name = "Images"+CLUSTER_TYPE

class Clusters(Base):
    __tablename__ = ClustersTable_name

    cluster_id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_median = Column(BLOB)

class ImagesClusters(Base):
    __tablename__ = ImagesClustersTable_name

    image_id = Column(Integer, ForeignKey(Images.image_id, ondelete="CASCADE"), primary_key=True)
    cluster_id = Column(Integer, ForeignKey(f'{ClustersTable_name}.cluster_id', ondelete="CASCADE"))

if IS_CLUSTER or IS_ONE_CLUSTER:
    # select cluster_median from Clusters
    results = session.execute(select(Clusters.cluster_id, Clusters.cluster_median)).fetchall()
    print("results", results)
    
    # store the results in a dictionary where the key is the cluster_id
    if results:
        cluster_medians = {}
        for i, row in enumerate(results, start=1):
            cluster_median = pickle.loads(row.cluster_median)
            cluster_medians[i] = cluster_median
            print("cluster_medians", i, cluster_median)
        sort.set_cluster_medians(cluster_medians)




# mongo_collection = mongo_db[io.dbmongo['collection']]


# construct mediapipe objects
# mp_drawing = mp.solutions.drawing_utils

# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,min_detection_confidence=0.5)



###################
# SQL  FUNCTIONS  #
###################

def selectSQL(cluster_no=None, topic_no=None):
    # print(f"cluster_no is")
    # print(cluster_no)
    if cluster_no is not None or topic_no is not None:
        cluster = " "
        if IS_CLUSTER or IS_ONE_CLUSTER:
            cluster +=f"AND ic.cluster_id = {str(cluster_no)} "
        if IS_TOPICS or IS_ONE_TOPIC:
            # cluster +=f"AND it.topic_id = {str(topic_no)} "
            if isinstance(topic_no, list):
                # Convert the list into a comma-separated string
                topic_ids = ', '.join(map(str, topic_no))
                # Use the IN operator to check if topic_id is in the list of values
                cluster += f"AND it.topic_id IN ({topic_ids}) "
            else:
                # If topic_no is not a list, simply check for equality
                if IS_ONE_TOPIC: cluster += f"AND it.topic_id = {str(topic_no)} "            
    else:
        cluster=""
    # print(f"cluster SELECT is {cluster}")
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} {cluster} LIMIT {str(LIMIT)};"
    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    return(resultsjson)


def select_cluster_median(cluster_no):
    cluster_selectsql = f"SELECT c.cluster_median FROM Clusters c WHERE cluster_id={cluster_no};"
    result = engine.connect().execute(text(cluster_selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    cluster_median = (resultsjson[0]['cluster_median'])
    return(cluster_median)




def save_segment_DB(df_segment):
    #save the df to a table
    # Assuming you have your DataFrame named 'df' containing the query results
    for _, row in df_segment.iterrows():
        instance = SegmentTable(
            image_id=row['image_id'],
            site_name_id=row['site_name_id'],
            contentUrl=row['contentUrl'],
            imagename=row['imagename'],
            face_x=row['face_x'],
            face_y=row['face_y'],
            face_z=row['face_z'],
            mouth_gap=row['mouth_gap'],
            face_landmarks=pickle.dumps(row['face_landmarks'], protocol=3),
            bbox=row['bbox'],
            face_encodings=pickle.dumps(row['face_encodings'], protocol=3),
            site_image_id=row['site_image_id']
        )
        session.add(instance)
    session.commit()



###################
# SORT FUNCTIONS  #
###################


# need to pass through start_img_enc rather than start_img_name
# for linear it is in the df_enc, but for itter, the start_img_name is in prev df_enc
# takes a dataframe of images and encodings and returns a df sorted by distance
def sort_by_face_dist_NN(df_enc):
    
    # create emtpy df_sorted with the same columns as df_enc
    df_sorted = pd.DataFrame(columns = df_enc.columns)

    if sort.CUTOFF < len(df_enc.index): itters = sort.CUTOFF
    else: itters = len(df_enc.index)
    

    # input enc1, df_128_enc, df_33_lmsNN
    # df = pd.DataFrame(face_distances, columns =['dist', 'folder', 'filename','site_name_id','face_landmarks', 'bbox'])


    # find the df_enc row with image_id = 10498233
    # test_row = df_enc.loc[df_enc['image_id'] == 10498233]
    # # print body_landmarks for this row
    # print("body_landmarks for test_row")
    # test_lms = test_row['body_landmarks']
    # print(test_lms)

    for i in range(itters):

        ## Find closest
        try:
            # send in both dfs, and return same dfs with 1+ rows sorted
            print("BEFORE sort_by_face_dist_NN _ for loop df_enc is", df_enc)

            df_enc, df_sorted = sort.get_closest_df_NN(df_enc, df_sorted)
    
            print("AFTER sort_by_face_dist_NN _ for loop df_enc is", df_enc)
            print("AFTER sort_by_face_dist_NN _ for loop df_sorted is", df_sorted)

            # # test to see if body_landmarks for row with image_id = 5251199 still is the same as test_lms
            # retest_row = df_enc.loc[df_enc['image_id'] == 10498233]
            # print("body_landmarks for retest_row")
            # retest_lms = retest_row['body_landmarks']
            # print(retest_lms)
            # calculate any different between test_lms to retest_lms


            dist = df_sorted.iloc[-1]['dist_enc1']
            print("sort_by_face_dist_NN _ for loop dist is", dist)

            # Break out of the loop if greater than MAXDIST
            if ONE_SHOT:
                df_sorted = pd.concat([df_sorted, df_enc])
                # only return the first x rows
                df_sorted = df_sorted.head(sort.CUTOFF)
                print("one shot, breaking out", df_sorted)
                break
            elif dist > sort.MAXD and sort.SHOT_CLOCK != 0:
                print("should breakout, dist is", dist)
                break

        except Exception as e:
            print("exception on going to get closest")
            print(str(e))
            traceback.print_exc()

    # use the colum site_name_id to asign the value of io.folder_list[site_name_id] to the folder column
    df_sorted['folder'] = df_sorted['site_name_id'].apply(lambda x: io.folder_list[x])
    
    # rename the distance column to dist
    df_sorted.rename(columns={'dist_enc1': 'dist'}, inplace=True)

    print("df_sorted", df_sorted)

    # make a list of df_sorted dist
    dist_list = df_sorted['dist'].tolist()
    print("dist_list", dist_list)
    
    # print all columns in df_sorted
    print("df_sorted.columns", df_sorted.columns)
    
    ## Set a start_img_name for next round --> for clusters
    try:
        #last_file = imagename from last row in df_sorted
        last_file = df_sorted.iloc[-1]['imagename']
        print("last_file ",last_file)
    except:
        last_file = this_start
        print("last_file is this_start",last_file)
    sort.counter_dict["start_img_name"] = last_file

    return df_sorted



def cycling_order(CYCLECOUNT, sort):
    img_array = []
    cycle = 0 
    # metamedian = get_metamedian(angle_list)
    metamedian = sort.metamedian
    d = sort.d

    print("CYCLE to test: ",cycle)

    while cycle < CYCLECOUNT:
        print("CYCLE: ",cycle)
        for angle in sort.angle_list:
            print("angle: ",str(angle))
            # # print(d[angle].iloc[(d[angle][SECOND_SORT]-metamedian).abs().argsort()[:2]])
            # # print(d[angle].size)
            try:
                # I don't remember exactly how this segments the data...!!!
                # [:CYCLECOUNT] gets the first [:0] value on first cycle?
                # or does it limit the total number of values to the number of cycles?
                print(d[angle])
                
                #this is a way of finding the image with closest second sort (Y)
                #mystery value is the image to be matched? 
                print("second sort, metamedian ",d[angle][sort.SECOND_SORT],sort.metamedian)
                mysteryvalue = (d[angle][sort.SECOND_SORT]-sort.metamedian)
                print('mysteryvalue ',mysteryvalue)
                #is mystery value a df?
                #this is finding the 
                mysterykey = mysteryvalue.abs().argsort()[:CYCLECOUNT]
                print('mysterykey: ',mysterykey)
                closest = d[angle].iloc[mysterykey]
                closest_file = closest.iloc[cycle]['imagename']
                closest_mouth = closest.iloc[cycle]['mouth_gap']
                print('closest: ')
                print(closest_file)
                img = cv2.imread(closest_file)
                height, width, layers = img.shape
                size = (width, height)
                img_array.append(img)
            except:
                print('failed cycle angle:')
                # print('failed:',row['imagename'])
        print('finished a cycle')
        sort.angle_list.reverse()
        cycle = cycle +1
        # print(angle_list)
    return img_array, size

def prep_encodings_NN(df_segment):
    def create_hsv_list(row):
        if row['hue_bb'] >= 0:
            # print("create_hsv_list bb", [row['hue_bb'], row['sat_bb'], row['lum_bb']])
            return [row['hue_bb']*HSV_NORMS["HUE"], row['sat_bb']*HSV_NORMS["SAT"], row['val_bb']*HSV_NORMS["VAL"]]
        else:
            return [row['hue']*HSV_NORMS["HUE"], row['sat']*HSV_NORMS["SAT"], row['val']*HSV_NORMS["VAL"]]    
    def create_lum_list(row):
        if row['lum_torso_bb'] >= 0 and row['lum_torso_bb'] != 1:
            # print("create_hsv_list bb", [row['hue_bb'], row['sat_bb'], row['lum_bb']])
            return [row['lum']*HSV_NORMS["LUM"], row['lum_torso_bb']*HSV_NORMS["LUM"]]
        else:
            return [row['lum']*HSV_NORMS["LUM"], row['lum_torso']*HSV_NORMS["LUM"]]    
    # def json_to_list(row):      
    #     bbox = row["bbox_"+str(OBJ_CLS_ID)]
    #     if bbox: return [bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]]
    #     else: return None
      
    #     bbox = json.loads(row["bbox_"+str(OBJ_CLS_ID)])
    #     bbox_list = [bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]]
    #     return bbox_list
    # def json_to_list(row):
    #     bbox = row["bbox_"+str(OBJ_CLS_ID)]
    #     print("type of bbox", type(bbox))
    #     print("bbox", bbox)
    #     bbox_json = json.loads((bbox))
    #     print("bbox_json", bbox_json)
    #     print("type of bbox_json", type(bbox_json))
    #     # If bbox is None, return None
    #     if bbox is None:
    #         print("bbox is None")
    #         return None
    #     else:
    #         print("type of bbox", type(bbox))
        
    #     # If bbox is a string, try to parse it as JSON
    #     if isinstance(bbox, str):
    #         print("going to ttry to load json")
    #         try:
    #             print("loading json")
    #             bbox_dict = json.loads(json.dumps(bbox))
    #             # bbox_dict = ast.literal_eval(bbox)
    #             # bbox_dict = json.loads(bbox)
    #             print("loaded json, type is", type(bbox_dict))
    #         except json.JSONDecodeError:
    #             print(f"Invalid JSON string: {bbox}")
    #             return None
    #     # If bbox is already a dictionary, use it directly
    #     elif isinstance(bbox, dict):
    #         bbox_dict = bbox
    #     else:
    #         print(f"Unexpected bbox format: {bbox}")
    #         return None
    #     print("type of bbox_dict", type(bbox_dict))        
    #     # Now bbox_dict should be a dictionary, so we can safely use .get()
    #     return [bbox_dict.get("left"), bbox_dict.get("top"), bbox_dict.get("right"), bbox_dict.get("bottom")]
    
    
    ########################################
    # THIS IS WHERE THE BUGS NEED TO BE FIXED
    ########################################

    print("degugging df_segment prep encodings", df_segment.columns)      
    # drop rows where body_landmarks_normalized is None
    df_segment = df_segment.dropna(subset=['body_landmarks_normalized'])
    print("df_segment length", len(df_segment.index))
    
    # print rows where body_landmarks_normalized is None
    null_encodings = df_segment[df_segment['body_landmarks_normalized'].isnull()]
    if not null_encodings.empty:
        print("Rows with invalid body_landmarks_normalized data:")
        print(null_encodings)
        
    print("debugging biiiiiiig nose landmarks")
    print(df_segment.size)
    print(df_segment['body_landmarks_normalized'])
    print(df_segment['body_landmarks'])


    # create a column for the hsv values using df_segment.apply(lambda row: create_hsv_list(row), axis=1)
    df_segment['hsv'] = df_segment.apply(lambda row: create_hsv_list(row), axis=1)
    df_segment['lum'] = df_segment.apply(lambda row: create_lum_list(row), axis=1)
    # load the OBJ_CLS_ID bbox as list into obj_bbox_list
    # df_segment['obj_bbox_list'] = df_segment.apply(lambda row: json_to_list(row), axis=1)
    if OBJ_CLS_ID > 0: 
        df_segment['obj_bbox_list'] = df_segment.apply(sort.json_to_list, axis=1)
        null_bboxes = df_segment[df_segment['obj_bbox_list'].isnull()]
        if not null_bboxes.empty:
            print("Rows with invalid bbox data:")
            print(null_bboxes)

    print("df_segment length", len(df_segment.index))
    if SORT_TYPE == "planar_body" and DROP_LOW_VIS:
        # if planar_body drop rows where self.BODY_LMS are low visibility
        df_segment['hand_visible'] = df_segment.apply(lambda row: any(sort.test_landmarks_vis(row)), axis=1)

        # delete rows where hand_visible is false
        df_segment = df_segment[df_segment['hand_visible'] == True].reset_index(drop=True)
        # df_segment = df_segment[df_segment['hand_visible'] == True]
        print("df_segment length visible hands", len(df_segment.index))

    return df_segment


def compare_images(last_image, img, face_landmarks, bbox):
    is_face = None
    face_diff = 100 # declaring full value, for first round
    skip_face = False
    #crop image here:
    if sort.EXPAND:
        cropped_image = sort.expand_image(img, face_landmarks, bbox)
        is_inpaint = False
    else:
        cropped_image, is_inpaint = sort.crop_image(img, face_landmarks, bbox)
    # print("cropped_image: ",cropped_image)
    # if cropped_image is not None and len(cropped_image)>1 :
    #     print(" >> have a cropped image trying to save", cropped_image.shape)
    # elif cropped_image is not None and len(cropped_image)==1 :
    #     print(" >> bad crop, will try inpainting and try again")
    # elif cropped_image is None:
    #     print(" >> no image here, trying next")

    # this code takes image i, and blends it with the subsequent image
    # next step is to test to see if mp can recognize a face in the image
    # if no face, a bad blend, try again with i+2, etc. 
    if cropped_image is not None and not is_inpaint:
        if VERBOSE: print("have a cropped image trying to save", cropped_image.shape)
        # try:
        #     print("last_image is ", type(last_image))
        # except:
        #     print("couldn't test last_image")
        try:
            if not sort.counter_dict["first_run"]:
                if VERBOSE:  print("testing is_face")
                if SORT_TYPE == "planar_body":
                    # skipping test_pair for body, b/c it is meant for face
                    is_face = True
                else:
                    is_face = sort.test_pair(last_image, cropped_image)
                    if is_face:
                        if VERBOSE: print("testing mse to see if same image")
                        face_diff = sort.unique_face(last_image,cropped_image)
                        # if VERBOSE: print ("mse ", mse) ########## mse not a variable
                    else:
                        print("failed is_face test")
                        # use cv2 to place last_image and cropped_image side by side in a new image


                        height = max(last_image.shape[0], cropped_image.shape[0])
                        last_image = cv2.resize(last_image, (last_image.shape[1], height))
                        cropped_image = cv2.resize(cropped_image, (cropped_image.shape[1], height))
                        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
                        # Concatenate images horizontally
                        combined_image = cv2.hconcat([last_image, cropped_image])
                        outpath_notface = os.path.join(sort.counter_dict["outfolder"],"notface",sort.counter_dict['last_description'][:30]+".jpg")
                        # sort.not_make_face.append(outpath_notfacecombined_image) ########## variable name error
                        # Save the new image
                        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

            else:
                print("first round, skipping the pair test")
        except:
            print("last_image try failed")
        # if is_face or first_run and sort.resize_factor < sort.resize_max:
        if face_diff > sort.FACE_DUPE_DIST or sort.counter_dict["first_run"]:
            sort.counter_dict["first_run"] = False
            last_image = cropped_image
            sort.counter_dict["good_count"] += 1
        else: 
            print("pair do not make a face, skipping")
            sort.counter_dict["isnot_face_count"] += 1
            return None, None, True
        
    elif cropped_image is None and sort.counter_dict["first_run"]:
        print("first run, but bad first image")
        last_image = cropped_image
        sort.counter_dict["cropfail_count"] += 1
    elif is_inpaint:
        print("bad crop, will try inpainting and try again")
        sort.counter_dict["inpaint_count"] += 1
    elif cropped_image is None:
        print("no image or resize to great, trying next")
        sort.counter_dict["cropfail_count"] += 1
        skip_face = True
    # print(type(cropped_image),"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    return cropped_image, face_diff, skip_face


def print_counters():
    print("good_count")
    print(sort.counter_dict["good_count"])
    print("isnot_face_count")
    print(sort.counter_dict["isnot_face_count"])
    print("cropfail_count")
    print(sort.counter_dict["cropfail_count"])
    print("sort.negmargin_count")
    print(sort.negmargin_count)
    print("sort.toosmall_count")
    print(sort.toosmall_count)
    print("failed_dist_count")
    print(sort.counter_dict["failed_dist_count"])
    print("total count")
    print(sort.counter_dict["counter"])
    print("inpaint count")
    print(sort.counter_dict["inpaint_count"])


def const_imgfilename_NN(UID, df, imgfileprefix):
    # print("filename", filename)
    # UID = filename.split('-id')[-1].split("/")[-1].replace(".jpg","")
    # print("UID ",UID)
    counter_str = str(sort.counter_dict["counter"]).zfill(len(str(df.size)))  # Add leading zeros to the counter
    imgfilename = imgfileprefix+"_"+str(counter_str)+"_"+str(UID)+".jpg"
    print("imgfilename ",imgfilename)
    return imgfilename

def fetch_selfie_bbox(target_image_id):
    select_image_ids_query = (select(ImagesBackground.image_id,ImagesBackground.selfie_bbox,ImagesBackground.is_left_shoulder, ImagesBackground.is_right_shoulder).filter(ImagesBackground.image_id == target_image_id))

    result = session.execute(select_image_ids_query).fetchall()
    image_id,selfie_bbox,is_left_shoulder,is_right_shoulder=result[0]
    # print("before json selie bbox",selfie_bbox,type(selfie_bbox))
    if type(selfie_bbox)==str:
        selfie_bbox=json.loads(selfie_bbox)
    # print("after json selie bbox",selfie_bbox,type(selfie_bbox))
    ##making cutoffs##
    if selfie_bbox:
        selfie_bbox["left"]=np.minimum(selfie_bbox["left"],BLUR_THRESH_MAX["left"])
        selfie_bbox["right"]=np.minimum(selfie_bbox["right"],BLUR_THRESH_MAX["right"])
        selfie_bbox["bottom"]=np.minimum(selfie_bbox["bottom"],BLUR_THRESH_MAX["bottom"])
        selfie_bbox["top"]=np.minimum(selfie_bbox["top"],BLUR_THRESH_MAX["top"])

        selfie_bbox["left"]=np.maximum(selfie_bbox["left"],BLUR_THRESH_MIN["left"])
        selfie_bbox["right"]=np.maximum(selfie_bbox["right"],BLUR_THRESH_MIN["right"])
        selfie_bbox["bottom"]=np.maximum(selfie_bbox["bottom"],BLUR_THRESH_MIN["bottom"])
        selfie_bbox["top"]=np.maximum(selfie_bbox["top"],BLUR_THRESH_MIN["top"])

    else:
        print("selfie bbox calculation not done")
    return selfie_bbox, is_left_shoulder,is_right_shoulder


def merge_inpaint(inpaint_image,img,extended_img,extension_pixels,selfie_bbox,blur_radius=BLUR_RADIUS):
    is_consistent = is_ext_UL_consistent = is_ext_top_consistent = is_inpaint_UL_consistent = is_inpaint_top_consistent = False
    height, width = img.shape[:2]
    top, bottom, left, right = extension_pixels["top"], extension_pixels["top"]+height, extension_pixels["left"],extension_pixels["left"]+width
    print("top, bottom, left, right", top, bottom, left, right)
    # test to see if top strip is consistent -- eg a seamless background
    area = [0,selfie_bbox["top"]],[0,width]
    if area[0][1] > 0 and area[1][1] > 0:
        is_consistent = sort.test_consistency(img,area)
    print("is_consistent", is_consistent)

    corners = sort.define_corners(extension_pixels,img.shape)
    if corners["top_left"]:
        # doubling test area, to compare it to areas around it
        test_corner = [[corners["top_left"][0][0],io.oddify(corners["top_left"][0][1]*1.5)],[corners["top_left"][1][0],io.oddify(corners["top_left"][1][1]*1.5)]]
        if test_corner[0][1] > 0 and test_corner[1][1] > 0:
            is_ext_UL_consistent = sort.test_consistency(extended_img,test_corner,10)
            is_inpaint_UL_consistent = sort.test_consistency(inpaint_image,test_corner,10)
        test_top = [[0,top+selfie_bbox["top"]],[0,inpaint_image.shape[1]]]
        if test_top[0][1] > 0 and test_top[1][1] > 0:                        
            is_ext_top_consistent = sort.test_consistency(extended_img,test_top,10)
            is_inpaint_top_consistent = sort.test_consistency(inpaint_image,test_top,10)
    print("is_ext_UL_consistent", is_ext_UL_consistent)

    mask_top = np.zeros(np.shape(inpaint_image))
    mask_left = np.zeros(np.shape(inpaint_image))
    mask_bottom = np.zeros(np.shape(inpaint_image))
    mask_right = np.zeros(np.shape(inpaint_image))

    mask_top[:top,:] = [255,255,255]
    mask_left[:,:left] = [255,255,255]
    mask_bottom[bottom:,:] = [255,255,255]
    mask_right[:,right:] = [255,255,255]

    blur_radius_left=io.oddify(selfie_bbox['left']*blur_radius)
    blur_radius_right=io.oddify(selfie_bbox['right']*blur_radius)
    blur_radius_top=io.oddify(extension_pixels['top']*blur_radius)
    blur_radius_bottom=io.oddify(extension_pixels['bottom']*blur_radius)
    
    mask_left = cv2.blur(mask_left, (blur_radius_left, 1))
    mask_right = cv2.blur(mask_right, (blur_radius_right, 1))
    mask_top = cv2.blur(mask_top, (1, blur_radius_top))
    mask_bottom = cv2.blur(mask_bottom, (1, blur_radius_bottom))

    # print("extension_pixels", extension_pixels)
    # print("selfie_bbox", selfie_bbox)

    # add the masks, keeping whites white, and allowing blacks to get full black
    mask=np.maximum(mask_left+mask_right,mask_top+mask_bottom)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # correction mask
    # if top is all same color, then don't use inpaint, leave mask black and use CV2
    if is_consistent and is_ext_UL_consistent and is_ext_top_consistent:
        if top <= 10:
            print("small top", top)
            invert_dist = selfie_bbox["top"]
            invert_blur = io.oddify(invert_dist/2)
        else: 
            invert_dist = top*2
            invert_blur = blur_radius_top
        print("invert_dist, invert_blur", invert_dist, invert_blur)
        mask_top_invert = np.zeros(np.shape(inpaint_image))
        mask_top_invert[:invert_dist,:] = [255,255,255]
        mask_top_invert = cv2.blur(mask_top_invert, (1, invert_blur))
        # mask_top_invert = cv2.bitwise_not(mask_top_invert) 
        # cv2.imshow("mask_top_invert",mask_top_invert)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # mask=np.minimum(mask,mask_top_invert)
        mask=mask-mask_top_invert
        mask = np.where(mask < 0, 0, mask) # zero out any negative numbers
    elif not is_inpaint_UL_consistent and not is_inpaint_top_consistent:
        return None, None
    # increase the white values by 4x, while keeping the black at 0
    # did this get lost???????
    
    # Expand the mask dimensions to match the image
    inpaint_merge = extended_img * (1 - mask / 255) + (mask / 255) * inpaint_image
    inpaint_merge=np.array(inpaint_merge,dtype=np.uint8)
    return inpaint_merge, mask

def extend_cv2(extended_img,mask,iR=3,method="NS"):
    if method=="NS":
        flags=cv2.INPAINT_NS
    elif method=="TELEA":
        flags=cv2.INPAINT_TELEA
    else:print("BRO!!! chose either NS or TELEA ")
    inpaint = cv2.inpaint(extended_img, mask, flags=flags, inpaintRadius=iR)
    return inpaint

def shift_bbox(bbox, extension_pixels):
    # if sort.VERBOSE:print("extension_pixelssssssssssssssssssssssssss",extension_pixels)
    x0,y0=extension_pixels["left"],extension_pixels["top"]
    print("before shifting",bbox)
    bbox['left']   = bbox['left']   + x0
    bbox['right']  = bbox['right']  + x0
    bbox['top']    = bbox['top']    + y0
    bbox['bottom'] = bbox['bottom'] + y0
    if sort.VERBOSE:print("after shifting",bbox)
    return bbox

def linear_test_df(df_sorted,df_segment,cluster_no, itter=None):
    def save_image_metas(row):
        print("row", row)
        print("save_image_metas for use in TTS")
        # parent_row = df_segment[df_segment['image_id'] == row['image_id']]
        # image_id = parent_row['image_id'].values[0] #NON NN
        image_id = row['image_id']
        if row['description']: description = row['description']
        else: description = None
        try: topic_score = row['topic_score']
        except: topic_score = 0
        # use image_id to retrieve description from mysql database 
        # this is temporary until I resegment the images with description in the segment
        # try:
        #     description = session.query(Images.description).filter(Images.image_id == image_id).first()
        # except Exception as e:
        #     traceback.print_exc()
        #     print(str(e))
        # description = parent_row['description'].values[0]
        if IS_TOPICS or IS_ONE_TOPIC:
            metas = [image_id, description[0], topic_score]
            metas_path = os.path.join(sort.counter_dict["outfolder"],METAS_FILE)
            io.write_csv(metas_path, metas)
        # print(image_id, description[0], topic_score)
        # return([image_id, description[0], topic_score])

    def in_out_paint(img, row):
        def check_extension(shape, extension_pixels, threshold):
            key = 0
            for index, (key, value) in enumerate(extension_pixels.items()):
                if index in [0,2]: dim = shape[0]
                else: dim = shape[1]
                ratio_factor = extension_pixels[key] / dim
                print("ratio_factor", ratio_factor)
                if ratio_factor > threshold[key]:
                    if VERBOSE: print("extension too big, not inpainting")
                    return False
            if VERBOSE: print("extension is big, going to be inpainting")
            return True

        cropped_image = None
        face_diff=None
        bailout=False
        extension_pixels=sort.get_extension_pixels(img)
        if sort.VERBOSE:print("extension_pixels",extension_pixels)
        # inpaint_file=os.path.join(os.path.join(os.path.dirname(row['folder']), "inpaint", os.path.basename(row['folder'])),row['filename'])
        inpaint_file=os.path.join(os.path.dirname(row['folder']), os.path.basename(row['folder'])+"_inpaint",row['imagename'])
        print("inpaint_file", inpaint_file)
        if USE_PAINTED and os.path.exists(inpaint_file):
            if sort.VERBOSE: print("path exists, loading image",inpaint_file)
            inpaint_image=cv2.imread(inpaint_file)
        else:
            if sort.VERBOSE: print("path doesnt exist, in_out_painting now")
            # print("maxkey", maxkey)
            # print("extension_pixels[maxkey]", extension_pixels[maxkey])
            ##################
            selfie_bbox, is_left_shoulder,is_right_shoulder=fetch_selfie_bbox(row['image_id'])
            if selfie_bbox["left"]==0: 
                if VERBOSE: print("head hits the top of the image, skipping -------------------> bailout !!!!!!!!!!!!!!!!!")
                do_inpaint = False
                bailout=True
            elif row["site_name_id"] == 2 and extension_pixels["bottom"]>0: 
                if VERBOSE: print("shutter at the bottom, skipping -------------------> bailout !!!!!!!!!!!!!!!!!")
                do_inpaint = False
                bailout=True
            elif is_left_shoulder or is_right_shoulder: 
                # if the selfie bbox is touching the R/L side of the image
                # but doesn't reach the bottom, it means there is a shoulder in the image
                # do_inpaint, but lower the threshold on left and right
                do_inpaint = check_extension(img.shape, extension_pixels, INPAINT_MAX_SHOULDERS)
                pass
            else:
                do_inpaint = check_extension(img.shape, extension_pixels, INPAINT_MAX)
            ##################
            if do_inpaint and not bailout:
                directory = os.path.dirname(inpaint_file)
                # Create the directory if it doesn't exist (creates directories even if skips below because extension too large)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                # maxkey = max(extension_pixels, key=lambda y: abs(extension_pixels[y]))
                print("inpainting small extension")
                # extimg is 50px smaller and mask is 10px bigger
                extended_img,mask,cornermask=sort.prepare_mask(img,extension_pixels)
                if SAVE_IMG_PROCESS:
                    cv2.imwrite(inpaint_file+"1_prepmask.jpg",extended_img)
                    cv2.imwrite(inpaint_file+"2_mask.jpg",mask)
                    cv2.imwrite(inpaint_file+"2.5_cornermask.jpg",cornermask)
                extended_img=extend_cv2(extended_img,mask,iR=3,method="NS")
                if SAVE_IMG_PROCESS: cv2.imwrite(inpaint_file+"3_extendcv2.jpg",extended_img)
                
                extended_img=extend_cv2(extended_img,cornermask,iR=3,method="TELEA")
                if SAVE_IMG_PROCESS: cv2.imwrite(inpaint_file+"3.5_extendcv2_corners.jpg",extended_img)

                inpaint_image=sort.extend_lama(extended_img, mask, downsampling_scale = 8)
                # print("inpaint_image shape after lama extend",np.shape(inpaint_image))
                # inpaint_image = inpaint_image[y:y+h, x:x+w]
                inpaint_image = inpaint_image[0:np.shape(extended_img)[0],0:np.shape(extended_img)[1]]
                # inpaint_image = cv2.crop(inpaint_image, (np.shape(extended_img)[1],np.shape(extended_img)[0]))
                # print("inpaint_image shape after transform",np.shape(inpaint_image))
                # print("extended_img shape after transform",np.shape(extended_img))                
                if SAVE_IMG_PROCESS: cv2.imwrite(inpaint_file+"4_premerge.jpg",inpaint_image)

                ### use inpainting for the extended part, but use original for non extend to keep image sharp ###
                # inpaint_image[extension_pixels["top"]:extension_pixels["top"]+np.shape(img)[0],extension_pixels["left"]:extension_pixels["left"]+np.shape(img)[1]]=img
                # move the boundary of the blur in 50px
                ########
                # inpaint_image=merge_inpaint(inpaint_image,img,extended_img,extension_pixels)
                inpaint_image, blurmask = merge_inpaint(inpaint_image,img,extended_img,extension_pixels,selfie_bbox)
                # cv2.imwrite(inpaint_file+"5_aftmerge.jpg",inpaint_image)
                # cv2.imshow('inpaint_image', inpaint_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if inpaint_image is not None:
                    print("we have an inpaint_image")
                    if SAVE_IMG_PROCESS:  cv2.imwrite(inpaint_file+"6_blurmask.jpg",blurmask)
                    ########
                    if SAVE_IMG_PROCESS: cv2.imwrite(inpaint_file+"_inpaint.jpg",inpaint_image) #for testing out
                    else: cv2.imwrite(inpaint_file,inpaint_image) #temp comment out
                    print("shape of inpaint_image",np.shape(inpaint_image))
                    print("inpainting done", inpaint_file,"shape",np.shape(inpaint_image))
                else: 
                    print("inpainting failed")
                    bailout=True
            elif check_extension(img.shape, extension_pixels, OUTPAINT_MAX) and OUTPAINT:
                directory = os.path.dirname(inpaint_file)
                # Create the directory if it doesn't exist (creates directories even if skips below because extension too large)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                print("outpainting medium extension")
                inpaint_image=outpaint(img,extension_pixels,downsampling_scale=1,prompt="",negative_prompt="")
                cv2.imwrite(inpaint_file,inpaint_image) 
            else:
                print("too big to inpaint -- attempting to bailout")
                # inpaint_image=0
                bailout=True
        if not bailout:
            print("not bailing out, going to compare_images")
            bbox=shift_bbox(row['bbox'],extension_pixels)
            cropped_image, face_diff, skip_face = compare_images(sort.counter_dict["last_image"], inpaint_image,row['face_landmarks'], bbox)
            if sort.VERBOSE:print("inpainting done","shape:",np.shape(cropped_image))
            if skip_face:print("still 1x1 image , you're probably shifting both landmarks and bbox, only bbox needs to be shifted")

        return cropped_image, face_diff

    #itter is a cap, to stop the process after a certain number of rounds
    print('linear_test_df writing images')
    imgfileprefix = f"X{str(sort.XLOW)}-{str(sort.XHIGH)}_Y{str(sort.YLOW)}-{str(sort.YHIGH)}_Z{str(sort.ZLOW)}-{str(sort.ZHIGH)}_ct{str(df_sorted.size)}"
    print(imgfileprefix)
    good = 0
    img_list = []
    metas_list = []
    description = None
    cropped_image = np.array([-10])
    for index, row in df_sorted.iterrows():
        # parent_row = df_segment[df_segment['imagename'] == row['filename']]
        # print("parent_row")
        # print(parent_row)

        print('-- linear_test_df [-] in loop, index is', str(index))
        if V_VERBOSE: print(row["body_landmarks"])
        # select the row in df_segment where the imagename == row['filename']
        try:
            imgfilename = const_imgfilename_NN(row['image_id'], df_sorted, imgfileprefix)
            outpath = os.path.join(sort.counter_dict["outfolder"],imgfilename)
            open_path = os.path.join(io.ROOT,row['folder'],row['imagename'])
            # print(outpath, open_path)
            try:
                img = cv2.imread(open_path)
            except:
                print("couldn't read image")
                continue
            if row['dist'] < sort.MAXD:
                # compare_images to make sure they are face and not the same
                # last_image is cv2 np.array
                cropped_image, face_diff, skip_face = compare_images(sort.counter_dict["last_image"], img, row['face_landmarks'], row['bbox'])
                print("type of cropped_image", type(cropped_image))
                if skip_face:
                    print("skipping face")
                    continue
                # if cropped_image[0][0] == -10:
                #     print("-10 is returned from compare_images, so resize is too big, skipping")
                #     continue
                elif cropped_image is None and not skip_face:
                # if len(cropped_image)==1 and (OUTPAINT or INPAINT):
                    print("gotta paint that shizzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
                    cropped_image, face_diff = in_out_paint(img, row)

                # for drawing landmarks on test image
                # landmarks_2d = sort.get_landmarks_2d(row['face_landmarks'], list(range(33)), "list")
                # print("landmarks_2d before drawing", landmarks_2d)
                # cropped_image = sort.draw_point(cropped_image, landmarks_2d, index = 0)                    

                # landmarks_2d = sort.get_landmarks_2d(row['face_landmarks'], list(range(420)), "list")
                # cropped_image = sort.draw_point(cropped_image, landmarks_2d, index = 0)                    

                temp_first_run = sort.counter_dict["first_run"]
                print("temp_first_run", temp_first_run)
                if sort.counter_dict["first_run"]:
                    sort.counter_dict["last_description"] = description
                    print("first run, setting last_description")
                elif face_diff and face_diff < 30:
                    print("face_diff too small")
                    # temp, until resegmenting
                    print(description[0])
                    print(sort.counter_dict["last_description"])
                    if description[0] == sort.counter_dict["last_description"]:
                        print("same description!!!")


                if cropped_image is not None:
                    img_list.append((outpath, cropped_image))
                    # this is done in compare function
                    # sort.counter_dict["good_count"] += 1
                    good += 1
                    # print("going to save image metas")
                    save_image_metas(row)
                    # metas_list.append(save_image_metas(row))
                    # parent_row = df_segment[df_segment['imagename'] == row['filename']]
                    # print("parent_row")
                    # print(parent_row)

                    
                    # print("row['filename']")
                    # print(row['filename'])
                    sort.counter_dict["start_img_name"] = row['imagename']
                    # print(sort.counter_dict["last_image"])
                    print("saved: ",outpath)
                    sort.counter_dict["counter"] += 1
                    if itter and good > itter:
                        print("breaking after this many itters,", str(good), str(itter))
                        continue
                    sort.counter_dict["last_image"] = img_list[-1][1]  #last pair in list, second item in pair
                else:
                    print("cropped_image is None")
            else:
                sort.counter_dict["failed_dist_count"] += 1
                print("MAXDIST too big:" , str(sort.MAXD))
        # print("sort.counter_dict with last_image???")
        # print(sort.counter_dict)

        except Exception as e:
            traceback.print_exc()
            print(str(e))
        print("metas_list")
        print(metas_list)
    return img_list
    
def write_images(img_list):
    for path_img in img_list:
        cv2.imwrite(path_img[0],path_img[1])


def process_iterr_angles(start_img_name, df_segment, cluster_no, sort):
    #cycling patch
    img_list = []
    cycle = 0 
    metamedian = sort.metamedian
    d = sort.d

    print("CYCLE to test: ",cycle, start_img_name)
    while cycle < CYCLECOUNT:
        print("CYCLE: ",cycle)
        for angle in sort.angle_list:
            print("angle: ",str(angle))
            # print(d[angle].iloc[(d[angle][sort.SECOND_SORT]-metamedian).abs().argsort()[:2]])
            if(d[angle].size) != 0:
                try:
                    print("sort.counter_dict[start_img_name] before sort_by_face_dist_NN")
                    print(sort.counter_dict["start_img_name"] )
                    if sort.counter_dict["start_img_name"] != "median":
                        try:
                            last_row = df_segment.loc[sort.counter_dict["start_img_name"]]
                            print("last_row")
                            print(last_row)
                        except Exception as e:
                            traceback.print_exc()
                            print(str(e))
                    df_enc, df_128_enc = prep_encodings(d[angle])
                    # # get dataframe sorted by distance
                    df_sorted = sort_by_face_dist_NN(df_enc, df_128_enc)
                    # print("df_sorted")
                    # print(df_sorted)
                    # print("sort.counter_dict before linear_test_df")
                    # print(sort.counter_dict)
                    if sort.counter_dict["last_image"] is None:
                        try:
                            sort.counter_dict["last_image"] = cv2.imread(sort.counter_dict["start_img_name"])
                        except:
                            print("failed to open sort.counter_dict[start_img_name]")
                    else:
                        print("sort.counter_dict has a last_image")
                    # write_images(df_sorted, cluster_no)
                    # print("df_sorted before linear_test_df")

                    # print(type(df_sorted.size))
                    # print(df_sorted.size)
                    # print(df_sorted)
                    # print("sort.counter_dict after linear_test_df")
                    # print(sort.counter_dict)
                    # print("img_list")
                    # print(img_list[0])
                    # print(len(img_list))
                    # # only write the first, closest one
                    # # in the future, prob want to assign each image list to
                    # # a list/df keyed by angle, so can iterate through it? 
                    if angle > 15 and motion['forward_smile'] == True:
                        img_list = linear_test_df(df_sorted,cluster_no)
                        write_images(img_list)
                    else:
                        print("sending in an itter cap")
                        img_list = linear_test_df(df_sorted,cluster_no, 1)
                        cv2.imwrite(img_list[0][0],img_list[0][1])
                    


                except:
                    print('failed cycle angle:')
                    # print('failed:',row['imagename'])
            else:
                print("skipping empty angle")
        print('finished a cycle')
        sort.angle_list.reverse()
        cycle = cycle +1
        # # print(angle_list)

    print_counters()


def process_linear(start_img_name, df_segment, cluster_no, sort):
    # linear sort by encoding distance
    print("processing linear")
    # preps the encodings for sort
    sort.set_counters(io.ROOT,cluster_no, start_img_name, start_site_image_id)  
    print("sort.counter_dict after sort.set_counters", sort.counter_dict)
    # df_enc, df_128_enc, df_33_lms = prep_encodings(df_segment)
    df_enc = prep_encodings_NN(df_segment)
    print("sort.counter_dict after prep_encodings_NN", sort.counter_dict)
    
    # if results in df_enc, then sort by face distance
    if not df_enc.empty:
        # # get dataframe sorted by distance
        df_sorted = sort_by_face_dist_NN(df_enc)
        # df_sorted = sort_by_face_dist(df_enc, df_128_enc, df_33_lms)

        # test to see if they make good faces
        img_list = linear_test_df(df_sorted,df_segment,cluster_no)
        write_images(img_list)
        write_images(sort.not_make_face)
        print_counters()


###################
#  MY MAIN CODE   #
###################

def main():
    # these are used in cleaning up fresh df from SQL
    # def io.unpickle_array(pickled_array):
    #     if pickled_array:
    #         try:
    #             # Attempt to unpickle using Protocol 3 in v3.7
    #             return pickle.loads(pickled_array, encoding='latin1')
    #         except TypeError:
    #             # If TypeError occurs, unpickle using specific protocl 3 in v3.11
    #             # return pickle.loads(pickled_array, encoding='latin1', fix_imports=True)
    #             try:
    #                 # Set the encoding argument to 'latin1' and protocol argument to 3
    #                 obj = pickle.loads(pickled_array, encoding='latin1', fix_imports=True, errors='strict', protocol=3)
    #                 return obj
    #             except pickle.UnpicklingError as e:
    #                 print(f"Error loading pickle data: {e}")
    #                 return None
    #     else:
    #         return None

    # def get_encodings_mongo(image_id):
    #     mongo_collection_face = mongo_db['encodings']

    #     if image_id:
    #         results_face = mongo_collection_face.find_one({"image_id": image_id})
    #         results_body = mongo_collection_body.find_one({"image_id": image_id})
    #         face_encodings68 = face_landmarks = body_landmarks = body_landmarks_normalized = None
    #         if results_body:
    #             body_landmarks_normalized = results_body["nlms"]
    #         if results_face:
    #             face_encodings68 = results_face['face_encodings68']
    #             face_landmarks = results_face['face_landmarks']
    #             body_landmarks = results_face['body_landmarks']
    #             # print("got encodings from mongo, types are: ", type(face_encodings68), type(face_landmarks), type(body_landmarks))
    #         return pd.Series([face_encodings68, face_landmarks, body_landmarks, body_landmarks_normalized])
    #         # else:
    #         #     return pd.Series([None, None, None])
    #     else:
    #         return pd.Series([None, None, None, None])


    # def unstring_json(json_string):
    #     eval_string = ast.literal_eval(json_string)
    #     if isinstance(eval_string, dict):
    #         return eval_string
    #     else:
    #         json_dict = json.loads(eval_string)
    #         return json_dict
    # def make_float(value):
    #     try:
    #         return float(value)
    #     except (ValueError, TypeError):
    #         return value
    def decode_64_array(encoded):
        decoded = base64.b64decode(encoded).decode('utf-8')
        return decoded
    def newname(contentUrl):
        file_name_path = contentUrl.split('?')[0]
        file_name = file_name_path.split('/')[-1]
        extension = file_name.split('.')[-1]
        if file_name.endswith(".jpeg"):
            file_name = file_name.replace(".jpeg",".jpg")
        elif file_name.endswith(".png") or file_name.endswith(".webm"):
            pass
        elif not file_name.endswith(".jpg"):
            file_name += ".jpg"    
        hash_folder1, hash_folder2 = io.get_hash_folders(file_name)
        newname = os.path.join(hash_folder1, hash_folder2, file_name)
        return newname
        # file_name = file_name_path.split('/')[-1]
    print("in main, making SQL query")


    ###################
    #  MAP THE IMGS   #
    ###################

    # this is the key function, which is called for each cluster
    # or only once if no clusters
    def map_images(resultsjson, cluster_no=None):
        # print(df_sql)

        # read the csv and construct dataframe
        try:
            df = pd.json_normalize(resultsjson)
            print(df)
        except:
            print('you forgot to change the filename DUH')
        if not df.empty:


            print("going to get mongo encodings")
            print("size",df.size)
            # use the image_id to query the mongoDB for face_encodings68, face_landmarks, body_landmarks
            df[['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized']] = df['image_id'].apply(io.get_encodings_mongo)
            print("got mongo encodings")

            # drop all rows where face_encodings68 is None TK revist this after migration to mongo
            df = df.dropna(subset=['face_encodings68'])
            print("size",df.size)
            # print the first row value for 'face_encodings68' column
            # print("face_encodings68")
            # print(df['face_encodings68'][0])

            # Apply the unpickling function to the 'face_encodings' column
            df['face_encodings68'] = df['face_encodings68'].apply(io.unpickle_array)
            df['face_landmarks'] = df['face_landmarks'].apply(io.unpickle_array)
            df['body_landmarks'] = df['body_landmarks'].apply(io.unpickle_array)
            df['body_landmarks_normalized'] = df['body_landmarks_normalized'].apply(io.unpickle_array)
            df['bbox'] = df['bbox'].apply(lambda x: io.unstring_json(x))
            if OBJ_CLS_ID > 0: df["bbox_"+str(OBJ_CLS_ID)] = df["bbox_"+str(OBJ_CLS_ID)].apply(lambda x: io.unstring_json(x))




            # this may be a big problem
            # turn URL into local hashpath (still needs local root folder)
            # df['imagename'] = df['contentUrl'].apply(newname)
            # make decimals into float
            columns_to_convert = ['face_x', 'face_y', 'face_z', 'mouth_gap']
            df[columns_to_convert] = df[columns_to_convert].applymap(io.make_float)

            ### SEGMENT THE DATA ###

            # make the segment based on settings
            print("going to segment")
            # need to make sure has HSV here
            df_segment = sort.make_segment(df)
            print("made segment")


            # this is to save files from a segment to the SSD
            print("will I save segment? ", SAVE_SEGMENT)
            if SAVE_SEGMENT:
                Base.metadata.create_all(engine)
                print(df_segment.size)
                save_segment_DB(df_segment)
                print("saved segment to segmentTable")
                quit()

            ### Set counter_dict ###

            sort.set_counters(io.ROOT,cluster_no, start_img_name,start_site_image_id)

            print("set sort.counter_dict:" )
            print(sort.counter_dict)


            ### Get cluster_median encodings for cluster_no ###

            if cluster_no is not None and cluster_no !=0 and IS_CLUSTER:
                # skips cluster 0 for pulling median because it was returning NULL
                # cluster_median = select_cluster_median(cluster_no)
                # image_id = insert_dict['image_id']
                # can I filter this by site_id? would that make it faster or slower? 

                # temp fix
                results = session.query(Clusters).filter(Clusters.cluster_id==cluster_no).first()


                print(results)
                cluster_median = io.unpickle_array(results.cluster_median)
                sort.counter_dict["last_image_enc"]=cluster_median


            ### SORT THE LIST OF SELECTED IMAGES ###
            ###    THESE ARE THE VARIATIONS      ###

            if motion["side_to_side"] is True and IS_ANGLE_SORT is False:
                # this is old, hasn't been refactored.
                img_list, size = cycling_order(CYCLECOUNT, sort)
                # size = sort.get_cv2size(ROOT, img_list[0])
            elif IS_ANGLE_SORT is True:
                # get list of all angles in segment
                angle_list = sort.createList(df_segment)

                # sort segment by angle list
                # creates sort.d attribute: a dataframe organized (indexed?) by angle list
                sort.get_divisor(df_segment)

                # # is this used anywhere? 
                # angle_list_pop = angle_list.pop()

                # get median for first sort
                median = sort.get_median()

                # get metamedian for second sort, creates sort.metamedian attribute
                sort.get_metamedian()
                # print(df_segment)

                process_iterr_angles(start_img_name,df_segment, cluster_no, sort)
            else:
                # hard coding override to just start from median
                # sort.counter_dict["start_img_name"] = "median"

                process_linear(start_img_name,df_segment, cluster_no, sort)
        elif df.empty and IS_CLUSTER:
            print('dataframe empty, but IS_CLUSTER so continuing to next cluster_no')

        else: 
            print('dataframe empty, and not IS_CLUSTER so probably bad path or bad SQL')
            sys.exit()



    ###          THE MAIN PART OF MAIN()           ###
    ### QUERY SQL BASED ON CLUSTERS AND MAP_IMAGES ###

    #creating my objects
    start = time.time()

    # to loop or not to loop that is the cluster
    if IS_CLUSTER and not IS_ONE_TOPIC:
        print(f"IS_CLUSTER is {IS_CLUSTER} with {N_CLUSTERS}")
        for cluster_no in range(N_CLUSTERS):
            print(f"SELECTing cluster {cluster_no} of {N_CLUSTERS}")
            resultsjson = selectSQL(cluster_no, None)
            print(f"resultsjson contains {len(resultsjson)} images")
            map_images(resultsjson, cluster_no)
    elif IS_CLUSTER and IS_ONE_TOPIC:
        print(f"IS_CLUSTER is {IS_CLUSTER} with {N_CLUSTERS}, and topic {TOPIC_NO}")
        for cluster_no in range(N_CLUSTERS):
            print(f"SELECTing cluster {cluster_no} of {N_CLUSTERS}")
            resultsjson = selectSQL(cluster_no, TOPIC_NO)
            print(f"resultsjson contains {len(resultsjson)} images")
            map_images(resultsjson, cluster_no)
    elif IS_TOPICS:
        print(f"IS_TOPICS is {IS_TOPICS} with {N_TOPICS}")
        for topic_no in range(N_TOPICS):
            print(f"SELECTing cluster {topic_no} of {N_TOPICS}")
            resultsjson = selectSQL(None, topic_no)
            print(f"resultsjson contains {len(resultsjson)} images")
            map_images(resultsjson, topic_no)
    elif IS_ONE_CLUSTER:
        print(f"SELECTing cluster {CLUSTER_NO}")
        resultsjson = selectSQL(CLUSTER_NO, None)
        print(f"resultsjson contains {len(resultsjson)} images")
        map_images(resultsjson, CLUSTER_NO)
    elif IS_ONE_TOPIC:
        print(f"SELECTing topic {TOPIC_NO}")
        resultsjson = selectSQL(None, TOPIC_NO)
        print(f"resultsjson contains {len(resultsjson)} images")
        map_images(resultsjson, TOPIC_NO) # passing in TOPIC_NO to use in saving folder name
    else:
        print("doing regular linear")
        resultsjson = selectSQL() 
        map_images(resultsjson)

    print("got results, count is: ",len(resultsjson))


if __name__ == '__main__':
    main()

