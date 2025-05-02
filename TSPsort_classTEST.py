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
from my_declarative_base import Base, Encodings, SegmentTable,ImagesBackground, Hands, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Images
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

# controls which type of sorting/column sorted on
SORT_TYPE = "128d"
# SORT_TYPE ="planar"
# SORT_TYPE = "planar_body"
# SORT_TYPE = "planar_hands"
# SORT_TYPE = "fingertips_positions"
FULL_BODY = False # this requires is_feet
TSP_SORT=True
# this is for controlling if it is using
# all clusters, 
IS_VIDEO_FUSION = False # used for constructing SQL query
GENERATE_FUSION_PAIRS = False # if true it will query based on MIN_VIDEO_FUSION_COUNT and create pairs
                                # if false, it will grab the list of pair lists below
MIN_VIDEO_FUSION_COUNT = 750
IS_HAND_POSE_FUSION = False # i'm not sure how this is different from the IS_VIDEO_FUSION
ONLY_ONE = False
IS_CLUSTER = False
if IS_HAND_POSE_FUSION or IS_VIDEO_FUSION:
    if SORT_TYPE in ["planar_hands", "fingertips_positions", "128d"]:
        # first sort on HandsPositions, then on HandsGestures
        CLUSTER_TYPE = "HandsPositions" # Select on 3d hands
        CLUSTER_TYPE_2 = "HandsGestures" # Sort on 2d hands
    elif SORT_TYPE == "planar_body":
        # if fusion, select on body and gesture, sort on hands positions
        SORT_TYPE = "planar_hands"
        CLUSTER_TYPE = "BodyPoses"
        CLUSTER_TYPE_2 = "HandsGestures"
    # CLUSTER_TYPE is passed to sort. below
else:
    # choose the cluster type manually here
    # CLUSTER_TYPE = "BodyPoses" # usually this one
    # CLUSTER_TYPE = "HandsPositions" # 2d hands
    # CLUSTER_TYPE = "HandsGestures"
    CLUSTER_TYPE = "Clusters" # manual override for 128d
    CLUSTER_TYPE_2 = None
DROP_LOW_VIS = False
USE_HEAD_POSE = False
N_HANDS = N_CLUSTERS = None # declared here, but set in the SQL query below
# this is for IS_ONE_CLUSTER to only run on a specific CLUSTER_NO
IS_ONE_CLUSTER = False
CLUSTER_NO = 0 # sort on this one as HAND_POSITION for IS_HAND_POSE_FUSION
                # if not IS_HAND_POSE_FUSION, then this is selecting HandsGestures
START_CLUSTER = 0
# I started to create a separate track for Hands, but am pausing for the moment
IS_HANDS = False
IS_ONE_HAND = False
HAND_POSE_NO = 5

# 80,74 fails between 300-400

# cut the kids
NO_KIDS = True
ONLY_KIDS = False
USE_PAINTED = True
OUTPAINT = False
INPAINT= True
INPAINT_MAX = {"top":.4,"right":.4,"bottom":.075,"left":.4}
INPAINT_MAX_SHOULDERS = {"top":.4,"right":.15,"bottom":.2,"left":.15}
OUTPAINT_MAX = {"top":.7,"right":.7,"bottom":.2,"left":.7}

BLUR_THRESH_MAX={"top":50,"right":100,"bottom":10,"left":100}
BLUR_THRESH_MIN={"top":0,"right":20,"bottom":10,"left":20}

BLUR_RADIUS = 1  ##computationally more expensive
BLUR_RADIUS = io.oddify(BLUR_RADIUS)

MASK_OFFSET = [50,50,50,50]
# if OUTPAINT: from outpainting_modular import outpaint, image_resize
VERBOSE = True
SAVE_IMG_PROCESS = False
# this controls whether it is using the linear or angle process
IS_ANGLE_SORT = False

# this control whether sorting by topics
IS_TOPICS = False
N_TOPICS = 64

IS_ONE_TOPIC = True
TOPIC_NO = [22]

#######################

#######################

ONE_SHOT = False # take all files, based off the very first sort order.
EXPAND = False # expand with white, as opposed to inpaint and crop
JUMP_SHOT = True # jump to random file if can't find a run (I don't think this applies to planar?)
USE_ALL = False # this is for outputting all images from a oneshot, forces ONE_SHOT
DRAW_TEST_LMS = False # this is for testing the landmarks

if SORT_TYPE == "planar_hands" and USE_ALL:
    SORT_TYPE = "planar_hands_USE_ALL"
    ONE_SHOT = True

NORMED_BODY_LMS = True

MOUTH_GAP = 0
# if planar_body set OBJ_CLS_ID for each object type
# 67 is phone, 63 is laptop, 26: 'handbag', 27: 'tie', 32: 'sports ball'
OBJ_CLS_ID = 0
DO_OBJ_SORT = True
OBJ_DONT_SUBSELECT = False # False means select for OBJ. this is a flag for selecting a specific object type when not sorting on obj
PHONE_BBOX_LIMITS = [1] # this is an attempt to control the BBOX placement. I don't think it is going to work, but with non-zero it will make a bigger selection. Fix this hack TK. 


if not GENERATE_FUSION_PAIRS:
    print("not generating FUSION_PAIRS, pulling from list")
    FUSION_PAIRS = [
        #CLUSTER_NO, HAND_POSE_NO
        [22,15],[24,70]
        ]
    print("FUSION_PAIRS", FUSION_PAIRS)
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
    WHERE = " s.is_dupe_of IS NULL "
    # this is the standard segment topics/clusters query for June 2024
    if PHONE_BBOX_LIMITS:
        WHERE += " AND s.face_x > -50 "
    else:
        WHERE += " AND s.face_x > -33 AND s.face_x < -27 AND s.face_y > -2 AND s.face_y < 2 AND s.face_z > -2 AND s.face_z < 2"
    # HIGHER
    # WHERE = "s.site_name_id != 1 AND face_encodings68 IS NOT NULL AND face_x > -27 AND face_x < -23 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2"

    if MOUTH_GAP: WHERE += f" AND mouth_gap > {MOUTH_GAP} "
    # WHERE += " AND s.age_id NOT IN (1,2,3,4) "
    # WHERE += " AND s.age_id > 4 "

    ## To add keywords to search
    # FROM += " JOIN ImagesKeywords ik ON s.image_id = ik.image_id JOIN Keywords k ON ik.keyword_id = k.keyword_id "
    # WHERE += " AND k.keyword_text LIKE 'shout%' "

    def add_topic_select():
        global FROM, WHERE, SELECT
        FROM += " JOIN ImagesTopics it ON s.image_id = it.image_id "
        WHERE += " AND it.topic_score > .3"
        SELECT += ", it.topic_score" # add description here, after resegmenting

    if IS_HAND_POSE_FUSION or IS_VIDEO_FUSION:
        FROM += f" JOIN Images{CLUSTER_TYPE} ihp ON s.image_id = ihp.image_id "
        FROM += f" JOIN Images{CLUSTER_TYPE_2} ih ON s.image_id = ih.image_id "
        # WHERE += " AND ihp.cluster_dist < 2.5" # isn't really working how I want it
        if IS_VIDEO_FUSION: add_topic_select()
    elif IS_ONE_CLUSTER and IS_ONE_TOPIC:
        FROM += f" JOIN Images{CLUSTER_TYPE} ic ON s.image_id = ic.image_id "
        add_topic_select()
        print(f"SELECTING ONE CLUSTER {CLUSTER_NO} AND ONE TOPIC {TOPIC_NO}. This is my WHERE: {WHERE}")
    else:
        if IS_CLUSTER or IS_ONE_CLUSTER:
            FROM += f" JOIN Images{CLUSTER_TYPE} ic ON s.image_id = ic.image_id "
        if IS_TOPICS or IS_ONE_TOPIC:
            add_topic_select()
            # TEMP OVERRIDE FOR FINGER POINT TESTING
        # if TOPIC_NO == [22]:
        #     CLUSTER_TYPE = "Clusters"
        #     FROM += f" JOIN Images{CLUSTER_TYPE} ic ON s.image_id = ic.image_id "
        #     WHERE += " AND ic.cluster_id = 126"



    if FULL_BODY:
        WHERE += " AND s.is_feet = 1 "
    if NO_KIDS:
        WHERE += " AND s.age_id NOT IN (1,2,3) "
    if ONLY_KIDS:
        WHERE += " AND s.age_id IN (1,2,3) "
    if HSV_BOUNDS:
        FROM += " JOIN ImagesBackground ibg ON s.image_id = ibg.image_id "
        # WHERE += " AND ibg.lum > .3"
        SELECT += ", ibg.lum, ibg.lum_bb, ibg.hue, ibg.hue_bb, ibg.sat, ibg.sat_bb, ibg.val, ibg.val_bb, ibg.lum_torso, ibg.lum_torso_bb " # add description here, after resegmenting
    ###############
    if OBJ_CLS_ID > 0 and not OBJ_DONT_SUBSELECT:
        FROM += " JOIN PhoneBbox pb ON s.image_id = pb.image_id "
        # SELECT += ", pb.bbox_67, pb.conf_67, pb.bbox_63, pb.conf_63, pb.bbox_26, pb.conf_26, pb.bbox_27, pb.conf_27, pb.bbox_32, pb.conf_32 "
        SELECT += ", pb.bbox_"+str(OBJ_CLS_ID)+", pb.conf_"+str(OBJ_CLS_ID)
        WHERE += " AND pb.bbox_"+str(OBJ_CLS_ID)+" IS NOT NULL "

        # for some reason I have to set OBJ_CLS_ID to 0 if I'm doing planar_body
        # but I want to store the value in OBJ_SUBSELECT to use in SQL
        if SORT_TYPE == "planar_body" and not DO_OBJ_SORT: OBJ_CLS_ID = 0
            
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
    LIMIT = 250

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
    "forward_smile": False,
    "forward_wider": True,
    "laugh": False,
    "forward_nosmile":  False,
    "static_pose":  False,
    "simple": False,
}

# face_height_output is how large each face will be. default is 750
face_height_output = 1000
image_edge_multiplier = [1.2, 1.2, 1.6, 1.2] # standard portrait
# sort.max_image_edge_multiplier is the maximum of the elements
# UPSCALE_MODEL_PATH=os.path.join(os.getcwd(), "models", "FSRCNN_x4.pb")
# construct my own objects
sort = SortPose(motion, face_height_output, image_edge_multiplier,EXPAND, ONE_SHOT, JUMP_SHOT, HSV_BOUNDS, VERBOSE,INPAINT, SORT_TYPE, OBJ_CLS_ID,TSP_SORT=TSP_SORT)
print("class inititalised")

def expand_face_encodings(df,encoding_col= "face_encodings68",):
    """
    Given a DataFrame with:
      - a 'face_encodings68' column where each entry is a length-128 list or array,
    return a new DataFrame of shape (n_rows, 128) where:
      - Columns 1..128 are the individual encoding dimensions,
        named 'enc_0', 'enc_1', ..., 'enc_127'.
    """
    # Helper: if entries are string representations of lists, eval to real lists
    def parse_encoding(x):
        if isinstance(x, str):
            return list(eval(x))
        return list(x)

    # Apply parsing and expand into a DataFrame
    encodings = df[encoding_col].apply(parse_encoding).tolist()
    enc_df = pd.DataFrame(encodings, columns=[f"enc_{i}" for i in range(128)])
    return enc_df



    
def main():
    # ROOT="C:/Users/jhash/Documents/GitHub/facemap2"
    ROOT="C:\\Users\\jhash\\Documents\\GitHub"
    # FOLDER="travelling_salesman/128d_cluster_close_faces"
    FOLDER="travelling_salesman\\128d_cluster_close_faces"

    ROOT = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/test_data"
    FOLDER = "travelling_salesman/128d_cluster_close_faces"

    # ─── DATASET & DISTANCES ─────────────────────────────────────────────────
    # np.random.seed(42)
    # points = np.random.rand(N_POINTS, DIM)
    INPUT_FILE="df_enc.csv"
    OUTPUT_FILE="df_enc_TSPsorted.csv"
    filepath=os.path.join(ROOT,FOLDER,INPUT_FILE)
    
    #MM only

    df_enc = pd.read_csv(filepath)



    df_clean=expand_face_encodings(df_enc)
    sort.set_TSP_sort(df_clean,START_IDX=None,END_IDX=None)
    # df_sorted = sort.get_closest_df_NN(df_enc, df_sorted, start_image_id, end_image_id)
    df_sorted=sort.do_TSP_SORT(df_enc)
    out_filepath=os.path.join(ROOT,FOLDER,OUTPUT_FILE)
    df_sorted.to_csv(out_filepath, index=False)
    print("file saved to ", out_filepath)


if __name__ == "__main__":
    main()