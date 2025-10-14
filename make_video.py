import math
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
from operator import add
from pick import pick

from mediapipe.framework.formats import landmark_pb2
from sqlalchemy import create_engine, text,select, MetaData, Table, Column, Numeric, Integer, VARCHAR, update, Float
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
# my ORM
from my_declarative_base import Base, Encodings, SegmentTable,ImagesBackground, Hands, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Images
import pymongo 

#mine
from mp_sort_pose import SortPose, TSPSorterTwoPhase
from mp_db_io import DataIO
from mediapipe.framework.formats import landmark_pb2
import ast

# Initialize sorter
sorter = TSPSorterTwoPhase(
    skip_frac=0.25,     # Skip 25% of points
    verbose=True,
    iterations=2        # 2-3 iterations recommended
)

title = 'Please choose your operation: '
options = ['sequence and save CSV', 'assemble images from CSV']
option, MODE = pick(options, title)

VIDEO = False
CYCLECOUNT = 1

# keep this live, even if not SSD
SegmentTable_name = 'SegmentOct20'
# SegmentHelper_name = None
# SegmentTable_name = 'SegmentBig_isface'
# SegmentTable_name = 'SegmentBig_isnotface'
# SegmentHelper_name = 'SegmentHelper_may2025_4x4faces'
# SegmentHelper_name = 'SegmentHelper_sept2025_heft_keywords'
SegmentHelper_name = None
# SATYAM, this is MM specific
# for when I'm using files on my SSD vs RAID
IS_SSD = True
#IS_MOVE is in move_toSSD_files.py

# I/O utils
io = DataIO(IS_SSD)
db = io.db

# declare all the globals as None in one line
N_CLUSTERS = N_HANDS = SORT_TYPE = FULL_BODY = IS_HAND_POSE_FUSION = ONLY_ONE = GENERATE_FUSION_PAIRS = MIN_VIDEO_FUSION_COUNT = FUSION_PAIR_DICT = IS_CLUSTER = IS_ONE_CLUSTER = CLUSTER_NO = START_CLUSTER = USE_POSE_CROP_DICT = IS_SEGONLY = HSV_CONTROL = VISIBLE_HAND_LEFT = VISIBLE_HAND_RIGHT = USE_NOSEBRIDGE = LIMIT = MIN_CYCLE_COUNT = IS_HANDS = IS_ONE_HAND = NO_KIDS = ONLY_KIDS = USE_PAINTED = OUTPAINT = INPAINT= INPAINT_COLOR= INPAINT_MAX_SHOULDERS= OUTPAINT_MAX= BLUR_THRESH_MAX= BLUR_THRESH_MIN= BLUR_RADIUS= MASK_OFFSET= VERBOSE= CALIBRATING= SAVE_IMG_PROCESS= IS_ANGLE_SORT= IS_TOPICS= N_TOPICS= IS_ONE_TOPIC= TOPIC_NO= NEG_TOPICS= POS_TOPICS= NEUTRAL_TOPICS= AFFECT_GROUPS_LISTS= USE_AFFECT_GROUPS= None
N_HSV = HAND_POSE_NO = 0 # defaults to no HSV clusters

# CSV_FOLDER = os.path.join(io.ROOT_DBx, "body3D_segmentbig_useall256_CSVs_MMtest")
# CSV_FOLDER = os.path.join("/Users/michaelmandiberg/Documents/projects-active/facemap_production/body3D_segmentbig_useall256_CSVs_test")
CSV_FOLDER = os.path.join("/Users/michaelmandiberg/Documents/projects-active/facemap_production/heft_keyword_fusion_clusters/meta_clusters_test_build")
# TENCH UNCOMMENT FOR YOUR COMP:
# CSV_FOLDER = os.path.join(io.ROOT_DBx, "body3D_segmentbig_useall256_CSVs_test")

# CSV_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/body3D_segmentbig_useall256_CSVs"
# overriding DB for testing
# io.db["name"] = "stock"
# io.db["name"] = "ministock"

MODES = ['paris_photo_torso_images_topics', 'paris_photo_torso_videos_topics', '3D_bodies_topics','3D_arms', 'heft_torso_keywords']
MODE_CHOICE = 2
CURRENT_MODE = MODES[MODE_CHOICE]
LIMIT = 1000 # this is the limit for the SQL query

# set defaults, including for all modes to False
FULL_BODY = IS_HAND_POSE_FUSION = ONLY_ONE = GENERATE_FUSION_PAIRS = USE_FUSION_PAIR_DICT = IS_CLUSTER = IS_ONE_CLUSTER = USE_POSE_CROP_DICT = IS_TOPICS= IS_ONE_TOPIC = USE_AFFECT_GROUPS = False
EXPAND = ONE_SHOT = JUMP_SHOT = USE_ALL = CHOP_FIRST = TSP_SORT = False
USE_PAINTED = OUTPAINT = INPAINT= False
FUSION_FOLDER = None
MIN_CYCLE_COUNT = 300
AUTO_EDGE_CROP = False # this triggers the dynamic cropping based on min_max_body_landmarks_for_crop

image_edge_multiplier = None
N_TOPICS = 64 # changing this to 14 triggers the affect topic fusion, 100 is keywords. 64 is default
if "paris" in CURRENT_MODE:
    IS_HAND_POSE_FUSION = True # do we use fusion clusters
    GENERATE_FUSION_PAIRS = False # if true it will query based on MIN_VIDEO_FUSION_COUNT and create pairs
                                    # if false, it will grab the list of pair lists below
    USE_FUSION_PAIR_DICT = True
    FUSION_PAIR_DICT_NAME = "FUSION_PAIR_DICT_TOPICS_64"
    N_HSV = 0 # don't do HSV clusters
    # this control whether sorting by topics
    # IS_TOPICS = True # if using Clusters only, must set this to False
    if CURRENT_MODE == 'paris_photo_torso_images_topics':
        # controls which type of sorting/column sorted on
        SORT_TYPE = "planar_hands"
        image_edge_multiplier = [1.3,2,2.9,2] # portrait crop for paris photo images < Aug 30

    elif CURRENT_MODE == 'paris_photo_torso_videos_topics':
        SORT_TYPE = "128d"
        MIN_VIDEO_FUSION_COUNT = 300

        JUMP_SHOT = True # jump to random file if can't find a run (I don't think this applies to planar?)

        # initializing default square crop
        # if this is defined, then it will not call min_max_body_landmarks_for_crop
        image_edge_multiplier = [1.3,1.85,2.4,1.85] # tighter square crop for paris photo videos < Oct 29 FINAL VERSION NOV 2024 DO NOT CHANGE

        USE_PAINTED = False
        INPAINT= True
        INPAINT_COLOR = None # "white" or "black" or None (none means generative inpainting with size limits)

        # when doing IS_HAND_POSE_FUSION code currently only supports one topic at a time
        IS_ONE_TOPIC = True
        TOPIC_NO = [32] # if doing an affect topic fusion, this is the wrapper topic
elif "3D" in CURRENT_MODE:
    if CURRENT_MODE == '3D_bodies_topics':
        SORT_TYPE = "body3D"
        FULL_BODY = True # this requires is_feet
        EXPAND = True # expand with white for prints, as opposed to inpaint and crop. (not video, which is controlled by INPAINT_COLOR) 
    elif CURRENT_MODE == '3D_arms':
        # SORT_TYPE = "arms3D"
        SORT_TYPE = "body3D"
        FULL_BODY = False 
        # either AUTO_EDGE_CROP or image_edge_multiplier must be set. Not both
        AUTO_EDGE_CROP = True # this triggers the dynamic cropping based on min_max_body_landmarks_for_crop
        if AUTO_EDGE_CROP:
            EXPAND = True
        else:
            image_edge_multiplier = [1.3,2,2.9,2] # portrait crop for paris photo images < Aug 30

    SegmentTable_name = 'SegmentBig_isface'
    ONLY_ONE = False # only one cluster, or False for video fusion, this_cluster = [CLUSTER_NO, HAND_POSE_NO]
    USE_POSE_CROP_DICT = False
    # GENERATE_FUSION_PAIRS = True # if true it will query based on MIN_VIDEO_FUSION_COUNT and create pairs
    USE_FUSION_PAIR_DICT = True
    FUSION_PAIR_DICT_NAME = "FUSION_PAIR_DICT_TOPICS_64"

    CHOP_FIRST = True
    ONE_SHOT = True # take all files, based off the very first sort order.
    USE_ALL = True # this is for outputting all images from a oneshot, forces ONE_SHOT, skips face comparison

    INPAINT_COLOR = "white" # "white" or "black" or None (none means generative inpainting with size limits)

    IS_CLUSTER = True
    # this is for IS_ONE_CLUSTER to only run on a specific CLUSTER_NO
    IS_ONE_CLUSTER = False # make IS_CLUSTER == False
    CLUSTER_NO = 242 # sort on this one as HAND_POSITION for IS_HAND_POSE_FUSION
                    # if not IS_HAND_POSE_FUSION, then this is selecting HandsGestures
                    # I think this is pose number from BodyPoses3D if SORT_TYPE == "body3D"
    START_CLUSTER = 0

elif CURRENT_MODE == 'heft_torso_keywords':
    # SegmentTable_name = 'SegmentOct20'
    SegmentTable_name = 'SegmentBig_isface'
    SegmentHelper_name = 'SegmentHelper_sept2025_heft_keywords' # TK revisit this for prodution run
    # SORT_TYPE = "planar_hands"
    SORT_TYPE = "arms3D"
    TESTING = False
    IS_HAND_POSE_FUSION = True # do we use fusion clusters
    GENERATE_FUSION_PAIRS = True # if true it will query based on MIN_VIDEO_FUSION_COUNT and create pairs
                                    # if false, it will grab the list of pair lists below
    USE_FUSION_PAIR_DICT = True
    FUSION_PAIR_DICT_NAME = "FUSION_PAIR_DICT_KEYWORDS_512"
    # N_HSV = 16
    N_HSV = 0 # don't do HSV clusters
    
    # TSP_SORT = True
    # CHOP_FIRST = True
    # if TESTING: IS_HAND_POSE_FUSION = GENERATE_FUSION_PAIRS = False
    MIN_VIDEO_FUSION_COUNT = 1300 # this is the cut off for the CSV fusion pairs
    MIN_CYCLE_COUNT = 1000 # this is the cut off for the SQL query results

    MIN_VIDEO_FUSION_COUNT = 400 # this is the cut off for the CSV fusion pairs
    MIN_CYCLE_COUNT = 300 # this is the cut off for the SQL query results

    # this control whether sorting by topics
    # IS_TOPICS = True # if using Clusters only, must set this to False

    ONE_SHOT = True # take all files, based off the very first sort order.
    JUMP_SHOT = True # jump to random file if can't find a run (I don't think this applies to planar?)

    # initializing default square crop
    # if this is defined, then it will not call min_max_body_landmarks_for_crop
    # image_edge_multiplier = [1.3,1.85,2.4,1.85] # tighter square crop for paris photo videos < Oct 29 FINAL VERSION NOV 2024 DO NOT CHANGE
    image_edge_multiplier = [1.3,2,2.9,2] # portrait crop for paris photo images < Aug 30

    USE_ALL = True # turns off face xyz and mouthgap filters && if SORT_TYPE == "planar_hands" forces ONE_SHOT

    USE_PAINTED = False
    INPAINT= True
    INPAINT_COLOR = "white" # "white" or "black" or None (none means generative inpainting with size limits)

    # when doing IS_HAND_POSE_FUSION code currently only supports one topic at a time
    IS_ONE_TOPIC = True
    N_TOPICS = 100
    TOPIC_NO = [22411] # if doing an affect topic fusion, this is the wrapper topic
    FUSION_FOLDER = "utilities/data/heft_keyword_fusion_clusters_hsv_meta"
    CSV_FOLDER = os.path.join("/Users/michaelmandiberg/Documents/projects-active/facemap_production/heft_keyword_fusion_clusters/meta_clusters_fusion_build/")

else:
    print("unknown mode, exiting")
    pass
    # SORT_TYPE ="planar"
    # SORT_TYPE = "planar_body"
    # SORT_TYPE = "fingertips_positions"
    USE_AFFECT_GROUPS = False

print(f"doing {CURRENT_MODE}: SORT_TYPE {SORT_TYPE}, IS_HAND_POSE_FUSION {IS_HAND_POSE_FUSION}, GENERATE_FUSION_PAIRS {GENERATE_FUSION_PAIRS}, MIN_VIDEO_FUSION_COUNT {MIN_VIDEO_FUSION_COUNT}, IS_TOPICS {IS_TOPICS}, IS_ONE_TOPIC {IS_ONE_TOPIC}, TOPIC_NO {TOPIC_NO}, USE_AFFECT_GROUPS {USE_AFFECT_GROUPS}")

if USE_AFFECT_GROUPS:
    # groupings of affect topics
    NEG_TOPICS = [0,1,3,5,8,9,13]
    POS_TOPICS = [4,6,7,10,11,12]
    NEUTRAL_TOPICS = [16]
    AFFECT_GROUPS_LISTS = [NEG_TOPICS, POS_TOPICS, NEUTRAL_TOPICS]

IS_SEGONLY= True # This is for when you only have the segment table. RW SQL query
HSV_CONTROL = False # defining so it doesn't break below, if commented out
# This tells it to pull luminosity. Comment out if not using
if HSV_CONTROL: HSV_BOUNDS = {"LUM_MIN": 0, "LUM_MAX": 40, "SAT_MIN": 0, "SAT_MAX": 1000, "HUE_MIN": 0, "HUE_MAX": 360}
else: HSV_BOUNDS = {"LUM_MIN": 0, "LUM_MAX": 100, "SAT_MIN": 0, "SAT_MAX": 1000, "HUE_MIN": 0, "HUE_MAX": 360}
HSV_BOUNDS["d128_WEIGHT"] = 1
HSV_BOUNDS["HSV_WEIGHT"] = 1
HSV_BOUNDS["LUM_WEIGHT"] = 1
# converts everything to a 0-1 scale
HSV_NORMS = {"LUM": .01, "SAT": 1,  "HUE": 0.002777777778, "VAL": 1}


VISIBLE_HAND_LEFT = False
VISIBLE_HAND_RIGHT = False
USE_NOSEBRIDGE = True 

# this is set dynamically based on SORT_TYPE set above
if IS_HAND_POSE_FUSION:
    if SORT_TYPE in ["planar_hands", "fingertips_positions", "128d"]:
        # first sort on HandsPositions, then on HandsGestures
        CLUSTER_TYPE = "HandsPositions" # Select on 3d hands
        CLUSTER_TYPE_2 = "HandsGestures" # Sort on 2d hands
    elif SORT_TYPE == "planar_body":
        # if fusion, select on body and gesture, sort on hands positions
        SORT_TYPE = "planar_hands"
        CLUSTER_TYPE = "BodyPoses"
        CLUSTER_TYPE_2 = "HandsGestures"
    elif SORT_TYPE == "body3D":
        # if fusion, select on body3D and gesture, sort on hands positions
        SORT_TYPE = "planar_hands"
        CLUSTER_TYPE = "BodyPoses3D"
        CLUSTER_TYPE_2 = "HandsGestures"
    elif SORT_TYPE == "arms3D":
        # if fusion, select on arms3D and gesture, sort on hands positions
        SORT_TYPE = "arms3D"
        CLUSTER_TYPE = "MetaBodyPoses3D"
        # CLUSTER_TYPE_2 = "HandsGestures" # commenting this out for now TK
        CLUSTER_TYPE_2 = None
        # if TESTING:
        #     CLUSTER_TYPE = "BodyPoses3D"
        #     CLUSTER_TYPE_2 = None
    # CLUSTER_TYPE is passed to sort. below
else:
    if SORT_TYPE == "arms3D":
        # if fusion, select on arms3D and gesture, sort on hands positions
        # SORT_TYPE = "planar_hands"
        CLUSTER_TYPE = "ArmsPoses3D"
    else:
        # choose the cluster type manually here
        # CLUSTER_TYPE = "BodyPoses" # usually this one
        CLUSTER_TYPE = "BodyPoses3D" # 
        # CLUSTER_TYPE = "HandsPositions" # 2d hands
        # CLUSTER_TYPE = "HandsGestures"
        # CLUSTER_TYPE = "Clusters" # manual override for 128d
    CLUSTER_TYPE_2 = None
print(f"SORT_TYPE {SORT_TYPE}, CLUSTER_TYPE {CLUSTER_TYPE}, CLUSTER_TYPE_2 {CLUSTER_TYPE_2}")
DROP_LOW_VIS = False
USE_HEAD_POSE = False

# cut the kids
NO_KIDS = True
ONLY_KIDS = False

# in/out paint constants
INPAINT_MAX_SHOULDERS = {"top":.4,"right":.15,"bottom":.2,"left":.15}
# if INPAINT_COLOR: INPAINT_MAX_SHOULDERS = INPAINT_MAX = {"top":3.4,"right":3.4,"bottom":3.075,"left":3.4}
if INPAINT_COLOR: INPAINT_MAX_SHOULDERS = INPAINT_MAX = {"top":10,"right":10,"bottom":10,"left":10}
else: INPAINT_MAX = {"top":.4,"right":.4,"bottom":.075,"left":.4}
OUTPAINT_MAX = {"top":.7,"right":.7,"bottom":.2,"left":.7}
BLUR_THRESH_MAX={"top":50,"right":100,"bottom":10,"left":100}
BLUR_THRESH_MIN={"top":0,"right":20,"bottom":10,"left":20}
BLUR_RADIUS = 1  ##computationally more expensive
BLUR_RADIUS = io.oddify(BLUR_RADIUS)
MASK_OFFSET = [50,50,50,50]
if OUTPAINT: from outpainting_modular import outpaint, image_resize

# process stuff
VERBOSE = True
CALIBRATING = False
SAVE_IMG_PROCESS = False
# this controls whether it is using the linear or angle process
IS_ANGLE_SORT = False




#######################

#######################

#  is isolated,  is business,  babies, 17 pointing
#  is doctor <<  covid
#  is hands
#  phone = 15
#  feeling frustrated
#  is hands to face
#  shout
# 7 is surprise
#  is yoga << planar,  planar,  fingers crossed

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
PHONE_BBOX_LIMITS = [0] # this is an attempt to control the BBOX placement. I don't think it is going to work, but with non-zero it will make a bigger selection. Fix this hack TK. 

# commenting this out Sept 2025
# if SegmentHelper_name != 'SegmentHelper_may2025_4x4faces' or SegmentHelper_name is None:
#     # OVERRIDES FOR TESTING
#     # SET IS_SSD = False ABOVE
#     IS_HAND_POSE_FUSION = False # this is for testing the planar hands
#     IS_TOPICS = False # this is for testing the planar hands
#     PHONE_BBOX_LIMITS = None
#     ONE_SHOT = True


KEYWORD_DICT = {
    22411: {"label":"phone", "OR":[22101,444,22191,16045,11549,133300,133777], "AND":[], "NOT":[]},
    22412: {"label":"money", "OR":[23029,25287,133768,24593], "AND":[], "NOT":[]},
    22412.24705: {"label":"money AND dollars", "OR":[23029,25287,133768,24593], "AND":[24705,25155], "NOT":[]},
    26270: {"label":"money AND euros", "OR":[23029,25287,133768,24593], "AND":[22412], "NOT":[]},
    5310: {"label":"gun", "OR":[], "AND":[], "NOT":[184]},

}


FUSION_PAIR_DICT_TOPICS_64 = {
    32: [[13, 109], [13, 24], [13, 85]]
}

FUSION_PAIR_DICT_3DBODIES_TOPICS_512 = {
    220: [[101, 0], [124, 0], [184, 0], [216, 0], [302, 0], [448, 0], [457,0]]
}
FUSION_PAIR_DICT_KEYWORDS_512 = {
    220: [[101, 0], [124, 0], [184, 0], [216, 0], [302, 0], [448, 0], [457,0]],
    553: [[216, 0], [230, 0], [89,0]],
    807: [[140, 0], [168, 0], [216, 0], [273, 0], [63,0]],
    827: [[191, 0], [229, 0], [322, 0], [340, 0], [396, 0], [423, 0], [447, 0], [448, 0], [51,0]],
    1070: [[122, 0], [14, 0], [22, 0], [342, 0], [391, 0], [53,0]],
    1644: [[158, 0], [161, 0], [345,0]],
    5310: [[180, 0], [245, 0], [265, 0], [315, 0], [324, 0], [445, 0], [451, 0], [476, 0], [498,0]],
    22269: [[351, 0], [449, 0], [53,0]],
    22411: [[158, 0], [240, 0], [304, 0], [306, 0], [342, 0], [391, 0], [40, 0], [423, 0], [449,0]],
    22412: [[89, 0], [101, 0], [140, 0], [184, 0], [273, 0], [391, 0], [396, 0], [426, 0], [430, 0], [446, 0], [449, 0], [481,0]]
}


ALL_FUSION_PAIRS_DICTS = {
    "FUSION_PAIR_DICT_TOPICS_64": FUSION_PAIR_DICT_TOPICS_64,
    "FUSION_PAIR_DICT_KEYWORDS_512": FUSION_PAIR_DICT_KEYWORDS_512,
    "FUSION_PAIR_DICT_3DBODIES_TOPICS_512": FUSION_PAIR_DICT_3DBODIES_TOPICS_512
    }
if USE_FUSION_PAIR_DICT:
    #set the actual dict from the name
    FUSION_PAIR_DICT = ALL_FUSION_PAIRS_DICTS[FUSION_PAIR_DICT_NAME]


if not GENERATE_FUSION_PAIRS:
    # from 512 testing
    # print("not generating FUSION_PAIRS, pulling from list")
    # FUSION_PAIR_DICT = {
    #     220: [[101, 0], [124, 0], [184, 0], [216, 0], [302, 0], [448, 0], [457,0]],
    #     553: [[216, 0], [230, 0], [89,0]],
    #     807: [[140, 0], [168, 0], [216, 0], [273, 0], [63,0]],
    #     827: [[191, 0], [229, 0], [322, 0], [340, 0], [396, 0], [423, 0], [447, 0], [448, 0], [51,0]],
    #     1070: [[122, 0], [14, 0], [22, 0], [342, 0], [391, 0], [53,0]],
    #     1644: [[158, 0], [161, 0], [345,0]],
    #     5310: [[180, 0], [245, 0], [265, 0], [315, 0], [324, 0], [445, 0], [451, 0], [476, 0], [498,0]],
    #     22269: [[351, 0], [449, 0], [53,0]],
    #     22411: [[158, 0], [240, 0], [304, 0], [306, 0], [342, 0], [391, 0], [40, 0], [423, 0], [449,0]],
    #     22412: [[89, 0], [101, 0], [140, 0], [184, 0], [273, 0], [391, 0], [396, 0], [426, 0], [430, 0], [446, 0], [449, 0], [481,0]]
    # }

    # meta clusters testing oct 11
    # print("not generating FUSION_PAIRS, pulling from list")
    # FUSION_PAIR_DICT = {
    #     22412: [[1,0], [2,0], [3,0], [4,0], [5,0], [6,0], [7,0], [8,0], [9,0], [10,0], [11,0], [12,0], [13,0], [14,0], [15,0], [16,0], [17,0], [18,0], [19,0], [20,0], [21,0], [22,0], [23,0], [24,0], [25,0], [26,0], [27,0], [28,0], [29,0], [30,0], [31,0], [32,0]],
    # }

    FUSION_PAIRS = [

        # Sept 2025 kludge
        # [198, 0], [206, 0], [208, 0], [215, 0], [216, 0], [219, 0], [220, 0], [229, 0], [234, 0], [240, 0], [244, 0], [250, 0], [257, 0], [264, 0], [265, 0], [273, 0], [277, 0], [279, 0], [288, 0], [302, 0], [304, 0], [305, 0], [306, 0], [316, 0], [317, 0], [322, 0], [324, 0], [331, 0], [340, 0], [342, 0], [351, 0], [352, 0], [364, 0], [370, 0], [390, 0], [391, 0], [396, 0], [412, 0], [422, 0], [423, 0], [425, 0], [426, 0], [445, 0], [446, 0], [448, 0], [449, 0], [457, 0], [464, 0], [470, 0], [472, 0], [476, 0], [479, 0], [481, 0], [482, 0], [485, 0], [487, 0], [498, 0], [500, 0], [507, 0]

        # Sept 2025, keyword money
        # [89, 0], [101, 0], [140, 0], [184, 0], [273, 0], [391, 0], [396, 0], [426, 0], [430, 0], [446, 0], [449, 0], [481]

        #CLUSTER_NO, HAND_POSE_NO

        # T0 sports
        # selects
        # [19, 61],
        # [22, 2], 
        # [9, 2], 
        # [9, 13],
        # [19, 95],  
        # [24, 13], # this one crashes the code after about 950
        # from fusion analysis
        # [19, 66], [8, 13], [29, 37], [17, 22], [19, 9], [4, 49], [6, 22], [6, 122], [21, 116], [24, 113], [26, 47], [26, 67]


        # T34 achieve scream
        # [30,6],[30,55],[30,81],[6,81],[15,38],[30,28],[15,28],[30,53],[15,53],[15,37],[30,110],[6,115],[1,113],[6,38],[21,72],[15,25],[15,92],[15,55],[6,6],[30,38],[30,37],[6,37]
        #T1 outside, think
        # [21,73]
        #T9 make up
        # [24,112]
        # T 15 muscl	romant
        # [6, 122], [16, 101]
        #T16 food
        # [1, 4],[16, 69]
        #T23 depressed
        # [21, 30], [9, 109]
        #T32 shock suprise
        # [13, 109],        [13, 24],        [13, 85]
        # [1, 125], [25, 125], [15, 37], [15, 91], [13, 116], [30, 37], [1, 65]
        #T35 skin care
        # [24, 51],[21, 13],[13, 118]
        #T59 Xmas
        # [8, 90],[8, 105],[23, 45],[16, 28],[22, 63],
        #T63 second
        # [8, 57],[22, 27]

        #T11 Business semi-unique
        # [25,64], [14,45], [23,47], [7,120], [12,33], [7,105], [26,10], [23,80], [12,100], [21,126], [7,64], [8,41], [7,57], [8,64], [12,19], [23,10], [12,24], [7,51], [12,118], [12,27], [12,93], [2,80], [23,110], [5,60], [26,60], [26,31], [23,45], [21,58], [16,110], [14,28], [14,110], [1,56], [21,5], [17,22], [7,54], [1,64], [18,8], [8,105], [8,90], [8,57], [7,41], [22,27], [13,106], [22,33], [7,1], [1,4], [14,86], [11,48], [18,29], [16,10], [20,102], [3,94], [8,54], [11,71], [6,81], [10,31], [30,86]
        #T22 finger point semi-unique
        # [30,36], [30,111], [30,114], [15,35], [1,29], [6,98], [6,78], [14,114], [13,71], [1,8], [18,99], [15,111], [30,89], [30,98], [1,123], [11,106], [11,33], [30,77], [15,78], [1,42], [15,74], [14,10], [16,114], [14,9], [6,74], [3,114], [30,78], [21,48], [18,40], [15,114], [24,70], [30,9], [11,108], [30,94], [24,56], [14,60], [28,19], [15,89], [24,64], [21,99], [16,86], [15,9], [22,15], [21,15], [13,48], [6,36], [8,111], [10,99], [4,48], [10,8], [30,74], [18,123], [11,17], [21,8], [1,54], [15,98], [15,115], [22,19], [30,10], [24,54], [30,115], [6,72], [1,99], [15,22], [1,56], [30,44], [3,77], [16,44], [15,86], [30,46], [21,83], [21,52], [15,55], [31,25], [18,8], [11,48], [6,115], [11,71], [18,29], [1,64], [20,102], [3,94], [21,72], [6,22], [30,55], [14,86], [30,86], [16,101], [22,33], [16,10], [8,54], [2,44], [10,31]
        #T22 finger point semi-unique TEST
        # [1,8],[1,56],[6,22],[14,9],[14,86],[20,102]
        # T47 handsome semi-unique
        # [21,84],[21,96],[21,18],[29,32],[21,14],[1,119],[16,6],[5,47],[21,103],[13,27],[16,85],[12,75],[11,127],[21,40],[21,81],[25,59],[7,57],[5,60],[12,24],[12,19],[12,27],[26,60],[15,38],[21,52],[30,74],[8,64],[21,58],[2,80],[21,5],[12,118],[18,123],[6,72],[23,10],[15,115],[30,10],[16,28],[21,83],[11,17],[30,115],[23,110],[12,93],[21,8],[7,51],[30,6],[16,110],[21,73],[1,54],[24,54],[15,98],[15,22],[30,81],[3,77],[22,19],[10,75],[21,30],[14,110],[15,91],[31,25],[14,28],[13,109],[26,31],[1,113],[30,38],[15,86],[16,44],[30,46],[30,44],[18,8],[6,115],[13,106],[7,54],[21,72],[11,48],[1,4],[22,27],[1,64],[20,102],[15,37],[8,57],[18,29],[30,55],[6,122],[6,22],[8,105],[3,94],[7,41],[10,31],[11,71],[7,1],[16,10],[22,33],[14,86],[6,81],[16,101],[8,90],[8,54],[24,4],[30,37],[2,44],[13,116],[30,86]

        # Topic 22 Finger
        # [24,99] #silence finger

        # Phone
        # [16, 21],  [22, 24], [22, 27], [23, 21], [23, 67]
        # Phone to ear
        # [13, 103], [21, 97] 
        # topic 25 beauty for video blur
        # face frame
        # [13, 76],  [21, 0], [21, 116],
        # glasses
        # [21, 58], [21, 84], [21,52],
        # hand to chin
        # [24, 126]

        # topic 35 skin care, 750plus selects
        # [21, 0]
        # both hands
        # [21, 116]
        # [13, 116], [21, 0], [21, 116], [24, 51], [24, 126]
        # both hands staring from 79236398
        # [24,51]
        # [13,3]
        # left hand
        # [13,76], [24, 126], [13,3]
        # right hand
        # [21,112], [24,112],[13, 0]

        #         # <3 
        # [16,101] #hands making heart shape

        # hands framing corners of photograph
        # [22,15],[24,70]
        # # topic 17 selects
        # [1,5]

        # topic 25 beauty, 750plus selects
        # [13,103], [24,42], [24,113], [24,126], [26,80], [26,47]

        # topic 25 beauty, right hand to face
        # [24, 4], [24, 11], [24, 13], [24, 23], [24, 42], [24, 51], [24, 57], [24, 97], [24, 99], [24, 112], [24, 113], [24, 119], [24, 126]
        # June 24 semi-unique
        # [10,75],[4,49],[7,1],[7,54],[13,106],[7,41]

        #depressed topic 23
        # [12,79], [13,106], [13,109], [13,117], [13,2], [13,55], [13,76], [14,45], [14,69], [15,92], [16,110], [21,0], [21,103], [21,109], [21,112], [21,113], [21,116], [21,118], [21,126], [21,13], [21,14], [21,18], [21,2], [21,30], [21,43], [21,52], [21,5], [21,68], [21,73], [21,76], [21,81], [21,83], [21,84], [21,87], [21,97], [24,113], [24,119], [24,11], [24,126], [24,13], [7,57], [9,109], [9,30], [9,68], [9,6], [9,97]
        # [21,13] # single

        # shocked single
        # [13, 2]

        # # topic 11 business singles
        # [23,80]
        # [26,10]
        # [7,57] # arms crossed
        # [30,10], [30,86] # thumbs up
        # [2,80], [7,51], [7,54], [7,57], [7,120], [12,27], [12,118] # arms crossed

        # # 34 Success
        # [11,3], [15,25], [15,28], [15,37], [15,38], [15,53], [15,55], [15,92], [1,113], [21,0], [21,116], [21,68], [30,110], [30,28], [30,37], [30,38], [30,53], [30,55], [30,6], [30,81], [6,115], [6,37], [6,38], [6,6], [6,81], [9,30]

        # Stressed single
        # [21, 30], [21, 68]

        # # topic 32, T64, ihp128
        # Book example
        # [21, 112], [21, 109], [21, 84], [21, 55]

        # topic 32, T64, ihp128 -- REWORK TESTING SEPT 13 2025
        # hands over face
        # [1, 65], [1, 88], [1, 125] 
     
        # # topic 32, T64, ihp128
        # # hands over face
        # [1, 65], [1, 88], [1, 125], [8, 125], [13, 2], [13, 24], [13, 65], [13, 85], [13, 117], [18, 88], [18, 125], [24, 11], [24, 13], [24, 57], [25, 65], [25, 88], [25, 125]

        # topic 32, T64, ihp128
        # open mouth
        # [4,124], [5,104], [6,14], [6,37], [6,38], [6,72], [6,115], [9,30], [9,68], [9,109], [10,31], [11,108], [13,76], [13,116], [15,37], [15,38], [15,53], [15,92], [18,7], [21,30], [21,37], [21,68], [21,76], [21,116], [22,127], [30,37], [30,38]

        # # topic 32, T64
        # [13, 11], [13, 29], [13, 47], [1, 37], [24, 12], [24, 8], [25, 25], [13, 37], [1, 53], [24, 33], [25, 11], [25, 37], [13, 28], [1, 25], [1, 60], [24, 37]

        # # topic 23 reselects
        # [1,65], [1,88], [1,125], [13,109], [13,24], [13,33], [13,85], [24,13], [24,85], [25,65], [13,117], [13,2], [13,65], [24,11], [24,7], [25,125], [25,88]

        # # topic 23 selects
        # [13,13], [13,15], [13,17], [13,23], [13,29], [13,7], [18,2], [1,13], [24,12], [24,24], [25,13]

        # no topics
        # [0,68], [35,11], [35,13], [59,30], [74,10], [113,2], [113,117]

        # 300 largest below
        # [1,21],[115,97],[53,21],[47,21],[47,67],[4,103],[5,115],[7,57],[52,8],[11,17],[112,57],[3,24],[11,71],[11,48],[120,115],[121,29],[15,74],[1,67],[121,123],[3,27],[115,112],[121,8],[74,67],[17,94],[99,62],[56,102],[126,102],[20,0],[31,24],[28,111],[0,68],[121,7],[35,99],[4,76],[18,97],[52,29],[84,25],[109,8],[66,54],[4,3],[38,48],[52,123],[2,64],[2,56],[94,19],[50,21],[68,21],[20,116],[5,78],[52,99],[1,80],
        # [7,105],[68,45],[80,35],[10,67],[33,104],[120,22],[88,57],[7,120],[46,124],[2,29],[113,3],[36,105],[28,89],[8,91],[120,78],[1,110],[18,87],[31,19],[36,120],[75,115],[96,90],[31,27],[73,63],[115,126],[10,45],[108,16],[29,127],[88,64],[35,126],[35,11],[15,89],[17,114],[83,57],[108,77],[4,52],[94,33],[94,27],[23,114],[111,90],[2,4],[11,108],[7,64],[115,58],[65,24],[84,108],[87,90],[88,54],[65,33],[87,120],[29,33],[83,41],[5,74],[11,127],[109,29],[73,27],[3,33],[89,102],[55,24],[79,94],[23,94],[73,24],[115,99],[121,4],[38,17],[49,44],[48,59],[2,54],[80,92],[4,71],[65,63],[55,19],[73,33],[88,120],[113,117],[106,41],[83,54],[62,35],[47,80],[40,44],[4,48],[68,67],[49,74],[7,54],[36,64],[3,63],[15,114],[49,46],[77,111],
        # [80,88],[92,73],[65,93],[77,89],[2,99],[112,54],[47,110],[15,94],[50,114],[53,80],[87,57],[57,92],[0,84],[3,19],[22,6],[57,9],[18,58],[35,113],[113,106],[15,92],[28,114],[50,107],[94,71],[57,85],[53,110],[108,88],[11,33],[47,45],[104,31],[13,63],[120,98],[53,28],[50,28],[126,89],[88,4],[2,120],[60,102],[52,4],[111,64],[85,67],[66,42],[88,90],[11,106],[92,5],[18,8],[4,127],[120,74],[125,104],[35,42],[94,12],[115,113],[31,2],[15,78],[57,114],[68,28],[1,89],[1,26],[55,100],[87,64],[126,107],[12,105],[2,42],[87,105],[38,71],[112,120],[50,45],[90,114],[113,2],[57,45],[115,119],[42,44],[118,45],
        # [3,93],[15,35],[90,89],[121,119],[1,16],[5,72],[52,58],[68,34],[0,72],[0,0],[51,77],[53,89],[111,4],[83,51],[112,11],[15,55],[18,112],[28,35],[83,105],[2,57],[90,21],[115,13],[85,47],[50,50],[22,21],[111,119],[103,82],[55,27],[78,67],[35,13],[56,25],[111,7],[51,94],[57,28],[29,108],[50,110],[50,34],[88,56],[92,81],[31,33],[59,30],[54,100],[74,10],[94,24],[67,102],[78,21],[3,79],[5,111],[35,57],[2,119],[1,85],[15,115],[80,74],[78,125],[18,126],[53,16],[1,125],[12,120],[112,13],[122,91],[112,64],[94,127],[12,124],[68,110],[113,27],[42,80],[80,78],[36,57],[15,111],[7,41],[97,94],[55,118],[28,17],[75,22],[84,127],[127,118],[52,40],[52,7],[80,115],[122,35],[15,28],[120,35],[112,113],[49,36],[15,102],[67,89],[80,22],[121,64]

        # phone bbox only
        # [1,21],[115,97],[53,21],[47,21],[47,67],[4,103],[7,57],[112,57],[3,24],[1,67],[3,27],[115,112],[31,24],[18,97],[74,67],[121,7],[20,0],[4,76],[96,90],[0,68],[1,80],[50,21],[88,57],[68,21],[1,110],[17,94],[115,126],[7,105],[31,27],[115,58],[126,102],[20,116],[4,3],[108,16],[7,120],[111,90],[36,105],[10,67],[4,52],[52,8],[94,27],[57,85],[94,19],[73,24]
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
    LIMIT = LIMIT

##################MICHAEL#####################
elif IS_SEGONLY and io.platform == "darwin":

    SAVE_SEGMENT = False
    # no JOIN just Segment table
    SELECT = "DISTINCT(s.image_id), s.site_name_id, s.contentUrl, s.imagename, s.description, s.face_x, s.face_y, s.face_z, s.mouth_gap, s.bbox, s.site_image_id"

    FROM =f"{SegmentTable_name} s "
    dupe_table_pre  = "s"
    if "SegmentBig_" in SegmentTable_name:
         # handles segmentbig which doesn't have is_dupe_of?
        FROM += f" JOIN Encodings e ON s.image_id = e.image_id "
        dupe_table_pre = "e"

    WHERE = f" {dupe_table_pre}.is_dupe_of IS NULL "
    # this is the standard segment topics/clusters query for June 2024
    if SegmentHelper_name is None:
        pass
    elif PHONE_BBOX_LIMITS:
        pass
        # WHERE += " AND s.face_x > -50 "
    else:
        WHERE += " AND s.face_x > -33 AND s.face_x < -27 AND s.face_y > -2 AND s.face_y < 2 AND s.face_z > -2 AND s.face_z < 2"
# OVERRIDE FOR TESTING
        # WHERE += " AND s.face_x > -27 AND s.face_x < 0 AND s.face_y > -5 AND s.face_y < 5 AND s.face_z > -5 AND s.face_z < 5"
    # HIGHER
    # WHERE = "s.site_name_id != 1 AND face_encodings68 IS NOT NULL AND face_x > -27 AND face_x < -23 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2"

    if MOUTH_GAP: WHERE += f" AND mouth_gap > {MOUTH_GAP} "
    # WHERE += " AND s.age_id NOT IN (1,2,3,4) "
    # WHERE += " AND s.age_id > 4 "

    ## To add keywords to search
    # FROM += " JOIN ImagesKeywords ik ON s.image_id = ik.image_id JOIN Keywords k ON ik.keyword_id = k.keyword_id "
    # WHERE += " AND k.keyword_text LIKE 'shout%' "

    def add_topic_select():
        global FROM, WHERE, SELECT, WrapperTopicTable
        if N_TOPICS == 14: 
            ImagesTopics = "ImagesTopics_affect"
            WrapperTopicTable = "ImagesTopics" # use this to narrow down an interative search through the affect topic(s)
            topic_keyword_id = "topic_id"
        elif N_TOPICS == 64: 
            ImagesTopics = "ImagesTopics"
            WrapperTopicTable = None
            topic_keyword_id = "topic_id"
        elif N_TOPICS == 100:
            ImagesTopics = "ImagesKeywords"
            WrapperTopicTable = None
            topic_keyword_id = "keyword_id"
        FROM += f" JOIN {ImagesTopics} it ON s.image_id = it.image_id "
        if not N_TOPICS == 100:
            WHERE += " AND it.topic_score > .1"
            SELECT += ", it.topic_score" # add description here, after resegmenting

    if IS_HAND_POSE_FUSION:
        # handle META situation, where arms3D needs to look for meta clusters
        if SORT_TYPE == "arms3D": 
            cluster_table = f"ImagesBodyPoses3D"
            # ihp as ClustersMetaBodyPoses3D - selecting the meta clusters id below
            # join Segment to ImagesBodyPoses3D
            # join ImagesBodyPoses3D to ClustersMetaBodyPoses3D

            # FROM += f" JOIN ClustersMetaBodyPoses3D ihp ON s.image_id = ihp.image_id "    
            FROM += f" JOIN ImagesBodyPoses3D ihp ON s.image_id = ihp.image_id "
            FROM += f" JOIN ClustersMetaBodyPoses3D cmp ON ihp.cluster_id = cmp.cluster_id "
        else: 
            cluster_table = f"Images{CLUSTER_TYPE}"
            FROM += f" JOIN {cluster_table} ihp ON s.image_id = ihp.image_id "
        if CLUSTER_TYPE_2: FROM += f" JOIN Images{CLUSTER_TYPE_2} ih ON s.image_id = ih.image_id "
        # WHERE += " AND ihp.cluster_dist < 2.5" # isn't really working how I want it
        if IS_HAND_POSE_FUSION: add_topic_select()
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

    if SegmentHelper_name is not None:
        FROM += f" JOIN {SegmentHelper_name} sh ON s.image_id = sh.image_id "

    if VISIBLE_HAND_LEFT or VISIBLE_HAND_RIGHT:
        if SegmentTable_name == "SegmentBig_isface":
            # handles segmentbig which doesn't have is_dupe_of?
            this_seg = "e"
        else:
            this_seg = "s"
        if VISIBLE_HAND_LEFT and VISIBLE_HAND_RIGHT:
            WHERE += f" AND {this_seg}.is_bodyhand_left = 1 AND {this_seg}.is_bodyhand_right = 1 "
        elif VISIBLE_HAND_LEFT: 
            WHERE += f" AND {this_seg}.is_bodyhand_left = 1 AND {this_seg}.is_bodyhand_right = 0 "  
        elif VISIBLE_HAND_RIGHT: 
            WHERE += f" AND {this_seg}.is_bodyhand_right = 1 AND {this_seg}.is_bodyhand_left = 0 "
    if FULL_BODY:
        if "Encodings" not in FROM: FROM += f" JOIN Encodings e ON s.image_id = e.image_id "
        WHERE += " AND e.is_feet = 1 "
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
    LIMIT = LIMIT

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
    LIMIT = LIMIT

    # TEMP TK TESTING
    # WHERE += " AND s.site_name_id = 8"
######################################

# construct the motion dictionary all false
motion = {
    "side_to_side": False,
    "forward_smile": False,
    "forward_wider": False,
    "laugh": False,
    "forward_nosmile":  False,
    "static_pose":  False,
    "simple": False,
    "use_all": False,
}

if USE_ALL:
    # if USE_ALL is True, set "use_all": True
    motion["use_all"] = True
else:
    # this sets the range of face xyz narrower
    motion["forward_wider"] = True



# face_height_output is how large each face will be. default is 750
face_height_output = 1000

# face_height_output = 250

# define crop ratios, in relationship to nose
# units are ratio of faceheight
# top, right, bottom, left
pose_crop_dict = {
    0: 11, 1: 1, 2: 5, 3: 2, 4: 9, 5: 7, 6: 2, 7: 5, 8: 9, 9: 8, 10: 3, 11: 1, 12: 5, 13: 1, 14: 0,
    15: 2, 16: 1, 17: 10, 18: 4, 19: 9, 20: 2, 21: 1, 22: 9, 23: 0, 24: 9, 25: 2, 26: 9, 27: 11, 28: 1, 29: 12,
    30: 5, 31: 2
}
# 9 includes standing. 

multiplier_list = [
    [1.5,3,3.3,3], # 0 2x3 but go lower
    [1.3,1.85,2.4,1.85], # 1 SQ
    [1.5,3,2.5,3], # 2 # 2x3 landscape 2025
    [1.5,2,4.5,2], # 3 # 3x2 portrait 2025
    [1.5,2.5,2.6,2.5], # 4 # 4x5 landscape 2025
    [1.5,2,3.5,2], # 5 # 5x4 portrait 2025
    [1.5,2.2,4,2.2], # 6 5x4 but go lower and wider 
    [1.5,2,4.5,2], # 7 ~6x2 full length portrait 2025 
    [4.5,3.5,5.5,3.5], # 8 arms raised lotus 
    [1.5,3.5,5.5,3.5], # 9 seated lotus
    [3.5,3.5,3.5,3.5], # 10 arms raised and gunshow 2025 
    [1.3,1.85,2.4,1.85], # 11 -- placeholder to test if SQ
    [1.5,3.75,3.5,3.75], # 12 extra wide 2x3 landscape (shruggie "why" pose)
]
# image_edge_multiplier = [1,1,1,1]
    # image_edge_multiplier = [1.4,2.6,1.9,2.6] # wider for hands (2023 finger point)
# image_edge_multiplier = [1.4,3.5,5.6,3.5] # yoga square crop for April 2025 videos < 
# sort.max_image_edge_multiplier is the maximum of the elements

UPSCALE_MODEL_PATH=os.path.join(os.getcwd(), "models", "FSRCNN_x4.pb")
# construct my own objects
sort = SortPose(motion, face_height_output, image_edge_multiplier,EXPAND, ONE_SHOT, JUMP_SHOT, HSV_BOUNDS, VERBOSE,INPAINT, SORT_TYPE, OBJ_CLS_ID,UPSCALE_MODEL_PATH=UPSCALE_MODEL_PATH,TSP_SORT=TSP_SORT)

# # TEMP TK TESTING
# sort.MIND = .5
# sort.MAXD = .8
sort.MIND = sort.MIND*2
sort.MAXD = sort.MAXD*30

if USE_NOSEBRIDGE: sort.ORIGIN = 6

# CLUSTER_TYPE is passed to sort. THIS SEEMS REDUNDANT!!!
# sort.set_subset_landmarks(CLUSTER_TYPE)

start_img_name = "median"
start_site_image_id = None

# start_img_name = "start_image_id"
# start_site_image_id = 57987995

# 9774337 screaming hands to head 10
# 10528975 phone right hand pose 4
# 15940552 two fingers up pose 4
# 107124684 hands to mouth shock pose 16
# 56159379 holding phone belly pose 8
# 105444295 right hand pointing up pose 11
# 4004183 whisper finger to lips pose 5
# 9875985 fingers pointing up
# 121076470 eyes closed mouth open, arms raised victory
# 38217385 mouth open pointing at phone
# 2752224 scream, hands on head
# 3279908 hands to face excited
# 6050372 hands to mouth in shock
# 126993730 87210848 43008591 11099323 phone held up

# start_site_image_id is not working Aug 2024
# start_img_name = "start_site_image_id"
# start_site_image_id = "F/F4/538577009.jpg"
# start_site_image_id = "0/02/159079944-hopeful-happy-young-woman-looking-amazed-winning-prize-standing-white-background.jpg"
# start_site_image_id = "0/08/158083627-man-in-white-t-shirt-gesturing-with-his-hands-studio-cropped.jpg"

# for PFP
# start_img_name = "start_face_encodings"
# start_site_image_id = [-0.13242901861667633, 0.09738104045391083, 0.003530653193593025, -0.04780442640185356, -0.13073976337909698, 0.07189705967903137, -0.006513072177767754, -0.051335446536540985, 0.1768932193517685, -0.03729865700006485, 0.1137416809797287, 0.13994133472442627, -0.23849385976791382, -0.08209677785634995, 0.06067033112049103, 0.07974598556756973, -0.1882513463497162, -0.24926315248012543, -0.011344537138938904, -0.10508193075656891, 0.010317208245396614, 0.06348179280757904, 0.02852417528629303, 0.06981766223907471, -0.14760875701904297, -0.34729471802711487, -0.014949701726436615, -0.09429284185171127, 0.08592978119850159, -0.11939340829849243, 0.04517041891813278, 0.06180906295776367, -0.1773814857006073, 0.011621855199337006, 0.010536111891269684, 0.12963438034057617, -0.07557092607021332, 0.0027374476194381714, 0.2890719771385193, 0.0692337155342102, -0.17323020100593567, 0.0724603682756424, 0.021229337900877, 0.361629843711853, 0.250482439994812, 0.021974680945277214, 0.018878426402807236, -0.022722169756889343, 0.09668144583702087, -0.29601603746414185, 0.11375367641448975, 0.2568872570991516, 0.11404240131378174, 0.04999732971191406, 0.02831254154443741, -0.15830034017562866, -0.031099170446395874, 0.028748074546456337, -0.180643692612648, 0.13169123232364655, 0.058790236711502075, -0.0858338251709938, 0.029470380395650864, -0.002784252166748047, 0.2532877027988434, 0.07375448942184448, -0.11085735261440277, -0.12285713106393814, 0.11346398293972015, -0.19246435165405273, -0.1447266787290573, 0.054258447140455246, -0.1335202157497406, -0.1264294683933258, -0.23741140961647034, 0.07753928005695343, 0.3753989636898041, 0.08984167128801346, -0.18434450030326843, 0.042485352605581284, -0.08978638052940369, -0.03871896490454674, 0.06451354175806046, 0.08044029772281647, -0.11364202201366425, -0.1158837378025055, -0.10755209624767303, 0.044953495264053345, 0.2573489546775818, 0.049939051270484924, -0.07680445909500122, 0.20810386538505554, 0.09711501002311707, 0.05330953001976013, 0.08986716717481613, 0.0984266921877861, -0.036112621426582336, -0.011795245110988617, -0.15438663959503174, -0.027118921279907227, -0.012514196336269379, -0.11667540669441223, 0.04242435097694397, 0.13383115828037262, -0.18503828346729279, 0.19057676196098328, 0.017584845423698425, -0.005235005170106888, 0.010936722159385681, 0.08952657878398895, -0.1809171438217163, -0.07223983108997345, 0.16210225224494934, -0.264881432056427, 0.3121953308582306, 0.21528613567352295, 0.02137373574078083, 0.12006716430187225, 0.08322857320308685, 0.0802738219499588, -0.013485163450241089, 0.005497157573699951, -0.0893208310008049, -0.06330209970474243, 0.017513029277324677, -0.007281661033630371, 0.06451432406902313, 0.10179871320724487]
# start_site_image_id = [-0.10581238567829132, 0.07088741660118103, 0.013263327069580555, -0.08114208281040192, -0.13992470502853394, 0.012888573110103607, -0.009552985429763794, -0.05837436020374298, 0.026127614080905914, 0.001093447208404541, 0.15341515839099884, 0.044287052005529404, -0.2721121311187744, -0.13441239297389984, -0.0026458948850631714, 0.11877364665269852, -0.15712828934192657, -0.1471686214208603, -0.10886535793542862, -0.09967300295829773, -0.011542147025465965, 0.0059587229043245316, -0.047813788056373596, 0.04775381088256836, -0.1761886328458786, -0.3028424084186554, -0.03370347619056702, -0.15713511407375336, 0.0005495026707649231, -0.1361989825963974, -0.015080012381076813, 0.025705486536026, -0.12947367131710052, -0.03306383639574051, 0.018395066261291504, 0.0488845556974411, -0.092781201004982, -0.1401013731956482, 0.15860462188720703, 0.08463308960199356, -0.14735937118530273, -0.009462594985961914, 0.08969609439373016, 0.30084139108657837, 0.2646666169166565, 0.036240506917238235, 0.06943795830011368, -0.026887238025665283, 0.1546226292848587, -0.23532000184059143, 0.08313022553920746, 0.1324366182088852, 0.11709924787282944, 0.08266870677471161, 0.05900813639163971, -0.20212897658348083, 0.10378697514533997, 0.057002242654561996, -0.29036426544189453, 0.057467326521873474, 0.027950018644332886, -0.12515060603618622, -0.10928650200366974, -0.020537540316581726, 0.20214180648326874, 0.09844112396240234, -0.14632004499435425, -0.11949113011360168, 0.11953604221343994, -0.19659164547920227, -0.08529043942689896, 0.018321029841899872, -0.12027868628501892, -0.17337237298488617, -0.2806975543498993, 0.08081948757171631, 0.3730350434780121, 0.24808505177497864, -0.23414771258831024, 0.015145592391490936, -0.059556588530540466, -0.029826462268829346, 0.1383151412010193, 0.14856243133544922, -0.05812269449234009, -0.06512188911437988, -0.11708509922027588, -0.02537207305431366, 0.11158326268196106, 0.013382695615291595, -0.06716612726449966, 0.19349130988121033, 0.04249662160873413, -0.045811235904693604, 0.07281351834535599, 0.06558635830879211, -0.19487519562244415, 0.01120009645819664, -0.013865187764167786, -0.09125098586082458, 0.11730070412158966, -0.1202155202627182, 0.03125461935997009, 0.08074315637350082, -0.12886971235275269, 0.21560832858085632, -0.00488397479057312, 0.0329570397734642, 0.0005348101258277893, -0.12098219245672226, -0.07969430088996887, -0.015188425779342651, 0.11801530420780182, -0.2579388916492462, 0.18724043667316437, 0.18778195977210999, 0.0005423035472631454, 0.15530824661254883, 0.13494034111499786, 0.05073530972003937, -0.027213752269744873, -0.024363964796066284, -0.15827980637550354, -0.08806312084197998, 0.039876870810985565, -0.03350042551755905, 0.12625662982463837, 0.010933175683021545]
# start_site_image_id = [-0.19413329660892487, 0.12061071395874023, 0.05011634901165962, -0.08183299750089645, -0.07337753474712372, -0.01627560332417488, -0.0838925838470459, -0.08846378326416016, 0.050066739320755005, -0.04880788177251816, 0.2662230134010315, -0.005381084978580475, -0.2440241426229477, -0.07016980648040771, -0.032612159848213196, 0.14663562178611755, -0.12449649721384048, -0.07680836319923401, -0.16958947479724884, -0.041098661720752716, -0.032985441386699677, 0.08393426239490509, 0.06522658467292786, 0.027064107358455658, -0.0450870618224144, -0.2766939699649811, -0.09460555762052536, -0.020049870014190674, 0.18320757150650024, -0.07277096807956696, 0.03758329153060913, 0.020734600722789764, -0.16676828265190125, -0.008788809180259705, 0.06788013875484467, 0.14876432716846466, -0.05156975984573364, -0.08320155739784241, 0.23632089793682098, -0.03169070929288864, -0.14393861591815948, 0.007112100720405579, 0.039216622710227966, 0.24194732308387756, 0.20701384544372559, 0.014645536430180073, 0.00043237628415226936, -0.11287529766559601, 0.049542635679244995, -0.20981638133525848, 0.15882457792758942, 0.12680980563163757, 0.09500034153461456, 0.07604669779539108, 0.06692805886268616, -0.11759510636329651, 0.029098421335220337, 0.2152121663093567, -0.17994603514671326, 0.007263883948326111, 0.08920442312955856, -0.06217777729034424, -0.12775185704231262, -0.13567441701889038, 0.11721442639827728, 0.11643578112125397, -0.16315968334674835, -0.20449669659137726, 0.1373467743396759, -0.16275277733802795, -0.042377498000860214, 0.13762398064136505, -0.06410378217697144, -0.08502615243196487, -0.30058538913726807, 0.0696333795785904, 0.33976882696151733, 0.1762707233428955, -0.15098142623901367, 0.009935759007930756, 0.016703439876437187, -0.06178131699562073, 0.01687360554933548, 0.026908770203590393, -0.16367696225643158, -0.09296569228172302, -0.03673135116696358, 0.011298641562461853, 0.12101420015096664, 0.07616612315177917, -0.03630882874131203, 0.2021653950214386, 0.022061089053750038, 0.01185181736946106, 0.034367188811302185, 0.06430231779813766, -0.02623981237411499, -0.08930035680532455, -0.06906800717115402, -0.016731463372707367, 0.06791259348392487, -0.12447217106819153, 0.006740108132362366, 0.0978136956691742, -0.13495850563049316, 0.19274848699569702, -0.0044151246547698975, 0.05091022700071335, 0.116733118891716, 0.016980648040771484, -0.10611657053232193, -0.016641631722450256, 0.1956191062927246, -0.26104623079299927, 0.20304137468338013, 0.15483921766281128, 0.04521116614341736, 0.07446543127298355, 0.19201408326625824, 0.1501380205154419, -0.011784471571445465, 0.00905616581439972, -0.16708143055438995, -0.11018393188714981, 0.05920220538973808, -0.07374550402164459, 0.09465853869915009, 0.056482456624507904]

# start_img_name = "start_bbox"
# start_site_image_id = [94, 428, 428,0]

# no gap
# start_site_image_id = "5/58/95516714-happy-well-dressed-man-holding-a-gift-on-white-background.jpg"


d = None


# override io.db for testing mode
# db['name'] = "123test"

######################################################################################
######################################################################################
######################## CODE CHANGED FOR IS TENCH####################################
######################################################################################
######################################################################################

if not io.IS_TENCH:
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

    class IsNotDupeOf(Base):
        __tablename__ = 'IsNotDupeOf'
        image_id_i = Column(Integer, ForeignKey('Images.image_id'), primary_key=True)
        image_id_j = Column(Integer, ForeignKey('Encodings.encoding_id'), primary_key=True)

    # not currently in use
    if IS_HAND_POSE_FUSION and CLUSTER_TYPE_2:
        ClustersTable_name_2 = CLUSTER_TYPE_2
        ImagesClustersTable_name_2 = "Images"+CLUSTER_TYPE_2

        class Clusters_2(Base):
            __tablename__ = ClustersTable_name_2

            cluster_id = Column(Integer, primary_key=True, autoincrement=True)
            cluster_median = Column(BLOB)

        class ImagesClusters_2(Base):
            __tablename__ = ImagesClustersTable_name_2

            image_id = Column(Integer, ForeignKey(Images.image_id, ondelete="CASCADE"), primary_key=True)
            cluster_id = Column(Integer, ForeignKey(f'{ClustersTable_name_2}.cluster_id', ondelete="CASCADE"))

    def prep_cluster_medians():
        # store the results in a dictionary where the key is the cluster_id
        if results:
            cluster_medians = {}
            for i, row in enumerate(results, start=1):
                cluster_median = pickle.loads(row.cluster_median)
                cluster_medians[i] = cluster_median
                # print("cluster_medians", i, cluster_median)
                N_CLUSTERS = i # will be the last cluster_id which is count of clusters

    # TK IS_HAND_POSE_FUSION ??
    if IS_CLUSTER or IS_ONE_CLUSTER or IS_HAND_POSE_FUSION:
        # select cluster_median from Clusters
        results = session.execute(select(Clusters.cluster_id, Clusters.cluster_median)).fetchall()
        cluster_medians, N_CLUSTERS = sort.prep_cluster_medians(results)
        sort.cluster_medians = cluster_medians
        # if any of the cluster_medians are empty, then we need to resegment
        # print("cluster results", len(results))
        # print("cluster_medians", len(cluster_medians))
        if cluster_medians is None:
        # if not any(cluster_medians):
            print("cluster results are empty", cluster_medians)
    if IS_HANDS or IS_ONE_HAND:
        results = session.execute(select(Hands.cluster_id, Hands.cluster_median)).fetchall()
        hands_medians, N_HANDS = sort.prep_cluster_medians(results)
        sort.hands_medians = hands_medians
        print("hands results len", len(results))

        # # store the results in a dictionary where the key is the cluster_id
        # if results:
        #     cluster_medians = {}
        #     for i, row in enumerate(results, start=1):
        #         cluster_median = pickle.loads(row.cluster_median)
        #         cluster_medians[i] = cluster_median
        #         # print("cluster_medians", i, cluster_median)
        #         N_CLUSTERS = i # will be the last cluster_id which is count of clusters
        #     sort.set_cluster_medians(cluster_medians)




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

    def selectSQL(cluster_no=None, topic_no=None, hsv_cluster=None):
        global SELECT, FROM, WHERE, LIMIT, WrapperTopicTable
        local_where = WHERE
        from_affect = where_affect = from_hsv = where_hsv = " "
        cluster_dict_AND = cluster_dict_NOT = None

        def is_query_list_string(topic_no, inclusion=True):
            if inclusion:
                list_comparison = "IN"
                int_comparison = "="
            else:
                list_comparison = "NOT IN"
                int_comparison = "!="

            if isinstance(topic_no, list):
                # Convert the list into a comma-separated string
                topic_ids = ', '.join(map(str, topic_no))
                query_string = f" {list_comparison} ({topic_ids})"
            else:
                query_string = f" {int_comparison} {str(topic_no)}"
            return query_string

        def access_keyword_dict(cluster_topic_no):
            # check if the topic_no has related keywords in the KEYWORD_DICT
            cluster_topic_no = int(math.floor(float(cluster_topic_no)))
            cluster_dict_values = KEYWORD_DICT.get(cluster_topic_no, None)
            if cluster_dict_values is not None:
                cluster_topic_OR = cluster_dict_values.get("OR", None)
                cluster_dict_AND = cluster_dict_values.get("AND", None)
                cluster_dict_NOT = cluster_dict_values.get("NOT", None)
                # merge the OR values and the cluster_topic_no
                if cluster_topic_OR is not None:
                    cluster_topic_OR.append(cluster_topic_no)
                    cluster_topic_no = cluster_topic_OR
                # AND and NOT will be handled below
            return cluster_topic_no, cluster_dict_AND, cluster_dict_NOT

        def cluster_topic_select(cluster_topic_table, cluster_topic_no):
            # convert topic_no to a query string
            IN_or_equal_query_string = is_query_list_string(cluster_topic_no)
            cluster = f"AND {cluster_topic_table} {IN_or_equal_query_string} "
            return cluster
            # if isinstance(cluster_topic_no, list):
            #     # Convert the list into a comma-separated string
            #     cluster_topic_ids = ', '.join(map(str, cluster_topic_no))
            #     # Use the IN operator to check if topic_id is in the list of values
            #     return f"AND {cluster_topic_table} IN ({cluster_topic_ids}) "
            # else:
            #     # If topic_no is not a list, simply check for equality
            #     return f"AND {cluster_topic_table} = {str(cluster_topic_no)} "            

        def handle_hsv_cluster_lists(cluster_topic_no, hsv_cluster):
            if isinstance(hsv_cluster, list) and hsv_cluster[0] == cluster_topic_no:
                # cluster_topic_no is a list that contains [metacluster, hsv]
                # remove the hsv from the list for passing into select_map_images
                hsv_cluster = hsv_cluster[1]
            return hsv_cluster

        cluster = " " #declare as empty string
        print(f"[selectSQL] cluster_no {cluster_no} and topic_no {topic_no} and hsv_cluster {hsv_cluster}")
        # handle the case where hsv_cluster is passed in as part of a list with cluster_no
        if hsv_cluster and isinstance(hsv_cluster, list) and len(hsv_cluster) == 2:
            hsv_cluster = handle_hsv_cluster_lists(cluster_no, hsv_cluster)
            print(f"after handling hsv, cluster_no {cluster_no} and hsv_cluster {hsv_cluster}")
            
        if IS_HAND_POSE_FUSION:
            if isinstance(cluster_no, list):
                print("cluster_no is a list", cluster_no)
                # set target column based on SORT_TYPE, arms3D means we have a meta_cluster_id
                if SORT_TYPE == "arms3D": cluster_target_col = "cmp.meta_cluster_id"
                else: cluster_target_col = "ihp.cluster_id"

                # we have two values, C1 and C2. C1 should be IHP, C2 should be IH
                cluster += f" AND {cluster_target_col} = {str(cluster_no[0])} "            
                if cluster_no[1]: cluster += f" AND ih.cluster_id = {str(cluster_no[1])} "            
        # elif cluster_no is not None or topic_no is not None:
        elif IS_CLUSTER or IS_ONE_CLUSTER:
            cluster += cluster_topic_select("ic.cluster_id", cluster_no)
        if IS_TOPICS or IS_ONE_TOPIC:
            if IS_TOPICS and IS_ONE_TOPIC and USE_AFFECT_GROUPS and WrapperTopicTable is not None:

                # topic fusion, so join to a second topics table
                from_affect = f" JOIN {WrapperTopicTable} iwt ON s.image_id = iwt.image_id "
                where_affect = f" AND iwt.topic_id = {TOPIC_NO[0]} AND iwt.topic_score > .3"
            # cluster +=f"AND it.topic_id = {str(topic_no)} "
            if N_TOPICS == 100: 
                this_id = "keyword_id"
                # check if the topic_no has related keywords in the KEYWORD_DICT
                # if it doesn't, it just returns the same topic_no
                topic_no, cluster_dict_AND, cluster_dict_NOT = access_keyword_dict(topic_no)
                print(f"after accessing keyword dict, topic_no is {topic_no}")
            else: 
                this_id = "topic_id"

            # convert topic_no to a query string
            IN_or_equal_topic_ids_string = is_query_list_string(topic_no)
            cluster += f"AND it.{this_id} {IN_or_equal_topic_ids_string} "

        if hsv_cluster is not None:
            IN_or_equal_hsv_string = is_query_list_string(hsv_cluster)
            where_hsv = f" AND ihsv.cluster_id {IN_or_equal_hsv_string} "

        if len(where_hsv) > 2:
            # if hsv is not " ", then we need to join the table
            from_hsv = " JOIN ImagesHSV ihsv ON s.image_id = ihsv.image_id "
            # set cluster_no here, as it doesn't catch any of the other condtionals. (admittedly messy)
            local_where += f" AND cmp.meta_cluster_id = {str(cluster_no)} "

        if bool(cluster_dict_AND):
            IN_or_equal_hsv_string = is_query_list_string(cluster_dict_AND)
            cluster +=  f" AND it.{this_id} {IN_or_equal_hsv_string} "
        if bool(cluster_dict_NOT):
            IN_or_equal_hsv_string = is_query_list_string(cluster_dict_NOT, inclusion=False)
            cluster +=  f" AND it.{this_id} {IN_or_equal_hsv_string} "

        print(f"cluster SELECT is {cluster}")

        selectsql = f"SELECT {SELECT} FROM {FROM + from_affect + from_hsv} WHERE {local_where + where_affect + where_hsv} {cluster} LIMIT {str(LIMIT)};"
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

######################################################################################
######################################################################################
######################## END CODE CHANGED FOR IS TENCH################################
######################################################################################
######################################################################################

###################
# SORT FUNCTIONS  #
###################


# need to pass through start_img_enc rather than start_img_name
# for linear it is in the df_enc, but for itter, the start_img_name is in prev df_enc
# takes a dataframe of images and encodings and returns a df sorted by distance

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

def sort_by_face_dist_NN(df_enc):
    
    # create emtpy df_sorted with the same columns as df_enc
    df_sorted = pd.DataFrame(columns = df_enc.columns)

    # # debugging -- will save full df_enc to csv
    # df_enc_outpath = os.path.join(sort.counter_dict["outfolder"],"df_enc.csv")
    # # write the dataframe df_enc to csv at df_enc_outpath
    # df_enc.to_csv(df_enc_outpath, index=False)


    if sort.CUTOFF < len(df_enc.index): itters = sort.CUTOFF
    else: itters = len(df_enc.index)
    # print("sort_by_face_dist_NN itters is", itters, "sort.CUTOFF is", sort.CUTOFF)

    # input enc1, df_128_enc, df_33_lmsNN
    # df = pd.DataFrame(face_distances, columns =['dist', 'folder', 'filename','site_name_id','face_landmarks', 'bbox'])


    # find the df_enc row with image_id = 10498233
    # test_row = df_enc.loc[df_enc['image_id'] == 10498233]
    # # print body_landmarks for this row
    # print("body_landmarks for test_row")
    # test_lms = test_row['body_landmarks']
    # print(test_lms)

    # print("row from df_enc with image_id = 893")
    # test_row = df_enc.loc[df_enc['image_id'] == 893]
    # print(test_row)    
    def get_closest_knn_or_break(df_enc, df_sorted):
        # send in both dfs, and return same dfs with 1+ rows sorted
        print("BEFORE sort_by_face_dist_NN _ for loop df_enc is", df_enc)
        is_break = False
        df_enc, df_sorted = sort.get_closest_df_NN(df_enc, df_sorted)

        print("AFTER sort_by_face_dist_NN _ for loop df_enc is", df_enc)
        print("AFTER sort_by_face_dist_NN _ for loop df_sorted is", df_sorted)

        # # test to see if body_landmarks for row with image_id = 5251199 still is the same as test_lms
        # retest_row = df_enc.loc[df_enc['image_id'] == 10498233]
        # print("body_landmarks for retest_row")
        # retest_lms = retest_row['body_landmarks']
        # print(retest_lms)
        # calculate any different between test_lms to retest_lms


        # Break out of the loop if greater than MAXDIST
        if df_enc.empty:
            is_break = True            
        elif ONE_SHOT:
            df_sorted = pd.concat([df_sorted, df_enc])
            # only return the first x rows
            df_sorted = df_sorted.head(sort.CUTOFF)
            print("one shot, breaking out", df_sorted)
            is_break = True
        # commenting out SHOT_CLOCK for now, Sept 28
        # elif dist > sort.MAXD and sort.SHOT_CLOCK != 0:
        else:
            print("df_sorted.iloc[-1], " , df_sorted.iloc[-1])
            dist = df_sorted.iloc[-1]['dist_enc1']
            print("sort_by_face_dist_NN _ for loop dist is", dist)
            if dist > sort.MAXD or df_enc.empty:
                print("should breakout, dist is", dist)
                is_break = True
        return df_enc, df_sorted, is_break

    if CHOP_FIRST:
        print(f"CHOP_FIRST is true with sort.counter_dict['start_img_name'], {sort.counter_dict['start_img_name']}")
        if sort.counter_dict["start_img_name"] is None:
            sort.counter_dict["start_img_name"] = start_img_name
        print(f"CHOP_FIRST is true with sort.counter_dict['start_img_name'], {sort.counter_dict['start_img_name']}")

        df_enc, df_sorted, is_break = get_closest_knn_or_break(df_enc, df_sorted)
        df_enc = df_sorted.head(sort.CUTOFF)
        print("AFTER CHOP_FIRST df_enc is", df_enc)
        print("AFTER CHOP_FIRST df_sorted is", df_sorted)


    ## SATYAM THIS IS WHAT WILL BE REPLACE BY TSP
    if TSP_SORT is True:
        

        # # Option 1: Work entirely with expanded dataframe
        # df_clean = expand_face_encodings(df_enc)
        # sorter.set_TSP_sort(df_clean, START_IDX=None, END_IDX=None)
        # df_sorted = sorter.do_TSP_SORT(df_clean)  # Same df!

        # Option 2: If you need to maintain link to original df_enc
        # You'll need to track the mapping yourself
        df_clean = expand_face_encodings(df_enc)
        df_clean['original_index'] = df_clean.index  # or however they map

        sorter.set_TSP_sort(df_clean, START_IDX=None, END_IDX=None)
        df_sorted_clean = sorter.do_TSP_SORT(df_clean)

        # Then map back if needed
        df_sorted = df_enc.iloc[df_sorted_clean['original_index']].reset_index(drop=True)

        # # Prepare and sort
        # df_clean = expand_face_encodings(df_enc)
        # sorter.set_TSP_sort(df_clean, START_IDX=None, END_IDX=None)
        # df_sorted = sorter.do_TSP_SORT(df_enc)

        # df_clean=expand_face_encodings(df_enc)
        # sort.set_TSP_sort(df_clean,START_IDX=None,END_IDX=None)
        # # df_sorted = sort.get_closest_df_NN(df_enc, df_sorted, start_image_id, end_image_id)
        # df_sorted=sort.do_TSP_SORT(df_enc)
    else:
        for i in range(itters):
            # print("sort_by_face_dist_NN _ for loop itters i is", i)
            ## Find closest
            try:
                df_enc, df_sorted, is_break = get_closest_knn_or_break(df_enc, df_sorted)
                if is_break: break
            except Exception as e:
                print("exception on going to get closest")
                print(str(e))
                traceback.print_exc()
    ## SATYAM THIS THE END OF WHAT WILL BE REPLACE BY TSP

    # use the colum site_name_id to asign the value of io.folder_list[site_name_id] to the folder column
    df_sorted['folder'] = df_sorted['site_name_id'].apply(lambda x: io.folder_list[x])
    
    # rename the distance column to dist
    df_sorted.rename(columns={'dist_enc1': 'dist'}, inplace=True)

    print("df_sorted", df_sorted)

    # # debugging -- will save full df_enc to csv
    # df_sorted_outpath = os.path.join(sort.counter_dict["outfolder"],"df_sorted.csv")
    # df_sorted.to_csv(df_sorted_outpath, index=False)

    # # make a list of df_sorted dist
    # dist_list = df_sorted['dist'].tolist()
    # print("dist_list", dist_list)
    
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
    
    def set_sort_col():
        if SORT_TYPE in ["body3D", "arms3D"] or CLUSTER_TYPE in ["BodyPoses3D", "ArmsPoses3D"]:
            # have to handle both, because handposefusion redefines SORT_TYPE 
            source_col = sort_column = "body_landmarks_3D"
        elif SORT_TYPE == "planar_body":
            if CLUSTER_TYPE == "HandsPositions":
                source_col = sort_column = "hand_landmarks"
                # source_col = None
                # source_col_2 = "right_hand_landmarks_norm"
            else:
                sort_column = "body_landmarks_array"
                source_col = "body_landmarks_normalized"
                # source_col_2 = None
        elif SORT_TYPE == "planar_hands" or SORT_TYPE == "planar_hands_USE_ALL" or SORT_TYPE == "fingertips_positions":
            source_col = sort_column = "hand_landmarks"
            # source_col = sort_column = "right_hand_landmarks_norm"
            # source_col = None
            # source_col_2 = "right_hand_landmarks_norm"
        elif SORT_TYPE == "128d":
            source_col = sort_column = "face_encodings68"

        return sort_column, source_col

    print("prep_encodings_NN df_segment columns", df_segment.columns)
    sort_column, source_col = set_sort_col()
    print(f"degugging df_segment prep encodings for {source_col}:", df_segment.columns)      
    # drop rows where body_landmarks_normalized is None
    # TK this needs to be adapted to handle left vs right hand. 
    # subset needs to be both of them, if both are na
    if not sort_column == "hand_landmarks":
        # hand_landmarks are all giving 0's if null, so no NA
        df_segment = df_segment.dropna(subset=[source_col])
        print("df_segment length", len(df_segment.index))
    
        # print rows where body_landmarks_normalized is None
        null_encodings = df_segment[df_segment[source_col].isnull()]
        if not null_encodings.empty:
            print("Rows with invalid body_landmarks_normalized data:")
            print(null_encodings)
        
    print(df_segment.size)
    print(df_segment[source_col])


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

    # convert body_landmarks_normalized to a list (only for SUBSET_LANDMARKS)
    # if sort_column == "hand_landmarks_array":
    #     print(f"prepping {sort_column} for {CLUSTER_TYPE}")
    #     print(df_segment[source_col].head())
    #     print(df_segment[source_col][0])
    #     print(len(df_segment[source_col][0]))
    #     # df_segment[sort_column] = df_segment["left_hand_landmarks_norm"].apply(lambda x: sort.prep_enc(x, structure="list3")) + df_segment["right_hand_landmarks_norm"].apply(lambda x: sort.prep_enc(x, structure="list")) # swittching to 3d
    #     df_segment[sort_column] = df_segment[source_col].apply(lambda x: sort.prep_enc(x, structure="list3"))
    #     print(sort_column, df_segment[sort_column].head())

    if sort_column == "body_landmarks_array":
        print(f"prepping {sort_column} for {CLUSTER_TYPE}")
        print(df_segment['body_landmarks_normalized'].head())
        df_segment[sort_column] = df_segment["body_landmarks_normalized"].apply(lambda x: sort.prep_enc(x, structure="list")) # swittching to 3d
        print(sort_column, df_segment[sort_column].head())
            # if USE_HEAD_POSE:
            #     df_segment = df_segment.apply(sort.weight_face_pose, axis=1)            
            #     head_columns = ['face_x', 'face_y', 'face_z', 'mouth_gap']

            #     # Add the face_x, face_y, face_z, and mouth_gap to the body_landmarks_array
            #     df_segment["body_landmarks_array"] = df_segment.apply(
            #         lambda row: row["body_landmarks_array"] + [row[col] for col in head_columns] if isinstance(row["body_landmarks_array"], list) else row["body_landmarks_array"],
            #         axis=1
            #     )            
            #     # print("body_landmarks_array after adding head pose", df_segment["body_landmarks_array"])
        
    if USE_HEAD_POSE:
        # not currently in use, so may need debugging
        df_segment = df_segment.apply(sort.weight_face_pose, axis=1)            
        head_columns = ['face_x', 'face_y', 'face_z', 'mouth_gap']

        # Add the face_x, face_y, face_z, and mouth_gap to the body_landmarks_array
        df_segment[sort_column] = df_segment.apply(
            lambda row: row[sort_column] + [row[col] for col in head_columns] if isinstance(row[sort_column], list) else row[sort_column],
            axis=1
        )            
        # print("body_landmarks_array after adding head pose", df_segment["body_landmarks_array"])

    if sort_column == "body_landmarks_array" and DROP_LOW_VIS:

        # if planar_body drop rows where self.BODY_LMS are low visibility
        df_segment['hand_visible'] = df_segment.apply(lambda row: any(sort.test_landmarks_vis(row)), axis=1)

        # delete rows where hand_visible is false
        df_segment = df_segment[df_segment['hand_visible'] == True].reset_index(drop=True)
        # df_segment = df_segment[df_segment['hand_visible'] == True]
        print("df_segment length visible hands", len(df_segment.index))

    return df_segment

def compare_images(last_image, img, face_landmarks_or_df, bbox_or_index):
    # for some reason the function is called with either a df and index, or face_landmarks and bbox
    # rw to handle both here. 
    if isinstance(face_landmarks_or_df, pd.DataFrame):
        df_sorted = face_landmarks_or_df
        index = bbox_or_index
        face_landmarks, bbox = df_sorted.iloc[index]['face_landmarks'], df_sorted.iloc[index]['bbox']
    else:
        face_landmarks = face_landmarks_or_df
        bbox = bbox_or_index

    is_face = None
    face_diff = 100 # declaring full value, for first round
    skip_face = False
    #crop image here:
    
    print(f". ---  compare_images for {SORT_TYPE} with EXPAND {sort.EXPAND} and FULL_BODY {FULL_BODY} and self.image_edge_multiplier {sort.image_edge_multiplier}")
    # this is where the image gets cropped or expanded
    if sort.EXPAND:
        cropped_image, resize = sort.expand_image(img, face_landmarks, bbox)
        
        if FULL_BODY or AUTO_EDGE_CROP: 
            print('running AUTO_EDGE_CROP')
            cropped_image = sort.auto_edge_crop(df_sorted, index, cropped_image, resize)
        else:   
            # cropp the 25K image back down to 10K
            # does this based on the incremental dimensions
            cropped_image = sort.crop_whitespace(cropped_image)
        
        # else: 
        #     # trim the top 25% of the image
        #     cropped_image = sort.trim_top_crop(cropped_image, 0.25)
        is_inpaint = False
    else:
        cropped_image, is_inpaint = sort.crop_image(img, face_landmarks, bbox)


    # this code takes image i, and blends it with the subsequent image
    # next step is to test to see if mp can recognize a face in the image
    # if no face, a bad blend, try again with i+2, etc. 
    if cropped_image is not None and not is_inpaint:
        if VERBOSE: print("have a cropped image trying to save", cropped_image.shape)
        try:
            if not sort.counter_dict["first_run"]:
                if VERBOSE:  print("testing is_face")

                if SORT_TYPE not in ("planar_body", "body3D"):
                #     # skipping test_pair for body, b/c it is meant for face
                    is_face = sort.test_pair(last_image, cropped_image)

                # sept 2025 removing this, as it is now handled in separate dedupe process
                # if is_face or SORT_TYPE in ("planar_body", "body3D"):
                #     if VERBOSE: print("testing mse to see if same image")
                #     error,face_diff = sort.unique_face(last_image,cropped_image)
                #     # if face diff is less than something very low (.04), then it is a duplicate and we are done
                #     # elif face diff is less than some other larger value, run additional tests
                #     # these tests will will check 
                #     if VERBOSE: print("compare_images face_diff ", face_diff)
                #     # if VERBOSE: print ("mse ", mse) ########## mse not a variable
                else:
                    print("failed is_face test")
                    # use cv2 to place last_image and cropped_image side by side in a new image

                    # I'm not 100% sure what this is doing, given that this is the FAIL loop

                    #Tench commented out 09.07.25 due to auto_edge_crop incompatability 
                    # height = max(last_image.shape[0], cropped_image.shape[0])
                    # last_image = cv2.resize(last_image, (last_image.shape[1], height))
                    # cropped_image = cv2.resize(cropped_image, (cropped_image.shape[1], height))
                    # #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
                    # # Concatenate images horizontally
                    # combined_image = cv2.hconcat([last_image, cropped_image])
                    # outpath_notface = os.path.join(sort.counter_dict["outfolder"],"notface",sort.counter_dict['last_description'][:30]+".jpg")
                    # # sort.not_make_face.append(outpath_notfacecombined_image) ########## variable name error
                    # # Save the new image
                    # #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
              #  face_embeddings_distance, body_landmarks_distance, same_description, same_site_name_id = sort.check_metadata_for_duplicate(df_sorted, index)

            else:
                print("first round, skipping the pair test")
        except:
            print("last_image try failed")
        # if is_face or first_run and sort.resize_factor < sort.resize_max:
        if face_diff > sort.FACE_DUPE_DIST or sort.counter_dict["first_run"]:
            # if successful, prepare for the next round
            sort.counter_dict["first_run"] = False
            last_image = cropped_image
            sort.counter_dict["good_count"] += 1
        else: 
            sort.counter_dict["isnot_face_count"] += 1
            print("pair do not make a face, skipping <<< is this really true? Isn't this for dupes?")
            return None, face_diff, True
        
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
    try: 
        selfie_top, selfie_bottom, selfie_left, selfie_right = selfie_bbox["top"], selfie_bbox["top"]+height, selfie_bbox["left"],selfie_bbox["left"]+width
    except: 
        print("selfie_bbox is None, returning None,None")
        return None, None
    # test to see if top strip is consistent -- eg a seamless background
    area = [0,selfie_top],[0,width]
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
        test_top = [[0,top+selfie_top],[0,inpaint_image.shape[1]]]
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

    blur_radius_left=io.oddify(selfie_left*blur_radius)
    blur_radius_right=io.oddify(selfie_right*blur_radius)
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
            invert_dist = selfie_top
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
    # shift sort.nose_2d by the same amount
    nose_2d_list = list(sort.nose_2d)
    nose_2d_list[0] = nose_2d_list[0] + x0
    nose_2d_list[1] = nose_2d_list[1] + y0
    sort.nose_2d = tuple(nose_2d_list)
    if sort.VERBOSE:print("after shifting",bbox)
    return bbox

def linear_test_df(df_sorted, itter=None):

    def save_image_metas(row):
        if sort.VERBOSE: print("row", row)
        if sort.VERBOSE: print("save_image_metas for use in TTS")
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
            if description is None:
                print("no description, skipping")
                description = None
            else:
                description = description
            metas = [image_id, description, topic_score]
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
        # aspect_ratio = '_'.join(sort.image_edge_multiplier)
        aspect_ratio = '_'.join(str(v) for v in sort.image_edge_multiplier)
        if INPAINT_COLOR:
            inpaint_file=os.path.join(os.path.dirname(row['folder']), os.path.basename(row['folder'])+"_inpaint_"+INPAINT_COLOR+"_"+aspect_ratio,row['imagename'])
        else:
            inpaint_file=os.path.join(os.path.dirname(row['folder']), os.path.basename(row['folder'])+"_inpaint_"+aspect_ratio,row['imagename'])
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
            print("selfie_bbox", selfie_bbox)
            if selfie_bbox is not None and selfie_bbox["left"]==0 and not INPAINT_COLOR: 
                if VERBOSE: print("head hits the top of the image, skipping -------------------> bailout !!!!!!!!!!!!!!!!!")
                do_inpaint = False
                bailout=True
            # no longer skipping because I trim above
            # elif row["site_name_id"] == 2 and extension_pixels["bottom"]>0: 
            #     if VERBOSE: print("shutter at the bottom, skipping -------------------> bailout !!!!!!!!!!!!!!!!!")
            #     do_inpaint = False
            #     bailout=True
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

                if INPAINT_COLOR:
                    print("going to inpaint black")
                    extended_img,mask,cornermask=sort.prepare_mask(img,extension_pixels,color=INPAINT_COLOR)
                    # if the image is black, then use the black image as the inpaint
                    # this is to avoid using a white image for the inpaint
                    # Fill the extended area with black instead of inpainting
                    inpaint_image = extended_img.copy()
                    # Set the masked (extended) area to black
                    # inpaint_image[mask > 0] = 0
                    print("just inpaint black", inpaint_image.shape)
                else:
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
                    print("inpaint_image shape after black OR lama extend",np.shape(inpaint_image))
                    # inpaint_image = inpaint_image[y:y+h, x:x+w]
                    inpaint_image = inpaint_image[0:np.shape(extended_img)[0],0:np.shape(extended_img)[1]]
                    # inpaint_image = cv2.crop(inpaint_image, (np.shape(extended_img)[1],np.shape(extended_img)[0]))
                    print("inpaint_image shape after transform",np.shape(inpaint_image))
                    print("extended_img shape after transform",np.shape(extended_img))                
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

    # def trim_bottom(img, site_name_id):
    #     if VERBOSE: print("trimming bottom")
    #     if site_name_id == 2: trim = 100
    #     elif site_name_id == 9: trim = 90
    #     img = img[0:img.shape[0]-trim, 0:img.shape[1]]
    #     return img
    
    #itter is a cap, to stop the process after a certain number of rounds
    print('linear_test_df writing images for this many images:', len(df_sorted))
    imgfileprefix = f"X{str(sort.XLOW)}-{str(sort.XHIGH)}_Y{str(sort.YLOW)}-{str(sort.YHIGH)}_Z{str(sort.ZLOW)}-{str(sort.ZHIGH)}_{sort.counter_dict['cluster_no']}"
    print(imgfileprefix)
    print("first row", df_sorted.iloc[0])
    if sort.ORIGIN == 6: sort.NOSE_BRIDGE_DIST = sort.calc_nose_bridge_dist(df_sorted.iloc[0]['face_landmarks'])    
    # print("nose bridge dist", sort.NOSE_BRIDGE_DIST)
    good = 0
    # img_list = []
    metas_list = []
    description = None
    cropped_image = np.array([-10])
    for index, row in df_sorted.iterrows():
        print('-- linear_test_df [-] in loop, index is', str(index))
        if sort.VERBOSE: print("row", row)
        sort.this_nose_bridge_dist = None
        try:
            # Open the Image
            imgfilename = const_imgfilename_NN(row['image_id'], df_sorted, imgfileprefix)
            outpath = os.path.join(sort.counter_dict["outfolder"],imgfilename)
            open_path = os.path.join(io.folder_list[row['site_name_id']],row['imagename'])
            print("open_path", open_path)
            print("open_path constructed from", io.folder_list[row['site_name_id']],row['folder'],row['imagename'])
            # print("io.folder_list:",row['site_name_id'],io.folder_list[row['site_name_id']])
            description = row['description']
            try:
                img = cv2.imread(open_path)

                if DRAW_TEST_LMS:
                    # for testing, draw in points
                    # list(range(20)
                    print("about to draw landmarks, subset", sort.SUBSET_LANDMARKS)
                    landmarks_2d = sort.get_landmarks_2d(row['left_hand_landmarks'], sort.SUBSET_LANDMARKS, "list")
                    landmarks_2d2 = sort.get_landmarks_2d(row['right_hand_landmarks'], sort.SUBSET_LANDMARKS, "list")
                    landmarks_2d = landmarks_2d + landmarks_2d2
                    print("landmarks_2d before drawing", landmarks_2d)
                    # transpose x and y in the landmarks    
                    img = sort.draw_point(img, landmarks_2d, index = 0)
                    # img = sort.draw_point(img, [.25,.25,.5,.5], index = 0)
                    
                    # cv2.imshow("img", img)

                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                if row["site_name_id"] in [2,9]: 
                    if VERBOSE: print("shutter alamy trimming at the bottom")
                    img = sort.trim_bottom(img, row["site_name_id"])
            except:
                print("trim failed")
                continue

            # establish the origin
            sort.get_image_face_data(img, row['face_landmarks'], row['bbox'])

            # control distance 
            this_dist = row.get('dist', None)
            if this_dist is not None and this_dist > sort.MAXD:
                sort.counter_dict["failed_dist_count"] += 1
                print("MAXDIST too big:" , str(sort.MAXD))
                continue

            # compare_images to make sure they are face and not the same
            # last_image is cv2 np.array
            cropped_image, face_diff, skip_face = compare_images(sort.counter_dict["last_image"], img, df_sorted, index)
            
            # test and handle duplicates 
            if cropped_image is None and skip_face:
                if VERBOSE: print("face_diff", face_diff)
                if face_diff == 0:
                    is_dupe_of = True
                elif SORT_TYPE == "planar_body" and face_diff < 10:
                    if VERBOSE: print("face_diff is small, so will check description")
                    if description == sort.counter_dict["last_description"]: 
                        if VERBOSE: print("same description, going to record as a dupe")
                        is_dupe_of = True
                    else:
                        pass
                        if VERBOSE: print("different description, not a dupe")
                        if VERBOSE: print("description", description)
                        if VERBOSE: print("sort.counter_dict[last_description]", sort.counter_dict["last_description"])

                        is_dupe_of = False
                ## TK NEED TO ADD IN CONDITIONAL FOR FACE DUPE DIST
                else:
                    is_dupe_of = False

                if is_dupe_of:
                    print(f"identical image, going to record {row['image_id']} as a dupe of ", sort.counter_dict["last_image_id"])
                    
                    session.query(Encodings).filter(Encodings.image_id == row['image_id']).update({
                            Encodings.is_dupe_of: sort.counter_dict["last_image_id"]
                        }, synchronize_session=False)
                    session.query(SegmentTable).filter(SegmentTable.image_id == row['image_id']).update({
                            SegmentTable.is_dupe_of: sort.counter_dict["last_image_id"]
                        }, synchronize_session=False)
                    session.commit()

            # handle USE_ALL and inpainting
            if skip_face and not USE_ALL:
                print("skipping face")
                continue
            elif cropped_image is None:
            # if len(cropped_image)==1 and (OUTPAINT or INPAINT):
                print("gotta paint that shizzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
                cropped_image, face_diff = in_out_paint(img, row)

            # handle/debug counter_dict
            temp_first_run = sort.counter_dict["first_run"]
            if VERBOSE: print("temp_first_run", temp_first_run)
            if sort.counter_dict["first_run"]:
                sort.counter_dict["last_description"] = description
                if VERBOSE: print("first run, setting last_description")
            elif face_diff and face_diff < sort.CHECK_DESC_DIST:
                # TK this doesn't seem to do anything, but maybe should be in dupe detection
                if VERBOSE: print("face_diff is small, so will check description:", face_diff)
                # temp, until resegmenting
                if VERBOSE: print("description", description)
                if VERBOSE: print("sort.counter_dict[last_description]", sort.counter_dict["last_description"])
                if description == sort.counter_dict["last_description"]:
                    print("same description!!!")
                    

            # save image
            if cropped_image is not None:
                cv2.imwrite(outpath, cropped_image)
                good += 1
                save_image_metas(row)
                sort.counter_dict["start_img_name"] = row['imagename']
                print("saved: ",outpath)
                sort.counter_dict["counter"] += 1
                if itter and good > itter:
                    print("breaking after this many itters,", str(good), str(itter))
                    continue
                sort.counter_dict["last_image"] = cropped_image  #last pair in list, second item in pair
                sort.counter_dict["last_image_id"] = row['image_id']  #last pair in list, second item in pair
            else:
                print("cropped_image is None")

        except Exception as e:
            traceback.print_exc()
            print(str(e))
        if VERBOSE: print("metas_list")
        if VERBOSE: print(metas_list)
    return

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


def process_linear(start_img_name, df_segment, file_prefix, sort):
    # linear sort by encoding distance
    print("processing linear")

    # redundant I think     
    # sort.set_counters(io.ROOT,cluster_no, start_img_name, start_site_image_id)  
    # print("sort.counter_dict after sort.set_counters", sort.counter_dict)

    # preps the encodings for sort
    # df_enc, df_128_enc, df_33_lms = prep_encodings(df_segment)
    df_enc = prep_encodings_NN(df_segment)
    segment_count = df_enc.shape[0]
    if VERBOSE: print("sort.counter_dict after prep_encodings_NN", sort.counter_dict)

    # if results in df_enc, then sort by face distance
    if not df_enc.empty:
        # # get dataframe sorted by distance
        df_sorted = sort_by_face_dist_NN(df_enc)
        # df_sorted = sort_by_face_dist(df_enc, df_128_enc, df_33_lms)

        # TK this is where i save df_sorted to csv
        # check to see if CSV_FOLDER exists and create if not
        if not os.path.exists(CSV_FOLDER): os.makedirs(CSV_FOLDER)
        df_sorted.to_csv(f"{CSV_FOLDER}/df_sorted_{file_prefix}_ct{segment_count}.csv", index=False)

        # test to see if they make good faces
        # write_images(img_list)
        # write_images(sort.not_make_face)
        # print_counters()

def parse_cluster_no(this_cluster):
    # if this_cluster is a list, then assign the first one to cluster_no
    # temp fix, to deal with passing in two values for FUSION
    # select on both, sort on CLUSTER_NO
    # for FUSION, CLUSTER_NO is HAND_POSITION and is the first value
    pose_no = cluster_no = None
    if isinstance(this_cluster, list):
        print("cluster_no is a list", this_cluster)
        if len(this_cluster) == 2:
            cluster_no = this_cluster[0]
            pose_no = this_cluster[1]
        elif len(this_cluster) == 1:
            cluster_no = this_cluster[0]
        else:
            print(" >> SOMETHINGS WRONG: cluster_no is a list, but len > 2", this_cluster)
        print(f"cluster_no: {cluster_no}, pose_no: {pose_no}")
    else:
        cluster_no = this_cluster
        print(" >> SOMETHINGS WRONG: cluster_no is not a list", this_cluster)
    print("map_images cluster_no", cluster_no)
    return cluster_no, pose_no

def set_my_counter_dict(this_topic=None, cluster_no=None, pose_no=None, hsv_no=None, start_img_name=None, start_site_image_id=None):
    ### Set counter_dict ###
    print("set_my_counter_dict this_topic, cluster_no, pose_no, hsv_no, start_img_name, start_site_image_id", this_topic, cluster_no, pose_no, hsv_no, start_img_name, start_site_image_id)
    cluster_string = const_prefix(this_topic, cluster_no, pose_no, hsv_cluster=hsv_no)
    print("cluster_string", cluster_string)
    sort.set_counters(io.ROOT,cluster_string, start_img_name,start_site_image_id)

    if VERBOSE: print("set sort.counter_dict:" )
    if VERBOSE: print(sort.counter_dict)

def const_prefix(this_topic, cluster_no, pose_no, hsv_cluster=None):
    # topic_string = ''.join([str(x) for x in this_topic]) if this_topic is not None else "None"
    # if topic is a list, then just use the first value
    if this_topic is not None and isinstance(this_topic, list):
        if VERBOSE: print("this_topic is a list", this_topic)
        this_topic = [this_topic[0]]

    file_prefix = f"c{cluster_no}_p{pose_no}_t{this_topic}" if this_topic is not None else f"c{cluster_no}_p{pose_no}_tNone"
    print("file_prefix before hsv", file_prefix)
    print("hsv_cluster", hsv_cluster)
    if hsv_cluster is not None:
        if isinstance(hsv_cluster, int) or isinstance(hsv_cluster, str):
            hsv_cluster = str(hsv_cluster)
        elif isinstance(hsv_cluster[1], list):
            if VERBOSE: print("hsv_cluster is a list", hsv_cluster)
            hsv_cluster = '_'.join([str(x) for x in hsv_cluster[1]])
        file_prefix = f"{file_prefix}_h{hsv_cluster}"
        print("file_prefix with hsv", file_prefix)
    else:
        print("file_prefix without hsv", file_prefix)
    return file_prefix


###################
#  MY MAIN CODE   #
###################

def main():
    # IDK why I need to declare this a global, but it borks otherwise
    global FUSION_PAIRS
    global FUSION_PAIR_DICT
    def set_multiplier_and_dims(df_segment, cluster_no=None, pose_no=None):
        # if pose_no, overide sort.image_edge_multiplier based on pose_no
        if pose_no is not None and USE_POSE_CROP_DICT:
            pose_type = pose_crop_dict.get(cluster_no, 1)
            sort.image_edge_multiplier = multiplier_list[pose_crop_dict[cluster_no]]
            if VERBOSE: print(f"using pose {cluster_no} getting pose_crop_dict value {pose_type} for image_edge_multiplier", sort.image_edge_multiplier)
        elif image_edge_multiplier is None:
            # if dynamic, set the first one here
            sort.image_edge_multiplier = sort.min_max_body_landmarks_for_crop(df_segment, 0)     
        # reset face_height_output for each round, in case it gets redefined inside loop
        sort.face_height_output = face_height_output
        # use image_edge_multiplier to crop for each
        sort.set_output_dims()

    ###################
    #  MAP THE IMGS   #
    ###################

    # this is the key function, which is called for each cluster
    # or only once if no clusters
    def map_images(resultsjson, this_cluster=None, this_topic=None, this_hsv_cluster=None):
        
        # get cluster and pose from this_cluster
        cluster_no, pose_no = parse_cluster_no(this_cluster)



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
            df[['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized', 'body_landmarks_3D', 'hand_results']] = df['image_id'].apply(io.get_encodings_mongo)
            print("got mongo encodings", df.columns)
            if VERBOSE: print("first row", df.iloc[0])

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
            df['body_landmarks_3D'] = df['body_landmarks_3D'].apply(io.unpickle_array)
            df['body_landmarks_normalized'] = df['body_landmarks_normalized'].apply(io.unpickle_array)
            # if hand_results has any values
            # if not df['hand_results'].isnull().all():
            
            df[['left_hand_landmarks', 'left_hand_world_landmarks', 'left_hand_landmarks_norm', 'right_hand_landmarks', 'right_hand_world_landmarks', 'right_hand_landmarks_norm']] = pd.DataFrame(df['hand_results'].apply(sort.prep_hand_landmarks).tolist(), index=df.index)
            if VERBOSE: print("about to split_landmarks_to_columns_or_list,", df.iloc[0])
            # df = sort.split_landmarks_to_columns_or_list(df, left_col="left_hand_world_landmarks", right_col="right_hand_world_landmarks", structure="list")
            df = sort.split_landmarks_to_columns_or_list(df, first_col="left_hand_landmarks_norm", second_col="right_hand_landmarks_norm", structure="list")
            df = sort.split_landmarks_to_columns_or_list(df, first_col="body_landmarks_3D", second_col=None, structure="list")
            if VERBOSE: print("after split_landmarks_to_columns_or_list, ", df.iloc[0])

            df['bbox'] = df['bbox'].apply(lambda x: io.unstring_json(x))
            if VERBOSE: print("df before bboxing,", df.columns)

            if OBJ_CLS_ID > 0: df["bbox_"+str(OBJ_CLS_ID)] = df["bbox_"+str(OBJ_CLS_ID)].apply(lambda x: io.unstring_json(x))




            columns_to_convert = ['face_x', 'face_y', 'face_z', 'mouth_gap']
            df[columns_to_convert] = df[columns_to_convert].applymap(io.make_float)

            ### SEGMENT THE DATA ###

            # make the segment based on settings
            print("going to segment")
            # need to make sure has HSV here
            df_segment = sort.make_segment(df)
            print("made segment")


            # this is to save files from a segment to the SSD
            if VERBOSE: print("will I save segment? ", SAVE_SEGMENT)
            if SAVE_SEGMENT:
                Base.metadata.create_all(engine)
                print(df_segment.size)
                save_segment_DB(df_segment)
                print("saved segment to segmentTable")
                quit()

            set_multiplier_and_dims(df_segment, cluster_no, pose_no)

            ### Set counter_dict with hsv_no=None ###
            set_my_counter_dict(this_topic, cluster_no, pose_no, None, start_img_name, start_site_image_id)

            ### Get cluster_median encodings for cluster_no ###

            if cluster_no is not None and cluster_no !=0 and (IS_CLUSTER) and not ONLY_ONE:
                # skips cluster 0 for pulling median because it was returning NULL
                # cluster_median = select_cluster_median(cluster_no)
                # image_id = insert_dict['image_id']
                # can I filter this by site_id? would that make it faster or slower? 

                # temp fix
                results = session.query(Clusters).filter(Clusters.cluster_id==cluster_no).first()


                if VERBOSE: print(results)
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
                file_prefix = const_prefix(this_topic, cluster_no, pose_no, this_hsv_cluster)
                # build file prefix for cluster, pose, and topic
                process_linear(start_img_name,df_segment, file_prefix, sort)
        elif df.empty and IS_CLUSTER:
            print('dataframe empty, but IS_CLUSTER so continuing to next cluster_no')

        else: 
            print('dataframe empty, and not IS_CLUSTER so probably bad path or bad SQL')
            sys.exit()


    def save_images_from_csv_folder():

        def recheck_is_dupe_of(session, df):
            '''
            Query the database to recheck if an image is a duplicate of another.
            '''
            if VERBOSE: print("Rechecking is_dupe_of in database for potential duplicates...")
            print("before dropping", df)
            image_id_list = df['image_id'].tolist()
            query = session.query(Encodings.image_id, Encodings.is_dupe_of).filter(Encodings.image_id.in_(image_id_list), Encodings.is_dupe_of.isnot(None)).all()
            image_dict = {image.image_id: image.is_dupe_of for image in query}
            print("image_dict", image_dict)
            if bool(image_dict):
                # if there are any values:
                # TK this needs to be refine when image_dict is not empty
                df['is_dupe_of'] = df['image_id'].map(image_dict)
                # Drop rows where is_dupe_of is None
                df = df.dropna(subset=['is_dupe_of'])
                print("after dropping", df)
            return df

        def get_not_dupes_sql(session):
            # Query the IsNotDupeOf table and return all values as list of tuples
            query = session.query(IsNotDupeOf.image_id_i, IsNotDupeOf.image_id_j).all()
            not_dupe_list = {(row.image_id_i, row.image_id_j): True for row in query}
            return not_dupe_list


        def load_df_sorted_from_csv(csv_file):
            df = pd.read_csv(csv_file)
            print("df head", df.head())
            if df.empty:
                print("dataframe is empty, skipping")
                return

            # Convert face_landmarks from string to mediapipe landmark object
            if "face_landmarks" in df.columns:
                df["face_landmarks"] = df["face_landmarks"].apply(sort.str_to_landmarks)
            df['bbox'] = df['bbox'].apply(lambda x: io.unstring_json(x))
            df['folder'] = df['folder'].apply(lambda x: os.path.join(io.ROOT, os.path.basename(x)))
            # convert 'face_encodings68' from string to list
            df["face_encodings68"] = df["face_encodings68"].apply(lambda x: eval(x) if isinstance(x, str) else x)
            df["body_landmarks_array"] = df["body_landmarks_array"].apply(lambda x: eval(x) if isinstance(x, str) else x)
            df["body_landmarks_normalized"] = df["body_landmarks_normalized"].apply(sort.str_to_landmarks)
            df["body_landmarks_normalized_array"] = df["body_landmarks_normalized"].apply(lambda x: sort.prep_enc(x, structure="list")) # convert mp lms to list
            df["body_landmarks_normalized_visible_array"] = df["body_landmarks_normalized"].apply(lambda x: sort.prep_enc(x, structure="visible")) # convert mp lms to list
            df["wrist_ankle_landmarks_normalized_array"] = df["body_landmarks_normalized"].apply(lambda x: sort.prep_enc(x, structure="wrists_and_ankles")) # convert mp lms to list
            df["face_xyz"] = df[['face_x','face_y', 'face_z']].apply(lambda x: [x[0], x[1], x[2]], axis=1)
            df['bbox_array'] = df['bbox'].apply(lambda x: list(x.values()))            
            df['body_landmarks_normalized_visible'] = df['body_landmarks_normalized'].apply(lambda x: sort.crop_prep(x))
            if VERBOSE: 
                # print("after unpickle_array, first row", df.iloc[0]['body_landmarks_normalized_visible'])
                print("list of columns", df.columns)
                print(df.iloc[0])
            # conver face_x	face_y	face_z	mouth_gap site_image_id to float
            columns_to_convert = ['face_x', 'face_y', 'face_z', 'mouth_gap', 'site_image_id']
            df[columns_to_convert] = df[columns_to_convert].applymap(io.make_float)
            # Process the dataframe as needed
            return df
        
        def find_parts(parts):
            if VERBOSE: print("finding parts in", parts)
            cluster_no = pose_no = segment_count = topic_no = None
            for part in parts:
                if part.startswith("ct"):
                    # Extract segment_count from the part that starts with "ct"
                    # e.g., ct5 -> 5
                    segment_count = part.split("ct")[1]
                elif part.startswith("p"):
                    # Extract pose_no from the part that starts with "p"
                    # e.g., p1 -> 1
                    pose_no = part.split("p")[1]
                elif part.startswith("t"):
                    # Extract topic from the part that starts with "t"
                    # e.g., t3 -> 3
                    topic_no = part.split("t")[1]
                elif part.startswith("c"):
                    # Extract cluster_no from the part that starts with "c"
                    # e.g., c2 -> 2
                    cluster_no = part.split("c")[1]
                elif part.startswith("h"):
                    # Extract hsv_no from the part that starts with "h"
                    # e.g., h2 -> 2
                    hsv_no = part.split("h")[1]
            return segment_count, pose_no, topic_no, cluster_no, hsv_no


        cluster_no = pose_no = segment_count = topic_no = hsv_no = None
        # list the files in the the CSV_FOLDER
        files_in_folder = os.listdir(CSV_FOLDER)
        # sort files alphabetically
        files_in_folder.sort()
        print("files in folder", files_in_folder)
        for csv_file in files_in_folder:
            print("csv_file", csv_file)
            if csv_file.endswith(".csv") and csv_file.startswith("df_sorted_"):
                # Extract cluster_no from filename, e.g., df_sorted_{cluster_no}_ct{segment_count}_p{pose_no}.csv
                parts = csv_file.replace(".csv", "").split("_")
                file_prefix = csv_file.replace("df_sorted_", "").replace(".csv", "")
                print("file_prefix", file_prefix)

                segment_count, pose_no, topic_no, cluster_no, hsv_no = find_parts(parts)
                if len(parts) >= 3:
                    # legacy for Tench's older file structure. delete on next file refresh. TK
                    cluster_no = parts[2]
                print(f"assembling cluster {cluster_no}, topic {topic_no}, pose {pose_no}, hsv {hsv_no} from csv file: {csv_file}")

                ### Set counter_dict (without start stuff which is not needed) ###
                set_my_counter_dict(topic_no, cluster_no, pose_no, hsv_no)
                df_sorted = load_df_sorted_from_csv(os.path.join(CSV_FOLDER, csv_file))


                #can delete this line after, using it to check dupe detection
                #first run use it to see the images to double check, then can comment out for speed
                #linear_test_df(df_sorted,segment_count,cluster_no)
                
                #Dedupe sorting here!
                # df_sorted.to_csv(os.path.join(io.ROOT_DBx, f"df_sorted_{cluster_no}_ct{segment_count}_p{pose_no}.csv"), index=False)
                # df_sorted = df_sorted.head(10)  # Keep only the top entries

                # don't pass in session if IS_TENCH
                if io.IS_TENCH == True: 
                    not_dupe_list = None
                else: 
                    if VERBOSE: print("before recheck_is_dupe_of, df_sorted size", df_sorted.size)
                    df_sorted = recheck_is_dupe_of(session, df_sorted)  # recheck is_dupe_of from DB to catch any new dupes since CSV was made
                    if VERBOSE: print("after recheck_is_dupe_of, df_sorted size", df_sorted.size)
                    not_dupe_list = get_not_dupes_sql(session)  # get list of image_ids that are not dupes
                    if VERBOSE: print("not_dupe_list size", len(not_dupe_list))
                df_sorted = sort.remove_duplicates(io.folder_list, df_sorted, not_dupe_list)

                # wrapping the multiplier in a function that sets dims too
                set_multiplier_and_dims(df_sorted, cluster_no, pose_no)

                if CALIBRATING: continue
                linear_test_df(df_sorted)



    # this is the start of the main part of main()

    if MODE == 1:
        print("MODE 1, assembling from CSV_FOLDER", CSV_FOLDER)
        save_images_from_csv_folder()

    else:
        # mode zero (and 2, though 2 isn't really functional
        first_loop = this_topic = this_cluster = n_cluster_topics = second_cluster_topic = None

        # make a list out of the topic no for looping
        if FUSION_PAIR_DICT is None:
            if IS_ONE_TOPIC:
                if GENERATE_FUSION_PAIRS:
                    print("doing GENERATE_FUSION_PAIRS and storing in FUSION_PAIR_DICT")
                    n_cluster_topics = sort.find_sorted_suitable_indices(TOPIC_NO,MIN_VIDEO_FUSION_COUNT,FUSION_FOLDER)
                    print("fusion_pairs", n_cluster_topics)
                    if N_HSV > 0:
                        print("N_HSV > 0, so making FUSION_PAIR_DICT with keyword: [metacluster, hsv] structure")
                        FUSION_PAIR_DICT = {str(TOPIC_NO[0]): n_cluster_topics}
                    else:
                        # this is how it was before. It doesn't make sense to me b/c key is fusion cluster and all have "no_pairs"
                        FUSION_PAIR_DICT = {str(TOPIC_NO[0]): "no_pairs" for topic in n_cluster_topics}
                    print("FUSION_PAIR_DICT", FUSION_PAIR_DICT)
                else:
                    # doing GENERATE_FUSION_PAIRS where no actual fusion document 
                    print(f"setting SELECT for IS_ONE_TOPIC {TOPIC_NO}")
                    this_topic = TOPIC_NO[0] if isinstance(TOPIC_NO, list) else TOPIC_NO
                    FUSION_PAIR_DICT = {this_topic: "no_pairs"}

        # pulls keys from the FUSION_PAIR_DICT, which are the topics/keywords
        # topic_keyword_list = FUSION_PAIR_DICT.keys()
        topic_keyword_list = []
        for key, value in FUSION_PAIR_DICT.items():
            topic_keyword_list.append(key)
        print("topic_keyword_list type", type(topic_keyword_list))
        print("topic_keyword_list[0]", topic_keyword_list[0])
        for this_topic in topic_keyword_list:

            ###          THE MAIN PART OF MAIN()           ###
            ### QUERY SQL BASED ON CLUSTERS AND MAP_IMAGES ###
    
            print(f"MODE 0 or 2, for this_topic {this_topic} sorting and saving CSV", CSV_FOLDER)
            #creating my objects
            start = time.time()

            # to loop or not to loop that is the cluster
            print(f"doing {CURRENT_MODE}: SORT_TYPE {SORT_TYPE}, IS_HAND_POSE_FUSION {IS_HAND_POSE_FUSION}, GENERATE_FUSION_PAIRS {GENERATE_FUSION_PAIRS}, MIN_VIDEO_FUSION_COUNT {MIN_VIDEO_FUSION_COUNT}, IS_TOPICS {IS_TOPICS}, IS_ONE_TOPIC {IS_ONE_TOPIC}, this_topic {this_topic}, USE_AFFECT_GROUPS {USE_AFFECT_GROUPS}")
            if IS_HAND_POSE_FUSION or IS_CLUSTER or IS_TOPICS: first_loop = True
            if VERBOSE: print("first_loop", first_loop)

            if IS_ONE_CLUSTER:
                print(f"setting SELECT for IS_ONE_CLUSTER {CLUSTER_NO}")
                this_cluster = CLUSTER_NO

            # selectSQL takes a cluster_no and this_topic
            if IS_HAND_POSE_FUSION and ONLY_ONE:
                print("IS_HAND_POSE_FUSION is True")
                # select on both, sort on CLUSTER_NO 
                # this sends pose and gesture in as a list, and an empty topic
                this_cluster = [CLUSTER_NO, HAND_POSE_NO]
            
            if IS_HAND_POSE_FUSION and not ONLY_ONE:
                # print("IS_HAND_POSE_FUSION and not ONLY_ONE")
                if GENERATE_FUSION_PAIRS and FUSION_PAIR_DICT is None:
                    print("FUSION_PAIR_DICT is None so doing GENERATE_FUSION_PAIRS and storing in FUSION_PAIR_DICT")
                    n_cluster_topics = sort.find_sorted_suitable_indices(this_topic,MIN_VIDEO_FUSION_COUNT,FUSION_FOLDER)
                elif GENERATE_FUSION_PAIRS and FUSION_PAIR_DICT is not None:
                    print("FUSION_PAIR_DICT is not None so using existing FUSION_PAIR_DICT")
                    n_cluster_topics = FUSION_PAIR_DICT[this_topic]
                    print("set n_cluster_topics from FUSION_PAIR_DICT", n_cluster_topics)
                elif USE_FUSION_PAIR_DICT:
                    n_cluster_topics = FUSION_PAIR_DICT[this_topic]
                else: 
                    n_cluster_topics = FUSION_PAIRS
                print("IS_HAND_POSE_FUSION and not ONLY_ONE: fusion_pairs", n_cluster_topics)
            elif IS_CLUSTER:
                n_cluster_topics = range(N_CLUSTERS)
                if IS_ONE_TOPIC: second_cluster_topic = this_topic
                print(f"IS_CLUSTER is {IS_CLUSTER} with {n_cluster_topics}, and second_cluster_topic {second_cluster_topic}")
            elif IS_TOPICS:
                if USE_AFFECT_GROUPS: n_cluster_topics = len(AFFECT_GROUPS_LISTS) # redefine for affect groups
                else: n_cluster_topics = range(N_TOPICS)
                if this_cluster is not None: second_cluster_topic = this_cluster
                # if USE_AFFECT_GROUPS: N_CLUSTERS = len(AFFECT_GROUPS_LISTS) # redefine for affect groups
                print(f"IS_TOPICS is {IS_TOPICS} with {n_cluster_topics}")

            if N_HSV > 0:
                n_hsv_clusters = range(1,N_HSV+1)
            else:
                n_hsv_clusters = [0]

            def select_map_images(this_cluster, this_topic, hsv_cluster):
                if VERBOSE: print("select_map_images this_cluster", this_cluster)
                if VERBOSE: print("select_map_images this_topic", this_topic)
                if VERBOSE: print("select_map_images hsv_cluster", hsv_cluster)
                resultsjson = selectSQL(this_cluster, this_topic, hsv_cluster)
                print("got results, count is: ",len(resultsjson))
                if len(resultsjson) < MIN_CYCLE_COUNT:
                    print(f"less than {MIN_CYCLE_COUNT} resultsjson, skipping this {this_cluster} and {this_topic}")
                    return
                else:
                    # folder_name = this_topic[0] if this_topic else this_cluster
                    map_images(resultsjson, this_cluster, this_topic, hsv_cluster)

            def sub_select_clusters_by_hsv(cluster_topic_no, n_cluster_topics):
                # N_HSV
                # make a list where the first value matches cluster_topic_no and the second value is assigned from range(1,N_HSV+1)
                all_potential_hsv_clusters = [[cluster_topic_no, i] for i in range(1, N_HSV + 1)]
                print(f"all_potential_hsv_clusters for cluster_topic_no {cluster_topic_no} is {all_potential_hsv_clusters}")
                this_n_hsv_clusters = []
                all_other_hsv_clusters = []
                for this_cluster in all_potential_hsv_clusters:
                    # print(f"checking cluster_topic_no {this_cluster} against hsv cluster {all_potential_hsv_clusters}")
                    if this_cluster in n_cluster_topics:
                        this_n_hsv_clusters.append(this_cluster)
                    else:
                        all_other_hsv_clusters.append(this_cluster[1])
                if VERBOSE: print(f"all_other_hsv_clusters for cluster_topic_no {cluster_topic_no} is {all_other_hsv_clusters}")
                if VERBOSE: print(f"all_potential_hsv_clusters for cluster_topic_no {cluster_topic_no} is {all_potential_hsv_clusters}")
                all_other_pair_list = [cluster_topic_no, all_other_hsv_clusters]
                this_n_hsv_clusters.append(all_other_pair_list)
                if VERBOSE: print(f"this_n_hsv_clusters for cluster_topic_no {cluster_topic_no} is {this_n_hsv_clusters}")

                # for this_cluster in n_cluster_topics:
                #     # print(f"checking cluster_topic_no {cluster_topic_no} against hsv cluster {this_cluster[0]}")
                #     if this_cluster[0] == cluster_topic_no:
                #         # print(f"appending cluster_topic_no {this_cluster} to this_n_hsv_clusters")
                #         this_n_hsv_clusters.append(this_cluster)
                # # print(f"this_n_hsv_clusters for cluster_topic_no {cluster_topic_no} is {this_n_hsv_clusters}")
                return this_n_hsv_clusters

            def do_first_select_map_images(cluster_topic_no, second_cluster_topic, hsv_cluster=None):
                print(f"cluster_topic_no: {cluster_topic_no}, hsv_cluster: {hsv_cluster}")
                if IS_CLUSTER and cluster_topic_no < START_CLUSTER: return
                if USE_AFFECT_GROUPS: 
                    cluster_topic_no = AFFECT_GROUPS_LISTS[cluster_topic_no] # redefine cluster_no with affect group list
                if IS_TOPICS and not IS_HAND_POSE_FUSION: 
                    print(f"IS_TOPICS and not IS_HAND_POSE_FUSION, passing in second_cluster_topic: {second_cluster_topic}, cluster_topic_no: {cluster_topic_no}, hsv_cluster: {hsv_cluster}")
                    select_map_images(second_cluster_topic, cluster_topic_no, hsv_cluster)
                elif IS_CLUSTER: 
                    print(f"IS_CLUSTER, passing in cluster_topic_no: {cluster_topic_no}, second_cluster_topic: {second_cluster_topic}, hsv_cluster: {hsv_cluster}")
                    select_map_images(cluster_topic_no, second_cluster_topic, hsv_cluster)
                elif IS_HAND_POSE_FUSION: 
                    # I think this is for one topic with pose/gesture fusion
                    print(f"IS_HAND_POSE_FUSION, passing in cluster_topic_no: {cluster_topic_no}, this_topic: {this_topic}, hsv_cluster: {hsv_cluster}")
                    select_map_images(cluster_topic_no,this_topic, hsv_cluster)

            if CURRENT_MODE == 'heft_torso_keywords' and N_HSV > 0:
                print("doing heft_torso_keywords with N_HSV > 0, so Keyword -> MetaClusters -> HSV")
                for cluster_topic_no in range(N_CLUSTERS):
                    print(f"cluster_topic_no range(N_CLUSTERS): {cluster_topic_no}, n_hsv_clusters: {n_hsv_clusters}")
                    if isinstance(n_cluster_topics, list) and len(n_cluster_topics) > 1:
                        print(f"cluster_topic_no: {cluster_topic_no}, this_n_hsv_clusters: {n_cluster_topics}")
                        this_n_hsv_clusters = sub_select_clusters_by_hsv(cluster_topic_no, n_cluster_topics)
                    elif not isinstance(n_cluster_topics, list) and n_cluster_topics is not None:
                        this_n_hsv_clusters = n_cluster_topics
                    else:
                        # when there are no hsv from fusion pairs, make a list containing one entry for all hsv values
                        this_n_hsv_clusters = [list(n_hsv_clusters)]
                    print(f"cluster_topic_no: {cluster_topic_no}, this_n_hsv_clusters: {this_n_hsv_clusters}")

                    if first_loop:
                        print("first loop is ", first_loop)
                        # for cluster_topic_no in this_n_cluster_topics:
                        for hsv_cluster in this_n_hsv_clusters:
                            do_first_select_map_images(cluster_topic_no, second_cluster_topic, hsv_cluster)
                    else:
                        print("doing regular linear")
                        select_map_images(this_cluster, this_topic, hsv_cluster)

            if "3D" in CURRENT_MODE:
                # do 3D bodies processing
                print("doing 3D bodies specific processing")
                if first_loop:
                    print("first loop is ", first_loop)
                    for cluster_topic_no in n_cluster_topics:
                        do_first_select_map_images(cluster_topic_no, second_cluster_topic)
                    else:
                        print("doing regular linear")
                        select_map_images(this_cluster, this_topic)

            if "paris" in CURRENT_MODE:
                print("doing Paris specific processing")
                if first_loop:
                    print("first loop is ", first_loop)
                    for cluster_topic_no in n_cluster_topics:
                        do_first_select_map_images(cluster_topic_no, second_cluster_topic)
                else:
                    print("doing regular linear")
                    select_map_images(this_cluster, this_topic)

            if MODE == 2:
                print("MODE 2, now assembling from CSV_FOLDER", CSV_FOLDER)
                save_images_from_csv_folder()


if __name__ == '__main__':
    main()