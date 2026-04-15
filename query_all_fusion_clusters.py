import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.declarative import declarative_base
import json
import os

import sys
if sys.platform == "darwin": sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
elif sys.platform == "win32": sys.path.insert(1, 'C:/Users/jhash/Documents/GitHub/facemap2/')
from mp_db_io import DataIO
from my_declarative_base import Images, Base, SegmentTable, Encodings, Clusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Float
from constants_make_video import *

io = DataIO()
db = io.db

io.db["name"] = "stock"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#######################################

# USE THIS TO MAKE THE FILE NECESSARY TO DO KEYWORD BASED MAKE VIDEO OUTPUT

# ROOT_FOLDER_PATH = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/heft_keyword_fusion_clusters'
ROOT_FOLDER_PATH = '/Users/michaelmandiberg/Documents/GitHub/takingstock/utilities/data/objectfusion_object_hsv'
MANIFEST_FILE = "fusion_manifest.json"
CONTRACT_VERSION = 1

# First-pass ObjectFusion object-HSV export controls.
# Uses the same temporary focus pattern as make_video for faster debug cycles.
TEMP_FOCUS_CLUSTER_HACK_LIST = [20]
OBJECT_HSV_EXPORT_CLASS_IDS = [27]

HACK_LIST_SKIP_DETECTIONS = [87,90,91,92]
MODE = "ArmsPoses3D" # Topics or Keywords or Detections or ArmsPoses3D
if MODE == "Topics": MODE_ID = "topic_id" 
elif "Detections" in MODE: MODE_ID = "class_id"
elif MODE == "ArmsPoses3D": MODE_ID = "cluster_id"
else: MODE_ID =  "keyword_id"

    # if "body3D" in CLUSTER_TYPE: cluster_count = 512
    # elif "BodyPoses3D" in CLUSTER_TYPE: cluster_count = 768
    # elif "hand_gesture_position" in CLUSTER_TYPE: cluster_count = 128
    # elif "MetaBodyPoses3D" in CLUSTER_TYPE: cluster_count = 64

CLUSTER_COUNT = 768
CLUSTER_DATA = {
    "ArmsPoses3D_MetaHSV": {"sql_template": "sql_query_template_MetaHSV_Body3D", "cluster_table_name": "ImagesArmsPoses3D", "hsv_type": "ClustersMetaHSV", "cluster_count": CLUSTER_COUNT},
    "ObjectFusion_ObjectHSV": {"sql_template": None, "cluster_table_name": "ImagesObjectFusion", "hsv_type": "Detections.meta_cluster_id", "cluster_count": CLUSTER_COUNT},
}

THIS_CLASS_ID = 0 # for object bbox normalization
KEYWORDS = [THIS_CLASS_ID] 
class_token = ID_SEGMENT_DICT.get(THIS_CLASS_ID, None)
if class_token: HELPER_TABLE = f'SegmentHelperObject_{class_token}' 
else: HELPER_TABLE = 'SegmentHelper_T11_Oct20_COCO_Custom_evens_quarters'

CLUSTER_TYPE = "ObjectFusion_ObjectHSV" # key to CLUSTER_DATA dict
# "ArmsPoses3D_MetaHSV" or "BodyPoses3D_MetaHSV" or "MetaBodyPoses3D" or "BodyPoses3D_HSV" or "body3D" or "hand_gesture_position" - determines whether it checks hand poses or body3D


# Create engine and session
engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

Session = sessionmaker(bind=engine)
session = Session()

# first 87
# KEYWORDS = [22137,184,502,135,22411,1991,11801,273,220,2150,22269,22233,5271,22040,133,22324,23100,827,22499,278,1070,13057,22412,5728,404,23084,22333,2472,22665,22042,420,553,1227,22228,665,23403,671,272,437,293,2514,22222,22961,27381,2467,5279,4265,1127,407,790,3856,133680,1204,703,1224,729,737,6286,2151,807,1585,699,1644,2756,786,698,730,133819,22692,2188,1223,1807,10765,24705,22247,133705,5310]

# second 100
# KEYWORDS = [232, 22251, 1575, 758, 22600, 424, 410, 1919, 25287, 5516, 2567, 3961, 9940, 22861, 25155, 919, 115, 8911, 818, 1263, 1222, 22617, 6970, 22139, 486, 5115, 22298, 13539, 697, 23512, 24327, 23825, 1073, 22217, 22910, 133822, 22105, 1421, 212, 4589, 133768, 4572, 805, 227, 133724, 13, 295, 24552, 13300, 133816, 5953, 2747, 24041, 1217, 133685, 24472, 514, 292, 22336, 761, 9028, 4361, 433, 223, 696, 13534, 327, 7266, 22851, 11605, 1121, 12472, 25083, 18066, 297, 830, 24399, 3977, 732, 736, 4667, 296, 753, 22628, 22968, 133834, 11203, 8962, 3706, 215, 12572, 5342, 2599, 23853, 5824, 2421, 1772, 6045, 789, 4714
# KEYWORDS = [22411,220,22269,827,1070,22412,553,807,1644,5310] # helper segment
# KEYWORDS = [21463,4222,13130,23084,79920,8874,736,8136] # helper segment
# KEYWORDS = [4222,23375,13130,21463,184,23726,8874,8136,133749,26241,22814,133787,4587,133627]

# SQL query template
sql_query_template = """
SELECT 
    ihp.cluster_id AS ihp_cluster,
    SUM(CASE WHEN ihg.cluster_id = 0 THEN 1 ELSE 0 END) AS ihg_0,
    SUM(CASE WHEN ihg.cluster_id = 1 THEN 1 ELSE 0 END) AS ihg_1,
    SUM(CASE WHEN ihg.cluster_id = 2 THEN 1 ELSE 0 END) AS ihg_2,
    SUM(CASE WHEN ihg.cluster_id = 3 THEN 1 ELSE 0 END) AS ihg_3,
    SUM(CASE WHEN ihg.cluster_id = 4 THEN 1 ELSE 0 END) AS ihg_4,
    SUM(CASE WHEN ihg.cluster_id = 5 THEN 1 ELSE 0 END) AS ihg_5,
    SUM(CASE WHEN ihg.cluster_id = 6 THEN 1 ELSE 0 END) AS ihg_6,
    SUM(CASE WHEN ihg.cluster_id = 7 THEN 1 ELSE 0 END) AS ihg_7,
    SUM(CASE WHEN ihg.cluster_id = 8 THEN 1 ELSE 0 END) AS ihg_8,
    SUM(CASE WHEN ihg.cluster_id = 9 THEN 1 ELSE 0 END) AS ihg_9,
    SUM(CASE WHEN ihg.cluster_id = 10 THEN 1 ELSE 0 END) AS ihg_10,
    SUM(CASE WHEN ihg.cluster_id = 11 THEN 1 ELSE 0 END) AS ihg_11,
    SUM(CASE WHEN ihg.cluster_id = 12 THEN 1 ELSE 0 END) AS ihg_12,
    SUM(CASE WHEN ihg.cluster_id = 13 THEN 1 ELSE 0 END) AS ihg_13,
    SUM(CASE WHEN ihg.cluster_id = 14 THEN 1 ELSE 0 END) AS ihg_14,
    SUM(CASE WHEN ihg.cluster_id = 15 THEN 1 ELSE 0 END) AS ihg_15,
    SUM(CASE WHEN ihg.cluster_id = 16 THEN 1 ELSE 0 END) AS ihg_16,
    SUM(CASE WHEN ihg.cluster_id = 17 THEN 1 ELSE 0 END) AS ihg_17,
    SUM(CASE WHEN ihg.cluster_id = 18 THEN 1 ELSE 0 END) AS ihg_18,
    SUM(CASE WHEN ihg.cluster_id = 19 THEN 1 ELSE 0 END) AS ihg_19,
    SUM(CASE WHEN ihg.cluster_id = 20 THEN 1 ELSE 0 END) AS ihg_20,
    SUM(CASE WHEN ihg.cluster_id = 21 THEN 1 ELSE 0 END) AS ihg_21,
    SUM(CASE WHEN ihg.cluster_id = 22 THEN 1 ELSE 0 END) AS ihg_22,
    SUM(CASE WHEN ihg.cluster_id = 23 THEN 1 ELSE 0 END) AS ihg_23,
    SUM(CASE WHEN ihg.cluster_id = 24 THEN 1 ELSE 0 END) AS ihg_24,
    SUM(CASE WHEN ihg.cluster_id = 25 THEN 1 ELSE 0 END) AS ihg_25,
    SUM(CASE WHEN ihg.cluster_id = 26 THEN 1 ELSE 0 END) AS ihg_26,
    SUM(CASE WHEN ihg.cluster_id = 27 THEN 1 ELSE 0 END) AS ihg_27,
    SUM(CASE WHEN ihg.cluster_id = 28 THEN 1 ELSE 0 END) AS ihg_28,
    SUM(CASE WHEN ihg.cluster_id = 29 THEN 1 ELSE 0 END) AS ihg_29,
    SUM(CASE WHEN ihg.cluster_id = 30 THEN 1 ELSE 0 END) AS ihg_30,
    SUM(CASE WHEN ihg.cluster_id = 31 THEN 1 ELSE 0 END) AS ihg_31,
    SUM(CASE WHEN ihg.cluster_id = 32 THEN 1 ELSE 0 END) AS ihg_32,
    SUM(CASE WHEN ihg.cluster_id = 33 THEN 1 ELSE 0 END) AS ihg_33,
    SUM(CASE WHEN ihg.cluster_id = 34 THEN 1 ELSE 0 END) AS ihg_34,
    SUM(CASE WHEN ihg.cluster_id = 35 THEN 1 ELSE 0 END) AS ihg_35,
    SUM(CASE WHEN ihg.cluster_id = 36 THEN 1 ELSE 0 END) AS ihg_36,
    SUM(CASE WHEN ihg.cluster_id = 37 THEN 1 ELSE 0 END) AS ihg_37,
    SUM(CASE WHEN ihg.cluster_id = 38 THEN 1 ELSE 0 END) AS ihg_38,
    SUM(CASE WHEN ihg.cluster_id = 39 THEN 1 ELSE 0 END) AS ihg_39,
    SUM(CASE WHEN ihg.cluster_id = 40 THEN 1 ELSE 0 END) AS ihg_40,
    SUM(CASE WHEN ihg.cluster_id = 41 THEN 1 ELSE 0 END) AS ihg_41,
    SUM(CASE WHEN ihg.cluster_id = 42 THEN 1 ELSE 0 END) AS ihg_42,
    SUM(CASE WHEN ihg.cluster_id = 43 THEN 1 ELSE 0 END) AS ihg_43,
    SUM(CASE WHEN ihg.cluster_id = 44 THEN 1 ELSE 0 END) AS ihg_44,
    SUM(CASE WHEN ihg.cluster_id = 45 THEN 1 ELSE 0 END) AS ihg_45,
    SUM(CASE WHEN ihg.cluster_id = 46 THEN 1 ELSE 0 END) AS ihg_46,
    SUM(CASE WHEN ihg.cluster_id = 47 THEN 1 ELSE 0 END) AS ihg_47,
    SUM(CASE WHEN ihg.cluster_id = 48 THEN 1 ELSE 0 END) AS ihg_48,
    SUM(CASE WHEN ihg.cluster_id = 49 THEN 1 ELSE 0 END) AS ihg_49,
    SUM(CASE WHEN ihg.cluster_id = 50 THEN 1 ELSE 0 END) AS ihg_50,
    SUM(CASE WHEN ihg.cluster_id = 51 THEN 1 ELSE 0 END) AS ihg_51,
    SUM(CASE WHEN ihg.cluster_id = 52 THEN 1 ELSE 0 END) AS ihg_52,
    SUM(CASE WHEN ihg.cluster_id = 53 THEN 1 ELSE 0 END) AS ihg_53,
    SUM(CASE WHEN ihg.cluster_id = 54 THEN 1 ELSE 0 END) AS ihg_54,
    SUM(CASE WHEN ihg.cluster_id = 55 THEN 1 ELSE 0 END) AS ihg_55,
    SUM(CASE WHEN ihg.cluster_id = 56 THEN 1 ELSE 0 END) AS ihg_56, 
    SUM(CASE WHEN ihg.cluster_id =  57 THEN 1 ELSE 0 END) AS ihg_57 ,
    SUM(CASE WHEN ihg.cluster_id =  58 THEN 1 ELSE 0 END) AS ihg_58 ,
    SUM(CASE WHEN ihg.cluster_id =  59 THEN 1 ELSE 0 END) AS ihg_59 ,
    SUM(CASE WHEN ihg.cluster_id =  60 THEN 1 ELSE 0 END) AS ihg_60 ,
    SUM(CASE WHEN ihg.cluster_id =  61 THEN 1 ELSE 0 END) AS ihg_61 ,
    SUM(CASE WHEN ihg.cluster_id =  62 THEN 1 ELSE 0 END) AS ihg_62 ,
    SUM(CASE WHEN ihg.cluster_id =  63 THEN 1 ELSE 0 END) AS ihg_63 ,
    SUM(CASE WHEN ihg.cluster_id =  64 THEN 1 ELSE 0 END) AS ihg_64 ,
    SUM(CASE WHEN ihg.cluster_id =  65 THEN 1 ELSE 0 END) AS ihg_65 ,
    SUM(CASE WHEN ihg.cluster_id =  66 THEN 1 ELSE 0 END) AS ihg_66 ,
    SUM(CASE WHEN ihg.cluster_id =  67 THEN 1 ELSE 0 END) AS ihg_67 ,
    SUM(CASE WHEN ihg.cluster_id =  68 THEN 1 ELSE 0 END) AS ihg_68 ,
    SUM(CASE WHEN ihg.cluster_id =  69 THEN 1 ELSE 0 END) AS ihg_69 ,
    SUM(CASE WHEN ihg.cluster_id =  70 THEN 1 ELSE 0 END) AS ihg_70 ,
    SUM(CASE WHEN ihg.cluster_id =  71 THEN 1 ELSE 0 END) AS ihg_71 ,
    SUM(CASE WHEN ihg.cluster_id =  72 THEN 1 ELSE 0 END) AS ihg_72 ,
    SUM(CASE WHEN ihg.cluster_id =  73 THEN 1 ELSE 0 END) AS ihg_73 ,
    SUM(CASE WHEN ihg.cluster_id =  74 THEN 1 ELSE 0 END) AS ihg_74 ,
    SUM(CASE WHEN ihg.cluster_id =  75 THEN 1 ELSE 0 END) AS ihg_75 ,
    SUM(CASE WHEN ihg.cluster_id =  76 THEN 1 ELSE 0 END) AS ihg_76 ,
    SUM(CASE WHEN ihg.cluster_id =  77 THEN 1 ELSE 0 END) AS ihg_77 ,
    SUM(CASE WHEN ihg.cluster_id =  78 THEN 1 ELSE 0 END) AS ihg_78 ,
    SUM(CASE WHEN ihg.cluster_id =  79 THEN 1 ELSE 0 END) AS ihg_79 ,
    SUM(CASE WHEN ihg.cluster_id =  80 THEN 1 ELSE 0 END) AS ihg_80 ,
    SUM(CASE WHEN ihg.cluster_id =  81 THEN 1 ELSE 0 END) AS ihg_81 ,
    SUM(CASE WHEN ihg.cluster_id =  82 THEN 1 ELSE 0 END) AS ihg_82 ,
    SUM(CASE WHEN ihg.cluster_id =  83 THEN 1 ELSE 0 END) AS ihg_83 ,
    SUM(CASE WHEN ihg.cluster_id =  84 THEN 1 ELSE 0 END) AS ihg_84 ,
    SUM(CASE WHEN ihg.cluster_id =  85 THEN 1 ELSE 0 END) AS ihg_85 ,
    SUM(CASE WHEN ihg.cluster_id =  86 THEN 1 ELSE 0 END) AS ihg_86 ,
    SUM(CASE WHEN ihg.cluster_id =  87 THEN 1 ELSE 0 END) AS ihg_87 ,
    SUM(CASE WHEN ihg.cluster_id =  88 THEN 1 ELSE 0 END) AS ihg_88 ,
    SUM(CASE WHEN ihg.cluster_id =  89 THEN 1 ELSE 0 END) AS ihg_89 ,
    SUM(CASE WHEN ihg.cluster_id =  90 THEN 1 ELSE 0 END) AS ihg_90 ,
    SUM(CASE WHEN ihg.cluster_id =  91 THEN 1 ELSE 0 END) AS ihg_91 ,
    SUM(CASE WHEN ihg.cluster_id =  92 THEN 1 ELSE 0 END) AS ihg_92 ,
    SUM(CASE WHEN ihg.cluster_id =  93 THEN 1 ELSE 0 END) AS ihg_93 ,
    SUM(CASE WHEN ihg.cluster_id =  94 THEN 1 ELSE 0 END) AS ihg_94 ,
    SUM(CASE WHEN ihg.cluster_id =  95 THEN 1 ELSE 0 END) AS ihg_95 ,
    SUM(CASE WHEN ihg.cluster_id =  96 THEN 1 ELSE 0 END) AS ihg_96 ,
    SUM(CASE WHEN ihg.cluster_id =  97 THEN 1 ELSE 0 END) AS ihg_97 ,
    SUM(CASE WHEN ihg.cluster_id =  98 THEN 1 ELSE 0 END) AS ihg_98 ,
    SUM(CASE WHEN ihg.cluster_id =  99 THEN 1 ELSE 0 END) AS ihg_99 ,
    SUM(CASE WHEN ihg.cluster_id =  100 THEN 1 ELSE 0 END) AS ihg_100 ,
    SUM(CASE WHEN ihg.cluster_id =  101 THEN 1 ELSE 0 END) AS ihg_101 ,
    SUM(CASE WHEN ihg.cluster_id =  102 THEN 1 ELSE 0 END) AS ihg_102 ,
    SUM(CASE WHEN ihg.cluster_id =  103 THEN 1 ELSE 0 END) AS ihg_103 ,
    SUM(CASE WHEN ihg.cluster_id =  104 THEN 1 ELSE 0 END) AS ihg_104 ,
    SUM(CASE WHEN ihg.cluster_id =  105 THEN 1 ELSE 0 END) AS ihg_105 ,
    SUM(CASE WHEN ihg.cluster_id =  106 THEN 1 ELSE 0 END) AS ihg_106 ,
    SUM(CASE WHEN ihg.cluster_id =  107 THEN 1 ELSE 0 END) AS ihg_107 ,
    SUM(CASE WHEN ihg.cluster_id =  108 THEN 1 ELSE 0 END) AS ihg_108 ,
    SUM(CASE WHEN ihg.cluster_id =  109 THEN 1 ELSE 0 END) AS ihg_109 ,
    SUM(CASE WHEN ihg.cluster_id =  110 THEN 1 ELSE 0 END) AS ihg_110 ,
    SUM(CASE WHEN ihg.cluster_id =  111 THEN 1 ELSE 0 END) AS ihg_111 ,
    SUM(CASE WHEN ihg.cluster_id =  112 THEN 1 ELSE 0 END) AS ihg_112 ,
    SUM(CASE WHEN ihg.cluster_id =  113 THEN 1 ELSE 0 END) AS ihg_113 ,
    SUM(CASE WHEN ihg.cluster_id =  114 THEN 1 ELSE 0 END) AS ihg_114 ,
    SUM(CASE WHEN ihg.cluster_id =  115 THEN 1 ELSE 0 END) AS ihg_115 ,
    SUM(CASE WHEN ihg.cluster_id =  116 THEN 1 ELSE 0 END) AS ihg_116 ,
    SUM(CASE WHEN ihg.cluster_id =  117 THEN 1 ELSE 0 END) AS ihg_117 ,
    SUM(CASE WHEN ihg.cluster_id =  118 THEN 1 ELSE 0 END) AS ihg_118 ,
    SUM(CASE WHEN ihg.cluster_id =  119 THEN 1 ELSE 0 END) AS ihg_119 ,
    SUM(CASE WHEN ihg.cluster_id =  120 THEN 1 ELSE 0 END) AS ihg_120 ,
    SUM(CASE WHEN ihg.cluster_id =  121 THEN 1 ELSE 0 END) AS ihg_121 ,
    SUM(CASE WHEN ihg.cluster_id =  122 THEN 1 ELSE 0 END) AS ihg_122 ,
    SUM(CASE WHEN ihg.cluster_id =  123 THEN 1 ELSE 0 END) AS ihg_123 ,
    SUM(CASE WHEN ihg.cluster_id =  124 THEN 1 ELSE 0 END) AS ihg_124 ,
    SUM(CASE WHEN ihg.cluster_id =  125 THEN 1 ELSE 0 END) AS ihg_125 ,
    SUM(CASE WHEN ihg.cluster_id =  126 THEN 1 ELSE 0 END) AS ihg_126 ,
    SUM(CASE WHEN ihg.cluster_id =  127 THEN 1 ELSE 0 END) AS ihg_127 
FROM 
    SegmentOct20 so
JOIN 
    ImagesHandsPositions ihp ON ihp.image_id = so.image_id
JOIN 
    ImagesHandsGestures ihg ON ihg.image_id = so.image_id
JOIN 
    Images{MODE} it ON it.image_id = so.image_id
WHERE it.{MODE_ID} = {THIS_MODE_ID}
GROUP BY
    ihp.cluster_id
ORDER BY 
    ihp_cluster;
"""


sql_query_template_body3D = """
SELECT 
    ihp.cluster_id AS ihp_cluster,
	COUNT(so.image_id)
FROM 
    SegmentBig_isface so
JOIN 
    SegmentHelper_sept2025_heft_keywords sh ON sh.image_id = so.image_id
JOIN 
    ImagesBodyPoses3D ihp ON ihp.image_id = so.image_id
JOIN 
    Images{MODE} it ON it.image_id = so.image_id
WHERE it.{MODE_ID} = {THIS_MODE_ID}
GROUP BY
    ihp.cluster_id
ORDER BY 
    ihp_cluster;

"""

#  NEED TO MAKE THIS WORK WITH NONAUTO INCREMENTING change to table
    # SUM(CASE WHEN ihsv.cluster_id = 0 THEN 1 ELSE 0 END) AS hsv_0,

sql_query_template_HSV_Body3D = """
SELECT 
    ibp.cluster_id AS ihp_cluster,
    SUM(CASE WHEN ihsv.cluster_id = 1 THEN 1 ELSE 0 END) AS hsv_1,
    SUM(CASE WHEN ihsv.cluster_id = 2 THEN 1 ELSE 0 END) AS hsv_2,
    SUM(CASE WHEN ihsv.cluster_id = 3 THEN 1 ELSE 0 END) AS hsv_3,
    SUM(CASE WHEN ihsv.cluster_id = 4 THEN 1 ELSE 0 END) AS hsv_4,
    SUM(CASE WHEN ihsv.cluster_id = 5 THEN 1 ELSE 0 END) AS hsv_5,
    SUM(CASE WHEN ihsv.cluster_id = 6 THEN 1 ELSE 0 END) AS hsv_6,
    SUM(CASE WHEN ihsv.cluster_id = 7 THEN 1 ELSE 0 END) AS hsv_7,
    SUM(CASE WHEN ihsv.cluster_id = 8 THEN 1 ELSE 0 END) AS hsv_8,
    SUM(CASE WHEN ihsv.cluster_id = 9 THEN 1 ELSE 0 END) AS hsv_9,
    SUM(CASE WHEN ihsv.cluster_id = 10 THEN 1 ELSE 0 END) AS hsv_10,
    SUM(CASE WHEN ihsv.cluster_id = 11 THEN 1 ELSE 0 END) AS hsv_11,
    SUM(CASE WHEN ihsv.cluster_id = 12 THEN 1 ELSE 0 END) AS hsv_12,
    SUM(CASE WHEN ihsv.cluster_id = 13 THEN 1 ELSE 0 END) AS hsv_13,
    SUM(CASE WHEN ihsv.cluster_id = 14 THEN 1 ELSE 0 END) AS hsv_14,
    SUM(CASE WHEN ihsv.cluster_id = 15 THEN 1 ELSE 0 END) AS hsv_15,
    SUM(CASE WHEN ihsv.cluster_id = 16 THEN 1 ELSE 0 END) AS hsv_16,
    SUM(CASE WHEN ihsv.cluster_id = 525 THEN 1 ELSE 0 END) AS hsv_525
  
FROM SegmentBig_isface so
JOIN SegmentHelper_sept2025_heft_keywords sh ON sh.image_id = so.image_id
JOIN ImagesBodyPoses3D ibp ON ibp.image_id = so.image_id
JOIN ImagesHSV ihsv ON ihsv.image_id = so.image_id
JOIN Images{MODE} it ON it.image_id = so.image_id
WHERE it.{MODE_ID} = {THIS_MODE_ID}
GROUP BY
    ibp.cluster_id
ORDER BY 
    ibp.cluster_id;
"""


sql_query_template_HSV_Arms_or_Body3D = """
SELECT 
    ibp.cluster_id AS ihp_cluster,
    SUM(CASE WHEN ihsv.cluster_id = 1 THEN 1 ELSE 0 END) AS hsv_1,
    SUM(CASE WHEN ihsv.cluster_id = 2 THEN 1 ELSE 0 END) AS hsv_2,
    SUM(CASE WHEN ihsv.cluster_id = 3 THEN 1 ELSE 0 END) AS hsv_3,
    SUM(CASE WHEN ihsv.cluster_id = 4 THEN 1 ELSE 0 END) AS hsv_4,
    SUM(CASE WHEN ihsv.cluster_id = 5 THEN 1 ELSE 0 END) AS hsv_5,
    SUM(CASE WHEN ihsv.cluster_id = 6 THEN 1 ELSE 0 END) AS hsv_6,
    SUM(CASE WHEN ihsv.cluster_id = 7 THEN 1 ELSE 0 END) AS hsv_7,
    SUM(CASE WHEN ihsv.cluster_id = 8 THEN 1 ELSE 0 END) AS hsv_8,
    SUM(CASE WHEN ihsv.cluster_id = 9 THEN 1 ELSE 0 END) AS hsv_9,
    SUM(CASE WHEN ihsv.cluster_id = 10 THEN 1 ELSE 0 END) AS hsv_10,
    SUM(CASE WHEN ihsv.cluster_id = 11 THEN 1 ELSE 0 END) AS hsv_11,
    SUM(CASE WHEN ihsv.cluster_id = 12 THEN 1 ELSE 0 END) AS hsv_12,
    SUM(CASE WHEN ihsv.cluster_id = 13 THEN 1 ELSE 0 END) AS hsv_13,
    SUM(CASE WHEN ihsv.cluster_id = 14 THEN 1 ELSE 0 END) AS hsv_14,
    SUM(CASE WHEN ihsv.cluster_id = 15 THEN 1 ELSE 0 END) AS hsv_15,
    SUM(CASE WHEN ihsv.cluster_id = 16 THEN 1 ELSE 0 END) AS hsv_16,
    SUM(CASE WHEN ihsv.cluster_id = 525 THEN 1 ELSE 0 END) AS hsv_525
  
FROM SegmentBig_isface so
JOIN SegmentHelper_sept2025_heft_keywords sh ON sh.image_id = so.image_id
JOIN ImagesBodyPoses3D ibp ON ibp.image_id = so.image_id
JOIN ImagesHSV ihsv ON ihsv.image_id = so.image_id
JOIN Images{MODE} it ON it.image_id = so.image_id
WHERE it.{MODE_ID} = {THIS_MODE_ID}
GROUP BY
    ibp.cluster_id
ORDER BY 
    ibp.cluster_id;
"""


sql_query_template_HSV_MetaBody3D = """
SELECT 
    ihp.meta_cluster_id AS ihp_cluster,
    SUM(CASE WHEN ihsv.cluster_id = 0 THEN 1 ELSE 0 END) AS hsv_0,
    SUM(CASE WHEN ihsv.cluster_id = 1 THEN 1 ELSE 0 END) AS hsv_1,
    SUM(CASE WHEN ihsv.cluster_id = 2 THEN 1 ELSE 0 END) AS hsv_2,
    SUM(CASE WHEN ihsv.cluster_id = 3 THEN 1 ELSE 0 END) AS hsv_3,
    SUM(CASE WHEN ihsv.cluster_id = 4 THEN 1 ELSE 0 END) AS hsv_4,
    SUM(CASE WHEN ihsv.cluster_id = 5 THEN 1 ELSE 0 END) AS hsv_5,
    SUM(CASE WHEN ihsv.cluster_id = 6 THEN 1 ELSE 0 END) AS hsv_6,
    SUM(CASE WHEN ihsv.cluster_id = 7 THEN 1 ELSE 0 END) AS hsv_7,
    SUM(CASE WHEN ihsv.cluster_id = 8 THEN 1 ELSE 0 END) AS hsv_8,
    SUM(CASE WHEN ihsv.cluster_id = 9 THEN 1 ELSE 0 END) AS hsv_9,
    SUM(CASE WHEN ihsv.cluster_id = 10 THEN 1 ELSE 0 END) AS hsv_10,
    SUM(CASE WHEN ihsv.cluster_id = 11 THEN 1 ELSE 0 END) AS hsv_11,
    SUM(CASE WHEN ihsv.cluster_id = 12 THEN 1 ELSE 0 END) AS hsv_12,
    SUM(CASE WHEN ihsv.cluster_id = 13 THEN 1 ELSE 0 END) AS hsv_13,
    SUM(CASE WHEN ihsv.cluster_id = 14 THEN 1 ELSE 0 END) AS hsv_14,
    SUM(CASE WHEN ihsv.cluster_id = 15 THEN 1 ELSE 0 END) AS hsv_15,
    SUM(CASE WHEN ihsv.cluster_id = 16 THEN 1 ELSE 0 END) AS hsv_16,
    SUM(CASE WHEN ihsv.cluster_id = 17 THEN 1 ELSE 0 END) AS hsv_17,
    SUM(CASE WHEN ihsv.cluster_id = 18 THEN 1 ELSE 0 END) AS hsv_18,
    SUM(CASE WHEN ihsv.cluster_id = 19 THEN 1 ELSE 0 END) AS hsv_19,    
    SUM(CASE WHEN ihsv.cluster_id = 20 THEN 1 ELSE 0 END) AS hsv_20,
    SUM(CASE WHEN ihsv.cluster_id = 21 THEN 1 ELSE 0 END) AS hsv_21,
    SUM(CASE WHEN ihsv.cluster_id = 22 THEN 1 ELSE 0 END) AS hsv_22,
    SUM(CASE WHEN ihsv.cluster_id = 23 THEN 1 ELSE 0 END) AS hsv_23,
    SUM(CASE WHEN ihsv.cluster_id = 24 THEN 1 ELSE 0 END) AS hsv_24,
    SUM(CASE WHEN ihsv.cluster_id = 25 THEN 1 ELSE 0 END) AS hsv_25,
    SUM(CASE WHEN ihsv.cluster_id = 26 THEN 1 ELSE 0 END) AS hsv_26,
    SUM(CASE WHEN ihsv.cluster_id = 27 THEN 1 ELSE 0 END) AS hsv_27,
    SUM(CASE WHEN ihsv.cluster_id = 28 THEN 1 ELSE 0 END) AS hsv_28,
    SUM(CASE WHEN ihsv.cluster_id = 29 THEN 1 ELSE 0 END) AS hsv_29,
    SUM(CASE WHEN ihsv.cluster_id = 30 THEN 1 ELSE 0 END) AS hsv_30,
    SUM(CASE WHEN ihsv.cluster_id = 31 THEN 1 ELSE 0 END) AS hsv_31,
    SUM(CASE WHEN ihsv.cluster_id = 32 THEN 1 ELSE 0 END) AS hsv_32,
    SUM(CASE WHEN ihsv.cluster_id = 33 THEN 1 ELSE 0 END) AS hsv_33,
    SUM(CASE WHEN ihsv.cluster_id = 34 THEN 1 ELSE 0 END) AS hsv_34,
    SUM(CASE WHEN ihsv.cluster_id = 35 THEN 1 ELSE 0 END) AS hsv_35,
    SUM(CASE WHEN ihsv.cluster_id = 36 THEN 1 ELSE 0 END) AS hsv_36,
    SUM(CASE WHEN ihsv.cluster_id = 37 THEN 1 ELSE 0 END) AS hsv_37,
    SUM(CASE WHEN ihsv.cluster_id = 38 THEN 1 ELSE 0 END) AS hsv_38,
    SUM(CASE WHEN ihsv.cluster_id = 39 THEN 1 ELSE 0 END) AS hsv_39,
    SUM(CASE WHEN ihsv.cluster_id = 40 THEN 1 ELSE 0 END) AS hsv_40,
    SUM(CASE WHEN ihsv.cluster_id = 41 THEN 1 ELSE 0 END) AS hsv_41,
    SUM(CASE WHEN ihsv.cluster_id = 42 THEN 1 ELSE 0 END) AS hsv_42,
    SUM(CASE WHEN ihsv.cluster_id = 43 THEN 1 ELSE 0 END) AS hsv_43,
    SUM(CASE WHEN ihsv.cluster_id = 44 THEN 1 ELSE 0 END) AS hsv_44,
    SUM(CASE WHEN ihsv.cluster_id = 45 THEN 1 ELSE 0 END) AS hsv_45,
    SUM(CASE WHEN ihsv.cluster_id = 46 THEN 1 ELSE 0 END) AS hsv_46,
    SUM(CASE WHEN ihsv.cluster_id = 47 THEN 1 ELSE 0 END) AS hsv_47,
    SUM(CASE WHEN ihsv.cluster_id = 48 THEN 1 ELSE 0 END) AS hsv_48,
    SUM(CASE WHEN ihsv.cluster_id = 49 THEN 1 ELSE 0 END) AS hsv_49,
    SUM(CASE WHEN ihsv.cluster_id = 50 THEN 1 ELSE 0 END) AS hsv_50,
    SUM(CASE WHEN ihsv.cluster_id = 51 THEN 1 ELSE 0 END) AS hsv_51,
    SUM(CASE WHEN ihsv.cluster_id = 52 THEN 1 ELSE 0 END) AS hsv_52,
    SUM(CASE WHEN ihsv.cluster_id = 53 THEN 1 ELSE 0 END) AS hsv_53,
    SUM(CASE WHEN ihsv.cluster_id = 54 THEN 1 ELSE 0 END) AS hsv_54,
    SUM(CASE WHEN ihsv.cluster_id = 55 THEN 1 ELSE 0 END) AS hsv_55,
    SUM(CASE WHEN ihsv.cluster_id = 56 THEN 1 ELSE 0 END) AS hsv_56,
    SUM(CASE WHEN ihsv.cluster_id = 57 THEN 1 ELSE 0 END) AS hsv_57,
    SUM(CASE WHEN ihsv.cluster_id = 58 THEN 1 ELSE 0 END) AS hsv_58,
    SUM(CASE WHEN ihsv.cluster_id = 59 THEN 1 ELSE 0 END) AS hsv_59,
    SUM(CASE WHEN ihsv.cluster_id = 60 THEN 1 ELSE 0 END) AS hsv_60,
    SUM(CASE WHEN ihsv.cluster_id = 61 THEN 1 ELSE 0 END) AS hsv_61,
    SUM(CASE WHEN ihsv.cluster_id = 62 THEN 1 ELSE 0 END) AS hsv_62,
    SUM(CASE WHEN ihsv.cluster_id = 63 THEN 1 ELSE 0 END) AS hsv_63,
    SUM(CASE WHEN ihsv.cluster_id = 64 THEN 1 ELSE 0 END) AS hsv_64,
    SUM(CASE WHEN ihsv.cluster_id = 65 THEN 1 ELSE 0 END) AS hsv_65,
    SUM(CASE WHEN ihsv.cluster_id = 66 THEN 1 ELSE 0 END) AS hsv_66,
    SUM(CASE WHEN ihsv.cluster_id = 67 THEN 1 ELSE 0 END) AS hsv_67,
    SUM(CASE WHEN ihsv.cluster_id = 68 THEN 1 ELSE 0 END) AS hsv_68,
    SUM(CASE WHEN ihsv.cluster_id = 69 THEN 1 ELSE 0 END) AS hsv_69,
    SUM(CASE WHEN ihsv.cluster_id = 70 THEN 1 ELSE 0 END) AS hsv_70,
    SUM(CASE WHEN ihsv.cluster_id = 71 THEN 1 ELSE 0 END) AS hsv_71,
    SUM(CASE WHEN ihsv.cluster_id = 72 THEN 1 ELSE 0 END) AS hsv_72,
    SUM(CASE WHEN ihsv.cluster_id = 73 THEN 1 ELSE 0 END) AS hsv_73,
    SUM(CASE WHEN ihsv.cluster_id = 74 THEN 1 ELSE 0 END) AS hsv_74,
    SUM(CASE WHEN ihsv.cluster_id = 75 THEN 1 ELSE 0 END) AS hsv_75,
    SUM(CASE WHEN ihsv.cluster_id = 76 THEN 1 ELSE 0 END) AS hsv_76,
    SUM(CASE WHEN ihsv.cluster_id = 77 THEN 1 ELSE 0 END) AS hsv_77,
    SUM(CASE WHEN ihsv.cluster_id = 78 THEN 1 ELSE 0 END) AS hsv_78,
    SUM(CASE WHEN ihsv.cluster_id = 79 THEN 1 ELSE 0 END) AS hsv_79,
    SUM(CASE WHEN ihsv.cluster_id = 80 THEN 1 ELSE 0 END) AS hsv_80,
    SUM(CASE WHEN ihsv.cluster_id = 81 THEN 1 ELSE 0 END) AS hsv_81,
    SUM(CASE WHEN ihsv.cluster_id = 82 THEN 1 ELSE 0 END) AS hsv_82,
    SUM(CASE WHEN ihsv.cluster_id = 83 THEN 1 ELSE 0 END) AS hsv_83,
    SUM(CASE WHEN ihsv.cluster_id = 84 THEN 1 ELSE 0 END) AS hsv_84,
    SUM(CASE WHEN ihsv.cluster_id = 85 THEN 1 ELSE 0 END) AS hsv_85,
    SUM(CASE WHEN ihsv.cluster_id = 86 THEN 1 ELSE 0 END) AS hsv_86,
    SUM(CASE WHEN ihsv.cluster_id = 87 THEN 1 ELSE 0 END) AS hsv_87,
    SUM(CASE WHEN ihsv.cluster_id = 88 THEN 1 ELSE 0 END) AS hsv_88,
    SUM(CASE WHEN ihsv.cluster_id = 89 THEN 1 ELSE 0 END) AS hsv_89,
    SUM(CASE WHEN ihsv.cluster_id = 90 THEN 1 ELSE 0 END) AS hsv_90,
    SUM(CASE WHEN ihsv.cluster_id = 91 THEN 1 ELSE 0 END) AS hsv_91,
    SUM(CASE WHEN ihsv.cluster_id = 92 THEN 1 ELSE 0 END) AS hsv_92,
    SUM(CASE WHEN ihsv.cluster_id = 93 THEN 1 ELSE 0 END) AS hsv_93,
    SUM(CASE WHEN ihsv.cluster_id = 94 THEN 1 ELSE 0 END) AS hsv_94,
    SUM(CASE WHEN ihsv.cluster_id = 95 THEN 1 ELSE 0 END) AS hsv_95,
    SUM(CASE WHEN ihsv.cluster_id = 96 THEN 1 ELSE 0 END) AS hsv_96

FROM SegmentBig_isface so
JOIN SegmentHelper_sept2025_heft_keywords sh ON sh.image_id = so.image_id
JOIN ImagesBodyPoses3D ibp ON ibp.image_id = so.image_id
JOIN ClustersMetaBodyPoses3D ihp ON ihp.cluster_id = ibp.cluster_id
JOIN ImagesHSV ihsv ON ihsv.image_id = so.image_id
JOIN Images{MODE} it ON it.image_id = so.image_id
WHERE it.{MODE_ID} = {THIS_MODE_ID}
GROUP BY
    ihp.meta_cluster_id
ORDER BY 
    ihp.meta_cluster_id;
"""


sql_query_template_MetaHSV_MetaBody3D = """
SELECT 
    ihp.meta_cluster_id AS ihp_cluster,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 0 THEN 1 ELSE 0 END) AS hsv_0,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 1 THEN 1 ELSE 0 END) AS hsv_1,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 2 THEN 1 ELSE 0 END) AS hsv_2,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 3 THEN 1 ELSE 0 END) AS hsv_3,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 4 THEN 1 ELSE 0 END) AS hsv_4,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 5 THEN 1 ELSE 0 END) AS hsv_5,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 6 THEN 1 ELSE 0 END) AS hsv_6,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 7 THEN 1 ELSE 0 END) AS hsv_7,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 8 THEN 1 ELSE 0 END) AS hsv_8,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 9 THEN 1 ELSE 0 END) AS hsv_9,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 10 THEN 1 ELSE 0 END) AS hsv_10,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 11 THEN 1 ELSE 0 END) AS hsv_11,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 12 THEN 1 ELSE 0 END) AS hsv_12,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 13 THEN 1 ELSE 0 END) AS hsv_13,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 14 THEN 1 ELSE 0 END) AS hsv_14,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 15 THEN 1 ELSE 0 END) AS hsv_15,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 16 THEN 1 ELSE 0 END) AS hsv_16,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 17 THEN 1 ELSE 0 END) AS hsv_17,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 18 THEN 1 ELSE 0 END) AS hsv_18,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 19 THEN 1 ELSE 0 END) AS hsv_19,    
    SUM(CASE WHEN cmhsv.meta_cluster_id = 20 THEN 1 ELSE 0 END) AS hsv_20,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 21 THEN 1 ELSE 0 END) AS hsv_21,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 22 THEN 1 ELSE 0 END) AS hsv_22

FROM SegmentBig_isface so
JOIN SegmentHelper_sept2025_heft_keywords sh ON sh.image_id = so.image_id
JOIN ImagesBodyPoses3D ibp ON ibp.image_id = so.image_id
JOIN ClustersMetaBodyPoses3D ihp ON ihp.cluster_id = ibp.cluster_id
JOIN ImagesHSV ihsv ON ihsv.image_id = so.image_id
JOIN ClustersMetaHSV cmhsv ON cmhsv.cluster_id = ihsv.cluster_id
JOIN Images{MODE} it ON it.image_id = so.image_id
WHERE it.{MODE_ID} = {THIS_MODE_ID}
GROUP BY
    ihp.meta_cluster_id
ORDER BY 
    ihp.meta_cluster_id;
"""


sql_query_template_MetaHSV_Body3D = """

SELECT 
    ibp.cluster_id AS ihp_cluster,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 0 THEN 1 ELSE 0 END) AS hsv_0,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 1 THEN 1 ELSE 0 END) AS hsv_1,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 2 THEN 1 ELSE 0 END) AS hsv_2,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 3 THEN 1 ELSE 0 END) AS hsv_3,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 4 THEN 1 ELSE 0 END) AS hsv_4,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 5 THEN 1 ELSE 0 END) AS hsv_5,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 6 THEN 1 ELSE 0 END) AS hsv_6,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 7 THEN 1 ELSE 0 END) AS hsv_7,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 8 THEN 1 ELSE 0 END) AS hsv_8,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 9 THEN 1 ELSE 0 END) AS hsv_9,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 10 THEN 1 ELSE 0 END) AS hsv_10,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 11 THEN 1 ELSE 0 END) AS hsv_11,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 12 THEN 1 ELSE 0 END) AS hsv_12,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 13 THEN 1 ELSE 0 END) AS hsv_13,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 14 THEN 1 ELSE 0 END) AS hsv_14,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 15 THEN 1 ELSE 0 END) AS hsv_15,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 16 THEN 1 ELSE 0 END) AS hsv_16,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 17 THEN 1 ELSE 0 END) AS hsv_17,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 18 THEN 1 ELSE 0 END) AS hsv_18,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 19 THEN 1 ELSE 0 END) AS hsv_19,    
    SUM(CASE WHEN cmhsv.meta_cluster_id = 20 THEN 1 ELSE 0 END) AS hsv_20,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 21 THEN 1 ELSE 0 END) AS hsv_21,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 22 THEN 1 ELSE 0 END) AS hsv_22
    
FROM {CLUSTER_TABLE} ibp
JOIN SegmentBig_isface so ON so.image_id = ibp.image_id
JOIN {HELPER_TABLE} sh ON sh.image_id = ibp.image_id
JOIN ImagesHSV ihsv ON ihsv.image_id = so.image_id
JOIN ClustersMetaHSV cmhsv ON cmhsv.cluster_id = ihsv.cluster_id
JOIN Images{MODE} it ON it.image_id = so.image_id
WHERE it.{MODE_ID} = {THIS_MODE_ID}
GROUP BY
    ibp.cluster_id
ORDER BY 
    ibp.cluster_id;
"""

# FROM SegmentBig_isface so
# JOIN {HELPER_TABLE} sh ON sh.image_id = so.image_id
# JOIN {CLUSTER_TABLE} ibp ON ibp.image_id = so.image_id
# JOIN ImagesHSV ihsv ON ihsv.image_id = so.image_id


def save_query_results_to_csv(query, topic_id):
    print(f"about to query for {MODE} {topic_id}")

    # Execute query and fetch data into a DataFrame
    df = pd.read_sql(query, engine)

    # add zero values for any missing rows in the ihp_cluster column
    
    # THIS SHOULD BE DEFINED IN DICT ABOVE
    # if "body3D" in CLUSTER_TYPE: cluster_count = 512
    # elif "BodyPoses3D" in CLUSTER_TYPE: cluster_count = 768
    # elif "hand_gesture_position" in CLUSTER_TYPE: cluster_count = 128
    # elif "MetaBodyPoses3D" in CLUSTER_TYPE: cluster_count = 64
    # else: raise ValueError("Unknown CLUSTER_TYPE")
    cluster_count = CLUSTER_DATA.get(CLUSTER_TYPE, {}).get('cluster_count', 0)
    for i in range(cluster_count):
        print(f"checking for ihp_cluster {i}")
        if i not in df['ihp_cluster'].values:
            print(f"adding missing ihp_cluster {i}")
            new_row = {'ihp_cluster': i}
            for col in df.columns:
                if col != 'ihp_cluster':
                    new_row[col] = 0
            print(f"new_row: {new_row}")
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # resort by ihp_cluster
    df = df.sort_values(by='ihp_cluster').reset_index(drop=True)

    # create new column that is the sum of hsv_3 to hsv_22
    if 'hsv_3' in df.columns and 'hsv_22' in df.columns:
        hsv_columns = [f'hsv_{i}' for i in range(3, 23)]
        df['hsv_3_to_22_sum'] = df[hsv_columns].sum(axis=1)

    # create add a total column that is the sum of all columns
    df['total'] = df.sum(axis=1)

    # Define file name for the CSV
    csv_file_path = f"{ROOT_FOLDER_PATH}/{MODE}_{topic_id}.csv"
    
    # Save to CSV
    df.to_csv(csv_file_path, index=False)
    print(f"Saved results for {MODE} {topic_id} to {csv_file_path}")
    update_fusion_manifest(csv_file_path, topic_id, df)


def infer_entity_type(mode):
    if mode == "Topics":
        return "topic"
    if "Detections" in mode:
        return "detection"
    if mode == "Keywords":
        return "keyword"
    return "cluster"


def infer_available_hsv_bins(df):
    hsv_bins = []
    for col in df.columns:
        if col.startswith("hsv_"):
            suffix = col.replace("hsv_", "")
        elif col.startswith("object_hsv_"):
            suffix = col.replace("object_hsv_", "")
        else:
            continue
        if suffix.isdigit():
            hsv_bins.append(int(suffix))
    return sorted(list(set(hsv_bins)))


def load_or_init_manifest(folder_path):
    manifest_path = os.path.join(folder_path, MANIFEST_FILE)
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return manifest_path, json.load(f)

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "generator": "query_all_fusion_clusters.py",
        "mode": MODE,
        "mode_id": MODE_ID,
        "cluster_type": CLUSTER_TYPE,
        "files": {},
    }
    return manifest_path, manifest


def update_fusion_manifest(
    csv_file_path,
    topic_id,
    df,
    entity_type=None,
    hsv_preset_name="background_default",
):
    folder_path = os.path.dirname(csv_file_path)
    manifest_path, manifest = load_or_init_manifest(folder_path)

    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(
            f"Manifest contract version mismatch in {manifest_path}: "
            f"expected {CONTRACT_VERSION}, got {manifest.get('contract_version')}"
        )

    file_name = os.path.basename(csv_file_path)
    resolved_entity_type = entity_type or infer_entity_type(MODE)
    available_hsv_bins = infer_available_hsv_bins(df)

    manifest.setdefault("files", {})
    manifest["files"][file_name] = {
        "entity_type": resolved_entity_type,
        "entity_id": int(topic_id),
        "csv_schema_version": 1,
        "has_hsv_summary_rows": "hsv_3_to_22_sum" in df.columns,
        "available_hsv_bins": available_hsv_bins,
        "hsv_preset_name": hsv_preset_name,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Updated manifest: {manifest_path}")


def _focus_cluster_where_sql(focus_cluster_ids):
    if not focus_cluster_ids:
        return ""
    ids = ",".join(str(int(x)) for x in focus_cluster_ids)
    return f" AND io.cluster_id IN ({ids}) "


def _class_where_sql(class_ids):
    if not class_ids:
        raise ValueError("OBJECT_HSV_EXPORT_CLASS_IDS cannot be empty")
    ids = ",".join(str(int(x)) for x in class_ids)
    return f" AND d.class_id IN ({ids}) "


def _ensure_hsv_columns(df, col_prefix="object_hsv", n_bins=23):
    for hsv_bin in range(n_bins):
        col_name = f"{col_prefix}_{hsv_bin}"
        if col_name not in df.columns:
            df[col_name] = 0
    ordered = [f"{col_prefix}_{hsv_bin}" for hsv_bin in range(n_bins)]
    lead_cols = [c for c in df.columns if c not in ordered]
    return df[lead_cols + ordered]


def export_objectfusion_object_hsv_csvs(root_folder_path, class_ids, focus_cluster_ids=None):
    """
    First pass export for ObjectFusion object-HSV analysis.
    Outputs:
      - Matrix A: per-class, rows=cluster_id cols=object_hsv_0..22
      - Matrix B: aggregate class_hsv columns by cluster_id
      - Table C: long relational table
    """
    os.makedirs(root_folder_path, exist_ok=True)

    class_where = _class_where_sql(class_ids)
    focus_where = _focus_cluster_where_sql(focus_cluster_ids)

    table_c_query = f"""
    SELECT
        io.cluster_id,
        d.class_id,
        d.meta_cluster_id AS object_hsv_bin,
        COUNT(DISTINCT io.image_id) AS image_count,
        COUNT(*) AS detection_count
    FROM ImagesObjectFusion io
    JOIN Detections d ON d.image_id = io.image_id
    WHERE d.meta_cluster_id IS NOT NULL
      {class_where}
      {focus_where}
    GROUP BY io.cluster_id, d.class_id, d.meta_cluster_id
    ORDER BY io.cluster_id, d.class_id, d.meta_cluster_id
    """

    print("Running ObjectFusion object-HSV long-table query...")
    df_long = pd.read_sql(table_c_query, engine)
    if df_long.empty:
        print("No rows returned for ObjectFusion object-HSV query. Nothing to export.")
        return

    # Table C
    class_tag = "-".join(str(int(x)) for x in class_ids)
    focus_tag = "all" if not focus_cluster_ids else "-".join(str(int(x)) for x in focus_cluster_ids)
    long_name = f"ObjectFusion_object_hsv_long_classes_{class_tag}_clusters_{focus_tag}.csv"
    long_path = os.path.join(root_folder_path, long_name)
    df_long.to_csv(long_path, index=False)
    print(f"Saved Table C: {long_path}")

    # Matrix A (one CSV per class)
    for class_id in class_ids:
        class_df = df_long[df_long["class_id"] == int(class_id)].copy()
        if class_df.empty:
            print(f"Skipping Matrix A for class {class_id}: no rows")
            continue
        matrix_a = class_df.pivot_table(
            index="cluster_id",
            columns="object_hsv_bin",
            values="image_count",
            aggfunc="sum",
            fill_value=0,
        )
        matrix_a.columns = [f"object_hsv_{int(c)}" for c in matrix_a.columns]
        matrix_a = matrix_a.reset_index().sort_values("cluster_id").reset_index(drop=True)
        matrix_a = _ensure_hsv_columns(matrix_a, col_prefix="object_hsv", n_bins=23)
        matrix_a["total_images_any_bin"] = matrix_a[[f"object_hsv_{i}" for i in range(23)]].sum(axis=1)

        matrix_a_name = f"ObjectFusion_object_hsv_matrix_class_{int(class_id)}_clusters_{focus_tag}.csv"
        matrix_a_path = os.path.join(root_folder_path, matrix_a_name)
        matrix_a.to_csv(matrix_a_path, index=False)
        print(f"Saved Matrix A: {matrix_a_path}")
        update_fusion_manifest(
            matrix_a_path,
            class_id,
            matrix_a,
            entity_type="object_fusion",
            hsv_preset_name="object_color_v1",
        )

    # Matrix B (aggregate class_hsv columns)
    df_b = df_long.copy()
    df_b["class_hsv_col"] = df_b.apply(
        lambda row: f"class_{int(row['class_id'])}_hsv_{int(row['object_hsv_bin'])}",
        axis=1,
    )
    matrix_b = df_b.pivot_table(
        index="cluster_id",
        columns="class_hsv_col",
        values="image_count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index().sort_values("cluster_id").reset_index(drop=True)
    matrix_b_name = f"ObjectFusion_object_hsv_matrix_class_hsv_clusters_{focus_tag}.csv"
    matrix_b_path = os.path.join(root_folder_path, matrix_b_name)
    matrix_b.to_csv(matrix_b_path, index=False)
    print(f"Saved Matrix B: {matrix_b_path}")


# Adjust the query template based on MODE
if CLUSTER_TYPE == "ObjectFusion_ObjectHSV":
    print("Running ObjectFusion object-HSV CSV exports (Matrix A/B + Table C)...")
    export_objectfusion_object_hsv_csvs(
        root_folder_path=ROOT_FOLDER_PATH,
        class_ids=OBJECT_HSV_EXPORT_CLASS_IDS,
        focus_cluster_ids=TEMP_FOCUS_CLUSTER_HACK_LIST,
    )
    session.close()
    sys.exit(0)

if MODE in ["Keywords", "ArmsPoses3D"] or "Detections" in MODE:
    print("Querying by Keywords with CLUSTER_TYPE:", CLUSTER_TYPE)
    # Loop through each keyword_id and save results to CSV
    for keyword_id in KEYWORDS:
        # if CLUSTER_TYPE == "body3D":
        #     sql_query_template = sql_query_template_body3D
        # elif CLUSTER_TYPE == "BodyPoses3D_HSV":
        #     sql_query_template = sql_query_template_HSV_Body3D
        # elif CLUSTER_TYPE == "MetaBodyPoses3D":
        #     sql_query_template = sql_query_template_HSV_MetaBody3D
        # elif CLUSTER_TYPE == "BodyPoses3D_MetaHSV":
        #     sql_query_template = sql_query_template_MetaHSV_Body3D
        sql_query_template_variable_name = CLUSTER_DATA.get(CLUSTER_TYPE, {}).get('sql_template', None)
        sql_query_template = globals().get(sql_query_template_variable_name, None)
        this_cluster_table = CLUSTER_DATA.get(CLUSTER_TYPE, {}).get('cluster_table_name', None)
        if sql_query_template is None or this_cluster_table is None:
            raise ValueError(f"SQL template or cluster table name not defined for CLUSTER_TYPE: {CLUSTER_TYPE}")
        if "Detections" in MODE: 
            sql_query_template = sql_query_template.replace("Images{MODE}", "{MODE}") # because Detections is a table itself
            if keyword_id in HACK_LIST_SKIP_DETECTIONS:
                print(f"removing Detection from query.")
                print("Before:", sql_query_template)
                sql_query_template = sql_query_template.replace("WHERE it.{MODE_ID} = {THIS_MODE_ID}", "WHERE 1=1") # remove the where clause for class_id
                print("After:", sql_query_template)
        if "ArmsPoses3D" in MODE:
            # remove the WHERE it.{MODE_ID} = {THIS_MODE_ID} as I want all the dat
            print(f"removing WHERE clause for ArmsPoses3D.")
            print("Before:", sql_query_template)
            sql_query_template = sql_query_template.replace("WHERE it.{MODE_ID} = {THIS_MODE_ID}", "WHERE 1=1") # remove the where clause for class_id
            print("After:", sql_query_template)
                
        sql_query_template = sql_query_template.replace("{MODE}", MODE).replace("{MODE_ID}", MODE_ID).replace("{THIS_MODE_ID}", str(keyword_id)).replace("{CLUSTER_TABLE}", this_cluster_table).replace("{HELPER_TABLE}", HELPER_TABLE) 
        print(sql_query_template)
        query = sql_query_template.format(keyword_id=keyword_id)
        save_query_results_to_csv(query, keyword_id)

elif MODE == "Topics":
    # Loop through each topic_id (0 to 63) and save results to CSV
    for topic_id in range(64):
        sql_query_template = sql_query_template.replace("{MODE}", MODE).replace("{MODE_ID}", MODE_ID).replace("{THIS_MODE_ID}", str(topic_id))
        query = sql_query_template.format(topic_id=topic_id)
        save_query_results_to_csv(query, topic_id)

# Close the session
session.close()
