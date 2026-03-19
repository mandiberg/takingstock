#sklearn imports
import gc
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.metrics import silhouette_score
from scipy.optimize import minimize
#from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
#from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
#from sklearn.model_selection import ParameterGrid
#from sklearn import metrics

import datetime   ####### for saving cluster analytics

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
# my ORM
from my_declarative_base import Encodings, Detections, Base, Images, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, ForeignKey

from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, update, Float
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool
import pymongo
from pick import pick

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import time
import pickle
from sys import platform

#mine
from mp_db_io import DataIO
from mp_sort_pose import SortPose
from tools_clustering import ToolsClustering

image_edge_multiplier_sm = [2.2, 2.2, 2.6, 2.2] # standard portrait
face_height_output = 500
motion = {"side_to_side": False, "forward_smile": True, "laugh": False, "forward_nosmile":  False, "static_pose":  False, "simple": False}
EXPAND = False
ONE_SHOT = False # take all files, based off the very first sort order.
JUMP_SHOT = False # jump to random file if can't find a run

'''
tracking time based on items, for speed predictions
earlier
47000, 240
100000, 695

Now
items, seconds
10000, 33
50000, 280
100000, 1030
200000, 3202
300000, 6275
1.1M, ???
'''


io = DataIO()
db = io.db
# io.db["name"] = "stock"
# io.db["name"] = "ministock"
# mongo_client = pymongo.MongoClient(io.dbmongo['host'])
# mongo_db = mongo_client[io.dbmongo['name']]
# io.mongo_db = mongo_db

# mongo_collection = mongo_db[io.dbmongo['collection']]


NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
title = 'Please choose your operation: '
options = ['kmeans cluster and save clusters', 'cluster assignment', 'calculate cluster medians, cluster_dist and save clusters', 'make meta clusters']
option, MODE = pick(options, title)
# MODE = 1
# CLUSTER_TYPE = "Clusters"
# CLUSTER_TYPE = "BodyPoses"
# CLUSTER_TYPE = "BodyPoses3D" # use this for META 3D body clusters, Arms will start build but messed up because of subset landmarks
# CLUSTER_TYPE = "ArmsPoses3D" 
CLUSTER_TYPE = "ObjectFusion" 
# CLUSTER_TYPE = "HandsPositions"
# CLUSTER_TYPE = "HandsGestures"
# CLUSTER_TYPE = "FingertipsPositions"
# CLUSTER_TYPE = "HSV" # only works with cluster save, not with assignment
VERBOSE = False
DEBUG_IMAGE_ID = None  # Set to None to disable single-image isolation
cl = ToolsClustering(CLUSTER_TYPE, VERBOSE=VERBOSE)
# Note: session will be passed to cl after engine/session creation below

if "3D" in cl.CLUSTER_TYPE:
    if cl.CLUSTER_TYPE == "ArmsPoses3D":
        # this is a hacky way of saying I want XYZ but not Vis
        LMS_DIMENSIONS = 3
    else:
        LMS_DIMENSIONS = 4
else:
    LMS_DIMENSIONS = 3
OFFSET = 0
# SELECT MAX(cmb.image_id) FROM ImagesBodyPoses3D cmb JOIN Encodings e ON cmb.image_id = e.image_id WHERE e.is_feet = 0;
# START_ID = 129478350 # only used in MODE 1
START_ID = 0 # only used in MODE 1
DET_CONF_THRESHOLD = 0.4 # only used for ObjectFusion clustering to filter out low confidence detections
# WHICH TABLE TO USE?
# SegmentTable_name = 'SegmentOct20'
SegmentTable_name = 'SegmentBig_isface'
# SegmentTable_name = 'Encodings'

# if doing MODE == 2, use SegmentHelper_name to subselect SQL query
# unless you know what you are doing, leave this as None
# SegmentHelper_name = None
# if cl.CLUSTER_TYPE == "ArmsPoses3D":
# SegmentHelper_name = 'SegmentHelper_sept2025_heft_keywords'
SegmentHelper_name = 'Detections'
# SegmentHelper_name = 'SegmentHelper_dec2025_body3D_outOfSegment'
# SegmentHelper_name = 'SegmentHelper_oct2025_every40'
FORCE_HAND_LANDMARKS = False # when doing ArmsPoses3D, default is True, so mongo_hand_landmarks = 1

# TESTING MODE - reduce dataset size for faster iteration using pre-filtered table
# Set to True to use SegmentHelper_oct2025_every40 (every 40th image, ~2.5% of full dataset)
# Set to False for production full dataset processing
SKIP_TESTING = False

# number of clusters produced. run GET_OPTIMAL_CLUSTERS and add that number here
# 32 for hand positions
# 128 for hand gestures
N_CLUSTERS = 768  # Increased from 768 - need more granularity to break up mega-clusters
N_META_CLUSTERS = 256
if MODE == 3: 
    META = True
    N_CLUSTERS = N_META_CLUSTERS
    LMS_DIMENSIONS = 4
else: META = False

ONE_SHOT= JUMP_SHOT= HSV_CONTROL=  INPAINT= OBJ_CLS_ID = UPSCALE_MODEL_PATH =None
# face_height_output, image_edge_multiplier, EXPAND=False, ONE_SHOT=False, JUMP_SHOT=False, HSV_CONTROL=None, VERBOSE=True,INPAINT=False, SORT_TYPE="128d", OBJ_CLS_ID = None,UPSCALE_MODEL_PATH=None, use_3D=False
cfg = {
    'motion': motion,
    'face_height_output': face_height_output,
    'image_edge_multiplier': image_edge_multiplier_sm,
    'EXPAND': EXPAND,
    'ONE_SHOT': ONE_SHOT,
    'JUMP_SHOT': JUMP_SHOT,
    'HSV_CONTROL': HSV_CONTROL,
    'VERBOSE': VERBOSE,
    'INPAINT': INPAINT,
    'SORT_TYPE': cl.CLUSTER_TYPE,
    'OBJ_CLS_ID': OBJ_CLS_ID,
    'UPSCALE_MODEL_PATH': UPSCALE_MODEL_PATH,
    'LMS_DIMENSIONS': LMS_DIMENSIONS
}
sort = SortPose(config=cfg)
# MM you need to use conda activate mps_torch310 
SUBSELECT_ONE_CLUSTER = 0

# overrides SUBSET_LANDMARKS which is now set in sort pose init
if cl.CLUSTER_TYPE == "BodyPoses3D": 
    if META: 
        # sort.SUBSET_LANDMARKS = sort.make_subset_landmarks(15,16)  # just wrists
        sort.SUBSET_LANDMARKS = sort.make_subset_landmarks(11,22)
        USE_SUBSET_MEDIANS = True
    # else: sort.SUBSET_LANDMARKS = None
    else:
        # setting SUBSET_LANDMARKS for to nose [0] and ears+mouth+rest of body [7-32]
        sort.SUBSET_LANDMARKS = sort.make_subset_landmarks(0,0) + sort.make_subset_landmarks(7,32)
        USE_SUBSET_MEDIANS = True
elif cl.CLUSTER_TYPE == "ArmsPoses3D":
    # print("OVERRIDE setting cl.CLUSTER_TYPE to BodyPoses3D for ArmsPoses3D subset median calculation: ",cl.CLUSTER_TYPE, sort.CLUSTER_TYPE)
    # sort.SUBSET_LANDMARKS = sort.make_subset_landmarks(0,0) + sort.make_subset_landmarks(7,22)
    sort.SUBSET_LANDMARKS = sort.make_subset_landmarks(0,22)
    USE_SUBSET_MEDIANS = True
    print("OVERRIDE after construction setting sort.SUBSET_LANDMARKS to: ",sort.SUBSET_LANDMARKS)
else:
    sort.SUBSET_LANDMARKS = None
    USE_SUBSET_MEDIANS = False
USE_HEAD_POSE = False

SHORTRANGE = False # controls a short range query for the face x,y,z and mouth gap
ANGLES = []
STRUCTURE = "list3D" # 2d "dict", 2d "list", 2d plus visibility "list3", 3d plus visibility "list3D"
print("STRUCTURE: ",STRUCTURE)
if "list3" in STRUCTURE: 
    print("setting 3D to True")
    sort.use_3D = True

# this works for using segment in stock, and for ministock
USE_SEGMENT = True

# get the best fit for clusters
GET_OPTIMAL_CLUSTERS=False

SAVE_FIG=False ##### option for saving the visualized data

# TK 4 HSV
# table_cluster_type is used in SQL queries and table class defs
table_cluster_type = cl.set_table_cluster_type(META)

# setting the data column and is_feet based on cl.CLUSTER_TYPE from dict
this_data_column = cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]

if USE_SEGMENT is True and (cl.CLUSTER_TYPE != "Clusters"):
    print("setting Poses SQL")
    dupe_table_pre  = "s" # set default dupe_table_pre to s
    FROM =f"{SegmentTable_name} s " # set base FROM
    if "SegmentBig_" in SegmentTable_name:
        # handles segmentbig which doesn't have is_dupe_of, etc
        FROM += f" JOIN Encodings e ON s.image_id = e.image_id "
        dupe_table_pre = "e"
        WHERE = f" {dupe_table_pre}.is_dupe_of IS NULL AND  {dupe_table_pre}.two_noses IS NULL "
    elif "Encodings" in SegmentTable_name:
        # handles segmentbig which doesn't have is_dupe_of, etc
        # FROM += f" JOIN Encodings e ON s.image_id = e.image_id "
        dupe_table_pre = "s" # because SegmentTable_name is Encodings and gets aliased as s 
        WHERE = f"  {dupe_table_pre}.is_dupe_of IS NULL AND  {dupe_table_pre}.two_noses IS NULL AND {dupe_table_pre}.is_face = 1 " # ensures we are still only using faces
    else:
        WHERE = f" {dupe_table_pre}.is_dupe_of IS NULL "
    # Basic Query, this works with SegmentOct20. Previously included s.face_x, s.face_y, s.face_z, s.mouth_gap
    SELECT = "DISTINCT(s.image_id)"

    # handle ObjectFusion, just get pitch, yaw, roll. Detections handled later:
    if cl.CLUSTER_TYPE == "ObjectFusion":
        SELECT += f" , {dupe_table_pre}.pitch, {dupe_table_pre}.yaw, {dupe_table_pre}.roll "

    if isinstance(this_data_column, list):
        if "HSV" in cl.CLUSTER_TYPE:
            SELECT = SELECT.replace(f"s.face_x, s.face_y, s.face_z, s.mouth_gap", f"ib.hue, ib.sat, ib.val ")
            FROM += f" JOIN ImagesBackground ib ON s.image_id = ib.image_id "
            WHERE += f" AND ib.hue IS NOT NULL AND ib.sat IS NOT NULL AND ib.val IS NOT NULL "
    else:
        WHERE += f" AND {dupe_table_pre}.{this_data_column} = 1 "

    if cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["is_feet"] is not None:
        WHERE += f" AND {dupe_table_pre}.is_feet = {cl.CLUSTER_DATA[cl.CLUSTER_TYPE]['is_feet']} "
    if cl.CLUSTER_DATA[cl.CLUSTER_TYPE].get("mongo_hand_landmarks") is not None and FORCE_HAND_LANDMARKS:
        WHERE += f" AND {dupe_table_pre}.mongo_hand_landmarks = {cl.CLUSTER_DATA[cl.CLUSTER_TYPE]['mongo_hand_landmarks']} "    
    # refactoring the above to use the dict Oct 11
    # if cl.CLUSTER_TYPE == "BodyPoses": WHERE += f" AND {dupe_table_pre}.mongo_body_landmarks = 1 and {dupe_table_pre}.is_feet = 1"
    # # elif cl.CLUSTER_TYPE == "BodyPoses3D": WHERE += f" AND {dupe_table_pre}.mongo_body_landmarks_3D = 1 and {dupe_table_pre}.is_feet = 1"
    # elif cl.CLUSTER_TYPE == "BodyPoses3D": WHERE += f" AND {dupe_table_pre}.mongo_body_landmarks_3D = 1"
    # elif cl.CLUSTER_TYPE == "ArmsPoses3D": WHERE += f" AND {dupe_table_pre}.mongo_body_landmarks_3D = 1"
    # elif cl.CLUSTER_TYPE == "HandsGestures": WHERE += f" AND {dupe_table_pre}.mongo_hand_landmarks = 1 "
    # elif cl.CLUSTER_TYPE in ["HandsPositions","FingertipsPositions"] : WHERE += f" AND {dupe_table_pre}.mongo_hand_landmarks_norm = 1 "

    if MODE == 0:
        if SHORTRANGE: WHERE += " AND s.face_x > -35 AND s.face_x < -24 AND s.face_y > -3 AND s.face_y < 3 AND s.face_z > -3 AND s.face_z < 3 "
    # FROM += f" INNER JOIN Encodings h ON h.image_id = s.image_id " 
    # FROM += f" INNER JOIN {HelperTable_name} h ON h.image_id = s.image_id " 
        if SUBSELECT_ONE_CLUSTER:
            if cl.CLUSTER_TYPE == "HandsGestures": subselect_cluster = "ImagesHandsPositions"
            elif cl.CLUSTER_TYPE == "HandsPositions": subselect_cluster = "ImagesHandsGestures"
            FROM += f" INNER JOIN {subselect_cluster} sc ON sc.image_id = s.image_id " 
            WHERE += f" AND sc.cluster_id = {SUBSELECT_ONE_CLUSTER} "
        if SegmentHelper_name:
            FROM += f" INNER JOIN {SegmentHelper_name} h ON h.image_id = s.image_id " 
        if cl.CLUSTER_TYPE == "ObjectFusion":
            WHERE += " AND s.image_id NOT IN (SELECT image_id FROM NoDetections) "
            WHERE += " AND s.image_id NOT IN (SELECT image_id FROM NoDetectionsCustom) "
            WHERE += f" AND h.bbox_norm IS NOT NULL AND h.conf >= {DET_CONF_THRESHOLD} " # ensures we have some kind of detection for ObjectFusion clustering
            if DEBUG_IMAGE_ID is not None:
                WHERE += f" AND s.image_id = {DEBUG_IMAGE_ID} "
                print(f"⚠️  DEBUG MODE: isolating to image_id = {DEBUG_IMAGE_ID}")
    elif MODE in (1,2):
        FROM += f" LEFT JOIN Images{table_cluster_type} ic ON s.image_id = ic.image_id"
        if MODE == 1: 
            WHERE += " AND ic.cluster_id IS NULL "
            if SegmentHelper_name:
                FROM += f" INNER JOIN {SegmentHelper_name} h ON h.image_id = s.image_id " 
                # WHERE += " AND h.is_body = 1"
            if START_ID:
                WHERE += f" AND s.image_id >= {START_ID} "
        elif MODE == 2: 
            # if doing MODE == 2, use SegmentHelper_name to subselect SQL query
            FROM += f" INNER JOIN {table_cluster_type} c ON c.cluster_id = ic.cluster_id"
            SELECT += ",ic.cluster_id, ic.cluster_dist, c.cluster_median"
            WHERE += " AND ic.cluster_id IS NOT NULL AND ic.cluster_dist IS NULL"
    elif MODE == 3:
        # make meta clusters. redefining as a simple full select of the clusters table
        SELECT = "*"
        FROM = table_cluster_type
        WHERE = " cluster_id IS NOT NULL "

    # WHERE += " AND h.is_body = 1"
    LIMIT = 10000000
    BATCH_LIMIT = 10000

    '''
    Poses
    500 11s
    1000 21s
    2000 43s
    4000 87s
    30000 2553 @ hands elbows x 3d
    100000 90s @ HAND_LMS
    1000000 1077s @ HAND_LMS
    1100000 1177 @ HAND_LMSx 3d
    '''

elif USE_SEGMENT is True and MODE == 0:
    print("setting Poses SQL MODE 0 where using regular Clusters and ImagesClusters tables")
    # where the script is looking for files list
    # do not use this if you are using the regular Clusters and ImagesClusters tables
    SegmentTable_name = 'SegmentOct20'

    # 3.8 M large table (for Topic Model)
    HelperTable_name = "SegmentHelperMar23_headon"
    HelperTable_name = None

    # Basic Query, this works with gettytest3
    SELECT = "DISTINCT(s.image_id)"
    FROM = f"{SegmentTable_name} s"
    if HelperTable_name: FROM += f" INNER JOIN {HelperTable_name} h ON h.image_id = s.image_id " 
    # WHERE = " s.mongo_body_landmarks = 1"
    WHERE = " s.mongo_face_landmarks = 1"

    # for selecting a specific topic
    FROM += f" INNER JOIN ImagesTopics it ON it.image_id = s.image_id " 
    WHERE += " AND it.topic_id = 22 "
    # WHERE = "s.face_x > -33 AND s.face_x < -27 AND s.face_y > -2 AND s.face_y < 2 AND s.face_z > -2 AND s.face_z < 2"
    LIMIT = 5000

    '''
    350k 1900s
    '''

    # # join with SSD tables. Satyam, use the one below
    # SELECT = "DISTINCT(e.image_id), e.face_encodings68"
    # FROM = "Encodings e"
    # QUERY = "e.image_id IN"
    # SUBQUERY = f"(SELECT seg1.image_id FROM {SegmentTable_name} seg1 )"
    # WHERE = f"{QUERY} {SUBQUERY}"
    # LIMIT = 1000000

elif USE_SEGMENT is True and MODE == 1:
    SegmentTable_name = 'SegmentOct20'
    # Basic Query, this works with gettytest3
    SELECT = "DISTINCT(s.image_id),s.face_encodings68"
    FROM = f"{SegmentTable_name} s LEFT JOIN Images{cl.CLUSTER_TYPE} ic ON s.image_id = ic.image_id"
    WHERE = "ic.cluster_id IS NULL"
    LIMIT = "100"
 
else:
    # Basic Query, this works with gettytest3
    SELECT = "DISTINCT(image_id),face_encodings68"
    FROM ="encodings"
    WHERE = "face_encodings68 IS NOT NULL"
    LIMIT = 1000
    SegmentTable_name = ""

if db['unix_socket']:
    # for MM's MAMP config
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), pool_pre_ping=True, pool_recycle=600, poolclass=NullPool)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), pool_pre_ping=True, pool_recycle=600, poolclass=NullPool)
# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Pass session to ToolsClustering instance for database access
cl.session = session

# Print startup configuration
print("\n" + "="*70)
print("CLUSTERING CONFIGURATION")
print("="*70)
print(f"MODE: {MODE} ({option})")
print(f"CLUSTER_TYPE: {cl.CLUSTER_TYPE}")
print(f"N_CLUSTERS: {N_CLUSTERS}")
if SKIP_TESTING:
    print(f"\n⚠️  TESTING MODE ACTIVE: Using SegmentHelper_oct2025_every40 (~2.5% of full dataset)")
    print(f"   Set SKIP_TESTING = False for production full dataset processing")
else:
    print(f"SKIP_TESTING: False (processing full dataset)")
print("="*70 + "\n")

# TK 4 HSV
Clusters, ImagesClusters, MetaClusters, ClustersMetaClusters = cl.construct_table_classes(table_cluster_type)

def selectSQL():
    if OFFSET: offset = f" OFFSET {str(OFFSET)}"
    else: offset = ""
    
    # Handle SKIP_TESTING for testing mode
    if SKIP_TESTING:
        # Add JOIN to pre-filtered testing table (every 40th image)
        selectsql = f"""
        SELECT {SELECT} FROM {FROM} 
        INNER JOIN SegmentHelper_oct2025_every40 test ON s.image_id = test.image_id
        WHERE {WHERE} LIMIT {str(LIMIT)} {offset};
        """
        print(f"TESTING MODE: Using SegmentHelper_oct2025_every40 (~1-2% of full dataset)")
    else:
        # Normal query without testing table
        selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)} {offset};"
    
    print("actual SELECT is: ", selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    
    # Log results
    print(f"Fetched {len(resultsjson):,} images")
    
    return(resultsjson)

def landmarks_to_df_columnar(df, add_list=False, fit_scaler=False):
    first_col = df.columns[1]
    print("first col: ",first_col)
    
    if cl.CLUSTER_TYPE == "ObjectFusion":
        print("cl.CLUSTER_TYPE == ObjectFusion, doing it via prepare_features_for_knn_v2", df)
        df_columnar = cl.prepare_features_for_knn_v2(df, fit_scaler=fit_scaler)
        # df_columnar = prepared_df.values
        return df_columnar
    
    # if the first column is an int, then the columns are integers
    if isinstance(first_col, int):
        numerical_columns = [col for col in df.columns if isinstance(col, int)]
        prefix_dim = False
    elif any(isinstance(col, str) and col.startswith('dim_') for col in df.columns):
        numerical_columns = [col for col in df.columns if col.startswith('dim_')]
        prefix_dim = True
    # set hand_columns = to the numerical_columns in sort.SUBSET_LANDMARKS
    print("lms df columns: ",df.columns)
    if sort.SUBSET_LANDMARKS and (len(numerical_columns)/len(sort.SUBSET_LANDMARKS)) % 1 != 0:
        # only do this if the number of numerical columns is not a multiple of the subset landmarks
        # which is to say: the lms have already been subsetted
        if prefix_dim: subset_columns = [f'dim_{i}' for i in sort.SUBSET_LANDMARKS]
        else: subset_columns = [i for i in sort.SUBSET_LANDMARKS]
        print("subset lms columns: ",subset_columns)
        if USE_HEAD_POSE:
            df = df.apply(sort.weight_face_pose, axis=1)
            head_columns = ['face_x', 'face_y', 'face_z', 'mouth_gap']
            subset_columns += head_columns
    elif cl.CLUSTER_TYPE == "HSV":
        subset_columns = ["hue", "sat", "val"]
    else:
        subset_columns = numerical_columns

    if "image_id" in df.columns:
        df_columnar = df[['image_id'] + subset_columns]
    else:
        df_columnar = df[subset_columns]

    if add_list:
        print("landmarks_to_df_columnar adding obj_bbox_list column")
        df_columnar["obj_bbox_list"] = df[subset_columns].values.tolist()

    # this is the old way before Arms3D subsetting. I'm not sure if there is an actual scenario where i want to assign the whole df
    # if add_list:
    #     print("landmarks_to_df_columnar adding obj_bbox_list column")
    #     df_columnar = df
    #     df_columnar["obj_bbox_list"] = df[subset_columns].values.tolist()
    # else:
    #     df_columnar = df[subset_columns]

    print("landmarks_to_df_columnar at the end these are the columns: ", df_columnar.columns)
    return df_columnar

# flatten_object_detections and prepare_features_for_knn have been moved to ToolsClustering class
# Use cl.flatten_object_detections(), cl.prepare_features_for_knn(), or cl.prepare_features_for_knn_v2() (with StandardScaler)

def kmeans_cluster(df, n_clusters=32, fit_scaler=True):
    # Select only the numerical columns (dim_0 to dim_65)
    print("kmeans_cluster sort.SUBSET_LANDMARKS: ",sort.SUBSET_LANDMARKS)



    if cl.CLUSTER_TYPE in ["BodyPoses", "BodyPoses3D", "ArmsPoses3D", "ObjectFusion"]:
        print("cl.CLUSTER_TYPE == BodyPoses || ArmsPoses3D", df)
        df_columnar = landmarks_to_df_columnar(df, fit_scaler=fit_scaler)
    else:
        df_columnar = df
    print("clustering subset data shape: ", df_columnar.shape)
    if hasattr(df_columnar, 'iloc'):
        print("first row of numerical data: ", df_columnar.iloc[0])
    else:
        print("columnar is not a df: ")


    kmeans = KMeans(n_clusters=n_clusters, n_init=10, init='k-means++', random_state=42, max_iter=300, verbose=1)
    kmeans.fit(df_columnar)
    clusters = kmeans.predict(df_columnar)
    return clusters
    
def best_score(df):
    print("starting best score", df)
    print("about to subset landmarks to thse columns: ",sort.SUBSET_LANDMARKS)
    df = landmarks_to_df_columnar(df)
    print("about to best score with subset data", df)

    n_list=np.linspace(4,24,6,dtype='int')
    score=np.zeros(len(n_list))
    for i,n_clusters in enumerate(n_list):
        kmeans = KMeans(n_clusters,n_init=10, init = 'k-means++', random_state = 42, max_iter = 300)
        preds = kmeans.fit_predict(df)
        score[i]=silhouette_score(df, preds)
    print(n_list, score)
    b_score=n_list[np.argmax(score)]
    
    return b_score
    
def geometric_median(X, eps=1e-5, zero_threshold=1e-6):
    """
    Compute the geometric median of an array of points using Weiszfeld's algorithm.
    Args:
        X: A 2D numpy array where each row is a point in n-dimensional space.
        eps: Convergence threshold.
        zero_threshold: Threshold below which values are set to zero.
    Returns:
        The geometric median or None if X is empty or invalid.
    """
    if len(X) == 0:
        return None

    def distance_sum(y, X):
        return np.sum(np.linalg.norm(X - y, axis=1))

    # Initial guess: mean of the points
    initial_guess = np.mean(X, axis=0)
    result = minimize(distance_sum, initial_guess, args=(X,), method='COBYLA', tol=eps)

    # Set values close to zero to exactly zero
    result_x = result.x
    result_x[np.abs(result_x) < zero_threshold] = 0
    return result_x

def calc_cluster_median(df, col_list, cluster_id):
    cluster_df = df[df['cluster_id'] == cluster_id]

    if "top_face_object" in cluster_df.columns:
        print("flattening object detection columns for cluster median calculation")
        cluster_df = cluster_df.copy()  # To avoid SettingWithCopyWarning
        # drop "image_d_id" if it exists, because it will mess with the knn input
        if 'image_id' in cluster_df.columns:
            cluster_df = cluster_df.drop(columns=['image_id'])
        # Use existing scaler (fit_scaler=False) for median calculation
        prepared_cluster_df = cl.prepare_features_for_knn_v2(cluster_df, fit_scaler=False)
        cluster_points = prepared_cluster_df.values
        print(f"Cluster {cluster_id} data after flattening: {prepared_cluster_df}")
        print(f"Cluster {cluster_id} points after flattening: {cluster_points}")
    else:
        print("calculating cluster median without object detection columns")
        print(f"Cluster {cluster_id} data: {cluster_df}")
        # Convert the selected dimensions into a NumPy array
        cluster_points = cluster_df[col_list].values
    print("cluster_points",(cluster_points[0]))
    # Check if there are valid points in the cluster
    if len(cluster_points) == 0 or np.isnan(cluster_points).any():
        print(f"No valid points for cluster {cluster_id}, skipping median calculation.")
        return None
    
    # print(f"Cluster {cluster_id} points: {cluster_points}")

    # Calculate the geometric median for the cluster points
    cluster_median = geometric_median(cluster_points)
    
    return cluster_median

def build_col_list(df):
    print("building col list for df columns: ", df.columns)
    col_list = {}
    col_list["left"] = col_list["right"] = col_list["body_lms"] = col_list["face"] = col_list["ObjectFusion"] =[]
    if cl.CLUSTER_TYPE == "ObjectFusion":
        # for ObjectFusion, we need to get pitch, yaw, roll and object detection columns
        col_list["ObjectFusion"] = ['pitch', 'yaw', 'roll', 'left_hand_object', 'right_hand_object',
                          'top_face_object', 'mouth_object', 'shoulder_object']
    elif "body" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]:
        # tests data_column, so works for ArmsPoses3D too
        second_column_name = df.columns[1]
        # if the second column == 0, then compress the columsn into a list:
        if second_column_name == 0:
            print("compressing body_lms columns into a list")
            # If columns are integers, convert them to strings before checking prefix
            col_list["body_lms"] = [col for col in df.columns]
        else:
            col_list["body_lms"] = [col for col in df.columns if col.startswith('dim_')]
    elif "hand" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]:
    # elif cl.CLUSTER_TYPE in ["HandsPositions", "HandsGestures", "FingertipsPositions"]:
        col_list["left"] = [col for col in df.columns if col.startswith('left_dim_')]
        col_list["right"] = [col for col in df.columns if col.startswith('right_dim_')]
    elif cl.CLUSTER_TYPE == "HSV":
        col_list["HSV"] = ["hue", "sat", "val"]
    elif cl.CLUSTER_TYPE == "Clusters":
        col_list["face"] = [col for col in df.columns if col.startswith('dim_')]
    return col_list

def zero_out_medians(cluster_median):
    for key in cluster_median.keys():
        if cluster_median[key] is not None and all([abs(val) < .01 for val in cluster_median[key]]):
            print(f" --- >>> Setting cluster {key} median to all zeros.")
            cluster_median[key] = [0.0 for _ in cluster_median[key]]
    return cluster_median

def calculate_cluster_medians(df):
    median_dict = {}
    col_list = build_col_list(df)

    print(f"Columns used for median calculation: {col_list}")
    print(f"All DataFrame columns for cl.CLUSTER_TYPE {cl.CLUSTER_TYPE}: {df.columns}")

    print(df)
    unique_clusters = set(df['cluster_id'])
    for cluster_id in unique_clusters:
        # cluster_median = calc_cluster_median(df, col_list, cluster_id)
        cluster_median = {}
        if cl.CLUSTER_TYPE == "ObjectFusion":
            print(f"[first call in elif] Calculating median for ObjectFusion cluster {cluster_id}")
            cluster_median["ObjectFusion"] = calc_cluster_median(df, col_list["ObjectFusion"], cluster_id)
            print(f"[first call in elif] Recalculated median for cluster {cluster_id}: {cluster_median}")
        elif "body" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]:
        # if cl.CLUSTER_TYPE in ("BodyPoses", "BodyPoses3D", "ArmsPoses3D"):
            cluster_median["body_lms"] = calc_cluster_median(df, col_list["body_lms"], cluster_id)
        elif "hand" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]:
        # elif cl.CLUSTER_TYPE in ["HandsPositions", "HandsGestures", "FingertipsPositions"]:
            cluster_median["left"] = calc_cluster_median(df, col_list["left"], cluster_id)
            cluster_median["right"] = calc_cluster_median(df, col_list["right"], cluster_id)
        elif cl.CLUSTER_TYPE == "HSV":
            cluster_median["HSV"] = calc_cluster_median(df, col_list["HSV"], cluster_id)
        elif cl.CLUSTER_TYPE == "Clusters":
            cluster_median["face"] = calc_cluster_median(df, col_list["face"], cluster_id)
            
        if cluster_median is not None:
            print(f"Recalculated median for cluster {cluster_id}: {cluster_median}")
            # if every value in the cluster median < .01, then set all values to 0.0
            cluster_median = zero_out_medians(cluster_median)
            # add left and right hands            
            if cl.CLUSTER_TYPE == "ObjectFusion":
                flattened_median = cluster_median["ObjectFusion"]
            elif "hand" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]:
            # if cl.CLUSTER_TYPE in ["HandsPositions", "HandsGestures", "FingertipsPositions"]:
                flattened_median = np.concatenate((cluster_median["left"], cluster_median["right"]))
            elif "body" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]:
                flattened_median = cluster_median["body_lms"]
            elif cl.CLUSTER_TYPE == "HSV":
                flattened_median = cluster_median["HSV"]
            elif cl.CLUSTER_TYPE == "Clusters":
                flattened_median = cluster_median["face"]
            print(f"compressed cluster_median for {cluster_id}: {flattened_median}")
            median_dict[cluster_id] = flattened_median
        else:
            print(f"Cluster {cluster_id} has no valid points or data.")
    return median_dict

# TK 4 HSV
def set_cluster_metacluster():
    if META:
        this_Cluster = MetaClusters
        this_CrosswalkClusters = ClustersMetaClusters
    else:
        this_Cluster = Clusters
        this_CrosswalkClusters = ImagesClusters
    return this_Cluster, this_CrosswalkClusters
        

def save_clusters_DB(median_dict, update=False):
    # Convert to set and Save the df to a table
    
    print("save_clusters_DB median_dict", median_dict)
    cluster_ids = median_dict.keys()
    print("save_clusters_DB cluster_ids", cluster_ids)
    this_cluster, this_crosswalk = set_cluster_metacluster()
    print("this_cluster: ", this_cluster)
    for cluster_id in cluster_ids:
        cluster_median = median_dict[cluster_id]
        
        # store the data in the database
        # Explicitly handle cluster_id 0
        # if cluster_id == 0:
        #     print("Handling cluster_id 0 explicitly, checking for cluster 1. this cluster is ", cluster_id)
        #     existing_record = session.query(Clusters).filter_by(cluster_id=1).first()
        # else:

        print("Checking for existing record with cluster_id ", cluster_id)
        # Check if the record already exists

        # this is where the timeout error is happening:
        '''
        sqlalchemy.exc.OperationalError: (pymysql.err.OperationalError) (2006, "MySQL server has gone away (BrokenPipeError(32, 'Broken pipe'))")
[SQL: SELECT `BodyPoses3D`.cluster_id AS `BodyPoses3D_cluster_id`, `BodyPoses3D`.cluster_median AS `BodyPoses3D_cluster_median` 
FROM `BodyPoses3D`'''
        existing_record = session.query(this_cluster).filter_by(cluster_id=cluster_id).first()

        if existing_record is None:
            # Save the geometric median into the database
            print(f"Saving new record with cluster_id {cluster_id}")
            instance = this_cluster(
                cluster_id=cluster_id,
                cluster_median=pickle.dumps(cluster_median)  # Serialize the geometric median
            )
            session.add(instance)
            session.flush()  # Force database insertion to catch issues early
            session.commit()  
            if cluster_id == 0:

                saved_record = session.query(this_cluster).filter_by(cluster_id=0).first()
                if saved_record:
                    print(f"Successfully saved cluster_id 0: {saved_record}")
                else:
                    print("Failed to save cluster_id 0.")
        elif existing_record is not None and update:
            print(f"Updating existing record with cluster_id {cluster_id} and median {cluster_median}")
            existing_record.cluster_median = pickle.dumps(cluster_median)
            
        else:
            print(f"Skipping duplicate record with cluster_id {cluster_id}")

    try:
        print(f"Attempting to commit session with {len(median_dict)}:")
        # for cluster_id, cluster_median in median_dict.items():
        #     print(f"Cluster ID: {cluster_id}, Median: {cluster_median}")
        session.commit()
        print("Data saved successfully.")
    except IntegrityError as e:
        session.rollback()
        print(f"Error occurred during data saving: {str(e)}")

# TK 4 HSV - after calculating cluster distances
def save_images_clusters_DB(df):
    #save the df to a table
    image_id = cluster_id = cluster_dist = this_cluster_id = meta_cluster_id = None
    print("save_images_clusters_DB df", df)
    print("columns: ",df.columns)
    this_cluster, this_crosswalk = set_cluster_metacluster()
    print("this_crosswalk: ", this_crosswalk)
    for idx, row in df.iterrows():
        # cluster_id = row['cluster_id']
        # cluster_dist = row['cluster_dist']

        if this_crosswalk == ImagesClusters:
            image_id = row['image_id']
            cluster_id = row['cluster_id']
            cluster_dist = row['cluster_dist']
            if any(pd.isna([image_id, cluster_id, cluster_dist])):
                print(f"Skipping row with NaN values: image_id={image_id}, cluster_id={cluster_id}, cluster_dist={cluster_dist}")
                continue
            existing_record = session.query(ImagesClusters).filter_by(image_id=image_id).first()
        elif this_crosswalk == ClustersMetaClusters:
            this_cluster_id = idx
            meta_cluster_id = row['cluster_id']
            cluster_dist = None
            print("this_cluster_id: ",this_cluster_id, " meta_cluster_id: ",meta_cluster_id)
            # look up the body3D cluster_id
            existing_record = session.query(ClustersMetaClusters).filter_by(cluster_id=this_cluster_id).first()

        if existing_record is None:
            # it may be easier to define this locally, and assign the name via cl.CLUSTER_TYPE
            if this_crosswalk == ImagesClusters:
                instance = ImagesClusters(
                    image_id=image_id,
                    cluster_id=cluster_id,
                    cluster_dist=cluster_dist
                )
            elif this_crosswalk == ClustersMetaClusters:
                instance = ClustersMetaClusters(
                    cluster_id=this_cluster_id,
                    meta_cluster_id=meta_cluster_id,  # here image_id is actually meta_cluster_id
                    cluster_dist=cluster_dist
                )
            session.add(instance)
        
        elif existing_record is not None:
            if existing_record.cluster_dist is None:
                if (image_id is not None and image_id % 100 == 0) or (this_cluster_id and this_cluster_id % 100 == 0):
                    print(f"Updating existing record with image_id {image_id} to cluster_dist {cluster_dist}")
                existing_record.cluster_dist = cluster_dist
            else:
                print(f"Skipping existing record with image_id {image_id} and cluster_dist {cluster_dist}")
        else:
            print(f"Skipping duplicate record with image_id {image_id}")

    try:
        print(f"Attempting to commit session with {len(df)}:")
        # for _, row in df.iterrows():
        #     # print(f"Image ID: {row['image_id']}, Cluster ID: {row['cluster_id']}, Cluster Dist: {row['cluster_dist']}")
        #     print(row)
        session.commit()
        print("Data saved successfully.")
    except IntegrityError as e:
        session.rollback()
        print(f"Error occurred during data saving: {str(e)}")

def calc_median_dist(enc1, enc2):
    # print("calc_median_dist enc1, enc2", enc1, enc2)
    # print("type enc1, enc2", type(enc1), type(enc2))
    # print("len enc1, enc2", len(enc1), len(enc2))
    return np.linalg.norm(enc1 - enc2, axis=0)

def process_landmarks_cluster_dist(df, df_subset_landmarks):
    first_col = df.columns[1]
    # print(f"process_landmarks_cluster_dist df type {type(df)} first col: ",first_col)
    # print(f"df_subset_landmarks type {type(df_subset_landmarks)} columns", df_subset_landmarks.columns)
    # if the first column is an int, then the columns are integers
    if isinstance(first_col, int):
        dim_columns = [col for col in df_subset_landmarks.columns if isinstance(col, int)]
    elif any(isinstance(col, str) and col.startswith('dim_') for col in df_subset_landmarks.columns):
    # Step 1: Identify columns that contain "_dim_"
        dim_columns = [col for col in df_subset_landmarks.columns if "dim_" in col]
    elif cl.CLUSTER_TYPE == "HSV":
        dim_columns = cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]
    else:
        print("process_landmarks_cluster_dist could not identify dim columns, defaulting to all columns except image_id and metadata")
        # Exclude image_id and hand position metadata columns from distance calculation
        exclude_cols = {'image_id', 'left_pointer_knuckle_norm', 'right_pointer_knuckle_norm', 'left_source', 'right_source'}
        dim_columns = [col for col in df_subset_landmarks.columns if col not in exclude_cols]
        print("process_landmarks_cluster_dist defaulted dim_columns: ", dim_columns)
    print("process_landmarks_cluster_dist dim_columns: ", dim_columns)
    # Step 2: Combine values from these columns into a list for each row
    df_subset_landmarks['enc1'] = df_subset_landmarks[dim_columns].values.tolist()

    # Step 3: Print the result to check
    print("df_subset_landmarks", df_subset_landmarks[['image_id', 'enc1']])
    print("df_subset_landmarks columns", df_subset_landmarks.columns)
    print("df first row", df.iloc[0])
    if 'cluster_id' not in df.columns:
        # assign clusters to all rows
        # df_subset_landmarks.loc[:, ['cluster_id', 'cluster_dist']] = zip(*df_subset_landmarks['enc1'].apply(cl.prep_pose_clusters_enc))
        cluster_results = df_subset_landmarks['enc1'].apply(cl.prep_pose_clusters_enc)
        # df_subset_landmarks['cluster_id'], df_subset_landmarks['cluster_dist'] = zip(*cluster_results)
        df_subset_landmarks[['cluster_id', 'cluster_dist']] = pd.DataFrame(cluster_results.tolist(), index=df_subset_landmarks.index)
    elif df['cluster_id'].isnull().values.any():
        # df_subset_landmarks["cluster_id"], df_subset_landmarks["cluster_dist"] = zip(*df_subset_landmarks["enc1"].apply(cl.prep_pose_clusters_enc))
        df_subset_landmarks.loc[df_subset_landmarks['cluster_id'].isnull(), ['cluster_id', 'cluster_dist']] = \
            zip(*df_subset_landmarks.loc[df_subset_landmarks['cluster_id'].isnull(), 'enc1'].apply(cl.prep_pose_clusters_enc))
    else:
        # Ensure cluster_median is present on df_subset_landmarks by merging on image_id
        if 'cluster_median' not in df_subset_landmarks.columns:
            if 'image_id' in df_subset_landmarks.columns and 'image_id' in df.columns and 'cluster_median' in df.columns:
                # merge cluster_median from df into df_subset_landmarks
                df_subset_landmarks = df_subset_landmarks.merge(df[['image_id', 'cluster_id','cluster_median']], on='image_id', how='left')
            else:
                # fallback: try to copy if same index alignment
                try:
                    df_subset_landmarks['cluster_median'] = df['cluster_median']
                except Exception:
                    print('Could not attach cluster_median to df_subset_landmarks')
        print("df_subset_landmarks before calc_median_dist", df_subset_landmarks.columns)
        # apply calc_median_dist to enc1 and cluster_median
        print("df_subset_landmarks enc1 and cluster_median", df_subset_landmarks[['enc1', 'cluster_median']].iloc[0])
        df_subset_landmarks["cluster_dist"] = df_subset_landmarks.apply(lambda row: calc_median_dist(row['enc1'], row['cluster_median']), axis=1)
    return df_subset_landmarks

# def prep_pose_clusters_enc(enc1):
#     # print("current image enc1", enc1)  
#     enc1 = np.array(enc1)
    
#     this_dist_dict = {}
#     for cluster_id in MEDIAN_DICT:
#         enc2 = MEDIAN_DICT[cluster_id]
#         # print("cluster_id enc2: ", cluster_id,enc2)
#         this_dist_dict[cluster_id] = np.linalg.norm(enc1 - enc2, axis=0)
    
#     cluster_id, cluster_dist = min(this_dist_dict.items(), key=lambda x: x[1])

#     # print(cluster_id)
#     return cluster_id, cluster_dist

def assign_images_clusters_DB(df):

    
    #assign clusters to each image's encodings
    print("assigning images to clusters, df at start",df)
    df_subset_landmarks = landmarks_to_df_columnar(df, add_list=True)
    print("df_subset_landmarks after landmarks_to_df_columnar", df_subset_landmarks)

    # if cl.CLUSTER_TYPE in ["BodyPoses","BodyPoses3D","ArmsPoses3D", "HandsGestures", "HandsPositions","FingertipsPositions"]:
    if "hand" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"] or "body" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]:
        # combine all columns that start with left_dim_ or right_dim_ or dim_ into one list in the "enc1" column
        df_subset_landmarks = process_landmarks_cluster_dist(df, df_subset_landmarks)
    else:
        # this is for obj_bbox_list
        df_subset_landmarks["cluster_id"] = df_subset_landmarks["obj_bbox_list"].apply(cl.prep_pose_clusters_enc)

    # if the cluster_id column contains tuples or lists, unpack them
    if isinstance(df_subset_landmarks["cluster_id"].iloc[0], (list, tuple)):
        df_subset_landmarks[['cluster_id', 'cluster_dist']] = pd.DataFrame(df_subset_landmarks['cluster_id'].tolist(), index=df_subset_landmarks.index)

    print("df_subset_landmarks clustered after apply")
    print(df_subset_landmarks)
    print(df_subset_landmarks[["image_id", "cluster_id","cluster_dist"]])

    # print all rows where cluster_id is 68
    # print(df_subset_landmarks[df_subset_landmarks["cluster_id"] == 68])

    save_images_clusters_DB(df_subset_landmarks)
    print ("saved to imagesclusters")

def df_list_to_cols(df, col_name):

    # Convert the string representation of lists to actual lists
    # df[col_name] = df[col_name].apply(eval)
    df_data = df.drop("image_id", axis=1)

    # drop any rows where col_name is None or NaN
    print(f"Dropping rows with NaN in column '{col_name}'")
    print("Before dropna, df_data len:", len(df_data))
    df_data = df_data.dropna(subset=[col_name])
    print("After dropna, df_data len:", len(df_data))
    try:
        # Create new columns for each coordinate
        num_coords = len(df_data[col_name].iloc[0])
    except Exception as e:
        print(f"Error determining number of coordinates in column '{col_name}': {e}")
        print("df_data", df_data)
        print("df_data", df_data[col_name])
        print("df_data", df_data[col_name].iloc[0])
        return df  # Return the original DataFrame if there's an error  
    for i in range(num_coords):
        df[f'dim_{i}'] = df[col_name].apply(lambda x: x[i] if x is not None else None)
    # Drop the original col_name column
    df = df.drop(col_name, axis=1)
    return df

def extract_face_geometry(df, sort):
    """
    Extract face geometry (nose position and face height) from each image.
    Updates df with nose_x, nose_y, face_height columns.
    """
    face_data = []
    for idx, row in df.iterrows():
        image_id = row['image_id']
        
        # Extract face landmarks and calculate position using SortPose methods
        try:
            face_landmarks = row['face_landmarks'] if 'face_landmarks' in row else None
            if pd.notna(face_landmarks):
                sort.faceLms = io.unpickle_array(face_landmarks)
                if sort.faceLms and hasattr(sort.faceLms, 'landmark') and len(sort.faceLms.landmark) > 0:
                    sort.w = row['w']
                    sort.h = row['h']

                    # Match utilities/draw_norm_lms.py process_face_doc()
                    sort.nose_2d = sort.get_face_2d_point(1)
                    sort.nose_x, sort.nose_y = sort.nose_2d
                    sort.get_faceheight_data()

                    face_data.append({
                        'image_id': image_id,
                        'nose_x': int(sort.nose_x) if sort.nose_x is not None else None,
                        'nose_y': int(sort.nose_y) if sort.nose_y is not None else None,
                        'face_height': int(sort.face_height) if sort.face_height is not None else None
                    })
                else:
                    face_data.append({'image_id': image_id, 'nose_x': None, 'nose_y': None, 'face_height': None})
            else:
                face_data.append({'image_id': image_id, 'nose_x': None, 'nose_y': None, 'face_height': None})
        except Exception as e:
            print(f"Error extracting face geometry for image_id {image_id}: {e}")
            face_data.append({'image_id': image_id, 'nose_x': None, 'nose_y': None, 'face_height': None})
    
    return pd.DataFrame(face_data)

def save_images_detections(df, engine):
    """
    Save hand positions and object detections to ImagesDetections table.
    
    Args:
        df: DataFrame with clustered data (must have left_pointer_knuckle_norm, right_pointer_knuckle_norm, 
                                           left_source, right_source, and detection columns)
        engine: SQLAlchemy engine for database connection
    """
    print("\n[DEBUG] save_images_detections called")
    print(f"[DEBUG] Input df shape: {df.shape}")
    print(f"[DEBUG] Columns in df: {list(df.columns)}")
    
    # Check if required columns exist
    required_cols = ['left_pointer_knuckle_norm', 'right_pointer_knuckle_norm', 'left_source', 'right_source']
    for col in required_cols:
        if col in df.columns:
            print(f"[DEBUG] ✓ Column '{col}' present")
        else:
            print(f"[DEBUG] ✗ Column '{col}' MISSING")
    
    if len(df) > 0:
        print(f"[DEBUG] Sample row (first image_id={df.iloc[0].get('image_id')}):")
        print(f"[DEBUG]   left_pointer_knuckle_norm: {df['left_pointer_knuckle_norm'].iloc[0] if 'left_pointer_knuckle_norm' in df.columns else 'N/A'}")
        print(f"[DEBUG]   right_pointer_knuckle_norm: {df['right_pointer_knuckle_norm'].iloc[0] if 'right_pointer_knuckle_norm' in df.columns else 'N/A'}")
        print(f"[DEBUG]   left_source: {df['left_source'].iloc[0] if 'left_source' in df.columns else 'N/A'}")
        print(f"[DEBUG]   right_source: {df['right_source'].iloc[0] if 'right_source' in df.columns else 'N/A'}")
    
    images_detections_records = []
    records_with_hand_data = 0
    records_with_default_only = 0
    
    for idx, row in df.iterrows():
        image_id = row.get('image_id', 'UNKNOWN')
        try:
            left_pos = row.get('left_pointer_knuckle_norm')
            right_pos = row.get('right_pointer_knuckle_norm')
            
            # Helper to extract scalar detection_id from potentially nested structures
            def extract_detection_id(val):
                # Check None first
                if val is None:
                    return None
                # Safely check for NaN/empty
                try:
                    if isinstance(val, dict):
                        detection_id = val.get('detection_id')
                        if detection_id is None:
                            return None
                        return int(detection_id)
                    if isinstance(val, (list, tuple)):
                        if len(val) == 0:
                            return None
                        first = val[0]
                        if first is None:
                            return None
                        return int(first)
                    elif isinstance(val, np.ndarray):
                        if val.size == 0:
                            return None
                        first = val.flat[0]
                        if first is None or (isinstance(first, float) and np.isnan(first)):
                            return None
                        return int(first)
                    else:
                        # Scalar
                        if isinstance(val, float) and np.isnan(val):
                            return None
                        return int(val)
                except (TypeError, ValueError, IndexError):
                    return None
            
            # Extract position coordinates, handling numpy arrays safely
            left_x = None
            left_y = None
            if left_pos is not None:
                try:
                    if hasattr(left_pos, '__len__') and len(left_pos) >= 2:
                        left_x = float(left_pos[0])
                        left_y = float(left_pos[1])
                except (TypeError, ValueError, IndexError) as e:
                    pass
            
            right_x = None
            right_y = None
            if right_pos is not None:
                try:
                    if hasattr(right_pos, '__len__') and len(right_pos) >= 2:
                        right_x = float(right_pos[0])
                        right_y = float(right_pos[1])
                except (TypeError, ValueError, IndexError) as e:
                    pass
            
            # Track whether this record has actual hand data (non-zero or non-default positions)
            has_hand_data = (left_x is not None and left_x != 0) or (right_x is not None and right_x != 0)
            if has_hand_data:
                records_with_hand_data += 1
            else:
                records_with_default_only += 1
            
            record = {
                'image_id': image_id,
                'left_pointer_x': left_x,
                'left_pointer_y': left_y,
                'left_source': row.get('left_source', 'default'),
                'right_pointer_x': right_x,
                'right_pointer_y': right_y,
                'right_source': row.get('right_source', 'default'),
                'left_hand_object_id': extract_detection_id(row.get('left_hand_object')),
                'right_hand_object_id': extract_detection_id(row.get('right_hand_object')),
                'top_face_object_id': extract_detection_id(row.get('top_face_object')),
                'mouth_object_id': extract_detection_id(row.get('mouth_object')),
                'shoulder_object_id': extract_detection_id(row.get('shoulder_object')),
            }
            images_detections_records.append(record)
        except Exception as e:
            import traceback
            print(f"[DEBUG] Error processing image_id {image_id}: {type(e).__name__}: {e}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            continue
    
    print(f"[DEBUG] Building {len(images_detections_records)} records for ImagesDetections...")
    print(f"[DEBUG]   Records with actual hand data (left_x or right_x != 0): {records_with_hand_data}")
    print(f"[DEBUG]   Records with default position only: {records_with_default_only}")
    
    if images_detections_records:
        df_to_insert = pd.DataFrame(images_detections_records)
        print(f"[DEBUG] DataFrame to insert shape: {df_to_insert.shape}")
        print(f"[DEBUG] Columns in df_to_insert: {list(df_to_insert.columns)}")
        
        # Show sample records with actual hand data
        hand_data_rows = df_to_insert[(df_to_insert['left_pointer_x'].notna() & (df_to_insert['left_pointer_x'] != 0)) | 
                                       (df_to_insert['right_pointer_x'].notna() & (df_to_insert['right_pointer_x'] != 0))]
        if len(hand_data_rows) > 0:
            print(f"[DEBUG] Sample record WITH hand data:")
            print(f"[DEBUG]   {hand_data_rows.iloc[0].to_dict()}")
        else:
            print(f"[DEBUG] WARNING: No records with actual hand positions found!")
            if len(df_to_insert) > 0:
                print(f"[DEBUG] Sample record (default position):")
                print(f"[DEBUG]   {df_to_insert.iloc[0].to_dict()}")
        
        try:
            df_to_insert.to_sql('ImagesDetections', con=engine, if_exists='append', index=False)
            print(f"✓ Saved {len(images_detections_records)} records to ImagesDetections table")
        except IntegrityError as e:
            print(f"Warning: Some records already exist in ImagesDetections (may be expected): {e}")
        except Exception as e:
            print(f"Error saving to ImagesDetections: {e}")
    else:
        print("WARNING: No valid records to save to ImagesDetections")

def prepare_df(df, process_object_detections=True, batch_label=None):
    print("columns: ",df.columns)
    print("prepare_df df with cl.CLUSTER_TYPE ", cl.CLUSTER_TYPE, df)
    print("prepare df first row",df.iloc[0])
    label = f"[{batch_label}] " if batch_label else ""
    columns_to_drop = []
    # apply io.convert_decimals_to_float to face_x, face_y, face_z, and mouth_gap 
    # if faxe_x, face_y, face_z, and mouth_gap are not already floats
    if 'face_x' in df.columns and df['face_x'].dtype != float:
        df[['face_x', 'face_y', 'face_z', 'mouth_gap']] = df[['face_x', 'face_y', 'face_z', 'mouth_gap']].astype(float)
    # if cluster_median column is in the df, unpickle it
    if 'cluster_median' in df.columns:
        df['cluster_median'] = df['cluster_median'].apply(io.unpickle_array)
    if cl.CLUSTER_TYPE == "ObjectFusion":

        print("first row of df",df.iloc[0].to_string())
        objectfusion_rows_in = len(df)
        print(f"{label}[COUNT] ObjectFusion rows entering prepare_df: {objectfusion_rows_in}")
        
        # Filter: Require body_landmarks_normalized (body pose data is mandatory)
        # This ensures we have reliable finger position data from body landmarks
        df = df[df['body_landmarks_normalized'].notna()]
        print(f"After filtering for body_landmarks_normalized: {len(df)} rows remaining")
        print(f"{label}[COUNT] Rows with body_landmarks_normalized: {len(df)} / {objectfusion_rows_in}")
        
        if len(df) == 0:
            print("WARNING: No rows with body_landmarks_normalized found. Cannot cluster.")
            return None
        
        # Replace NaN values with None before applying prep functions
        df['hand_results'] = df['hand_results'].apply(lambda x: None if pd.isna(x) else x)
        df[['left_hand_landmarks', 'left_hand_world_landmarks', 'left_hand_landmarks_norm', 'right_hand_landmarks', 'right_hand_world_landmarks', 'right_hand_landmarks_norm']] = pd.DataFrame(df['hand_results'].apply(sort.prep_hand_landmarks).tolist(), index=df.index)
        # Debug: check first row landmarks (using iloc to handle any index)
        if len(df) > 0 and len(df['left_hand_landmarks_norm'].iloc[0]) > 0:
            print(f"Sample left_hand_landmarks_norm (first landmark): {df['left_hand_landmarks_norm'].iloc[0][0]}")
        
        # Extract finger positions from body landmarks (primary source)
        # hand_results parameter kept for compatibility but body_landmarks_normalized is used
        print("\n[DEBUG] Extracting finger positions from body landmarks...")
        print(f"[DEBUG] df shape: {df.shape}")
        
        # CRITICAL: Unpickle body_landmarks_normalized before using
        print("[DEBUG] Unpickling body_landmarks_normalized...")
        df['body_landmarks_normalized'] = df['body_landmarks_normalized'].apply(io.unpickle_array)
        # print(f"[DEBUG] Sample body_landmarks_normalized (first row, after unpickling): {df['body_landmarks_normalized'].iloc[0] if len(df) > 0 else 'N/A'}")
        
        knuckle_results = df.apply(lambda row: sort.prep_knuckle_landmarks(row['hand_results'], row['body_landmarks_normalized']), axis=1).tolist()
        print(f"[DEBUG] prep_knuckle_landmarks returned {len(knuckle_results)} results")
        if len(knuckle_results) > 0:
            print(f"[DEBUG] Sample result (first row): {knuckle_results[0]}")
        
        df[["left_pointer_knuckle_norm","right_pointer_knuckle_norm","left_source","right_source"]] = pd.DataFrame(
            knuckle_results, 
            index=df.index)

        left_body_count = int((df['left_source'] == 'body').sum())
        right_body_count = int((df['right_source'] == 'body').sum())
        left_default_count = int((df['left_source'] == 'default').sum())
        right_default_count = int((df['right_source'] == 'default').sum())
        left_nondefault_count = int(df['left_pointer_knuckle_norm'].apply(lambda value: value != [0.0, 8.0, 0.0] if isinstance(value, list) else False).sum())
        right_nondefault_count = int(df['right_pointer_knuckle_norm'].apply(lambda value: value != [0.0, 8.0, 0.0] if isinstance(value, list) else False).sum())
        print(f"{label}[COUNT] Left knuckles from body/default: {left_body_count}/{left_default_count}")
        print(f"{label}[COUNT] Right knuckles from body/default: {right_body_count}/{right_default_count}")
        print(f"{label}[COUNT] Rows with non-default left/right knuckles: {left_nondefault_count}/{right_nondefault_count}")
        
        # Log source distribution for monitoring
        left_source_counts = df['left_source'].value_counts()
        right_source_counts = df['right_source'].value_counts()
        print(f"[DEBUG] Left finger position sources: {left_source_counts.to_dict()}")
        print(f"[DEBUG] Right finger position sources: {right_source_counts.to_dict()}")
        print(f"[DEBUG] Sample left_pointer_knuckle_norm (first row): {df['left_pointer_knuckle_norm'].iloc[0] if len(df) > 0 else 'N/A'}")
        print(f"[DEBUG] Sample right_pointer_knuckle_norm (first row): {df['right_pointer_knuckle_norm'].iloc[0] if len(df) > 0 else 'N/A'}")
        
        print("after getting finger positions from body landmarks",df.iloc[0].to_string())
        # df = sort.split_landmarks_to_columns(df, left_col="left_hand_landmarks_norm", right_col="right_hand_landmarks_norm")
        # df = sort.split_landmarks_to_columns_or_list(df, first_col="left_hand_landmarks_norm", second_col="right_hand_landmarks_norm", structure="cols")
    # def split_landmarks_to_columns_or_list(self, df, first_col="left_hand_world_landmarks", second_col="right_hand_world_landmarks", structure="cols"):
        print("after split",df)
        columns_to_drop = ['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_3D','body_landmarks_normalized', 
                           'hand_results', 'left_hand_landmarks', 'right_hand_landmarks', 
                           'left_hand_world_landmarks', 'right_hand_world_landmarks',
                           'left_hand_landmarks_norm', 'right_hand_landmarks_norm']
    # if "Object" in cl.CLUSTER_TYPE:
        print("first row before query",df.iloc[0].to_string())
        # apply query_detections to each image_id to get the object data
        # df['image_id'].apply(lambda image_id: query_detections(image_id))
        if process_object_detections:
            df = cl.process_detections_for_df(df)
            object_assignment_cols = ['left_hand_object', 'right_hand_object', 'top_face_object', 'mouth_object', 'shoulder_object']
            rows_with_any_object = int(df[object_assignment_cols].notna().any(axis=1).sum())
            rows_with_no_objects = int(len(df) - rows_with_any_object)
            print(f"{label}[COUNT] Rows with any object assignment: {rows_with_any_object}")
            print(f"{label}[COUNT] Rows with no object assignments: {rows_with_no_objects}")
            for col in object_assignment_cols:
                print(f"{label}[COUNT] {col} assigned: {int(df[col].notna().sum())}")
        else:
            for col in ['left_hand_object', 'right_hand_object', 'top_face_object', 'mouth_object', 'shoulder_object']:
                if col not in df.columns:
                    df[col] = None

        pass
    # if cl.CLUSTER_TYPE in ("BodyPoses", "BodyPoses3D", "ArmsPoses3D"):
    elif "body" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]:
        print(f"processing body landmarks for cl.CLUSTER_TYPE: {cl.CLUSTER_TYPE}: cl.CLUSTER_DATA[cl.CLUSTER_TYPE]['data_column']: {cl.CLUSTER_DATA[cl.CLUSTER_TYPE]['data_column']}")
        if "3D" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]:
            keep_col = "body_landmarks_3D"
            drop_col = "body_landmarks_normalized"
        else:
            keep_col = "body_landmarks_normalized"
            drop_col = "body_landmarks"

        df = df.dropna(subset=[keep_col])
        df[keep_col] = df[keep_col].apply(io.unpickle_array)
        # body = self.get_landmarks_2d(enc1, list(range(33)), structure)
        print(f"df size {len(df)} before get_landmarks_2d with {keep_col}", df)
        # getting errors here. I think it is because I have accumulated so many None's that it fills the whole df
        # this is because a lot of the is_body do not actually have mongo data
        # workaround is to update the start_id to skip the bad data
        df['body_landmarks_array'] = df[keep_col].apply(lambda x: sort.get_landmarks_2d(x, list(range(33)), structure=STRUCTURE))

        # apply io.convert_decimals_to_float to face_x, face_y, face_z, and mouth_gap 
        # df['body_landmarks_array'] = df.apply(lambda row: io.convert_decimals_to_float(row['body_landmarks_array'] + [row['face_x'], row['face_y'], row['face_z'], row['mouth_gap']]), axis=1)
        # drop the columns that are not needed
        # if not USE_HEAD_POSE: df = df.drop(columns=['face_x', 'face_y', 'face_z', 'mouth_gap']) 
        columns_to_drop=['face_encodings68', 'face_landmarks', 'body_landmarks', keep_col, drop_col]
        print("before cols",df.iloc[0])
        print("before cols",df.iloc[0]['body_landmarks_array'])
        df_list_to_cols(df, 'body_landmarks_array')
        print("after cols",df.iloc[0])
        if cl.CLUSTER_TYPE == "ArmsPoses3D":
            df = sort.make_subset_df_lms(df)
            print("after make_subset_df_lms",df)
    # elif cl.CLUSTER_TYPE == "HandsPositions":
    elif cl.CLUSTER_TYPE in ["HandsPositions","FingertipsPositions"]:
        print("first row of df",df.iloc[0])
        df[['left_hand_landmarks', 'left_hand_world_landmarks', 'left_hand_landmarks_norm', 'right_hand_landmarks', 'right_hand_world_landmarks', 'right_hand_landmarks_norm']] = pd.DataFrame(df['hand_results'].apply(sort.prep_hand_landmarks).tolist(), index=df.index)
        print("after prep",df)
        df = sort.split_landmarks_to_columns_or_list(df, first_col="left_hand_landmarks_norm", second_col="right_hand_landmarks_norm", structure="cols")
        print("after split",df)
        columns_to_drop = ['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized', 
                           'hand_results', 'left_hand_landmarks', 'right_hand_landmarks', 
                           'left_hand_world_landmarks', 'right_hand_world_landmarks',
                           'left_hand_landmarks_norm', 'right_hand_landmarks_norm']

    elif cl.CLUSTER_TYPE == "HandsGestures":
        # Replace NaN values with None before applying prep functions
        df['hand_results'] = df['hand_results'].apply(lambda x: None if pd.isna(x) else x)
        df[['left_hand_landmarks', 'left_hand_world_landmarks', 'left_hand_landmarks_norm', 'right_hand_landmarks', 'right_hand_world_landmarks', 'right_hand_landmarks_norm']] = pd.DataFrame(df['hand_results'].apply(sort.prep_hand_landmarks).tolist(), index=df.index)
        df = sort.split_landmarks_to_columns_or_list(df, first_col="left_hand_world_landmarks", second_col="right_hand_world_landmarks", structure="cols")
        # drop the columns that are not needed
        columns_to_drop = ['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized', 
                           'hand_results', 'left_hand_landmarks', 'right_hand_landmarks', 'left_hand_world_landmarks', 'right_hand_world_landmarks']
    elif cl.CLUSTER_TYPE == "HSV":
        # Replace NaN values with appropriate default before applying prep functions
        df['hue'] = df['hue'].apply(lambda x: None if pd.isna(x) else x)
        df['hue'] = pd.DataFrame(df["hue"].apply(sort.prep_hsv).tolist(), index=df.index)
        # df[['hsv']] = pd.DataFrame(df.apply(sort.prep_hsv, axis=1), index=df.index)
        # columns_to_drop = ['hue']

    elif cl.CLUSTER_TYPE == "Clusters":
        df = df.dropna(subset=['face_encodings68'])

        # Apply the unpickling function to the 'face_encodings' column
        df['face_encodings68'] = df['face_encodings68'].apply(io.unpickle_array)
        df['face_landmarks'] = df['face_landmarks'].apply(io.unpickle_array)
        df['body_landmarks'] = df['body_landmarks'].apply(io.unpickle_array)
        columns_to_drop=['face_landmarks', 'body_landmarks', 'body_landmarks_normalized']
        df_list_to_cols(df, 'face_encodings68')

    elif cl.CLUSTER_TYPE == "ObjectFusion":

        print("first row of df",df.iloc[0].to_string())
        objectfusion_rows_in = len(df)
        print(f"{label}[COUNT] ObjectFusion rows entering prepare_df: {objectfusion_rows_in}")
        
        # Filter: Require body_landmarks_normalized (body pose data is mandatory)
        # This ensures we have reliable finger position data from body landmarks
        df = df[df['body_landmarks_normalized'].notna()]
        print(f"After filtering for body_landmarks_normalized: {len(df)} rows remaining")
        print(f"{label}[COUNT] Rows with body_landmarks_normalized: {len(df)} / {objectfusion_rows_in}")
        
        if len(df) == 0:
            print("WARNING: No rows with body_landmarks_normalized found. Cannot cluster.")
            return None
        
        # Replace NaN values with None before applying prep functions
        df['hand_results'] = df['hand_results'].apply(lambda x: None if pd.isna(x) else x)
        df[['left_hand_landmarks', 'left_hand_world_landmarks', 'left_hand_landmarks_norm', 'right_hand_landmarks', 'right_hand_world_landmarks', 'right_hand_landmarks_norm']] = pd.DataFrame(df['hand_results'].apply(sort.prep_hand_landmarks).tolist(), index=df.index)
        # Debug: check first row landmarks (using iloc to handle any index)
        if len(df) > 0 and len(df['left_hand_landmarks_norm'].iloc[0]) > 0:
            print(f"Sample left_hand_landmarks_norm (first landmark): {df['left_hand_landmarks_norm'].iloc[0][0]}")
        
        # Extract finger positions from body landmarks (primary source)
        # hand_results parameter kept for compatibility but body_landmarks_normalized is used
        print("\n[DEBUG] Extracting finger positions from body landmarks...")
        print(f"[DEBUG] df shape: {df.shape}")
        
        # CRITICAL: Unpickle body_landmarks_normalized before using
        print("[DEBUG] Unpickling body_landmarks_normalized...")
        df['body_landmarks_normalized'] = df['body_landmarks_normalized'].apply(io.unpickle_array)
        # print(f"[DEBUG] Sample body_landmarks_normalized (first row, after unpickling): {df['body_landmarks_normalized'].iloc[0] if len(df) > 0 else 'N/A'}")
        
        knuckle_results = df.apply(lambda row: sort.prep_knuckle_landmarks(row['hand_results'], row['body_landmarks_normalized']), axis=1).tolist()
        print(f"[DEBUG] prep_knuckle_landmarks returned {len(knuckle_results)} results")
        if len(knuckle_results) > 0:
            print(f"[DEBUG] Sample result (first row): {knuckle_results[0]}")
        
        df[["left_pointer_knuckle_norm","right_pointer_knuckle_norm","left_source","right_source"]] = pd.DataFrame(
            knuckle_results, 
            index=df.index)

        left_body_count = int((df['left_source'] == 'body').sum())
        right_body_count = int((df['right_source'] == 'body').sum())
        left_default_count = int((df['left_source'] == 'default').sum())
        right_default_count = int((df['right_source'] == 'default').sum())
        left_nondefault_count = int(df['left_pointer_knuckle_norm'].apply(lambda value: value != [0.0, 8.0, 0.0] if isinstance(value, list) else False).sum())
        right_nondefault_count = int(df['right_pointer_knuckle_norm'].apply(lambda value: value != [0.0, 8.0, 0.0] if isinstance(value, list) else False).sum())
        print(f"{label}[COUNT] Left knuckles from body/default: {left_body_count}/{left_default_count}")
        print(f"{label}[COUNT] Right knuckles from body/default: {right_body_count}/{right_default_count}")
        print(f"{label}[COUNT] Rows with non-default left/right knuckles: {left_nondefault_count}/{right_nondefault_count}")
        
        # Log source distribution for monitoring
        left_source_counts = df['left_source'].value_counts()
        right_source_counts = df['right_source'].value_counts()
        print(f"[DEBUG] Left finger position sources: {left_source_counts.to_dict()}")
        print(f"[DEBUG] Right finger position sources: {right_source_counts.to_dict()}")
        print(f"[DEBUG] Sample left_pointer_knuckle_norm (first row): {df['left_pointer_knuckle_norm'].iloc[0] if len(df) > 0 else 'N/A'}")
        print(f"[DEBUG] Sample right_pointer_knuckle_norm (first row): {df['right_pointer_knuckle_norm'].iloc[0] if len(df) > 0 else 'N/A'}")
        
        print("after getting finger positions from body landmarks",df.iloc[0].to_string())
        # df = sort.split_landmarks_to_columns(df, left_col="left_hand_landmarks_norm", right_col="right_hand_landmarks_norm")
        # df = sort.split_landmarks_to_columns_or_list(df, first_col="left_hand_landmarks_norm", second_col="right_hand_landmarks_norm", structure="cols")
    # def split_landmarks_to_columns_or_list(self, df, first_col="left_hand_world_landmarks", second_col="right_hand_world_landmarks", structure="cols"):
        print("after split",df)
        columns_to_drop = ['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_3D','body_landmarks_normalized', 
                           'hand_results', 'left_hand_landmarks', 'right_hand_landmarks', 
                           'left_hand_world_landmarks', 'right_hand_world_landmarks',
                           'left_hand_landmarks_norm', 'right_hand_landmarks_norm']
    # if "Object" in cl.CLUSTER_TYPE:
        print("first row before query",df.iloc[0].to_string())
        # apply query_detections to each image_id to get the object data
        # df['image_id'].apply(lambda image_id: query_detections(image_id))
        if process_object_detections:
            df = cl.process_detections_for_df(df)
            object_assignment_cols = ['left_hand_object', 'right_hand_object', 'top_face_object', 'mouth_object', 'shoulder_object']
            rows_with_any_object = int(df[object_assignment_cols].notna().any(axis=1).sum())
            rows_with_no_objects = int(len(df) - rows_with_any_object)
            print(f"{label}[COUNT] Rows with any object assignment: {rows_with_any_object}")
            print(f"{label}[COUNT] Rows with no object assignments: {rows_with_no_objects}")
            for col in object_assignment_cols:
                print(f"{label}[COUNT] {col} assigned: {int(df[col].notna().sum())}")
        else:
            for col in ['left_hand_object', 'right_hand_object', 'top_face_object', 'mouth_object', 'shoulder_object']:
                if col not in df.columns:
                    df[col] = None

        pass
    
    if not USE_HEAD_POSE: 
        print("not using head pose, dropping face_x, face_y, face_z, mouth_gap")
        # add 'face_x', 'face_y', 'face_z', 'mouth_gap' to existing columns_to_drop
        if 'face_x' in df.columns:
            columns_to_drop += ['face_x', 'face_y', 'face_z', 'mouth_gap']
    
    print(f"[DEBUG] Before dropping columns:")
    print(f"[DEBUG]   df shape: {df.shape}")
    print(f"[DEBUG]   Columns: {list(df.columns)}")
    if 'left_pointer_knuckle_norm' in df.columns:
        print(f"[DEBUG]   ✓ left_pointer_knuckle_norm present")
    else:
        print(f"[DEBUG]   ✗ left_pointer_knuckle_norm MISSING")
    
    df = df.drop(columns=columns_to_drop)
    
    print(f"[DEBUG] After dropping columns:")
    print(f"[DEBUG]   df shape: {df.shape}")
    print(f"[DEBUG]   Columns: {list(df.columns)}")
    if 'left_pointer_knuckle_norm' in df.columns:
        print(f"[DEBUG]   ✓ left_pointer_knuckle_norm present")
    else:
        print(f"[DEBUG]   ✗ left_pointer_knuckle_norm MISSING")
    
    # if body_landmarks_array is in the df columns,
    # drop any rows where body_landmarks_array is None or NaN
    if 'body_landmarks_array' in df.columns:
        df = df.dropna(subset=['body_landmarks_array'])

    print(f"[DEBUG] Final prepared df:")
    print(f"[DEBUG]   shape: {len(df)}")
    print(f"[DEBUG]   Columns: {list(df.columns)}")
    if 'left_pointer_knuckle_norm' in df.columns:
        print(f"[DEBUG]   ✓ left_pointer_knuckle_norm present in final df")
        print(f"[DEBUG]   Sample value: {df['left_pointer_knuckle_norm'].iloc[0] if len(df) > 0 else 'N/A'}")
    else:
        print(f"[DEBUG]   ✗ left_pointer_knuckle_norm MISSING from final df!")
    return df

def fetch_mongo_for_batch(batch_df):
    """
    Fetch MongoDB encodings for a single batch DataFrame.
    Returns batch_df with MongoDB columns populated.
    """
    batch_results = []
    missing_count = 0
    error_count = 0
    success_count = 0
    
    for idx, row in batch_df.iterrows():
        image_id = int(row['image_id'])
        try:
            result = io.get_encodings_mongo(image_id)
            
            # result is a pd.Series with 6 values
            # Check if all values are None (missing document)
            if isinstance(result, pd.Series):
                is_all_none = result.isna().all() or all(v is None for v in result.values)
            else:
                is_all_none = all(v is None for v in result)
            
            if is_all_none:
                missing_count += 1
                if missing_count <= 5:
                    print(f"Missing/Empty result for image_id {image_id}")
                batch_results.append((None, None, None, None, None, None))
            else:
                success_count += 1
                # Convert Series to tuple if needed
                if isinstance(result, pd.Series):
                    batch_results.append(tuple(result.values))
                else:
                    batch_results.append(result)
                    
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                error_msg = str(e)[:150]
                print(f"ERROR fetching image_id {image_id}: {error_msg}")
            batch_results.append((None, None, None, None, None, None))
    
    # Convert results to DataFrame columns - CRITICAL: match batch_df's index
    results_df = pd.DataFrame(batch_results, columns=['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized', 'body_landmarks_3D', 'hand_results'], index=batch_df.index)
    batch_df[['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized', 'body_landmarks_3D', 'hand_results']] = results_df
    
    if missing_count > 0 or error_count > 0:
        print(f"Batch summary: success={success_count}, missing={missing_count}, errors={error_count}")
    
    return batch_df

def fetch_encodings_mongo_batched(df, batch_size=5000, mongo_reconnect_interval=5):
    """
    Fetch MongoDB encodings in batches with periodic reconnection to prevent timeouts.
    Handles missing documents and NumPy type conversions.
    Note: get_encodings_mongo() returns a pd.Series, not a tuple
    """
    print(f"Fetching encodings in batches of {batch_size}...")
    
    total_rows = len(df)
    results_list = []
    missing_count = 0
    error_count = 0
    success_count = 0
    
    # Debug: test one call to see what we get
    if total_rows > 0:
        test_image_id = int(df.iloc[0]['image_id'])
        test_result = io.get_encodings_mongo(test_image_id)
        # print(f"DEBUG: Sample result for image_id {test_image_id}:")
        # print(f"  Type: {type(test_result)}")
        # print(f"  Values: {test_result.tolist() if isinstance(test_result, pd.Series) else test_result}")
        # print(f"  All None?: {test_result.isna().all() if isinstance(test_result, pd.Series) else all(v is None for v in test_result)}")
    
    for batch_num, batch_start in enumerate(range(0, total_rows, batch_size)):
        batch_end = min(batch_start + batch_size, total_rows)
        batch_size_actual = batch_end - batch_start
        
        print(f"[Batch {batch_num + 1}] Fetching rows {batch_start} to {batch_end} ({batch_size_actual} rows)...")
        
        batch_df = df.iloc[batch_start:batch_end]
        batch_results = []
        
        for idx, row in batch_df.iterrows():
            image_id = int(row['image_id'])
            try:
                result = io.get_encodings_mongo(image_id)
                
                # result is a pd.Series with 6 values
                # Check if all values are None (missing document)
                if isinstance(result, pd.Series):
                    is_all_none = result.isna().all() or all(v is None for v in result.values)
                else:
                    is_all_none = all(v is None for v in result)
                
                if is_all_none:
                    missing_count += 1
                    if missing_count <= 10:
                        print(f"[Batch {batch_num + 1}] Missing/Empty result for image_id {image_id}")
                    batch_results.append((None, None, None, None, None, None))
                else:
                    success_count += 1
                    # Convert Series to tuple if needed
                    if isinstance(result, pd.Series):
                        batch_results.append(tuple(result.values))
                    else:
                        batch_results.append(result)
                    
            except Exception as e:
                error_count += 1
                if error_count <= 10:
                    error_msg = str(e)[:150]
                    print(f"[Batch {batch_num + 1}] ERROR fetching image_id {image_id}: {error_msg}")
                batch_results.append((None, None, None, None, None, None))
        
        results_list.extend(batch_results)
        gc.collect()
        batch_summary = f"success={success_count}, missing={missing_count}, errors={error_count}"
        print(f"[Batch {batch_num + 1}] Completed ({batch_summary})...")
    
    # Convert results list to DataFrame columns
    print(f"\n=== Summary ===")
    print(f"Total rows processed: {total_rows}")
    print(f"Successful: {success_count}")
    print(f"Missing/Empty: {missing_count}")
    print(f"Errors: {error_count}")
    print(f"=== End Summary ===\n")
    
    results_df = pd.DataFrame(results_list, columns=['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized', 'body_landmarks_3D', 'hand_results'])
    df[['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized', 'body_landmarks_3D', 'hand_results']] = results_df
    
    print(f"Finished fetching {total_rows} encodings")
    return df

# ==================== OBJECT-HAND RELATIONSHIP FUNCTIONS ====================
# These functions have been moved to ToolsClustering class.
# Use cl.query_and_classify_detections() and cl.process_detections_for_df() instead.

def fetch_and_prepare_batch(batch_df, batch_num):
    """
    Fetch MongoDB encodings AND prepare (including MySQL queries) for one batch.
    Returns prepared DataFrame ready for clustering.
    Refreshes MySQL connection to prevent timeouts.
    """
    global session, engine, cl
    
    # Refresh MySQL connection every batch to prevent timeouts
    print(f"[Batch {batch_num}] Refreshing MySQL connection...")
    session.close()
    engine.dispose()
    
    if db['unix_socket']:
        engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
            user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
        ), pool_pre_ping=True, pool_recycle=3600, poolclass=NullPool)
    else:
        engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(
            host=db['host'], db=db['name'], user=db['user'], pw=db['pass']
        ), pool_pre_ping=True, pool_recycle=3600, poolclass=NullPool)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    cl.session = session

    if cl.CLUSTER_TYPE == "ObjectFusion":
        print(f"[Batch {batch_num}] Hydrating precomputed ImagesDetections first...")
        batch_prepared = batch_df.copy()
        batch_prepared, missing_image_ids = cl.hydrate_detections_from_precomputed_table(batch_prepared)
        hits = len(batch_prepared) - len(missing_image_ids)
        print(f"[Batch {batch_num}] Precomputed hits: {hits}, missing: {len(missing_image_ids)}")

        batch_prepared['newly_processed_detection'] = False

        if missing_image_ids:
            print(f"[Batch {batch_num}] Fetching MongoDB encodings for missing IDs only...")
            missing_df = batch_df[batch_df['image_id'].isin(missing_image_ids)].copy()
            print(f"[Batch {batch_num}] [COUNT] Missing rows before Mongo fetch: {len(missing_df)}")
            missing_df = fetch_mongo_for_batch(missing_df)

            print(f"[Batch {batch_num}] Preparing missing rows and classifying detections...")
            missing_prepared = prepare_df(missing_df, batch_label=f"Batch {batch_num} missing")

            if missing_prepared is not None and len(missing_prepared) > 0:
                missing_prepared['newly_processed_detection'] = True

                update_cols = [
                    'left_hand_object', 'right_hand_object', 'top_face_object', 'mouth_object', 'shoulder_object',
                    'left_pointer_knuckle_norm', 'right_pointer_knuckle_norm', 'left_source', 'right_source',
                    'newly_processed_detection'
                ]

                missing_update_df = missing_prepared[['image_id'] + [c for c in update_cols if c in missing_prepared.columns]].copy()
                batch_prepared = batch_prepared.merge(missing_update_df, on='image_id', how='left', suffixes=('', '__missing'))

                for col in update_cols:
                    missing_col = f"{col}__missing"
                    if missing_col not in batch_prepared.columns:
                        continue
                    if col not in batch_prepared.columns:
                        batch_prepared[col] = batch_prepared[missing_col]
                    else:
                        batch_prepared[col] = batch_prepared[missing_col].combine_first(batch_prepared[col])
                    batch_prepared.drop(columns=[missing_col], inplace=True)

                print(f"[Batch {batch_num}] [COUNT] Missing rows surviving prepare_df: {len(missing_prepared)}")
                if 'left_pointer_knuckle_norm' in batch_prepared.columns:
                    print(f"[Batch {batch_num}] [COUNT] Batch rows with left knuckle column populated: {int(batch_prepared['left_pointer_knuckle_norm'].notna().sum())}")
                if 'right_pointer_knuckle_norm' in batch_prepared.columns:
                    print(f"[Batch {batch_num}] [COUNT] Batch rows with right knuckle column populated: {int(batch_prepared['right_pointer_knuckle_norm'].notna().sum())}")
                for col in ['left_hand_object', 'right_hand_object', 'top_face_object', 'mouth_object', 'shoulder_object']:
                    if col in batch_prepared.columns:
                        print(f"[Batch {batch_num}] [COUNT] Batch {col} populated: {int(batch_prepared[col].notna().sum())}")
            else:
                print(f"[Batch {batch_num}] [COUNT] Missing rows surviving prepare_df: 0")
        else:
            print(f"[Batch {batch_num}] No missing IDs; skipping MongoDB fetch.")
    else:
        # Non-ObjectFusion path still requires MongoDB data for prepare_df
        print(f"[Batch {batch_num}] Fetching MongoDB encodings...")
        batch_df = fetch_mongo_for_batch(batch_df)

        # Prepare df (includes MySQL Detections query via cl.process_detections_for_df)
        print(f"[Batch {batch_num}] Preparing data (includes MySQL queries)...")
        batch_prepared = prepare_df(batch_df, batch_label=f"Batch {batch_num}")
    
    return batch_prepared


# defining globally # TK 4 HSV
MEDIAN_DICT = cl.get_cluster_medians(session, Clusters, USE_SUBSET_MEDIANS, sort.SUBSET_LANDMARKS)
print("MEDIAN_DICT len: ",len(MEDIAN_DICT))

def main():
    start = time.time()
    global MEDIAN_DICT

    def calculate_clusters_and_save(enc_data):
        # I drop image_id, etc as I pass it to knn bc I need it later, but knn can't handle strings
        print("df columns: ",enc_data.columns)
        columns_to_drop = []
        columns_to_check = ["image_id", "cluster_id", "body_landmarks_array", "left_hand_landmarks_norm", "right_hand_landmarks_norm", "hand_results", "face_encodings68"]
        columns_to_drop += [col for col in columns_to_check if col in enc_data.columns]
        print("columns to drop: ",columns_to_drop)
        
        print(f"\n[DEBUG] Before KMeans clustering:")
        print(f"[DEBUG]   enc_data shape: {enc_data.shape}")
        print(f"[DEBUG]   Columns: {list(enc_data.columns)}")
        if 'left_pointer_knuckle_norm' in enc_data.columns:
            print(f"[DEBUG]   ✓ left_pointer_knuckle_norm present before clustering")
        else:
            print(f"[DEBUG]   ✗ left_pointer_knuckle_norm MISSING before clustering")
        
        # fit_scaler=True for MODE 0 (creating clusters from scratch)
        enc_data["cluster_id"] = kmeans_cluster(enc_data.drop(columns=columns_to_drop), n_clusters=N_CLUSTERS, fit_scaler=True)
        
        print("enc_data", enc_data)
        print("as list", set(enc_data["cluster_id"].tolist()))
    
        median_dict = calculate_cluster_medians(enc_data)
        save_clusters_DB(median_dict)
        
        # add the correct median_dict for each cluster_id to the enc_data
        enc_data["cluster_median"] = enc_data["cluster_id"].apply(lambda x: median_dict[x])
        if cl.CLUSTER_TYPE != "HSV":
            # this is specific to lms, not hsv
            # Use fit_scaler=False here - scaler already fitted above
            print(f"\n[DEBUG] Before landmarks_to_df_columnar:")
            print(f"[DEBUG]   enc_data shape: {enc_data.shape}")
            print(f"[DEBUG]   Columns: {list(enc_data.columns)}")
            if 'left_pointer_knuckle_norm' in enc_data.columns:
                print(f"[DEBUG]   ✓ left_pointer_knuckle_norm present before landmarks_to_df_columnar")
            else:
                print(f"[DEBUG]   ✗ left_pointer_knuckle_norm MISSING before landmarks_to_df_columnar")
            
            # CRITICAL: Preserve hand position metadata columns before transformation
            hand_position_cols = ['image_id', 'left_pointer_knuckle_norm', 'right_pointer_knuckle_norm', 'left_source', 'right_source']
            hand_position_metadata = enc_data[[col for col in hand_position_cols if col in enc_data.columns]].copy()
            print(f"[DEBUG] Saving hand position metadata: {list(hand_position_metadata.columns)}")
            
            df_columnar = landmarks_to_df_columnar(enc_data, add_list=True, fit_scaler=False)
            
            print(f"\n[DEBUG] After landmarks_to_df_columnar:")
            print(f"[DEBUG]   df_columnar shape: {df_columnar.shape}")
            print(f"[DEBUG]   Columns: {list(df_columnar.columns)}")
            if 'left_pointer_knuckle_norm' in df_columnar.columns:
                print(f"[DEBUG]   ✓ left_pointer_knuckle_norm present after landmarks_to_df_columnar")
            else:
                print(f"[DEBUG]   ✗ left_pointer_knuckle_norm MISSING after landmarks_to_df_columnar")
            
            # CRITICAL: Merge hand position metadata back into df_columnar
            print(f"\n[DEBUG] Restoring hand position metadata to df_columnar...")
            df_columnar = df_columnar.merge(hand_position_metadata, on='image_id', how='left')
            print(f"[DEBUG] After restoring hand position metadata:")
            print(f"[DEBUG]   df_columnar shape: {df_columnar.shape}")
            if 'left_pointer_knuckle_norm' in df_columnar.columns:
                print(f"[DEBUG]   ✓ left_pointer_knuckle_norm RESTORED to df_columnar")
            else:
                print(f"[DEBUG]   ✗ left_pointer_knuckle_norm STILL MISSING after restore!")
        else:
            df_columnar = enc_data
        print("df_columnar", df_columnar)
        if not META:
            df_columnar = process_landmarks_cluster_dist(enc_data,df_columnar)
            print("df_columnar after process_landmarks", df_columnar)

        if len(df_columnar) <= BATCH_LIMIT:
            save_images_clusters_DB(df_columnar)
        else:
            print(f"Large dataset ({len(df_columnar)} rows) — processing in batches of {BATCH_LIMIT}")
            for start in range(0, len(df_columnar), BATCH_LIMIT):
                end = min(start + BATCH_LIMIT, len(df_columnar))
                print(f"Processing batch rows {start} to {end}...")
                batch_df = df_columnar.iloc[start:end].copy()
                save_images_clusters_DB(batch_df)
                # Free memory between batches
                gc.collect()
        
        # Save face geometry and hand detection data for ObjectFusion clustering
        if cl.CLUSTER_TYPE == "ObjectFusion":
            print("\n=== Saving ImagesDetections table ===")
            try:
                # Extract face geometry from original df
                face_geometry_df = extract_face_geometry(df, sort)
                
                print(f"[DEBUG] Before merge:")
                print(f"[DEBUG]   enc_data shape: {enc_data.shape if 'image_id' in enc_data.columns else 'N/A (will use df_columnar)'}")
                print(f"[DEBUG]   df_columnar shape: {df_columnar.shape}")
                print(f"[DEBUG]   face_geometry_df shape: {face_geometry_df.shape}")
                
                # Merge face geometry with full ObjectFusion data (contains knuckle/source columns)
                if 'image_id' in enc_data.columns:
                    print(f"[DEBUG]   Using enc_data for merge (has image_id)")
                    print(f"[DEBUG]   enc_data columns: {list(enc_data.columns)}")
                    df_with_geometry = enc_data.merge(face_geometry_df, on='image_id', how='left')
                else:
                    print(f"[DEBUG]   Using df_columnar for merge (enc_data missing image_id)")
                    print(f"[DEBUG]   df_columnar columns: {list(df_columnar.columns)}")
                    df_with_geometry = df_columnar.merge(face_geometry_df, on='image_id', how='left')
                
                print(f"[DEBUG] After merge:")
                print(f"[DEBUG]   df_with_geometry shape: {df_with_geometry.shape}")
                print(f"[DEBUG]   df_with_geometry columns: {list(df_with_geometry.columns)}")
                
                # Check for hand position columns
                hand_cols = ['left_pointer_knuckle_norm', 'right_pointer_knuckle_norm', 'left_source', 'right_source']
                for col in hand_cols:
                    if col in df_with_geometry.columns:
                        print(f"[DEBUG]   ✓ Column '{col}' found in df_with_geometry")
                    else:
                        print(f"[DEBUG]   ✗ Column '{col}' MISSING from df_with_geometry")

                # Update Encodings table with face geometry using raw SQL so ORM class can lag schema
                updated_count = 0
                try:
                    col_check_sql = text("""
                        SELECT COLUMN_NAME
                        FROM information_schema.COLUMNS
                        WHERE TABLE_SCHEMA = DATABASE()
                          AND TABLE_NAME = 'Encodings'
                          AND COLUMN_NAME IN ('nose_x', 'nose_y', 'face_height')
                    """)
                    existing_cols = {row[0] for row in session.execute(col_check_sql)}
                except Exception as e:
                    print(f"Error checking Encodings schema: {e}")
                    existing_cols = set()

                required_cols = {'nose_x', 'nose_y', 'face_height'}
                if required_cols.issubset(existing_cols):
                    update_sql = text("""
                        UPDATE Encodings
                        SET nose_x = :nose_x,
                            nose_y = :nose_y,
                            face_height = :face_height
                        WHERE image_id = :image_id
                    """)
                    for idx, row in face_geometry_df.iterrows():
                        try:
                            session.execute(update_sql, {
                                'image_id': row['image_id'],
                                'nose_x': row['nose_x'],
                                'nose_y': row['nose_y'],
                                'face_height': row['face_height']
                            })
                            updated_count += 1
                        except Exception as e:
                            print(f"Error updating Encodings for image_id {row['image_id']}: {e}")
                    session.commit()
                    print(f"✓ Updated Encodings table with face geometry for {updated_count} images")
                else:
                    missing_cols = required_cols - existing_cols
                    print(f"Skipping Encodings geometry update; missing columns: {sorted(missing_cols)}")

                # Save ImagesDetections table only for newly processed IDs
                if 'newly_processed_detection' in df_with_geometry.columns:
                    df_new = df_with_geometry[df_with_geometry['newly_processed_detection'] == True].copy()
                    print(f"[DEBUG] Newly processed rows to persist: {len(df_new)}")
                    if len(df_new) > 0:
                        save_images_detections(df_new, engine)
                    else:
                        print("[DEBUG] No newly processed rows. Skipping ImagesDetections save.")
                else:
                    print("[DEBUG] newly_processed_detection column missing; defaulting to full save.")
                    save_images_detections(df_with_geometry, engine)
            except Exception as e:
                print(f"Error in ObjectFusion post-processing: {e}")
        
        # save_images_clusters_DB(df_columnar)
        print("saved segment to clusters")

    # create_my_engine(db)
    global N_CLUSTERS
    if META:
        print("making meta clusters from existing clusters")
        # convert MEDIAN_DICT to a dataframe
        enc_data = pd.DataFrame.from_dict(MEDIAN_DICT, orient='index')
        enc_data.reset_index(inplace=True)
        # this is where the weird cluster_id column comes from. could remove this here and the correction elsewhere
        enc_data.rename(columns={'index': 'cluster_id'}, inplace=True)

    else:
        print("about to SQL: ",SELECT,FROM,WHERE,LIMIT)
        resultsjson = selectSQL()
        if resultsjson is None or len(resultsjson) == 0:
            print("No results found from SQL query.")
            return False
        print("got results, count is: ",len(resultsjson))
        enc_data=pd.DataFrame()
        df = pd.json_normalize(resultsjson)
        print(df)


        # tell sort_pose which columns to NOT query
        if cl.CLUSTER_TYPE in ("BodyPoses", "BodyPoses3D", "ArmsPoses3D"): io.query_face = sort.query_face = io.query_hands = sort.query_hands = False
        elif cl.CLUSTER_TYPE == "HandsGestures": io.query_body = sort.query_body = io.query_face = sort.query_face = False
        elif cl.CLUSTER_TYPE == "Clusters": io.query_body = sort.query_body = io.query_hands = sort.query_hands = False
        if not USE_HEAD_POSE: io.query_head_pose = sort.query_head_pose = False
        
        if cl.CLUSTER_TYPE != "HSV":
            # hsv does not need any encodings from mongo
            # Use interleaved batch processing: MongoDB + MySQL per batch to prevent connection timeouts
            print("\n=== Starting interleaved batch processing (MongoDB + MySQL per batch) ===")
            all_prepared_batches = []
            total_rows = len(df)
            
            for batch_start in range(0, total_rows, 5000):
                batch_end = min(batch_start + 5000, total_rows)
                batch_num = batch_start // 5000 + 1
                
                print(f"\n=== Processing Batch {batch_num}: rows {batch_start}-{batch_end} ({batch_end - batch_start} rows) ===")
                batch_df = df.iloc[batch_start:batch_end].copy()
                
                # Interleaved: MongoDB + MySQL for this batch
                batch_prepared = fetch_and_prepare_batch(batch_df, batch_num)
                
                all_prepared_batches.append(batch_prepared)
                gc.collect()
                
                print(f"[Batch {batch_num}] Complete. Total batches collected: {len(all_prepared_batches)}")
            
            # Combine all prepared batches into one dataset
            print("\n=== Combining all batches for clustering ===")
            enc_data = pd.concat(all_prepared_batches, ignore_index=True)
            print(f"Total prepared rows: {len(enc_data)}")
        else:
            # HSV doesn't need MongoDB encodings, just prepare directly
            enc_data = prepare_df(df)
    
    # choose if you want optimal cluster size or custom cluster size using the parameter GET_OPTIMAL_CLUSTERS
    if MODE == 0:
        if GET_OPTIMAL_CLUSTERS is True: 
            # this is mostly defunct/deprecated. would need refactoring to work with norm/3D/etc
            OPTIMAL_CLUSTERS = best_score(enc_data.drop(["image_id", "body_landmarks_array"], axis=1))   #### Input ONLY encodings into clustering algorithm
            print(OPTIMAL_CLUSTERS)
            N_CLUSTERS = OPTIMAL_CLUSTERS
        print("enc_data", enc_data)
        

        # if body_landmarks_array is one of the df.columns, drop it
        # don't need to write a CSV
        # enc_data.to_csv('clusters_clusterID_byImageID.csv')

        # if USE_SEGMENT:
        #     Base.metadata.create_all(engine)
        
        # Process in batches if dataset is large to avoid crashing the shell
        total_rows = len(enc_data)
        calculate_clusters_and_save(enc_data)
        
    elif MODE == 1:
        total_rows = len(enc_data)
        if total_rows <= BATCH_LIMIT:
            assign_images_clusters_DB(enc_data)
        else:
            print(f"Large dataset ({total_rows} rows) — processing in batches of {BATCH_LIMIT}")
            for start in range(0, total_rows, BATCH_LIMIT):
                end = min(start + BATCH_LIMIT, total_rows)
                print(f"Processing batch rows {start} to {end}...")
                batch_df = enc_data.iloc[start:end].copy()
                assign_images_clusters_DB(batch_df)
                # Free memory between batches
                gc.collect()

        # assign_images_clusters_DB(enc_data)
        print("assigned and saved segment to clusters")

    elif MODE == 2:
        # reprocesses the data to get the cluster medians
        print("reprcessing data to get cluster medians and distances")
        print(enc_data)

        # if any values in the cluster_median are None or NULL
        if df['cluster_median'].isnull().values.any():
            print("Cluster median contains NULL values, recalculating cluster medians.")
            median_dict = calculate_cluster_medians(enc_data)
            save_clusters_DB(median_dict, update=True)
            #assign median_dict to MEDIAN_DICT for global use in assigning clusters
            MEDIAN_DICT = median_dict
        else:
            print("Cluster median contains valid values, skipping recalculation.")

        assign_images_clusters_DB(enc_data)

        # median_dict = calculate_cluster_medians(enc_data)
        # save_clusters_DB(median_dict, update=True)
        print("assigned and saved segment to clusters")
    elif MODE == 3:
        # make meta clusters using MEDIAN_DICT
        calculate_clusters_and_save(enc_data)
        print("made and saved meta clusters")


    end = time.time()
    print (end - start)
    return True

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    