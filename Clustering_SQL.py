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
from my_declarative_base import Base, Images, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, ForeignKey

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
CLUSTER_TYPE = "ArmsPoses3D" 
# CLUSTER_TYPE = "HandsPositions"
# CLUSTER_TYPE = "HandsGestures"
# CLUSTER_TYPE = "FingertipsPositions"
# CLUSTER_TYPE = "HSV" # only works with cluster save, not with assignment
VERBOSE = True
cl = ToolsClustering(CLUSTER_TYPE, VERBOSE=VERBOSE)

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
# START_ID = 114468990 # only used in MODE 1
START_ID = 0 # only used in MODE 1

# WHICH TABLE TO USE?
# SegmentTable_name = 'SegmentOct20'
SegmentTable_name = 'SegmentBig_isface'
# SegmentTable_name = 'SegmentBig_isnotface'

# if doing MODE == 2, use SegmentHelper_name to subselect SQL query
# unless you know what you are doing, leave this as None
SegmentHelper_name = None
# if cl.CLUSTER_TYPE == "ArmsPoses3D":
# SegmentHelper_name = 'SegmentHelper_sept2025_heft_keywords'
SegmentHelper_name = 'SegmentHelper_oct2025_evens'
# SegmentHelper_name = 'SegmentHelper_nov2025_placard'

# number of clusters produced. run GET_OPTIMAL_CLUSTERS and add that number here
# 32 for hand positions
# 128 for hand gestures
N_CLUSTERS = 128
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
    # Basic Query, this works with SegmentOct20. Previously included s.face_x, s.face_y, s.face_z, s.mouth_gap
    SELECT = "DISTINCT(s.image_id)"
    WHERE = f" {dupe_table_pre}.is_dupe_of IS NULL "

    if isinstance(this_data_column, list):
        if "HSV" in cl.CLUSTER_TYPE:
            SELECT = SELECT.replace(f"s.face_x, s.face_y, s.face_z, s.mouth_gap", f"ib.hue, ib.sat, ib.val ")
            FROM += f" JOIN ImagesBackground ib ON s.image_id = ib.image_id "
            WHERE += f" AND ib.hue IS NOT NULL AND ib.sat IS NOT NULL AND ib.val IS NOT NULL "
    else:
        WHERE += f" AND {dupe_table_pre}.{this_data_column} = 1 "

    if cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["is_feet"] is not None:
        WHERE += f" AND {dupe_table_pre}.is_feet = {cl.CLUSTER_DATA[cl.CLUSTER_TYPE]['is_feet']} "
    if cl.CLUSTER_DATA[cl.CLUSTER_TYPE].get("mongo_hand_landmarks") is not None:
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
    LIMIT = 50000000
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
    ), poolclass=NullPool)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# TK 4 HSV
Clusters, ImagesClusters, MetaClusters, ClustersMetaClusters = cl.construct_table_classes(table_cluster_type)

def selectSQL():
    if OFFSET: offset = f" OFFSET {str(OFFSET)}"
    else: offset = ""
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)} {offset};"
    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    return(resultsjson)

def make_subset_landmarks(df,add_list=False):
    first_col = df.columns[1]
    print("first col: ",first_col)
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
        numerical_data = df[['image_id'] + subset_columns]
    else:
        numerical_data = df[subset_columns]

    if add_list:
        print("make_subset_landmarks adding obj_bbox_list column")
        numerical_data["obj_bbox_list"] = df[subset_columns].values.tolist()

    # this is the old way before Arms3D subsetting. I'm not sure if there is an actual scenario where i want to assign the whole df
    # if add_list:
    #     print("make_subset_landmarks adding obj_bbox_list column")
    #     numerical_data = df
    #     numerical_data["obj_bbox_list"] = df[subset_columns].values.tolist()
    # else:
    #     numerical_data = df[subset_columns]

    print("make_subset_landmarks at the end these are the columns: ", numerical_data.columns)
    return numerical_data

def kmeans_cluster(df, n_clusters=32):
    # Select only the numerical columns (dim_0 to dim_65)
    print("kmeans_cluster sort.SUBSET_LANDMARKS: ",sort.SUBSET_LANDMARKS)
    if cl.CLUSTER_TYPE in ["BodyPoses", "BodyPoses3D", "ArmsPoses3D"]:
        print("cl.CLUSTER_TYPE == BodyPoses || ArmsPoses3D", df)
        numerical_data = make_subset_landmarks(df)
    else:
        numerical_data = df
    print("clustering subset data", numerical_data)
    print("frist row of numerical data: ", numerical_data.iloc[0])


    kmeans = KMeans(n_clusters=n_clusters, n_init=10, init='k-means++', random_state=42, max_iter=300, verbose=1)
    kmeans.fit(numerical_data)
    clusters = kmeans.predict(numerical_data)
    return clusters
    
def best_score(df):
    print("starting best score", df)
    print("about to subset landmarks to thse columns: ",sort.SUBSET_LANDMARKS)
    df = make_subset_landmarks(df)
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
    # if META:
    #     # for some reason the meta clusters have the cluster_id in the first column
    #     # need to remove it
    #     if len(col_list)/len(sort.SUBSET_LANDMARKS) % 1 != 0:
    #         print(" \/\/\/\/ imbalance in col_list and subset_landmarks")
    #         if (len(col_list)-1)/len(sort.SUBSET_LANDMARKS) % 1 == 0:
    #             # if removing one column makes it balanced, then do that

    #             # this is a hacky attempt to handle weird scenario when making metaclusters
    #             # it reads in the cluster_id from mysql into df, etc. 
    #             print('remove cluster_id column from cluster_df and col_list', col_list)
    #             print(len(col_list))
    #             col_list.remove(col_list[0])
    #             print('removeDDDDD cluster_id column from cluster_df and col_list', col_list)
    #             print(len(col_list))
    #             cluster_df = cluster_df.drop(columns=['cluster_id'])
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
    col_list["left"] = col_list["right"] = col_list["body_lms"] = col_list["face"] = []
    if "body" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]:
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
    print(f"All DataFrame columns: {df.columns}")

    print(df)
    unique_clusters = set(df['cluster_id'])
    for cluster_id in unique_clusters:
        # cluster_median = calc_cluster_median(df, col_list, cluster_id)
        cluster_median = {}
        if "body" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]:
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
            if "hand" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]:
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
    print("process_landmarks_cluster_dist first col: ",first_col)
    # if the first column is an int, then the columns are integers
    if isinstance(first_col, int):
        dim_columns = [col for col in df_subset_landmarks.columns if isinstance(col, int)]
    elif any(isinstance(col, str) and col.startswith('dim_') for col in df_subset_landmarks.columns):
    # Step 1: Identify columns that contain "_dim_"
        dim_columns = [col for col in df_subset_landmarks.columns if "dim_" in col]
    elif cl.CLUSTER_TYPE == "HSV":
        dim_columns = cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]
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
    df_subset_landmarks = make_subset_landmarks(df, add_list=True)
    print("df_subset_landmarks after make_subset_landmarks", df_subset_landmarks)

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

def prepare_df(df):
    print("columns: ",df.columns)
    print("prepare_df df",df)
    print("prepare df first row",df.iloc[0])
    columns_to_drop = []
    # apply io.convert_decimals_to_float to face_x, face_y, face_z, and mouth_gap 
    # if faxe_x, face_y, face_z, and mouth_gap are not already floats
    if 'face_x' in df.columns and df['face_x'].dtype != float:
        df[['face_x', 'face_y', 'face_z', 'mouth_gap']] = df[['face_x', 'face_y', 'face_z', 'mouth_gap']].astype(float)
    # if cluster_median column is in the df, unpickle it
    if 'cluster_median' in df.columns:
        df['cluster_median'] = df['cluster_median'].apply(io.unpickle_array)
    if "body" in cl.CLUSTER_DATA[cl.CLUSTER_TYPE]["data_column"]:
    # if cl.CLUSTER_TYPE in ("BodyPoses", "BodyPoses3D", "ArmsPoses3D"):
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
        df = sort.split_landmarks_to_columns(df, left_col="left_hand_landmarks_norm", right_col="right_hand_landmarks_norm")
        print("after split",df)
        columns_to_drop = ['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized', 
                           'hand_results', 'left_hand_landmarks', 'right_hand_landmarks', 
                           'left_hand_world_landmarks', 'right_hand_world_landmarks',
                           'left_hand_landmarks_norm', 'right_hand_landmarks_norm']

    elif cl.CLUSTER_TYPE == "HandsGestures":
        df[['left_hand_landmarks', 'left_hand_world_landmarks', 'left_hand_landmarks_norm', 'right_hand_landmarks', 'right_hand_world_landmarks', 'right_hand_landmarks_norm']] = pd.DataFrame(df['hand_results'].apply(sort.prep_hand_landmarks).tolist(), index=df.index)
        df = sort.split_landmarks_to_columns(df, left_col="left_hand_world_landmarks", right_col="right_hand_world_landmarks")
        # drop the columns that are not needed
        columns_to_drop = ['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized', 
                           'hand_results', 'left_hand_landmarks', 'right_hand_landmarks', 'left_hand_world_landmarks', 'right_hand_world_landmarks']
    elif cl.CLUSTER_TYPE == "HSV":
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
    if not USE_HEAD_POSE: 
        print("not using head pose, dropping face_x, face_y, face_z, mouth_gap")
        # add 'face_x', 'face_y', 'face_z', 'mouth_gap' to existing columns_to_drop
        if 'face_x' in df.columns:
            columns_to_drop += ['face_x', 'face_y', 'face_z', 'mouth_gap']
    df = df.drop(columns=columns_to_drop)
    # if body_landmarks_array is in the df columns,
    # drop any rows where body_landmarks_array is None or NaN
    if 'body_landmarks_array' in df.columns:
        df = df.dropna(subset=['body_landmarks_array'])

    print("final prepared df len", len(df))
    return df

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
        enc_data["cluster_id"] = kmeans_cluster(enc_data.drop(columns=columns_to_drop), n_clusters=N_CLUSTERS)
        
        print("enc_data", enc_data)
        print("as list", set(enc_data["cluster_id"].tolist()))
        median_dict = calculate_cluster_medians(enc_data)
        save_clusters_DB(median_dict)
        # add the correct median_dict for each cluster_id to the enc_data
        enc_data["cluster_median"] = enc_data["cluster_id"].apply(lambda x: median_dict[x])
        if cl.CLUSTER_TYPE != "HSV":
            # this is specific to lms, not hsv
            df_subset_landmarks = make_subset_landmarks(enc_data, add_list=True)
        else:
            df_subset_landmarks = enc_data
        print("df_subset_landmarks", df_subset_landmarks)
        if not META:
            df_subset_landmarks = process_landmarks_cluster_dist(enc_data,df_subset_landmarks)
            print("df_subset_landmarks after process_landmarks", df_subset_landmarks)

        if len(df_subset_landmarks) <= BATCH_LIMIT:
            save_images_clusters_DB(df_subset_landmarks)
        else:
            print(f"Large dataset ({len(df_subset_landmarks)} rows) — processing in batches of {BATCH_LIMIT}")
            for start in range(0, len(df_subset_landmarks), BATCH_LIMIT):
                end = min(start + BATCH_LIMIT, len(df_subset_landmarks))
                print(f"Processing batch rows {start} to {end}...")
                batch_df = df_subset_landmarks.iloc[start:end].copy()
                save_images_clusters_DB(batch_df)
                # Free memory between batches
                gc.collect()
        # save_images_clusters_DB(df_subset_landmarks)
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
            df[['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized', 'body_landmarks_3D', 'hand_results']] = df['image_id'].apply(io.get_encodings_mongo)
        # face_encodings68, face_landmarks, body_landmarks, body_landmarks_normalized = sort.get_encodings_mongo(mongo_db,row["image_id"], is_body=True, is_face=False)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    