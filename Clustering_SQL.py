#sklearn imports
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

image_edge_multiplier_sm = [2.2, 2.2, 2.6, 2.2] # standard portrait
face_height_output = 500
motion = {"side_to_side": False, "forward_smile": True, "laugh": False, "forward_nosmile":  False, "static_pose":  False, "simple": False}
EXPAND = False
ONE_SHOT = False # take all files, based off the very first sort order.
JUMP_SHOT = False # jump to random file if can't find a run
sort = SortPose(motion, face_height_output, image_edge_multiplier_sm,EXPAND, ONE_SHOT, JUMP_SHOT, None, None, use_3D=True)
# MM you need to use conda activate mps_torch310 

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

start = time.time()

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
options = ['kmeans cluster and save clusters', 'cluster assignment', 'calculate cluster medians and save clusters']
option, MODE = pick(options, title)
# MODE = 1
# CLUSTER_TYPE = "Clusters"
# CLUSTER_TYPE = "BodyPoses"
# CLUSTER_TYPE = "HandsPositions"
CLUSTER_TYPE = "HandsGestures"
# CLUSTER_TYPE = "FingertipsPositions"
sort.set_subset_landmarks(CLUSTER_TYPE)
SUBSELECT_ONE_CLUSTER = 0

# SUBSET_LANDMARKS is now set in sort pose init
USE_HEAD_POSE = False

SHORTRANGE = True
ANGLES = []
STRUCTURE = "list3"
print("STRUCTURE: ",STRUCTURE)
if STRUCTURE == "list3": 
    print("setting 3D to True")
    sort.use_3D = True

# this works for using segment in stock, and for ministock
USE_SEGMENT = True

# get the best fit for clusters
GET_OPTIMAL_CLUSTERS=False

# number of clusters produced. run GET_OPTIMAL_CLUSTERS and add that number here
# 24 for body poses
# 128 for hands 
N_CLUSTERS = 64
SAVE_FIG=False ##### option for saving the visualized data

if USE_SEGMENT is True and (CLUSTER_TYPE != "Clusters"):
    print("setting Poses SQL")
    SegmentTable_name = 'SegmentOct20'

    # 3.8 M large table (for Topic Model)
    # HelperTable_name = "SegmentHelperMar23_headon"

    ######################################################
    #TK need to rework this query for CLUSTER_TYPE
    ###################################################### 

    # Basic Query, this works with gettytest3
    SELECT = "DISTINCT(s.image_id), s.face_x, s.face_y, s.face_z, s.mouth_gap"
    if CLUSTER_TYPE == "BodyPoses": WHERE = " s.mongo_body_landmarks = 1 "
    elif CLUSTER_TYPE == "HandsGestures": WHERE = " s.mongo_hand_landmarks = 1 "
    elif CLUSTER_TYPE in ["HandsPositions","FingertipsPositions"] : WHERE = " s.mongo_hand_landmarks_norm = 1 "
    WHERE += " AND s.is_dupe_of IS NULL "
    if MODE == 0:
        FROM = f"{SegmentTable_name} s"
        if SHORTRANGE: WHERE += " AND s.face_x > -35 AND s.face_x < -24 AND s.face_y > -3 AND s.face_y < 3 AND s.face_z > -3 AND s.face_z < 3 "
    # FROM += f" INNER JOIN Encodings h ON h.image_id = s.image_id " 
    # FROM += f" INNER JOIN {HelperTable_name} h ON h.image_id = s.image_id " 
        if SUBSELECT_ONE_CLUSTER:
            if CLUSTER_TYPE == "HandsGestures": subselect_cluster = "ImagesHandsPositions"
            elif CLUSTER_TYPE == "HandsPositions": subselect_cluster = "ImagesHandsGestures"
            FROM += f" INNER JOIN {subselect_cluster} sc ON sc.image_id = s.image_id " 
            WHERE += f" AND sc.cluster_id = {SUBSELECT_ONE_CLUSTER} "
    elif MODE in (1,2):
        FROM = f"{SegmentTable_name} s LEFT JOIN Images{CLUSTER_TYPE} ic ON s.image_id = ic.image_id"
        if MODE == 1: 
            WHERE += " AND ic.cluster_id IS NULL "
        elif MODE == 2: 
            FROM += f" INNER JOIN {CLUSTER_TYPE} c ON c.cluster_id = ic.cluster_id"
            SELECT += ",ic.cluster_id, ic.cluster_dist, c.cluster_median"
            WHERE += " AND ic.cluster_id IS NOT NULL "

    # WHERE += " AND h.is_body = 1"
    LIMIT = 5000000


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

    # Basic Query, this works with gettytest3
    SELECT = "DISTINCT(s.image_id)"
    FROM = f"{SegmentTable_name} s"
    FROM += f" INNER JOIN {HelperTable_name} h ON h.image_id = s.image_id " 
    WHERE = " s.mongo_body_landmarks = 1"
    # WHERE = "face_encodings68 IS NOT NULL AND face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2"
    LIMIT = 100


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
    FROM = f"{SegmentTable_name} s LEFT JOIN Images{CLUSTER_TYPE} ic ON s.image_id = ic.image_id"
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
    cluster_dist = Column(Float)

def get_cluster_medians():


    # store the results in a dictionary where the key is the cluster_id
    # if results:
    #     cluster_medians = {}
    #     for i, row in enumerate(results, start=1):
    #         cluster_median = pickle.loads(row.cluster_median)
    #         cluster_medians[i] = cluster_median
    #         # print("cluster_medians", i, cluster_median)
    #         N_CLUSTERS = i # will be the last cluster_id which is count of clusters

    ####################################
    # this has a LIMIT on it, that I didn't see before
    # hmm... wait, that just limits to 1000 clusters. I don't have that many
    ####################################

    # Create a SQLAlchemy select statement
    select_query = select(Clusters.cluster_id, Clusters.cluster_median).limit(1000)

    # Execute the query using your SQLAlchemy session
    results = session.execute(select_query)
    median_dict = {}

    # Process the results as needed
    for row in results:
        # print(row)
        cluster_id, cluster_median_pickle = row
        # print("cluster_id: ",cluster_id)

        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        # print the pickle using pprint
        # pp.pprint(cluster_median_pickle)

        cluster_median = pickle.loads(cluster_median_pickle)
        # print("cluster_median: ", cluster_id, cluster_median)
        # Check the type and content of the deserialized object
        if not isinstance(cluster_median, np.ndarray):
            print(f"Deserialized object is not a numpy array, it's of type: {type(cluster_median)}")
        
        if sort.SUBSET_LANDMARKS:
            # handles body lms subsets
            subset_cluster_median = []
            for i in range(len(cluster_median)):
                if i in sort.SUBSET_LANDMARKS:
                    subset_cluster_median.append(cluster_median[i])
            cluster_median = subset_cluster_median
        median_dict[cluster_id] = cluster_median
    # print("median dict: ",median_dict)
    return median_dict 



def selectSQL():
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)};"
    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    return(resultsjson)

def make_subset_landmarks(df,add_list=False):
    numerical_columns = [col for col in df.columns if col.startswith('dim_')]
    # set hand_columns = to the numerical_columns in sort.SUBSET_LANDMARKS
    print("subset df columns: ",df.columns)
    if sort.SUBSET_LANDMARKS:
        subset_columns = [f'dim_{i}' for i in sort.SUBSET_LANDMARKS]
        print("subset columns: ",subset_columns)
        if USE_HEAD_POSE:
            df = df.apply(sort.weight_face_pose, axis=1)
            head_columns = ['face_x', 'face_y', 'face_z', 'mouth_gap']
            subset_columns += head_columns
    else:
        subset_columns = numerical_columns
    if add_list:
        numerical_data = df
        numerical_data["obj_bbox_list"] = df[subset_columns].values.tolist()
    else:
        numerical_data = df[subset_columns]
    return numerical_data

def kmeans_cluster(df, n_clusters=32):
    # Select only the numerical columns (dim_0 to dim_65)
    print(" : ",sort.SUBSET_LANDMARKS)
    if CLUSTER_TYPE == "BodyPoses":
        print("CLUSTER_TYPE == BodyPoses", df)
        numerical_data = make_subset_landmarks(df)
    else:
        numerical_data = df
    print("clustering subset data", numerical_data)
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
    # print(f"Cluster {cluster_id} data: {cluster_df}")
    # Convert the selected dimensions into a NumPy array
    cluster_points = cluster_df[col_list].values
    
    # Check if there are valid points in the cluster
    if len(cluster_points) == 0 or np.isnan(cluster_points).any():
        print(f"No valid points for cluster {cluster_id}, skipping median calculation.")
        return None
    
    # print(f"Cluster {cluster_id} points: {cluster_points}")

    # Calculate the geometric median for the cluster points
    cluster_median = geometric_median(cluster_points)
    
    return cluster_median

def build_col_list(df):
    col_list = {}
    col_list["left"] = [col for col in df.columns if col.startswith('left_dim_')]
    col_list["right"] = [col for col in df.columns if col.startswith('right_dim_')]
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

    # print(f"Columns used for median calculation: {col_list}")
    # print(f"All DataFrame columns: {df.columns}")

    # print(df)
    unique_clusters = set(df['cluster_id'])
    for cluster_id in unique_clusters:
        # cluster_median = calc_cluster_median(df, col_list, cluster_id)
        cluster_median = {}
        cluster_median["left"] = calc_cluster_median(df, col_list["left"], cluster_id)
        cluster_median["right"] = calc_cluster_median(df, col_list["right"], cluster_id)
        if cluster_median is not None:
            print(f"Recalculated median for cluster {cluster_id}: {cluster_median}")
            # if every value in the cluster median < .01, then set all values to 0.0
            cluster_median = zero_out_medians(cluster_median)
            # add left and right hands
            flattened_median = np.concatenate((cluster_median["left"], cluster_median["right"]))
            print(f"compressed cluster_median for {cluster_id}: {flattened_median}")
            median_dict[cluster_id] = flattened_median
        else:
            print(f"Cluster {cluster_id} has no valid points or data.")
    return median_dict




def save_clusters_DB(median_dict, update=False):
    # Convert to set and Save the df to a table
    
    print("save_clusters_DB median_dict", median_dict)
    cluster_ids = median_dict.keys()
    print("save_clusters_DB cluster_ids", cluster_ids)
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
        existing_record = session.query(Clusters).filter_by(cluster_id=cluster_id).first()

        if existing_record is None:
            # Save the geometric median into the database
            print(f"Saving new record with cluster_id {cluster_id}")
            instance = Clusters(
                cluster_id=cluster_id,
                cluster_median=pickle.dumps(cluster_median)  # Serialize the geometric median
            )
            session.add(instance)
            session.flush()  # Force database insertion to catch issues early
            if cluster_id == 0:

                saved_record = session.query(Clusters).filter_by(cluster_id=0).first()
                if saved_record:
                    print(f"Successfully saved cluster_id 0: {saved_record}")
                else:
                    print("Failed to save cluster_id 0.")
        if existing_record is not None and update:
            print(f"Updating existing record with cluster_id {cluster_id} and median {cluster_median}")
            existing_record.cluster_median = pickle.dumps(cluster_median)
            
        else:
            print(f"Skipping duplicate record with cluster_id {cluster_id}")

    try:
        session.commit()
        print("Data saved successfully.")
    except IntegrityError as e:
        session.rollback()
        print(f"Error occurred during data saving: {str(e)}")


def save_images_clusters_DB(df):
    #save the df to a table
    for _, row in df.iterrows():
        image_id = row['image_id']
        cluster_id = row['cluster_id']
        cluster_dist = row['cluster_dist']
        existing_record = session.query(ImagesClusters).filter_by(image_id=image_id).first()

        if existing_record is None:
            # it may be easier to define this locally, and assign the name via CLUSTER_TYPE
            instance = ImagesClusters(
                image_id=image_id,
                cluster_id=cluster_id,
                cluster_dist=cluster_dist
            )
            session.add(instance)
        
        elif existing_record is not None:
            if existing_record.cluster_dist is None:
                if image_id % 100 == 0:
                    print(f"Updating existing record with image_id {image_id} to cluster_dist {cluster_dist}")
                existing_record.cluster_dist = cluster_dist
            else:
                print(f"Skipping existing record with image_id {image_id} and cluster_dist {cluster_dist}")
        else:
            print(f"Skipping duplicate record with image_id {image_id}")

    try:
        session.commit()
        print("Data saved successfully.")
    except IntegrityError as e:
        session.rollback()
        print(f"Error occurred during data saving: {str(e)}")


def assign_images_clusters_DB(df):
    def prep_pose_clusters_enc(enc1):
        # print("current image enc1", enc1)  
        enc1 = np.array(enc1)
        this_dist_dict = {}
        for cluster_id in MEDIAN_DICT:
            enc2 = MEDIAN_DICT[cluster_id]
            # print("cluster_id enc2: ", cluster_id,enc2)
            this_dist_dict[cluster_id] = np.linalg.norm(enc1 - enc2, axis=0)
        
        cluster_id, cluster_dist = min(this_dist_dict.items(), key=lambda x: x[1])

        # print(cluster_id)
        return cluster_id, cluster_dist

    def calc_median_dist(enc1, enc2):
        return np.linalg.norm(enc1 - enc2, axis=0)
    
    #assign clusters to each image's encodings
    print("assigning images to clusters, df at start",df)
    df_subset_landmarks = make_subset_landmarks(df, add_list=True)


    if CLUSTER_TYPE in ["BodyPoses","HandsGestures", "HandsPositions","FingertipsPositions"]:
        # combine all columns that start with left_dim_ or right_dim_ or dim_ into one list in the "enc1" column

        # Step 1: Identify columns that contain "_dim_"
        dim_columns = [col for col in df_subset_landmarks.columns if "_dim_" in col]

        # Step 2: Combine values from these columns into a list for each row
        df_subset_landmarks['enc1'] = df_subset_landmarks[dim_columns].values.tolist()

        # Step 3: Print the result to check
        print("df_subset_landmarks", df_subset_landmarks[['image_id', 'enc1']])

        if df['cluster_id'].isnull().values.any():
            # df_subset_landmarks["cluster_id"], df_subset_landmarks["cluster_dist"] = zip(*df_subset_landmarks["enc1"].apply(prep_pose_clusters_enc))
            df_subset_landmarks.loc[df_subset_landmarks['cluster_id'].isnull(), ['cluster_id', 'cluster_dist']] = \
                zip(*df_subset_landmarks.loc[df_subset_landmarks['cluster_id'].isnull(), 'enc1'].apply(prep_pose_clusters_enc))
        else:
            # apply calc_median_dist to enc1 and cluster_median
            df_subset_landmarks["cluster_dist"] = df_subset_landmarks.apply(lambda row: calc_median_dist(row['enc1'], row['cluster_median']), axis=1)

    else:
        # this is for obj_bbox_list
        df_subset_landmarks["cluster_id"] = df_subset_landmarks["obj_bbox_list"].apply(prep_pose_clusters_enc)

    print("df_subset_landmarks clustered after apply")
    print(df_subset_landmarks[["image_id", "cluster_id","cluster_dist"]])

    # print all rows where cluster_id is 68
    # print(df_subset_landmarks[df_subset_landmarks["cluster_id"] == 68])

    save_images_clusters_DB(df_subset_landmarks)
    print ("saved to imagesclusters")

def df_list_to_cols(df, col_name):

    # Convert the string representation of lists to actual lists
    # df[col_name] = df[col_name].apply(eval)
    df_data = df.drop("image_id", axis=1)
    # Create new columns for each coordinate
    num_coords = len(df_data[col_name].iloc[0])
    for i in range(num_coords):
        df[f'dim_{i}'] = df[col_name].apply(lambda x: x[i])

    # Drop the original col_name column
    df = df.drop(col_name, axis=1)
    return df

def prepare_df(df):
    # apply io.convert_decimals_to_float to face_x, face_y, face_z, and mouth_gap 
    # if faxe_x, face_y, face_z, and mouth_gap are not already floats
    if df['face_x'].dtype != float:
        df[['face_x', 'face_y', 'face_z', 'mouth_gap']] = df[['face_x', 'face_y', 'face_z', 'mouth_gap']].astype(float)
    # if cluster_median column is in the df, unpickle it
    if 'cluster_median' in df.columns:
        df['cluster_median'] = df['cluster_median'].apply(io.unpickle_array)
    if CLUSTER_TYPE == "BodyPoses":
        df = df.dropna(subset=['body_landmarks_normalized'])
        df['body_landmarks_normalized'] = df['body_landmarks_normalized'].apply(io.unpickle_array)
        # body = self.get_landmarks_2d(enc1, list(range(33)), structure)
        df['body_landmarks_array'] = df['body_landmarks_normalized'].apply(lambda x: sort.get_landmarks_2d(x, list(range(33)), structure=STRUCTURE))

        # apply io.convert_decimals_to_float to face_x, face_y, face_z, and mouth_gap 
        # df['body_landmarks_array'] = df.apply(lambda row: io.convert_decimals_to_float(row['body_landmarks_array'] + [row['face_x'], row['face_y'], row['face_z'], row['mouth_gap']]), axis=1)
        # drop the columns that are not needed
        # if not USE_HEAD_POSE: df = df.drop(columns=['face_x', 'face_y', 'face_z', 'mouth_gap']) 
        columns_to_drop=['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized']
        print("before cols",df)

        df_list_to_cols(df, 'body_landmarks_array')
        print("after cols",df)
    # elif CLUSTER_TYPE == "HandsPositions":
    elif CLUSTER_TYPE in ["HandsPositions","FingertipsPositions"]:
        print("first row of df",df.iloc[0])
        df[['left_hand_landmarks', 'left_hand_world_landmarks', 'left_hand_landmarks_norm', 'right_hand_landmarks', 'right_hand_world_landmarks', 'right_hand_landmarks_norm']] = pd.DataFrame(df['hand_results'].apply(sort.prep_hand_landmarks).tolist(), index=df.index)
        print("after prep",df)
        df = sort.split_landmarks_to_columns(df, left_col="left_hand_landmarks_norm", right_col="right_hand_landmarks_norm")
        print("after split",df)
        columns_to_drop = ['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized', 
                           'hand_results', 'left_hand_landmarks', 'right_hand_landmarks', 
                           'left_hand_world_landmarks', 'right_hand_world_landmarks',
                           'left_hand_landmarks_norm', 'right_hand_landmarks_norm']

    elif CLUSTER_TYPE == "HandsGestures":
        df[['left_hand_landmarks', 'left_hand_world_landmarks', 'left_hand_landmarks_norm', 'right_hand_landmarks', 'right_hand_world_landmarks', 'right_hand_landmarks_norm']] = pd.DataFrame(df['hand_results'].apply(sort.prep_hand_landmarks).tolist(), index=df.index)
        df = sort.split_landmarks_to_columns(df, left_col="left_hand_world_landmarks", right_col="right_hand_world_landmarks")
        # drop the columns that are not needed
        columns_to_drop = ['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized', 
                           'hand_results', 'left_hand_landmarks', 'right_hand_landmarks', 'left_hand_world_landmarks', 'right_hand_world_landmarks']

    elif CLUSTER_TYPE == "Clusters":
        df = df.dropna(subset=['face_encodings68'])

        # Apply the unpickling function to the 'face_encodings' column
        df['face_encodings68'] = df['face_encodings68'].apply(io.unpickle_array)
        df['face_landmarks'] = df['face_landmarks'].apply(io.unpickle_array)
        df['body_landmarks'] = df['body_landmarks'].apply(io.unpickle_array)
        columns_to_drop=['face_landmarks', 'body_landmarks', 'body_landmarks_normalized']
        df_list_to_cols(df, 'face_encodings68')
    if not USE_HEAD_POSE: 
        # add 'face_x', 'face_y', 'face_z', 'mouth_gap' to existing columns_to_drop
        columns_to_drop += ['face_x', 'face_y', 'face_z', 'mouth_gap']
    df = df.drop(columns=columns_to_drop)
    return df

# defining globally 
MEDIAN_DICT = get_cluster_medians()

def main():
    # create_my_engine(db)
    global N_CLUSTERS
    print("about to SQL: ",SELECT,FROM,WHERE,LIMIT)
    resultsjson = selectSQL()
    print("got results, count is: ",len(resultsjson))
    enc_data=pd.DataFrame()
    df = pd.json_normalize(resultsjson)
    print(df)
    # tell sort_pose which columns to NOT query
    if CLUSTER_TYPE == "BodyPoses": io.query_face = sort.query_face = io.query_hands = sort.query_hands = False
    elif CLUSTER_TYPE == "HandsGestures": io.query_body = sort.query_body = io.query_face = sort.query_face = False
    elif CLUSTER_TYPE == "Clusters": io.query_body = sort.query_body = io.query_hands = sort.query_hands = False
    if not USE_HEAD_POSE: io.query_head_pose = sort.query_head_pose = False
    df[['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized', 'hand_results']] = df['image_id'].apply(io.get_encodings_mongo)
    # face_encodings68, face_landmarks, body_landmarks, body_landmarks_normalized = sort.get_encodings_mongo(mongo_db,row["image_id"], is_body=True, is_face=False)
    enc_data = prepare_df(df)
    
    # choose if you want optimal cluster size or custom cluster size using the parameter GET_OPTIMAL_CLUSTERS
    if MODE == 0:
        if GET_OPTIMAL_CLUSTERS is True: 
            OPTIMAL_CLUSTERS = best_score(enc_data.drop(["image_id", "body_landmarks_array"], axis=1))   #### Input ONLY encodings into clustering algorithm
            print(OPTIMAL_CLUSTERS)
            N_CLUSTERS = OPTIMAL_CLUSTERS
        print(enc_data)
        

        # I drop image_id as I pass it to knn bc I need it later, but knn can't handle strings
        # if body_landmarks_array is one of the df.columns, drop it
        print("df columns: ",enc_data.columns)
        columns_to_drop = ["image_id"]
        if "body_landmarks_array" in enc_data.columns: columns_to_drop.append("body_landmarks_array")
        if "left_hand_landmarks_norm" in enc_data.columns: columns_to_drop.append("left_hand_landmarks_norm")
        if "right_hand_landmarks_norm" in enc_data.columns: columns_to_drop.append("right_hand_landmarks_norm")
        print("columns to drop: ",columns_to_drop)
        enc_data["cluster_id"] = kmeans_cluster(enc_data.drop(columns=columns_to_drop), n_clusters=N_CLUSTERS)
        
        print(enc_data)
        print(set(enc_data["cluster_id"].tolist()))
        # don't need to write a CSV
        # enc_data.to_csv('clusters_clusterID_byImageID.csv')

        # if USE_SEGMENT:
        #     Base.metadata.create_all(engine)
        median_dict = calculate_cluster_medians(enc_data)
        save_clusters_DB(median_dict)
        save_images_clusters_DB(enc_data)
        print("saved segment to clusters")
    elif MODE == 1:

        assign_images_clusters_DB(enc_data)
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

    end = time.time()
    print (end - start)
    return True

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    