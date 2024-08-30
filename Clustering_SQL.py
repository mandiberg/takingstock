#sklearn imports
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.metrics import silhouette_score

#from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
#from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
#from sklearn.model_selection import ParameterGrid
#from sklearn import metrics

import datetime   ####### for saving cluster analytics
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
# import plotly as py
# import plotly.graph_objs as go


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

import numpy as np
import pandas as pd
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
sort = SortPose(motion, face_height_output, image_edge_multiplier_sm,EXPAND, ONE_SHOT, JUMP_SHOT, None, None)

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
MODE = 1
# CLUSTER_TYPE = "Clusters"
CLUSTER_TYPE = "Poses"
# SUBSET_LANDMARKS is now set in sort pose init

ANGLES = []
STRUCTURE = "list"
# this works for using segment in stock, and for ministock
USE_SEGMENT = True

# get the best fit for clusters
GET_OPTIMAL_CLUSTERS=False

# number of clusters produced. run GET_OPTIMAL_CLUSTERS and add that number here
N_CLUSTERS = 24
SAVE_FIG=False ##### option for saving the visualized data

if USE_SEGMENT is True and CLUSTER_TYPE == "Poses":
    print("setting Poses SQL")
    SegmentTable_name = 'SegmentOct20'

    # 3.8 M large table (for Topic Model)
    # HelperTable_name = "SegmentHelperMar23_headon"

    ######################################################
    #TK need to rework this query for CLUSTER_TYPE
    ###################################################### 

    # Basic Query, this works with gettytest3
    SELECT = "DISTINCT(s.image_id)"
    WHERE = " s.mongo_body_landmarks = 1 "
    if MODE == 0:
        FROM = f"{SegmentTable_name} s"
        WHERE += " AND s.face_x > -35 AND s.face_x < -24 AND s.face_y > -3 AND s.face_y < 3 AND s.face_z > -3 AND s.face_z < 3 "
    # FROM += f" INNER JOIN Encodings h ON h.image_id = s.image_id " 
    # FROM += f" INNER JOIN {HelperTable_name} h ON h.image_id = s.image_id " 
    elif MODE == 1:
        FROM = f"{SegmentTable_name} s LEFT JOIN Images{CLUSTER_TYPE} ic ON s.image_id = ic.image_id"
        WHERE += " AND ic.cluster_id IS NULL "

    # WHERE += " AND h.is_body = 1"
    LIMIT = 5000000


    '''
    Poses
    500 11s
    1000 21s
    2000 43s
    4000 87s
    30000 2553 @ hands elbows x 3d
    40000 2248
    200000 18664 @ 33
    300000 32475 @ 4 x 3d

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
    LIMIT = 1000

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

def get_cluster_medians():


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
        print(row)
        cluster_id, cluster_median = row
        cluster_median = pickle.loads(cluster_median, encoding='latin1')
        if sort.SUBSET_LANDMARKS:
            # handles body lms subsets
            subset_cluster_median = []
            for i in range(len(cluster_median)):
                if i in sort.SUBSET_LANDMARKS:
                    subset_cluster_median.append(cluster_median[i])
            cluster_median = subset_cluster_median
        median_dict[cluster_id] = cluster_median
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
    if sort.SUBSET_LANDMARKS:
        subset_columns = [f'dim_{i}' for i in sort.SUBSET_LANDMARKS]
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
    print("about to subset landmarks to thse columns: ",sort.SUBSET_LANDMARKS)
    numerical_data = make_subset_landmarks(df)
    print("clustering subset data", numerical_data)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, init='k-means++', random_state=42, max_iter=300, verbose=1)
    kmeans.fit(numerical_data)
    clusters = kmeans.predict(numerical_data)
    return clusters
    
def export_html_clusters(enc_data,n_clusters):
    x = datetime.datetime.now()
    d_time=x.strftime("%c").replace(":","-").replace(" ","_")
    title = "Visualizing Clusters in Two Dimensions Using PCA"
    filename="cluster_analytics"+str(d_time)+".html"
    
    class MplColorHelper:     ### making a class using matplotlib to choose colors instead of doing it manually

        def __init__(self, cmap_name, start_val, stop_val):
            self.cmap_name = cmap_name
            self.cmap = plt.get_cmap(cmap_name)
            self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
            self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

        def get_rgb(self, val):
            return self.scalarMap.to_rgba(val)
    COL = MplColorHelper('magma', 0, n_clusters-1)
    c=COL.get_rgb(np.arange(n_clusters))           ####### determining color for each cluster
    
    pca_2d = PCA(n_components=2)


    PCs_2d = pd.DataFrame(pca_2d.fit_transform(enc_data.drop(["cluster_id"], axis=1)))
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]
    plotX = pd.concat([plotX,PCs_2d], axis=1, join='inner')
    data=[]
    for i in range(n_clusters):
        cluster=plotX[plotX["cluster_id"] == i]        
        data.append(go.Scatter(
                        x = cluster["PC1_2d"],
                        y = cluster["PC2_2d"],
                        mode = "markers",
                        name = "Cluster "+str(i),
                        marker = dict(color = 'rgba'+str(tuple(c[i]))),  ## it has to be tuples
                        text = None)
                    )


    layout = dict(title = title,
                  xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
                  yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
                 )

    fig = dict(data = data, layout = layout)
    py.offline.plot(fig, filename=filename, auto_open=False)
    
    return

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
    
# def encodings_split(encodings):
#     col="encodings"
#     col_list=[]
#     for i in range(128):col_list.append(col+str(i))
#     encoding_data=pd.DataFrame({col:[np.array(encodings)]})
#     #splitting the encodings column
#     df = pd.DataFrame(encoding_data["encodings"].tolist(), columns=col_list)
#     return df

def encodings_split(encodings):
    col_list = [f"encodings{i}" for i in range(128)]
    df = pd.DataFrame([encodings], columns=col_list)
    return df

def landmarks_preprocess(landmarks):
    col_list = [f"landmarks{i}" for i in range(33)]
    df = pd.DataFrame([landmarks], columns=col_list)
    return df

def save_clusters_DB(df):
    # col="encodings"
    # col_list=[]
    # for i in range(128):col_list.append(col+str(i))
    col_list = [col for col in df.columns if col.startswith('dim_')]


    # this isn't saving means for each cluster. all cluster_median are the same

    # Convert to set and Save the df to a table
    unique_clusters = set(df['cluster_id'])
    for cluster_id in unique_clusters:
        cluster_df=df[df['cluster_id']==cluster_id]
        cluster_mean=np.array(cluster_df[col_list].mean())
        existing_record = session.query(Clusters).filter_by(cluster_id=cluster_id).first()

        if existing_record is None:
            instance = Clusters(
                cluster_id=cluster_id,
                cluster_median=pickle.dumps(cluster_mean)
            )
            session.add(instance)
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
        existing_record = session.query(ImagesClusters).filter_by(image_id=image_id).first()

        if existing_record is None:
            # it may be easier to define this locally, and assign the name via CLUSTER_TYPE
            instance = ImagesClusters(
                image_id=image_id,
                cluster_id=cluster_id,
            )
            session.add(instance)

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
        # print(enc1)  
        enc1 = np.array(enc1)
        this_dist_dict = {}
        for cluster_id in MEDIAN_DICT:
            enc2 = MEDIAN_DICT[cluster_id]
            this_dist_dict[cluster_id] = np.linalg.norm(enc1 - enc2, axis=0)
        
        cluster_id = min(this_dist_dict, key=this_dist_dict.get)
        # print(cluster_id)
        return cluster_id

    #assign clusters to each image's encodings
    # print("assigning images to clusters, df at start",df)
    df_subset_landmarks = make_subset_landmarks(df, add_list=True)

    df_subset_landmarks["cluster_id"] = df_subset_landmarks["obj_bbox_list"].apply(prep_pose_clusters_enc)
    print("df_subset_landmarks clustered after apply")
    print(df_subset_landmarks[["image_id", "cluster_id"]].head())

    save_images_clusters_DB(df_subset_landmarks)


    return
    for index, row in df.iterrows():
        image_id = row['image_id']        
        if CLUSTER_TYPE == "Poses":
            this_enc = df_subset_landmarks.iloc[index].tolist()
        elif CLUSTER_TYPE == "Clusters":
            this_enc = [row[f'encodings{i}'] for i in range(128)]

        this_dist_dict = {}
        enc1 = np.array(this_enc)
        for cluster_id in median_dict:
            enc2 = median_dict[cluster_id]
            this_dist_dict[cluster_id] = np.linalg.norm(enc1 - enc2, axis=0)
        
        cluster_id = min(this_dist_dict, key=this_dist_dict.get)
        
        if cluster_id:
            print(f"Assigning image_id {image_id} to cluster_id {cluster_id}")
            instance = ImagesClusters(
                image_id=image_id,
                cluster_id=cluster_id,
            )
            session.add(instance)
        else:
            print(f"Something went wrong with image_id {image_id}")

        # Increment the batch counter
        batch_counter += 1

        # If the batch counter reaches the batch size, commit the session
        if batch_counter % batch_size == 0:
            try:
                # session.commit()
                print(f"Batch committed successfully. Processed {batch_counter} rows so far.")
            except IntegrityError as e:
                session.rollback()
                print(f"Error occurred during batch commit: {str(e)}")

    # Commit any remaining records after the loop
    try:
        # session.commit()
        print("Final batch committed successfully.")
    except IntegrityError as e:
        session.rollback()
        print(f"Error occurred during final commit: {str(e)}")
        
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
    if CLUSTER_TYPE == "Poses":
        df = df.dropna(subset=['body_landmarks_normalized'])
        df['body_landmarks_normalized'] = df['body_landmarks_normalized'].apply(io.unpickle_array)
        # body = self.get_landmarks_2d(enc1, list(range(33)), structure)
        df['body_landmarks_array'] = df['body_landmarks_normalized'].apply(lambda x: io.get_landmarks_2d(x, list(range(33)), structure=STRUCTURE))
        # drop the columns that are not needed
        df = df.drop(columns=['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized'])
        print("before cols",df)

        df_list_to_cols(df, 'body_landmarks_array')
        print("after cols",df)

    elif CLUSTER_TYPE == "Clusters":
        df = df.dropna(subset=['face_encodings68'])

        # Apply the unpickling function to the 'face_encodings' column
        df['face_encodings68'] = df['face_encodings68'].apply(io.unpickle_array)
        df['face_landmarks'] = df['face_landmarks'].apply(io.unpickle_array)
        df['body_landmarks'] = df['body_landmarks'].apply(io.unpickle_array)
        df = df.drop(columns=['face_landmarks', 'body_landmarks', 'body_landmarks_normalized'])
        df_list_to_cols(df, 'face_encodings68')

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
    is_body = is_face = True
    df = pd.json_normalize(resultsjson)
    print(df)
    if CLUSTER_TYPE == "Poses": io.query_face = False
    elif CLUSTER_TYPE == "Clusters": io.query_body = False
    df[['face_encodings68', 'face_landmarks', 'body_landmarks', 'body_landmarks_normalized']] = df['image_id'].apply(io.get_encodings_mongo)
    # face_encodings68, face_landmarks, body_landmarks, body_landmarks_normalized = io.get_encodings_mongo(mongo_db,row["image_id"], is_body=True, is_face=False)
    enc_data = prepare_df(df)

    # print(enc_data)

    # for row in resultsjson:
    #     # gets contentUrl
    #     if CLUSTER_TYPE == "Poses":
    #         ##################
    #         #TK need to rework this for CLUSTER_TYPE
    #         # to prep the data properly 128d vs 4-33lms
    #         ##################
    #         print(row)
    #         face_encodings68, face_landmarks, body_landmarks, body_landmarks_normalized = io.get_encodings_mongo(mongo_db,row["image_id"], is_body=True, is_face=False)
    #         print("got encodings from mongo, types are: ", type(face_encodings68), type(face_landmarks), type(body_landmarks_normalized))
    #         quit()
    #         pass
    #     else:
    #         df=encodings_split(pickle.loads(row["face_encodings68"], encoding='latin1'))
    #         df["image_id"]=row["image_id"]
    #         enc_data = pd.concat([enc_data,df],ignore_index=True) 
    
    # choose if you want optimal cluster size or custom cluster size using the parameter GET_OPTIMAL_CLUSTERS
    if MODE == 0:
        if GET_OPTIMAL_CLUSTERS is True: 
            OPTIMAL_CLUSTERS = best_score(enc_data.drop(["image_id", "body_landmarks_array"], axis=1))   #### Input ONLY encodings into clustering algorithm
            print(OPTIMAL_CLUSTERS)
            N_CLUSTERS = OPTIMAL_CLUSTERS
        print(enc_data)
        # I drop image_id as I pass it to knn bc I need it later, but knn can't handle strings
        enc_data["cluster_id"] = kmeans_cluster(enc_data.drop(["image_id", "body_landmarks_array"], axis=1),n_clusters=N_CLUSTERS)
        
        if SAVE_FIG: export_html_clusters(enc_data.drop("image_id", axis=1))
        print(enc_data)
        print(set(enc_data["cluster_id"].tolist()))
        # don't need to write a CSV
        # enc_data.to_csv('clusters_clusterID_byImageID.csv')

        # if USE_SEGMENT:
        #     Base.metadata.create_all(engine)
        save_clusters_DB(enc_data)
        save_images_clusters_DB(enc_data)
        print("saved segment to clusters")
    elif MODE == 1:

        assign_images_clusters_DB(enc_data)
        print("assigned and saved segment to clusters")


    end = time.time()
    print (end - start)
    return True

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    