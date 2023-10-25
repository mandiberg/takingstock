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
import plotly as py
import plotly.graph_objs as go


from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
# my ORM
from my_declarative_base import Base, Images, Clusters, ImagesClusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, ForeignKey

from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, update, Float
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

import numpy as np
import pandas as pd
import os
import time
import pickle
from sys import platform

#mine
from mp_db_io import DataIO

# MM you need to use conda activate minimal_ds 

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
1340000, ???
'''

start = time.time()

io = DataIO()
db = io.db
io.db["name"] = "ministock1023"
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
MODE = 0

# this works for using segment in stock, and for ministock
USE_SEGMENT = True

# get the best fit for clusters
GET_OPTIMAL_CLUSTERS=False

# number of clusters produced. run GET_OPTIMAL_CLUSTERS and add that number here
N_CLUSTERS = 128
SAVE_FIG=False ##### option for saving the visualized data

if USE_SEGMENT is True and MODE == 0:

    # where the script is looking for files list
    # do not use this if you are using the regular Clusters and ImagesClusters tables
    SegmentTable_name = 'SegmentOct20'

    # Basic Query, this works with gettytest3
    SELECT = "DISTINCT(image_id),face_encodings68"
    FROM = SegmentTable_name
    WHERE = "face_encodings68 IS NOT NULL"
    LIMIT = 1500000

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
    FROM = f"{SegmentTable_name} s LEFT JOIN ImagesClusters ic ON s.image_id = ic.image_id"
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

# # define new cluster table names based on segment name
# SegmentClustersTable_name = "Clusters_"+SegmentTable_name
# SegmentImagesClustersTable_name = "ImagesClusters_"+SegmentTable_name

# class SegmentClustersTable(Base):
#     __tablename__ = SegmentClustersTable_name

#     cluster_id = Column(Integer, primary_key=True, autoincrement=True)
#     cluster_median = Column(BLOB)

# class SegmentImagesClustersTable(Base):
#     __tablename__ = SegmentImagesClustersTable_name

#     image_id = Column(Integer, ForeignKey('Images.image_id'), primary_key=True)
#     cluster_id = Column(Integer, ForeignKey('Clusters.cluster_id'))


def selectSQL():
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)};"
    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    return(resultsjson)


def kmeans_cluster(df,n_clusters=32):
    kmeans = KMeans(n_clusters,n_init=10, init = 'k-means++', random_state = 42, max_iter = 300)
    kmeans.fit(df)
    clusters = kmeans.predict(df)
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
    n_list=np.linspace(64,128,10,dtype='int')
    score=np.zeros(len(n_list))
    for i,n_clusters in enumerate(n_list):
        kmeans = KMeans(n_clusters,n_init=10, init = 'k-means++', random_state = 42, max_iter = 300)
        preds = kmeans.fit_predict(df)
        score[i]=silhouette_score(df, preds)
    print(n_list, score)
    b_score=n_list[np.argmax(score)]
    
    return b_score
    
def encodings_split(encodings):
    col="encodings"
    col_list=[]
    for i in range(128):col_list.append(col+str(i))
    encoding_data=pd.DataFrame({col:[np.array(encodings)]})
    #splitting the encodings column
    df = pd.DataFrame(encoding_data["encodings"].tolist(), columns=col_list)
    return df

def save_clusters_DB(df):
    col="encodings"
    col_list=[]
    for i in range(128):col_list.append(col+str(i))


    # this isn't saving means for each cluster. all cluster_median are the same

    # Convert to set and Save the df to a table
    unique_clusters = set(df['cluster_id'])
    for cluster_id in unique_clusters:
        cluster_df=df[df['cluster_id']==cluster_id]
        cluster_mean=np.array(df[col_list].mean())
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

def get_cluster_medians():

    # Create a SQLAlchemy select statement
    select_query = select([Clusters.cluster_id, Clusters.cluster_median]).limit(1000)

    # Execute the query using your SQLAlchemy session
    results = session.execute(select_query)
    median_dict = {}

    # Process the results as needed
    for row in results:
        print(row)
        cluster_id, cluster_median = row
        median_dict[cluster_id] = pickle.loads(cluster_median, encoding='latin1')
    return median_dict 


def assign_images_clusters_DB(df):
    #assign clusters to each image's encodings
    for _, row in df.iterrows():
        median_dict = get_cluster_medians()
        print(median_dict)

        image_id = row['image_id']
        face_encodings68 = [row[f'encodings{i}'] for i in range(128)]
        # print(existing_record)
        # face_encodings68 = row['face_encodings68']
        # existing_record = session.query(ImagesClusters).filter_by(image_id=image_id).first()

        this_dist_dict = {}
        for cluster_id in median_dict:
            enc1=np.array(face_encodings68)
            enc2=median_dict[cluster_id]
            # enc2=np.array(median_dict[cluster_id])
            this_dist_dict[cluster_id]=np.linalg.norm(enc1 - enc2, axis=0)
        cluster_id = min(this_dist_dict, key=this_dist_dict.get)
        print(this_dist_dict)
        print(cluster_id)
        quit()
        if cluster_id:
            instance = ImagesClusters(
                image_id=image_id,
                cluster_id=cluster_id,
            )
            session.add(instance)

        else:
            print(f"Something went wrong with image_id {image_id}")

    try:
        session.commit()
        print("Data saved successfully.")
    except IntegrityError as e:
        session.rollback()
        print(f"Error occurred during data saving: {str(e)}")


def main():
    # create_my_engine(db)
    global N_CLUSTERS
    print("about to SQL: ",SELECT,FROM,WHERE,LIMIT)
    resultsjson = selectSQL()
    print("got results, count is: ",len(resultsjson))
    enc_data=pd.DataFrame()
    for row in resultsjson:
        # gets contentUrl
        df=encodings_split(pickle.loads(row["face_encodings68"], encoding='latin1'))
        df["image_id"]=row["image_id"]
        enc_data = pd.concat([enc_data,df],ignore_index=True) 
    
    # choose if you want optimal cluster size or custom cluster size using the parameter GET_OPTIMAL_CLUSTERS
    if MODE == 0:
        if GET_OPTIMAL_CLUSTERS is True: 
            OPTIMAL_CLUSTERS= best_score(enc_data.drop("image_id", axis=1))   #### Input ONLY encodings into clustering alhorithm
            print(OPTIMAL_CLUSTERS)
            N_CLUSTERS = OPTIMAL_CLUSTERS
        print(enc_data)
        enc_data["cluster_id"] = kmeans_cluster(enc_data.drop("image_id", axis=1),n_clusters=N_CLUSTERS)
        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    