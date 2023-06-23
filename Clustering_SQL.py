#sklearn imports
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
from sklearn.model_selection import ParameterGrid
from sklearn import metrics

from sqlalchemy import create_engine
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

start = time.time()

io = DataIO()
db = io.db
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES

# Satyam, you want to set this to False
USE_SEGMENT = True

# number of clusters produced
opt_c_size=True
N_CLUSTERS = 128


if USE_SEGMENT is True:

    # where the script is looking for files list
    # do not use this if you are using the regular Clusters and ImagesClusters tables
    SegmentTable_name = 'May25segment123side_to_side'

    # join with SSD tables. Satyam, use the one below
    SELECT = "DISTINCT(e.image_id), e.face_encodings"
    FROM = "Encodings e"
    QUERY = "e.image_id IN"
    SUBQUERY = f"(SELECT seg1.image_id FROM {SegmentTable_name} seg1 )"
    WHERE = f"{QUERY} {SUBQUERY}"
    LIMIT = 100000

else:
    # Basic Query, this works with gettytest3
    SELECT = "DISTINCT(image_id),face_encodings"
    FROM ="encodings"
    WHERE = "face_encodings IS NOT NULL"
    LIMIT = 1000
    SegmentTable_name = ""


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
    
def best_score(df):
    n_list=np.linspace(64,128,10,dtype='int')
    score=np.zeros(len(n_list))
    for i,n_clusters in enumerate(n_list):
        kmeans = KMeans(n_clusters,n_init=10, init = 'k-means++', random_state = 42, max_iter = 300)
        preds = kmeans.fit_predict(df)
        score[i]=silhouette_score(df, preds)
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


# could also use df drop tolist
# unique_clusters = df['cluster_id'].drop_duplicates().tolist()
# saving this here, because may need to roll back to this
# when I have to calculate the median enc for each cluster. 
# # that will require access to all the encodings in the df? 
# def save_clusters_DB(df):
#     #save the df to a table
#     for _, row in df.iterrows():
#         cluster_id = row['cluster_id']
#         existing_record = session.query(Clusters).filter_by(cluster_id=cluster_id).first()

#         if existing_record is None:
#             instance = Clusters(
#                 cluster_id=cluster_id,
#                 cluster_median=None
#             )
#             session.add(instance)
#         else:
#             print(f"Skipping duplicate record with cluster_id {cluster_id}")

#     try:
#         session.commit()
#         print("Data saved successfully.")
#     except exc.IntegrityError as e:
#         session.rollback()
#         print(f"Error occurred during data saving: {str(e)}")

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


def main():
    # create_my_engine(db)
    print("about to SQL: ",SELECT,FROM,WHERE,LIMIT)
    resultsjson = selectSQL()
    print("got results, count is: ",len(resultsjson))
    enc_data=pd.DataFrame()
    for row in resultsjson:
        # gets contentUrl
        df=encodings_split(pickle.loads(row["face_encodings"], encoding='latin1'))
        df["image_id"]=row["image_id"]
        enc_data = pd.concat([enc_data,df],ignore_index=True) 
    # choose if you want optimal cluster size or custom cluster size using the parameter opt_c_size
    if opt_c_size: N_clusters= best_score(df)
    enc_data["cluster_id"] = kmeans_cluster(enc_data,n_clusters=N_CLUSTERS)
    print(enc_data)
    print(set(enc_data["cluster_id"].tolist()))
    # if USE_SEGMENT:
    #     Base.metadata.create_all(engine)
    save_clusters_DB(enc_data)
    save_images_clusters_DB(enc_data)
    print("saved segment to clusters")

    end = time.time()
    print (end - start)
    return True

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    