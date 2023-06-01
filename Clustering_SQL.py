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

# number of clusters created, will be added to table names if USE_SEGMENT
N_CLUSTERS = 12

# Satyam, you want to set this to False
USE_SEGMENT = False

io = DataIO()
db = io.db
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES

engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


if USE_SEGMENT is True:

    # where the script is looking for files list
    # do not use this if you are using the regular Clusters and ImagesClusters tables
    SegmentTable_name = 'May25seg123y'

    # MM join with SSD tables. Satyam, use the one below
    SELECT = "DISTINCT(e.image_id), e.face_encodings"
    FROM = "Encodings e"
    QUERY = "e.image_id IN"
    SUBQUERY = f"(SELECT seg1.image_id FROM {SegmentTable_name} seg1 )"
    WHERE = f"{QUERY} {SUBQUERY}"
    LIMIT = 1000

    # define new cluster table names based on segment name
    SegmentClustersTable_name = "Clusters_"+SegmentTable_name+str(N_CLUSTERS)
    print("SegmentClustersTable_name")
    print(SegmentClustersTable_name)
    SegmentImagesClustersTable_name = "ImagesClusters_"+SegmentTable_name+str(N_CLUSTERS)

    class SegmentClustersTable(Base):
        __tablename__ = SegmentClustersTable_name

        cluster_id = Column(Integer, primary_key=True, autoincrement=True)
        cluster_median = Column(BLOB)

    class SegmentImagesClustersTable(Base):
        __tablename__ = SegmentImagesClustersTable_name

        image_id = Column(Integer, ForeignKey(Images.image_id), primary_key=True)
        cluster_id = Column(Integer, ForeignKey(SegmentClustersTable.cluster_id))
    
    # class SegmentClustersTable(Base):
    #     __tablename__ = SegmentClustersTable_name

    #     cluster_id = Column(Integer, primary_key=True, autoincrement=True)
    #     cluster_median = Column(BLOB)

    # class SegmentImagesClustersTable(Base):
    #     __tablename__ = SegmentImagesClustersTable_name

    #     image_id = Column(Integer, ForeignKey(Images.image_id), primary_key=True)
    #     cluster_id = Column(Integer, ForeignKey(SegmentClustersTable.cluster_id))


else:
    # Basic Query, this works with gettytest3
    SELECT = "DISTINCT(image_id),face_encodings"
    FROM ="encodings"
    WHERE = "face_encodings IS NOT NULL"
    LIMIT = 1000
    SegmentTable_name = None





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
    
def encodings_split(encodings):
    col="encodings"
    col_list=[]
    for i in range(128):col_list.append(col+str(i))
    encoding_data=pd.DataFrame({col:[np.array(encodings)]})
    #splitting the encodings column
    df = pd.DataFrame(encoding_data["encodings"].tolist(), columns=col_list)
    return df

def save_clusters_DB(df):
    # Convert to set and Save the df to a table
    unique_clusters = set(df['cluster_id'])
    for cluster_id in unique_clusters:
        if SegmentTable_name is None:
            print("first condition")
            existing_record = session.query(Clusters).filter_by(cluster_id=cluster_id).first()
        elif SegmentTable_name:
            print("second condition")

            existing_record = session.query(SegmentClustersTable).filter_by(cluster_id=cluster_id).first()
        else:
            print("failing the existing_record record test in save_clusters_DB")

        if existing_record is None and SegmentTable_name is False:
            instance = Clusters(
                cluster_id=cluster_id,
                cluster_median=None
            )
            session.add(instance)
        elif existing_record is None and SegmentTable_name:
            instance = SegmentClustersTable(
                cluster_id=cluster_id,
                cluster_median=None
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
        if SegmentTable_name is None:
            existing_record = session.query(ImagesClusters).filter_by(image_id=image_id).first()
        elif SegmentTable_name:
            existing_record = session.query(SegmentImagesClustersTable).filter_by(image_id=image_id).first()
        else:
            print("failing the existing_record record test in save_images_clusters_DB")


        if existing_record is None and pd.isnull(SegmentTable_name):
            instance = ImagesClusters(
                image_id=image_id,
                cluster_id=cluster_id,
            )
            session.add(instance)

        elif existing_record is None and SegmentTable_name:
            instance = SegmentImagesClustersTable(
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
        df=encodings_split(pickle.loads(row["face_encodings"]))
        df["image_id"]=row["image_id"]
        enc_data = pd.concat([enc_data,df],ignore_index=True) 
    # I changed n_clusters to 128, from 3, and now it returns 128 clusters
    enc_data["cluster_id"] = kmeans_cluster(enc_data,n_clusters=N_CLUSTERS)
    print(enc_data)
    print(set(enc_data["cluster_id"].tolist()))
    if USE_SEGMENT:
        tables = [
            Table(SegmentClustersTable.__tablename__, Base.metadata),
            Table(SegmentImagesClustersTable.__tablename__, Base.metadata),
        ]

        # tables = [SegmentClustersTable.__tablename__, SegmentImagesClustersTable.__tablename__]
        Base.metadata.create_all(engine, tables=tables)
    save_clusters_DB(enc_data)
    save_images_clusters_DB(enc_data)
    print("saved segment to clusters")

    end = time.time()
    print (end - start)
    return True

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    