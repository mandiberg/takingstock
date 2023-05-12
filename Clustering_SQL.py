#sklearn imports
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
from sklearn.model_selection import ParameterGrid
from sklearn import metrics

from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

import numpy as np
import pandas as pd
import os
import time
import pickle
from sys import platform


# platform specific file folder (mac for michael, win for satyam)
if platform == "darwin":
    ####### Michael's OS X Credentials ########
    db = {
        "host":"localhost",
        "name":"stock1",            
        "user":"root",
        "pass":"Fg!27Ejc!Mvr!GT"
    }
    ROOT= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production") ## only on Mac
    NUMBER_OF_PROCESSES = 8
elif platform == "win32":
    ######## Satyam's WIN Credentials #########
    db = {
        "host":"localhost",
        "name":"gettytest3",                 
        "user":"root",
        "pass":"SSJ2_mysql"
    }
    ROOT= os.path.join("D:/"+"Documents/projects-active/facemap_production") ## SD CARD
    NUMBER_OF_PROCESSES = 4


SELECT = "DISTINCT(image_id),face_encodings"
FROM ="encodings"
WHERE = "face_encodings IS NOT NULL"
LIMIT = 1000

engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
metadata = MetaData(engine)

start = time.time()

def selectSQL():
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)};"
    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))

    resultsjson = ([dict(row) for row in result.mappings()])

    return(resultsjson)
\

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
    
    
def main():
    
    # create_my_engine(db)
    resultsjson = selectSQL()
    print("got results, count is: ",len(resultsjson))
    enc_data=pd.DataFrame()
    for row in resultsjson:
        # gets contentUrl
        #print (row["image_id"],pickle.loads(row["face_encodings"]))
        df=encodings_split(pickle.loads(row["face_encodings"]))
        df["image_id"]=row["image_id"]
        enc_data = pd.concat([enc_data,df],ignore_index=True) 
    enc_data["cluster_id"] = kmeans_cluster(enc_data,n_clusters=3)
    print("about to SQL: ",SELECT,FROM,WHERE,LIMIT)
    print(set(enc_data["cluster_id"].tolist()))
    end = time.time()
    print (end - start)
    return True

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    