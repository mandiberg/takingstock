# import datetime   ####### for saving cluster analytics
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import plotly as py
# import plotly.graph_objs as go

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
# my ORM
from my_declarative_base import Base, Images, Clusters68, ImagesClusters68, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, ForeignKey

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
###########
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import nltk
from gensim import corpora, models
from pprint import pprint

nltk.download('wordnet') ##only first time

# MM you need to use conda activate minimal_ds 

'''
tracking time based on items, for speed predictions
items, seconds
47000, 240
100000, 695
'''

start = time.time()

io = DataIO()
db = io.db
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES

# Satyam, you want to set this to False
USE_SEGMENT = False

MODEL="TF" ## OR TF  ## Bag of words or TF-IDF
NUM_TOPICS=48

stemmer = SnowballStemmer('english')

if USE_SEGMENT is True:

    # where the script is looking for files list
    # do not use this if you are using the regular Clusters and ImagesClusters tables
    SegmentTable_name = 'July15segment123straight'
    

    # join with SSD tables. Satyam, use the one below
    SELECT = "DISTINCT(e.image_id), e.face_encodings68"
    FROM = "Encodings e"
    QUERY = "e.image_id IN"
    SUBQUERY = f"(SELECT seg1.image_id FROM {SegmentTable_name} seg1 )"
    WHERE = f"{QUERY} {SUBQUERY}"
    LIMIT = 1100

else:
    # Basic Query, this works with gettytest3
    SELECT = "DISTINCT(image_id),description,keyword_list"
    FROM ="bagofkeywords"
    WHERE = "keyword_list IS NOT NULL"
    LIMIT = 10000
    SegmentTable_name = ""

# if db['unix_socket']:
    # # for MM's MAMP config
    # engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        # user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    # ), poolclass=NullPool)
# else:
    # engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                # .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)


# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

def selectSQL():
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)};"
    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    return(resultsjson)

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def process(processed_txt,MODEL):
    dictionary = gensim.corpora.Dictionary(processed_txt)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_txt] ## BOW corpus
    corpus=bow_corpus
    if MODEL=="TF":   
        tfidf = models.TfidfModel(bow_corpus)  ## converting BOW to TDIDF corpus
        tfidf_corpus = tfidf[bow_corpus]
        corpus=tfidf_corpus
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=2, workers=2)
    return lda_model
    
def main():
    # create_my_engine(db)
    print("about to SQL: ",SELECT,FROM,WHERE,LIMIT)
    resultsjson = selectSQL()
    print("got results, count is: ",len(resultsjson))
    txt = pd.DataFrame(index=range(len(resultsjson)),columns=["description","keywords","index","score"])
    for i,row in enumerate(resultsjson):
        # gets contentUrl
        txt.at[i,"description"]=row["description"]
        #txt.at[i,"keyword_list"]=" ".join(pickle.loads(row["keyword_list"]))
    #print(txt.tail())
    processed_txt=txt['description'].map(preprocess)
    #processed_txt=txt['keyword_list'].map(preprocess)
    lda_model=process(processed_txt,MODEL)
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
    # for i in range(len(resultsjson)):
        # index,score=sorted(lda_model_tfidf[bow_corpus[i]], key=lambda tup: -1*tup[1])
        # txt[i,"index"]=index[0]
        # txt[i,"score"]=score[0]
        
        
    # if USE_SEGMENT:
    #     Base.metadata.create_all(engine)

    end = time.time()
    print (end - start)
    return True

if __name__ == '__main__':
    main()





