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
from my_declarative_base import Base, Images, Topics,ImagesTopics,Clusters68, ImagesClusters68, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, ForeignKey

from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, update, Float
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool
from pick import pick

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
from gensim.test.utils import get_tmpfile
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import os

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
title = 'Please choose your operation: '
options = ['Topic modelling', 'Topic indexing']
OPTION, MODE = pick(options, title)



start = time.time()

io = DataIO()
db = io.db
io.db["name"] = "ministock"


NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#MODEL_PATH=io.ROOT+"/model"
MODEL_PATH=os.path.join(io.ROOT,"model")
#BOW_CORPUS_PATH=io.ROOT+"/BOW_lda_corpus.mm"
BOW_CORPUS_PATH=os.path.join(io.ROOT,"BOW_lda_corpus.mm")
#TFIDF_CORPUS_PATH=io.ROOT+"/TFIDF_lda_corpus.mm"
TFIDF_CORPUS_PATH=os.path.join(io.ROOT,"TFIDF_lda_corpus.mm")
# Satyam, you want to set this to False
USE_SEGMENT = False

MODEL="TF" ## OR TF  ## Bag of words or TF-IDF
NUM_TOPICS=75

stemmer = SnowballStemmer('english')

# Basic Query, this works with gettytest3
SELECT = "DISTINCT(image_id),description,keyword_list"
FROM ="bagofkeywords"
if MODE==0:
    WHERE = "keyword_list IS NOT NULL "
    LIMIT = 328894
elif MODE==1:
    WHERE = "keyword_list IS NOT NULL AND image_id NOT IN (SELECT image_id FROM imagestopics)"
    LIMIT=1000

if db['unix_socket']:
    # for MM's MAMP config
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

#engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)


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

def save_model(lda_model,tfidf_corpus,bow_corpus):
    #CORPUS_PATH = get_tmpfile(CORPUS_PATH)
    corpora.MmCorpus.serialize(TFIDF_CORPUS_PATH, tfidf_corpus)
    corpora.MmCorpus.serialize(BOW_CORPUS_PATH, bow_corpus)
    lda_model.save(MODEL_PATH)
    print("model saved")
    return

def process(processed_txt,MODEL):
    print("processing the model now")
    dictionary = gensim.corpora.Dictionary(processed_txt)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_txt] ## BOW corpus
    corpus=bow_corpus
    if MODEL=="TF":   
        tfidf = models.TfidfModel(bow_corpus)  ## converting BOW to TDIDF corpus
        tfidf_corpus = tfidf[bow_corpus]
        corpus=tfidf_corpus
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=2, workers=2)
    save_model(lda_model,tfidf_corpus,bow_corpus)
    return lda_model

def write_topics(lda_model):
    print("writing data to the topic table")
    for idx, topic_list in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic_list))

       # Create a BagOfKeywords object
        topics_entry = Topics(
        topic_id = idx,
        topic = "".join(topic_list)
        )

    # Add the BagOfKeywords object to the session
        session.add(topics_entry)
        print("Updated topic_id {}".format(idx))
    session.commit()
    return

def write_imagetopics(resultsjson,lda_model_tfidf,bow_corpus):
    print("writing data to the imagetopic table")
    for i,row in enumerate(resultsjson):
        index,score=sorted(lda_model_tfidf[bow_corpus[i]], key=lambda tup: -1*tup[1])[0]
        imagestopics_entry=ImagesTopics(
        image_id=row["image_id"],
        topic_id=index,
        topic_score=score
        )
        session.add(imagestopics_entry)
        print("Updated image_id {}".format(row["image_id"]))


    # Add the imagestopics object to the session
    session.commit()
    return


def main():
    
    # create_my_engine(db)
    print("about to SQL: ",SELECT,FROM,WHERE,LIMIT)
    resultsjson = selectSQL()
    print("got results, count is: ",len(resultsjson))
    if MODE==0:
        #######TOPIC MODELING ############
        txt = pd.DataFrame(index=range(len(resultsjson)),columns=["description","keywords","index","score"])
        for i,row in enumerate(resultsjson):
            #txt.at[i,"description"]=row["description"]
            txt.at[i,"keyword_list"]=" ".join(pickle.loads(row["keyword_list"]))
        #processed_txt=txt['description'].map(preprocess)
        processed_txt=txt['keyword_list'].map(preprocess)
        lda_model=process(processed_txt,MODEL)
        write_topics(lda_model)
        
    elif MODE==1:
        ###########TOPIC INDEXING#########################
        bow_corpus = corpora.MmCorpus(BOW_CORPUS_PATH)
        lda_model_tfidf = gensim.models.LdaModel.load(MODEL_PATH)
        #lda_dict = corpora.Dictionary.load(MODEL_PATH+'.id2word')
        print("model loaded successfully")
        write_imagetopics(resultsjson,lda_model_tfidf,bow_corpus)
        
    # if USE_SEGMENT:
    #     Base.metadata.create_all(engine)

    end = time.time()
    print (end - start)
    return True

if __name__ == '__main__':
    main()




