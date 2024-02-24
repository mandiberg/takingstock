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
from my_declarative_base import Base, Images, Topics,ImagesTopics, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, ForeignKey

from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, update, Float, LargeBinary
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool
from pick import pick

import numpy as np
import pandas as pd
import os
import time
import pickle
from sys import platform
import ast
import csv

#mine
from mp_db_io import DataIO
###########
import gensim
from gensim.test.utils import get_tmpfile
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import os

import nltk
from gensim import corpora, models
from pprint import pprint

#nltk.download('wordnet') ##only first time

# MM you need to use conda activate gensim311 

'''
tracking time based on items, for speed predictions, 88topics
items, seconds
100000, 61
500000, 1015
1000000, 486
2000000, 4503
4000000, 
'''

'''
Gen_corpus, Feb24
1M, 145s
2M, 360s
4M, 
'''


title = 'Please choose your operation: '
options = ['Make Dictionary and BoW Corpus','Model topics', 'Index topics','calculate optimum_topics']
io = DataIO()
db = io.db
io.db["name"] = "stock"
io.ROOT = "/Users/michaelmandiberg/Documents/GitHub/facemap/topic_model"

NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
MODEL_PATH=os.path.join(io.ROOT,"model")
DICT_PATH=os.path.join(io.ROOT,"dictionary.dict")
BOW_CORPUS_PATH=os.path.join(io.ROOT,"BOW_lda_corpus.mm")
TOKEN_PATH=os.path.join(io.ROOT,"tokenized_keyword_list.csv")
TFIDF_CORPUS_PATH=os.path.join(io.ROOT,"TFIDF_lda_corpus.mm")
# Satyam, you want to set this to False
USE_SEGMENT = False
VERBOSE = True
RANDOM = False
global_counter = 0
QUERY_LIMIT = 4000000
# started at 9:45PM, Feb 17
BATCH_SIZE = 100

MODEL="TF" ## OR TF  ## Bag of words or TF-IDF
NUM_TOPICS=88

stemmer = SnowballStemmer('english')

def set_query():
    # Basic Query, this works with gettytest3
    SELECT = "DISTINCT(image_id),description,keyword_list"
    FROM ="bagofkeywords"
    WHERE = "keyword_list IS NOT NULL "
    if RANDOM: 
        WHERE += "AND image_id >= (SELECT FLOOR(MAX(image_id) * RAND()) FROM bagofkeywords)"
    LIMIT = QUERY_LIMIT
    if MODE==1:
        # assigning topics
        WHERE = "keyword_list IS NOT NULL AND image_id NOT IN (SELECT image_id FROM imagestopics)"
        # WHERE = "image_id = 423638"
        LIMIT=100000
    return SELECT, FROM, WHERE, LIMIT


# open and read a csv file, and assign each row as an element in a list
def read_csv(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data

# removing all keywords that are stored in gender, ethnicity, and age tables
GENDER_LIST = read_csv(os.path.join(io.ROOT, "stopwords_gender.csv"))
ETH_LIST = read_csv(os.path.join(io.ROOT, "stopwords_ethnicity.csv"))
AGE_LIST = read_csv(os.path.join(io.ROOT, "stopwords_age.csv"))                       
MY_STOPWORDS = gensim.parsing.preprocessing.STOPWORDS.union(set(GENDER_LIST+ETH_LIST+AGE_LIST))

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

class BagOfKeywords(Base):
    __tablename__ = 'BagOfKeywords'

    image_id = Column(Integer, primary_key=True)
    keyword_list = Column(LargeBinary)
    tokenized_keyword_list = Column(LargeBinary)

ambig_key_dict = { "black-and-white": "black_and_white", "black and white background": "black_and_white background", "black and white portrait": "black_and_white portrait", "black amp white": "black_and_white", "white and black": "black_and_white", "black and white film": "black_and_white film", "black and white wallpaper": "black_and_white wallpaper", "black and white cover photos": "black_and_white cover photos", "black and white outfit": "black_and_white outfit", "black and white city": "black_and_white city", "blackandwhite": "black_and_white", "black white": "black_and_white", "black friday": "black_friday", "black magic": "black_magic", "black lives matter": "black_lives_matter black_ethnicity", "black out tuesday": "black_out_tuesday black_ethnicity", "black girl magic": "black_girl_magic black_ethnicity", "beautiful black women": "beautiful black_ethnicity women", "black model": "black_ethnicity model", "black santa": "black_ethnicity santa", "black children": "black_ethnicity children", "black history": "black_ethnicity history", "black family": "black_ethnicity family", "black community": "black_ethnicity community", "black owned business": "black_ethnicity owned business", "black holidays": "black_ethnicity holidays", "black models": "black_ethnicity models", "black girl bullying": "black_ethnicity girl bullying", "black santa claus": "black_ethnicity santa claus", "black hands": "black_ethnicity hands", "black christmas": "black_ethnicity christmas", "white and black girl": "white_ethnicity and black_ethnicity girl", "white woman": "white_ethnicity woman", "white girl": "white_ethnicity girl", "white people": "white_ethnicity", "red white and blue": "red_white_and_blue"}
def clarify_keywords(text):
    # // if text contains either of the strings "black" or "white", replace with "black_and_white"
    if "black" in text or "white" in text:
        for key, value in ambig_key_dict.items():
            text = text.replace(key, value)
        # print("clarified text: ",text)
    return text

def selectSQL():
    SELECT, FROM, WHERE, LIMIT = set_query()
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)};"
    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    return(resultsjson)

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    global global_counter
    result = []
    if global_counter % 10000 == 0:
        print("preprocessed: ",global_counter)
    text = clarify_keywords(text.lower())
    global_counter += 1

    for token in gensim.utils.simple_preprocess(text):
        if token not in MY_STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def load_corpus():
    print("loading corpus and dictionary")
    dictionary = corpora.Dictionary.load(DICT_PATH)
    corpus = corpora.MmCorpus(TFIDF_CORPUS_PATH)
    return dictionary, corpus 

    
def LDA_model(num_topics):
    dictionary, corpus = load_corpus()
    print("processing the model now")
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=NUMBER_OF_PROCESSES)
    lda_model.save(MODEL_PATH)
    print("processed all")
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

def write_imagetopics(resultsjson,lda_model_tfidf,dictionary):
    print("writing data to the imagetopic table")
    idx_list, topic_list = zip(*lda_model_tfidf.print_topics(-1))
    for i,row in enumerate(resultsjson):
        # print(row)
        keyword_list=" ".join(pickle.loads(row["keyword_list"]))

        # handles empty keyword_list
        if keyword_list:
            word_list = keyword_list
        else:
            word_list = row["description"]

        bow_vector = dictionary.doc2bow(preprocess(word_list))

        #index,score=sorted(lda_model_tfidf[bow_corpus[i]], key=lambda tup: -1*tup[1])[0]
        index,score=sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1])[0]
        imagestopics_entry=ImagesTopics(
        image_id=row["image_id"],
        topic_id=index,
        topic_score=score
        )
        session.add(imagestopics_entry)
        # print(f'image_id {row["image_id"]} -- topic_id {index} -- topic tokens {topic_list[index][:100]}')
        # print(f"keyword list {keyword_list}")

        if row["image_id"] % 1000 == 0:
            print("Updated image_id {}".format(row["image_id"]))


    # Add the imagestopics object to the session
    session.commit()
    return
def calc_optimum_topics():

    dictionary, corpus = load_corpus()

    # #######TOPIC MODELING ############
    # txt = pd.DataFrame(index=range(len(resultsjson)),columns=["description","keywords","index","score"])
    # for i,row in enumerate(resultsjson):
    #     #txt.at[i,"description"]=row["description"]
    #     txt.at[i,"keyword_list"]=" ".join(pickle.loads(row["keyword_list"]))
    # #processed_txt=txt['description'].map(preprocess)
    # processed_txt=txt['keyword_list'].map(preprocess)

    # gen_corpus(processed_txt,MODEL)
    # corpus = corpora.MmCorpus(BOW_CORPUS_PATH)
    # dictionary = corpora.Dictionary.load(MODEL_PATH+'.id2word')

    
    # num_topics_list=[80,90,100,110,120]
    num_topics_list=[40,80,120]
    coher_val_list=np.zeros(len(num_topics_list))
    for i,num_topics in enumerate(num_topics_list):
        lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=NUMBER_OF_PROCESSES)
        cm = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
        coher_val_list[i]=cm.get_coherence()
    print(num_topics_list,coher_val_list)  # get coherence value

def gen_corpus():
    print("generating corpus")
    query = session.query(BagOfKeywords.tokenized_keyword_list).filter(BagOfKeywords.tokenized_keyword_list.isnot(None)).limit(QUERY_LIMIT)
    results = query.all()
    total_rows = query.count()
    if VERBOSE: 
        print(query.statement.compile(compile_kwargs={"literal_binds": True}))  # Print the SQL query
        print("total_rows in query: ",total_rows)
        print("results length: ",len(results))
        # for row in results: print("row: ",row.tokenized_keyword_list)
    
    token_lists = [pickle.loads(row.tokenized_keyword_list) for row in results]
    if VERBOSE: print("token_lists: ",token_lists[:1])

    dictionary = gensim.corpora.Dictionary(token_lists)
    if VERBOSE: print("gen_corpus: created dictionary")
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    if VERBOSE: print("gen_corpus: filtered extremes")
    bow_corpus = [dictionary.doc2bow(doc) for doc in token_lists] ## BOW corpus
    if VERBOSE: print("gen_corpus: created bow_corpus")
    if MODEL=="TF":   
        tfidf = models.TfidfModel(bow_corpus)  ## converting BOW to TDIDF corpus
        tfidf_corpus = tfidf[bow_corpus]
        if VERBOSE: print("gen_corpus: created tfidf_corpus")
    dictionary.save(DICT_PATH)
    if VERBOSE: print("gen_corpus: saved dictionary")
    corpora.MmCorpus.serialize(TFIDF_CORPUS_PATH, tfidf_corpus)
    if VERBOSE: print("gen_corpus: saved tfidf_corpus")
    corpora.MmCorpus.serialize(BOW_CORPUS_PATH, bow_corpus)
    if VERBOSE: print("gen_corpus: saved bow_corpus")
    return 


def topic_model():
    # #######TOPIC MODELING ############
    # txt = pd.DataFrame(index=range(len(resultsjson)),columns=["description","keywords","index","score"])
    # for i,row in enumerate(resultsjson):
    #     #txt.at[i,"description"]=row["description"]
    #     txt.at[i,"keyword_list"]=" ".join(pickle.loads(row["keyword_list"]))
    # #processed_txt=txt['description'].map(preprocess)
    # processed_txt=txt['keyword_list'].map(preprocess)
    # gen_corpus(processed_txt,MODEL)
    
    lda_model=LDA_model(NUM_TOPICS)

    write_topics(lda_model)
    
    return


def topic_index(resultsjson):
    ###########TOPIC INDEXING#########################
    bow_corpus = corpora.MmCorpus(BOW_CORPUS_PATH)
    #dictionary = corpora.Dictionary.load(DICT_PATH)
    lda_model_tfidf = gensim.models.LdaModel.load(MODEL_PATH)
    lda_dict = corpora.Dictionary.load(MODEL_PATH+'.id2word')
    print("model loaded successfully")
    while True:
        # go get LIMIT number of items (will duplicate initial select)
        print("about to SQL:")
        resultsjson = selectSQL()
        print("got results, count is: ",len(resultsjson))
        if len(resultsjson) == 0:
            break

        write_imagetopics(resultsjson,lda_model_tfidf,lda_dict)
        print("updated cells")
    print("DONE")

    return
    
def main():
    global MODE
    OPTION, MODE = pick(options, title)

    start = time.time()
    # create_my_engine(db)

    # 
    if MODE==2:
        resultsjson = selectSQL()
        print("got results, count is: ",len(resultsjson))

    if MODE==0:gen_corpus()
    elif MODE==1:topic_model()
    elif MODE==2:topic_index(resultsjson)
    elif MODE==3:calc_optimum_topics()
    end = time.time()
    print (end - start)
    return True

if __name__ == '__main__':
    main()




