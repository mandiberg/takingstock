# import datetime   ####### for saving cluster analytics
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import plotly as py
# import plotly.graph_objs as go

from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
# my ORM
from my_declarative_base import Base, Images, SegmentBig, Topics,ImagesTopics, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, ForeignKey
import pymongo

from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, update, Float, LargeBinary, select, and_
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
from gensim import corpora
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
Gen_corpus, Aug26
1M, 101s
5M, 320s
10M, 721s
20M,
'''

'''

'''

title = 'Please choose your operation: '
options = ['Make Dictionary and BoW Corpus','Model topics', 'Index topics','calculate optimum_topics', 'Make Dict & BoW in batches', 'Merge corpus batches']
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
USE_SEGMENT = True
USE_BIGSEGMENT = True
VERBOSE = True
RANDOM = False
global_counter = 0
QUERY_LIMIT = 50000000
# started at 9:45PM, Feb 17


MODEL="TF" ## OR TF  ## Bag of words or TF-IDF
NUM_TOPICS=30

stemmer = SnowballStemmer('english')


# open and read a csv file, and assign each row as an element in a list
def read_csv(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data

# removing all keywords that are stored in gender, ethnicity, and age tables
GENDER_LIST = read_csv(os.path.join(io.ROOT, "stopwords_gender.csv"))
ETH_LIST = read_csv(os.path.join(io.ROOT, "stopwords_ethnicity.csv"))
AGE_LIST = read_csv(os.path.join(io.ROOT, "stopwords_age.csv"))                       
SKIP_TOKEN_LIST = read_csv(os.path.join(io.ROOT, "skip_tokens.csv"))                       

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

mongo_client = pymongo.MongoClient(io.dbmongo['host'])
mongo_db = mongo_client[io.dbmongo['name']]
mongo_collection = mongo_db['tokens']

if USE_BIGSEGMENT:
    SegmentTable = SegmentBig
    SegmentTable_name = 'SegmentBig_isface'
else:
    # this is prob redundant, and could be replaced by calling the SegmentTable object from Base
    SegmentTable_name = 'SegmentOct20'

    # to create new SegmentTable with variable as name
    class SegmentTable(Base):
        __tablename__ = SegmentTable_name

        image_id = Column(Integer, primary_key=True)
        site_name_id = Column(Integer)
        site_image_id = Column(String(50))
        contentUrl = Column(String(300), nullable=False)
        imagename = Column(String(200))
        description = Column(String(150))
        face_x = Column(DECIMAL(6, 3))
        face_y = Column(DECIMAL(6, 3))
        face_z = Column(DECIMAL(6, 3))
        mouth_gap = Column(DECIMAL(6, 3))
        face_landmarks = Column(BLOB)
        bbox = Column(JSON)
        face_encodings = Column(BLOB)
        face_encodings68 = Column(BLOB)
        site_image_id = Column(String(50), nullable=False)
        keyword_list = Column(BLOB)  # Pickled list
        tokenized_keyword_list = Column(BLOB)  # Pickled list
        ethnicity_list = Column(BLOB)  # Pickled list


ambig_key_dict = { "black-and-white": "black_and_white", "black and white background": "black_and_white background", "black and white portrait": "black_and_white portrait", "black amp white": "black_and_white", "white and black": "black_and_white", "black and white film": "black_and_white film", "black and white wallpaper": "black_and_white wallpaper", "black and white cover photos": "black_and_white cover photos", "black and white outfit": "black_and_white outfit", "black and white city": "black_and_white city", "blackandwhite": "black_and_white", "black white": "black_and_white", "black friday": "black_friday", "black magic": "black_magic", "black lives matter": "black_lives_matter black_ethnicity", "black out tuesday": "black_out_tuesday black_ethnicity", "black girl magic": "black_girl_magic black_ethnicity", "beautiful black women": "beautiful black_ethnicity women", "black model": "black_ethnicity model", "black santa": "black_ethnicity santa", "black children": "black_ethnicity children", "black history": "black_ethnicity history", "black family": "black_ethnicity family", "black community": "black_ethnicity community", "black owned business": "black_ethnicity owned business", "black holidays": "black_ethnicity holidays", "black models": "black_ethnicity models", "black girl bullying": "black_ethnicity girl bullying", "black santa claus": "black_ethnicity santa claus", "black hands": "black_ethnicity hands", "black christmas": "black_ethnicity christmas", "white and black girl": "white_ethnicity and black_ethnicity girl", "white woman": "white_ethnicity woman", "white girl": "white_ethnicity girl", "white people": "white_ethnicity", "red white and blue": "red_white_and_blue"}
def clarify_keywords(text):
    # // if text contains either of the strings "black" or "white", replace with "black_and_white"
    if "black" in text or "white" in text:
        for key, value in ambig_key_dict.items():
            text = text.replace(key, value)
        # print("clarified text: ",text)
    return text

def set_query():
    # currently only used for indexing
    # not refactored for mongo (despite the one WHERE line)

    # mongofy, for indexing:
    SELECT = "DISTINCT(image_id),description"
    FROM = SegmentTable_name
    WHERE = " mongo_tokens IS NOT NULL "
    if RANDOM: 
        WHERE += "AND image_id >= (SELECT FLOOR(MAX(image_id) * RAND()) FROM bagofkeywords)"
    LIMIT = QUERY_LIMIT
    if MODE==2 and USE_BIGSEGMENT:
        # assigning topics
        WHERE = " image_id NOT IN (SELECT image_id FROM imagestopics)"
    elif MODE==2:
        # assigning topics
        WHERE = "tokenized_keyword_list IS NOT NULL AND image_id NOT IN (SELECT image_id FROM imagestopics)"

    return SELECT, FROM, WHERE, LIMIT

def selectSQL():
    # currently only used for indexing
    SELECT, FROM, WHERE, LIMIT = set_query()
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)};"
    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))

    # image_id_col = Column('image_id', Integer, primary_key=True, nullable=False)
    # description_col = Column('description', String(150))
    # tokenized_keyword_list_col = Column('tokenized_keyword_list', BLOB)

    # # Define your select query using the columns
    # select_query = (
    #     select(image_id_col, description_col, tokenized_keyword_list_col)
    #     .select_from(SegmentTable)
    #     .where(
    #         and_(
    #             tokenized_keyword_list_col != None,  # Check for non-null tokenized_keyword_list
    #             ~image_id_col.in_(select([ImagesTopics.c.image_id]))  # Subquery to exclude image_id in ImagesTopics
    #         )
    #     )
    #     .limit(QUERY_LIMIT)
    # )

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

       # Create a Topics object
        topics_entry = Topics(
        topic_id = idx,
        topic = "".join(topic_list)
        )

    # Add the Topics object to the session
        session.add(topics_entry)
        print("Updated topic_id {}".format(idx))
    print(" >>>>>>>>>>>> DID NOT SAVE TOPICS TO DATABASE <<<<<<<<<<<<")
    return
    session.commit()
    return

def write_imagetopics(resultsjson,lda_model_tfidf,dictionary):
    print("writing data to the imagetopic table")
    idx_list, topic_list = zip(*lda_model_tfidf.print_topics(-1))
    for i,row in enumerate(resultsjson):
        print("row: ",row)
        # mongofy:
        results = mongo_collection.find_one({"image_id": row["image_id"]})
        if results:
            print("results: ",results)
            keyword_list=" ".join(pickle.loads(results['tokenized_keyword_list']))
        else:
            print("mongo results are empty, using description instead")
            keyword_list = row["description"]
        print(keyword_list)
        # keyword_list=" ".join(pickle.loads(row["tokenized_keyword_list"]))


        # # handles empty keyword_list
        # if keyword_list:
        #     word_list = keyword_list
        # else:
        #     word_list = row["description"]

        bow_vector = dictionary.doc2bow(preprocess(keyword_list))

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
    # num_topics_list=[40,80,120]
    num_topics_list=[30,40,50]
    coher_val_list=np.zeros(len(num_topics_list))
    for i,num_topics in enumerate(num_topics_list):
        lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=NUMBER_OF_PROCESSES)
        cm = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
        coher_val_list[i]=cm.get_coherence()
    print(num_topics_list,coher_val_list)  # get coherence value

def gen_corpus():
    # this takes the tokenized keyword list and generates a corpus saved to disk
    print("generating corpus")
    # query = session.query(SegmentTable.tokenized_keyword_list).filter(SegmentTable.tokenized_keyword_list.isnot(None)).limit(QUERY_LIMIT)
    query = session.query(SegmentTable.image_id).filter(SegmentTable.mongo_tokens.isnot(None)).limit(QUERY_LIMIT)
    results = query.all()
    total_rows = query.count()
    if VERBOSE: 
        print(query.statement.compile(compile_kwargs={"literal_binds": True}))  # Print the SQL query
        print("total_rows in query: ",total_rows)
        print("results length: ",len(results))
        # for row in results: print("row: ",row.tokenized_keyword_list)
    if results: 
        image_id_list = [row[0] for row in results]
    else: 
        print("no image_id results")
        return
        # get list of image_id
    
    # query mongo tokens collection for tokenized_keyword_list
    # query = mongo_collection.find({"image_id": {"$in": image_id_list}})
    # results = list(query)

    batch_size = 100  # Set your desired batch size

    all_results = []  # To accumulate all results

    # Break image_id_list into batches
    for batch_start in range(0, len(image_id_list), batch_size):
        batch = image_id_list[batch_start:batch_start + batch_size]
        
        # Perform a query for the current batch
        query = mongo_collection.find({"image_id": {"$in": batch}})
        
        # Convert the cursor to a list and append to the accumulated results
        batch_results = list(query)
        all_results.extend(batch_results)

    # Now, all_results contains all the documents retrieved in batches

    if VERBOSE:  print("mongo results length: ", len(all_results))
    if not all_results: 
        print("no mongo results")
        return

    # Ensure we are working with all_results, not results
    token_lists = [pickle.loads(row["tokenized_keyword_list"]) for row in all_results]
    token_lists = [[token for token in doc if token not in SKIP_TOKEN_LIST] for doc in token_lists]

    if VERBOSE: print("token_lists first entry: ",token_lists[:1])
    dictionary = gensim.corpora.Dictionary(token_lists)
    if VERBOSE: print("gen_corpus: created dictionary")
    dictionary.filter_extremes(no_below=100, no_above=0.5, keep_n=100000)
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


def gen_corpus_in_batches():
    # I think this was an attempt to batch the corpuse generation but you can't quite add them back together
    # deleting. it is in the repo if needed
    pass

def merge_corpus_batches():
    # I think this was an attempt to batch the corpuse generation but you can't quite add them back together
    # deleting. it is in the repo if needed
    pass

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
        # go get LIMIT number of items (will duplicate initial select, but only the initial one)
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
    elif MODE==4:gen_corpus_in_batches()
    elif MODE==5:merge_corpus_batches()

    end = time.time()
    print (end - start)
    return True

if __name__ == '__main__':
    main()




