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
from my_declarative_base import Base, Images, SegmentBig, Topics,Topics_isnotface, Topics_affect, ImagesTopics, ImagesTopics_isnotface, ImagesTopics_isnotface_isfacemodel, ImagesTopics_affect, imagestopics_ALLgetty4faces_isfacemodel, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, ForeignKey
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
import gensim_test
from gensim_test import corpora
from gensim.test.utils import get_tmpfile
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import os

import nltk
from gensim_test import corpora, models
from pprint import pprint

#nltk.download('wordnet') ##only first time

# MM you need to use conda activate gensim311 


'''
Gen_corpus, Aug26
1M, 101s
5M, 320s
8.8M (5x5) 1133s
14.9M (8x8) 3157s
TK M (9x9) 3800s
'''

'''
LDA_model, Aug26
14.9M, 302s
9x9, 402s Sept 7 30 topics



testing coherence for 30,40,50 topics sept 7: 21367s
'''

title = 'Please choose your operation: '
options = ['Make Dictionary and BoW Corpus','Model topics', 'Index topics','calculate optimum_topics', 'Make Dict & BoW in batches', 'Merge corpus batches']
io = DataIO()
db = io.db
io.db["name"] = "stock"
io.ROOT = "/Users/michaelmandiberg/Documents/GitHub/facemap/model_files"

# Satyam, you want to set this to False
USE_SEGMENT = False # only used for indexing
USE_BIGSEGMENT = True # sets declarative base object. Seem to need to be True for corpus generation. limit with ANGLE instead
IS_GETTYONLY = False # this is for the NOT FACE to constrain the database to only getty images for testing
IS_NOT_FACE = False # this turns of the xyz angle filter for faces pointing forward and returns all images
USE_EXISTING_MODEL = True # this is for the NOT FACE data, to use the FACE model, not used elsewhere
IS_AFFECT = True # switches to the affect model
VERBOSE = False
RANDOM = False # selects random image_ids from the DB. not tested. maybe runs very slow. 
global_counter = 0
QUERY_LIMIT = 10000
query_start_counter = 0 # only used in write image topics
ANGLE = 1 # controls x/y face angle in +/-, set to 9 for building the full model, then indexed
MIN_TOKEN_LENGTH = 2 # minimum token length for the model

if IS_AFFECT:
    MODEL_FOLDER = os.path.join(io.ROOT,"model_affect")
elif IS_NOT_FACE and not USE_EXISTING_MODEL:
    MODEL_FOLDER = os.path.join(io.ROOT,"model_isnotface")
else:
    MODEL_FOLDER = os.path.join(io.ROOT,"model_isface")

NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
MODEL_PATH=os.path.join(MODEL_FOLDER,"model")
DICT_PATH=os.path.join(MODEL_FOLDER,"dictionary.dict")
BOW_CORPUS_PATH=os.path.join(MODEL_FOLDER,"BOW_lda_corpus.mm")
TFIDF_CORPUS_PATH=os.path.join(MODEL_FOLDER,"TFIDF_lda_corpus.mm")


MODEL="TF" ## OR TF  ## Bag of words or TF-IDF
NUM_TOPICS=14

stemmer = SnowballStemmer('english')


# open and read a csv file, and assign each row as an element in a list
def read_csv(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data

if IS_AFFECT:
    # load ALL keys
    ALL_KEYWORDS = read_csv("/Users/michaelmandiberg/Documents/GitHub/facemap/utilities/keys/Keywords_202408151415.csv")
    # load only affect keys
    AFFECT_CSV = os.path.join(io.ROOT, "go_words_affect_april2025.csv")
    # make a list of the values in the third column
    AFFECT_LIST = []
    with open(AFFECT_CSV, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > 2:
                AFFECT_LIST.append(row[2])
    print(AFFECT_LIST)
    # subtract the affect keys from the ALL keywords
    # NOT_AFFECT_LIST = [word for word in ALL_KEYWORDS if word not in AFFECT_LIST]
    # print("NOT_AFFECT_LIST: ", NOT_AFFECT_LIST[0:50])
    # MY_STOPWORDS = gensim.parsing.preprocessing.STOPWORDS.union(set(NOT_AFFECT_LIST))
    SKIP_TOKEN_LIST = read_csv(os.path.join(io.ROOT, "skip_tokens_affect.csv"))                       
    
else:
    # removing all keywords that are stored in gender, ethnicity, and age tables
    GENDER_LIST = read_csv(os.path.join(io.ROOT, "stopwords_gender.csv"))
    ETH_LIST = read_csv(os.path.join(io.ROOT, "stopwords_ethnicity.csv"))
    AGE_LIST = read_csv(os.path.join(io.ROOT, "stopwords_age.csv"))                       
    SKIP_TOKEN_LIST = read_csv(os.path.join(io.ROOT, "skip_tokens.csv"))                       
    # MY_STOPWORDS = gensim.parsing.preprocessing.STOPWORDS.union(set(GENDER_LIST+ETH_LIST+AGE_LIST))


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
mongo_tokens = "mongo_tokens" # default
if IS_NOT_FACE and not USE_EXISTING_MODEL:
    mongo_collection = mongo_db['tokens_noface']
    topics_table = "topics_isnotface"
    images_topics_table = "imagestopics_isnotface"
    SegmentTable = SegmentBig
    SegmentTable_name = 'SegmentBig_isnotface'
elif IS_GETTYONLY and USE_EXISTING_MODEL:
    mongo_collection = mongo_db['tokens_gettyonly']
    topics_table = "topics"
    images_topics_table = "imagestopics_ALLgetty4faces_isfacemodel"
    SegmentTable = SegmentBig
    SegmentTable_name = 'SegmentBig_ALLgetty4faces'
elif IS_NOT_FACE and USE_EXISTING_MODEL:
    mongo_collection = mongo_db['tokens_noface']
    topics_table = "topics"
    images_topics_table = "imagestopics_isnotface_isfacemodel"
    SegmentTable = SegmentBig
    SegmentTable_name = 'SegmentBig_isnotface'
elif IS_AFFECT:
    mongo_collection = mongo_db['tokens_affect']
    topics_table = "topics_affect"
    images_topics_table = "imagestopics_affect"
    SegmentTable = SegmentBig
    SegmentTable_name = 'SegmentBig_isface'
    mongo_tokens = "mongo_tokens_affect" # redefine for affect
else:
    mongo_collection = mongo_db['tokens']
    topics_table = "topics"
    images_topics_table = "imagestopics"
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
    print("setting query from MODE: ",MODE)
    # mongofy, for indexing:
    SELECT = "DISTINCT(image_id),description"
    FROM = SegmentTable_name
    WHERE = f" {mongo_tokens} IS NOT NULL "
    WHERE += " AND face_x > -35 AND face_x < -24 AND face_y > -3 AND face_y < 3 AND face_z > -3 AND face_z < 3 "
    if RANDOM: WHERE += "AND image_id >= (SELECT FLOOR(MAX(image_id) * RAND()) FROM bagofkeywords)"
    LIMIT = QUERY_LIMIT
    if MODE==2 and USE_BIGSEGMENT:
        print("assigning topics via bigsegment")
        # assigning topics
        WHERE = f" {mongo_tokens} IS NOT NULL AND image_id NOT IN (SELECT image_id FROM {images_topics_table})"
    elif MODE==2 and USE_SEGMENT:
        print("assigning topics via small segment")
        WHERE = " face_x > -35 AND face_x < -24 AND face_y > -3 AND face_y < 3 AND face_z > -3 AND face_z < 3 AND "
        # WHERE = " face_x > -40 AND face_x < -20 AND face_y > -5 AND face_y < 5 AND face_z > -5 AND face_z < 5 AND "
        WHERE += f" {mongo_tokens} IS NOT NULL AND image_id NOT IN (SELECT image_id FROM {images_topics_table})"

    elif MODE==2:
        print("assigning topics without any segment")
        # assigning topics
        # I'm not sure how this is different from USE_BIGSEGMENT
        WHERE = f" {mongo_tokens} = 1 AND image_id NOT IN (SELECT image_id FROM {images_topics_table})"

    WHERE += f" AND image_id > {query_start_counter} "

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
def preprocess(text, MY_STOPWORDS):
    global global_counter
    result = []
    if global_counter % 10000 == 0:
        print("preprocessed: ",global_counter)
    text = clarify_keywords(text.lower())
    global_counter += 1

    for token in gensim_test.utils.simple_preprocess(text):
        if token not in MY_STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def load_corpus():
    print("loading corpus and dictionary")
    dictionary = corpora.Dictionary.load(DICT_PATH)
    corpus = corpora.MmCorpus(TFIDF_CORPUS_PATH)
    return dictionary, corpus 

def print_word_counts(dictionary, corpus):
    # This function will print the word counts for each token in the corpus
    from collections import defaultdict

    # Initialize a dictionary to store token counts
    token_counts = defaultdict(int)

    # Iterate through each document in the corpus
    for doc in corpus:
        print(len(doc),doc)
        # Each doc is a list of (token_id, count) tuples
        for token_id, count in doc:
            token_counts[token_id] += count

    # Now output each word with its count
    for token_id, count in token_counts.items():
        word = dictionary[token_id]
        print(f"{word},{count}")

    # Optionally, sort by count (descending) for better insights
    sorted_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 20 most frequent words:")
    for token_id, count in sorted_counts[:20]:
        word = dictionary[token_id]
        print(f"Word: {word}, Count: {count}")

    # This will give you the document frequency (number of documents containing each word)
    for token_id in dictionary.keys():
        word = dictionary[token_id]
        doc_freq = dictionary.dfs[token_id]  # Number of documents this word appears in
        print(f"Word: {word}, Document Frequency: {doc_freq}")
            
def LDA_model(num_topics):
    dictionary, corpus = load_corpus()
    print_word_counts(dictionary, corpus)    
    print("processing the model now")

    filtered_corpus = [doc for doc in corpus if len(doc) >= MIN_TOKEN_LENGTH]
    # Print stats before and after filtering
    print(f"Original corpus size: {len(corpus)} documents")
    print(f"Filtered corpus size: {len(filtered_corpus)} documents")
    print(f"Removed {len(corpus) - len(filtered_corpus)} documents with fewer than {MIN_TOKEN_LENGTH} tokens")
    lda_model = gensim_test.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=NUMBER_OF_PROCESSES)
    # alpha=(50/num_topics), eta = 0.1,
    lda_model.save(MODEL_PATH)
    print("processed all")
    return lda_model
    
def write_topics(lda_model):
    print("writing data to the topic table")
    for idx, topic_list in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic_list))

        if IS_NOT_FACE:
            # Create a Topics object
            print("IS_NOT_FACE new topic is", idx)
            topics_entry = Topics_isnotface(
            topic_id = idx,
            topic = "".join(topic_list)
            )
        elif IS_AFFECT:
            # Create a Topics object
            print("IS_AFFECT new topic is", idx)
            topics_entry = Topics_affect(
            topic_id = idx,
            topic = "".join(topic_list)
            )
        else:
            # Create a Topics object
            topics_entry = Topics(
            topic_id = idx,
            topic = "".join(topic_list)
            )

    # Add the Topics object to the session
        session.add(topics_entry)
        print("Updated topic_id {}".format(idx))
    session.commit()
    return

def write_imagetopics(resultsjson,lda_model_tfidf,dictionary,MY_STOPWORDS):
    global query_start_counter
    print("writing data to the imagetopic table")
    idx_list, topic_list = zip(*lda_model_tfidf.print_topics(-1))
    for i,row in enumerate(resultsjson):
        if VERBOSE: print("row: ",row)
        # mongofy:
        results = mongo_collection.find_one({"image_id": row["image_id"]})
        if results:
            if VERBOSE: print("results: ",results)
            keyword_list=" ".join(pickle.loads(results['tokenized_keyword_list']))
        else:
            print("mongo results are empty, using description instead")
            keyword_list = row["description"]
        if VERBOSE: print(keyword_list)
        # keyword_list=" ".join(pickle.loads(row["tokenized_keyword_list"]))


        # # handles empty keyword_list
        # if keyword_list:
        #     word_list = keyword_list
        # else:
        #     word_list = row["description"]

        bow_vector = dictionary.doc2bow(preprocess(keyword_list,MY_STOPWORDS))

        # #index,score=sorted(lda_model_tfidf[bow_corpus[i]], key=lambda tup: -1*tup[1])[0]
        # index, score = sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1])[0]
        # index2, score2 = sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1])[1]
        # index3, score3 = sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1])[2]
        sorted_topics = sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1 * tup[1])

        # Extract the first topic
        if len(sorted_topics) > 0:
            index, score = sorted_topics[0]
        else:
            index, score = None, None

        # Extract the second topic
        if len(sorted_topics) > 1:
            index2, score2 = sorted_topics[1]
        else:
            index2, score2 = None, None

        # Extract the third topic
        if len(sorted_topics) > 2:
            index3, score3 = sorted_topics[2]
        else:
            index3, score3 = None, None

        if IS_GETTYONLY:            
            if VERBOSE: print("IS_GETTYONLY")
            imagestopics_entry=imagestopics_ALLgetty4faces_isfacemodel(
                image_id=row["image_id"],
                topic_id=index,
                topic_score=score,
                topic_id2=index2,
                topic_score2=score2,
                topic_id3=index3,
                topic_score3=score3
            )
        elif IS_NOT_FACE and not USE_EXISTING_MODEL:            
            if VERBOSE: print("IS_NOT_FACE and USE_EXISTING_MODEL")
            imagestopics_entry=ImagesTopics_isnotface(
                image_id=row["image_id"],
                topic_id=index,
                topic_score=score,
                topic_id2=index2,
                topic_score2=score2,
                topic_id3=index3,
                topic_score3=score3
            )
        elif IS_NOT_FACE and USE_EXISTING_MODEL:            
            if VERBOSE: print("IS_NOT_FACE and USE_EXISTING_MODEL")
            imagestopics_entry=ImagesTopics_isnotface_isfacemodel(
                image_id=row["image_id"],
                topic_id=index,
                topic_score=score,
                topic_id2=index2,
                topic_score2=score2,
                topic_id3=index3,
                topic_score3=score3
            )
        elif IS_AFFECT:            
            if VERBOSE: print("IS_AFFECT")
            print(f"image_id: {row['image_id']}, {keyword_list} topic_id: {index}, topic_score: {score}")
            imagestopics_entry=ImagesTopics_affect(
                image_id=row["image_id"],
                topic_id=index,
                topic_score=score,
                topic_id2=index2,
                topic_score2=score2,
                topic_id3=index3,
                topic_score3=score3
            )
        else:
            if VERBOSE: print("REGULAR")
            imagestopics_entry=ImagesTopics(
                image_id=row["image_id"],
                topic_id=index,
                topic_score=score,
                topic_id2=index2,
                topic_score2=score2,
                topic_id3=index3,
                topic_score3=score3
            )
        session.add(imagestopics_entry)
        # print(f'image_id {row["image_id"]} -- topic_id {index} -- topic tokens {topic_list[index][:100]}')
        # print(f"keyword list {keyword_list}")

        if row["image_id"] % 1000 == 0:
            print("Updated image_id {}".format(row["image_id"]))
            query_start_counter = row["image_id"]


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
        lda_model = gensim_test.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=NUMBER_OF_PROCESSES)
        cm = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
        coher_val_list[i]=cm.get_coherence()
        print("num_topics: ",num_topics,"coherence: ",coher_val_list[i])
    print(num_topics_list,coher_val_list)  # get coherence value

def gen_corpus():
    # this takes the tokenized keyword list and generates a corpus saved to disk
    print("generating corpus, will save here:", DICT_PATH)
    # query = session.query(SegmentTable.tokenized_keyword_list).filter(SegmentTable.tokenized_keyword_list.isnot(None)).limit(QUERY_LIMIT)
    if IS_NOT_FACE:
        query = session.query(SegmentTable.image_id).filter(
            SegmentTable.mongo_tokens.isnot(None)
        ).limit(QUERY_LIMIT)
    elif IS_AFFECT:
        query = session.query(SegmentTable.image_id).filter(
            SegmentTable.mongo_tokens_affect.isnot(None)
        ).limit(QUERY_LIMIT)
    else:
        query = session.query(SegmentTable.image_id).filter(
            SegmentTable.mongo_tokens.isnot(None),
            SegmentTable.face_y > ANGLE*-1,
            SegmentTable.face_y < ANGLE,
            SegmentTable.face_z > ANGLE*-1,
            SegmentTable.face_z < ANGLE
        ).limit(QUERY_LIMIT)
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
    dictionary = gensim_test.corpora.Dictionary(token_lists)
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
    lda_model_tfidf = gensim_test.models.LdaModel.load(MODEL_PATH)
    lda_dict = corpora.Dictionary.load(MODEL_PATH+'.id2word')
    if IS_AFFECT:
        print(f"IS_AFFECT, AFFECT_LIST: {AFFECT_LIST[0]} ALL_KEYWORDS: {ALL_KEYWORDS[0]}")
        # subtract the affect keys from the ALL keywords
        NOT_AFFECT_LIST = [word for word in ALL_KEYWORDS.split(',') if word not in AFFECT_LIST]
        print("NOT_AFFECT_LIST: ", NOT_AFFECT_LIST[0:50])
        MY_STOPWORDS = gensim_test.parsing.preprocessing.STOPWORDS.union(set(NOT_AFFECT_LIST))        
    else:
        MY_STOPWORDS = gensim_test.parsing.preprocessing.STOPWORDS.union(set(GENDER_LIST+ETH_LIST+AGE_LIST))

    print("model loaded successfully")
    while True:
        # go get LIMIT number of items (will duplicate initial select, but only the initial one)
        # LIMIT is set to a reasonably small number, so as to itterate, (not the whole db)
        print("about to SQL:")
        resultsjson = selectSQL()
        print("got results, count is: ",len(resultsjson))
        if len(resultsjson) == 0:
            break

        write_imagetopics(resultsjson,lda_model_tfidf,lda_dict,MY_STOPWORDS)
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




