from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import sessionmaker,scoped_session
from sqlalchemy.pool import NullPool
from sqlalchemy import and_, or_, func

from sqlalchemy.ext.declarative import declarative_base
# my ORM
# from my_declarative_base import Base, Clusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Images

from my_declarative_base import Base,Images, ImagesTopics, SegmentBig, SegmentTable, BagOfKeywords,Keywords,ImagesKeywords,ImagesEthnicity, Encodings, Column, Integer, String, DECIMAL, BLOB, ForeignKey, JSON  # Replace 'your_module' with the actual module where your SQLAlchemy models are defined
from mp_db_io import DataIO
import pickle
import numpy as np
from pick import pick
import threading
import queue
import csv
import os
import gensim
from collections import Counter
import pymongo

# nltk stuff
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

############################################
# I think this is the most current version #
############################################

'''
1. Opt 0: add image_id and some encodings info to a SegmentBig table via create_table_from_encodings() with all the image_ids from encodings that are within the x,y,z range
1. Opt 0:
'''
io = DataIO()
db = io.db
# io.db["name"] = "ministock"

VERBOSE = True
SegmentHelper_name = 'SegmentHelperAug16_SegOct20_preAlamy'
TOKEN_COUNT_PATH = "token_counts.csv"

# Create a database engine
if db['unix_socket']:
    # for MM's MAMP config
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

# Create a session
session = scoped_session(sessionmaker(bind=engine))

# Connect to MongoDB
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["stock"]
mongo_collection = mongo_db["tokens"]

# create a stemmer object for preprocessing
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

# open and read a csv file, and assign each row as an element in a list
def read_csv(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data
# removing all keywords that are stored in gender, ethnicity, and age tables
io.ROOT = "/Users/michaelmandiberg/Documents/GitHub/facemap/topic_model"
GENDER_LIST = read_csv(os.path.join(io.ROOT, "stopwords_gender.csv"))
ETH_LIST = read_csv(os.path.join(io.ROOT, "stopwords_ethnicity.csv"))
AGE_LIST = read_csv(os.path.join(io.ROOT, "stopwords_age.csv"))                       
MY_STOPWORDS = gensim.parsing.preprocessing.STOPWORDS.union(set(GENDER_LIST+ETH_LIST+AGE_LIST))

def make_key_dict(filepath):
    keys_dict = {}
    with open(filepath, 'r') as file:
        keys = csv.reader(file)
        next(keys)
        for row in keys:
            # print(row)
            keys_dict[int(row[0])] = row[2]
        return keys_dict
keys_dict = make_key_dict(os.path.join(io.ROOT, "Keywords_202405151718.csv"))


title = 'Please choose your operation: '
options = ['Create helper table', 'Fetch keywords list and make tokens', 'Fetch ethnicity list', 'Prune Table where is_face == None', 
           'move new segment image_ids to existing segment','fetch description/Image metas if None','count tokens',
           'fetch body_landmarks'
           ]
option, index = pick(options, title)

LIMIT= 1
# Initialize the counter
counter = 0

# Number of thread
num_threads = io.NUMBER_OF_PROCESSES

class SegmentHelper(Base):
    __tablename__ = SegmentHelper_name

    image_id = Column(Integer, primary_key=True)

# def create_TempTable(row, lock, session):
#     # image_id, bbox, face_x, face_y, face_z, mouth_gap, face_landmarks, face_encodings68, body_landmarks = row

#     image_id = row[0]

#     # Create a SegmentTable object
#     segment_table = SegmentHelper(
#         image_id=image_id
#     )


#     # # Create a SegmentTable object
#     # segment_table = SegmentHelper(
#     #     image_id=image_id,
#     #     bbox=bbox,
#     #     face_x=face_x,
#     #     face_y=face_y,
#     #     face_z=face_z,
#     #     mouth_gap=mouth_gap,
#     #     face_landmarks=face_landmarks,
#     #     face_encodings68=face_encodings68,
#     #     body_landmarks=body_landmarks
#     # )
    
#     # # Create a BagOfKeywords object
#     # bag_of_keywords = BagOfKeywords(
#     #     image_id=image_id,
#     #     description=description,
#     #     gender_id=gender_id,
#     #     age_id=age_id,
#     #     location_id=location_id,
#     #     keyword_list=None,  # Set this to None or your desired value
#     #     tokenized_keyword_list=None,  # Set this to None or your desired value
#     #     ethnicity_list=None  # Set this to None or your desired value
#     # )
    
#     # Add the BagOfKeywords object to the session
#     session.add(segment_table)

#     with lock:
#         # Increment the counter using the lock to ensure thread safety
#         global counter
#         counter += 1
#         session.commit()

#     # Print a message to confirm the update
#     # print(f"Keyword list for image_id {image_id} updated successfully.")
#     if counter % 1000 == 0:
#         print(f"Created SegmentTable number: {counter}")



def create_table(row, lock, session):
    image_id, description, gender_id, age_id, location_id = row
    
    # Create a BagOfKeywords object
    bag_of_keywords = BagOfKeywords(
        image_id=image_id,
        description=description,
        gender_id=gender_id,
        age_id=age_id,
        location_id=location_id,
        keyword_list=None,  # Set this to None or your desired value
        tokenized_keyword_list=None,  # Set this to None or your desired value
        ethnicity_list=None  # Set this to None or your desired value
    )
    
    # Add the BagOfKeywords object to the session
    session.add(bag_of_keywords)

    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()

    # Print a message to confirm the update
    # print(f"Keyword list for image_id {image_id} updated successfully.")
    if counter % 1000 == 0:
        print(f"Created BagOfKeywords number: {counter}")

def create_table_from_encodings(row, lock, session):
    image_id, bbox, face_x, face_y, face_z, mouth_gap, face_landmarks, face_encodings68, body_landmarks = row
    # print(row)
    # Create a BagOfKeywords object
    segment_big = SegmentBig(
        image_id=image_id,
        # description=None,
        # gender_id=None,
        # age_id=None,
        # location_id=None,
        # keyword_list=None,  # Set this to None or your desired value
        # tokenized_keyword_list=None,  # Set this to None or your desired value
        # ethnicity_list=None,  # Set this to None or your desired value
        face_x = face_x, 
        face_y = face_y, 
        face_z = face_z, 
        mouth_gap = mouth_gap, 
        face_landmarks = face_landmarks, 
        bbox = bbox, 
        face_encodings68 = face_encodings68, 
        body_landmarks = body_landmarks
    )
    

    # Add the BagOfKeywords object to the session
    session.add(segment_big)

    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()

    # Print a message to confirm the update
    # print(f"Keyword list for image_id {image_id} updated successfully.")
    if counter % 1000 == 0:
        print(f"Created BagOfKeywords number: {counter}")

def fetch_description(image_id_with_no_description, lock, session):
    # image_id_with_no_description = row[0]
    select_description_query = (
        select(Images.description)
        .filter(Images.image_id == image_id_with_no_description)
    )

    # Execute the query and fetch the result as a list of keyword_ids
    result = session.execute(select_description_query).fetchall()
    description = result[0][0]
    print(description)

    existing_segment_entry = (
        session.query(SegmentTable)
        .filter(SegmentTable.image_id == image_id_with_no_description)
        .first()
    )

    if existing_segment_entry:
        # print(f"image_id {image_id_with_no_description} will be added .")
        existing_segment_entry.description=description        
        # Add the Segment object to the session
        # session.add(segment_table)
    else:
        print(f"NO ACTION image_id {image_id_with_no_description} already exists.")
        return

    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()
        print("added description: ", image_id_with_no_description , description)

    # Print a message to confirm the update
    # print(f"Keyword list for image_id {image_id} updated successfully.")
    if counter % 1000 == 0:
        print(f"Added description: {counter}")
    

def fetch_images_metadata(image_id_with_no_meta, lock, session):
    # image_id_with_no_meta = row[0]
    select_Images_metas_query = (
        select(
        Images.site_name_id, 
        Images.site_image_id, 
        Images.contentUrl, 
        Images.imagename, 
        Images.description, 
        Images.age_id, 
        Images.gender_id, 
        Images.location_id)
        .filter(Images.image_id == image_id_with_no_meta)
    )

    select_Encodings_metas_query = (
        select( 
        Encodings.face_x, 
        Encodings.face_y, 
        Encodings.face_z, 
        Encodings.mouth_gap, 
        Encodings.face_landmarks, 
        Encodings.bbox, 
        Encodings.face_encodings68, 
        Encodings.body_landmarks)
        .join(Images, Encodings.image_id == Images.image_id)
        .filter(Images.image_id == image_id_with_no_meta)
    )

    existing_segment_entry = (
        session.query(SegmentTable)
        .filter(SegmentTable.image_id == image_id_with_no_meta)
        .first()
    )
    # if no description, add description, and also check for all metas
    if not existing_segment_entry.description:
        # Execute the query and fetch the result as a list of keyword_ids
        result_Images = session.execute(select_Images_metas_query).fetchall()
        # print(result_Images)
        existing_segment_entry.description= result_Images[0][4]
        print(f"image_id {image_id_with_no_meta} Description will be added.")
        if not existing_segment_entry.site_image_id:
            print(f"image_id {image_id_with_no_meta} Image Metas will be added.")
            existing_segment_entry.site_name_id= result_Images[0][0]
            existing_segment_entry.site_image_id= result_Images[0][1]
            existing_segment_entry.contentUrl= result_Images[0][2]
            existing_segment_entry.imagename= result_Images[0][3]
            existing_segment_entry.age_id= result_Images[0][5]
            existing_segment_entry.gender_id=result_Images[0][6]
            existing_segment_entry.location_id=result_Images[0][7]
    else:
        if VERBOSE: print(f"NO ACTION image_id {image_id_with_no_meta} Metas already exists completely.")

    if not existing_segment_entry.gender_id:
        # Execute the query and fetch the result as a list of keyword_ids
        result_Images = session.execute(select_Images_metas_query).fetchall()
        # print(result_Images)
        print(f"image_id {image_id_with_no_meta} Image Metas will be added.")
        print(result_Images[0][5], result_Images[0][6], result_Images[0][7])
        existing_segment_entry.age_id= result_Images[0][5]
        existing_segment_entry.gender_id=result_Images[0][6]
        existing_segment_entry.location_id=result_Images[0][7]
        
    else:
        if VERBOSE: print(f"NO ACTION image_id {image_id_with_no_meta} Metas gender age location already exists completely.")


    if not existing_segment_entry.bbox:
        # Execute the query and fetch the result as a list of keyword_ids
        result_Encodings = session.execute(select_Encodings_metas_query).fetchall()
        # print(result_Encodings)
        existing_segment_entry.face_x= result_Encodings[0][0]
        existing_segment_entry.face_y= result_Encodings[0][1]
        existing_segment_entry.face_z= result_Encodings[0][2]
        existing_segment_entry.mouth_gap= result_Encodings[0][3]
        existing_segment_entry.face_landmarks= result_Encodings[0][4]
        existing_segment_entry.bbox= result_Encodings[0][5]
        existing_segment_entry.face_encodings68= result_Encodings[0][6]
        existing_segment_entry.body_landmarks= result_Encodings[0][7]
        print(f"image_id {image_id_with_no_meta} Encodings will be added.")

    else:
        if VERBOSE: print(f"NO ACTION image_id {image_id_with_no_meta} Encodings already exists.")

    if not existing_segment_entry.description and not existing_segment_entry.bbox:
        if VERBOSE: print(f"NO ACTION image_id {image_id_with_no_meta} hard stop return.")
        return

    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()
        if VERBOSE: print("added metas/encodings: ", image_id_with_no_meta)

    # Print a message to confirm the update
    # print(f"Keyword list for image_id {image_id} updated successfully.")
    if counter % 1000 == 0:
        print(f"Added description: {counter}")
    

def fetch_encodings_bodylandmarks(image_id_with_no_meta, lock, session):

    select_Encodings_metas_query = (
        select( 
        Encodings.face_x, 
        Encodings.face_y, 
        Encodings.face_z, 
        Encodings.mouth_gap, 
        Encodings.face_landmarks, 
        Encodings.bbox, 
        Encodings.face_encodings68, 
        Encodings.body_landmarks)
        .join(Images, Encodings.image_id == Images.image_id)
        .filter(Images.image_id == image_id_with_no_meta)
    )

    existing_segment_entry = (
        session.query(SegmentTable)
        .filter(SegmentTable.image_id == image_id_with_no_meta)
        .first()
    )

    if not existing_segment_entry.body_landmarks:
        # Execute the query and fetch the result as a list of keyword_ids
        result_Encodings = session.execute(select_Encodings_metas_query).fetchall()
        # print(result_Encodings)
        existing_segment_entry.face_x= result_Encodings[0][0]
        existing_segment_entry.face_y= result_Encodings[0][1]
        existing_segment_entry.face_z= result_Encodings[0][2]
        existing_segment_entry.mouth_gap= result_Encodings[0][3]
        existing_segment_entry.face_landmarks= result_Encodings[0][4]
        existing_segment_entry.bbox= result_Encodings[0][5]
        existing_segment_entry.face_encodings68= result_Encodings[0][6]
        existing_segment_entry.body_landmarks= result_Encodings[0][7]
        print(f"image_id {image_id_with_no_meta} Encodings will be added.")

    else:
        if VERBOSE: print(f"NO ACTION image_id {image_id_with_no_meta} Encodings already exists.")

    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()
        if VERBOSE: print("added metas/encodings: ", image_id_with_no_meta)

    # Print a message to confirm the update
    # print(f"Keyword list for image_id {image_id} updated successfully.")
    if counter % 1000 == 0:
        print(f"Added description: {counter}")


def move_segment_to_segment(image_id_to_move, lock, session):
    # image_id_to_move = row[0]

    existing_segment_entry = (
        session.query(SegmentTable)
        .filter(SegmentTable.image_id == image_id_to_move)
        .first()
    )

    if not existing_segment_entry:
        print(f"image_id {image_id_to_move} will be added .")

        # Create a SegmentTable object
        segment_table = SegmentTable(
            image_id=image_id_to_move
        )

        # Add the BagOfKeywords object to the session
        session.add(segment_table)

    else:
        print(f"NO ACTION image_id {image_id_to_move} already exists.")
        return

    
    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()
        print("moved: ",image_id_to_move)

    # Print a message to confirm the update
    # print(f"Keyword list for image_id {image_id} updated successfully.")
    if counter % 1000 == 0:
        print(f"Added to SegmentTable number: {counter}")



def prune_table(image_id, lock, session):
    # Acquire the lock to ensure thread safety
    with lock:
        # Delete rows from BagOfKeywords where image_id matches the provided image_id
        delete_stmt = delete(BagOfKeywords).where(BagOfKeywords.image_id == image_id)        
        session.execute(delete_stmt)        
        session.commit()

        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()

    # Print a message to confirm the update
    # print(f"Keyword list for image_id {image_id} updated successfully.")
    if counter % 100 == 0:
        print(f"Created BagOfKeywords number: {counter} for image_id {image_id}")
    # print("processed: ",image_id)

def preprocess_keywords(target_image_id, lock,session):
    ambig_key_dict = { "black-and-white": "black_and_white", "black and white background": "black_and_white background", "black and white portrait": "black_and_white portrait", "black amp white": "black_and_white", "white and black": "black_and_white", "black and white film": "black_and_white film", "black and white wallpaper": "black_and_white wallpaper", "black and white cover photos": "black_and_white cover photos", "black and white outfit": "black_and_white outfit", "black and white city": "black_and_white city", "blackandwhite": "black_and_white", "black white": "black_and_white", "black friday": "black_friday", "black magic": "black_magic", "black lives matter": "black_lives_matter black_ethnicity", "black out tuesday": "black_out_tuesday black_ethnicity", "black girl magic": "black_girl_magic black_ethnicity", "beautiful black women": "beautiful black_ethnicity women", "black model": "black_ethnicity model", "black santa": "black_ethnicity santa", "black children": "black_ethnicity children", "black history": "black_ethnicity history", "black family": "black_ethnicity family", "black community": "black_ethnicity community", "black owned business": "black_ethnicity owned business", "black holidays": "black_ethnicity holidays", "black models": "black_ethnicity models", "black girl bullying": "black_ethnicity girl bullying", "black santa claus": "black_ethnicity santa claus", "black hands": "black_ethnicity hands", "black christmas": "black_ethnicity christmas", "white and black girl": "white_ethnicity and black_ethnicity girl", "white woman": "white_ethnicity woman", "white girl": "white_ethnicity girl", "white people": "white_ethnicity", "red white and blue": "red_white_and_blue"}
    def clarify_keyword(text):
        # // if text contains either of the strings "black" or "white", replace with "black_and_white"
        if "black" in text or "white" in text:
            for key, value in ambig_key_dict.items():
                text = text.replace(key, value)
            # print("clarified text: ",text, text)
        return text
    def lemmatize_stemming(text):
        return stemmer.stem(lemmatizer.lemmatize(text, pos='v'))
    def preprocess_list(keyword_list):
        result = []
        # text = clarify_keywords(text.lower())
        individual_words = [word for phrase in keyword_list for word in phrase.split()]
        for token in individual_words:
            token = clarify_keyword(token.lower())
            if token not in MY_STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result

    ####
    ## this is the correct regular version, creating a keyword list from the keywords table
    ####

    #global session
    # Build a select query to retrieve keyword_ids for the specified image_id
    select_keyword_ids_query = (
        select(ImagesKeywords.keyword_id)
        .filter(ImagesKeywords.image_id == target_image_id)
    )

    # Execute the query and fetch the result as a list of keyword_ids
    result = session.execute(select_keyword_ids_query).fetchall()
    # keyword_ids = [row.keyword_id for row in result]
    # print(keys_dict)
    # for row in result:
    #     print(row.keyword_id) 
    if result:
        keyword_list = [keys_dict[row.keyword_id] for row in result]
    else:
        print(f"Keywords entry for image_id {target_image_id} not found.")

        select_description_query = (
            select(SegmentTable.description)
            .filter(SegmentTable.image_id == target_image_id)
        )

        # Execute the query and fetch the result as a list of keyword_ids
        result = session.execute(select_description_query).fetchone()
        print(result[0])
        if result[0]:
            keyword_list = result[0].replace(".","").split()
        else:
            print(f"Description entry for image_id {target_image_id} not found.")
            return
        # print(keyword_list)
        
    # print(keyword_list)

    # # this pulls each key text from db - refactoring
    # # Build a select query to retrieve keywords for the specified keyword_ids
    # select_keywords_query = (
    #     select(Keywords.keyword_text)
    #     .filter(Keywords.keyword_id.in_(keyword_ids))
    #     .order_by(Keywords.keyword_id)
    # )
    # # Execute the query and fetch the results as a list of keyword_text
    # result = session.execute(select_keywords_query).fetchall()
    # keyword_list = [row.keyword_text for row in result]

    with lock:

        # prepare the keyword_list (no pickles, return a string)
        token_list = preprocess_list(keyword_list)

    # Pickle the keyword_list
    # print(token_list)
    keyword_list_pickle = pickle.dumps(token_list)

    # have to do the Mongo here

    # Update the BagOfKeywords entry with the corresponding image_id
    #OLD
    # Segment_keywords_entry = (
    #     session.query(SegmentTable)
    #     .filter(SegmentTable.image_id == target_image_id)
    #     .first()
    # )
    #NEW
    
    # create a SegmentBig entry
    SegmentBig_entry = (
        session.query(SegmentBig)
        .filter(SegmentBig.image_id == target_image_id)
        .first()
    )

    # query the mongo collection to see if the tokens exist for image_id
    query = {"image_id": target_image_id}
    result = mongo_collection.find(query)
    if result and token_list:
        print(f"Tokens for image_id {target_image_id} already exist, setting mongo_tokens to 1.")
        SegmentBig_entry.mongo_tokens = 1
        insert_mongo = False
    elif token_list:
        # insert the tokens into the mongo collection
        insert_mongo = True
        SegmentBig_entry.mongo_tokens = 1
        print(f"Keyword list for image_id {target_image_id} will be updated.")
    else:
        print(f"Keywords entry for image_id {target_image_id} not found.")

    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        # commented out for testing
        if insert_mongo: mongo_collection.insert_one({"image_id": target_image_id, "tokenized_keyword_list": keyword_list_pickle})
        session.commit()

    if counter % 10000 == 0:
        print(f"Keyword list updated: {counter}")

    return

def fetch_ethnicity(target_image_id, lock, session):
    select_ethnicity_ids_query = (
        select(ImagesEthnicity.ethnicity_id)
        .filter(ImagesEthnicity.image_id == target_image_id)
    )

    result = session.execute(select_ethnicity_ids_query).fetchall()
    ethnicity_list = [row.ethnicity_id for row in result]

    ethnicity_list_pickle = pickle.dumps(ethnicity_list)

    # Update the BagOfKeywords entry with the corresponding image_id
    BOK_ethnicity_entry = (
        session.query(BagOfKeywords)
        .filter(BagOfKeywords.image_id == target_image_id)
        .first()
    )

    if BOK_ethnicity_entry:
        BOK_ethnicity_entry.ethnicity_list = ethnicity_list_pickle
        #session.commit()
        print(f"Ethnicity list for image_id {target_image_id} updated successfully.")
    else:
        print(f"ethnicity entry for image_id {target_image_id} not found.")
    
    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()

    return



#######MULTI THREADING##################
# Create a lock for thread synchronization
lock = threading.Lock()
threads_completed = threading.Event()



# Create a queue for distributing work among threads
work_queue = queue.Queue()

if index == 0:
    function=create_table_from_encodings
    # function=create_TempTable
    
    ################# CREATE TABLE From ENCODINGS ###########
    # creates a table with every image_id from encodings within the x,y,z range

    # select_query = select(Images.image_id, Images.description, Images.gender_id, Images.age_id, Images.location_id).\
    #     select_from(Images).outerjoin(BagOfKeywords, Images.image_id == BagOfKeywords.image_id).filter(BagOfKeywords.image_id == None, Images.site_name_id.in_([2,4])).limit(LIMIT)
    
    # select max image_id from SegmentBig
    select_query = select(func.max(SegmentBig.image_id))
    result = session.execute(select_query).fetchone()
    max_image_id = result[0]
    print("max image_id: ", max_image_id)

    # this is the regular one to use
    select_query = (
        select(Encodings.image_id, Encodings.bbox, Encodings.face_x, Encodings.face_y, Encodings.face_z, Encodings.mouth_gap, Encodings.face_landmarks, Encodings.face_encodings68, Encodings.body_landmarks)
        # select(Encodings.image_id)
        .select_from(Encodings)
        .outerjoin(SegmentBig, Encodings.image_id == SegmentBig.image_id)
        .filter(SegmentBig.image_id == None)
        .filter(and_(
            Encodings.image_id >= max_image_id,
            Encodings.face_x > -45,
            Encodings.face_x < -20,
            Encodings.face_y > -10,
            Encodings.face_y < 10,
            Encodings.face_z > -10,
            Encodings.face_z < 10
        ))
        .limit(LIMIT)
    )
            # there are NULL image_ids in the Encodings table!!!
            # Encodings.image_id.isnot(None),


    # # this is for one specific topic
    # select_query = (
    #     # select(Encodings.image_id, Encodings.bbox, Encodings.face_x, Encodings.face_y, Encodings.face_z, Encodings.mouth_gap, Encodings.face_landmarks, Encodings.face_encodings68, Encodings.body_landmarks)
    #     select(Encodings.image_id)
    #     .select_from(Encodings)
    #     .join(ImagesTopics, Encodings.image_id == ImagesTopics.image_id)
    #     .outerjoin(SegmentHelper, Encodings.image_id == SegmentHelper.image_id)
    #     .filter(SegmentHelper.image_id == None)

    #     # .filter(and_(
    #     #     Encodings.face_x > -40,
    #     #     Encodings.face_x < -25,
    #     #     Encodings.face_y > -4,
    #     #     Encodings.face_y < 4,
    #     #     Encodings.face_z > -4,
    #     #     Encodings.face_z < 4,
    #     #     ImagesTopics.topic_id == 17
    #     # ))

    #     .filter(and_(
    #         Encodings.face_x > -33,
    #         Encodings.face_x < -27,
    #         Encodings.face_y > -2,
    #         Encodings.face_y < 2,
    #         Encodings.face_z > -2,
    #         Encodings.face_z < 2
    #     ))

    #     .limit(LIMIT)
    # )


    # this is for subselecting the segment table
    # select_query = (
    #     # select(Encodings.image_id, Encodings.bbox, Encodings.face_x, Encodings.face_y, Encodings.face_z, Encodings.mouth_gap, Encodings.face_landmarks, Encodings.face_encodings68, Encodings.body_landmarks)
    #     select(SegmentTable.image_id)
    #     .select_from(SegmentTable)
    #     .outerjoin(SegmentHelper, SegmentTable.image_id == SegmentHelper.image_id)
    #     .filter(SegmentHelper.image_id == None)
    #     .filter(and_(
    #         SegmentTable.face_x > -33,
    #         SegmentTable.face_x < -27,
    #         SegmentTable.face_y > -2,
    #         SegmentTable.face_y < 2,
    #         SegmentTable.face_z > -2,
    #         SegmentTable.face_z < 2
    #     ))

    #     .limit(LIMIT)
    # )

    result = session.execute(select_query).fetchall()
    # print the length of the result
    print(len(result), "rows")
    for row in result:
        work_queue.put(row)


elif index == 1:
    function=preprocess_keywords
    ################FETCHING KEYWORDS AND CREATING TOKENS #################
    distinct_image_ids_query = select(SegmentTable.image_id.distinct()).filter(SegmentTable.mongo_tokens == None).limit(LIMIT)
    distinct_image_ids = [row[0] for row in session.execute(distinct_image_ids_query).fetchall()]

    # print the length of the result
    print(len(distinct_image_ids), "rows")
    for target_image_id in distinct_image_ids:
        work_queue.put(target_image_id)

       
elif index == 2:
    function=fetch_ethnicity
    #################FETCHING ETHNICITY####################################
    distinct_image_ids_query = select(BagOfKeywords.image_id.distinct()).filter(BagOfKeywords.ethnicity_list == None).limit(LIMIT)
    distinct_image_ids = [row[0] for row in session.execute(distinct_image_ids_query).fetchall()]
    for target_image_id in distinct_image_ids:
        work_queue.put(target_image_id)        

elif index == 3:
    function=prune_table
    #################PRUNE THE TABLE#######################################

    # Construct the query to select distinct image_ids where Encodings.is_face is None
    distinct_image_ids_query = select(Encodings.image_id.distinct()).filter(Encodings.is_face == None).limit(LIMIT)

    # Execute the query and fetch the results
    distinct_image_ids = [row[0] for row in session.execute(distinct_image_ids_query).fetchall()]
    print(len(distinct_image_ids), "rows to prune")    
    for target_image_id in distinct_image_ids:
        work_queue.put(target_image_id)

elif index == 4:
    function=move_segment_to_segment
    #################MOVE SEGMENT TO SEGMENT#######################################
    existing_image_ids_query = select(SegmentTable.image_id.distinct()).limit(LIMIT)
    existing_image_ids = [row[0] for row in session.execute(existing_image_ids_query).fetchall()]
    print(len(existing_image_ids), "existing rows")   

    new_image_ids_query = select(SegmentHelper.image_id.distinct()).limit(LIMIT)
    new_image_ids = [row[0] for row in session.execute(new_image_ids_query).fetchall()]
    print(len(new_image_ids), "new rows")   

    # Convert the lists to sets
    existing_image_ids_set = set(existing_image_ids)
    new_image_ids_set = set(new_image_ids)

    # Find the IDs that are present in new_image_ids but not in existing_image_ids
    to_move_image_ids_set = new_image_ids_set - existing_image_ids_set

    # Convert the result back to a list
    to_move_image_ids = list(to_move_image_ids_set)

    # Print the length of the to_move_image_ids list
    print(len(to_move_image_ids), "IDs to move")
    
    for target_image_id in to_move_image_ids:
        work_queue.put(target_image_id)

elif index == 5:
    # function=fetch_description
    function=fetch_images_metadata
    #################PRUNE THE TABLE#######################################

    # No Description
    # distinct_image_ids_query = select(SegmentTable.image_id.distinct()).filter(SegmentTable.description == None).limit(LIMIT)

    # No Metas or description
    distinct_image_ids_query = select(SegmentTable.image_id.distinct())\
        .filter(or_(SegmentTable.description == None, SegmentTable.bbox == None, SegmentTable.gender_id == None))\
        .limit(LIMIT)

    # Execute the query and fetch the results
    distinct_image_ids = [row[0] for row in session.execute(distinct_image_ids_query).fetchall()]
    print(len(distinct_image_ids), "rows without description or bbox")    
    for target_image_id in distinct_image_ids:
        work_queue.put(target_image_id)

elif index == 6:
    #################COUNT ALL TOKENS IN PICKLED LIST#######################################
    # this is not threaded

    # Query all rows from the SegmentTable
    # rows = session.query(SegmentTable).all()
    rows = session.query(SegmentTable.tokenized_keyword_list).all()

    # Initialize a Counter to count the occurrences of each string
    string_counter = Counter()

    # Iterate over each row
    for row in rows:
        try:
            tokenized_keywords = pickle.loads(row.tokenized_keyword_list)
            # Update the counter with these token keywords
            string_counter.update(tokenized_keywords)
        except:
            print("error, probably NULL")
    # Write the counts to a CSV file
    with open(TOKEN_COUNT_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(['count', 'string'])
        # Write the counts
        for string, count in string_counter.items():
            writer.writerow([count, string])

elif index == 7:
    # function=fetch_description
    function=fetch_encodings_bodylandmarks
    #################PRUNE THE TABLE#######################################

    # No bodylandmarks and in specific topic
    distinct_image_ids_query = select(SegmentTable.image_id.distinct())\
        .join(SegmentHelper, SegmentTable.image_id == SegmentHelper.image_id)\
        .filter(SegmentTable.body_landmarks == None)\
        .limit(LIMIT)


    # Execute the query and fetch the results
    distinct_image_ids = [row[0] for row in session.execute(distinct_image_ids_query).fetchall()]
    print(len(distinct_image_ids), "rows without body_landmarks in topic: ", SegmentHelper_name)    
    for target_image_id in distinct_image_ids:
        work_queue.put(target_image_id)


def threaded_fetching():
    while not work_queue.empty():
        param = work_queue.get()
        function(param, lock, session)
        work_queue.task_done()

def threaded_processing():
    thread_list = []
    for _ in range(num_threads):
        thread = threading.Thread(target=threaded_fetching)
        thread_list.append(thread)
        thread.start()
    # Wait for all threads to complete
    for thread in thread_list:
        thread.join()
    # Set the event to signal that threads are completed
    threads_completed.set()
    
threaded_processing()
# Commit the changes to the database
threads_completed.wait()

print("done")
# Close the session
session.commit()
session.close()
