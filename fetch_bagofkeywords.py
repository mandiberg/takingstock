#fetch bok
from sqlalchemy import create_engine, select, delete, distinct, func
from sqlalchemy.orm import sessionmaker,scoped_session
from sqlalchemy.pool import NullPool
from my_declarative_base import Images, BagOfKeywords,Keywords, Encodings_Site2, SegmentTable, SegmentBig, ImagesKeywords,ImagesEthnicity, Encodings  # Replace 'your_module' with the actual module where your SQLAlchemy models are defined
from mp_db_io import DataIO
import pickle
import numpy as np
from pick import pick
import threading
import queue
import csv
import os
import gensim
import pymongo
from pymongo.errors import DuplicateKeyError

# nltk stuff
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *


## this is not the current version. go to fetch_segment_keywords.py

io = DataIO()
db = io.db
# io.db["name"] = "ministock"

# Create a database engine
# engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)

# Create a session
session = scoped_session(sessionmaker(bind=engine))

# Connect to MongoDB
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["stock"]
# currently set up to do the noface segmentbig
mongo_collection = mongo_db["tokens_noface"]

# create a stemmer object for preprocessing
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

# open and read a csv file, and assign each row as an element in a list
def read_csv(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data
# removing all keywords that are stored in gender, ethnicity, and age tables
io.ROOT = "/Users/michaelmandiberg/Documents/GitHub/facemap"
GENDER_LIST = read_csv(os.path.join(io.ROOT, "topic_model", "stopwords_gender.csv"))
ETH_LIST = read_csv(os.path.join(io.ROOT, "topic_model", "stopwords_ethnicity.csv"))
AGE_LIST = read_csv(os.path.join(io.ROOT, "topic_model", "stopwords_age.csv"))                       
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
keys_dict = make_key_dict(os.path.join(io.ROOT, "utilities", "keys", "Keywords_202408151415.csv"))


title = 'Please choose your operation: '
options = ['Create table', 'Fetch keywords list and make tokens', 'Fetch ethnicity list', 'Prune Table where is_face == None','Insert from segment']
option, index = pick(options, title)

LIMIT= 20000
MAXID = 100000000
# Initialize the counter
counter = 0

# Number of threads
num_threads = io.NUMBER_OF_PROCESSES

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


def create_seg_table(row, lock, session):
    # image_id, site_name_id, site_image_id, contentUrl, imagename, description, age_id, gender_id, location_id, face_x, face_y, face_z, mouth_gap, face_landmarks, bbox, face_encodings68, body_landmarks = row
    image_id, site_name_id, site_image_id, contentUrl, imagename, description, age_id, gender_id, location_id, face_x, face_y, face_z, mouth_gap, bbox = row
    
    # Create a BagOfKeywords object
    segment_big = SegmentBig(
        image_id = image_id,
        site_name_id = site_name_id,
        site_image_id = site_image_id,
        contentUrl = contentUrl,
        imagename = imagename,
        description = description,
        age_id = age_id,
        gender_id = gender_id,
        location_id = location_id,
        face_x = face_x,
        face_y = face_y,
        face_z = face_z,
        mouth_gap = mouth_gap,
        # face_landmarks = face_landmarks,
        bbox = bbox,
        # face_encodings68 = face_encodings68,
        # body_landmarks = body_landmarks,
        keyword_list=None,  # Set this to None or your desired value
        tokenized_keyword_list=None,  # Set this to None or your desired value
        ethnicity_list=None,  # Set this to None or your desired value
        mongo_tokens = None
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
        print(f"Created SegmentBig number: {counter}")


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

def fetch_keywords(target_image_id, lock,session):
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
        for token in keyword_list:
            token = clarify_keyword(token.lower())
            if token not in MY_STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result

    # print(f"my ID is {target_image_id}")
    ####
    ## this is the correct regular version, creating a keyword list from the keywords table
    ####

    #global session
    # Build a select query to retrieve keyword_ids for the specified image_id
    select_keyword_ids_query = (
        select(ImagesKeywords.keyword_id)
        .filter(ImagesKeywords.image_id == target_image_id)
    )
    # print(target_image_id)
    # Execute the query and fetch the result as a list of keyword_ids
    result = session.execute(select_keyword_ids_query).fetchall()
    # keyword_ids = [row.keyword_id for row in result]
    # print(keys_dict)
    # for row in result:
    #     print(row.keyword_id) 
    if result:
        # only process if there are keywords, otherwise, skip. 
        keyword_list = [keys_dict[row.keyword_id] for row in result]
    else:
        print("no keywords for image_id: ",target_image_id)
        return
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
    keyword_list_pickle = pickle.dumps(token_list)

    # # Update the BagOfKeywords entry with the corresponding image_id
    # BOK_keywords_entry = (
    #     session.query(BagOfKeywords)
    #     .filter(BagOfKeywords.image_id == target_image_id)
    #     .first()
    # )

    # Update the SegmentBig entry with the corresponding image_id
    BOK_keywords_entry = (
        session.query(SegmentBig)
        .filter(SegmentBig.image_id == target_image_id)
        .first()
    )

    if BOK_keywords_entry:
        # this is the old version, storing in mysql. 
        # BOK_keywords_entry.tokenized_keyword_list = keyword_list_pickle
        # this is the new version, storing in mongo. 

        # print(f"this is my unpickled list {token_list}")

        BOK_keywords_entry.mongo_tokens = 1
        try:
            mongo_collection.insert_one({
                "image_id": target_image_id,
                "tokenized_keyword_list": keyword_list_pickle,
            })
            print(f"Keyword list for image_id {target_image_id} updated successfully.")
        except DuplicateKeyError as e:
            print(f"Duplicate key error for image_id: {target_image_id}")
            print(f"Error details: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


    else:
        print(f"Keywords entry for image_id {target_image_id} not found.")
        
    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
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
    ################# CREATE TABLE ###########
    

    # create_seg_table for whole shebang
    # 5/26/2024 this is how i making SegmentBig_isface 
    # with all the data for bbox/is_face is not None 
    # IS_REGULAR = True
    # 12/27/2024, to work with isnotface, change declarative_base to SegmentBig_isnotface
    # and redefine IS_FACE = 0
    IS_REGULAR = False
    IS_HELPER = True
    # HELPER_SITE = 2

    function=create_seg_table
    if IS_HELPER:
        max_image_id_query = select(func.max(SegmentBig.image_id)).join(Encodings_Site2, SegmentBig.image_id == Encodings_Site2.image_id)
    else:    
        max_image_id_query = select(func.max(SegmentBig.image_id))
    
    # handles first run
    max_image_id = session.execute(max_image_id_query).fetchone()[0]
    if max_image_id is None: max_image_id = 0  

    if IS_REGULAR:
        select_query = select(
            distinct(Images.image_id), Images.site_name_id, Images.site_image_id, 
                Images.contentUrl, Images.imagename, Images.description, Images.age_id, Images.gender_id, Images.location_id, 
                Encodings.face_x, Encodings.face_y, Encodings.face_z, Encodings.mouth_gap, 
                Encodings.bbox).\
            select_from(Images).\
            outerjoin(Encodings, Images.image_id == Encodings.image_id).\
            filter(Images.image_id > max_image_id, Images.image_id < MAXID, Encodings.face_x.is_not(None)).\
            limit(LIMIT)
    elif not IS_REGULAR and not IS_HELPER:
        print("IS_REGULAR is False")
        select_query = select(
            distinct(Images.image_id), Images.site_name_id, Images.site_image_id, 
                Images.contentUrl, Images.imagename, Images.description, Images.age_id, Images.gender_id, Images.location_id, 
                Encodings.face_x, Encodings.face_y, Encodings.face_z, Encodings.mouth_gap, 
                Encodings.bbox).\
            select_from(Images).\
            outerjoin(Encodings, Images.image_id == Encodings.image_id).\
            filter(Images.image_id > max_image_id, Images.image_id < MAXID, Encodings.is_face == 0, Images.site_name_id == 1).\
            limit(LIMIT)

            # removed this from the filter, as I will be moving emb to noSQL
            # Encodings.face_landmarks, Encodings.bbox, Encodings.face_encodings68, Encodings.body_landmarks).\

        # filter(Images.image_id > max_image_id, Encodings.face_x.is_not(None)).\
            # filter(Images.image_id > max_image_id, Encodings.bbox.is_(None)).\
    elif not IS_REGULAR and IS_HELPER:
        print("IS_HELPER is True")
        # will need to refactor this to pull from SegmentHelper, rather than Encodings_Site2 which is deprecated
        select_query = select(
            distinct(Images.image_id), Images.site_name_id, Images.site_image_id, 
                Images.contentUrl, Images.imagename, Images.description, Images.age_id, Images.gender_id, Images.location_id,
                Encodings_Site2.face_x, Encodings_Site2.face_y, Encodings_Site2.face_z, Encodings_Site2.mouth_gap, 
                Encodings_Site2.bbox).\
            select_from(Images).\
            outerjoin(Encodings_Site2, Images.image_id == Encodings_Site2.image_id).\
            filter(Images.image_id > max_image_id, Images.image_id < MAXID, Encodings_Site2.is_face == 0).\
            limit(LIMIT)

    
    # # create_table for just BOW
    # function=create_table
    
    # max_image_id_query = select(func.max(BagOfKeywords.image_id))
    # max_image_id = session.execute(max_image_id_query).fetchone()[0]
    # if max_image_id is None: max_image_id = 0  

    # # select_query = select(distinct(Images.image_id), Images.description, Images.gender_id, Images.age_id, Images.location_id).\
    # #     select_from(Images).outerjoin(BagOfKeywords, Images.image_id == BagOfKeywords.image_id).\
    # #     outerjoin(Encodings, Images.image_id == Encodings.image_id).\
    # #         filter(BagOfKeywords.image_id == None, Encodings.is_face.is_not(None)).limit(LIMIT)

    # select_query = select(distinct(Images.image_id), Images.description, Images.gender_id, Images.age_id, Images.location_id).\
    #     select_from(Images).\
    #     outerjoin(Encodings, Images.image_id == Encodings.image_id).\
    #         filter(Images.image_id > max_image_id, Encodings.is_face.is_not(None)).limit(LIMIT)
    
    # Execute the query and fetch the results
    result = session.execute(select_query).fetchall()
    # print the length of the result
    print(len(result), "rows")
    for row in result:
        # print(row)
        # if len(row) == 9:
        #     # add five None to the end of the list
        #     row = row + [None, None, None, None, None]

        # print(row)
        work_queue.put(row)


elif index == 1:
    function=fetch_keywords
    ################FETCHING KEYWORDS AND CREATING TOKENS #################
    # distinct_image_ids_query = select(BagOfKeywords.image_id.distinct()).filter(BagOfKeywords.tokenized_keyword_list == None).limit(LIMIT)
    distinct_image_ids_query = select(SegmentBig.image_id.distinct()).filter(SegmentBig.mongo_tokens == None).limit(LIMIT)
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
    function=create_table 

    select_query = select(distinct(SegmentTable.image_id), SegmentTable.description, SegmentTable.gender_id, SegmentTable.age_id, SegmentTable.location_id).\
        select_from(SegmentTable).outerjoin(BagOfKeywords, SegmentTable.image_id == BagOfKeywords.image_id).\
            filter(BagOfKeywords.image_id == None).limit(LIMIT)
    result = session.execute(select_query).fetchall()
    # print the length of the result
    print(len(result), "rows")
    for row in result:
        work_queue.put(row)

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
