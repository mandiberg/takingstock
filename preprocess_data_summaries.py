from sqlalchemy import create_engine, select, delete, and_, func, insert
from sqlalchemy.orm import sessionmaker,scoped_session, declarative_base, relationship, join
from sqlalchemy.pool import NullPool
# from sqlalchemy.ext.declarative import declarative_base

# from my_declarative_base import Images,ImagesBackground, SegmentTable, Site 
from mp_db_io import DataIO
import pickle
import numpy as np
from pick import pick
import threading
import queue
import csv
import os
import cv2
# import mediapipe as mp
import shutil
import pandas as pd
import json
from my_declarative_base import Base, Clusters, Location, Ethnicity, ImagesEthnicity, Gender, Age, Images,ImagesTopics, SegmentBig, SegmentTable, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Images
#from sqlalchemy.ext.declarative import declarative_base
from mp_sort_pose import SortPose

Base = declarative_base()
VERBOSE = False

# 3.8 M large table (for Topic Model)
# HelperTable_name = "SegmentHelperMar23_headon"

# 7K for topic 7
HelperTable_name = "SegmentBig_isface"

#TK need to ALTER and ADD     missing INT DEFAULT 0,

# MM controlling which folder to use
IS_SSD = False

io = DataIO(IS_SSD)
db = io.db
io.db["name"] = "stock"
# io.db["name"] = "ministock"


# Create a database engine
if db['unix_socket']:
    # for MM's MAMP config
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)




title = 'Please choose your operation: '
options = ["CountGender_Location_so", "CountGender_Topic_so", "CountEthnicity_Location_so", "CountEthnicity_Topics_so", "CountAge_Location_so", "CountAge_Topics_so", 
           "CountGender_Location", "CountGender_Topic", "CountEthnicity_Location", "CountEthnicity_Topics", "CountAge_Location", "CountAge_Topics"]
option, index = pick(options, title)
print(f"Selected option: {option}, index: {index}")
if index <= 5: 
    gender_location = "CountGender_Location_so"
    gender_topic = "CountGender_Topics_so"
    ethnicity_location = "CountEthnicity_Location_so"
    ethnicity_topic = "CountEthnicity_Topics_so"
    age_location = "CountAge_Location_so"
    age_topic = "CountAge_Topics_so"
else: 
    print(f"setting tables for option: {option}, index: {index}")
    gender_location = "CountGender_Location"
    gender_topic = "CountGender_Topics" # placeholder, not functional
    ethnicity_location = "CountEthnicity_Location"
    ethnicity_topic = "CountEthnicity_Topics" # placeholder, not functional
    age_location = "CountAge_Location"
    age_topic = "CountAge_Topics"

# Initialize the counter
counter = 0

# Number of threads
#num_threads = io.NUMBER_OF_PROCESSES
num_threads = 1


Base = declarative_base()

class Location(Base):
    __tablename__ = 'Location'
    location_id = Column(Integer, primary_key=True)
    nation_name = Column(String)
    # Define the relationship with CountGender_Location_so, etc
    gender_location_counts = relationship("CountGender_Location_so", back_populates="location")
    # Define the relationship with CountEthnicity_Location_so
    ethnicity_location_counts = relationship("CountEthnicity_Location_so", back_populates="location")
    # Define the relationship with CountAge_Location_so
    age_location_counts = relationship("CountAge_Location_so", back_populates="location")

class CountGender_Location_so(Base):
    __tablename__ = gender_location

    location_id = Column(Integer, ForeignKey('Location.location_id'), primary_key=True)
    men = Column(Integer, default=0)
    nogender = Column(Integer, default=0)
    oldmen = Column(Integer, default=0)
    oldwomen = Column(Integer, default=0)
    nonbinary = Column(Integer, default=0)
    other = Column(Integer, default=0)
    trans = Column(Integer, default=0)
    women = Column(Integer, default=0)
    youngmen = Column(Integer, default=0)
    youngwomen = Column(Integer, default=0)
    manandwoman = Column(Integer, default=0)
    intersex = Column(Integer, default=0)
    androgynous = Column(Integer, default=0)

    # Define a relationship with the Location table
    location = relationship("Location", back_populates="gender_location_counts")

class CountEthnicity_Location_so(Base):
    __tablename__ = ethnicity_location

    location_id = Column(Integer, ForeignKey('Location.location_id'), primary_key=True)
    POC = Column(Integer, default=0)
    Black = Column(Integer, default=0)
    caucasian = Column(Integer, default=0)
    eastasian = Column(Integer, default=0)
    hispaniclatino = Column(Integer, default=0)
    middleeastern = Column(Integer, default=0)
    mixedraceperson = Column(Integer, default=0)
    nativeamericanfirstnations = Column(Integer, default=0)
    pacificislander = Column(Integer, default=0)
    southasian = Column(Integer, default=0)
    southeastasian = Column(Integer, default=0)
    afrolatinx = Column(Integer, default=0)
    personofcolor = Column(Integer, default=0)

    # Define a relationship with the Location table
    location = relationship("Location", back_populates="ethnicity_location_counts")

class CountAge_Location_so(Base):
    __tablename__ = age_location

    location_id = Column(Integer, ForeignKey('Location.location_id'), primary_key=True)
    baby = Column(Integer, default=0)
    infant = Column(Integer, default=0)
    child = Column(Integer, default=0)
    teenager = Column(Integer, default=0)
    young = Column(Integer, default=0)
    adult = Column(Integer, default=0)
    old = Column(Integer, default=0)

    # Define a relationship with the Location table
    location = relationship("Location", back_populates="age_location_counts")

class Topics(Base):
    __tablename__ = 'Topics'
    topic_id = Column(Integer, primary_key=True)
    # Relationships
    ethnicity_topic_counts = relationship("CountEthnicity_Topics_so", back_populates="topics")
    age_topic_counts = relationship("CountAge_Topics_so", back_populates="topics")

class CountGender_Topics(Base):
    __tablename__ = gender_topic

    topic_id = Column(Integer, ForeignKey('Topics.topic_id'), primary_key=True)
    men = Column(Integer, default=0)
    nogender = Column(Integer, default=0)
    oldmen = Column(Integer, default=0)
    oldwomen = Column(Integer, default=0)
    nonbinary = Column(Integer, default=0)
    other = Column(Integer, default=0)
    trans = Column(Integer, default=0)
    women = Column(Integer, default=0)
    youngmen = Column(Integer, default=0)
    youngwomen = Column(Integer, default=0)
    manandwoman = Column(Integer, default=0)
    intersex = Column(Integer, default=0)
    androgynous = Column(Integer, default=0)

    # Define a relationship with the Location table
    topics = relationship("Topics", back_populates="gender_topics_counts")


class CountEthnicity_Topics_so(Base):
    __tablename__ = ethnicity_topic
    topic_id = Column(Integer, ForeignKey('Topics.topic_id'), primary_key=True)
    # Ethnicity columns
    POC = Column(Integer, default=0)
    Black = Column(Integer, default=0)
    caucasian = Column(Integer, default=0)
    eastasian = Column(Integer, default=0)
    hispaniclatino = Column(Integer, default=0)
    middleeastern = Column(Integer, default=0)
    mixedraceperson = Column(Integer, default=0)
    nativeamericanfirstnations = Column(Integer, default=0)
    pacificislander = Column(Integer, default=0)
    southasian = Column(Integer, default=0)
    southeastasian = Column(Integer, default=0)
    afrolatinx = Column(Integer, default=0)
    personofcolor = Column(Integer, default=0)
    # Define a relationship back to Topics
    topics = relationship("Topics", back_populates="ethnicity_topic_counts")

class CountAge_Topics_so(Base):
    __tablename__ = age_topic
    topic_id = Column(Integer, ForeignKey('Topics.topic_id'), primary_key=True)
    # Age columns
    baby = Column(Integer, default=0)
    infant = Column(Integer, default=0)
    child = Column(Integer, default=0)
    teenager = Column(Integer, default=0)
    young = Column(Integer, default=0)
    adult = Column(Integer, default=0)
    old = Column(Integer, default=0)
    # Define a relationship back to Topics
    topics = relationship("Topics", back_populates="age_topic_counts")

# Define a relationship with the CountGender_Location_so table
Location.gender_location_counts = relationship("CountGender_Location_so", back_populates="location")
Topics.gender_topics_counts = relationship("CountGender_Topics", back_populates="topics")


class HelperTable(Base):
    __tablename__ = HelperTable_name
    seg_image_id=Column(Integer,primary_key=True, autoincrement=True)
    image_id = Column(Integer, primary_key=True, autoincrement=True)


# Create a session
session = scoped_session(sessionmaker(bind=engine))

def pivot_table(result):
    # Define a dictionary to store the counts for each id
    id_dimension_counts = {}

    # Iterate through the query result
    for row in result:
        this_id, dimension, count = row

        # Check if the this_id exists in the dictionary
        if this_id not in id_dimension_counts:
            # If not, initialize a dictionary for the this_id
            id_dimension_counts[this_id] = {}

        # Store the count for the dimension category under the this_id
        id_dimension_counts[this_id][dimension] = count

    # Now, id_dimension_counts dictionary contains the counts organized by this_id and dimension category
    return(id_dimension_counts)

def save_gender(id_dimension_counts, id_type):
    # Iterate through the id_dimension_counts dictionary
    for this_id, dimension_counts in id_dimension_counts.items():
        # Construct the INSERT query
        if id_type == "location_id":
            insert_query = insert(CountGender_Location_so).values(
                location_id=this_id,
                men=(dimension_counts.get('men', 0)+dimension_counts.get('oldmen', 0)+dimension_counts.get('youngmen', 0)),
                nogender=dimension_counts.get('nogender', 0),
                nonbinary=dimension_counts.get('nonbinary', 0),
                other=dimension_counts.get('other', 0),
                trans=dimension_counts.get('trans', 0),
                women=(dimension_counts.get('women', 0)+dimension_counts.get('youngwomen', 0)+dimension_counts.get('oldwomen', 0)),
                manandwoman=dimension_counts.get('manandwoman', 0),
                intersex=dimension_counts.get('intersex', 0),
                androgynous=dimension_counts.get('androgynous', 0)
            )
        elif id_type == "topic_id":
            insert_query = insert(CountGender_Topics).values(
                topic_id=this_id,
                men=(dimension_counts.get('men', 0)+dimension_counts.get('oldmen', 0)+dimension_counts.get('youngmen', 0)),
                nogender=dimension_counts.get('nogender', 0),
                nonbinary=dimension_counts.get('nonbinary', 0),
                other=dimension_counts.get('other', 0),
                trans=dimension_counts.get('trans', 0),
                women=(dimension_counts.get('women', 0)+dimension_counts.get('youngwomen', 0)+dimension_counts.get('oldwomen', 0)),
                manandwoman=dimension_counts.get('manandwoman', 0),
                intersex=dimension_counts.get('intersex', 0),
                androgynous=dimension_counts.get('androgynous', 0)
            )

        # Execute the INSERT query
        session.execute(insert_query)
        print(f"executed {this_id} successfully.")
    # Commit the changes
    session.commit()

def save_age(id_dimension_counts, id_type):
    # Iterate through the id_dimension_counts dictionary
    for this_id, dimension_counts in id_dimension_counts.items():
        # Construct the INSERT query
        if id_type == "location_id":
            insert_query = insert(CountAge_Location_so).values(
                location_id=this_id,
                baby = dimension_counts.get('baby', 0),
                infant = dimension_counts.get('infant', 0),
                child = dimension_counts.get('child', 0),
                teenager = dimension_counts.get('teenager', 0),
                young = dimension_counts.get('young', 0),
                adult = dimension_counts.get('adult', 0),
                old = dimension_counts.get('old', 0),

            )
        elif id_type == "topic_id":
            insert_query = insert(CountAge_Topics_so).values(
                topic_id=this_id,
                baby = dimension_counts.get('baby', 0),
                infant = dimension_counts.get('infant', 0),
                child = dimension_counts.get('child', 0),
                teenager = dimension_counts.get('teenager', 0),
                young = dimension_counts.get('young', 0),
                adult = dimension_counts.get('adult', 0),
                old = dimension_counts.get('old', 0),
            )

        # Execute the INSERT query
        session.execute(insert_query)
        print(f"executed {this_id} successfully.")
    # Commit the changes
    session.commit()

def save_ethnicity(id_dimension_counts, id_type):
    # Iterate through the id_dimension_counts dictionary
    for this_id, dimension_counts in id_dimension_counts.items():
        # Construct the INSERT query
        if id_type == "location_id":

            insert_query = insert(CountEthnicity_Location_so).values(
                location_id=this_id,
                POC = dimension_counts.get(99, 0),
                Black=dimension_counts.get(1, 0),
                caucasian=dimension_counts.get(2, 0),
                eastasian=dimension_counts.get(3, 0),
                hispaniclatino=dimension_counts.get(4, 0),
                middleeastern=dimension_counts.get(5, 0),
                mixedraceperson=dimension_counts.get(6, 0),
                nativeamericanfirstnations=dimension_counts.get(7, 0),
                pacificislander=dimension_counts.get(8, 0),
                southasian=dimension_counts.get(9, 0),
                southeastasian=dimension_counts.get(10, 0),
                afrolatinx=dimension_counts.get(12, 0),
                personofcolor=dimension_counts.get(13, 0)
            )
        elif id_type == "topic_id":
            insert_query = insert(CountEthnicity_Topics_so).values(
                topic_id=this_id,
                POC = dimension_counts.get(99, 0),
                Black=dimension_counts.get(1, 0),
                caucasian=dimension_counts.get(2, 0),
                eastasian=dimension_counts.get(3, 0),
                hispaniclatino=dimension_counts.get(4, 0),
                middleeastern=dimension_counts.get(5, 0),
                mixedraceperson=dimension_counts.get(6, 0),
                nativeamericanfirstnations=dimension_counts.get(7, 0),
                pacificislander=dimension_counts.get(8, 0),
                southasian=dimension_counts.get(9, 0),
                southeastasian=dimension_counts.get(10, 0),
                afrolatinx=dimension_counts.get(12, 0),
                personofcolor=dimension_counts.get(13, 0)
            )

        # Execute the INSERT query
        session.execute(insert_query)
        print(f"executed {this_id} successfully.")
    # Commit the changes
    session.commit()


def query(select_query, select_POC_query=None):
    print(select_query)
    print(select_POC_query)
    

    result = session.execute(select_query).fetchall()
    try:
        print("POC query")
        result_POC = session.execute(select_POC_query).fetchall()
        print(result_POC)
        for row in result_POC:
            this_id, POC_count = row
            new_row = (this_id, 99, POC_count)
            result.append(new_row)
    except:
        print("probably no select_POC_query")
    id_dimension_counts = pivot_table(result)
    return id_dimension_counts



def count_gender_location(this_table):
    select_query = select(
        Location.location_id,
        Gender.gender,
        func.count().label('gender_count')
    ).\
    join(this_table, Location.location_id == this_table.location_id).\
    join(Gender, this_table.gender_id == Gender.gender_id).\
    group_by(Location.location_id, Gender.gender)

    id_dimension_counts = query(select_query)
    save_gender(id_dimension_counts, "location_id")

def count_gender_topic(this_table):

    select_query = select(
        Topics.topic_id,
        Gender.gender,
        func.count().label('gender_count')
    ).\
    join(ImagesTopics, Topics.topic_id == ImagesTopics.topic_id).\
    join(this_table, ImagesTopics.image_id == this_table.image_id).\
    join(Gender, this_table.gender_id == Gender.gender_id).\
    group_by(Topics.topic_id, Gender.gender)

    id_dimension_counts = query(select_query)
    save_gender(id_dimension_counts, "topic_id")

def count_ethnicity_location(this_table):
    select_query = select(
        Location.location_id,
        Ethnicity.ethnicity_id,
        func.count().label('ethnicity_count')
    ).\
    join(this_table, Location.location_id == this_table.location_id).\
    join(ImagesEthnicity, this_table.image_id == ImagesEthnicity.image_id).\
    join(Ethnicity, ImagesEthnicity.ethnicity_id == Ethnicity.ethnicity_id).\
    group_by(Location.location_id, Ethnicity.ethnicity_id)


    select_POC_query = select(
        Location.location_id,
        func.count(func.distinct(ImagesEthnicity.image_id)).label('distinct_POC_count')
    ).\
    join(this_table, Location.location_id == this_table.location_id).\
    join(ImagesEthnicity, this_table.image_id == ImagesEthnicity.image_id).\
    join(Ethnicity, ImagesEthnicity.ethnicity_id == Ethnicity.ethnicity_id).\
    filter(Ethnicity.ethnicity_id != 2).\
    group_by(Location.location_id)


    id_dimension_counts = query(select_query, select_POC_query)
    print(len(id_dimension_counts))
    print(id_dimension_counts)
    save_ethnicity(id_dimension_counts, "location_id")

def count_ethnicity_topic(this_table):
    
    select_query = select(
        Topics.topic_id,
        Ethnicity.ethnicity_id,
        func.count().label('ethnicity_count')
    ).\
    join(ImagesTopics, Topics.topic_id == ImagesTopics.topic_id).\
    join(this_table, ImagesTopics.image_id == this_table.image_id).\
    join(ImagesEthnicity, this_table.image_id == ImagesEthnicity.image_id).\
    join(Ethnicity, ImagesEthnicity.ethnicity_id == Ethnicity.ethnicity_id).\
    group_by(Topics.topic_id, Ethnicity.ethnicity_id)


    select_POC_query = select(
        Topics.topic_id,
        func.count(func.distinct(this_table.image_id)).label('distinct_POC_count')
    ).\
    join(ImagesTopics, Topics.topic_id == ImagesTopics.topic_id).\
    join(this_table, ImagesTopics.image_id == this_table.image_id).\
    join(ImagesEthnicity, this_table.image_id == ImagesEthnicity.image_id).\
    join(Ethnicity, ImagesEthnicity.ethnicity_id == Ethnicity.ethnicity_id).\
    filter(Ethnicity.ethnicity_id != 2).\
    group_by(Topics.topic_id)


    id_dimension_counts = query(select_query, select_POC_query)
    # id_dimension_counts = query(select_query)
    print(len(id_dimension_counts))
    print(id_dimension_counts)
    save_ethnicity(id_dimension_counts, "topic_id")

def count_age_location(this_table):
    select_query = select(
        Location.location_id,
        Age.age,
        func.count().label('age_count')
    ).\
    join(this_table, Location.location_id == this_table.location_id).\
    join(Age, this_table.age_id == Age.age_id).\
    group_by(Location.location_id, Age.age)

    id_dimension_counts = query(select_query)
    save_age(id_dimension_counts, "location_id")

def count_age_topic(this_table):

    select_query = select(
        Topics.topic_id,
        Age.age,
        func.count().label('age_count')
    ).\
    join(ImagesTopics, Topics.topic_id == ImagesTopics.topic_id).\
    join(this_table, ImagesTopics.image_id == this_table.image_id).\
    join(Age, this_table.age_id == Age.age_id).\
    group_by(Topics.topic_id, Age.age)

    id_dimension_counts = query(select_query)
    save_age(id_dimension_counts, "topic_id")



#######MULTI THREADING##################
# Create a lock for thread synchronization
lock = threading.Lock()
threads_completed = threading.Event()



# Create a queue for distributing work among threads
work_queue = queue.Queue()

# segment table queries
if index == 0:
    count_gender_location(SegmentTable)
elif index == 1:
    count_gender_topic(SegmentTable)

elif index == 2:
    count_ethnicity_location(SegmentTable)
elif index == 3:
    count_ethnicity_topic(SegmentTable)
elif index == 4:
    count_age_location(SegmentTable)
elif index == 5:
    count_age_topic(SegmentTable)

# full Images table queries
elif index == 6:
    count_gender_location(SegmentBig)
elif index == 7:
    count_gender_topic(SegmentBig)
elif index == 8:    
    # get_bg_database()
    count_ethnicity_location(SegmentBig)
elif index == 9:
    count_ethnicity_topic(SegmentBig)
elif index == 10:
    count_age_location(SegmentBig)
elif index == 12:
    count_age_topic(SegmentBig)
        
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
if index!=2:
    threaded_processing()
    # Commit the changes to the database
    threads_completed.wait()

print("done")
# Close the session
session.commit()
session.close()
