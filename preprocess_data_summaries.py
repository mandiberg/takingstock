from sqlalchemy import create_engine, select, delete, and_, func, insert
from sqlalchemy.orm import sessionmaker,scoped_session, declarative_base, relationship
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
from my_declarative_base import Base, Clusters, Location, Gender, Images,ImagesTopics, SegmentTable, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Images
#from sqlalchemy.ext.declarative import declarative_base
from mp_sort_pose import SortPose

Base = declarative_base()
VERBOSE = False

# 3.8 M large table (for Topic Model)
# HelperTable_name = "SegmentHelperMar23_headon"

# 7K for topic 7
HelperTable_name = "SegmentHelperApril12_2x2x33x27"

# MM controlling which folder to use
IS_SSD = False

io = DataIO(IS_SSD)
db = io.db
io.db["name"] = "stock"
# io.db["name"] = "ministock"


# Create a database engine
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)



title = 'Please choose your operation: '
options = ["CountGender_Location_so", "CountGender_Topic_so", "CountGender_Location", "CountGender_Topic"]
option, index = pick(options, title)

if index in [0,1]: 
    gender_location = "CountGender_Location_so"
    gender_topic = "CountGender_Topics_so"
elif index in [2,3]: 
    gender_location = "CountGender_Location"
    gender_topic = "CountGender_Topics"

LIMIT= 500000
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

class Topics(Base):
    __tablename__ = 'Topics'
    topic_id = Column(Integer, primary_key=True)
    gender_topics_counts = relationship("CountGender_Topics", back_populates="topics")

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

    # Define a dictionary to store the counts for each location_id
    location_dimension_counts = {}

    # Iterate through the query result
    for row in result:
        location_id, dimension, count = row

        # Check if the location_id exists in the dictionary
        if location_id not in location_dimension_counts:
            # If not, initialize a dictionary for the location_id
            location_dimension_counts[location_id] = {}

        # Store the count for the dimension category under the location_id
        location_dimension_counts[location_id][dimension] = count

    # Now, location_dimension_counts dictionary contains the counts organized by location_id and dimension category
    return(location_dimension_counts)

def save_gender(location_gender_counts, id_type):
    # Iterate through the location_gender_counts dictionary
    for this_id, gender_counts in location_gender_counts.items():
        # Construct the INSERT query
        if id_type == "location_id":
            insert_query = insert(CountGender_Location_so).values(
                location_id=this_id,
                men=(gender_counts.get('men', 0)+gender_counts.get('oldmen', 0)+gender_counts.get('youngmen', 0)),
                nogender=gender_counts.get('nogender', 0),
                nonbinary=gender_counts.get('nonbinary', 0),
                other=gender_counts.get('other', 0),
                trans=gender_counts.get('trans', 0),
                women=(gender_counts.get('women', 0)+gender_counts.get('youngwomen', 0)+gender_counts.get('oldwomen', 0)),
                manandwoman=gender_counts.get('manandwoman', 0),
                intersex=gender_counts.get('intersex', 0),
                androgynous=gender_counts.get('androgynous', 0)
            )
        elif id_type == "topic_id":
            insert_query = insert(CountGender_Topics).values(
                topic_id=this_id,
                men=(gender_counts.get('men', 0)+gender_counts.get('oldmen', 0)+gender_counts.get('youngmen', 0)),
                nogender=gender_counts.get('nogender', 0),
                nonbinary=gender_counts.get('nonbinary', 0),
                other=gender_counts.get('other', 0),
                trans=gender_counts.get('trans', 0),
                women=(gender_counts.get('women', 0)+gender_counts.get('youngwomen', 0)+gender_counts.get('oldwomen', 0)),
                manandwoman=gender_counts.get('manandwoman', 0),
                intersex=gender_counts.get('intersex', 0),
                androgynous=gender_counts.get('androgynous', 0)
            )

        # Execute the INSERT query
        session.execute(insert_query)
        print(f"executed {this_id} successfully.")
    # Commit the changes
    session.commit()

def query_save(select_query, id_type):
    result = session.execute(select_query).fetchall()
    location_gender_counts = pivot_table(result)

    # this is going to have to be mod to handle eth
    save_gender(location_gender_counts, id_type)


def count_gender_location(this_table):
    select_query = select(
        Location.location_id,
        Gender.gender,
        func.count().label('gender_count')
    ).\
    join(this_table, Location.location_id == this_table.location_id).\
    join(Gender, this_table.gender_id == Gender.gender_id).\
    group_by(Location.location_id, Gender.gender)

    query_save(select_query, "location_id")

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

    query_save(select_query, "topic_id")

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

# full Images table queries
elif index == 2:
    count_gender_location(Images)
elif index == 3:
    # get_bg_database()
    count_gender_topic(Images)
        
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
