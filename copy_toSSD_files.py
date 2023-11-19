import hashlib
import os
import shutil
from shutil import move
import re
import csv
from time import sleep

from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, Boolean, DECIMAL, BLOB, JSON, String, Date, ForeignKey, update, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from my_declarative_base import Base, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from threading import Thread

from mp_db_io import DataIO

sig = '''
     __ __ _  
\\\\ (_ (_ | \\ 
/// __)__)|_/  
'''

# testname = "https://images.pexels.com/photos/9304005/pexels-photo-9304005.jpeg?auto=compress&cs=tinysrgb&w=1440"
# PATH= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production/gettyimages") 

# this script COPIES files from one PATH to PATH2 without deleting
# for moving segment to SSD

#where the images are:
PATH = "/Volumes/SSD4/"
#where the images are going:
PATH2 = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/segment_images"
SEGMENTTABLE_NAME = 'SegmentNov19'

# i don'tthink this does anything
IMAGES_THREAD_COUNTER = 0

'''
1   getty
2   shutterstock
3   adobe
4   istock
5   pexels
6   unsplash
7   pond5
8   123rf
9   alamy
10  visualchinagroup
'''

# right now this is only working for one site at a time
SITE_NAME_ID = 4
IMAGES_FOLDER_NAME = 'images_istock'
NEWIMAGES_FOLDER_NAME = 'images_istock'
NUMBER_OF_THREADS_IMAGES_DOWNLOAD =15
OLDPATH = os.path.join(PATH, IMAGES_FOLDER_NAME)
NEWPATH = os.path.join(PATH2, NEWIMAGES_FOLDER_NAME)
CSV_COUNTOUT_PATH = os.path.join(PATH2,"countout2ssd.csv")

# platform specific credentials
io = DataIO()
db = io.db
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES


# engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
#                                 .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

# MAMP
engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

class SegmentTable(Base):
    __tablename__ = SEGMENTTABLE_NAME

    image_id = Column(Integer, primary_key=True)
    site_name_id = Column(Integer)
    contentUrl = Column(String(300), nullable=False)
    imagename = Column(String(200))
    face_x = Column(DECIMAL(6, 3))
    face_y = Column(DECIMAL(6, 3))
    face_z = Column(DECIMAL(6, 3))
    mouth_gap = Column(DECIMAL(6, 3))
    face_landmarks = Column(BLOB)
    bbox = Column(JSON)
    face_encodings = Column(BLOB)
    face_encodings68 = Column(BLOB)
    body_landmarks = Column(BLOB)
    site_image_id = Column(String(50), nullable=False)




def copy_files(source, destination):
    isExist = os.path.exists(destination)
    try:
        if not isExist: 
            shutil.copy(source, destination)
    except Exception as e:
        print(e)

def main():

    counter = 1
    start_counter = io.get_counter(CSV_COUNTOUT_PATH)


    io.touch(NEWPATH)
    io.make_hash_folders(NEWPATH)
    # Create a session
    session = Session()

    query = session.query(SegmentTable.image_id, SegmentTable.imagename)\
        .filter(SegmentTable.site_name_id==SITE_NAME_ID)\
        # .limit(100)

    # Fetch the results
    results = query.all()

    print(str(len(results)), "images")
    # Copy the files
    for result in results:
        image_id, imagename = result

        # temporarily remove this counter function
        # if counter < start_counter:
        #     counter += 1
        #     continue

        # Source file path
        source_file = os.path.join(OLDPATH, imagename)
        
        # Destination file path
        destination_file = os.path.join(NEWPATH,imagename)

        # Copy the file
        # print(source_file, destination_file)
        copy_files(source_file, destination_file)

        if counter % 1000 == 0:
            print(counter)
            save_counter = [counter]
            io.write_csv(CSV_COUNTOUT_PATH, save_counter)
        
        counter += 1

    print("moved the files")
    # Close the session
    session.close()




if __name__ == '__main__':
    print(sig)
    try:
        main()
    except KeyboardInterrupt as _:
        print('[-] User cancelled.\n', flush=True)
    except Exception as e:
        print('[__main__] %s' % str(e), flush=True)
