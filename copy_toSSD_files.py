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
from my_declarative_base import Base, Column, SegmentTable, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

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
# PATH = "/Volumes/RAID54/"
PATH = "/Volumes/RAID18/"
# PATH = "/Volumes/OWC4/"
# PATH = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/"
#where the images are going:
PATH2 = "/Volumes/OWC4/segment_images"
SEGMENTTABLE_NAME = 'SegmentOct20'

# i don'tthink this does anything
IMAGES_THREAD_COUNTER = 0

'''
1   getty - aug17 128175
2   shutterstock - aug17 1534850
3   adobe - aug17 1740113
4   istock - aug17 351105
5   pexels - aug17 11557
6   unsplash - did I DL these???
7   pond5 - renamed
8   123rf - aug17 823542
9   alamy - WIP
10  visualchinagroup - aug17 DONE
11	picxy - aug17 DONE (only 1500?)
12	pixerf
13	imagesbazaar - aug17 DONE
14	indiapicturebudget - aug17 DONE
15	iwaria - aug17 DONE (only 30?)
16	nappy - aug17 DONE (only 3?)
17	picha - aug17 DONE (only 410)
18	afripics


MAIN_FOLDER = "/Volumes/RAID18/images_pond5"
# MAIN_FOLDER = "/Volumes/OWC4/images_alamy"
# MAIN_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/images_picha"
# MAIN_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/afripics_v2/images"

'''

# right now this is only working for one site at a time
SITE_NAME_ID = 7
NEWIMAGES_FOLDER_NAME = IMAGES_FOLDER_NAME = 'images_pond5'
# SITE_NAME_ID = 2
# IMAGES_FOLDER_NAME = 'images_shutterstock'
# NEWIMAGES_FOLDER_NAME = 'images_shutterstock'
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

# Create a database engine
if db['unix_socket']:
    # for MM's MAMP config
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# class SegmentTable(Base):
#     __tablename__ = SEGMENTTABLE_NAME

#     image_id = Column(Integer, primary_key=True)
#     site_name_id = Column(Integer)
#     contentUrl = Column(String(300), nullable=False)
#     imagename = Column(String(200))
#     face_x = Column(DECIMAL(6, 3))
#     face_y = Column(DECIMAL(6, 3))
#     face_z = Column(DECIMAL(6, 3))
#     mouth_gap = Column(DECIMAL(6, 3))
#     face_landmarks = Column(BLOB)
#     bbox = Column(JSON)
#     face_encodings = Column(BLOB)
#     face_encodings68 = Column(BLOB)
#     body_landmarks = Column(BLOB)
#     site_image_id = Column(String(50), nullable=False)




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
