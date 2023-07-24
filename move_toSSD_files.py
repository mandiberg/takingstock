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
\\\ (_ (_ | \ 
/// __)__)|_/  
'''

# testname = "https://images.pexels.com/photos/9304005/pexels-photo-9304005.jpeg?auto=compress&cs=tinysrgb&w=1440"
# PATH= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production/gettyimages") 

#where the images are:
PATH = "/Volumes/RAID54/"
#where the images are going:
PATH2 = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/"
SEGMENTTABLE_NAME = 'July15segment123straight'

COPY=True
IMAGES_THREAD_COUNTER = 0

# right now this is only working for one site at a time
IMAGES_FOLDER_NAME = 'images_123rf'
NEWIMAGES_FOLDER_NAME = 'images_123rf'
NUMBER_OF_THREADS_IMAGES_DOWNLOAD =15
OLDPATH = os.path.join(PATH, IMAGES_FOLDER_NAME)
NEWPATH = os.path.join(PATH2, NEWIMAGES_FOLDER_NAME)
CSV_COUNTOUT_PATH = "/Volumes/RAID54/CSVs_to_ingest/123rfCSVs/countout2ssd.csv"

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
    site_image_id = Column(String(50), nullable=False)



# startpoint = 622000

#setup alphabet list
#long to crate full directory structure
# alphabet = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
# # alphabet = '0'  
# alphabet2 = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
# # alphabet = 'A B C 0 1 2'   #short alphabet for testing purposes
# # alphabet2 = 'A B C 0 1 2'   #short alphabet for testing purposes
# alphabet = alphabet.split()
# alphabet2 = alphabet2.split()


# def get_hash_folders(filename):
#     m = hashlib.md5()
#     m.update(filename.encode('utf-8'))
#     d = m.hexdigest()
#     return d[0].upper(), d[0:2].upper()
    
# def touch(folder):
#     if not os.path.exists(folder):
#         os.makedirs(folder)

# def make_hash_folders(path):
#     #create depth 0
#     for letter in alphabet:
#         # print (letter)
#         pth = os.path.join(path,letter)
#         touch(pth)
#         for letter2 in alphabet2:
#             # print (letter2)

#             pth2 = os.path.join(path,letter,letter+letter2)
#             touch(pth2)


def copy_files(source, destination):
    isExist = os.path.exists(destination)
    if not isExist: 
        shutil.copy(source, destination)



def main():

    counter = 1
    start_counter = io.get_counter(CSV_COUNTOUT_PATH)


    io.touch(NEWPATH)
    io.make_hash_folders(NEWPATH)
    # Create a session
    session = Session()

    query = session.query(SegmentTable.image_id, SegmentTable.imagename)\
        .filter(SegmentTable.site_name_id==8)\
        # .limit(100)

    # Fetch the results
    results = query.all()

    print(len(results))
    # Copy the files
    for result in results:
        image_id, imagename = result

        if counter < start_counter:
            counter += 1
            continue

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
