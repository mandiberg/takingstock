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
PATH = "/Volumes/Test36/"
#where the images are going:
PATH2 = "/Volumes/Test36/"

COPY=True
UNIQUE_FILES_PATH="/Volumes/Test36/scraping phase 2/CSVs_to_ingest/unsplashCSVs/unique_images.csv"
IMAGES_THREAD_COUNTER = 0
IMAGES_FOLDER_NAME = 'images_unsplash'
NEWIMAGES_FOLDER_NAME = 'new_images_unsplash'
NUMBER_OF_THREADS_IMAGES_DOWNLOAD =15
OLDPATH = os.path.join(PATH, IMAGES_FOLDER_NAME)
NEWPATH = os.path.join(PATH2, NEWIMAGES_FOLDER_NAME)

# platform specific credentials
io = DataIO()
db = io.db
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES


engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
metadata = MetaData(engine)

Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

class Ethnicity(Base):
    __tablename__ = 'ethnicity'
    ethnicity_id = Column(Integer, primary_key=True, autoincrement=True)
    ethnicity = Column(String(40))

class Gender(Base):
    __tablename__ = 'gender'
    gender_id = Column(Integer, primary_key=True, autoincrement=True)
    gender = Column(String(20))

class Age(Base):
    __tablename__ = 'age'
    age_id = Column(Integer, primary_key=True, autoincrement=True)
    age = Column(String(20))

class Site(Base):
    __tablename__ = 'site'
    site_name_id = Column(Integer, primary_key=True, autoincrement=True)
    site_name = Column(String(20))

class Location(Base):
    __tablename__ = 'location'
    location_id = Column(Integer, primary_key=True, autoincrement=True)
    location_text = Column(String(50))
    location_number = Column(String(50))
    location_code = Column(String(50))

class Images(Base):
    __tablename__ = 'images'
    image_id = Column(Integer, primary_key=True, autoincrement=True)
    site_name_id = Column(Integer, ForeignKey('site.site_name_id'))
    site_image_id = Column(String(50), nullable=False)
    age_id = Column(Integer, ForeignKey('age.age_id'))
    gender_id = Column(Integer, ForeignKey('gender.gender_id'))
    location_id = Column(Integer, ForeignKey('location.location_id'))
    author = Column(String(100))
    caption = Column(String(150))
    contentUrl = Column(String(300), nullable=False)
    description = Column(String(150))
    imagename = Column(String(200))
    uploadDate = Column(Date)

    site = relationship("Site")
    age = relationship("Age")
    gender = relationship("Gender")
    location = relationship("Location")

class Keywords(Base):
    __tablename__ = 'keywords'
    keyword_id = Column(Integer, primary_key=True, autoincrement=True)
    keyword_number = Column(Integer)
    keyword_text = Column(String(50), nullable=False)
    keytype = Column(String(50))
    weight = Column(Integer)
    parent_keyword_id = Column(String(50))
    parent_keyword_text = Column(String(50))

class ImagesKeywords(Base):
    __tablename__ = 'imageskeywords'
    image_id = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    keyword_id = Column(Integer, ForeignKey('keywords.keyword_id'), primary_key=True)

class ImagesEthnicity(Base):
    __tablename__ = 'imagesethnicity'
    image_id = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    ethnicity_id = Column(Integer, ForeignKey('ethnicity.ethnicity_id'), primary_key=True)

class Encodings(Base):
    __tablename__ = 'encodings'
    encoding_id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.image_id'))
    is_face = Column(Boolean)
    is_body = Column(Boolean)
    is_face_distant = Column(Boolean)
    face_x = Column(DECIMAL(6, 3))
    face_y = Column(DECIMAL(6, 3))
    face_z = Column(DECIMAL(6, 3))
    mouth_gap = Column(DECIMAL(6, 3))
    face_landmarks = Column(BLOB)
    bbox = Column(JSON)
    face_encodings = Column(BLOB)
    body_landmarks = Column(BLOB)


# startpoint = 622000

#setup alphabet list
#long to crate full directory structure
alphabet = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
# alphabet = '0'  
alphabet2 = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
# alphabet = 'A B C 0 1 2'   #short alphabet for testing purposes
# alphabet2 = 'A B C 0 1 2'   #short alphabet for testing purposes
alphabet = alphabet.split()
alphabet2 = alphabet2.split()


def get_hash_folders(filename):
    m = hashlib.md5()
    m.update(filename.encode('utf-8'))
    d = m.hexdigest()
    # csvWriter1.writerow(["https://upload.wikimedia.org/wikipedia/commons/"+d[0]+'/'+d[0:2]+'/'+filename])
    return d[0].upper(), d[0:2].upper()
# print(get_hash_folders(testname))
    
def touch(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def make_hash_folders(path):
    #create depth 0
    for letter in alphabet:
        # print (letter)
        pth = os.path.join(path,letter)
        touch(pth)
        for letter2 in alphabet2:
            # print (letter2)

            pth2 = os.path.join(path,letter,letter+letter2)
            touch(pth2)

# def get_hash_folders(filename):
#     m = hashlib.md5()
#     m.update(filename.encode('utf-8'))
#     d = m.hexdigest()
#     return (os.path.join(d[0], d[0:2]), d)

def generate_local_image_filepath(image_name):
    file_name_path = image_name.split('?')[0]
    file_name = file_name_path.split('/')[-1]
    extension = file_name.split('.')[-1]
    hash_folder, image_hashed_filename = get_hash_folders(file_name)
    print(hash_folder, image_hashed_filename, file_name, extension)
    return os.path.join(
        IMAGES_FOLDER_NAME, hash_folder, '{}.{}'.format(image_hashed_filename, extension))

def generate_local_unhashed_image_filepath(image_name):
    file_name_path = image_name.split('?')[0]
    file_name = file_name_path.split('/')[-1]
    extension = file_name.split('.')[-1]
    hash_folder, image_hashed_filename = get_hash_folders(file_name)
    return os.path.join(NEWIMAGES_FOLDER_NAME, hash_folder,file_name)
        # IMAGES_FOLDER_NAME, hash_folder, '{}.{}'.format(file_name, extension))

def copy_files(source, destination):
    shutil.copy(source, destination)


def unhash_files():
    global IMAGES_THREAD_COUNTER

    def thread(image_hashpath, image_unhashed_path, retry=0):
        global IMAGES_THREAD_COUNTER
        try:
            #THIS IS WHERE I WILL MOVE THE STUFF
            # move(src,dest)
            print(image_hashpath, image_unhashed_path)
            print("moved")
        except:
            if retry < 5:
                thread(image_hashpath, image_unhashed_path, retry+1)
        if IMAGES_THREAD_COUNTER > 0:
            IMAGES_THREAD_COUNTER -= 1

    def read_csv(path):
        with open(path, "r") as f1:
            last_line = f1.readlines()[-1]
        return int(last_line)

    def write_log_csv(path,max_pages):
        headers = ["maxpages"]
        with open(path, 'a') as csvfile: 
            writer=csv.writer(csvfile, delimiter=',')
            writer.writerow([max_pages])

    # check to make sure the old files are actualy there
    try:
        if not os.path.exists(OLDPATH):
            print("[-] No folder here: ", OLDPATH)
            quit()
        #     os.mkdir(IMAGES_FOLDER_NAME)
        # # initialize hash folders
        if not os.path.exists(os.path.join(OLDPATH, "A")):
            print("[-] No folder here: ", OLDPATH)
            quit()
        #     make_hash_folders(IMAGES_FOLDER_NAME)
    except:
        print('[download_images_from_cache] unable to find folder\n', flush=True)

    # check to see if the new folder structure is in place, and if not make it. 
    try:
        if not os.path.exists(NEWPATH):
            os.mkdir(NEWPATH)
        # # initialize hash folders
        if not os.path.exists(os.path.join(NEWPATH, "A")):
            print("no subfolders")
            make_hash_folders(NEWPATH)
            print("just made new hash_folder")
    except:
        print('[download_images_from_cache] unable to create folder\n', flush=True)

    if not os.path.exists(UNIQUE_FILES_PATH):
        print('[-] cache `%s` not found.')
        exit(0)

    with open(UNIQUE_FILES_PATH, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Loop over each row in the file
        start_counter = 0
        counter = start_counter
        alreadyDL = 0
        print("starting from start_counter: ",start_counter)

        for i in range(start_counter):
            next(reader)  # skip the line

        # print('starting to traverse the file, starting from: ',str(start_counter))
        for row in reader:
            # obj = json.loads(item)
            # print(counter)
            # while start_counter > counter:counter is:  
            #     # print("in while")
            #     counter += 1
            #     print("skipping, ",counter)
            #     continue

            if row[0] is None:
                continue
            # if startpoint > 0 and startpoint > counter:
            #     continue

            # this stores images in hashed folders, to ensure reasonable
            # number of files per folder
            image_url = row[0]
            # where the old images is (PATH)
            image_hashpath = os.path.join(PATH,generate_local_image_filepath(image_url))
            # where the new images goes (PATH2)
            image_unhashed_path = os.path.join(PATH2,generate_local_unhashed_image_filepath(image_url.replace('.jpeg','.jpg')))
            # print out to countout every 1000 batches
            # print(image_hashpath, image_unhashed_path)
            # continue
            if start_counter % 10 == 0:
                print("start_counter is: ",start_counter)
            start_counter += 1
            counter += 1

            if counter % 1000 == 0 and counter > start_counter:
                print("counter is: ",counter)
                # write_log_csv(CSV_COUNTOUT_PATH,counter)

            if os.path.isfile(image_hashpath):
                print("this file will be moved", str(counter), image_hashpath)
            else:
                alreadyDL += 1
                print("nobody there", str(alreadyDL), image_hashpath)
                continue
            if IMAGES_THREAD_COUNTER < NUMBER_OF_THREADS_IMAGES_DOWNLOAD:
                Thread(target=thread, args=[
                       image_hashpath, image_unhashed_path], daemon=True).start()
                IMAGES_THREAD_COUNTER += 1
            # print("IMAGES_THREAD_COUNTER ",str(IMAGES_THREAD_COUNTER))
            while IMAGES_THREAD_COUNTER >= NUMBER_OF_THREADS_IMAGES_DOWNLOAD:
                sleep(.1)
                print('[-] Processing batch #%s' %
                      (int(counter / NUMBER_OF_THREADS_IMAGES_DOWNLOAD)), end='\r')
                      # (int(counter / NUMBER_OF_THREADS_IMAGES_DOWNLOAD)),  (int(total_count / NUMBER_OF_THREADS_IMAGES_DOWNLOAD)), end='\r')
    print('[-] All images have been downloaded successfully\n')



def main():
    # Create a session
    session = Session()

    # Continue from the previous code


    subquery = session.query(Images.imagename). \
        join(ImagesKeywords, Images.image_id == ImagesKeywords.image_id). \
        join(Keywords, ImagesKeywords.keyword_id == Keywords.keyword_id). \
        join(ImagesEthnicity, Images.image_id == ImagesEthnicity.image_id). \
        join(Ethnicity, ImagesEthnicity.ethnicity_id == Ethnicity.ethnicity_id). \
        outerjoin(Encodings, Images.image_id == Encodings.image_id). \
        filter(Encodings.encoding_id.isnot(None)). \
        filter(Images.site_name_id == 8). \
        filter(Images.age_id == 5). \
        group_by(Images.imagename). \
        subquery()

    query = session.query(Images.image_id, Images.site_name_id, Images.site_image_id, Images.contentUrl, Images.imagename). \
        join(subquery, Images.imagename == subquery.c.imagename). \
        limit(5000)

    # Fetch the results
    results = query.all()

    # Destination directory
    destination_dir = '/Volumes/Test36/images_123rf_toSSD'

    # Copy the files
    for result in results:
        image_id, site_name_id, site_image_id, contentUrl, imagename = result

        # Source file path
        source_file = f'/Volumes/Test36/images_123rf/{imagename}'

        # Destination file path
        destination_file = f'{destination_dir}/{imagename}'

        # Copy the file
        print(source_file, destination_file)
        # copy_files(source_file, destination_file)

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
