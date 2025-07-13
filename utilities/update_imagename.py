import os
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
from pathlib import Path


# importing from another folder
import sys
ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/facemap/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)
from mp_db_io import DataIO
from my_declarative_base import Images, Base, Clusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

######## Michael's Credentials ########
# platform specific credentials
io = DataIO()
db = io.db
# overriding DB for testing
io.db["name"] = "stock"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#######################################

#######################################
# DEPRECATED, USE CLEANUP.SQL INSTEAD #
# unless creating md5 hash folders    #
#######################################



engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

SEGMENTTABLE_NAME = 'SegmentOct20'

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

# Define the function for generating imagename
def generate_local_unhashed_image_filepath(contentUrl):
    file_name_path = contentUrl.split('?')[0]
    file_name = file_name_path.split('/')[-1]
    if ".jpeg" in file_name:
        file_name = file_name.replace(".jpeg", ".jpg")
    elif ".jpg" in file_name:
        pass
    else: 
        file_name = file_name+".jpg"
        contentUrl = contentUrl+".jpg"
    # extension = file_name.split('.')[-1]
    hash_folder, hash_subfolder = io.get_hash_folders(file_name)
    # print("hash_folder: ", hash_folder)
    # print("hash_subfolder: ", hash_subfolder)
    # print(os.path.join(hash_folder, hash_subfolder, file_name))
    return os.path.join(hash_folder, hash_subfolder, file_name), contentUrl

# Define the function for generating imagename
def generate_site_name_id_filepath(contentUrl):
    file_name_path = contentUrl.split('?')[0]
    file_name = file_name_path.split('/')[-1]
    if ".jpeg" in file_name:
        file_name = file_name.replace(".jpeg", ".jpg")
    elif ".jpg" in file_name:
        pass
    else: 
        file_name = file_name+".jpg"
        contentUrl = contentUrl+".jpg"
    # extension = file_name.split('.')[-1]
    hash_folder, hash_subfolder = io.get_hash_folders(file_name)
    # print("hash_folder: ", hash_folder)
    # print("hash_subfolder: ", hash_subfolder)
    # print(os.path.join(hash_folder, hash_subfolder, file_name))
    return os.path.join(hash_folder, hash_subfolder, file_name), contentUrl

# Define the function for generating imagename
def generate_pond5_filepath(contentUrl):
    # https://images.pond5.com/business-people-and-globe-photo-147339920_iconl_nowm.jpeg
    file_name_path = contentUrl.split('?')[0]
    file_name_full = file_name_path.split('/')[-1]
    file_name = file_name_full.split('-')[-1].replace("_iconl_nowm", "").replace("_iconl_wide_nowm", "")
    if ".jpeg" in file_name:
        file_name = file_name.replace(".jpeg", ".jpg")
    elif ".jpg" in file_name:
        pass
    else: 
        file_name = file_name+".jpg"
        contentUrl = contentUrl+".jpg"
    # remove any leading zeros
    file_name = file_name.lstrip("0")
    
    hash_folder, hash_subfolder = io.get_hash_folders(file_name)
    # print("hash_folder: ", hash_folder)
    # print("hash_subfolder: ", hash_subfolder)
    # print(os.path.join(hash_folder, hash_subfolder, file_name))
    return os.path.join(hash_folder, hash_subfolder, file_name)


# Define the function for generating imagename
def generate_proper_getty_path(imagename):
    # print("imagename: ", imagename)
    # print("imagename type: ", type(imagename))

    new_imagename = imagename.replace("/Volumes/SSD4/images_getty_reDL","")
    new_imagename = new_imagename.replace("/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/getty_33333_china/images_china_lastset","")
    new_imagename = new_imagename.replace("/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/getty_22222_us/images_usa_lastset","")
    new_imagename = new_imagename.replace("/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/getty_22222_serbia/images_serbia_lastset","")
    new_imagename = new_imagename.replace("/images","")
        
    if new_imagename.startswith("/"):
        # Replace "/" with empty string for first character, do only once
        # I had the "//" in the replace above, but it was treating it as an escape character
        new_imagename = new_imagename[0:].replace("/", "", 1)        
    return new_imagename

# Define the batch size
batch_size = 1000


# currently set up for SegmentTable. need to change SegmentTable to Images if you want to use on main table

try:
    # Query the Images table for image_id and contentUrl where site_name_id is 7
    # results = session.query(Images.image_id, Images.site_image_id,Images.site_name_id, SegmentTable.imagename, SegmentTable.contentUrl).filter(SegmentTable.site_name_id == 7).limit(2000)
    # results = session.query(SegmentTable.image_id, SegmentTable.imagename, SegmentTable.contentUrl).filter(SegmentTable.site_name_id == 7).all()
    results = session.query(Images.image_id, Images.imagename, Images.contentUrl).filter(Images.site_name_id == 7).all()

    # Initialize counters
    total_processed = 0
    current_batch = []

    for image_id, imagename, contentUrl in results:
        # for unhashpath
        # new_imagename, contentUrl = generate_local_unhashed_image_filepath(contentUrl)

        # for getty SNAFU
        # print("imagename: ", image_id, imagename, contentUrl)
        # new_imagename = generate_proper_getty_path(imagename)
        new_imagename = generate_pond5_filepath(contentUrl)
        # print("new_imagename: ", new_imagename)
    
        if new_imagename != imagename:
            print(f"Updating Image ID: {image_id}, Imagename: {new_imagename}, contentUrl: {contentUrl}")

            # Update both imagename and contentUrl columns for the current image_id
            current_batch.append((image_id, new_imagename, contentUrl))
        else:
            print(f"-- NO CHANGES Image ID: {image_id}, Imagename: {imagename}, contentUrl: {contentUrl}")
        total_processed += 1

        # Check if the current batch is ready for commit
        if len(current_batch) >= batch_size:
            session.bulk_update_mappings(Images, [{"image_id": image_id, "imagename": new_imagename, "contentUrl": contentUrl} for image_id, new_imagename, contentUrl in current_batch])
            session.commit()
            print(f"{total_processed} Changes committed for {batch_size} rows.")
            current_batch = []
    # Commit any remaining changes
    if current_batch:
        session.bulk_update_mappings(Images, [{"image_id": image_id, "imagename": new_imagename, "contentUrl": contentUrl} for image_id, new_imagename, contentUrl in current_batch])
        session.commit()
        print(f"Changes committed for the remaining {len(current_batch)} rows.")

    print("All changes committed.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the session
    session.close()