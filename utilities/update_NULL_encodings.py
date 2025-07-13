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
from my_declarative_base import Images, Encodings, Base, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

######## Michael's Credentials ########
# platform specific credentials
io = DataIO()
db = io.db
# overriding DB for testing
io.db["name"] = "stock"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#######################################


engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

SEGMENTTABLE_NAME = 'SegmentAug30Straightahead'

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


# Define the batch size
batch_size = 10000


# currently 3512089 in stock, then 2070

try:
    # Query the Encodings table for image_id where face_encodings is NOT NULL
    results = session.query(Encodings.image_id).filter(Encodings.face_encodings.isnot(None)).limit(10)
    # .all()

    # Initialize counters
    total_processed = 0

    for (image_id,) in results:
        # print(f"Deleting face_encodings for Image ID: {image_id}")

        # Delete the entries in the Encodings table for the specified columns
        session.query(Encodings).filter(Encodings.image_id == image_id).update({
            Encodings.face_encodings: None,
            Encodings.face_encodings_J3: None,
            Encodings.face_encodings_J5: None,
            Encodings.face_encodings68_J3: None,
            Encodings.face_encodings68_J5: None
        })

        total_processed += 1

        # Check if the current batch is ready for commit
        if total_processed % batch_size == 0:
            session.commit()
            print(f"{total_processed} Changes committed, up to and including Image ID: {image_id}")
            # quit()

    # Commit any remaining changes
    session.commit()
    print(f"All changes committed.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the session
    session.close()