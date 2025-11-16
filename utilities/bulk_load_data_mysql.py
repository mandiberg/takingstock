import sqlalchemy
from sqlalchemy import create_engine, select, distinct, and_, or_, not_, update, insert, text
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.mysql import insert as mysql_insert
import pandas as pd

# go get IO class from parent folder
# caution: path[0] is reserved for script path (or '' in REPL)
import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
# import file
from my_declarative_base import Base, SegmentTable, ImagesKeywords, Encodings, Images, Column, Integer, DECIMAL, BLOB, String, JSON
from mp_db_io import DataIO

######## Michael's Credentials ########
# platform specific credentials
io = DataIO()
db = io.db
# overriding DB for testing
# io.db["name"] = "gettytest3"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#######################################

# ADD TO SEGMENTS WHEN SQL/WORKBENCH IS BORKING

HelperTable_name = "SegmentHelper_nov2025_SQL_only_still"


engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

Base.metadata.bind = engine

Session = sessionmaker(bind=engine)
session = Session()

class HelperTable(Base):
    __tablename__ = HelperTable_name
    seg_image_id=Column(Integer,primary_key=True, autoincrement=True)
    image_id = Column(Integer, primary_key=True, autoincrement=True)


# engine = create_engine("mysql+pymysql://user:pass@localhost/db?unix_socket=/Applications/MAMP/tmp/mysql/mysql.sock")

chunksize = 50000
reader = pd.read_csv("/Volumes/OWC4/segment_images/mongo_exports_oct19_sets/still_sql_only_encodings_allunique.txt", names=["image_id"])

# if first row in reader first column is "image_id", skip it
first_row = next(reader.iterrows())[1]
if first_row[0] == "image_id":
    print("Skipping header row")
    reader = pd.read_csv("/Users/michaelmandiberg/Documents/projects-active/facemap_production/validating_sql_mongo/Encodings_202511152103_faces_without_bbox_etc.csv", names=["image_id"], skiprows=1)
reader.to_sql(HelperTable_name, con=engine, if_exists="append", index=False, chunksize=chunksize)
