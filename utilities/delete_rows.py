import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool


# go get IO class from parent folder
# caution: path[0] is reserved for script path (or '' in REPL)
import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
# import file
from my_declarative_base import Base, ImagesKeywords
from mp_db_io import DataIO

######## Michael's Credentials ########
# platform specific credentials
io = DataIO()
db = io.db
# overriding DB for testing
io.db["name"] = "ministock1023"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#######################################

# Connect to the database
engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)


Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Define the batch size for deleting rows
batch_size = 10000
deleted = 0

# Continue deleting rows until no rows are left to delete
while True:
    # Execute the delete query in batches

    # for ImagesKeywords
    # delete_query = """
    # DELETE FROM ImagesKeywords 
    # WHERE image_id NOT IN (SELECT image_id FROM SegmentOct20)
    # LIMIT :batch_size;
    # """

    # for ImagesKeywords
    delete_query = """
    DELETE FROM Images
    WHERE image_id NOT IN (SELECT image_id FROM SegmentOct20)
    AND image_id IS NOT NULL
    LIMIT :batch_size;
    """

    
    result = session.execute(delete_query, {"batch_size": batch_size})
    session.commit()
    deleted += batch_size
    print(deleted)
    # Check if no more rows were deleted
    if result.rowcount == 0:
        break

# Close the session
session.close()

