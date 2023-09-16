import os
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool

# importing from another folder
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from mp_db_io import DataIO
from my_declarative_base import Images

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

try:
    # Query the Images table for image_id and contentUrl where site_name_id is 1
    result = session.query(Images.image_id, Images.contentUrl).filter(Images.site_name_id == 1).all()

    # Iterate through the results and update imagename
    for image_id, contentUrl in result:
        imagename, contentUrl = generate_local_unhashed_image_filepath(contentUrl)
        print(f"Updating Image ID: {image_id}, Imagename: {imagename}, contentUrl: {contentUrl}")

        # Update both imagename and contentUrl columns for the current image_id
        # session.query(Images).filter(Images.image_id == image_id).update({
        #     "imagename": imagename,
        #     "contentUrl": contentUrl
        # })
    
    # Commit the changes
    session.commit()
    print("Changes committed.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the session
    session.close()
