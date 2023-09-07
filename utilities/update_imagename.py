import os
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from my_declarative_base import Images, create_engine

# importing from another folder
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from mp_db_io import DataIO

######## Michael's Credentials ########
# platform specific credentials
io = DataIO()
db = io.db
# overriding DB for testing
io.db["name"] = "gettytest3"
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
def generate_local_unhashed_image_filepath(image_name):
    file_name_path = image_name.split('?')[0]
    file_name = file_name_path.split('/')[-1].replace(".jpeg", ".jpg")
    # extension = file_name.split('.')[-1]
    hash_folder, hash_subfolder = io.get_hash_folders(file_name)
    print("hash_folder: ", hash_folder)
    print("hash_subfolder: ", hash_subfolder)
    print(os.path.join(hash_folder, hash_subfolder, file_name))
    return os.path.join(hash_folder, hash_subfolder, file_name)

try:
    # Query the Images table for image_id and contentUrl where site_name_id is 1
    result = session.query(Images.image_id, Images.contentUrl).filter(Images.site_name_id == 1).all()

    # Iterate through the results and update imagename
    for image_id, contentUrl in result:
        imagename = generate_local_unhashed_image_filepath(contentUrl)
        print(f"Updating Image ID: {image_id}, Imagename: {imagename}")

        # Update the imagename for the current image_id
        session.query(Images).filter(Images.image_id == image_id).update({"imagename": imagename})
    
    # Commit the changes
    session.commit()
    print("Changes committed.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the session
    session.close()
