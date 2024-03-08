# import cv2
# import numpy as np
# import mediapipe as mp
# import matplotlib.pyplot as plt
# import os
# import pandas as pd
# import shutil

# FOLDER_PATH = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/bg_color/0900"
# results = []
# SORTTYPE = "luminosity"  # "hue" or "luminosity"
# output_folder = os.path.join(FOLDER_PATH, SORTTYPE)
# os.makedirs(output_folder, exist_ok=True)

# get_background_mp = mp.solutions.selfie_segmentation
# get_bg_segment = get_background_mp.SelfieSegmentation()

# '''
# now itterates through folder for testing.
# luminosity works well.
# hue isn't as useful, as it is only the chroma, so it treats 95% white with a blue cast the same as a vivid blue

# the next step is to pull from db and write back to db. 
# let's stick to the segment table.
# I revised fetch_bagofkeywords.py, and I would suggest using that as a template for this.
# it uses myslqalchemy now. and the threading is stable

# it needs to pull the bbox, and crop the image to bbox and then run that through the function
# because that is what will actually be in the final image
# most images have even continuous backgrounds, but some have uneven backgrounds that throw off the average.
# '''


# def get_bg_folder(folder_path):
    # for filename in os.listdir(folder_path):
        # if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # file_path = os.path.join(folder_path, filename)
            # hue, lum = get_bg_hue_lum(file_path)
            # results.append({"file": filename, "hue": hue, "luminosity": lum})

    # # Create DataFrame from results and sort by SORTYPE
    # df = pd.DataFrame(results)
    # df_sorted = df.sort_values(by=SORTTYPE)

    # print(df_sorted)

    # # Iterate over sorted DataFrame and save copies of each file to output folder
    # counter = 0
    # total = len(df_sorted)

    # for index, row in df_sorted.iterrows():
        # old_file_path = os.path.join(folder_path, row["file"])
        # filename = f"{str(counter)}_{int(row[SORTTYPE])}_{row['file']}"
        # print(filename)
        # new_file_path = os.path.join(output_folder, filename)
        # shutil.copyfile(old_file_path, new_file_path)
        # print(f"File '{row['file']}' copied to '{filename}'")
        # counter += 1

    # print("Files saved to", output_folder)


# get_bg_folder(FOLDER_PATH)

#################################

from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import sessionmaker,scoped_session
from sqlalchemy.pool import NullPool
from my_declarative_base import Images,ImagesBG,Site  # Replace 'your_module' with the actual module where your SQLAlchemy models are defined
from mp_db_io import DataIO
import pickle
import numpy as np
from pick import pick
import threading
import queue
import csv
import os
import cv2
import mediapipe as mp

io = DataIO()
db = io.db
io.db["name"] = "ministock"

# Create a database engine
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

get_background_mp = mp.solutions.selfie_segmentation
get_bg_segment = get_background_mp.SelfieSegmentation()



# Create a session
session = scoped_session(sessionmaker(bind=engine))

# i was checking which port am i pointing to
# # Execute a query to retrieve database names
# result = session.execute(text("SHOW DATABASES"))

# # Iterate through the result set and print database names
# for row in result:
    # print(row[0])

title = 'Please choose your operation: '
options = ['Create table', 'Fetch BG color stats']
option, index = pick(options, title)

LIMIT= 10
# Initialize the counter
counter = 0

# Number of threads
#num_threads = io.NUMBER_OF_PROCESSES
num_threads = 1

def get_bg_hue_lum(file):
    sample_img = cv2.imread(file)
    result = get_bg_segment.process(sample_img[:,:,::-1])
    mask=np.repeat((1-result.segmentation_mask)[:, :, np.newaxis], 3, axis=2)
    masked_img=mask*sample_img[:,:,::-1]/255 ##RGB format
    # Identify black pixels where R=0, G=0, B=0
    black_pixels_mask = np.all(masked_img == [0, 0, 0], axis=-1)
    # Filter out black pixels and compute the mean color of the remaining pixels
    mean_color = np.mean(masked_img[~black_pixels_mask], axis=0)[np.newaxis,np.newaxis,:] # ~ is negate
    hue=cv2.cvtColor(mean_color, cv2.COLOR_RGB2HSV)[0,0,0]
    lum=cv2.cvtColor(mean_color, cv2.COLOR_RGB2LAB)[0,0,0]
    return hue,lum


def create_table(row, lock, session):
    image_id,imagename,site_name_id = row
    
    # Create a BagOfKeywords object
    images_bg = ImagesBG(
        image_id=image_id,
        hue=None,  # Set this to None or your desired value
        lum=None,  # Set this to None or your desired value
    )
    
    # Add the BagOfKeywords object to the session
    session.add(images_bg)

    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()

    # Print a message to confirm the update
    # print(f"BG list list for image_id {image_id} updated successfully.")
    if counter % 100 == 0:
        print(f"Created Images_BG number: {counter}")


def get_filename(target_image_id):
    ## get the image somehow
    select_image_ids_query = (
        select(Images.site_name_id,Images.imagename)
        .filter(Images.image_id == target_image_id)
    )

    result = session.execute(select_image_ids_query).fetchall()
    site_name_id,imagename=result[0]
    site_specific_root_folder = io.folder_list[site_name_id]
    file=site_specific_root_folder+"/"+imagename  ###os.path.join was acting wierd so had to do this

    return file

def fetch_BG_stat(target_image_id, lock, session):

    file=get_filename(target_image_id)
    #filename=get_filename(imagename)
    hue,lum=get_bg_hue_lum(file)
    
    # Update the BagOfKeywords entry with the corresponding image_id
    ImagesBG_entry = (
        session.query(ImagesBG)
        .filter(ImagesBG.image_id == target_image_id)
        .first()
    )

    if ImagesBG_entry:
        ImagesBG_entry.hue = hue
        ImagesBG_entry.lum = lum
        #session.commit()
        print(f"BG stat for image_id {target_image_id} updated successfully.")
    else:
        print(f"BG stat entry for image_id {target_image_id} not found.")
    
    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()

    return

#######MULTI THREADING##################
# Create a lock for thread synchronization
lock = threading.Lock()
threads_completed = threading.Event()



# Create a queue for distributing work among threads
work_queue = queue.Queue()

if index == 0:
    function=create_table
    ################# CREATE TABLE ###########
    select_query = select(Images.image_id,Images.imagename,Images.site_name_id).\
        select_from(Images).outerjoin(ImagesBG,Images.image_id == ImagesBG.image_id).filter(ImagesBG.image_id == None).limit(LIMIT)
    result = session.execute(select_query).fetchall()
    # print the length of the result
    print(len(result), "rows")
    for row in result:
        work_queue.put(row)
        
elif index == 1:
    function=fetch_BG_stat
    #################FETCHING BG stat####################################
    distinct_image_ids_query = select(ImagesBG.image_id.distinct()).filter(ImagesBG.hue == None).limit(LIMIT)
    distinct_image_ids = [row[0] for row in session.execute(distinct_image_ids_query).fetchall()]
    for target_image_id in distinct_image_ids:
        work_queue.put(target_image_id)        

        
def threaded_fetching():
    while not work_queue.empty():
        param = work_queue.get()
        function(param, lock, session)
        work_queue.task_done()

def threaded_processing():
    thread_list = []
    for _ in range(num_threads):
        thread = threading.Thread(target=threaded_fetching)
        thread_list.append(thread)
        thread.start()
    # Wait for all threads to complete
    for thread in thread_list:
        thread.join()
    # Set the event to signal that threads are completed
    threads_completed.set()
    
threaded_processing()
# Commit the changes to the database
threads_completed.wait()

print("done")
# Close the session
session.commit()
session.close()
