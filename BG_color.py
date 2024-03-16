# FOLDER_PATH = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/bg_color/0900"
# results = []
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
import shutil
import pandas as pd
from my_declarative_base import Base, Clusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Images
from sqlalchemy.ext.declarative import declarative_base
from mp_sort_pose import SortPose

Base = declarative_base()
USE_BBOX=True


# MM controlling which folder to use
IS_SSD = True

io = DataIO(IS_SSD)
db = io.db
io.db["name"] = "ministock1023"
# io.db["name"] = "ministock"


# Create a database engine
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

get_background_mp = mp.solutions.selfie_segmentation
get_bg_segment = get_background_mp.SelfieSegmentation()

image_edge_multiplier = [1.5,1.5,2,1.5] # bigger portrait
face_height_output = 500
motion = {
    "side_to_side": False,
    "forward_smile": True,
    "laugh": False,
    "forward_nosmile":  False,
    "static_pose":  False,
    "simple": False,
}

EXPAND = False
ONE_SHOT = False # take all files, based off the very first sort order.
JUMP_SHOT = False # jump to random file if can't find a run

sort = SortPose(motion, face_height_output, image_edge_multiplier,EXPAND, ONE_SHOT, JUMP_SHOT)


# if USE_BBOX:FOLDER_PATH = os.path.join(io.ROOT_PROD, "bg_color/0900_bb")
# else:FOLDER_PATH = os.path.join(io.ROOT_PROD, "bg_color/0900")
FOLDER_PATH = os.path.join(io.ROOT_PROD, "bg_color")
SORTTYPE = "luminosity"  # "hue" or "luminosity"
output_folder = os.path.join(FOLDER_PATH, SORTTYPE)
print(output_folder)
os.makedirs(output_folder, exist_ok=True)


# Create a session
session = scoped_session(sessionmaker(bind=engine))

title = 'Please choose your operation: '
options = ['Create table', 'Fetch BG color stats',"test sorting"]
option, index = pick(options, title)

LIMIT= 1000
# Initialize the counter
counter = 0

# Number of threads
#num_threads = io.NUMBER_OF_PROCESSES
num_threads = 1

class SegmentOct20(Base):
    __tablename__ = 'SegmentOct20'
    seg_image_id=Column(Integer,primary_key=True, autoincrement=True)
    image_id = Column(Integer)
    site_name_id = Column(Integer)
    site_image_id = Column(String(50),nullable=False)
    contentUrl = Column(String(300), nullable=False)
    imagename = Column(String(200))
    age_id = Column(Integer)
    age_detail_id = Column(Integer)
    gender_id = Column(Integer)
    location_id = Column(Integer)
    face_x = Column(DECIMAL(6, 3))
    face_y = Column(DECIMAL(6, 3))
    face_z = Column(DECIMAL(6, 3))
    mouth_gap = Column(DECIMAL(6, 3))
    face_landmarks = Column(BLOB)
    bbox = Column(JSON)
    face_encodings = Column(BLOB)
    face_encodings68 = Column(BLOB)
    body_landmarks = Column(BLOB)


def get_bg_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            hue, lum = get_bg_hue_lum(file_path)
            results.append({"file": filename, "hue": hue, "luminosity": lum})

    # Create DataFrame from results and sort by SORTYPE
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by=SORTTYPE)

    print(df_sorted)

    # Iterate over sorted DataFrame and save copies of each file to output folder
    counter = 0
    total = len(df_sorted)

    for index, row in df_sorted.iterrows():
        old_file_path = os.path.join(folder_path, row["file"])
        filename = f"{str(counter)}_{int(row[SORTTYPE])}_{row['file']}"
        print(filename)
        new_file_path = os.path.join(output_folder, filename)
        shutil.copyfile(old_file_path, new_file_path)
        print(f"File '{row['file']}' copied to '{filename}'")
        counter += 1

    print("Files saved to", output_folder)

def sort_files_onBG():
    # Define the select statement to fetch all columns from the table
    images_bg = ImagesBG.__table__

    # Construct the select query
    query = select([images_bg])

    # Optionally limit the number of rows fetched
    if LIMIT:
        query = query.limit(LIMIT)

    # Execute the query and fetch all results
    result = session.execute(query).fetchall()

    # Convert the result to a DataFrame
    # df = pd.DataFrame(result, columns=result[0].keys()) if result else pd.DataFrame()
    # print(df)
    # return df

    results=[]
    counter = 0

    for row in result:
        image_id =row[0]
        if row[3] > 0:
            hue = row[3]
            lum = row[4]
        else:
            hue = row[1]
            lum = row[2]
        print(hue,lum)
        filename=get_filename(image_id)
        results.append({"file": filename, "hue": hue, "luminosity": lum})

    # if there are positive values for hub_bb and lum_bb (not -1), use those
    # if not, use hue and lum
    
    # Create DataFrame from results and sort by SORTYPE
    df = pd.DataFrame(results)
    print(df)
    df_sorted = df.sort_values(by=SORTTYPE)

    print(df_sorted)
    for index, row in df_sorted.iterrows():
        #old_file_path = os.path.join(folder_path, row["file"])
        old_file_path=row['file']

        filename = f"{str(counter)}_{int(row[SORTTYPE])}_{row['file'].split('/')[-1]}"
        new_file_path = os.path.join(output_folder,filename)
        print(old_file_path, new_file_path)
        shutil.copyfile(old_file_path, new_file_path)
        print(f"File '{row['file']}' copied to '{filename}'")
        counter += 1

    print("Files saved to", output_folder)



def get_bg_hue_lum(img,bbox=None,face_landmarks=None):
    if bbox:
        #sample_img=sample_img[bbox['top']:bbox['bottom'],bbox['left']:bbox['right'],:]
        img = sort.crop_image(img, face_landmarks, bbox)
        #print(type(sample_img),"@@@@@@@@@@@@")
        if img is None: return -1,-1 ## if TOO_BIG==true, checking if cropped image is empty
        
    result = get_bg_segment.process(img[:,:,::-1])
    mask=np.repeat((1-result.segmentation_mask)[:, :, np.newaxis], 3, axis=2)
    masked_img=mask*img[:,:,::-1]/255 ##RGB format
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


def get_filename(target_image_id, return_endfile=False):
    ## get the image somehow
    select_image_ids_query = (
        select(Images.site_name_id,Images.imagename)
        .filter(Images.image_id == target_image_id)
    )

    result = session.execute(select_image_ids_query).fetchall()
    site_name_id,imagename=result[0]
    site_specific_root_folder = io.folder_list[site_name_id]
    file=site_specific_root_folder+"/"+imagename  ###os.path.join was acting wierd so had to do this
    end_file=imagename.split('/')[2]
    if return_endfile: return file,endfile
    return file
 


def get_bbox(target_image_id):
    select_image_ids_query = (
        select(SegmentOct20.bbox,SegmentOct20.face_landmarks)
        .filter(SegmentOct20.image_id == target_image_id)
    )
    result = session.execute(select_image_ids_query).fetchall()
    bbox=result[0][0]
    
    #face_landmarks=faceLms(pickle.loads(result[0][1]))
    face_landmarks=pickle.loads(result[0][1])

    return bbox,face_landmarks
    
def fetch_BG_stat(target_image_id, lock, session):

    file=get_filename(target_image_id)
    #filename=get_filename(imagename)
    img = cv2.imread(file)    
    bbox=None
    facelandmark=None

    hue,lum=get_bg_hue_lum(img,bbox,facelandmark)    
    if USE_BBOX:
        #will do a second round for bbox with same cv2 image
        bbox,facelandmark=get_bbox(target_image_id)
        hue_bb,lum_bb=get_bg_hue_lum(img,bbox,facelandmark)
    
    # Update the BagOfKeywords entry with the corresponding image_id
    ImagesBG_entry = (
        session.query(ImagesBG)
        .filter(ImagesBG.image_id == target_image_id)
        .first()
    )

    if ImagesBG_entry:
        if USE_BBOX:
            ImagesBG_entry.hue_bb = hue_bb
            ImagesBG_entry.lum_bb = lum_bb

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
    if USE_BBOX:distinct_image_ids_query = select(ImagesBG.image_id.distinct()).filter(ImagesBG.hue_bb == None).limit(LIMIT)
    else:distinct_image_ids_query = select(ImagesBG.image_id.distinct()).filter(ImagesBG.hue == None).limit(LIMIT)
    distinct_image_ids = [row[0] for row in session.execute(distinct_image_ids_query).fetchall()]
    for target_image_id in distinct_image_ids:
        work_queue.put(target_image_id)        

elif index == 2:
    # get_bg_database()
    sort_files_onBG()
        
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
if index!=2:
    threaded_processing()
    # Commit the changes to the database
    threads_completed.wait()

print("done")
# Close the session
session.commit()
session.close()
