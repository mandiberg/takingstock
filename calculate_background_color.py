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

from sqlalchemy import create_engine, select, delete, and_
from sqlalchemy.orm import sessionmaker,scoped_session, declarative_base
from sqlalchemy.pool import NullPool
# from my_declarative_base import Images,ImagesBackground, SegmentTable, Site 
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
import json
from my_declarative_base import Base, Clusters, Images,ImagesBackground, SegmentTable, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Images
#from sqlalchemy.ext.declarative import declarative_base
from mp_sort_pose import SortPose
import pymongo

Base = declarative_base()
USE_BBOX=True
VERBOSE = True

# 3.8 M large table (for Topic Model)
# HelperTable_name = "SegmentHelperMar23_headon"

# 7K for topic 7
# HelperTable_name = "SegmentHelperApril12_2x2x33x27"

# for fingerpoint
HelperTable_name = "SegmentHelperMay7_fingerpoint"
# MM controlling which folder to use
IS_SSD = False

io = DataIO(IS_SSD)
db = io.db
# io.db["name"] = "stock"
# io.db["name"] = "ministock"
mongo_client = pymongo.MongoClient(io.dbmongo['host'])
mongo_db = mongo_client[io.dbmongo['name']]
mongo_collection = mongo_db[io.dbmongo['collection']]

# Create a database engine
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

get_background_mp = mp.solutions.selfie_segmentation
get_bg_segment = get_background_mp.SelfieSegmentation()

image_edge_multiplier = [1.5,1.5,2,1.5] # bigger portrait
image_edge_multiplier_sm = [1.2, 1.2, 1.6, 1.2] # standard portrait
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

sort = SortPose(motion, face_height_output, image_edge_multiplier_sm,EXPAND, ONE_SHOT, JUMP_SHOT)


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

LIMIT= 10
# Initialize the counter
counter = 0

# Number of threads
#num_threads = io.NUMBER_OF_PROCESSES
num_threads = 1

class SegmentTable(Base):
    # __tablename__ = 'SegmentTable'
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

class HelperTable(Base):
    __tablename__ = HelperTable_name
    seg_image_id=Column(Integer,primary_key=True, autoincrement=True)
    image_id = Column(Integer, primary_key=True, autoincrement=True)

# def get_bg_folder(folder_path):
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
#             file_path = os.path.join(folder_path, filename)
#             hue, lum = get_bg_hue_lum(file_path)
#             results.append({"file": filename, "hue": hue, "luminosity": lum})

#     # Create DataFrame from results and sort by SORTYPE
#     df = pd.DataFrame(results)
#     df_sorted = df.sort_values(by=SORTTYPE)

#     print(df_sorted)

#     # Iterate over sorted DataFrame and save copies of each file to output folder
#     counter = 0
#     total = len(df_sorted)

#     for index, row in df_sorted.iterrows():
#         old_file_path = os.path.join(folder_path, row["file"])
#         filename = f"{str(counter)}_{int(row[SORTTYPE])}_{row['file']}"
#         print(filename)
#         new_file_path = os.path.join(output_folder, filename)
#         shutil.copyfile(old_file_path, new_file_path)
#         print(f"File '{row['file']}' copied to '{filename}'")
#         counter += 1

#     print("Files saved to", output_folder)

def sort_files_onBG():
    # Define the select statement to fetch all columns from the table
    images_bg = ImagesBackground.__table__

    # Construct the select query
    #query = select([images_bg]) ## this DOESNT work on windows somehow
    query = select(images_bg)

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
    #####################
    #make sure that in my_declarative_base and in database both the sequence is
    #hue,lum,sat,hue_bb,lum_bb,sat_bb
    # and NOT
    #hue,lum,hue_bb,lum_bb,sat,sat_bb
    #####################
    
    for row in result:
        image_id =row[0]
        if row[4] > 0:
            hue = row[4]
            lum = row[5]
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

def get_selfie_bbox(segmentation_mask):
    bbox=None
    scaled_mask = (segmentation_mask * 255).astype(np.uint8)
    # Apply a binary threshold to get a binary image
    _, binary = cv2.threshold(scaled_mask, 127, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Assume the largest contour is the shape
        contour = max(contours, key=cv2.contourArea)
        # Get the bounding box of the shape
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the bounding box for visualization
        bbox={"top":y,"right":scaled_mask.shape[1] - (x + w),"bottom":scaled_mask.shape[0] - (y + h),"left":x}
    else:
        print("No contours were found")
    if bbox is None: print("bbox is empty, figure out what happened")
    else:
        if VERBOSE:print("bbox=",bbox)
    return bbox

def get_segmentation_mask(img,bbox=None,face_landmarks=None):
    print("[get_bg_hue_lum] about to go for segemntation")

    if bbox:
        try:
            if type(bbox)==str:
                bbox=json.loads(bbox)
                if VERBOSE: print("bbox type", type(bbox))
            #sample_img=sample_img[bbox['top']:bbox['bottom'],bbox['left']:bbox['right'],:]
            # passing in bbox as a str
            img = sort.crop_image(img, face_landmarks, bbox)
            if img is None: return -1,-1,-1,-1,-1 ## if TOO_BIG==true, checking if cropped image is empty
        except:
            if VERBOSE: print("FAILED CROPPING, bad bbox",bbox)
            return -2,-2,-2,-2,-2
        print("bbox['bottom'], ", bbox['bottom'])

    result = get_bg_segment.process(img[:,:,::-1]) #convert RBG to BGR then process with mp
    print("[get_bg_hue_lum] got result")
    return result.segmentation_mask


# def get_bg_hue_lum(img,segmentation_mask,bbox):
#     hue = sat = val = lum = lum_torso = None
    
#     mask      =np.repeat((1-segmentation_mask)[:, :, np.newaxis], 3, axis=2) 
#     mask_torso=np.repeat((segmentation_mask)[:, :, np.newaxis], 3, axis=2) 

#     masked_img=mask*img[:,:,::-1]/255 ##RGB format
#     masked_img_torso=mask_torso*img[:,:,::-1]/255 ##RGB format

#     # Identify black pixels where R=0, G=0, B=0
#     black_pixels_mask = np.all(masked_img == [0, 0, 0], axis=-1)
#     black_pixels_mask_torso = np.all(masked_img_torso == [0, 0, 0], axis=-1)

#     # Filter out black pixels and compute the mean color of the remaining pixels
#     mean_color = np.mean(masked_img[~black_pixels_mask], axis=0)[np.newaxis,np.newaxis,:] # ~ means negate/remove
#     hue=cv2.cvtColor(mean_color, cv2.COLOR_RGB2HSV)[0,0,0]
#     sat = cv2.cvtColor(mean_color, cv2.COLOR_RGB2HSV)[0, 0, 1]
#     val = cv2.cvtColor(mean_color, cv2.COLOR_RGB2HSV)[0, 0, 2]
#     lum=cv2.cvtColor(mean_color, cv2.COLOR_RGB2LAB)[0,0,0]

#     if VERBOSE: print("NOTmasked_img_torso size", masked_img_torso.shape, black_pixels_mask_torso.shape)
#     if bbox:
#         # SJ something is broken in here. It returns an all black image which produces a lum of 100
#         masked_img_torso = masked_img_torso[bbox['bottom']:]
#         black_pixels_mask_torso = black_pixels_mask_torso[bbox['bottom']:]
#     # else:
#     #     print("YIKES! no bbox. Here's a hacky hack to crop to the bottom 20%")
#     #     bottom_fraction = masked_img_torso.shape[0] // 5
#     #     masked_img_torso = masked_img_torso[-bottom_fraction:]
#     #     black_pixels_mask_torso = black_pixels_mask_torso[-bottom_fraction:]

#     if VERBOSE: print("masked_img_torso size", masked_img_torso.shape, black_pixels_mask_torso.shape)
#     mean_color = np.mean(masked_img_torso[~black_pixels_mask_torso], axis=0)[np.newaxis,np.newaxis,:] # ~ is negate
#     lum_torso=cv2.cvtColor(mean_color, cv2.COLOR_RGB2LAB)[0,0,0]

#     if VERBOSE: 
#         print("HSV, lum", hue,sat,val,lum, lum_torso)
#     return hue,sat,val,lum,lum_torso


def create_table(row, lock, session):
    image_id,imagename,site_name_id = row
    
    # Create a BagOfKeywords object
    images_bg = ImagesBackground(
        image_id=image_id,
        hue=None,  # Set this to None or your desired value
        lum=None,  # Set this to None or your desired value
        sat = None,
        val = None,
        lum_torso = None,
        hue_bb = None,
        lum_bb = None,
        sat_bb = None,
        val_bb = None,
        lum_torso_bb = None
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
        select(SegmentTable.site_name_id,SegmentTable.imagename)
        .filter(SegmentTable.image_id == target_image_id)
    )

    result = session.execute(select_image_ids_query).fetchall()
    site_name_id,imagename=result[0]
    site_specific_root_folder = io.folder_list[site_name_id]
    file=site_specific_root_folder+"/"+imagename  ###os.path.join was acting wierd so had to do this
    end_file=imagename.split('/')[2]
    if VERBOSE:print("file name:",file)
    if return_endfile: return file,end_file
    return file
 
def get_landmarks_mongo(image_id):
    if image_id:
        results = mongo_collection.find_one({"image_id": image_id})
        if results:
            face_landmarks = results['face_landmarks']
            # print("got encodings from mongo, types are: ", type(face_encodings68), type(face_landmarks), type(body_landmarks))
            return face_landmarks
        else:
            return None
    else:
        return None

def unpickle_array(pickled_array):
    if pickled_array:
        try:
            # Attempt to unpickle using Protocol 3 in v3.7
            return pickle.loads(pickled_array, encoding='latin1')
        except TypeError:
            # If TypeError occurs, unpickle using specific protocl 3 in v3.11
            # return pickle.loads(pickled_array, encoding='latin1', fix_imports=True)
            try:
                # Set the encoding argument to 'latin1' and protocol argument to 3
                obj = pickle.loads(pickled_array, encoding='latin1', fix_imports=True, errors='strict', protocol=3)
                return obj
            except pickle.UnpicklingError as e:
                print(f"Error loading pickle data: {e}")
                return None
    else:
        return None

def get_bbox(target_image_id):
    select_image_ids_query = (
        select(SegmentTable.bbox)
        .filter(SegmentTable.image_id == target_image_id)
    )
    result = session.execute(select_image_ids_query).fetchall()
    bbox=result[0][0]
    
    face_landmarks=unpickle_array(get_landmarks_mongo(target_image_id))

    return bbox,face_landmarks
    
def fetch_BG_stat(target_image_id, lock, session):
    ImagesBG_entry = (
        session.query(ImagesBackground)
        .filter(ImagesBackground.image_id == target_image_id)
        .first()
    )

    if ImagesBG_entry.hue :
        FULL_ANALYSIS=False
        if VERBOSE:print("color data already present, will only add selfie bbox")
    else:
        FULL_ANALYSIS=True
        if VERBOSE:print("doing full analysis")


    file=get_filename(target_image_id)
    #filename=get_filename(imagename)
    if os.path.exists(file):
        img = cv2.imread(file)    
    else:
        print(f"image not found {target_image_id} {file}")
        return
    bbox=None
    face_landmarks=None
    ########This specific case is for image with apostrophe in their name like "hand's"#############
    ########It messes with reading/writing somehow, os.exists says it exists
    ########cv.imread reads it and produces None, because it reads "hands" not "hand's"
    if img is None:return
    #####################
    # hue,sat,val,lum, lum_torso=get_bg_hue_lum(img,bbox,facelandmark)
    segmentation_mask=get_segmentation_mask(img,bbox,face_landmarks)
    if FULL_ANALYSIS:
        hue,sat,val,lum, lum_torso=sort.get_bg_hue_lum(img,segmentation_mask,bbox)  
        if USE_BBOX:
            #will do a second round for bbox with same cv2 image
            bbox,face_landmarks=get_bbox(target_image_id)
            hue_bb,sat_bb, val_bb, lum_bb, lum_torso_bb =sort.get_bg_hue_lum(img,segmentation_mask,bbox)  
            print("sat values before insert", hue_bb,sat_bb, val_bb, lum_bb, lum_torso_bb)
            # hue_bb,sat_bb, val_bb, lum_bb, lum_torso_bb =get_bg_hue_lum(img,bbox,facelandmark)

    selfie_bbox=get_selfie_bbox(segmentation_mask)
    
    # Update the BG entry with the corresponding image_id


    if ImagesBG_entry:
        if FULL_ANALYSIS:
            if USE_BBOX:
                ImagesBG_entry.hue_bb = hue_bb
                ImagesBG_entry.lum_bb = lum_bb
                ImagesBG_entry.sat_bb = sat_bb
                ImagesBG_entry.val_bb = val_bb
                ImagesBG_entry.lum_torso_bb = lum_torso_bb

            ImagesBG_entry.hue = hue
            ImagesBG_entry.lum = lum
            ImagesBG_entry.sat = sat
            ImagesBG_entry.val = val
            ImagesBG_entry.lum_torso = lum_torso   

        ImagesBG_entry.selfie_bbox=selfie_bbox

        if VERBOSE:
            print("image_id:", ImagesBG_entry.image_id)
            print("hue_bb:", ImagesBG_entry.hue_bb)
            print("lum_bb:", ImagesBG_entry.lum_bb)
            print("sat_bb:", ImagesBG_entry.sat_bb)
            print("val_bb:", ImagesBG_entry.val_bb)
            print("lum_torso_bb:", ImagesBG_entry.lum_torso_bb)
            print("hue:", ImagesBG_entry.hue)
            print("lum:", ImagesBG_entry.lum)
            print("sat:", ImagesBG_entry.sat)
            print("val:", ImagesBG_entry.val)
            print("lum_torso:", ImagesBG_entry.lum_torso)
            print("selfie bbox",selfie_bbox)

        #session.commit()
        print(f"BG stat for image_id {target_image_id} updated successfully.")
    else:
        print(f"BG stat entry for image_id {target_image_id} not found.")
    
    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()
    if counter % 100 == 0:
        print(f"Updated Images_BG number: {counter}")
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
    # select_query = select(Images.image_id,Images.imagename,Images.site_name_id).\
    #     select_from(Images).outerjoin(ImagesBackground,Images.image_id == ImagesBackground.image_id).filter(ImagesBackground.image_id == None).limit(LIMIT)
    
    # pulling directly frmo segment, to filter on face_x etc
    select_query = select(SegmentTable.image_id,SegmentTable.imagename,SegmentTable.site_name_id).\
        select_from(SegmentTable).outerjoin(ImagesBackground,SegmentTable.image_id == ImagesBackground.image_id).filter(ImagesBackground.image_id == None).limit(LIMIT)
    
    # pulling from segment with a join to the helper table
    # select_query = select(
    #     SegmentTable.image_id,
    #     SegmentTable.imagename,
    #     SegmentTable.site_name_id
    # ).\
    # select_from(SegmentTable).\
    # outerjoin(ImagesBackground, SegmentTable.image_id == ImagesBackground.image_id).\
    # outerjoin(HelperTable, SegmentTable.image_id == HelperTable.image_id).\
    # filter(ImagesBackground.image_id == None).\
    # filter(HelperTable.image_id != None).\
    # limit(LIMIT)
    ######################
    # select_query = select(
    #     SegmentTable.image_id,
    #     SegmentTable.imagename,
    #     SegmentTable.site_name_id
    # ).\
    # select_from(SegmentTable).\
    # outerjoin(ImagesBackground, SegmentTable.image_id == ImagesBackground.image_id).\
    # filter(ImagesBackground.image_id == None).\
    # filter(not_(SegmentTable.age_id.in_([1, 2, 3]))).\
    # limit(LIMIT)
    #####################
    ####################
    select_query = select(
        SegmentTable.image_id,
        SegmentTable.imagename,
        SegmentTable.site_name_id
    ).\
    select_from(SegmentTable).\
    outerjoin(ImagesBackground, SegmentTable.image_id == ImagesBackground.image_id).\
    filter(ImagesBackground.image_id == None).\
    limit(LIMIT)    
    ####################
    #####################
    #for some reason ''' select ([xyx])''' produces error
    #but ''' select(xyz)''' doesn't, atleast on windows
    ############################
    
    # select_query = select([SegmentTable.image_id, SegmentTable.imagename, SegmentTable.site_name_id]). \
    # select_from(SegmentTable). \
    # outerjoin(ImagesBackground, SegmentTable.image_id == ImagesBackground.image_id). \
    # filter(ImagesBackground.image_id == None). \
    # filter(and_(
        # SegmentTable.face_x >= -33,
        # SegmentTable.face_x <= -26,
        # SegmentTable.face_y >= -2,
        # SegmentTable.face_y <= 2,
        # SegmentTable.face_z >= -2,
        # SegmentTable.face_z <= 2
    # )). \
    # limit(LIMIT)


    result = session.execute(select_query).fetchall()
    # print the length of the result
    print(len(result), "rows")
    for row in result:
        work_queue.put(row)
        
elif index == 1:
    function=fetch_BG_stat
    #################FETCHING BG stat####################################
    # # for reprocessing bad bboxes with sm portrait, joined to helper table (note the offset)
    # if USE_BBOX:distinct_image_ids_query = select(ImagesBackground.image_id.distinct()).\
    #     outerjoin(HelperTable, ImagesBackground.image_id == HelperTable.image_id).\
    #     filter(HelperTable.image_id != None).\
    #     filter(ImagesBackground.hue_bb == -1).limit(LIMIT).offset(3000)

    # for reprocessing torso+row only for subsegment through join to helper table
    # if USE_BBOX:distinct_image_ids_query = select(ImagesBackground.image_id.distinct()).\
    #     outerjoin(HelperTable, ImagesBackground.image_id == HelperTable.image_id).\
    #     filter(HelperTable.image_id != None).\
    #     filter(ImagesBackground.lum_torso == None).limit(LIMIT)

    # # for helpertable
    # if USE_BBOX:distinct_image_ids_query = select(HelperTable.image_id.distinct()).\
    #     join(ImagesBackground, ImagesBackground.image_id == HelperTable.image_id).\
    #     filter(ImagesBackground.lum_torso == None).limit(LIMIT)

    # if USE_BBOX:distinct_image_ids_query = select(ImagesBackground.image_id.distinct()).\
    #     filter(ImagesBackground.lum_torso == None).limit(LIMIT)
    # FOR SELFIE BBOX
    if USE_BBOX:distinct_image_ids_query = select(ImagesBackground.image_id.distinct()).\
        filter(ImagesBackground.selfie_bbox == None).limit(LIMIT)

    # if USE_BBOX:distinct_image_ids_query = select(ImagesBackground.image_id.distinct()).filter(ImagesBackground.hue_bb == None).limit(LIMIT)
    else:distinct_image_ids_query = select(ImagesBackground.image_id.distinct()).filter(ImagesBackground.hue == None).limit(LIMIT)
    distinct_image_ids = [row[0] for row in session.execute(distinct_image_ids_query).fetchall()]
    for counter,target_image_id in enumerate(distinct_image_ids):
        if counter%10000==0:print("###########"+str(counter)+"images processed ##########")
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
