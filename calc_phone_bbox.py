from ultralytics import YOLO
import os
import cv2
import numpy as np

from sqlalchemy import create_engine, select, delete, and_
from sqlalchemy.orm import sessionmaker,scoped_session, declarative_base
from sqlalchemy.pool import NullPool
# from my_declarative_base import Images,ImagesBackground, SegmentTable, Site 
from mp_db_io import DataIO
from pick import pick
import threading
import queue
import json
from my_declarative_base import Base, ImagesTopics,PhoneBbox, SegmentTable


title = 'Please choose your operation: '
options = ['Create table', 'Object detection']
option, index = pick(options, title)

Base = declarative_base()
# MM controlling which folder to use
IS_SSD = True
VERBOSE = True
io = DataIO(IS_SSD)
db = io.db
# io.db["name"] = "stock"
io.db["name"] = "ministock"
# Create a database engine
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

# Create a session
session = scoped_session(sessionmaker(bind=engine))



LIMIT= 100
# Initialize the counter
counter = 0
OBJ_CLS_ID=67 ## 67 for "cell phone"
OBJ_CLS_LIST=[67,63,26,27,32] ## 
OBJ_CLS_NAME={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat'\
   , 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat'\
    , 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe'\
    , 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard'\
    , 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard'\
    , 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl'\
    , 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza'\
    , 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet'\
    , 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster'\
    , 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier'\
    , 79: 'toothbrush'}
TOPIC_ID=15 ## '0.129*"phone" + 0.091*"mobil" + 0.043*"stop" + 0.043*"talk" + 0.038*"text" + 0.038*"messag" + 0.035*"smart" + 0.033*"communic" + 0.032*"technolog" + 0.029*"telephon"'
#SJ topic 26 is "phone" and MM is 15
# Number of threads
#num_threads = io.NUMBER_OF_PROCESSES
num_threads = 1

MODEL_SIZE="MEDIUM"
# DETAILS : https://b2633864.smushcdn.com/2633864/wp-content/uploads/2023/05/yolov8-model-comparison-1024x360.png?lossy=2&strip=1&webp=1

def create_table_one(image_id, lock, session):
    # Create a BagOfKeywords object
    PhoneBbox_entry = PhoneBbox(
        image_id=image_id,
        bbox=None,  # Set this to None or your desired value
        conf=None,  # Set this to None or your desired value
    )
    
    # Add the BagOfKeywords object to the session
    session.add(PhoneBbox_entry)

    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()

    # Print a message to confirm the update
    # print(f"BG list list for image_id {image_id} updated successfully.")
    if counter % 100 == 0:
        print(f"Created Phone_bbox number: {counter}")

    return

def create_table(image_id, lock, session):
    # Create a BagOfKeywords object
    PhoneBbox_entry = PhoneBbox(
        image_id=image_id,
        bbox_67=None,  # Set this to None or your desired value
        conf_67=None,  # Set this to None or your desired value
        bbox_63=None,  # Set this to None or your desired value
        conf_63=None,  # Set this to None or your desired value
        bbox_26=None,  # Set this to None or your desired value
        conf_26=None,  # Set this to None or your desired value
        bbox_27=None,  # Set this to None or your desired value
        conf_27=None,  # Set this to None or your desired value
        bbox_32=None,  # Set this to None or your desired value
        conf_32=None,  # Set this to None or your desired value
    )
    
    # Add the BagOfKeywords object to the session
    session.add(PhoneBbox_entry)

    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()

    # Print a message to confirm the update
    # print(f"BG list list for image_id {image_id} updated successfully.")
    if counter % 100 == 0:
        print(f"Created Phone_bbox number: {counter}")

    return

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
    if return_endfile: return file,end_file
    return file

def return_bbox_one(image):
    result = model(image,classes=[OBJ_CLS_ID])[0]
    bbox,conf=None,-1
    if len(result.boxes)==1:
        for box in result.boxes:
            bbox = box.xyxy[0].tolist()    #the coordinates of the box as an array [x1,y1,x2,y2]
            bbox = {"left":round(bbox[0]),"top":round(bbox[1]),"right":round(bbox[2]),"bottom":round(bbox[3])}
            conf = round(box.conf[0].item(), 2)
    elif len(result.boxes)>1:
        print("multiple phones detected so ignoring")
    else:
        print("no phones detected")
    
    return bbox,conf

def return_bbox(image):
    result = model(image,classes=[OBJ_CLS_LIST])[0]
    bbox_dict={}
    bbox_count=np.zeros(len(OBJ_CLS_LIST))
    for i,OBJ_CLS_ID in enumerate(OBJ_CLS_LIST):
        for box in result.boxes:
            if int(box.cls[0].item())==OBJ_CLS_ID:
                bbox = box.xyxy[0].tolist()    #the coordinates of the box as an array [x1,y1,x2,y2]
                bbox = {"left":round(bbox[0]),"top":round(bbox[1]),"right":round(bbox[2]),"bottom":round(bbox[3])}
                bbox=json.dumps(bbox)
                # bbox=json.dumps(bbox, indent = 4) 
                conf = round(box.conf[0].item(), 2)                
                bbox_count[i]+=1 
                bbox_dict[OBJ_CLS_ID]={"bbox": bbox, "conf": conf}
                if VERBOSE:print("object IS detected",result.names[box.cls[0].item()])

    for i,OBJ_CLS_ID in enumerate(OBJ_CLS_LIST):
        if bbox_count[i]>1: # checking to see it there are more than one objects of a class and removing 
            bbox_dict.pop(OBJ_CLS_ID)
            bbox_dict[OBJ_CLS_ID]={"bbox": None, "conf": -1} ##setting to default
            if VERBOSE:print("popping because too many",OBJ_CLS_NAME[OBJ_CLS_ID])
        if bbox_count[i]==0:
            bbox_dict[OBJ_CLS_ID]={"bbox": None, "conf": -1} ##setting to default
            if VERBOSE:print("object NOT detected",OBJ_CLS_NAME[OBJ_CLS_ID])

    return bbox_dict

def write_bbox_one(target_image_id, lock, session):
    file=get_filename(target_image_id)
    if os.path.exists(file):
        img = cv2.imread(file)    
    else:
        print(f"image not found {file}")
        return
    
    ########This specific case is for image with apostrophe in their name like "hand's"#############
    ########It messes with reading/writing somehow, os.exists says it exists
    ########cv.imread reads it and produces None, because it reads "hands" not "hand's"
    if img is None:return
    #####################
    bbox,conf=return_bbox(img)
    if VERBOSE:
        if conf==-1:
            folder = "no_phone"
        else:
            folder = "phone"
        cv2.imwrite(os.path.join(io.ROOT_PROD, folder, str(target_image_id)+".jpg"), img)

    # print(bbox,conf)
    PhoneBbox_entry = (
        session.query(PhoneBbox)
        .filter(PhoneBbox.image_id == target_image_id)
        .first()
    )

    if PhoneBbox_entry:
        PhoneBbox_entry.bbox = bbox
        PhoneBbox_entry.conf = conf
        if VERBOSE:
            print("image_id:", PhoneBbox_entry.image_id)
            print("bbox:", PhoneBbox_entry.bbox)
            print("conf:", PhoneBbox_entry.conf)

        #session.commit()
        print(f"Bbox for image_id {target_image_id} updated successfully.")
    else:
        print(f"Bbox for image_id {target_image_id} not found.")
    
    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()
    if counter%100==0:print("###########"+str(counter)+"images processed ##########")

    return

def write_bbox(target_image_id, lock, session):
    file=get_filename(target_image_id)
    if os.path.exists(file):
        img = cv2.imread(file)    
    else:
        print(f"image not found {file}")
        return
    
    ########This specific case is for image with apostrophe in their name like "hand's"#############
    ########It messes with reading/writing somehow, os.exists says it exists
    ########cv.imread reads it and produces None, because it reads "hands" not "hand's"
    if img is None:return
    #####################
    bbox_dict=return_bbox(img)
    for OBJ_CLS_ID in OBJ_CLS_LIST:
        if VERBOSE:
            if bbox_dict[OBJ_CLS_ID]["conf"]==-1:
                folder = "no_phone"
            else:
                folder = "phone"
            # cv2.imwrite(os.path.join(io.ROOT_PROD, folder, str(target_image_id)+".jpg"), img)

    # print(bbox,conf)
        PhoneBbox_entry = (
            session.query(PhoneBbox)
            .filter(PhoneBbox.image_id == target_image_id)
            .first()
        )

        if PhoneBbox_entry:
            setattr(PhoneBbox_entry, "bbox_{0}".format(OBJ_CLS_ID), bbox_dict[OBJ_CLS_ID]["bbox"])
            setattr(PhoneBbox_entry, "conf_{0}".format(OBJ_CLS_ID), bbox_dict[OBJ_CLS_ID]["conf"])
            if VERBOSE:
                print("image_id:", PhoneBbox_entry.image_id)
                print("bbox:", OBJ_CLS_NAME[OBJ_CLS_ID],getattr(PhoneBbox_entry, "bbox_{0}".format(OBJ_CLS_ID)))
                print("conf:", OBJ_CLS_NAME[OBJ_CLS_ID],getattr(PhoneBbox_entry, "conf_{0}".format(OBJ_CLS_ID)))
            #session.commit()
            print(f"Bbox for image_id {target_image_id} updated successfully.")
        else:
            print(f"Bbox for image_id {target_image_id} not found.")
    
    with lock:
        # Increment the counter using the lock to ensure thread safety
        global counter
        counter += 1
        session.commit()
    if counter%100==0:print("###########"+str(counter)+"images processed ##########")

    return


#######MULTI THREADING##################
# Create a lock for thread synchronization
lock = threading.Lock()
threads_completed = threading.Event()

# Create a queue for distributing work among threads
work_queue = queue.Queue()

if index == 0:
    function=create_table
    # Query to retrieve entries where topic_id is equal to 26 and image_id is not present in PhoneBbox table
    select_query = select(ImagesTopics.image_id).select_from(ImagesTopics).filter(ImagesTopics.topic_id == TOPIC_ID).\
        outerjoin(PhoneBbox, ImagesTopics.image_id == PhoneBbox.image_id).filter(PhoneBbox.image_id == None).\
        outerjoin(SegmentTable, ImagesTopics.image_id == SegmentTable.image_id).filter(SegmentTable.site_name_id != 1).limit(LIMIT)
    
    result = session.execute(select_query).fetchall()
    # print the length of the result
    print(len(result), "rows")
    for image_id in result:
        work_queue.put(image_id[0])
        
elif index == 1:
    if MODEL_SIZE=="NANO":
        model = YOLO("yolov8n.pt")   #NANO
    elif MODEL_SIZE=="SMALL":
        model = YOLO("yolov8s.pt")   #SMALL
    elif MODEL_SIZE=="MEDIUM":
        model = YOLO("yolov8m.pt")   #MEDIUM

    function=write_bbox
    if VERBOSE:
        # making folders for testing
        if not os.path.exists(os.path.join(io.ROOT_PROD, "phone")): 
            os.makedirs(os.path.join(io.ROOT_PROD, "phone"))
            os.makedirs(os.path.join(io.ROOT_PROD, "no_phone"))
    distinct_image_ids_query = select(PhoneBbox.image_id.distinct()).select_from(PhoneBbox).filter(PhoneBbox.conf_67 == None).limit(LIMIT)
    distinct_image_ids = [row[0] for row in session.execute(distinct_image_ids_query).fetchall()]
    for counter,target_image_id in enumerate(distinct_image_ids):
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