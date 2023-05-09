from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import csv
import os
import hashlib
import cv2
import math
import pickle
import sys # can delete for production
from sys import platform

import numpy as np
import mediapipe as mp
import pandas as pd

from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from mp_pose_est import SelectPose

#####new imports #####
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import dlib
import face_recognition_models

'''

select 1k images from database with no is_face variable

SELECT column_names
FROM table_name
WHERE column_name IS NULL; 

send each row into the 
        tasks_to_accomplish.put("Task no " + str(i))

i think i want the whole row:
 i need the UID to add the data back, and the filepath to read it
 so easiest just to send the whole thing in?

'''


# platform specific credentials
if platform == "darwin":
    # OS X

    ####### Michael's Credentials ########
    db = {
        "host":"localhost",
        "name":"gettytest3",            
        "user":"root",
        "pass":"Fg!27Ejc!Mvr!GT"
    }

    ROOT= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production") ## only on Mac
    NUMBER_OF_PROCESSES = 8
    ######################################

elif platform == "win32":
    # Windows...

    ######## Satyam's Credentials #########
    db = {
        "host":"localhost",
        "name":"gettytest3",                 
        "user":"root",
        "pass":"SSJ2_mysql"
    }
    #ROOT= os.path.join(os.environ['HOMEDRIVE'],os.environ['HOMEPATH'], "Documents/projects-active/facemap_production") ## local WIN
    ROOT= os.path.join("D:/"+"Documents/projects-active/facemap_production") ## SD CARD
    NUMBER_OF_PROCESSES = 4
    #######################################



folder ="gettyimages"
sortfolder ="getty_test"
http="https://media.gettyimages.com/photos/"
# folder ="files_for_testing"
outputfolder = os.path.join(ROOT,folder+"_output_febmulti")
SAVE_ORIG = False
DRAW_BOX = False
MINSIZE = 700
# number_of_task = 10
SLEEP_TIME=0

# table_search ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id"
SELECT = "DISTINCT(i.image_id), i.gender_id, author, caption, contentUrl, description, imagename"
FROM ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings e ON i.image_id = e.image_id "
WHERE = "e.image_id IS NULL"
LIMIT = 10


#creating my objects
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=1, static_image_mode=True)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

####### new imports and models ########
mp_face_detection = mp.solutions.face_detection #### added face detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
###############

# testing to see if it works better to collect all rows, and then insert
df_all = pd.DataFrame(columns=['image_id','is_face','is_body','is_face_distant','face_x','face_y','face_z','mouth_gap','face_landmarks','face_landmarks_pickle','face_encodings','body_landmarks'])

engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
metadata = MetaData(engine)

start = time.time()

# not sure if I'm using this
class Object:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

def get_hash_folders(filename):
    m = hashlib.md5()
    m.update(filename.encode('utf-8'))
    d = m.hexdigest()
    # csvWriter1.writerow(["https://upload.wikimedia.org/wikipedia/commons/"+d[0]+'/'+d[0:2]+'/'+filename])
    return d[0], d[0:2]

def read_csv(csv_file):
    with open(csv_file, encoding="utf-8", newline="") as in_file:
        reader = csv.reader(in_file, delimiter=",")
        next(reader)  # Header row

        for row in reader:
            yield row



def save_image_elsewhere(image, path):
    #saves a CV2 image elsewhere -- used in setting up test segment of images
    oldfolder = "newimages"
    newfolder = "testimages"
    outpath = path.replace(oldfolder, newfolder)
    try:
        print(outpath)
        cv2.imwrite(outpath, image)
        print("wrote")

    except:
        print("couldn't write")

def save_image_by_path(image, sort, name):
    global sortfolder
    def do_isExist(outfolder):
        isExist = os.path.exists(outfolder)
        if not isExist: 
            os.mkdir(outfolder)

    sortfolder_path = os.path.join(ROOT,sortfolder)
    outfolder = os.path.join(sortfolder_path,sort)
    outpath = os.path.join(outfolder, name)
    do_isExist(sortfolder)
    do_isExist(outfolder)

    try:
        cv2.imwrite(outpath, image)

    except:
        print("couldn't write")



    outpath = os.path.join(ROOT, sortfolder, folder)
    try:
        print(outpath)
        cv2.imwrite(outpath, image)
        print("wrote")

    except:
        print("couldn't write")

# def create_my_engine(db):

#     # Create SQLAlchemy engine to connect to MySQL Database
#     engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
#                                     .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']))

def insertignore(dataframe,table):

     # creating column list for insertion
     cols = "`,`".join([str(i) for i in dataframe.columns.tolist()])

     # Insert DataFrame recrds one by one.
     for i,row in dataframe.iterrows():
         sql = "INSERT IGNORE INTO `"+table+"` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
         engine.connect().execute(sql, tuple(row))


def insertignore_df(dataframe,table_name, engine):

     # Convert the DataFrame to a SQL table using pandas' to_sql method
     with engine.connect() as connection:
         dataframe.to_sql(name=table_name, con=connection, if_exists='append', index=False)


def insertignore_dict(dict_data,table_name):

     # # creating column list for insertion
     # # cols = "`,`".join([str(i) for i in dataframe.columns.tolist()])
     # cols = "`,`".join([str(i) for i in list(dict.keys())])
     # tup = tuple(list(dict.values()))

     # sql = "INSERT IGNORE INTO `"+table+"` (`" +cols + "`) VALUES (" + "%s,"*(len(tup)-1) + "%s)"
     # engine.connect().execute(sql, tup)

     # Create a SQLAlchemy Table object representing the target table
     target_table = Table(table_name, metadata, extend_existing=True, autoload_with=engine)

     # Insert the dictionary data into the table using SQLAlchemy's insert method
     with engine.connect() as connection:
         connection.execute(target_table.insert(), dict_data)


def selectSQL():

    # selectsql = "SELECT UID from Images Where UID = '"+str(1351300526)+"';"
    # selectsql = "SELECT "+ get +" from "+ table +" Where "+ column +" "+ value +" LIMIT "+ str(limit) +";"
    # selectsql = f"SELECT {get} FROM {table} WHERE {column} {value} LIMIT {str(limit)};"
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)};"
    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))

    resultsjson = ([dict(row) for row in result.mappings()])

    return(resultsjson)
    # try:
    #     myUID = resultsjson[0]['UID']
    #     alreadyDL = True
    # except:
    #     alreadyDL = False

    # return alreadyDL



def find_face(image, df):
    height, width, _ = image.shape
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_det: 
        results_det=face_det.process(image[:,:,::-1])  ## [:,:,::-1] is the shortcut for converting BGR to RGB
        
    '''
    0 type model: When we will select the 0 type model then our face detection model will be able to detect the 
    faces within the range of 2 meters from the camera.
    1 type model: When we will select the 1 type model then our face detection model will be able to detect the 
    faces within the range of 5 meters. Though the default value is 0.
    '''
    is_face = False

    if results_det.detections:
        faceDet=results_det.detections[0]
        bbox = faceDet.location_data.relative_bounding_box
        print("bbox ", bbox)
        xy_min = _normalized_to_pixel_coordinates(bbox.xmin, bbox.ymin, width,height)
        xy_max = _normalized_to_pixel_coordinates(bbox.xmin + bbox.width, bbox.ymin + bbox.height,width,height)
        if xy_min and xy_max:
            # TOP AND BOTTOM WERE FLIPPED 
            # both in xy_min assign, and in face_mesh.process(image[np crop])
            left,top =xy_min
            right,bottom = xy_max
            bbox={"left":left,"right":right,"top":top,"bottom":bottom
            }
            print("bbox ", bbox)
            with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                             refine_landmarks=False,
                                             max_num_faces=1,
                                             min_detection_confidence=0.5
                                             ) as face_mesh:
            # Convert the BGR image to RGB and cropping it around face boundary and process it with MediaPipe Face Mesh.
                                # crop_img = img[y:y+h, x:x+w]
                results = face_mesh.process(image[top:bottom,left:right,::-1])    
            #read any image containing a face
            if results.multi_face_landmarks:
                
                # # debugging the flipped landmarks
                # start_point = ((left, top))
                # end_point = ((right, bottom))
                # color = (255, 0, 0)
                # thickness = 1
                # image = cv2.rectangle(image, start_point, end_point, color, thickness)
                # image = cv2.line(image, start_point, end_point, color, thickness)
                # image = cv2.circle(image, start_point, 20, color, 1)

                #construct pose object to solve pose
                is_face = True
                pose = SelectPose(image)

                #get landmarks
                faceLms = pose.get_face_landmarks(results, image,bbox)


                #calculate base data from landmarks
                pose.calc_face_data(faceLms)

                # get angles, using r_vec property stored in class
                # angles are meta. there are other meta --- size and resize or something.
                angles = pose.rotationMatrixToEulerAnglesToDegrees()
                mouth_gap = pose.get_mouth_data(faceLms)
                             ##### calculated face detection results
                if is_face:
                    # Calculate Face Encodings if is_face = True
                    print("in encodings conditional")
                    # turning off to debug
                    encodings = calc_encodings(image, faceLms,faceDet) ## changed parameters
                    print(encodings)
                #df.at['1', 'is_face'] = is_face
                #df.at['1', 'is_face_distant'] = is_face_distant
                df.at['1', 'face_x'] = angles[0]
                df.at['1', 'face_y'] = angles[1]
                df.at['1', 'face_z'] = angles[2]
                df.at['1', 'mouth_gap'] = mouth_gap
                df.at['1', 'face_landmarks'] = pickle.dumps(faceLms)
                df.at['1', 'face_encodings'] = pickle.dumps(encodings)
    
    df.at['1', 'is_face'] = is_face


    return df

def calc_encodings(image, faceLms,faceDet):## changed parameters and rebuilt
    #print("calc_encodings")
    height, width, _ = image.shape
    landmark_points_5 = [ 263, #left eye away from centre
                       362, #left eye towards centre
                       33,  #right eye away from centre
                       133, #right eye towards centre
                        2 #bottom of nose tip 
                    ]
    raw_landmark_set = []
    for index in landmark_points_5:                       ######### CORRECTION: landmark_points_5_3 is the correct one for sure
        x = int(faceLms.landmark[index].x * width)
        y = int(faceLms.landmark[index].y * height)
        landmark_point=dlib.point([x,y])
        raw_landmark_set.append(landmark_point)
    all_points=dlib.points(raw_landmark_set)
    
    bbox = faceDet.location_data.relative_bounding_box
    xy_min = _normalized_to_pixel_coordinates(bbox.xmin, bbox.ymin, height,width)
    xy_max = _normalized_to_pixel_coordinates(bbox.xmin + bbox.width, bbox.ymin + bbox.height,height,width)
    if xy_min is not None and xy_max is not None:
        xmin,ymin =xy_min
        xmax,ymax = xy_max
        b_box= dlib.rectangle(left=xmin, top=ymax, right=xmax, bottom=ymin)
        #in_bounds=True
    #else:
        #print("face out of frame")
        #in_bounds=False

    if (all_points is None) or (b_box is None):return 
    
    raw_landmark_set=dlib.full_object_detection(b_box,all_points)
    encodings=face_encoder.compute_face_descriptor(image, raw_landmark_set, num_jitters=1)


    return np.array(encodings)

def find_body(image,df):
    #print("find_body")
    with mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5) as pose:
      # for idx, file in enumerate(file_list):
        try:
            # image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            bodyLms = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print("bodyLms, ", bodyLms)
            # bodyLms = results.pose_landmarks.landmark
            if not bodyLms.pose_landmarks:
                is_body = False
            else: 
                is_body = True
                # bodyLms.pose_landmarks = is_body

                # it seems like this is borking the function
                # turning this off to debug df
                df.at['1', 'body_landmarks'] = pickle.dumps(bodyLms.pose_landmarks)

            df.at['1', 'is_body'] = is_body

            # return is_body, bodyLms.pose_landmarks.toJSON()
        except:
            print(f"[find_body]this item failed: {image}")

        return df


def process_image(task):
    #print("process_image")
    def save_image_triage(image,df):
        #saves a CV2 image elsewhere -- used in setting up test segment of images
        if df.at['1', 'is_face']:
            sort = "face"
        elif df.at['1', 'is_body']:
            sort = "body"
        else:
            sort = "none"
        name = str(df.at['1', 'image_id'])+".jpg"
        save_image_by_path(image, sort, name)

    df = pd.DataFrame(columns=['image_id','is_face','is_body','is_face_distant','face_x','face_y','face_z','mouth_gap','face_landmarks','face_encodings','body_landmarks'])
    df.at['1', 'image_id'] = task[0]
    try:
        image = cv2.imread(task[1])        
        # this is for when you need to move images into a testing folder structure
        # save_image_elsewhere(image, task)
    except:
        print(f"[process_image]this item failed: {task}")

    if image is not None and image.shape[0]>MINSIZE and image.shape[1]>MINSIZE:
        # Do FaceMesh
        df = find_face(image, df)
        # Do Body Pose
        df = find_body(image, df)
    else:
        print('toooooo smallllll')

    # for testing: this will save images into folders for is_face, is_body, and none. 
    save_image_triage(image,df)

    # store data
    try:
        insertignore_df(df,"encodings", engine)  ### made it all lower case to avoid discrepancy
    except OperationalError as e:
        print(e)


        # save image based on is_face
def do_job(tasks_to_accomplish, tasks_that_are_done):
    #print("do_job")
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:

            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            process_image(task)
            # tasks_that_are_done.put(task + ' is done by ' + current_process().name)
            time.sleep(SLEEP_TIME)
    return True


def main():
    #print("main")
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    print("about to SQL: ",SELECT,FROM,WHERE,LIMIT)
    # create_my_engine(db)
    resultsjson = selectSQL()
    print("got results, count is: ",len(resultsjson))


    for row in resultsjson:
        # gets contentUrl
        print(row)
        image_id = row["image_id"]
        item = row["contentUrl"]
        if folder == "gettyimages":
            orig_filename = item.replace(http, "")+".jpg"
            d0, d02 = get_hash_folders(orig_filename)
            imagepath=os.path.join(ROOT,folder, "newimages",d0, d02, orig_filename)
            isExist = os.path.exists(imagepath)
            if isExist: 
                task = (image_id,imagepath)
                tasks_to_accomplish.put(task)
                # print(imagepath)
                # print(tasks_to_accomplish.qsize())

        else:
            imagepath=os.path.join(ROOT,folder, item)
            orig_filename = item
            print("starting: " +item)


    # for i in range(number_of_task):
    #     tasks_to_accomplish.put("Task no " + str(i))

    # creating processes
    for w in range(NUMBER_OF_PROCESSES):
        p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        # print("completing process")
        p.join()

    # # print the output
    # while not tasks_that_are_done.empty():
    #     print("tasks are done")
    #     print(tasks_that_are_done.get())

    end = time.time()
    print (end - start)

        # need to pull count from tasks_that_are_done
        # imgpermin = tasks_that_are_done.count()/((time.time() - start)/60)
        # hours = (time.time() - start)/3600

        # print("--- %s images per minute ---" % (imgpermin))
        # print("--- %s images per day ---" % (imgpermin*1440))
        # if imgpermin:
        #     print("--- %s days per 1M images ---" % (1000000/(imgpermin*1440)))



    return True


if __name__ == '__main__':
    main()



