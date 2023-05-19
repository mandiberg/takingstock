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
import json

import numpy as np
import mediapipe as mp
import pandas as pd

from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from mp_pose_est import SelectPose
from mp_db_io import DataIO

#####new imports #####
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import dlib
import face_recognition_models

# platform specific credentials
io = DataIO()
db = io.db
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES

sortfolder ="getty_test"
http="https://media.gettyimages.com/photos/"
# outputfolder = os.path.join(ROOT,folder+"_output_febmulti")
SAVE_ORIG = False
DRAW_BOX = False
MINSIZE = 700
SLEEP_TIME=0

SELECT = "DISTINCT(i.image_id), i.site_name_id, i.contentUrl, i.imagename, e.encoding_id, i.site_image_id, e.face_landmarks, e.bbox"
# SELECT = "DISTINCT(i.image_id), i.gender_id, author, caption, contentUrl, description, imagename"


# DEBUGGING --> need to change this back to "encodings"
FROM ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings e ON i.image_id = e.image_id INNER JOIN Allmaps am ON i.site_image_id = am.site_image_id"

# WHERE = "e.is_body IS TRUE AND e.bbox IS NULL AND e.face_x IS NOT NULL"
# WHERE = "e.face_encodings IS NULL"
# WHERE = "e.image_id IS NULL AND i.site_name_id = 5 AND k.keyword_text LIKE 'work%'"
# WHERE = "(e.image_id IS NULL AND k.keyword_text LIKE 'smil%')OR (e.image_id IS NULL AND k.keyword_text LIKE 'happ%')OR (e.image_id IS NULL AND k.keyword_text LIKE 'laugh%')"
# WHERE = "e.face_landmarks IS NOT NULL AND e.bbox IS NULL AND i.site_name_id = 1"
WHERE = "i.site_name_id = 1 AND i.site_image_id LIKE '1402424532'"
LIMIT = 100

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
        print("save_image_elsewhere couldn't write")

def save_image_by_path(image, sort, name):
    global sortfolder
    def mkExist(outfolder):
        isExist = os.path.exists(outfolder)
        if not isExist: 
            os.mkdir(outfolder)

    sortfolder_path = os.path.join(ROOT,sortfolder)
    outfolder = os.path.join(sortfolder_path,sort)
    outpath = os.path.join(outfolder, name)
    mkExist(sortfolder)
    mkExist(outfolder)

    try:
        print(outpath)

        cv2.imwrite(outpath, image)

    except:
        print("save_image_by_path couldn't write")



    # outpath = os.path.join(ROOT, sortfolder, folder)
    # try:
    #     print(outpath)
    #     cv2.imwrite(outpath, image)
    #     print("wrote")

    # except:
    #     print("couldn't write")

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
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)};"
    # print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    return(resultsjson)

def get_bbox(faceDet, height, width):
    bbox = {}
    bbox_obj = faceDet.location_data.relative_bounding_box
    xy_min = _normalized_to_pixel_coordinates(bbox_obj.xmin, bbox_obj.ymin, width,height)
    xy_max = _normalized_to_pixel_coordinates(bbox_obj.xmin + bbox_obj.width, bbox_obj.ymin + bbox_obj.height,width,height)
    if xy_min and xy_max:
        # TOP AND BOTTOM WERE FLIPPED 
        # both in xy_min assign, and in face_mesh.process(image[np crop])
        left,top =xy_min
        right,bottom = xy_max
        bbox={"left":left,"right":right,"top":top,"bottom":bottom}
    else:
        print("no results???")
    return(bbox)

def retro_bbox(image):
    height, width, _ = image.shape
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_det: 
        results_det=face_det.process(image)  ## [:,:,::-1] is the shortcut for converting BGR to RGB
        
    # is_face = False
    bbox_json = None
    if results_det.detections:
        faceDet=results_det.detections[0]
        bbox = get_bbox(faceDet, height, width)
        if bbox:
            bbox_json = json.dumps(bbox, indent = 4)
    else:
        print("no results???")
    return bbox_json

def find_face_wholeimage(image, df):
    # window_name = 'image'
    # cv2.imshow(window_name, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    height, width, _ = image.shape
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_det: 
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # results_det=face_det.process(image)  ## [:,:,::-1] is the shortcut for converting BGR to RGB
       
        results_det=face_det.process(image)  ## [:,:,::-1] is the shortcut for converting BGR to RGB
        
    '''
    0 type model: When we will select the 0 type model then our face detection model will be able to detect the 
    faces within the range of 2 meters from the camera.
    1 type model: When we will select the 1 type model then our face detection model will be able to detect the 
    faces within the range of 5 meters. Though the default value is 0.
    '''
    is_face = False

    if results_det.detections:
        faceDet=results_det.detections[0]
        bbox = get_bbox(faceDet, height, width)
        # bbox = faceDet.location_data.relative_bounding_box
        # xy_min = _normalized_to_pixel_coordinates(bbox.xmin, bbox.ymin, width,height)
        # xy_max = _normalized_to_pixel_coordinates(bbox.xmin + bbox.width, bbox.ymin + bbox.height,width,height)
        # if xy_min and xy_max:
        #     # TOP AND BOTTOM WERE FLIPPED 
        #     # both in xy_min assign, and in face_mesh.process(image[np crop])
        #     left,top =xy_min
        #     right,bottom = xy_max
        #     bbox={"left":left,"right":right,"top":top,"bottom":bottom
        #     }
        if bbox:
            with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                             refine_landmarks=False,
                                             max_num_faces=1,
                                             min_detection_confidence=0.5
                                             ) as face_mesh:
            # Convert the BGR image to RGB and cropping it around face boundary and process it with MediaPipe Face Mesh.
                                # crop_img = img[y:y+h, x:x+w]
                # old version, crop to bbox
                # results = face_mesh.process(image[bbox["top"]:bbox["bottom"],bbox["left"]:bbox["right"]])    
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # results = face_mesh.process(image)    
                results = face_mesh.process(image)    

            #read any image containing a face
            if results.multi_face_landmarks:
                print("faceLmsfaceLmsfaceLmsfaceLmsfaceLmsfaceLms")
                print(results.multi_face_landmarks)

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
                faceLms = pose.get_face_landmarks_wholeimage(results, image,bbox)


                #calculate base data from landmarks
                pose.calc_face_data(faceLms)

                # get angles, using r_vec property stored in class
                # angles are meta. there are other meta --- size and resize or something.
                angles = pose.rotationMatrixToEulerAnglesToDegrees()
                mouth_gap = pose.get_mouth_data(faceLms)
                             ##### calculated face detection results
                if is_face:
                    # Calculate Face Encodings if is_face = True
                    # print("in encodings conditional")
                    # turning off to debug
                    encodings = calc_encodings(image, faceLms,bbox) ## changed parameters
                    print(encodings)
                    exit()
                #df.at['1', 'is_face'] = is_face
                #df.at['1', 'is_face_distant'] = is_face_distant
                bbox_json = json.dumps(bbox, indent = 4) 

                df.at['1', 'face_x'] = angles[0]
                df.at['1', 'face_y'] = angles[1]
                df.at['1', 'face_z'] = angles[2]
                df.at['1', 'mouth_gap'] = mouth_gap
                df.at['1', 'face_landmarks'] = pickle.dumps(faceLms)
                df.at['1', 'bbox'] = bbox_json
                df.at['1', 'face_encodings'] = pickle.dumps(encodings)
    df.at['1', 'is_face'] = is_face
    return df
def find_face(image, df):

    height, width, _ = image.shape
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_det: 
        results_det=face_det.process(image)  ## [:,:,::-1] is the shortcut for converting BGR to RGB
        
    '''
    0 type model: When we will select the 0 type model then our face detection model will be able to detect the 
    faces within the range of 2 meters from the camera.
    1 type model: When we will select the 1 type model then our face detection model will be able to detect the 
    faces within the range of 5 meters. Though the default value is 0.
    '''
    is_face = False

    if results_det.detections:
        faceDet=results_det.detections[0]
        bbox = get_bbox(faceDet, height, width)
        # bbox = faceDet.location_data.relative_bounding_box
        # xy_min = _normalized_to_pixel_coordinates(bbox.xmin, bbox.ymin, width,height)
        # xy_max = _normalized_to_pixel_coordinates(bbox.xmin + bbox.width, bbox.ymin + bbox.height,width,height)
        # if xy_min and xy_max:
        #     # TOP AND BOTTOM WERE FLIPPED 
        #     # both in xy_min assign, and in face_mesh.process(image[np crop])
        #     left,top =xy_min
        #     right,bottom = xy_max
        #     bbox={"left":left,"right":right,"top":top,"bottom":bottom
        #     }
        if bbox:
            with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                             refine_landmarks=False,
                                             max_num_faces=1,
                                             min_detection_confidence=0.5
                                             ) as face_mesh:
            # Convert the BGR image to RGB and cropping it around face boundary and process it with MediaPipe Face Mesh.
                                # crop_img = img[y:y+h, x:x+w]
                results = face_mesh.process(image[bbox["top"]:bbox["bottom"],bbox["left"]:bbox["right"]])    
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
                    # print("in encodings conditional")
                    # turning off to debug
                    encodings = calc_encodings(image, faceLms,bbox) ## changed parameters
                    print(encodings)
                    exit()
                #df.at['1', 'is_face'] = is_face
                #df.at['1', 'is_face_distant'] = is_face_distant
                bbox_json = json.dumps(bbox, indent = 4) 

                df.at['1', 'face_x'] = angles[0]
                df.at['1', 'face_y'] = angles[1]
                df.at['1', 'face_z'] = angles[2]
                df.at['1', 'mouth_gap'] = mouth_gap
                df.at['1', 'face_landmarks'] = pickle.dumps(faceLms)
                df.at['1', 'bbox'] = bbox_json
                df.at['1', 'face_encodings'] = pickle.dumps(encodings)
    df.at['1', 'is_face'] = is_face
    return df

def calc_encodings(image, faceLms,bbox):## changed parameters and rebuilt

    # # original, treats uncropped image as .shape
    height, width, _ = image.shape

    # second attempt, tries to project faceLms from bbox origin
    # width = (bbox["right"]-bbox["left"])
    # height = (bbox["bottom"]-bbox["top"])

    # third attempt, crops image to bbox, and keeps faceLms relative to bbox 
    # print("bbox:")
    # print(bbox)
    # print(bbox["top"])
    # bbox = json.loads(bbox)
    # print("bbox")
    # print(bbox)
    # top = int(bbox["top"])
    # bottom = int(bbox["bottom"])
    # left = int(bbox["left"])
    # right = int(bbox["right"])
    # image = image[top:bottom, left:right]

    # # image = image[bbox["top"]:bbox["bottom"],bbox["left"]:bbox["right"]]
    # height, width, _ = image.shape


    landmark_points_5 = [ 263, #left eye away from centre
                       362, #left eye towards centre
                       33,  #right eye away from centre
                       133, #right eye towards centre
                        2 #bottom of nose tip 
                    ]
    raw_landmark_set = []
    for index in landmark_points_5:                       ######### CORRECTION: landmark_points_5_3 is the correct one for sure
        # print(faceLms.landmark[index].x)

        # second attempt, tries to project faceLms from bbox origin
        # x = int(faceLms.landmark[index].x * width + bbox["left"])
        # y = int(faceLms.landmark[index].y * height + bbox["top"])

        x = int(faceLms.landmark[index].x * width)
        y = int(faceLms.landmark[index].y * height)

        # print(x)
        # print(y)
        landmark_point=dlib.point([x,y])
        raw_landmark_set.append(landmark_point)
    all_points=dlib.points(raw_landmark_set)
    print("all_points", all_points)
    print(bbox)
        
    # bbox = faceDet.location_data.relative_bounding_box
    # xy_min = _normalized_to_pixel_coordinates(bbox.xmin, bbox.ymin, height,width)
    # xy_max = _normalized_to_pixel_coordinates(bbox.xmin + bbox.width, bbox.ymin + bbox.height,height,width)
    # if xy_min and xy_max:
    #     xmin,ymin =xy_min
    #     xmax,ymax = xy_max
    #     b_box= dlib.rectangle(left=xmin, top=ymax, right=xmax, bottom=ymin)
    #     #in_bounds=True
    # #else:
    #     #print("face out of frame")
    #     #in_bounds=False

    # I'm unsure if top should be ymax or ymin. 
    # ymin ("top") would be y value for top left point.
    bbox_rect= dlib.rectangle(left=bbox["left"], top=bbox["top"], right=bbox["right"], bottom=bbox["bottom"])

    # # Here is alt_encodings that match SJ's original structure: left=xmin, top=ymax, right=xmax, bottom=ymin
    # # ymax ("bottom") would be y value for top left point.
    # bbox_rect= dlib.rectangle(left=bbox["left"], top=bbox["bottom"], right=bbox["right"], bottom=bbox["top"])

    if (all_points is None) or (bbox is None):return 
    
    raw_landmark_set=dlib.full_object_detection(bbox_rect,all_points)

    # window_name = 'image'
    # cv2.imshow(window_name, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
            # print("bodyLms, ", bodyLms)
            # bodyLms = results.pose_landmarks.landmark
            if not bodyLms.pose_landmarks:
                is_body = False
            else: 
                is_body = True
                df.at['1', 'body_landmarks'] = pickle.dumps(bodyLms.pose_landmarks)

            df.at['1', 'is_body'] = is_body
        except:
            print(f"[find_body]this item failed: {image}")
        return df

def capitalize_directory(path):
    dirname, filename = os.path.split(path)
    parts = dirname.split('/')
    capitalized_parts = [part if i < len(parts) - 2 else part.upper() for i, part in enumerate(parts)]
    capitalized_dirname = '/'.join(capitalized_parts)
    return os.path.join(capitalized_dirname, filename)

# this was for reprocessing the missing bbox
def process_image_bbox(task):
    # df = pd.DataFrame(columns=['image_id','bbox'])
    print("task is: ",task)
    encoding_id = task[0]
    cap_path = capitalize_directory(task[1])
    try:
        image = cv2.imread(cap_path)        
        # this is for when you need to move images into a testing folder structure
        # save_image_elsewhere(image, task)
    except:
        print(f"[process_image]this item failed, even after uppercasing: {task}")
    print("processing: ")
    print(encoding_id)
    if image is not None and image.shape[0]>MINSIZE and image.shape[1]>MINSIZE:
        # Do FaceMesh
        bbox_json = retro_bbox(image)
        print(bbox_json)
        if bbox_json: 
            for _ in range(io.max_retries):
                try:
                    update_sql = f"UPDATE Encodings SET bbox = '{bbox_json}' WHERE encoding_id = {encoding_id};"
                    engine.connect().execute(text(update_sql))
                    print("bboxxin:")
                    print(encoding_id)
                    break  # Transaction succeeded, exit the loop
                except OperationalError as e:
                    print(e)
                    time.sleep(io.retry_delay)
        else:
            print("no bbox")

    else:
        print('toooooo smallllll')
        # I should probably assign no_good here...?

    # store data


def process_image_enc_only(task):
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

    df = pd.DataFrame(columns=['image_id','is_face','is_body','is_face_distant','face_x','face_y','face_z','mouth_gap','face_landmarks','bbox','face_encodings','body_landmarks'])
    # print(task)
    # df.at['1', 'image_id'] = task[0]
    encoding_id = task[0]
    faceLms = task[2]
    bbox = task[3]
    cap_path = capitalize_directory(task[1])
    try:
        image = cv2.imread(cap_path)        
        # this is for when you need to move images into a testing folder structure
        # save_image_elsewhere(image, task)
    except:
        print(f"[process_image]this item failed: {task}")

    if image is not None and image.shape[0]>MINSIZE and image.shape[1]>MINSIZE:
        face_enc = calc_encodings(image, faceLms,bbox)
        # Do FaceMesh
        df = find_face(image, df)
        # Do Body Pose
        df = find_body(image, df)
        # for testing: this will save images into folders for is_face, is_body, and none. 
        # only save images that aren't too smallllll
        # save_image_triage(image,df)
    else:
        print('toooooo smallllll')
        # I should probably assign no_good here...?

    # store data
    # print(df)
    try:
        update_sql = f"UPDATE Encodings4 SET bbox = '{face_encodings}' WHERE encoding_id = {encoding_id};"
        # print(update_sql)
        # quit()
        engine.connect().execute(text(update_sql))
        print("face_encodingsssssssssssss:")
        print(encoding_id)

        # # DEBUGGING --> need to change this back to "encodings"
        # insertignore_df(df,"encodings4", engine)  ### made it all lower case to avoid discrepancy
    except OperationalError as e:
        print(e)

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

    df = pd.DataFrame(columns=['image_id','is_face','is_body','is_face_distant','face_x','face_y','face_z','mouth_gap','face_landmarks','bbox','face_encodings','body_landmarks'])
    print(task)
    df.at['1', 'image_id'] = task[0]
    cap_path = capitalize_directory(task[1])
    try:
        image = cv2.imread(cap_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        # this is for when you need to move images into a testing folder structure
        # save_image_elsewhere(image, task)
    except:
        print(f"[process_image]this item failed: {task}")

    if image is not None and image.shape[0]>MINSIZE and image.shape[1]>MINSIZE:
        # Do FaceMesh
        df = find_face_wholeimage(image, df)
        # Do Body Pose
        df = find_body(image, df)
        # for testing: this will save images into folders for is_face, is_body, and none. 
        # only save images that aren't too smallllll
        # save_image_triage(image,df)
    else:
        print('toooooo smallllll')
        # I should probably assign no_good here...?

    # store data
    # print(df)
    try:
        # DEBUGGING --> need to change this back to "encodings"
        insertignore_df(df,"encodings3", engine)  ### made it all lower case to avoid discrepancy
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
    print("main")
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []
    count = 0
    last_round = False

    while True:
        # print("about to SQL: ",SELECT,FROM,WHERE,LIMIT)
        resultsjson = selectSQL()    
        print("got results, count is: ",len(resultsjson))
        # print(resultsjson)
        #catches the last round, where it returns less than full results
        if last_round == True:
            print("last_round caught, should break")
            break
        elif len(resultsjson) != LIMIT:
            last_round = True
            print("last_round just assigned")

        # print(last_round)
        for row in resultsjson:
            # print(row)
            # face_landmarks, e2.bbox
            encoding_id = row["encoding_id"]
            image_id = row["image_id"]
            item = row["contentUrl"]
            hashed_path = row["imagename"]
            site_id = row["site_name_id"]
            # print(hashed_path)
            if site_id == 1:
                # print("fixing gettyimages hash")
                orig_filename = item.replace(http, "")+".jpg"
                d0, d02 = get_hash_folders(orig_filename)
                hashed_path = os.path.join(d0, d02, orig_filename)
            
            # gets folder via the folder list, keyed with site_id integer
            imagepath=os.path.join(io.folder_list[site_id], hashed_path)
            isExist = os.path.exists(imagepath)
            if isExist: 
                if row["face_landmarks"] is not None:
                    task = (encoding_id,imagepath,pickle.loads(row["face_landmarks"]),row["bbox"])
                else:
                    task = (image_id,imagepath)
                tasks_to_accomplish.put(task)
                # print("tasks_to_accomplish.put(task) ",imagepath)
            else:
                print("this file is missssssssssing --------> ",imagepath)
        # print("tasks_to_accomplish.qsize()", str(tasks_to_accomplish.qsize()))
        # print(tasks_to_accomplish.qsize())

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
        count += len(resultsjson)
        print("completed round, total results processed is: ",count)


    end = time.time()
    print (end - start)
    print ("total processed ",count)
    return True

if __name__ == '__main__':
    main()



