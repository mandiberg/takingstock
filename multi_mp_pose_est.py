from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import csv
import os
import hashlib
import cv2
import math

import numpy as np
import mediapipe as mp
import pandas as pd

from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, update

from mp_pose_est import SelectPose

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

ROOT= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production") 
folder ="gettyimages"
http="https://media.gettyimages.com/photos/"
# folder ="files_for_testing"
outputfolder = os.path.join(ROOT,folder+"_output_febmulti")
SAVE_ORIG = False
DRAW_BOX = False
MINSIZE = 700


db = {
    "host":"localhost",
    "name":"gettytest3",
    "user":"root",
    "pass":"Fg!27Ejc!Mvr!GT"
}


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

engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']))

metadata = MetaData(engine)

# # initialize the Metadata Object
# meta = MetaData(bind=engine)
# MetaData.reflect(meta)

start = time.time()

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
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                         refine_landmarks=False,
                                         max_num_faces=1,
                                         min_detection_confidence=0.9
                                         ) as face_mesh:
        # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # print("has results")

    #read any image containing a face
    if results.multi_face_landmarks:
        #construct pose object to solve pose
        is_face = True
        pose = SelectPose(image)

        #get landmarks
        #added returning meshimage (was image)
        faceLms = pose.get_face_landmarks(results, image)

        #calculate base data from landmarks
        pose.calc_face_data(faceLms)

        # get angles, using r_vec property stored in class
        # angles are meta. there are other meta --- size and resize or something.
        angles = pose.rotationMatrixToEulerAnglesToDegrees()
        mouth_gap = pose.get_mouth_data(faceLms)

        df.at['1', 'is_face'] = is_face
        # df.at['1', 'is_face_distant'] = is_face_distant
        df.at['1', 'face_x'] = angles[0]
        df.at['1', 'face_y'] = angles[1]
        df.at['1', 'face_z'] = angles[2]
        df.at['1', 'mouth_gap'] = mouth_gap
        # turning off to debug
        # df.at['1', 'face_landmarks'] = faceLms

        # data_to_store = (angles[0], angles[1], angles[2], mouth_gap)
        # print(data_to_store)

    else: 
        # print(f"no face found")
        is_face = False
        df.at['1', 'is_face'] = is_face
        # data_to_store = (is_face)
        # # print(data_to_store)
        # faceLms= is_face
    # return is_face, data_to_store, faceLms
    print("is_face: ",is_face)
    return df

def calc_encodings(image, df):
    # dlib code will go here
    encodings = ""
    df.at['1', 'face_encodings'] = encodings

    return df

def find_body(image,df):
    with mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5) as pose:
      # for idx, file in enumerate(file_list):
        try:
            # image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            bodyLms = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # bodyLms = results.pose_landmarks.landmark
            if not bodyLms.pose_landmarks:
                is_body = False
            else: 
                is_body = True
                # bodyLms.pose_landmarks = is_body

                # turning this off to debug df
                # df.at['1', 'body_landmarks'] = bodyLms.pose_landmarks.toJSON()

            df.at['1', 'is_body'] = is_body

            # return is_body, bodyLms.pose_landmarks.toJSON()
        except:
            print(f"this item failed: {image}")

        print("is_body: ",is_body)
        return df


def process_image(task):
    df = pd.DataFrame(columns=['image_id','is_face','is_body','is_face_distant','face_x','face_y','face_z','mouth_gap','face_landmarks','face_encodings','body_landmarks'])
    # data = {}
    # # df['image_id'] = task[0]
    # df.at['1', 'image_id'] = task[0]
    # # print(task[0])
    # print("df['image_id'] ", df['image_id'])
    try:
        image = cv2.imread(task[1]) 
        
        # this is for when you need to move images into a testing folder structure
        # save_image_elsewhere(image, task)
    except:
        print(f"this item failed: {task}")

    if image is not None and image.shape[0]>MINSIZE and image.shape[1]>MINSIZE:

        # Do FaceMesh
        # is_face, data_to_store, faceLms = find_face(image)
        df = find_face(image, df)

        #         data_to_store = (angles[0], angles[1], angles[2], mouth_gap)

        # Do Body Pose
        df = find_body(image, df)

        # data['is_face'] = is_face
        # data['is_body'] = is_body
        # 'is_face_distant'] = is_face_distant

        # print(df.at['1', 'is_face'])

        if df.at['1', 'is_face']:
            # Calculate Face Encodings if is_face = True
            print("in encodings conditional")
            # turning off to debug
            # df = calc_encodings(image, df)


            #prepare data package here
            # data['face_x'] = data_to_store[0]
            # data['face_y'] = data_to_store[0]
            # data['face_z'] = data_to_store[0]
            # data['mouth_gap'] = data_to_store[0]
            # data['face_landmarks'] = faceLms
            # data['face_encodings'] = encodings


        # if df.at['1', 'is_body']:
        #     print("processed image")
        #     #prepare data package here
        #     data['body_landmarks'] = bodyLms

        if not df.at['1', 'is_face'] and not df.at['1', 'is_body']:
            print("no face or body found")
            #prepare data package here


    else:
        print('toooooo smallllll')
        # os.remove(item)
        # do I say no face, no body, no double face?
    print("\n\n")

    insertignore_df(df,"Encodings", engine)
    #store data here


def do_job(tasks_to_accomplish, tasks_that_are_done):
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
            # time.sleep(.5)
    return True


def main():
    # number_of_task = 10
    number_of_processes = 8
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
    for w in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        print("completing process")
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



