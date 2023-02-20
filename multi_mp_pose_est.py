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

csv_file = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/test2500.csv"
ROOT= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production") 
folder ="gettyimages"
http="https://media.gettyimages.com/photos/"
# folder ="files_for_testing"
outputfolder = os.path.join(ROOT,folder+"_output_febmulti")
SAVE_ORIG = False
DRAW_BOX = False
MINSIZE = 700


#creating my objects
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=1, static_image_mode=True)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

start = time.time()


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


def find_face(image):
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

        data_to_store = (is_face, angles[0], angles[1], angles[2], mouth_gap, faceLms)
        # print(data_to_store)

    else: 
        # print(f"no face found")
        is_face = False

        data_to_store = (is_face)
        # print(data_to_store)
    return data_to_store

def calc_encodings(image):
    # dlib code will go here
    encodings = ""
    return encodings

def find_body(image):
    with mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5) as pose:
      # for idx, file in enumerate(file_list):
        try:
            # image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                is_body = False
            else: 
                is_body = True
            return is_body, results
        except:
            print(f"this item failed: {file}")


def process_image(task):
    try:
        image = cv2.imread(task) 
    except:
        print(f"this item failed: {task}")

    if image is not None and image.shape[0]>MINSIZE and image.shape[1]>MINSIZE:

        # Do FaceMesh
        data_to_store = find_face(image)

        # Do Body Pose
        is_body, results = find_body(image)

        if not data_to_store:
            # Calculate Face Encodings if is_face = True
            encodings = calc_encodings(image)
        if data_to_store or is_body:
            print("processed image")
        else:
            print("no face or body found")

    else:
        print('toooooo smallllll')
        # os.remove(item)


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
            tasks_that_are_done.put(task + ' is done by ' + current_process().name)
            # time.sleep(.5)
    return True


def main():
    number_of_task = 10
    number_of_processes = 8
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    for row in read_csv(csv_file):
        # gets contentUrl
        item = (row[4])
        if folder == "gettyimages":
            orig_filename = item.replace(http, "")+".jpg"
            d0, d02 = get_hash_folders(orig_filename)
            imagepath=os.path.join(ROOT,folder, "newimages",d0, d02, orig_filename)
            isExist = os.path.exists(imagepath)
            if isExist: 
                tasks_to_accomplish.put(imagepath)
                # print(imagepath)
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
        p.join()

    # print the output
    while not tasks_that_are_done.empty():
        print(tasks_that_are_done.get())

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



