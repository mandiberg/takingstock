from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import csv
import os
import hashlib
import cv2


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

do_job will
mediapipe post est
write is_face and landmarks
write calc data


/Users/michaelmandiberg/Documents/projects-active/facemap_production/gettyimages/newimages/9/94/sporty-girl-picture-id1098415730.jpg
/Users/michaelmandiberg/Documents/projects-active/facemap_production/gettyimages/newimages/9/94/sporty-girl-picture-id1098415730.jpg

'''

csv_file = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/test2500.csv"
ROOT= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production") 
folder ="gettyimages"
http="https://media.gettyimages.com/photos/"
# folder ="files_for_testing"
outputfolder = os.path.join(ROOT,folder+"_output_febbig")
SAVE_ORIG = False
DRAW_BOX = False


#creating my objects
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=1, static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


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

def process_image(image):
    try:
        image = cv2.imread(task) 
    except:
        print(f"this item failed: {task}")

    # if image is not None and image.shape[0]>MINSIZE and image.shape[1]>MINSIZE:
    if image is not None:

        # Initialize FaceMesh
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                             refine_landmarks=False,
                                             max_num_faces=1,
                                             min_detection_confidence=0.9
                                             ) as face_mesh:
            # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print("has results")

        #read any image containing a face
        if results.multi_face_landmarks:
            #construct pose object to solve pose
            pose = SelectPose(image)
            print("has pose")
        



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
            # print(task)
            # print("this is where you do stuff?")
            process_image(image)
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
                print(isExist)
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

    return True


if __name__ == '__main__':
    main()



