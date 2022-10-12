import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import mediapipe as mp
import pandas as pd
from mp_pose_est import SelectPose

import os
import math
import time


#creating my objects
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

start = time.time()

#regular 31.8s
#concurrent 

#declariing path and image before function, but will reassign in the main loop
#location of source and output files outside repo
ROOT= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production") 
# folder ="sourceimages"
folder ="files_for_testing3"
outputfolder = os.path.join(ROOT,folder+"_output")

dfallmaps = pd.DataFrame(columns=['name', 'cropX', 'x', 'y', 'z', 'resize', 'newname', 'color']) 
MINSIZE = 700


def touch(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
touch(outputfolder)


pi = 22.0/7.0
def eulerToDegree(euler):
    return ( (euler) / (2 * pi) ) * 360

def get_dir_files(folder):
    # counter = 1

    # directory = folder
    directory = os.path.join(ROOT,folder)
    print(directory)

    meta_file_list = []
    os.chdir(directory)
    print(len(os.listdir(directory)))
    for filename in os.listdir(directory):
    # for item in os.listdir(root):
        # print (counter)

        if not filename.startswith('.') and os.path.isfile(os.path.join(directory, filename)):
            meta_file_list.append(filename)
    return meta_file_list

#constructing image object with first image as placeholder, just so these next functions don't bork b/c they expect obj
# image = cv2.imread(os.path.join(root,folder, meta_file_list[0]))  # read any image containing a face

meta_file_list = get_dir_files(folder)
# print(meta_file_list)

for item in meta_file_list:
    print("starting: " +item)
    try:
        image = cv2.imread(os.path.join(ROOT,folder, item)) 
    except:
        print(f"this item failed: {item}")
    if image is not None and image.shape[0]>MINSIZE and image.shape[1]>MINSIZE:


        # Initialize FaceMesh
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                             refine_landmarks=False,
                                             max_num_faces=1,
                                             min_detection_confidence=0.5
                                             ) as face_mesh:
            # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        #read any image containing a face

        if results.multi_face_landmarks:
            #construct pose_estimator object to solve pose
            pose_estimator = SelectPose(image)


            #get landmarks
            #added returning meshimage (was image)
            faceLms = pose_estimator.get_face_landmarks(results)
            meshimage = image
            mp_drawing.draw_landmarks(meshimage, faceLms, landmark_drawing_spec=drawing_spec, connections=mp_face_mesh.FACEMESH_TESSELATION) # draw every match

            pose_estimator.draw_annotation_box(meshimage, pose_estimator.r_vec, pose_estimator.t_vec)

            cropped_image = pose_estimator.crop_image(image)

            this_meta=["foo","foo","foo","foo","foo",]
            toobig=False
            # crop=True
            # this_meta[0]="foo"
            # this_meta[1]="foo"
            # this_meta[2]="foo"
            # this_meta[3]="foo"
            # this_meta[4]="foo"

            filename=f"{this_meta[0]}_{this_meta[1]}_{this_meta[2]}_{this_meta[3]}_{this_meta[4]}"
            cropname=os.path.join(ROOT,outputfolder,f"crop_{filename}_{item}")
            dfthismap = pd.DataFrame({'name': item, 'cropX':this_meta[0], 'x':this_meta[1], 'y':this_meta[2], 'z':this_meta[3], 'resize':this_meta[4], 'newname':cropname}, index=[0])
            dfallmaps = pd.concat([dfallmaps, dfthismap], ignore_index=True, sort=False)

            markedname=os.path.join(ROOT,outputfolder,f"marked_{filename}_{item}")
            cv2.imwrite(markedname, meshimage)
            if (toobig==False) and (cropped_image is not None):
                cv2.imwrite(cropname, cropped_image)



        #commenting out for refactor
        # if results.multi_face_landmarks:
        #     #get landmarks
        #     #added returning meshimage (was image)
        #     face_landmarks, crop, meshimage, filename, toobig, this_meta = get_face_landmarks(results)

        #     filename=f"{this_meta[0]}_{this_meta[1]}_{this_meta[2]}_{this_meta[3]}_{this_meta[4]}"
        #     cropname=os.path.join(ROOT,outputfolder,f"crop_{filename}_{item}")
        #     dfthismap = pd.DataFrame({'name': item, 'cropX':this_meta[0], 'x':this_meta[1], 'y':this_meta[2], 'z':this_meta[3], 'resize':this_meta[4], 'newname':cropname}, index=[0])
        #     dfallmaps = pd.concat([dfallmaps, dfthismap], ignore_index=True, sort=False)

        #     markedname=os.path.join(ROOT,outputfolder,f"marked_{filename}_{item}")
        #     cv2.imwrite(markedname, meshimage)
        #     if (toobig==False) and (crop is not None):
        #         cv2.imwrite(cropname, crop)

        else: 
            print(f"no face found {item}")
            # failedname=os.path.join(root,outputfolder,f"failed_{item}")
            # os.remove(item)


            # cv2.imwrite(failedname, image)
    else:
        print('toooooo smallllll')
        # os.remove(item)
csv_name = "allmaps_"+str(len(dfallmaps))+".csv"
dfallmaps.to_csv(os.path.join(ROOT,csv_name), index=False)
print('just wrote csv')
end = time.time()
print (end - start)


