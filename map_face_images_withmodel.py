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
import hashlib


#creating my objects
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=1, static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

start = time.time()

#regular 31.8s
#concurrent 

#declariing path and image before function, but will reassign in the main loop
#location of source and output files outside repo
ROOT= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production") 
folder ="gettyimages"
http="https://media.gettyimages.com/photos/"
# folder ="files_for_testing"
outputfolder = os.path.join(ROOT,folder+"_output")
SAVE_ORIG = True
DRAW_BOX = True

#comment this out to run testing mode with variables above
SOURCEFILE="_SELECT_FROM_faceimages_query_mouthopen.csv"
# SOURCEFILE="test2000.csv"

dfallmaps = pd.DataFrame(columns=['name', 'cropX', 'x', 'y', 'z', 'resize', 'newname', 'mouth_gap']) 
MINSIZE = 700

def touch(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
touch(outputfolder)

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

def get_hash_folders(filename):
    m = hashlib.md5()
    m.update(filename.encode('utf-8'))
    d = m.hexdigest()
    # csvWriter1.writerow(["https://upload.wikimedia.org/wikipedia/commons/"+d[0]+'/'+d[0:2]+'/'+filename])
    return d[0], d[0:2]

counter = 0
#setting the workfile list
if folder == "gettyimages":
    csvpath = os.path.join(ROOT, SOURCEFILE)
    df_files = pd.read_csv(csvpath)
    meta_file_list = df_files['contentUrl'].tolist()
else:
    meta_file_list = get_dir_files(folder)

for item in meta_file_list:

    if folder == "gettyimages":
        orig_filename = item.replace(http, "")+".jpg"
        d0, d02 = get_hash_folders(orig_filename)
        imagepath=os.path.join(ROOT,folder, "newimages",d0, d02, orig_filename)
        isExist = os.path.exists(imagepath)
        print(isExist)
    else:
        imagepath=os.path.join(ROOT,folder, item)
        orig_filename = item
        print("starting: " +item)

    try:
        image = cv2.imread(imagepath) 
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
            #construct pose object to solve pose
            pose = SelectPose(image)

            #prod is fullsized image with no cv drawing on it
            prodimage = image.copy()

            #get landmarks
            #added returning meshimage (was image)
            faceLms = pose.get_face_landmarks(results, image)

            #calculate base data from landmarks
            pose.calc_face_data(faceLms)

            # draw mesh on meshimage
            mp_drawing.draw_landmarks(image, faceLms, landmark_drawing_spec=drawing_spec, connections=mp_face_mesh.FACEMESH_TESSELATION) # draw every match

            # draw pose on meshimage
            pose.draw_annotation_box(image)

            # # TEMP draw nose and pose on production image
            if DRAW_BOX is True:
                pose.draw_annotation_box(prodimage)
                pose.draw_nose(prodimage)


            # get angles, using r_vec property stored in class
            # angles are meta. there are other meta --- size and resize or something.
            angles = pose.rotationMatrixToEulerAnglesToDegrees()
            print("angles")
            print(angles)

            # crop image - needs to be refactored, not sure which image object is being touched here.
            # should this return an object, or just save it?
            mouth_gap = pose.get_mouth_data(faceLms)
            cropped_image, resize = pose.crop_image(prodimage, faceLms)

            #annotate full size image
            pose.draw_crop_frame(image)
            
            sinY = math.sin(np.round(angles[1],2)* (math.pi/180.0))
            print(sinY)

            cv2.putText(image, "x: " + str(np.round(angles[0],2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(angles[1],2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(angles[2],2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "ytemp: "+str(np.round(angles[2],2))+" sinytemp: " + str(sinY), (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "rotation SIN: " + str(sinY), (500, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            
            filename_meta=f"{pose.crop_multiplier}_{angles[0]}_{angles[1]}_{angles[2]}_{resize}"
            cropname=os.path.join(ROOT,outputfolder,f"crop_{filename_meta}_{orig_filename}")
            markedname=os.path.join(ROOT,outputfolder,f"marked_{filename_meta}_{orig_filename}")
            print(cropname)
            print(markedname)
            print(pose.crop_multiplier)
            if cropped_image is None:
                print("null value for cropped image")
            # temporarily not writing main image
            if SAVE_ORIG is True:
                cv2.imwrite(markedname, image)
            if (pose.crop_multiplier==1) and (cropped_image is not None):
                # only writes to file and CSV if the file is cropped well and not too big
                cv2.imwrite(cropname, cropped_image)
                print("just wrote file")
                dfthismap = pd.DataFrame({'name': item, 'cropX':pose.crop_multiplier, 'x':angles[0], 'y':angles[1], 'z':angles[2], 'resize':resize, 'newname':cropname, 'mouth_gap':mouth_gap}, index=[0])
                dfallmaps = pd.concat([dfallmaps, dfthismap], ignore_index=True, sort=False)
                print("just wrote DataFrame")
            else:
                print("not hitting the write cropped_image loop")


        else: 
            print(f"no face found {item}")
            # failedname=os.path.join(root,outputfolder,f"failed_{item}")
            # os.remove(item)


            # cv2.imwrite(failedname, image)
    else:
        print('toooooo smallllll')
        # os.remove(item)
    counter = counter+1
csv_name = "allmaps_"+str(len(dfallmaps))+".csv"
dfallmaps.to_csv(os.path.join(ROOT,csv_name), index=False)
print('just wrote csv')
end = time.time()
print (end - start)

imgpermin = counter/((time.time() - start)/60)
hours = (time.time() - start)/3600

print("--- %s images per minute ---" % (imgpermin))
print("--- %s images per day ---" % (imgpermin*1440))
if imgpermin:
    print("--- %s days per 1M images ---" % (1000000/(imgpermin*1440)))

