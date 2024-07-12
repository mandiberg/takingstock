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
import base64

import numpy as np
import mediapipe as mp
import pandas as pd

from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, Boolean, DECIMAL, BLOB, JSON, String, Date, ForeignKey, update, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
# my ORM
from my_declarative_base import Base, Images, Keywords, SegmentTable, ImagesKeywords, Encodings, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects import mysql
import pymongo
from pymongo.errors import DuplicateKeyError

from mp_pose_est import SelectPose
from mp_db_io import DataIO
from mp_sort_pose import SortPose

#####new imports #####
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import dlib
import face_recognition_models




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

SMALL_MODEL = False
NUM_JITTERS = 1


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

def calc_encodings(image, faceLms,bbox):## changed parameters and rebuilt
    def get_dlib_all_points(landmark_points):
        raw_landmark_set = []
        for index in landmark_points:                       ######### CORRECTION: landmark_points_5_3 is the correct one for sure
            # print(faceLms.landmark[index].x)

            # second attempt, tries to project faceLms from bbox origin
            x = int(faceLms.landmark[index].x * width + bbox["left"])
            y = int(faceLms.landmark[index].y * height + bbox["top"])

            landmark_point=dlib.point([x,y])
            raw_landmark_set.append(landmark_point)
        dlib_all_points=dlib.points(raw_landmark_set)
        return dlib_all_points
        # print("all_points", all_points)
        # print(bbox)


    # second attempt, tries to project faceLms from bbox origin
    width = (bbox["right"]-bbox["left"])
    height = (bbox["bottom"]-bbox["top"])

    landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,
                      288,323,454,389,71,63,105,66,107,336,296,334,293,
                      301,168,197,5,4,75,97,2,326,305,33,160,158,133,
                      153,144,362,385,387,263,373,380,61,39,37,0,267,
                      269,291,405,314,17,84,181,78,82,13,312,308,317,
                      14,87]
                      
    landmark_points_5 = [ 263, #left eye away from centre
                       362, #left eye towards centre
                       33,  #right eye away from centre
                       133, #right eye towards centre
                        2 #bottom of nose tip 
                    ]
                    
    if SMALL_MODEL is True:landmark_points=landmark_points_5
    else:landmark_points=landmark_points_68
    
    # dlib_all_points = get_dlib_all_points(landmark_points)

    # temp test hack
    # dlib_all_points5 = get_dlib_all_points(landmark_points_5)
    dlib_all_points68 = get_dlib_all_points(landmark_points_68)

    # ymin ("top") would be y value for top left point.
    bbox_rect= dlib.rectangle(left=bbox["left"], top=bbox["top"], right=bbox["right"], bottom=bbox["bottom"])


    # if (dlib_all_points is None) or (bbox is None):return 
    # full_object_detection=dlib.full_object_detection(bbox_rect,dlib_all_points)
    # encodings=face_encoder.compute_face_descriptor(image, full_object_detection, num_jitters=NUM_JITTERS)

    if (dlib_all_points68 is None) or (bbox is None):return 
    
    # full_object_detection5=dlib.full_object_detection(bbox_rect,dlib_all_points5)
    # encodings5=face_encoder.compute_face_descriptor(image, full_object_detection5, num_jitters=NUM_JITTERS)
    # encodings5j=face_encoder.compute_face_descriptor(image, full_object_detection5, num_jitters=3)
    # encodings5v2=facerec.compute_face_descriptor(image, full_object_detection5, num_jitters=NUM_JITTERS)

    full_object_detection68=dlib.full_object_detection(bbox_rect,dlib_all_points68)
    encodings68=face_encoder.compute_face_descriptor(image, full_object_detection68, num_jitters=NUM_JITTERS)
    # encodings68j=face_encoder.compute_face_descriptor(image, full_object_detection68, num_jitters=3)
    # encodings68v2=facerec.compute_face_descriptor(image, full_object_detection68, num_jitters=NUM_JITTERS)

    # # hack of full dlib
    # dets = detector(image, 1)
    # for k, d in enumerate(dets):
    #     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    #         k, d.left(), d.top(), d.right(), d.bottom()))
    #     # Get the landmarks/parts for the face in box d.
    #     shape = sp(image, d)
    #     # print("shape")
    #     # print(shape.pop())
    #     face_descriptor = facerec.compute_face_descriptor(image, shape)
    #     # print(face_descriptor)
    #     encD=np.array(face_descriptor)


    encodings = encodings68

    # enc1=np.array(encodings5)
    # enc2=np.array(encodings68)
    # d=np.linalg.norm(enc1 - enc2, axis=0)

    # # distance = pose.get_d(encodings5, encodings68)
    # print("distance between 5 and 68 ")    
    # print(d)    


    # d=np.linalg.norm(encD - enc2, axis=0)

    # # distance = pose.get_d(encodings5, encodings68)
    # print("distance between dlib and mp hack - 68 ")    
    # print(d)    


    # # enc12=np.array(encodings5v2)
    # # enc22=np.array(encodings68v2)
    # # d=np.linalg.norm(enc12 - enc22, axis=0)

    # # # distance = pose.get_d(encodings5, encodings68)
    # # print("distance between 5v2 and 68v2 ")    
    # # print(d)    


    # enc1j=np.array(encodings5j)
    # enc2j=np.array(encodings68j)
    # d=np.linalg.norm(enc1j - enc2j, axis=0)

    # # distance = pose.get_d(encodings5, encodings68)
    # print("distance between 5j and 68j ")    
    # print(d)    

    # d=np.linalg.norm(enc1j - enc1, axis=0)
    # # distance = pose.get_d(encodings5, encodings68)
    # print("distance between 5 and 5j ")    
    # print(d)    


    # d=np.linalg.norm(enc2j - enc2, axis=0)
    # # distance = pose.get_d(encodings5, encodings68)
    # print("distance between 68 and 68j ")    
    # print(d)    


    # # d=np.linalg.norm(enc2 - enc22, axis=0)
    # # # distance = pose.get_d(encodings5, encodings68)
    # # print("distance between 68v and 68v2 ")    
    # # print(d)    


    # print(len(encodings))
    return np.array(encodings).tolist()


def find_face(image, df):
    # image is RGB
    height, width, _ = image.shape
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_det: 
        # print(">> find_face SPLIT >> with mp.solutions constructed")
        # ff_split = print_get_split(find_face_start)

        results_det=face_det.process(image)  ## [:,:,::-1] is the shortcut for converting BGR to RGB

        # print(">> find_face SPLIT >> face_det.process(image)")
        # ff_split = print_get_split(ff_split)
        
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
        # print(">> find_face SPLIT >> get_bbox()")
        # ff_split = print_get_split(ff_split)

        if bbox:

            with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                             refine_landmarks=False,
                                             max_num_faces=1,
                                             min_detection_confidence=0.5
                                             ) as face_mesh:
            # Convert the BGR image to RGB and cropping it around face boundary and process it with MediaPipe Face Mesh.
                                # crop_img = img[y:y+h, x:x+w]
                # print(">> find_face SPLIT >> const face_mesh")
                # ff_split = print_get_split(ff_split)

                results = face_mesh.process(image[bbox["top"]:bbox["bottom"],bbox["left"]:bbox["right"]])   
                # print(">> find_face SPLIT >> face_mesh.process")
                # ff_split = print_get_split(ff_split)
 
            #read any image containing a face
            if results.multi_face_landmarks:
                
                #construct pose object to solve pose
                is_face = True
                pose = SelectPose(image)

                #get landmarks
                faceLms = pose.get_face_landmarks(results, image,bbox)

                # print(">> find_face SPLIT >> got lms")
                # ff_split = print_get_split(ff_split)


                #calculate base data from landmarks
                pose.calc_face_data(faceLms)

                # get angles, using r_vec property stored in class
                # angles are meta. there are other meta --- size and resize or something.
                angles = pose.rotationMatrixToEulerAnglesToDegrees()
                mouth_gap = pose.get_mouth_data(faceLms)
                             ##### calculated face detection results
                # old version, encodes everything

                # print(">> find_face SPLIT >> done face calcs")
                # ff_split = print_get_split(ff_split)


                if is_face:

                # # new version, attempting to filter the amount that get encoded
                # if is_face  and -20 < angles[0] < 10 and np.abs(angles[1]) < 4 and np.abs(angles[2]) < 3 :
                    # Calculate Face Encodings if is_face = True
                    # print("in encodings conditional")
                    # turning off to debug
                    encodings = calc_encodings(image, faceLms,bbox) ## changed parameters
                    print(">> find_face SPLIT >> calc_encodings")
                    print(encodings)
                    
                    # ff_split = print_get_split(ff_split)

                    # # image is currently BGR so converting back to RGB
                    # image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
                    
                    # # gets bg hue and lum without bbox
                    # hue,sat,val,lum, lum_torso=sort.get_bg_hue_lum(image_rgb)
                    # print("HSV", hue, sat, val)

                    # gets bg hue and lum with bbox
                    # hue_bb,sat_bb, val_bb, lum_bb, lum_torso_bb =sort.get_bg_hue_lum(image,bbox,faceLms)
                    # print("HSV", hue_bb,sat_bb, val_bb)
                    # quit()
                #     print(encodings)
                #     exit()
                # #df.at['1', 'is_face'] = is_face

                # # debug
                # else:
                #     print("bad angles")
                #     print(angles[0])
                #     print(angles[1])
                #     print(angles[2])

                #df.at['1', 'is_face_distant'] = is_face_distant
                bbox_json = json.dumps(bbox, indent = 4) 

                df.at['1', 'face_x'] = angles[0]
                df.at['1', 'face_y'] = angles[1]
                df.at['1', 'face_z'] = angles[2]
                df.at['1', 'mouth_gap'] = mouth_gap
                df.at['1', 'face_landmarks'] = pickle.dumps(faceLms)
                df.at['1', 'bbox'] = bbox_json
                if SMALL_MODEL is True:
                    df.at['1', 'face_encodings'] = pickle.dumps(encodings)
                else:
                    df.at['1', 'face_encodings68'] = pickle.dumps(encodings)
    df.at['1', 'is_face'] = is_face
    # print(">> find_face SPLIT >> prepped dataframe")
    # ff_split = print_get_split(ff_split)

    return df


task=[1001,"/Users/michaelmandiberg/Documents/projects-active/facemap_production/PFP/misha_PFP.jpg"]

df = pd.DataFrame(columns=['image_id','is_face','is_body','is_face_distant','face_x','face_y','face_z','mouth_gap','face_landmarks','bbox','face_encodings','face_encodings68_J','body_landmarks'])
# print(task)
df.at['1', 'image_id'] = task[0]
cap_path = task[1]
# print(">> SPLIT >> made DF, about to imread")
# pr_split = print_get_split(pr_split)

try:
    # i think i'm doing this twice. I should just do it here. 
    image = cv2.imread(cap_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
    # image is RGB now 
    # this is for when you need to move images into a testing folder structure
    # save_image_elsewhere(image, task)
except:
    print(f"[process_image]this item failed: {task}")

# print(">> SPLIT >> done imread, about to find face")
# pr_split = print_get_split(pr_split)

if image is not None:
    # Do FaceMesh
    df = find_face(image, df)
    # print(">> SPLIT >> done find_face")
    # pr_split = print_get_split(pr_split)

    # Do Body Pose
    # temporarily commenting this out
    # to reactivate, will have to accept return of is_body, body_landmarks
    # df = find_body(image, df)

    # print(">> SPLIT >> done find_body")
    # pr_split = print_get_split(pr_split)

    # for testing: this will save images into folders for is_face, is_body, and none. 
    # only save images that aren't too smallllll
    # save_image_triage(image,df)
else:
    print('toooooo smallllll')
    # I should probably assign no_good here...?

print(df)