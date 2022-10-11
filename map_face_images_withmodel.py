import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import mediapipe as mp
import pandas as pd

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
folder ="files_for_testing"
outputfolder = os.path.join(ROOT,folder+"_output")

# file = "auto-service-workerowner-picture-id931914734.jpg"
# # path = "sourceimages/auto-service-workerowner-picture-id931914734.jpg"
# image = cv2.imread(os.path.join(root,folder, file))  # read any image containing a face
dfallmaps = pd.DataFrame(columns=['name', 'cropX', 'x', 'y', 'z', 'resize', 'newname', 'color']) 
MINSIZE = 700


#carlos
# 3D model points.
face3Dmodel = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
],dtype=np.float64)


def touch(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
touch(outputfolder)


pi = 22.0/7.0
def eulerToDegree(euler):
    return ( (euler) / (2 * pi) ) * 360

def x_element(elem):
    return elem[0]
def y_element(elem):
    return elem[1]

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

def draw_pose_estimation(img, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
    img_h, img_w, img_c = img.shape

    img_size=(img_w, img_h)
    size = img_size
    focal_length = size[1]
    camera_center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, camera_center[0]],
         [0, focal_length, camera_center[1]],
         [0, 0, 1]], dtype="double")
    # Assuming no lens distortion
    dist_coeefs = np.zeros((4, 1))


    """Draw a 3D box as annotation of pose"""
    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

def get_face_landmarks(results):
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    height, width = image.shape[:2]
    size = image.shape
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist=[]
    for faceNum, faceLms in enumerate(results.multi_face_landmarks):                            # loop through all matches
        mp_drawing.draw_landmarks(image, faceLms, landmark_drawing_spec=drawing_spec, connections=mp_face_mesh.FACEMESH_TESSELATION) # draw every match
        faceXY = []
        for id,lm in enumerate(faceLms.landmark):                           # loop over all land marks of one face
            ih, iw, _ = image.shape
            x,y = int(lm.x*iw), int(lm.y*ih)
            # print(lm)
            faceXY.append((x, y))                                           # put all xy points in neat array
        image_points = np.array([
            faceXY[1],      # "nose"
            faceXY[152],    # "chin"
            faceXY[226],    # "left eye"
            faceXY[446],    # "right eye"
            faceXY[57],     # "left mouth"
            faceXY[287]     # "right mouth"
        ], dtype="double")

        for i in image_points:
            cv2.circle(image,(int(i[0]),int(i[1])),4,(255,0,0),-1)
        maxXY = max(faceXY, key=x_element)[0], max(faceXY, key=y_element)[1]
        minXY = min(faceXY, key=x_element)[0], min(faceXY, key=y_element)[1]

        xcenter = (maxXY[0] + minXY[0]) / 2
        ycenter = (maxXY[1] + minXY[1]) / 2

        dist.append((faceNum, (int(((xcenter-width/2)**2+(ycenter-height/2)**2)**.4)), maxXY, minXY))     # faceID, distance, maxXY, minXY

        # print(image_points)

        (success, rotation_vector, translation_vector) = cv2.solvePnP(face3Dmodel, image_points,  camera_matrix, dist_coeffs)
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        print(rotation_vector)
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(image, p1, p2, (255, 0, 0), 2)
        draw_pose_estimation(image, rotation_vector, translation_vector)
        return image


    # dist.sort(key=y_element)
    # # print(dist)

    # for i,faceLms in enumerate(results.multi_face_landmarks):
    #     if i == 0:
    #         cv2.rectangle(img,dist[i][2],dist[i][3],(0,255,0),2)
    #     else:
    #         cv2.rectangle(img, dist[i][2], dist[i][3], (0, 0, 255), 2)


    # for face_landmarks in results.multi_face_landmarks:
    #     for idx, lm in enumerate(face_landmarks.landmark):
    #         if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199 or idx == 10 or idx == 152:
    #             if idx == 1:
    #                 nose_2d = (lm.x * img_w, lm.y * img_h)
    #                 nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
    #             elif idx == 10:
    #                 top_2d = (lm.x * img_w, lm.y * img_h)
    #             elif idx == 152:
    #                 bottom_2d = (lm.x * img_w, lm.y * img_h)

    #             x, y = int(lm.x * img_w), int(lm.y * img_h)

    #             # Get the 2D Coordinates
    #             face_2d.append([x, y])

    #             # Get the 3D Coordinates
    #             face_3d.append([x, y, lm.z])       
        
    #     # Convert it to the NumPy array
    #     # image points
    #     face_2d = np.array(face_2d, dtype=np.float64)

    #     # Convert it to the NumPy array
    #     # face model
    #     face_3d = np.array(face_3d, dtype=np.float64)

    #     # The camera matrix
    #     focal_length = 1 * img_w

    #     cam_matrix = np.array([ [focal_length, 0, img_h / 2],
    #                             [0, focal_length, img_w / 2],
    #                             [0, 0, 1]])

    #     # The distortion parameters
    #     dist_matrix = np.zeros((4, 1), dtype=np.float64)

    #     # Solve PnP
    #     success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    #     # Get rotational matrix
    #     rmat, jac = cv2.Rodrigues(rot_vec)
    #     # print(rmat)
    #     # Get angles
    #     angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    #     # print(angles)

    #     # Get the y rotation degree
    #     x = eulerToDegree(angles[0])
    #     y = eulerToDegree(angles[1])
    #     z = eulerToDegree(angles[2])


    #     # --this is me trying to debug the pose estimation errors--
    #     # print(f"x: {x} y: {y} z {z}")
    #     # print(f"Qx: {Qx} Qy: {Qy} Qz {Qz}")

    #     # print('*' * 80)
    #     # print("Angle: ", angles)
    #     # # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
    #     # x = np.arctan2(Qx[2][1], Qx[2][2])
    #     # y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
    #     # z = np.arctan2(Qz[0][0], Qz[1][0])
    #     # print("AxisX: ", x)
    #     # print("AxisY: ", y)
    #     # print("AxisZ: ", z)
    #     # print('*' * 80)

    #     # Display the nose direction
    #     nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        
    #     # display the pose estimation box projection

    #     rotation_vector=rot_vec
    #     translation_vector=trans_vec
        
    #     # print(image)
    #     print(rotation_vector)
    #     print(translation_vector)
    #     draw_pose_estimation(image, rotation_vector, translation_vector)



    #     # draw_pose_estimation(image, rotation_vector, translation_vector)


    #     #set main points for drawing/cropping
    #     #p1 is tip of nose
    #     p1 = (int(nose_2d[0]), int(nose_2d[1]))
    #     p2 = (int(nose_2d[0] + y * 70) , int(nose_2d[1] - x * 70))
    #     ptop = (int(top_2d[0]), int(top_2d[1]))
    #     pbot = (int(bottom_2d[0]), int(bottom_2d[1]))
    #     height = int(pbot[1]-ptop[1])

    #     # math.atan2(dy, dx)
    #     # ptop = (int(top_2d[0]), int(top_2d[1]))
    #     # pbot = (int(bottom_2d[0]), int(bottom_2d[1]))
    #     tanZ = math.degrees(math.atan2((top_2d[1]-bottom_2d[1]),(top_2d[0]-bottom_2d[0])))+90
    #     # (y2 - y1)/(x2-x1)

    #     # print(f"is {p1[1]} greater than {height}")
    #     # print(f"is {img_h-p1[1]} greater than {height}")

    #     toobig = False
    #     # if p1[1]>(height*1.5) and (img_h-p1[1])>(height*1.5):
    #     #     crop_multiplier = 1.5
    #     if p1[1]>(height*1) and (img_h-p1[1])>(height*1):
    #         crop_multiplier = 1
    #     # elif p1[1]>(height*.75) and (img_h-p1[1])>(height*.75):
    #     #     crop_multiplier = .75
    #     # elif p1[1]>(height*.65) and (img_h-p1[1])>(height*.65):
    #     #     crop_multiplier = .65
    #     # elif p1[1]>(height*.5) and (img_h-p1[1])>(height*.5):
    #     #     crop_multiplier = .5
    #     else:
    #         crop_multiplier = .25
    #         print('face too biiiiigggggg')
    #         toobig=True

    #     # print(crop_multiplier)
    #     # img_h - p1[1]
    #     # top_overlap = p1[1]-height

        
    #     # noselinelength=p2[0]-p1[0]
    #     # noselineheight=p2[1]-p1[1]
    #     # r = 1
    #     # displacement = r* math.cos(x)
    #     # print(displacement)
    #     # # displacement = r* cos O 
                

    #     #set crop
    #     # crop_multiplier = 1
    #     leftcrop = int(p1[0]-(height*crop_multiplier))
    #     rightcrop = int(p1[0]+(height*crop_multiplier))
    #     topcrop = int(p1[1]-(height*crop_multiplier))
    #     botcrop = int(p1[1]+(height*crop_multiplier))
    #     newsize = 750
          

    #     #convert this to a list, and return it. then take this structure into the main function, and do it there, but with list[0,1,etc]
    #     this_meta = [crop_multiplier, np.round(x, 3), np.round(y, 3), np.round(tanZ, 3), np.round(newsize/(height*2.5), 3)]
    #     filename=f"{crop_multiplier}_{np.round(x, 3)}_{np.round(y, 3)}_{np.round(tanZ, 3)}_{np.round(newsize/(height*2.5), 3)}"
    #     # cv2.putText(crop, "scale ratio: " + str(np.round(newsize/(height*2.5),2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



    #     #moved this back up so it would NOT     draw map on both sets of images
    #     try:
    #         crop = cv2.resize(image[topcrop:botcrop, leftcrop:rightcrop], (newsize,newsize), interpolation= cv2.INTER_LINEAR)
    #     except:
    #         crop = None
    #         print(img_h, img_w, img_c)
                
    #     #draw data
    #     cv2.circle(image, (p1), radius=5, color=(0, 0, 255), thickness=-1)
    #     cv2.line(image, p1, p2, (255, 0, 0), 3)
    #     cv2.line(image, ptop, pbot, (0, 255, 0), 3)
    #     cv2.rectangle(image, (leftcrop,topcrop), (rightcrop,botcrop), (255,0,0), 2)
    #     # Add the text on the image
    #     cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #     cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #     cv2.putText(image, "z: " + str(np.round(tanZ,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #     #draw mesh on image
    #     mp_drawing.draw_landmarks(
    #                 image=image,
    #                 landmark_list=face_landmarks,
    #                 connections=mp_face_mesh.FACEMESH_TESSELATION,
    #                 landmark_drawing_spec=drawing_spec,
    #                 connection_drawing_spec=drawing_spec)

    #         #added returning meshimage as image
    #     return face_landmarks, crop, image, filename, toobig, this_meta




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
            #get landmarks
            #added returning meshimage (was image)
            meshimage = get_face_landmarks(results)

            this_meta=["foo","foo","foo","foo","foo",]
            toobig=False
            crop=True
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
            if (toobig==False) and (crop is not None):
                cv2.imwrite(cropname, crop)



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


