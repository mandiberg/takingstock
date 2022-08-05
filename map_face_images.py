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
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

start = time.time()

#regular 31.8s
#concurrent 

#declariing path and image before function, but will reassign in the main loop
ROOT="/Users/michaelmandiberg/Documents/projects-active/facemap_production/"
# folder ="sourceimages"
folder ="images5test"
# file = "auto-service-workerowner-picture-id931914734.jpg"
# # path = "sourceimages/auto-service-workerowner-picture-id931914734.jpg"
# image = cv2.imread(os.path.join(root,folder, file))  # read any image containing a face
dfallmaps = pd.DataFrame(columns=['name', 'cropX', 'x', 'y', 'z', 'resize', 'newname', 'color']) 
MINSIZE = 700

def touch(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

outputfolderRGB = os.path.join(ROOT,"face_mesh_outputsRGB")
outputfolderBW = os.path.join(ROOT,"face_mesh_outputsBW")
outputfolderMEH = os.path.join(ROOT,"face_mesh_outputsMEH")
outputfolder = os.path.join(ROOT,"face_mesh_outputsORIG")

touch(outputfolderBW)
touch(outputfolderRGB)
touch(outputfolderMEH)
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

#constructing image object with first image as placeholder, just so these next functions don't bork b/c they expect obj
# image = cv2.imread(os.path.join(root,folder, meta_file_list[0]))  # read any image containing a face


def image_iscolor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # split the channels
    el, ea, eb = cv2.split(image)
    # obtain difference between A and B channel at every pixel location
    de = abs(ea-eb)
    # find the mean of this difference
    mean_e = np.mean(de)
    std_a = np.std(ea)
    std_b = np.std(eb)
    return (mean_e, std_a, std_b)


def get_face_landmarks(results):
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    for face_landmarks in results.multi_face_landmarks:
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199 or idx == 10 or idx == 152:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                elif idx == 10:
                    top_2d = (lm.x * img_w, lm.y * img_h)
                elif idx == 152:
                    bottom_2d = (lm.x * img_w, lm.y * img_h)

                x, y = int(lm.x * img_w), int(lm.y * img_h)

                # Get the 2D Coordinates
                face_2d.append([x, y])

                # Get the 3D Coordinates
                face_3d.append([x, y, lm.z])       
        
        # Convert it to the NumPy array
        face_2d = np.array(face_2d, dtype=np.float64)

        # Convert it to the NumPy array
        face_3d = np.array(face_3d, dtype=np.float64)

        # The camera matrix
        focal_length = 1 * img_w

        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

        # The distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)
        # print(rmat)
        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        # print(angles)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        # print(f"x: {x} y: {y} z {z}")
        # print(f"Qx: {Qx} Qy: {Qy} Qz {Qz}")

        # print('*' * 80)
        # print("Angle: ", angles)
        # # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
        # x = np.arctan2(Qx[2][1], Qx[2][2])
        # y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
        # z = np.arctan2(Qz[0][0], Qz[1][0])
        # print("AxisX: ", x)
        # print("AxisY: ", y)
        # print("AxisZ: ", z)
        # print('*' * 80)

        # Display the nose direction
        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

        #set main points for drawing/cropping
        #p1 is tip of nose
        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 70) , int(nose_2d[1] - x * 70))
        ptop = (int(top_2d[0]), int(top_2d[1]))
        pbot = (int(bottom_2d[0]), int(bottom_2d[1]))
        height = int(pbot[1]-ptop[1])

        # math.atan2(dy, dx)
        # ptop = (int(top_2d[0]), int(top_2d[1]))
        # pbot = (int(bottom_2d[0]), int(bottom_2d[1]))
        tanZ = math.degrees(math.atan2((top_2d[1]-bottom_2d[1]),(top_2d[0]-bottom_2d[0])))+90
        # (y2 - y1)/(x2-x1)

        # print(f"is {p1[1]} greater than {height}")
        # print(f"is {img_h-p1[1]} greater than {height}")

        toobig = False
        # if p1[1]>(height*1.5) and (img_h-p1[1])>(height*1.5):
        #     crop_multiplier = 1.5
        if p1[1]>(height*1) and (img_h-p1[1])>(height*1):
            crop_multiplier = 1
        # elif p1[1]>(height*.75) and (img_h-p1[1])>(height*.75):
        #     crop_multiplier = .75
        # elif p1[1]>(height*.65) and (img_h-p1[1])>(height*.65):
        #     crop_multiplier = .65
        # elif p1[1]>(height*.5) and (img_h-p1[1])>(height*.5):
        #     crop_multiplier = .5
        else:
            crop_multiplier = .25
            print('face too biiiiigggggg')
            toobig=True

        # print(crop_multiplier)
        # img_h - p1[1]
        # top_overlap = p1[1]-height

        
        # noselinelength=p2[0]-p1[0]
        # noselineheight=p2[1]-p1[1]
        # r = 1
        # displacement = r* math.cos(x)
        # print(displacement)
        # # displacement = r* cos O 
                

        #set crop
        # crop_multiplier = 1
        leftcrop = int(p1[0]-(height*crop_multiplier))
        rightcrop = int(p1[0]+(height*crop_multiplier))
        topcrop = int(p1[1]-(height*crop_multiplier))
        botcrop = int(p1[1]+(height*crop_multiplier))
        newsize = 750
          

        #convert this to a list, and return it. then take this structure into the main function, and do it there, but with list[0,1,etc]
        this_meta = [crop_multiplier, np.round(x, 3), np.round(y, 3), np.round(tanZ, 3), np.round(newsize/(height*2.5), 3)]
        filename=f"{crop_multiplier}_{np.round(x, 3)}_{np.round(y, 3)}_{np.round(tanZ, 3)}_{np.round(newsize/(height*2.5), 3)}"
        # cv2.putText(crop, "scale ratio: " + str(np.round(newsize/(height*2.5),2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



        #moved this back up so it would NOT     draw map on both sets of images
        try:
            crop = cv2.resize(image[topcrop:botcrop, leftcrop:rightcrop], (newsize,newsize), interpolation= cv2.INTER_LINEAR)
        except:
            crop = None
            print(img_h, img_w, img_c)
                
        #draw data
        cv2.circle(image, (p1), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.line(image, p1, p2, (255, 0, 0), 3)
        cv2.line(image, ptop, pbot, (0, 255, 0), 3)
        cv2.rectangle(image, (leftcrop,topcrop), (rightcrop,botcrop), (255,0,0), 2)
        # Add the text on the image
        cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "tanZ: " + str(np.round(tanZ,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #draw mesh on image
        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

            #added returning meshimage as image
        return face_landmarks, crop, image, filename, toobig, this_meta


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
            face_landmarks, crop, meshimage, filename, toobig, this_meta = get_face_landmarks(results)
            
            #check if color/bw
            colordata = image_iscolor(image)
            # print(colordata)
            if  (colordata[0] > 6 and colordata[1] > 12 and colordata[2] > 12) or (colordata[1] > 20 or colordata[2] > 20):
                print('its a color image...')
                color=True
                outputfolder = outputfolderRGB

            # elif colordata[1] > 20 or colordata[2] > 20 :
            #     print('its a color image...')
            #     color=True
            #     outputfolder = outputfolderRGB

            elif (colordata[0] < 6) or (colordata[0] > 100 and colordata[1] < 10 and colordata[2] < 10 ) :
                print('Black and white image...')
                color=False
                outputfolder = outputfolderBW

            # elif colordata[0] > 100 and colordata[1] < 10 and colordata[2] < 10:
            #     color=False
            #     outputfolder = outputfolderBW

            else:
                print('Not Sure...')
                color=None
                outputfolder = outputfolderMEH

            # #diagnostic data for lab color/bw determination
            # cv2.putText(crop, f"diff: {round(colordata[0])} a: {round(colordata[1])} b:{    round(colordata[2])}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            filename=f"{this_meta[0]}_{this_meta[1]}_{this_meta[2]}_{this_meta[3]}_{this_meta[4]}"
            cropname=os.path.join(ROOT,outputfolder,f"crop_{filename}_{item}")
            dfthismap = pd.DataFrame({'name': item, 'cropX':this_meta[0], 'x':this_meta[1], 'y':this_meta[2], 'z':this_meta[3], 'resize':this_meta[4], 'newname':cropname, 'color':color}, index=[0])
            dfallmaps = pd.concat([dfallmaps, dfthismap], ignore_index=True, sort=False)

            # #draw mesh on image / moved to function
            # mp_drawing.draw_landmarks(
            #             image=crop,
            #             landmark_list=face_landmarks,
            #             connections=mp_face_mesh.FACEMESH_TESSELATION,
            #             landmark_drawing_spec=drawing_spec,
            #             connection_drawing_spec=drawing_spec)
            markedname=os.path.join(ROOT,outputfolder,f"marked_{filename}_{item}")
            # print(markedname)
            # print(meshimage)
            cv2.imwrite(markedname, meshimage)
            if (toobig==False) and (crop is not None):
                cv2.imwrite(cropname, crop)

        else: 
            print(f"no face found {item}")
            # failedname=os.path.join(root,outputfolder,f"failed_{item}")
            # os.remove(item)


            # cv2.imwrite(failedname, image)
    else:
        print('toooooo smallllll')
        os.remove(item)
csv_name = "allmaps_"+str(len(dfallmaps))+".csv"
dfallmaps.to_csv(os.path.join(ROOT,csv_name), index=False)
print('just wrote csv')
end = time.time()
print (end - start)


