import numpy as np
import mediapipe as mp
import cv2
import os
import time

def x_element(elem):
    return elem[0]
def y_element(elem):
    return elem[1]

cap = cv2.VideoCapture(0)
pTime = 0
faceXY = []
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
# faceMesh = mpFaceMesh.FaceMesh(max_num_faces=5, min_detection_confidence=.9, min_tracking_confidence=.01)
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(0,1,1)

success, img = cap.read()
height, width = img.shape[:2]
size = img.shape

# 3D model points.
face3Dmodel = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
],dtype=np.float64)


dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

def draw_pose_estimation(image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
    img_h, img_w, img_c = image.shape

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

ROOT= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production") 
# folder ="sourceimages"
folder ="files_for_testing"
outputfolder = os.path.join(ROOT,folder+"_output")

# file = "auto-service-workerowner-picture-id931914734.jpg"
# # path = "sourceimages/auto-service-workerowner-picture-id931914734.jpg"
# image = cv2.imread(os.path.join(root,folder, file))  # read any image containing a face
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


meta_file_list = get_dir_files(folder)
# print(meta_file_list)

for item in meta_file_list:

# while True:
    # success, img = cap.read()
    img = cv2.imread(os.path.join(ROOT,folder, item)) 

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:                                            # if faces found
        dist=[]
        for faceNum, faceLms in enumerate(results.multi_face_landmarks):                            # loop through all matches
            mpDraw.draw_landmarks(img, faceLms, landmark_drawing_spec=drawSpec, connections=mpFaceMesh.FACEMESH_TESSELATION) # draw every match
            faceXY = []
            for id,lm in enumerate(faceLms.landmark):                           # loop over all land marks of one face
                ih, iw, _ = img.shape
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
                cv2.circle(img,(int(i[0]),int(i[1])),4,(255,0,0),-1)
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

            cv2.line(img, p1, p2, (255, 0, 0), 2)
            draw_pose_estimation(img, rotation_vector, translation_vector)



        dist.sort(key=y_element)
        # print(dist)

        for i,faceLms in enumerate(results.multi_face_landmarks):
            if i == 0:
                cv2.rectangle(img,dist[i][2],dist[i][3],(0,255,0),2)
            else:
                cv2.rectangle(img, dist[i][2], dist[i][3], (0, 0, 255), 2)


    cv2.imshow("Image", img)
    time.sleep(2) 

    cv2.waitKey(1)