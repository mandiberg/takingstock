"""Estimate head pose according to the facial landmarks"""
import cv2
import numpy as np
import math


class SelectPose:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, image):
        self.image = image
        self.size = (image.shape[0], image.shape[1])
        self.h = image.shape[0]
        self.w = image.shape[1]

        # self.image = image
        # self.size = (image.shape[0], image.shape[1])

        # 3D model points.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Mouth left corner
            (150.0, -150.0, -125.0)      # Mouth right corner
        ]) / 4.5

        # self.model_points_68 = self._get_full_model_points()

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])
        # self.r_vec = None
        # self.t_vec = None


    def x_element(self, elem):
        return elem[0]
    def y_element(self, elem):
        return elem[1]

    def get_face_landmarks(self,results, image):

        height = self.h
        width = self.w
        center = (self.size[1] / 2, self.size[0] / 2)

        dist=[]
        for faceNum, faceLms in enumerate(results.multi_face_landmarks):                            # loop through all matches
            faceXY = []
            for id,lm in enumerate(faceLms.landmark):                           # loop over all land marks of one face
                # ih, iw, _ = self.image.shape
                # gone direct to obj dimensions
                x,y = int(lm.x*self.w), int(lm.y*self.h)
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
            maxXY = max(faceXY, key=self.x_element)[0], max(faceXY, key=self.y_element)[1]
            minXY = min(faceXY, key=self.x_element)[0], min(faceXY, key=self.y_element)[1]

            xcenter = (maxXY[0] + minXY[0]) / 2
            ycenter = (maxXY[1] + minXY[1]) / 2

            dist.append((faceNum, (int(((xcenter-width/2)**2+(ycenter-height/2)**2)**.4)), maxXY, minXY))     # faceID, distance, maxXY, minXY

            # print(image_points)

            (success, self.r_vec, self.t_vec) = cv2.solvePnP(self.model_points, image_points,  self.camera_matrix, self.dist_coeefs)
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), self.r_vec, self.t_vec, self.camera_matrix, self.dist_coeefs)
            print(self.r_vec)
            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            # p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            # cv2.line(self.image, p1, p2, (255, 0, 0), 2)
            return faceLms




    def draw_annotation_box(self, image, color=(255, 255, 255), line_width=2):
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
                                          self.r_vec,
                                          self.t_vec,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    def eulerToDegree(self, euler):
        return ( (euler) / (2 * math.pi) ) * 360
        # Checks if a matrix is a valid rotation matrix.

    def isRotationMatrix(self, R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # pi = 22.0/7.0
    # def eulerToDegree(euler):
    #     return ( (euler) / (2 * pi) ) * 360

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAnglesToDegrees(self):
        #R is Rotation Matrix
        R, jac = cv2.Rodrigues(self.r_vec)
        print("r matrix ",R)
        #make sure it is actually a rmatrix
        assert(self.isRotationMatrix(R))

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        degreeX = self.eulerToDegree(x)
        if degreeX > 0:
            newx = 180-degreeX
        elif degreeX <0:
            newx = -180-degreeX

        #swap x and z? 
        # this is for returning euler
        # return np.array([x, y, z])

        # # this is for returning degrees
        return np.array([newx, self.eulerToDegree(y), self.eulerToDegree(z)])

    # def get_angles(self):
    #     # Get rotational matrix
    #     rmat, jac = cv2.Rodrigues(self.r_vec)
    #     # print(rmat)
    #     # Get angles
    #     angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    #     print(angles)

    #     # Get the y rotation degree
    #     x = self.eulerToDegree(angles[0])
    #     y = self.eulerToDegree(angles[1])
    #     z = self.eulerToDegree(angles[2])

    #     angles_degrees =[x,y,z]
    #     print(angles_degrees)
    #     return angles_degrees

    #     #needs to return a set of angles. This is the meta. 

    def crop_image(self,cropped_image, faceLms):
        # I don't think i need all of this. but putting it here.
        img_h = self.h
        img_w = self.w
        face_3d = []
        face_2d = []


        for idx, lm in enumerate(faceLms.landmark):
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
        # image points
        face_2d = np.array(face_2d, dtype=np.float64)

        # Convert it to the NumPy array
        # face model
        face_3d = np.array(face_3d, dtype=np.float64)


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
        if p1[1]>(height*1) and (img_h-p1[1])>(height*1):
            crop_multiplier = 1
        else:
            crop_multiplier = .25
            print('face too biiiiigggggg')
            toobig=True

        # print(crop_multiplier)
        img_h - p1[1]
        top_overlap = p1[1]-height

        
        noselinelength=p2[0]-p1[0]
        noselineheight=p2[1]-p1[1]
        r = 1
        displacement = r* math.cos(x)
        # print(displacement)
        # displacement = r* cos O 
                    

        #set crop
        # crop_multiplier = 1
        leftcrop = int(p1[0]-(height*crop_multiplier))
        rightcrop = int(p1[0]+(height*crop_multiplier))
        topcrop = int(p1[1]-(height*crop_multiplier))
        botcrop = int(p1[1]+(height*crop_multiplier))
        newsize = 750
        resize = np.round(newsize/(height*2.5), 3)

        #convert this to a list, and return it. then take this structure into the main function, and do it there, but with list[0,1,etc]
        this_meta = [crop_multiplier, np.round(x, 3), np.round(y, 3), np.round(tanZ, 3), np.round(newsize/(height*2.5), 3)]
        filename=f"{crop_multiplier}_{np.round(x, 3)}_{np.round(y, 3)}_{np.round(tanZ, 3)}_{np.round(newsize/(height*2.5), 3)}"
        # cv2.putText(crop, "scale ratio: " + str(np.round(newsize/(height*2.5),2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



        #moved this back up so it would NOT     draw map on both sets of images
        try:
            cropped_image = cv2.resize(cropped_image[topcrop:botcrop, leftcrop:rightcrop], (newsize,newsize), interpolation= cv2.INTER_LINEAR)
        except:
            cropped_image = None
            print(img_h, img_w, img_c)
               
        return cropped_image, crop_multiplier, resize, toobig


    # def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    # def _get_full_model_points(self, filename='assets/model.txt'):
    #     """Get all 68 3D model points from file"""
    #     raw_value = []
    #     with open(filename) as file:
    #         for line in file:
    #             raw_value.append(line)
    #     model_points = np.array(raw_value, dtype=np.float32)
    #     model_points = np.reshape(model_points, (3, -1)).T

    #     # Transform the model into a front view.
    #     model_points[:, 2] *= -1

    #     return model_points


    # def draw_axis(self, img, R, t):
    #     points = np.float32(
    #         [[30, 0, 0], [0, 30, 0], [0, 0, 30], [0, 0, 0]]).reshape(-1, 3)

    #     axisPoints, _ = cv2.projectPoints(
    #         points, R, t, self.camera_matrix, self.dist_coeefs)
    #     print(axisPoints)
    #     img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
    #     img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
    #     img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
    #         axisPoints[2].ravel()), (0, 0, 255), 3)

    # def draw_axes(self, img, R, t):
    #     img	= cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeefs, R, t, 30)


    # def get_pose_marks(self, marks):
        """Get marks ready for pose estimation from 68 marks"""
        pose_marks = []
        pose_marks.append(marks[30])    # Nose tip
        pose_marks.append(marks[8])     # Chin
        pose_marks.append(marks[36])    # Left eye left corner
        pose_marks.append(marks[45])    # Right eye right corner
        pose_marks.append(marks[48])    # Mouth left corner
        pose_marks.append(marks[54])    # Mouth right corner
        return pose_marks