"""Estimate head pose according to the facial landmarks"""
import cv2
import numpy as np


class SelectPose:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, image):
        self.image = image
        self.size = (image.shape[0], image.shape[1])

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

    def get_face_landmarks(self,results):

        height, width = self.image.shape[:2]
        center = (self.size[1] / 2, self.size[0] / 2)

        dist=[]
        for faceNum, faceLms in enumerate(results.multi_face_landmarks):                            # loop through all matches
            faceXY = []
            for id,lm in enumerate(faceLms.landmark):                           # loop over all land marks of one face
                ih, iw, _ = self.image.shape
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
                cv2.circle(self.image,(int(i[0]),int(i[1])),4,(255,0,0),-1)
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




    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
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

    def crop_image(self,cropped_image):

        return cropped_image


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