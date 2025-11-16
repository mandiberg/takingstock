"""Estimate head pose according to the facial landmarks"""
import cv2
import numpy as np
import math
import mediapipe as mp


class SelectPose:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, image):

        # image is mp.Image
        self.image = self.ensure_image_mp(image)
        self.size = (self.image.height, self.image.width)
        self.h = self.image.height
        self.w = self.image.width
        self.VERBOSE = False

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


    def ensure_image_cv2(self,image):
        # convert image back to numpy array if it's a mediapipe image
        if isinstance(image, mp.Image):
            image = image.numpy_view()
        # Ensure image is 3-channel (RGB) and uint8 for dlib
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = image.astype(np.uint8)
        return image

    def ensure_image_mp(self,image):
        # convert image back to mediapipe image if it's a numpy array
        if isinstance(image, np.ndarray):
            image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))
        return image

    def x_element(self, elem):
        return elem[0]
    def y_element(self, elem):
        return elem[1]



    def get_face_landmarks(self,results,bbox):

        height = self.h
        width = self.w
        center = (self.size[1] / 2, self.size[0] / 2)

        dist=[]
        for faceNum, faceLms in enumerate(results.multi_face_landmarks):                            # loop through all matches
            faceXY, image_points = self.extract_key_lms_for_faceXYZ(bbox, faceLms)

            # #this is where the face points are written to the image
            # # turning this off for production run
            # for i in image_points:
            #     cv2.circle(image,(int(i[0]),int(i[1])),4,(255,0,0),-1)

            maxXY = max(faceXY, key=self.x_element)[0], max(faceXY, key=self.y_element)[1]
            minXY = min(faceXY, key=self.x_element)[0], min(faceXY, key=self.y_element)[1]

            xcenter = (maxXY[0] + minXY[0]) / 2
            ycenter = (maxXY[1] + minXY[1]) / 2

            dist.append((faceNum, (int(((xcenter-width/2)**2+(ycenter-height/2)**2)**.4)), maxXY, minXY))     # faceID, distance, maxXY, minXY

            # if self.VERBOSE: print(image_points)

            (success, self.r_vec, self.t_vec) = cv2.solvePnP(self.model_points, image_points,  self.camera_matrix, self.dist_coeefs)
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), self.r_vec, self.t_vec, self.camera_matrix, self.dist_coeefs)
            # if self.VERBOSE: print(self.r_vec)
            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            # p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            # cv2.line(self.image, p1, p2, (255, 0, 0), 2)
            return faceLms

    # def extract_key_lms_for_faceXYZ(self, bbox, faceLms):
    #     faceXY = []
    #     for id, lm in enumerate(faceLms.landmark):
    #         x = int(lm.x * (bbox["right"]-bbox["left"]) + bbox["left"])
    #         y = int(lm.y * (bbox["bottom"]-bbox["top"]) + bbox["top"])
    #         faceXY.append((x, y))

    #     # Use more landmarks for stability
    #     landmark_indices = [
    #         1,    # nose tip
    #         152,  # chin
    #         33,   # left eye outer
    #         133,  # left eye inner
    #         263,  # right eye inner
    #         362,  # right eye outer
    #         61,   # left mouth corner
    #         291,  # right mouth corner
    #         199,  # lower lip
    #         10,   # upper lip center (forehead area)
    #         234,  # left cheek
    #         454,  # right cheek
    #     ]
        
    #     image_points = np.array([faceXY[i] for i in landmark_indices], dtype="double")
        
    #     # Update model_points accordingly with 3D coordinates
    #     return faceXY, image_points

    def draw_face_landmarks(self, image, faceLms, bbox):
        # Draw the landmarks
        for id, lm in enumerate(faceLms.landmark):
            # if self.VERBOSE: print(bbox)
            bbox_width = bbox["right"]-bbox["left"]
            bbox_height = bbox["bottom"]-bbox["top"]
            x = int(bbox["left"] + lm.x * bbox_width)
            y = int(bbox["top"] + lm.y * bbox_height)
            # if self.VERBOSE: print(x,y)
            cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
        return image



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

    def extract_key_lms_for_faceXYZ(self, bbox, faceLms):
        """Extract landmark points and convert to image coordinates"""
        faceXY = []
        for id, lm in enumerate(faceLms.landmark):
            x = int(lm.x * (bbox["right"]-bbox["left"]) + bbox["left"])
            y = int(lm.y * (bbox["bottom"]-bbox["top"]) + bbox["top"]) 
            faceXY.append((x, y))

        # Use 6 proven landmarks
        landmark_indices = [
            1,    # nose tip
            152,  # chin
            226,  # left eye left corner
            446,  # right eye right corner  
            57,   # left mouth corner
            287,  # right mouth corner
        ]
        
        image_points = np.array([faceXY[i] for i in landmark_indices], dtype="double")
        return faceXY, image_points


    def get_model_points_corrected(self):
        """
        6-point 3D model with CORRECTED coordinate system.
        
        The issue: OpenCV's camera coordinate system has:
        - X pointing right
        - Y pointing DOWN (not up!)
        - Z pointing away from camera (into scene)
        
        Most face models assume Y points UP. We need to flip Y coordinates.
        """
        # Standard model (Y points up)
        model_points_standard = np.array([
            (0.0, 0.0, 0.0),          # Nose tip (origin)
            (0.0, -330.0, -65.0),     # Chin (below nose)
            (-225.0, 170.0, -135.0),  # Left eye (above nose)
            (225.0, 170.0, -135.0),   # Right eye (above nose)
            (-150.0, -150.0, -125.0), # Left mouth (below nose)
            (150.0, -150.0, -125.0)   # Right mouth (below nose)
        ], dtype="double")
        
        # Flip Y axis to match OpenCV's camera coordinates (Y down)
        model_points = model_points_standard.copy()
        model_points[:, 1] *= -1  # Flip Y
        
        return model_points


    def solve_head_pose_robust(self, image_points):
        """
        Solve for head pose with multiple fallback strategies.
        """
        # Strategy 1: Try ITERATIVE first (most stable)
        try:
            success, r_vec, t_vec = cv2.solvePnP(
                self.model_points, 
                image_points,  
                self.camera_matrix, 
                self.dist_coeefs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success and r_vec is not None:
                return success, r_vec, t_vec, "ITERATIVE"
        except cv2.error:
            pass
        
        # Strategy 2: Try EPNP (faster, sometimes more robust)
        try:
            success, r_vec, t_vec = cv2.solvePnP(
                self.model_points, 
                image_points,  
                self.camera_matrix, 
                self.dist_coeefs,
                flags=cv2.SOLVEPNP_EPNP
            )
            if success and r_vec is not None:
                return success, r_vec, t_vec, "EPNP"
        except cv2.error:
            pass
        
        # Strategy 3: Try RANSAC as last resort (only if above fail)
        try:
            success, r_vec, t_vec, inliers = cv2.solvePnPRansac(
                self.model_points, 
                image_points,  
                self.camera_matrix, 
                self.dist_coeefs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=8.0,
                iterationsCount=200,
                confidence=0.95
            )
            if success and r_vec is not None:
                return success, r_vec, t_vec, "RANSAC"
        except cv2.error:
            pass
        
        return False, None, None, "FAILED"


    def rotationMatrixToEulerAngles(self):
        """
        Convert rotation vector to Euler angles in RADIANS.
        Returns pitch, yaw, roll (X, Y, Z rotations).
        
        Standard Euler extraction - angles may be outside [-pi/2, pi/2].
        """
        if self.r_vec is None:
            return np.array([0.0, 0.0, 0.0])
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(self.r_vec)
        
        # Extract Euler angles using standard decomposition
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])   # pitch
            y = np.arctan2(-R[2,0], sy)       # yaw
            z = np.arctan2(R[1,0], R[0,0])    # roll
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        
        return np.array([x, y, z])


    def normalize_euler_angles(self, pitch, yaw, roll):
        """
        Normalize Euler angles to intuitive ranges.
        Handles the wrap-around cases.
        
        Args:
            pitch, yaw, roll: angles in degrees
        
        Returns:
            Normalized angles in degrees where:
            - Pitch: [-90, 90] (negative = down, positive = up)
            - Yaw: [-90, 90] (negative = left, positive = right)  
            - Roll: [-90, 90] (negative = tilt left, positive = tilt right)
        """
        # Handle roll wrap-around
        # If roll is near ±180, it's actually a small tilt in opposite direction
        if roll > 90:
            roll = roll - 180
            pitch = 180 - pitch
            yaw = -yaw
        elif roll < -90:
            roll = roll + 180
            pitch = 180 - pitch
            yaw = -yaw
        
        # Normalize pitch to [-180, 180] then to [-90, 90]
        while pitch > 180:
            pitch -= 360
        while pitch < -180:
            pitch += 360
        
        # If pitch is way off, it might be due to coordinate system flip
        if pitch > 90:
            pitch = 180 - pitch
            yaw = -yaw
            roll = -roll
        elif pitch < -90:
            pitch = -180 - pitch
            yaw = -yaw
            roll = -roll
        
        # Normalize yaw to [-180, 180]
        while yaw > 180:
            yaw -= 360
        while yaw < -180:
            yaw += 360
        
        return pitch, yaw, roll


    def rotationMatrixToEulerDegrees(self):
        """Convert rotation vector to normalized Euler angles in DEGREES"""
        rad = self.rotationMatrixToEulerAngles()
        deg = np.degrees(rad)
        
        # Normalize to intuitive ranges
        pitch, yaw, roll = self.normalize_euler_angles(deg[0], deg[1], deg[2])
        
        return np.array([pitch, yaw, roll])


    def get_roll_from_landmarks(self, faceLms):
        """
        Estimate roll angle from eye landmarks.
        This is often more reliable than PnP for roll.
        Returns angle in DEGREES in range [-90, +90].
        """
        # Get eye corners
        left_eye_outer = faceLms.landmark[33]   # left eye outer
        right_eye_outer = faceLms.landmark[263] # right eye outer
        
        # Calculate angle from eye line
        dx = right_eye_outer.x - left_eye_outer.x
        dy = right_eye_outer.y - left_eye_outer.y
        
        # Roll angle (positive = head tilted right/clockwise)
        roll_rad = math.atan2(dy, dx)
        roll_deg = math.degrees(roll_rad)
        
        # Normalize to [-90, 90] range
        if roll_deg > 90:
            roll_deg = roll_deg - 180
        elif roll_deg < -90:
            roll_deg = roll_deg + 180
        
        return roll_deg


    def calculate_face_pose_final(self, bbox, faceLms):
        """
        Complete face pose calculation with best practices.
        Returns dict with pitch, yaw, roll in degrees.
        All angles normalized to reasonable ranges:
        - Pitch: [-90, +90] negative=down, positive=up
        - Yaw: [-90, +90] negative=left, positive=right
        - Roll: [-90, +90] negative=tilt left, positive=tilt right
        """
        # Extract landmarks
        faceXY, image_points = self.extract_key_lms_for_faceXYZ(bbox, faceLms)
        
        # Solve for pose
        success, r_vec, t_vec, method = self.solve_head_pose_robust(image_points)
        
        if not success:
            return None
        
        # Store results
        self.r_vec = r_vec
        self.t_vec = t_vec
        
        # Get PnP angles (now normalized in rotationMatrixToEulerAngles)
        pnp_angles = self.rotationMatrixToEulerDegrees()
        
        # Get roll from eye line (often more stable)
        eye_roll = self.get_roll_from_landmarks(faceLms)
        
        # For roll: Use eye-based estimate as primary, PnP as secondary
        # Eye-based is more geometric and stable for roll
        # Weight more heavily toward eye-based (80/20 instead of 70/30)
        blended_roll = 0.2 * pnp_angles[2] + 0.8 * eye_roll
        
        return {
            'pitch': pnp_angles[0],
            'yaw': pnp_angles[1],
            'roll': blended_roll,
            'roll_pnp': pnp_angles[2],
            'roll_eyes': eye_roll,
            'method': method,
            'r_vec': r_vec,
            't_vec': t_vec
        }


    def point(self,coords):
        newpoint = (int(coords[0]), int(coords[1]))
        return newpoint

    def dist(self,p, q):
        """ 
        Return euclidean distance between points p and q
        assuming both to have the same number of dimensions
        """
        # sum of squared difference between coordinates
        s_sq_difference = 0
        for p_i,q_i in zip(p,q):
            s_sq_difference += (p_i - q_i)**2
        
        # take sq root of sum of squared difference
        distance = s_sq_difference**0.5
        return distance    

    def calc_face_data(self, faceLms):

        # check for face_2d, and if not exist, then get
        if not hasattr(self, 'face_2d'): 
            self.get_face_2d_3d(faceLms)

        # check for face height, and if not exist, then get
        if not hasattr(self, 'face_height'): 
            self.get_faceheight_data(faceLms)
        

    def get_face_2d_point(self, faceLms, point):
        # I don't think i need all of this. but putting it here.
        img_h = self.h
        img_w = self.w
        for idx, lm in enumerate(faceLms.landmark):
            if idx == point:
                pointXY = (lm.x * img_w, lm.y * img_h)
        return pointXY

    def get_face_2d_3d(self, faceLms):
        # I don't think i need all of this. but putting it here.
        img_h = self.h
        img_w = self.w
        face_3d = []
        face_2d = []
        for idx, lm in enumerate(faceLms.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199 or idx == 10 or idx == 152:
                x, y = int(lm.x * img_w), int(lm.y * img_h)

                # Get the 2D Coordinates
                face_2d.append([x, y])

                # Get the 3D Coordinates
                face_3d.append([x, y, lm.z])      

        # Convert it to the NumPy array
        # image points
        self.face_2d = np.array(face_2d, dtype=np.float64)

        # Convert it to the NumPy array
        # face model
        self.face_3d = np.array(face_3d, dtype=np.float64)

        return face_2d, face_3d

    def get_eye_pitch(self, faceLms):
        def get_average_y(landmarks, point1, point2):
            # if self.VERBOSE: print("point1, point2",point1,point2)
            y1 = landmarks.landmark[point1].y
            y2 = landmarks.landmark[point2].y
            if self.VERBOSE: print("y1, y2",y1,y2)
            return (y1 + y2) / 2
        # eye pitch
        left_eye_tops = get_average_y(faceLms, 159,145)
        left_eye_top = faceLms.landmark[159].y
        left_eye_bottom = faceLms.landmark[145].y
        left_eye_sides = get_average_y(faceLms, 33,133)
        left_eye_top_delta = left_eye_sides - left_eye_top 
        left_eye_bottom_delta = left_eye_bottom - left_eye_sides
        left_pitch = left_eye_bottom_delta - left_eye_top_delta
        if self.VERBOSE: print("left_eye_top[1]",left_eye_top)
        if self.VERBOSE: print("left_eye_bottom[1]",left_eye_bottom)
        if self.VERBOSE: print("left_eye_sides",left_eye_sides)
        if self.VERBOSE: print("left eye top delta",left_eye_top_delta)
        if self.VERBOSE: print("left eye bottom delta",left_eye_bottom_delta)
        if self.VERBOSE: print("left pitch",left_pitch)

        right_eye_tops = get_average_y(faceLms, 386,374)
        right_eye_top = self.get_face_2d_point(faceLms, 386)
        right_eye_bottom = self.get_face_2d_point(faceLms, 374)
        right_eye_sides = get_average_y(faceLms, 362,263)
        right_eye_top_delta = right_eye_top[1] - right_eye_sides
        right_eye_bottom_delta = right_eye_bottom[1] - right_eye_sides
        
        # if self.VERBOSE: print(left_eye_top_delta, left_eye_bottom_delta, right_eye_top_delta, right_eye_bottom_delta)
        left_pitch = (left_eye_tops - left_eye_sides) / self.face_height*100
        right_pitch = (right_eye_tops - right_eye_sides) / self.face_height*100
        average_pitch = (left_pitch + right_pitch) / 2
        # if self.VERBOSE: print(average_pitch)
        return average_pitch
        # left eye left corner 33, rt 133
        # right eye left corner ,  362, rt 263

    def get_dist_btwn_landmarks(self, faceLms, point1, point2, style="new"):
        if self.VERBOSE: print("point1, point2",point1,point2)
        # point 1 is the top point, point 2 is the bottom point
        top_point = self.get_face_2d_point(faceLms,point1)
        bot_point = self.get_face_2d_point(faceLms,point2)
        if self.VERBOSE: print("top_point, bot_point",top_point,bot_point)
        #calculate  gap
        if style == "new":
            gap = top_point[1] - bot_point[1]
        else:
            gap = self.dist(self.point(bot_point), self.point(top_point))
        if self.VERBOSE: print("gap",gap)
        # check for face height, and if not exist, then get
        if not hasattr(self, 'face_height'): 
            self.get_faceheight_data(faceLms)
 
        gap_pct = gap/self.face_height*100
        if self.VERBOSE: print(gap_pct)
        return gap_pct
    
    def get_mouth_data(self, faceLms):
        # toplip = self.get_face_2d_point(faceLms,13)
        # botlip = self.get_face_2d_point(faceLms,14)
        # #calculate mouth gap
        # mouth_gap = self.dist(self.point(botlip), self.point(toplip))
 
        # # check for face height, and if not exist, then get
        # if not hasattr(self, 'face_height'): 
        #     self.get_faceheight_data(faceLms)
 
        # mouth_pct = mouth_gap/self.face_height*100
        # # if self.VERBOSE: print(mouth_pct)
        mouth_pct = self.get_dist_btwn_landmarks(faceLms,13,14, style="old")
        return mouth_pct


    def get_faceheight_data(self, faceLms):
        top_2d = self.get_face_2d_point(faceLms,10)
        bottom_2d = self.get_face_2d_point(faceLms,152)
        self.ptop = (int(top_2d[0]), int(top_2d[1]))
        self.pbot = (int(bottom_2d[0]), int(bottom_2d[1]))
        # height = int(pbot[1]-ptop[1])
        self.face_height = self.dist(self.point(self.pbot), self.point(self.ptop))

        # return ptop, pbot, face_height

    def get_crop_data_simple(self, faceLms):
        
        #it would prob be better to do this with a dict and a loop
        nose_2d = self.get_face_2d_point(faceLms,1)
        # if self.VERBOSE: print("self.sinY: ",self.sinY)
        #set main points for drawing/cropping
        #p1 is tip of nose
        p1 = (int(nose_2d[0]), int(nose_2d[1]))


        toobig = False
        if p1[1]>(self.face_height*1) and (self.h-p1[1])>(self.face_height*1):
            if p1[0]>(self.face_height*1) and (self.w-p1[0])>(self.face_height*1):
                self.crop_multiplier = 1
            else:
                if self.VERBOSE: print('face too wiiiiiiiide')
                self.crop_multiplier = .25
                toobig=True

        else:
            self.crop_multiplier = .25
            if self.VERBOSE: print('face too biiiiigggggg')
            toobig=True

        # if self.VERBOSE: print(crop_multiplier)
        self.h - p1[1]
        top_overlap = p1[1]-self.face_height

        #set crop
        # crop_multiplier = 1
        leftcrop = int(p1[0]-(self.face_height*self.crop_multiplier))
        rightcrop = int(p1[0]+(self.face_height*self.crop_multiplier))
        topcrop = int(p1[1]-(self.face_height*self.crop_multiplier))
        botcrop = int(p1[1]+(self.face_height*self.crop_multiplier))
        self.simple_crop = [topcrop, rightcrop, botcrop, leftcrop]


    def get_crop_data(self, faceLms, sinY, export_size=2500):
        #it would prob be better to do this with a dict and a loop
        nose_2d = self.get_face_2d_point(faceLms,1)
        # if self.VERBOSE: print("sinY: ",sinY)

        #cludge to get the new script to not move for neck
        sinY = 0

        #set main points for drawing/cropping
        #p1 is tip of nose
        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        # if self.VERBOSE: print(crop_multiplier)
        # self.h - p1[1]
        top_overlap = p1[1]-self.face_height
        neck_offset = sinY*int(self.face_height)
        #neck is point to crop image off of
        neck = (p1[0]+neck_offset,p1[1])
        # if self.VERBOSE: print("nose ",p1[0])
        # if self.VERBOSE: print("neck ",neck[0])
        self.crop =[0,0]
        # determine crop shape/ratio
        # crops = [.75,1,1.5,2,2.5,3]

        #cludge to get the new script to not mess with cropping
        crops = [1]

        toobig = False
        balance = 1
        for ratio in crops:
            if neck[0]>(self.face_height*ratio) and (self.w-neck[0])>(self.face_height*ratio):
                self.crop[0]=ratio
                maxcrop = True
            # this isn't totally working. I'm trying to add space below, but it isn't working.
            # it is still centering on neck and isn't passing the values to the actual crop 
            if neck[1]>(self.face_height*crops[0]+self.face_height*(ratio-crops[0])) and (self.h-neck[1])>(self.face_height*crops[0]):
                self.crop[1]=ratio
            try:
                balance = self.crop[0]/self.crop[1]
                if self.VERBOSE: print(balance)
                #this might not be set right. seems to be weird with < or >
                if .6 > balance > 1.5:
                    balance = 1
                    continue
            except:
                toobig = True
                balance = 1

        if self.crop[0] == 0 or self.crop[1] == 0:
            toobig = True
        if self.VERBOSE: print("toobig and crop ",toobig,self.crop)

        #set crop
        # crop_multiplier = 1
        leftcrop = int(neck[0]-(self.face_height*self.crop[0]))
        rightcrop = int(neck[0]+(self.face_height*self.crop[0]))
        topcrop = int(neck[1]-(self.face_height*self.crop[1]))
        botcrop = int(neck[1]+(self.face_height*self.crop[1]))
        self.crop_points = [topcrop, rightcrop, botcrop, leftcrop]

        #set padding
        # figures out how far each dimensions is from nose
        # subtracts edge_to_nose from export_size/2
        # crop_multiplier = 1
        # if self.VERBOSE: print("neck, faceheight, crop")
        # if self.VERBOSE: print(neck[0])
        # if self.VERBOSE: print(self.face_height)
        # if self.VERBOSE: print(self.crop[0])
        leftpadding = int(export_size/2 - int(neck[0]))
        rightpadding = int(export_size/2 - (self.w - int(neck[0])))
        toppadding = int(export_size/2 - int(neck[1]))
        botpadding = int(export_size/2 - (self.h - int(neck[1])))
        self.padding_points = [toppadding, rightpadding, botpadding, leftpadding]


    def draw_nose(self,image):
        #it would prob be better to do this with a dict and a loop
        # nose_2d = self.get_face_2d_point(faceLms,1)
        nose_2d = self.face_2d[0]

        #set main points for drawing/cropping
        #p1 is tip of nose
        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        cv2.circle(image,(int(p1[0]),int(p1[1])),4,(255,0,0),-1)


    def draw_crop_frame(self,image):
  
      # cv2.rectangle(image, (leftcrop,topcrop), (rightcrop,botcrop), (255,0,0), 2)
        cv2.rectangle(image, (self.crop_points[3],self.crop_points[0]), (self.crop_points[1],self.crop_points[2]), (255,0,0), 2)

        pass

    def add_margin(self, src, padding_points):
        top, right, bottom, left = padding_points   
        borderType = cv2.BORDER_CONSTANT
        BLUE = [255,255,255]
        if self.VERBOSE: print(top)
        if self.VERBOSE: print(type(top))
        # width, height = pil_img.size
        # new_width = width + right + left
        # new_height = height + top + bottom
        
        padded_image = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value = BLUE)

        # result = Image.new(pil_img.mode, (new_width, new_height), color)
        # result.paste(pil_img, (left, top))
        return padded_image

    def crop_image(self,cropped_image, faceLms, sinY):


        #I'm not sure the diff between nose_2d and p1. May be redundant.
        #it would prob be better to do this with a dict and a loop
        nose_2d = self.get_face_2d_point(faceLms,1)

        # check for crop, and if not exist, then get
        if not hasattr(self, 'crop'): 
            # self.get_crop_data_simple(faceLms)

            # this is the in progress neck rotation stuff
            self.get_crop_data(faceLms, sinY)

        
        # if self.VERBOSE: print (self.padding_points)
        #set main points for drawing/cropping
        #p1 is tip of nose
        p1 = (int(nose_2d[0]), int(nose_2d[1]))

        # if self.VERBOSE: print(crop_multiplier)
        self.h - p1[1]
        top_overlap = p1[1]-self.face_height

        #adding this in to padd image
        try:
            padded_image = self.add_margin(cropped_image, self.padding_points)
        except:
            padded_image = False
        basesize = 750
        newsize = (basesize*self.crop[0],basesize*self.crop[1])
        resize = np.round(newsize[0]/(self.face_height*2.5), 3)

        #moved this back up so it would NOT     draw map on both sets of images
        try:
            # crop[0] is top, and clockwise from there. Right is 1, Bottom is 2, Left is 3. 
            cropped_image = cv2.resize(cropped_image[self.crop_points[0]:self.crop_points[2], self.crop_points[3]:self.crop_points[1]], (newsize), interpolation= cv2.INTER_LINEAR)
        except:
            cropped_image = None
            if self.VERBOSE: print("not cropped_image loop")

            if self.VERBOSE: print(self.h, self.w)
               
        return padded_image, cropped_image, resize


##### HAND LANDMARKS #####


    def calculate_hand_landmarks(self,image):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # this version allows specific model to be loaded, so more flexible, but not going to use
        base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        detector = vision.HandLandmarker.create_from_options(options)

        # STEP 4: Detect hand landmarks from the input image.
        detection_result = detector.detect(image)

        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        # annotated_image = np.copy(image)
        return detection_result

    def display_landmarks(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)
        
        height, width, _ = annotated_image.shape  # Get image dimensions

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)): 
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            score = handedness[0].score
            hand_type = handedness[0].category_name
            if self.VERBOSE: print(f"Hand {idx}:")
            if self.VERBOSE: print(f"    Type: {hand_type}")
            if self.VERBOSE: print(f"    Score: {score}")

            # Pointer finger tip is landmark 8
            pointer_finger_tip = hand_landmarks[8]

            # Convert normalized coordinates to pixel values
            pointer_finger_x = int(pointer_finger_tip.x * width)
            pointer_finger_y = int(pointer_finger_tip.y * height)

            # Draw the point on the image
            cv2.circle(annotated_image, (pointer_finger_x, pointer_finger_y), 25, (0, 255, 0), -1)
            if self.VERBOSE: print(f"  >>>  Pointer finger (2D) location: x = {pointer_finger_x}, y = {pointer_finger_y}")

            # Display the image with the annotation
            cv2.imshow("marked image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated_image


    def store_hand_landmarks(self, image_id, hands_data, mongo_hand_collection):
        hand_data_dict = {}

        for hand_data in hands_data:
            # Prepare the data for each hand
            hand_landmarks_data = {
                "image_landmarks": hand_data["image_landmarks"],
                "world_landmarks": hand_data["world_landmarks"],
                "confidence_score": hand_data["confidence_score"]
            }

            # Store the hand data based on handedness
            if hand_data["handedness"] == "Right":
                hand_data_dict["right_hand"] = hand_landmarks_data
            elif hand_data["handedness"] == "Left":
                hand_data_dict["left_hand"] = hand_landmarks_data

        # Store both left and right hand data in the same MongoDB document
        # if self.VERBOSE: print(f"Storing data for image_id: {image_id}")
        mongo_hand_collection.update_one(
            {"image_id": image_id},
            {"$set": hand_data_dict},
            upsert=True  # Insert if doesn't exist or update if it does
        )
        # if self.VERBOSE: print(f"----------- >>>>>>>>   MongoDB hand data updated for image_id: {image_id}")


    def extract_hand_landmarks(self, detection_result):
        hands_data = []

        # Loop through each hand detected and extract the necessary details
        if detection_result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_result.multi_hand_landmarks):
                # Extract landmarks in image coordinates (x, y, z)
                image_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                # Extract landmarks in world coordinates (for 3D space)
                world_landmarks = [(lm.x, lm.y, lm.z) for lm in detection_result.multi_hand_world_landmarks[idx].landmark]

                # Extract confidence score and handedness (left or right hand)
                handedness = detection_result.multi_handedness[idx].classification[0]
                confidence_score = handedness.score
                hand_label = handedness.label  # "Left" or "Right"

                # Create a dictionary to store all information for this hand
                hand_data = {
                    "image_landmarks": image_landmarks,
                    "world_landmarks": world_landmarks,
                    "handedness": hand_label,
                    "confidence_score": confidence_score
                }

                # Append the hand data to the list
                hands_data.append(hand_data)

        return hands_data


    def draw_landmarks_on_image(self,rgb_image, detection_result):
        MARGIN = 10    # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
        from mediapipe import solutions
        from mediapipe.framework.formats import landmark_pb2
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image
