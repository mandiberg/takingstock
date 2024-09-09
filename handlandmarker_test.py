from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import pymongo
from pymongo.errors import DuplicateKeyError

from mp_pose_est import SelectPose

import numpy as np
import cv2
import pickle

detection_result = None

# mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
# mongo_db = mongo_client["stock"]
# # mongo_collection = mongo_db["encodings"]
# mongo_hand_collection = mongo_db["hand_landmarks"]


# img = cv2.imread("woman_hands.jpg")
image_path = "woman_hands.jpg"
image_id = 1

# calculate_hand_landmarks() wants an mp image object not cv2
image = mp.Image.create_from_file(image_path)

placeholder_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
pose = SelectPose(placeholder_image)

detection_result = pose.calculate_hand_landmarks(image)


# store detection result in mongo_hand_collection
# if detection_result:
#     mongo_hand_collection.update_one(
#         {"image_id": image_id},
#         {"$set": {"nlms": pickle.dumps(detection_result)}},
#         upsert=True
#     )
#     print("----------- >>>>>>>>   mongo hand_landmarks updated:", image_id)


annotated_image = pose.display_landmarks(image.numpy_view(), detection_result)

# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = pose.draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow("marked image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)
cv2.destroyAllWindows()
