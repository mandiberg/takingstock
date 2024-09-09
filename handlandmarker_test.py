from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

MARGIN = 10    # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
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

def get_hand_bboxes(results, image_width, image_height):
    # Compute the bounding box from the landmarks
    # handedness_list = detection_result.handedness
    hand_bboxes = []
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            print("Hand landmarks:", hand_landmarks)
            # handedness = handedness_list[idx]
            # hand_type = handedness[0].category_name
            # print("bbox for Hand type:", hand_type)
            # Initialize bounding box coordinates
            min_x = float('inf')
            max_x = float('-inf')
            min_y = float('inf')
            max_y = float('-inf')
            
            # Iterate over all landmarks for the hand
            for landmark in hand_landmarks:
                # Update bounding box coordinates
                min_x = min(min_x, landmark.x)
                max_x = max(max_x, landmark.x)
                min_y = min(min_y, landmark.y)
                max_y = max(max_y, landmark.y)
            
            # Print the bounding box (normalized coordinates)
            # print(f"Bounding Box: x_min = {min_x}, x_max = {max_x}, y_min = {min_y}, y_max = {max_y}")
            
            # If needed, convert normalized coordinates to pixel values based on image dimensions
            # Assuming you know the width and height of the image
            # image_width = 640  # Example image width
            # image_height = 480  # Example image height
            hand_bbox = {}
            hand_bbox["top"] = int(min_y * image_height)
            hand_bbox["right"]  = int(max_x * image_width)
            hand_bbox["bottom"] = int(max_y * image_height)
            hand_bbox["left"] = int(min_x * image_width)

            # print(f"Bounding Box in pixels: x_min = {x_min_pixel}, x_max = {x_max_pixel}, y_min = {y_min_pixel}, y_max = {y_max_pixel}")
            print(f"Bounding Box in pixels: left = {hand_bbox['left']}, right = {hand_bbox['right']}, top = {hand_bbox['top']}, bottom = {hand_bbox['bottom']}")
            print(hand_bbox)
            hand_bboxes.append(hand_bbox)
    return hand_bboxes

def project_hand_point(point, hand_bbox):
    # Get the bounding box coordinates
    x_min = hand_bbox["left"]
    x_max = hand_bbox["right"]
    y_min = hand_bbox["top"]
    y_max = hand_bbox["bottom"]
    
    # Get the normalized coordinates of the point
    x = point.x
    y = point.y
    
    # Project the point to the image coordinates
    x_pixel = int(x * (x_max - x_min) + x_min)
    y_pixel = int(y * (y_max - y_min) + y_min)

    return x_pixel, y_pixel
    
    return x_pixel, y_pixel
def return_landmarks(rgb_image, detection_result):
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
        print(f"Hand {idx}:")
        print(f"    Type: {hand_type}")
        print(f"    Score: {score}")

        # Pointer finger tip is landmark 8
        pointer_finger_tip = hand_landmarks[8]

        # Convert normalized coordinates to pixel values
        pointer_finger_x = int(pointer_finger_tip.x * width)
        pointer_finger_y = int(pointer_finger_tip.y * height)

        # Draw the point on the image
        cv2.circle(annotated_image, (pointer_finger_x, pointer_finger_y), 25, (0, 255, 0), -1)
        print(f"  >>>  Pointer finger (2D) location: x = {pointer_finger_x}, y = {pointer_finger_y}")

        # Display the image with the annotation
        cv2.imshow("marked image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Draw the hand landmarks.
        # hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        # hand_landmarks_proto.landmark.extend([
        #     landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        # ])
        # solutions.drawing_utils.draw_landmarks(
        #     annotated_image,
        #     hand_landmarks_proto,
        #     solutions.hands.HAND_CONNECTIONS,
        #     solutions.drawing_styles.get_default_hand_landmarks_style(),
        #     solutions.drawing_styles.get_default_hand_connections_style()
        # )

        # Get the top left corner of the detected hand's bounding box.
        # height, width, _ = annotated_image.shape
        # x_coordinates = [landmark.x for landmark in hand_landmarks]
        # y_coordinates = [landmark.y for landmark in hand_landmarks]
        # text_x = int(min(x_coordinates) * width)
        # text_y = int(min(y_coordinates) * height) - MARGIN

        # # Draw handedness (left or right hand) on the image.
        # cv2.putText(annotated_image, f"{handedness[0].category_name}",
        #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


# img = cv2.imread("woman_hands.jpg")

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("woman_hands.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

landmarks = return_landmarks(image.numpy_view(), detection_result)

# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow("marked image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)
cv2.destroyAllWindows()
