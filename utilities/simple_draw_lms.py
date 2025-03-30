import cv2
import mediapipe as mp
import os

def process_image(image_path, mode="face", landmark_thickness=1, connection_thickness=1, nose_circle_radius=5):
    """
    Detects faces, bodies, or hands in an image, calculates landmarks, draws them, and saves the modified image.

    Args:
        image_path (str): The path to the input image.
        mode (str): "face", "body", or "hands". Determines which landmarks to detect and draw.
        landmark_thickness (int): Thickness of the landmark points.
        connection_thickness (int): Thickness of the lines connecting landmarks.
        nose_circle_radius (int): Radius of the red circle drawn on the nose landmark (face mode only).
    """

    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=connection_thickness, circle_radius=landmark_thickness)

    if mode == "face":
        mp_landmarks = mp.solutions.face_mesh
        connections = [
            mp_landmarks.FACEMESH_TESSELATION,
            mp_landmarks.FACEMESH_CONTOURS,
            mp_landmarks.FACEMESH_IRISES,
        ]
        landmark_processor = mp_landmarks.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    elif mode == "body":
        mp_landmarks = mp.solutions.pose
        connections = [mp_landmarks.POSE_CONNECTIONS]
        landmark_processor = mp_landmarks.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )

    elif mode == "hands":
        mp_landmarks = mp.solutions.hands
        connections = [mp_landmarks.HAND_CONNECTIONS]
        landmark_processor = mp_landmarks.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
    else:
        print("Invalid mode. Choose 'face', 'body', or 'hands'.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    results = landmark_processor.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if mode == "face":
        if not results.multi_face_landmarks:
            print("No face detected")
            return
        landmark_list = results.multi_face_landmarks
    elif mode == "body":
        if not results.pose_landmarks:
            print("No body detected")
            return
        landmark_list = results.pose_landmarks
    elif mode == "hands":
        if not results.multi_hand_landmarks:
            print("No hands detected")
            return
        landmark_list = results.multi_hand_landmarks

    annotated_image = image.copy()

    if landmark_list:
        if mode == "face":
            h, w = annotated_image.shape[:2]
            for landmarks in landmark_list:
                cx, cy = int(landmarks.landmark[1].x * w), int(landmarks.landmark[1].y * h)
                cv2.circle(annotated_image, (cx, cy), nose_circle_radius, (0, 0, 255), -1) # Draw red circle on nose
                for connection in connections:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=annotated_image,
                        landmark_list=landmarks,
                        connections=connection,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_spec,
                    )
        else:
            for landmarks in [landmark_list] if mode == "body" else landmark_list:
                for connection in connections:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=annotated_image,
                        landmark_list=landmarks,
                        connections=connection,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_spec,
                    )

    base_name, extension = os.path.splitext(image_path)
    output_path = f"{base_name}_{mode}_lms{extension}"
    cv2.imwrite(output_path, annotated_image)
    print(f"{mode.capitalize()} landmarks drawn and saved to {output_path}")

    if mode == "face":
        landmark_processor.close()
    elif mode == "body":
        landmark_processor.close()
    elif mode == "hands":
        landmark_processor.close()

if __name__ == "__main__":
    image_path = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/hand_tests/19882612.jpg"  # Replace with your image path
    mode = "face"  # Choose "face", "body", or "hands"
    landmark_thickness = 20
    connection_thickness = 2
    nose_circle_radius = 5

    if os.path.exists(image_path):
        process_image(image_path, mode, landmark_thickness, connection_thickness, nose_circle_radius)
    else:
        print(f"Error: Image not found at {image_path}")