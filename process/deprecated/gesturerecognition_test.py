#@markdown We implemented some functions to visualize the gesture recognition results. <br/> Run the following cell to activate the functions.
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import os

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


from mp_db_io import DataIO
IS_SSD = True
io = DataIO(IS_SSD)

def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)


def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.
    images = [image.numpy_view() for image in images]
    gestures = [top_gesture for (top_gesture, _) in results]
    multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    # Display gestures and hand landmarks.
    for i, (image, gestures) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
        title = f"{gestures.category_name} ({gestures.score:.2f})"
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
        annotated_image = image.copy()

        for hand_landmarks in multi_hand_landmarks_list[i]:
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
          ])

          mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()


import urllib

IMAGE_FILENAMES = ['5443657-isolated-shot-of-a-brunette-with-her-eyes-shut.jpg', 
                    '5585842-pretty-young-female-pointing-away-isolated-on-white-background.jpg',
                    '5928710-happy-businessman-gesturing-ok.jpg', 
                    '5932691-bright-picture-of-young-woman-making-stop-gesture.jpg',
                    '11900621-displeased-businessman-holds-both-hands-up-to-show-a-stop-signal.jpg',
                    '13074998-young-woman-practicing-yoga-isolated-on-a-white-background.jpg',
                    '13360857-businesswoman-in-shirt-and-tie-smiling.jpg',
                    '13742631-happy-girl-and-soap-bubbles-on-the-black-background.jpg',
                    '13826042-portrait-of-a-positive-business-man-excited-on-black-suit-on-isolated-background.jpg',
                    '14840309-beautiful-and-happy-young-woman-standing-isolated-on-white-background.jpg',
                    '14873940-attractive-young-brunette-businesswoman-shows-the-sign-thumbs-up-isolated-against-white-background.jpg',
                    '16829532-muscular-young-woman-in-yoga-pose-wearing-sports-outfit-on-white-background.jpg',
                    '17056476-happy-southeast-asian-chinese-male-in-cheongsam-hands-holding-two-red-packets-ang-pow-isolated-on.jpg',
                    '17165971-pretty-young-woman-asks-for-calm.jpg',
                    '17501222-surprised-latin-child-isolated-on-white-background.jpg',
                    '23880518-businessman-hand-pushing-screen.jpg'
                    ]

# IMAGE_FILENAMES = ['174657738-pretty-young-woman-doing-yoga-in-the-park-woman-doing-exercise-in-the-garden-yoga-in-the-grass.jpg']

# for name in IMAGE_FILENAMES:
#     url = f'https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{name}'
#     urllib.request.urlretrieve(url, name)



import cv2

import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow("images", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


folder = "images_123rf/0/00"

# # Preview the images.
# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():

# for image_file_name in IMAGE_FILENAMES:
#     image_path = os.path.join(io.ROOT, folder, image_file_name)

#     print(image_path)
#     resize_and_show(cv2.imread(image_path))


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

images = []
results = []

for image_file_name in IMAGE_FILENAMES:
    image_path = os.path.join(io.ROOT, folder, image_file_name)
    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(image_path)

    # STEP 4: Recognize gestures in the input image.
    recognition_result = recognizer.recognize(image)

    # STEP 5: Process the result. In this case, visualize it.
    images.append(image)
    top_gesture = recognition_result.gestures[0][0]
    hand_landmarks = recognition_result.hand_landmarks
    results.append((top_gesture, hand_landmarks))

display_batch_of_images_with_gestures_and_hand_landmarks(images, results)