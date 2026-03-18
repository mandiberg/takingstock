'''
This script draws hand, body, and detection landmarks on the original images, and saves them to a new folder for review.
'''

import os
import gc
import pickle
import cv2
import pymongo
from pathlib import Path
from sqlalchemy import create_engine, MetaData, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import sys
sys.path.insert(1, '/Users/michael.mandiberg/Documents/GitHub/takingstock/')
from my_declarative_base import Encodings, Base, Detections, Images
from mp_db_io import DataIO
from mp_sort_pose import SortPose

IS_SSD = False
VERBOSE = True
LIMIT = 10000
SegmentHelper_name = "SegmentHelper_mar2026_hand_body_lms_testing"

io = DataIO(IS_SSD)
db = io.db
IMPORT_DIR = "/Users/michael.mandiberg/Documents/projects-active/facemap_production/debug_bodies"
EXPORT_DIR = os.path.join(os.path.dirname(IMPORT_DIR), "draw_norm_lms")  # Directory to save BSON files
print(f"Export directory: {EXPORT_DIR}")

# init SortPose to unify with existing code
image_edge_multiplier = [1.5,1.5,2,1.5] # bigger portrait
image_edge_multiplier_sm = [1.2, 1.2, 1.6, 1.2] # standard portrait
face_height_output = 500
motion = {"side_to_side": False, "forward_smile": True, "laugh": False, "forward_nosmile": False, "static_pose": False, "simple": False}
EXPAND = False
ONE_SHOT = False # take all files, based off the very first sort order.
JUMP_SHOT = False # jump to random file if can't find a run

sort = SortPose(config={'motion': motion, 'face_height_output': face_height_output, 'image_edge_multiplier': image_edge_multiplier_sm, 'EXPAND': EXPAND, 'ONE_SHOT': ONE_SHOT, 'JUMP_SHOT': JUMP_SHOT, 'HSV_CONTROL': None, 'VERBOSE': VERBOSE, 'INPAINT': False, 'SORT_TYPE': 'planar_hands', 'OBJ_CLS_ID': 0})


def init_session():
    global engine, Session, session
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
    Session = sessionmaker(bind=engine)
    session = Session()

def close_session():
    session.close()
    engine.dispose()

def init_mongo():
    global mongo_client, mongo_db, mongo_collection, body_normed_collection, body_world_collection, mongo_hand_collection
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    mongo_db = mongo_client["stock"]
    mongo_collection = mongo_db["encodings"]
    body_normed_collection = mongo_db["body_landmarks_norm"]
    body_world_collection = mongo_db["body_world_landmarks"]
    mongo_hand_collection = mongo_db["hand_landmarks"]

def close_mongo():
    mongo_client.close()

def ensure_export_dir():
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)

def build_export_path(image_id, imagename):
    src_name = Path(str(imagename)).name
    stem = Path(src_name).stem if src_name else str(image_id)
    safe_stem = stem.replace("/", "_").replace("\\", "_")
    return os.path.join(EXPORT_DIR, f"{image_id}_{safe_stem}.jpg")

def query_mysql():
    query = text(f"""
        SELECT 
            s.image_id,
            s.site_name_id,
            s.imagename,
            s.bbox,
            i.h,
            i.w,
            e.encoding_id
        FROM SegmentBig_isface s
        INNER JOIN Images i ON s.image_id = i.image_id
        INNER JOIN Encodings e ON s.image_id = e.image_id
        INNER JOIN {SegmentHelper_name} sh ON s.image_id = sh.image_id
        WHERE e.is_dupe_of IS NULL
        AND e.mongo_hand_landmarks_norm = 1
        AND e.mongo_body_landmarks_norm = 1
        ORDER BY s.image_id
        LIMIT {LIMIT}
    """)
    result = session.execute(query)
    return result.fetchall()

def process_face_doc(encodings_doc):
    sort.nose_x, sort.nose_y, sort.face_height = None, None, None
    if encodings_doc and "face_landmarks" in encodings_doc:
        sort.faceLms = pickle.loads(encodings_doc["face_landmarks"])
        if sort.faceLms and len(sort.faceLms.landmark) > 0:
            sort.nose_2d = sort.get_face_2d_point(1)
            # sort.nose_x = sort.faceLms.landmark[0].x * sort.w
            # sort.nose_y = sort.faceLms.landmark[0].y * sort.h
            sort.nose_x, sort.nose_y = sort.nose_2d
            # set the face height
            sort.get_faceheight_data() # stores it in sort.face_height

def draw_baseline_info(image):
    # draw a red dot at the nose position for reference
    if sort.nose_x is not None and sort.nose_y is not None:
        cv2.circle(image, (int(sort.nose_x), int(sort.nose_y)), 5, (0, 255, 255), -1)
    # draw a blue rectangle representing the face height for reference
    if sort.nose_x is not None and sort.nose_y is not None and sort.face_height is not None:
        top_left = (int(sort.nose_x - sort.face_height/2), int(sort.nose_y - sort.face_height/2))
        bottom_right = (int(sort.nose_x + sort.face_height/2), int(sort.nose_y + sort.face_height/2))
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)

def get_class_names(session):
    # query YoloClasses table to get class_id and class_name
    result = session.execute(text("SELECT class_id, class_name FROM YoloClasses"))
    class_names = result.fetchall()
    return {class_id: class_name for class_id, class_name in class_names}


def draw_bbox(image, norm_data, color_map, hand):
    if norm_data and isinstance(norm_data, list):
        x_coords = [point[0] for point in norm_data]
        y_coords = [point[1] for point in norm_data]
        if x_coords and y_coords:
            min_x, max_x = int(min(x_coords)), int(max(x_coords))
            min_y, max_y = int(min(y_coords)), int(max(y_coords))
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color_map[hand], 3)
        # draw hand label
        if "conf" in hand:
            conf = hand.split("conf_0")[1]
            print(f"Drawing bbox for {hand} with confidence {conf}")
            x_width = max_x - min_x
            x_location = int(min_x + x_width - (x_width * float(conf)))
        else:
            x_location = min_x
        cv2.putText(image, hand, (x_location, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_map[hand], 2)

def denornm_lms(lms, image):
    global class_names
    def denorm_landmark(x,y,z, color=(0,255,0), draw = True):
        # print(f"Original normalized landmark: ({x}, {y}, {z})")
        x = sort.nose_x + (x * sort.face_height)  # scale by face height and position relative to nose
        y = sort.nose_y + (y * sort.face_height)  # scale by face height and position relative to nose
        if z is not None: z = sort.face_height
        # draw the landmark on the image for visualization
        if draw:
            cv2.circle(image, (int(x), int(y)), 5, color, -1)
        # draw line from nose to landmark for visualization
        # print(f"denormed landmark: ({x}, {y}, {z})")
        # if sort.nose_x is not None and sort.nose_y is not None:
        #     cv2.line(image, (int(sort.nose_x), int(sort.nose_y)), (int(x), int(y)), color, 1)
        return x,y,z

    print("Denorming landmarks...")
    print(type(lms))
    if hasattr(lms, 'landmark'):
        for lm in lms.landmark:
            x, y, z = denorm_landmark(lm.x, lm.y, getattr(lm, 'z', None), color=(0,255,0))
        print("body Denorming complete:" + f"Sample denormed landmark (first landmark): ({lms.landmark[0].x}, {lms.landmark[0].y}, {getattr(lms.landmark[0], 'z', 'N/A')})")
    # handle hand_landmarks which is json object
    elif isinstance(lms, dict):
        hands = ["left_hand", "right_hand"]
        color_map = {"left_hand": (255,0,0), "right_hand": (0,0,255)}
        for hand in hands:
            hand_data = lms.get(hand, None)
            norm_data = hand_data.get("hand_landmarks_norm", []) if hand_data else []
            if norm_data and isinstance(norm_data, list):
                denormed_points = []
                for lm in norm_data:
                    print(f"Processing {hand} landmark list: {lm}")
                    x, y, z = denorm_landmark(lm[0], lm[1], lm[2], color=color_map[hand], draw = False)
                    denormed_points.append((x,y,z))
                draw_bbox(image, denormed_points, color_map, hand)
                print(f"{hand} Denorming complete)")
            else:
                print(f"Warning: No valid 'hand_landmarks_norm' data found for {hand}. Skipping denorming for this hand.")
    elif isinstance(lms, list) and len(lms) == 3:
        class_id, conf, bbox_norm = lms
        bbox_norm = io.unstring_json(bbox_norm) if isinstance(bbox_norm, str) else bbox_norm
        if bbox_norm and isinstance(bbox_norm, dict):
            denormed_points = []
            ul = bbox_norm.get('left', None), bbox_norm.get('top', None)
            ur = bbox_norm.get('right', None), bbox_norm.get('top', None)
            bl = bbox_norm.get('left', None), bbox_norm.get('bottom', None)
            br = bbox_norm.get('right', None), bbox_norm.get('bottom', None)
            norm_points = [ul, ur, bl, br]
            for point in norm_points:
                if None not in point:
                    x, y, z = denorm_landmark(point[0], point[1], None, color=(0,0,255), draw = False)
                    denormed_points.append((x,y,z))
            # for key in ['left', 'top', 'right', 'bottom']:
            #     if key in bbox_norm:
            #         x = sort.nose_x + (bbox_norm[key] * sort.face_height)  # scale by face height and position relative to nose
            #         y = sort.nose_y + (bbox_norm[key] * sort.face_height)  # scale by face height and position relative to nose
            #         denormed_points.append((x,y))
            if len(denormed_points) == 4:
                this_class_name = class_names.get(class_id, f"class_{class_id}")
                det_label=f"{this_class_name} conf_{conf:.2f}"

                draw_bbox(image, denormed_points, color_map={f"{det_label}": (0,0,255)}, hand=det_label)
                print(f"Detection bbox Denorming complete)")

        # this is the case for detections, where lms is actually [class_id, conf, bbox]
        print(f"Received detection data: {lms}. Skipping denorming since this is not landmark data.")
    else:
        print("Warning: Landmark data format not recognized. Expected an object with 'landmark' attribute or a dict with hand landmarks. Skipping denorming.")
        print(f"Received landmark data: {lms}")


def main():
    global class_names
    init_session()
    init_mongo()
    ensure_export_dir()
    class_names = get_class_names(session)

    # get list from segment
    print("Querying database for segment data...")
    results = query_mysql()
    print(f"Retrieved {len(results)} records from database.")
    for idx, row in enumerate(results):
        image_id = row.image_id
        site_name_id = row.site_name_id
        imagename = row.imagename
        sort.h = row.h
        sort.w = row.w
        encoding_id = row.encoding_id
        sort.bbox = io.unstring_json(row.bbox)

        print(f"Processing image {idx+1}/{len(results)}: ID={image_id}, Name={imagename}, Size=({sort.w}x{sort.h})")

        # get the corresponding MongoDB documents for this image_id
        encodings_doc = mongo_collection.find_one({"image_id": image_id})
        body_normed_doc = body_normed_collection.find_one({"image_id": image_id})
        body_world_doc = body_world_collection.find_one({"image_id": image_id})
        hand_doc = mongo_hand_collection.find_one({"image_id": image_id})

        detections = session.query(Detections).filter(Detections.image_id == image_id).all()
        print(f"Found {len(detections)} detection(s) in MySQL for image ID {image_id}.")

        if body_normed_doc or body_world_doc or hand_doc:
            folder_name = io.folder_list[site_name_id]
            print(f"Found MongoDB documents for image ID {image_id} in folder '{folder_name}'. Proceeding with processing.")
            print(IMPORT_DIR, os.path.basename(folder_name), imagename)
            image_path = os.path.join(IMPORT_DIR, os.path.basename(folder_name), imagename)
            print(f"Constructed image path: {image_path}")
            image = cv2.imread(image_path)


            process_face_doc(encodings_doc) # stores in sort.nose_x, sort.nose_y, sort.face_height
            print(f"Extracted face data - Image ID: {image_id} - Nose: ({sort.nose_x}, {sort.nose_y}), Face Height: {sort.face_height}")
            draw_baseline_info(image)
             # stores in sort.body_landmarks_3D
            if body_normed_doc: denornm_lms(pickle.loads(body_normed_doc["nlms"]), image)
            # if body_world_doc: denornm_lms(pickle.loads(body_world_doc["body_world_landmarks"]), image)
            if hand_doc: denornm_lms(hand_doc, image)
            if len(detections) > 0: 
                for det in detections:
                    denornm_lms([det.class_id, det.conf, det.bbox_norm], image)
            cv2.imshow("Original Image", image)
            cv2.waitKey(120)  # Display the image for 100 milliseconds

            save_path = build_export_path(image_id, imagename)
            saved = cv2.imwrite(save_path, image)
            if saved:
                print(f"Saved image with drawn landmarks: {save_path}")
            else:
                print(f"ERROR: cv2.imwrite failed for image ID {image_id} -> {save_path}")
            # Here you would add the code to draw the landmarks on the image and save it to EXPORT_DIR
            # For example, you could use OpenCV to read the original image, draw the landmarks, and save it.
            # This is where you would implement the drawing logic based on the data in the MongoDB documents.
        else:
            print(f"Warning: Missing MongoDB documents for image ID {image_id}. Skipping drawing.")


    close_session()
    close_mongo()
    print("Export complete.")

if __name__ == "__main__":
    main()