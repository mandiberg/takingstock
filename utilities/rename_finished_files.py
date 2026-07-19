import os
from pathlib import Path
import sys
import pandas as pd

from sqlalchemy import Float, create_engine, Column, Integer, Boolean, String
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy.pool import NullPool
from pymediainfo import MediaInfo

ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/takingstock/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)
from mp_db_io import DataIO 
from my_declarative_base import Base, ImagesObjectSignatures, YoloClasses

'''
This script reads a folder of finished files and renames them to a new naming convention.
The finished files have names like:
output_folder__FINISHED_WORK__hsv_bg_2tier_100clustercc1_p119_t2341_h7-13_1782990020.413111_p34_st1_ct8.mp4
output_folder__FINISHED_WORK__looping_june24_itter100clustercc537_p282_t0_1782312201.079829_p34_st1_ct8.mp4
output_folder__FINISHED_WORK__looping_clipboardsclustercc172_p3984_t0_1782848710.082335_p37_st1_ct8.mp4
merged_cluster458_pNone_t63_156.jpg

It will output a new name like:
TakingStock_T{TOPIC}_p{folder_arms_pose}_s{folder_signature}_obj_{obj}_h{folder_hsv}.mp4
'''

FOLDER = "/Volumes/OWC52/_finished_work.mirrorRAID18/_FINISHED_WORK_THEOFFICE/T11_1920"
MP4_ONLY = True 
io = DataIO()
DRY_RUN = False

TOPIC = 11

OBJECT_SIGNATURE_EXPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "utilities",
    "data",
    "ImagesObjectSignatures_ObjectSignatures_202604272237.csv",
)
OBJECT_SIGNATURE_EXPORT_PATH = OBJECT_SIGNATURE_EXPORT_PATH.replace("utilities/utilities", "utilities") # if called from utilities folder, fix path

HSV_DICT = {
    "3-6+22": "Red",
    "3-6+9+22": "Red",
    "7-13": "Yellow",
    "15-21": "Blue",
    "15-19": "Blue",
    "3-22": "Red-Yellow-Blue",
    "0": "Black",
    "0-1": "Black-Grey",
    "2": "White",
    "0-22": "None",
}

# load custom class map from yolo repo:

db = io.db
engine = create_engine(
    f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
    pool_pre_ping=True,
    pool_recycle=600,
    poolclass=NullPool
)
Session = sessionmaker(bind=engine)
session = Session()
ImagesArmsPoses3D = io.create_class_from_reflection(engine, "ImagesObjectSignatures", "ImagesArmsPoses3D")
    # Encodings_Migration = io.create_class_from_reflection(engine, 'encodings', 'encodings_migration')

object_signature_registry = io.load_object_signature_registry(OBJECT_SIGNATURE_EXPORT_PATH)
# print(f"object_signature_registry: {object_signature_registry}")
folder_list = io.get_folders(FOLDER)
# add root folder to folder_list if it is not already in the list
if FOLDER not in folder_list:
    folder_list.append(FOLDER)
print(f"folder_list: {folder_list}")
if len(folder_list) == 0:
    print(f" ❌ WARNING: No folders found in {FOLDER}, so setting folder_list to [FOLDER]")
    folder_list = [FOLDER]

def load_yolo_classes(session):
    yolo_classes = session.query(YoloClasses).all()
    class_map = {yolo_class.class_id: yolo_class.class_name for yolo_class in yolo_classes}
    # capitalize the first letter of each word in class name (including values with multiple words, eg Baseball Bat)
    class_map = {k: v.title() for k, v in class_map.items()}
    return class_map

CLASS_MAP = load_yolo_classes(session)
print(f"CLASS_MAP: {CLASS_MAP}")

def get_modal_cluster_id(main_folder, session):
    modal_cluster_dict = {}
    print(f"Getting modal cluster id for all folders in {main_folder}")
    # this looks in a folder that holds other folders, and does a query per folder
    folders = io.get_folders(main_folder)
    for folder in folders:
        print(f"Checking folder: {folder}")
        folder_path = os.path.join(main_folder, folder)
        if os.path.isdir(folder_path):
            # img_list = io.get_img_list(folder)
            try:
                image_id_list = io.get_existing_image_ids_from_jpgs(folder_path)
            except Exception as e:
                print(f"Error occurred while getting image IDs from {folder_path}: {e}")
                continue
            print(f"image_id_list: {image_id_list}")
            if len(image_id_list) == 0:
                print(f" ❌ WARNING: No image IDs found in {folder_path}, skipping")
                continue
            # use session to query ImagesObjectSignatures for each image's signature, and return the most common signature
            cluster_id_list = []
            for image_id in image_id_list:
                cluster_id = session.query(ImagesArmsPoses3D.cluster_id).filter(ImagesArmsPoses3D.image_id == image_id).first()
                if cluster_id is not None:
                    cluster_id_list.append(cluster_id[0])
            if len(cluster_id_list) == 0:
                return None
            # get the most common signature
            # print(f"cluster_id_list: {cluster_id_list}")
            modal_cluster_id = max(set(cluster_id_list), key=cluster_id_list.count)
            print(f"modal_cluster_id for {folder}: {modal_cluster_id}")
            modal_cluster_dict[folder] = modal_cluster_id
    print(f"modal_cluster_dict: {modal_cluster_dict}")

    return modal_cluster_dict

def format_title(topic, folder_arms_pose, folder_hands_gesture, folder_signature, obj_str, folder_hsv):
    title = ""

    obj_id_string = obj_name_string = hsv_name = hsv_value  = None
    # format obj and hsv strings, but not add yet
    if obj_str is not None:
        if "-" in obj_str:
            obj_list = obj_str.split("-")
            # create a string like "Object obj1, obj2, and obj3"
            obj_id_list = [f"{obj_id} " for obj_id in obj_list]
            # join the list with commas and "and"
            obj_id_string = f"Objects {', '.join(obj_id_list[:-1])}and {obj_id_list[-1]}"
            obj_name_string = " and ".join([f"{CLASS_MAP.get(int(obj_id), 'Unknown')}" for obj_id in obj_list]).replace("_", " ")
            
            # obj_id_and_name_list = [f"{obj_id} ({CLASS_MAP.get(int(obj_id), 'Unknown')})" for obj_id in obj_list]
            # obj_str = ", ".join(obj_id_and_name_list[:-1]) + f" and {obj_id_and_name_list[-1]}"
            # obj_id += f"Objects {obj_str}, "
        else: 
            obj_id = f"{obj_str}"
            obj_name = f"{CLASS_MAP.get(int(obj_str), 'Unknown')}"
            obj_id_string = f"Object {obj_id}, "
            obj_name_string = f"{obj_name}"
    
    if folder_hsv is not None:
        if folder_hsv == 1 or folder_hsv == "1":
            folder_hsv = "0-1"
        print(f"folder_hsv: {folder_hsv}")
        hsv_name = HSV_DICT.get(folder_hsv, None)
        if hsv_name != "None" and hsv_name is not None:
            hsv_value = f"HSV {folder_hsv} "
            # hsv_name += f"{hsv_name}
    else: 
        hsv_name = None
        hsv_value = None
    print(f"format_title: topic: {topic}, folder_arms_pose: {folder_arms_pose}, folder_hands_gesture: {folder_hands_gesture}, folder_signature: {folder_signature}, obj_id_string: {obj_id_string}, obj_name_string: {obj_name_string}, hsv_name: {hsv_name}, hsv_value: {hsv_value}")
    if obj_name_string is not None and hsv_name is not None:
        title += f"{obj_name_string}, {hsv_name} "
    elif obj_name_string is not None:
        title += f"{obj_name_string} "
    elif hsv_name is not None and hsv_name != "None":
        title += f"{hsv_name} "
    title += "("
    if topic is not None:
        title += f"Topic {topic}, "
    if folder_arms_pose is not None:
        title += f"Pose {folder_arms_pose}, "
    if folder_hands_gesture is not None:
        title += f"Gesture {folder_hands_gesture}, "
    if obj_id_string is not None:
        title += f"{obj_id_string}, "
    if hsv_value is not None:
        title += f"{hsv_value}, "
    # if folder_signature is not None:
    #     title += f"Signature {folder_signature}, "
    # remove trailing comma and space
    title = title.rstrip(", ").replace(", ,", ",")
    title += ")"


    # chomp any trailing comma and space
    title = title.rstrip(", ")
    print(f"format_title: title: {title}")
    return title

# go get modal cluster id for everything, just in case
folder_modal_signatures = get_modal_cluster_id(FOLDER, session)
print(f"modal_signature: {folder_modal_signatures}")

# construct df for TOPIC}_p{folder_arms_pose}_g{folder_hands_gesture}_s{folder_signature}_obj{obj_str}_h{folder_hsv
df = pd.DataFrame(columns=["new_name", "title", "folder", "img", "folder_arms_pose", "folder_hands_gesture", "folder_signature", "folder_hsv", "obj_str", ])
for folder in folder_list:
    # looks in each folder, and acts on the mp4 files in that folder
    img_list = io.get_img_list(folder, force_ls=True, sort=True, walk=False)
    print(f"Processing folder: {folder}, found {len(img_list)} images")



    # print(f"img_list: {img_list}")
    for img in img_list:
        original_suffix = img.split("_")[-1]
        if "." in original_suffix:
            original_suffix = original_suffix.split(".")[-1]
        topic_id = None
        folder_arms_pose = folder_hands_gesture = folder_signature = folder_hsv = obj_str = frame_count = None
        # img == the mp4 file
        if MP4_ONLY and not "mp4" in img: continue
        print(f" >> this is the file we are going to act on: {img}")
        # get the file dimensions
        # use pymediainfo to get the dimensions of the image/video
        media_info = MediaInfo.parse(os.path.join(folder, img))
        media_width = media_height = media_length = None
        if "mp4" in img:
            # Extract dimensions from the video track
            for track in media_info.tracks:
                if track.track_type == 'Video':
                    media_width = track.width
                    media_height = track.height
                    media_length = track.duration
        elif "jpg" in img or "jpeg" in img:
            # Check if an image track exists and extract dimensions
            if media_info.image_tracks:
                image_track = media_info.image_tracks[0]
                print(f"Dimensions: {image_track.width}x{image_track.height}")
                media_width = image_track.width
                media_height = image_track.height
        print(f"media_width: {media_width}, media_height: {media_height}, media_length: {media_length}")

        

        folder_arms_pose, folder_hands_gesture, folder_signature, folder_hsv, topic_id, frame_count = io.extract_fusion_cluster(img)
        if topic_id is None:
            topic_id = TOPIC
        if frame_count is not None:
            print(f"Found frame_count: {frame_count} in filename {img}")
            frame_token = f"_fr{frame_count}"
        if folder_arms_pose == -1:
            # find the right key from the folder_modal_signatures dict based on folder_signature and folder_hsv
            for key, value in folder_modal_signatures.items():
                if str(folder_signature) in key and folder_hsv is None:
                    folder_arms_pose = value
                    print(f"folder_arms_pose not found in filename, using modal signature from folder: {folder_arms_pose}")
                    break
                elif str(folder_signature) in key and folder_hsv is not None:
                    print(f" ✖️ key: {key}, value: {value}")
                    print(f"folder_signature: {folder_signature}, folder_hsv: {folder_hsv}")
                    print(type(folder_hsv))

        print(f"extract_fusion_cluster DONE: folder_arms_pose: {folder_arms_pose}, folder_hands_gesture: {folder_hands_gesture}, folder_signature: {folder_signature}, folder_hsv: {folder_hsv}, topic_id: {topic_id}, frame_count: {frame_count}")
        obj = object_signature_registry.get(io.normalize_cluster_token(folder_signature), None)
        # print(type(obj))
        
        # (folder_arms_pose is not None or folder_arms_pose is not "None") and
        if (obj == "None" or obj is None):
            if folder_signature is None:
                print(f" X Maybe Concern: no object, but also no folder_signature, so probably a Body3D still image. Proceeding")
            else:
            # print(f" cluster info is folder_arms_pose, folder_hands_gesture, folder_signature, obj_str, folder_hsv: {folder_arms_pose}, {folder_hands_gesture}, {folder_signature}, {obj}, {folder_hsv}")
                print(f" ❌ WARNING: object {folder_signature} not found in object_signature_registry, skipping")
                continue
        else: 
            obj_str = obj.replace(", ", "-").replace("[", "").replace("]", "")
        # print(f"obj: {obj}, obj_str: {obj_str}")

        new_name = f"TakingStock_T{topic_id}_p{folder_arms_pose}_g{folder_hands_gesture}_s{folder_signature}_obj{obj_str}_h{folder_hsv}{frame_token if frame_count is not None else ''}.{original_suffix}"

        title = format_title(topic_id, folder_arms_pose, folder_hands_gesture, folder_signature, obj_str, folder_hsv)

        this_dict = {
            "new_name": new_name,
            "title": title,
            "folder": folder,
            "img": img,
            "folder_arms_pose": folder_arms_pose,
            "folder_hands_gesture": folder_hands_gesture,
            "folder_signature": folder_signature,
            "folder_hsv": folder_hsv,
            "obj_str": obj_str,
            "media_width": media_width,
            "media_height": media_height,
            "media_length": media_length,
            "frame_count": frame_count,
            "long_video_name": None
        }

        df = pd.concat([df, pd.DataFrame([this_dict])], ignore_index=True)


        if DRY_RUN:
            print(f"Dry run: would rename {img} to {new_name} in folder {folder}")
            continue
        # rename the file
        try:
            os.rename(os.path.join(folder, img), os.path.join(folder, new_name))
            print(f"Renamed {img} to {new_name} in folder {folder}")
        except OSError as e:
            print(f"Error occurred while renaming {img}: {e}")
        # print(f"new_name: {new_name}")
        # print(f"title: {title}")


# if two rows have the same title, and one is frame_count 100 and the other is frame_count 600
# assign the long_video_name of the 600 frame_count row to the new_name of the 100 frame_count row
# and drop the 600 frame_count row from the df
for existing_row in df.itertuples():
    title = existing_row.title
    frame_count = existing_row.frame_count
    if frame_count is not None and frame_count == 100:
        # find the row with the same title and frame_count 600
        long_video_row = df[(df["title"] == title) & (df["frame_count"] == 600)]
        if len(long_video_row) > 0:
            long_video_name = long_video_row.iloc[0]["new_name"]
            df.loc[df["title"] == title, "long_video_name"] = long_video_name
            # drop the long video row from the df
            df = df.drop(long_video_row.index)

# sort df on title
df = df.sort_values(by=["title"]).reset_index(drop=True)
print(f"df: {df}")

df.to_csv(os.path.join(FOLDER, "renamed_files.csv"), index=False)