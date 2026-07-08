import os
from pathlib import Path
import sys
import pandas as pd

from sqlalchemy import Float, create_engine, Column, Integer, Boolean, String
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy.pool import NullPool

ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/facemap/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)
from mp_db_io import DataIO
import importlib.util


'''
This script reads a folder of finished files and renames them to a new naming convention.
The finished files have names like:
output_folder__FINISHED_WORK__hsv_bg_2tier_100clustercc1_p119_t2341_h7-13_1782990020.413111_p34_st1_ct8.mp4
output_folder__FINISHED_WORK__looping_june24_itter100clustercc537_p282_t0_1782312201.079829_p34_st1_ct8.mp4
output_folder__FINISHED_WORK__looping_clipboardsclustercc172_p3984_t0_1782848710.082335_p37_st1_ct8.mp4

It will output a new name like:
TakingStock_T{TOPIC}_p{folder_arms_pose}_s{folder_signature}_obj_{obj}_h{folder_hsv}.mp4
'''

FOLDER = "/Volumes/OWC52/_finished_work.mirrorRAID18/_FINISHED_WORK_THEOFFICE"
MP4_ONLY = True 
io = DataIO()

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
    "15-21": "Blue-Purple",
    "15-19": "Blue",
    "3-22": "Red-Yellow-Blue",
    "0": "Black",
    "1": "Grey",
    "2": "White",
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


# Specify the precise path to the python file
file_path = "/Users/michaelmandiberg/Documents/GitHub/taking-stock-yolo/class_map_utils.py"
class_map_json = "/Users/michaelmandiberg/Documents/GitHub/taking-stock-yolo/config/custom_class_map.json"
module_name = "class_map" # The name you want to assign to the module

# Load the file location specification
spec = importlib.util.spec_from_file_location(module_name, file_path)
class_map = importlib.util.module_from_spec(spec)

# Execute the module to make its functions available
spec.loader.exec_module(class_map)

# Use the imported module
CLASS_MAP = class_map.get_id_to_name(class_map_json)

object_signature_registry = io.load_object_signature_registry(OBJECT_SIGNATURE_EXPORT_PATH)
# print(f"object_signature_registry: {object_signature_registry}")
folder_list = io.get_folders(FOLDER)
# print(f"folder_list: {folder_list}")

def get_modal_signature(folder):
    img_list = io.get_img_list(folder)
    signature_list = []
    for img in img_list:
        folder_arms_pose, folder_hands_gesture, folder_signature, folder_hsv = io.extract_fusion_cluster(img)
        signature_list.append(folder_signature)
    # get the most common signature
    modal_signature = max(set(signature_list), key=signature_list.count)
    return modal_signature

def format_title(topic, folder_arms_pose, folder_hands_gesture, folder_signature, obj_str, folder_hsv):

    title = "Taking Stock "
    if topic is not None:
        title += f"Topic {topic}, "
    if folder_arms_pose is not None:
        title += f"Pose {folder_arms_pose}, "
    if folder_hands_gesture is not None:
        title += f"Gesture {folder_hands_gesture}, "
    # if folder_signature is not None:
    #     title += f"Signature {folder_signature}, "
    if obj_str is not None:
        if "-" in obj_str:
            obj_list = obj_str.split("-")
            # create a string like "Object obj1, obj2, and obj3"
            obj_id_and_name_list = [f"{obj_id} ({CLASS_MAP.get(int(obj_id), 'Unknown')})" for obj_id in obj_list]
            obj_str = ", ".join(obj_id_and_name_list[:-1]) + f" and {obj_id_and_name_list[-1]}"
            title += f"Objects {obj_str}, "
        else: 
            obj_id_and_name = f"{obj_str} ({CLASS_MAP.get(int(obj_str), 'Unknown')})"
            title += f"Object {obj_id_and_name}, "
    if folder_hsv is not None:
        hsv_name = HSV_DICT.get(folder_hsv, "")
        title += f"HSV {folder_hsv} ({hsv_name}) "

    # chomp any trailing comma and space
    title = title.rstrip(", ")

    return title


# construct df for TOPIC}_p{folder_arms_pose}_g{folder_hands_gesture}_s{folder_signature}_obj{obj_str}_h{folder_hsv
df = pd.DataFrame(columns=["new_name", "title", "folder", "img", "folder_arms_pose", "folder_hands_gesture", "folder_signature", "folder_hsv", "obj_str", ])
for folder in folder_list:
    img_list = io.get_img_list(folder)
    # print(f"img_list: {img_list}")
    for img in img_list:
        if MP4_ONLY and not "mp4" in img: continue
        # print(f"img: {img}")
        folder_arms_pose, folder_hands_gesture, folder_signature, folder_hsv = io.extract_fusion_cluster(img)
        # print(f"folder_arms_pose: {folder_arms_pose}, folder_hands_gesture: {folder_hands_gesture}, folder_signature: {folder_signature}, folder_hsv: {folder_hsv}")
        obj = object_signature_registry.get(io.normalize_cluster_token(folder_signature), None)
        # print(type(obj))

        if obj == "None" or obj is None:
            print(f" ❌ WARNING: object {folder_signature} not found in object_signature_registry, skipping")
            continue
        else: 
            obj_str = obj.replace(", ", "-").replace("[", "").replace("]", "")
        # print(f"obj: {obj}, obj_str: {obj_str}")

        new_name = f"TakingStock_T{TOPIC}_p{folder_arms_pose}_g{folder_hands_gesture}_s{folder_signature}_obj{obj_str}_h{folder_hsv}.mp4"

        title = format_title(TOPIC, folder_arms_pose, folder_hands_gesture, folder_signature, obj_str, folder_hsv)

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
        }

        df = pd.concat([df, pd.DataFrame([this_dict])], ignore_index=True)

        print(f"new_name: {new_name}")
        print(f"title: {title}")
