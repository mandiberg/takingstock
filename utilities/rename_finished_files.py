import os
from pathlib import Path
import sys

ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/facemap/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)
from mp_db_io import DataIO

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

object_signature_registry = io.load_object_signature_registry(OBJECT_SIGNATURE_EXPORT_PATH)
# print(f"object_signature_registry: {object_signature_registry}")
folder_list = io.get_folders(FOLDER)
# print(f"folder_list: {folder_list}")

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
        print(f"new_name: {new_name}")