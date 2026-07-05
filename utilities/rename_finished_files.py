import os
from pathlib import Path
import sys

ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/takingstock/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)
from mp_db_io import DataIO
from constants_make_video import SIG_OBJECT_DICT

'''
This script reads a folder of finished files and renames them to a new naming convention.
The finished files have names like:
output_folder__FINISHED_WORK__hsv_bg_2tier_100clustercc1_p119_t2341_h7-13_1782990020.413111_p34_st1_ct8.mp4
output_folder__FINISHED_WORK__looping_june24_itter100clustercc537_p282_t0_1782312201.079829_p34_st1_ct8.mp4
output_folder__FINISHED_WORK__looping_clipboardsclustercc172_p3984_t0_1782848710.082335_p37_st1_ct8.mp4

It will output a new name like:
TakingStock_T{TOPIC}_p{folder_arms_pose}_s{folder_signature}_obj_{obj}_h{folder_hsv}.mp4
'''

FOLDER = "/Volumes/LaCie/output_folder/_FINISHED_WORK_THEOFFICE"
MP4_ONLY = True 
io = DataIO()

TOPIC = 11

folder_list = io.get_folders(FOLDER)
# print(f"folder_list: {folder_list}")

for folder in folder_list:
    img_list = io.get_img_list(folder)
    # print(f"img_list: {img_list}")
    for img in img_list:
        if MP4_ONLY and not "mp4" in img: continue
        print(f"img: {img}")
        folder_arms_pose, folder_signature, folder_hsv = io.extract_fusion_cluster(img)
        print(f"folder_arms_pose: {folder_arms_pose}, folder_signature: {folder_signature}, folder_hsv: {folder_hsv}")
        obj = SIG_OBJECT_DICT.get(int(folder_signature), None)
        if obj is None:
            print(f"WARNING: folder_signature {folder_signature} not found in SIG_OBJECT_DICT, skipping")
            continue
        new_name = f"TakingStock_T{TOPIC}_p{folder_arms_pose}_s{folder_signature}_obj_{obj}_h{folder_hsv}.mp4"
        print(f"new_name: {new_name}")