import os
import sys
from pathlib import Path

ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/takingstock/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)
from mp_db_io import DataIO

'''
This script counts files in subfolders based on fusion cluster. 
This is for when you are making a looping video and need to know how many images are in each cluster folder.
Useful to figuring out the min values
And for comparing one run to an other to see if any need further pruning
'''

FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/_looping_june22_BER"
FOLDER = "/Volumes/LaCie/output_folder/_looping_june22_BK_trimmed"

def get_list(folderpath):
    image_list = []
    folder_list = []
    filelist = os.listdir(folderpath)
    for file in filelist:
        # print("file to sort: ", file)
        if "jpg" in file:
            image_list.append(file)
            # print("is jpg", file)
            # print("current image_list", image_list)
            # print("current folder_list", folder_list)
        if "mp4" in file or "csv" in file or "DS_Store" in file:
            continue
        else:
            # print("is folder,", file)
            folder_list.append(file)
    #         print("current image_list", image_list)
            # print("current folder_list", folder_list)
    # print(f"image_list, {image_list}")
    # print(f"folder_list {folder_list}")
    return image_list, folder_list

def main():
    image_list, folder_list = get_list(FOLDER)
    for folder in folder_list:
        if "DS_Store" in folder: continue
        folderpath = os.path.join(FOLDER,folder)
        # print(f"Processing file: {folderpath}")
        this_arms_pose, this_signature, this_hsv = DataIO.extract_fusion_cluster(folder)
        image_list, folder_list = get_list(folderpath)
        print(this_arms_pose, this_signature, len(image_list))



if __name__ == "__main__":
    main()