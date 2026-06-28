import os
import shutil

'''
This script is for the cluster cleaning process -- for the looping videos
Save your good images in a folder, and then move them into a subfolder 
so you only are cleaning *new* images.
This will walk the folder and move for all subfolders
'''

FOLDER = "/Volumes/LaCie/output_folder/_small_1000/"
GOOD_IMAGES = "/Volumes/LaCie/output_folder/_current_good_images"




GOOD_IDS = set()
# open folder and load files

def load_files_from_folder(folder):
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            files.append(os.path.join(folder, filename))
    return files

good_filenames = load_files_from_folder(GOOD_IMAGES)
# filenames look like this: X-180-180_Y-180-180_Z-180-180_cc-1_p321_t0_00001_114665785.jpg
# extract the last number before the file extension and store that in GOOD_IDS
for filename in good_filenames:
    base = os.path.basename(filename)
    image_id = base.split("_")[-1].split(".")[0]
    GOOD_IDS.add(image_id)
print(f"Loaded {len(GOOD_IDS)} good image IDs from {GOOD_IMAGES}")
# for each UID in GOOD_IDS, find the file with that UID in the filename and move it to a new folder
# NEW_FOLDER = os.path.join(FOLDER, "good_ids")
# if folder contains folders, walk through them and do the same thing
for root, dirs, files in os.walk(FOLDER):
    for dir in dirs:
        this_new_folder = os.path.join(FOLDER,dir, "good_ids")
        os.makedirs(this_new_folder, exist_ok=True)

        dir_path = os.path.join(root, dir)
        files = load_files_from_folder(dir_path)
        for uid in GOOD_IDS:
            for filename in files:
                if str(uid) in filename:
                    new_path = os.path.join(FOLDER, this_new_folder, os.path.basename(filename))
                    print(f"Moving {filename} to {new_path}")
                    try:
                        os.rename(filename, new_path)
                    except OSError as e:
                        print(f"Error occurred while moving {filename}: {e}")