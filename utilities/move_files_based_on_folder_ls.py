import os
import shutil

GOOD_IMAGES = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/dumbell_sort/exclude/labels"
FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/dumbell_sort/86_dumbbell/images"
NESTED_FOLDERS = False
ACCEPT_TEXT = True
# print(f"GOOD_IMAGES: {GOOD_IMAGES}")


GOOD_IDS = set()
# open folder and load files

def load_files_from_folder(folder):
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            files.append(os.path.join(folder, filename))
        elif ACCEPT_TEXT and filename.endswith(".txt"):
            files.append(os.path.join(folder, filename))
    return files

good_filenames = load_files_from_folder(GOOD_IMAGES)
# filenames look like this: X-180-180_Y-180-180_Z-180-180_cc-1_p321_t0_00001_114665785.jpg
# extract the last number before the file extension and store that in GOOD_IDS
for filename in good_filenames:
    # print(f"Loading good image ID from {filename}")
    base = os.path.basename(filename).replace("_YOLO_debug", "")
    image_id = base.split("_")[-1].split(".")[0]
    # print(f"base: {base}, image_id: {image_id}")
    GOOD_IDS.add(image_id)
print(f"Loaded {len(GOOD_IDS)} good image IDs from {GOOD_IMAGES}")
# for each UID in GOOD_IDS, find the file with that UID in the filename and move it to a new folder

if not NESTED_FOLDERS:
    NEW_FOLDER = os.path.join(FOLDER, "good_ids")
    os.makedirs(NEW_FOLDER, exist_ok=True)
    for uid in GOOD_IDS:
        files = load_files_from_folder(FOLDER)
        for filename in files:
            if str(uid) in filename:
                new_path = os.path.join(NEW_FOLDER, os.path.basename(filename))
                os.rename(filename, new_path)
else:
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
                        # try:
                        #     os.rename(filename, new_path)
                        # except OSError as e:
                        #     print(f"Error occurred while moving {filename}: {e}")