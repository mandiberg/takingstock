import os
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from mp_db_io import DataIO

######## Michael's Credentials ########
# platform specific credentials
io = DataIO()
db = io.db
# overriding DB for testing



ROOT_FOLDER = '/Volumes/ExFAT_SSD4_/images_adobe'

# walk through the root folder and get all folder paths
def get_all_folder_paths(root_folder):
    folder_paths = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if dirnames:  # Only add directories that have subdirectories
            for dirname in dirnames:
                print(f"Found folder: {dirname} in {dirpath}")
                folder_paths.append(os.path.join(dirpath,dirname))
    return folder_paths

print("Getting all folder paths...")
folder_paths = get_all_folder_paths(ROOT_FOLDER)
print(f"Found {len(folder_paths)} folders.")
print("Folder paths:", folder_paths)

for folder in folder_paths:
    img_list = io.save_img_list(folder)
    print(f"Saved image list for folder: {folder} with {len(img_list)} images.")
