'''
Walk ROOT folder. 
For each folder in ROOT, get file list of all jpgs. 
If either dimension is greater than MIN_DIM = 1280, use open cv to reduce the file size to exactly 1/4 the original dimensions.  
Save back to the same filename, overwriting the original.
'''

import os
import cv2

def parse_folder(folderpath):
    this_folders_new_width = this_folders_new_height = None
    for filename in os.listdir(folderpath):
        if filename.endswith(".jpg"):
            # open image with open cv, check dimensions, and if either dimension is greater than MIN_DIM, resize to 1/4 the original dimensions and save back to same filename
            image_path = os.path.join(folderpath, filename)
            image = cv2.imread(image_path)
            if this_folders_new_width is None:
                this_folders_new_height, this_folders_new_width, _ = image.shape
            if this_folders_new_height > MIN_DIM or this_folders_new_width > MIN_DIM:
                new_width = this_folders_new_width // DIVISOR
                new_height = this_folders_new_height // DIVISOR
                resized_image = cv2.resize(image, (new_width, new_height))
                cv2.imwrite(image_path, resized_image)
                print(f"Resized {filename} to {new_width}x{new_height}")
    print(f"Finished processing folder: {folderpath}")
    return

# WALK_FOLDERS = False
MIN_DIM = 1280
DIVISOR = 3
# keyword_id = None

# ROOT= "/Volumes/OWC4/segment_images/phone_tests_nov4_money1000"
# ROOT= "/Users/michaelmandiberg/Documents/projects-active/facemap_production/heft_keyword_fusion_clusters/clustercc332_pNone_t553_h0_FOR_SQLinput"
ROOT = "/Volumes/LaCie/output_folder/_obj_misc_priority2"
# ROOT = "/Volumes/LaCie/dedupe/exclude"


# if "_t" in ROOT:
#     keyword_id = ROOT.split("_t")[1].split("_")[0]
# all_image_ids = []
for foldername in os.listdir(ROOT):
    # foldername is 
    print(f"Processing folder: {foldername}")
    folderpath = os.path.join(ROOT, foldername)
    if os.path.isdir(folderpath):
        parse_folder(folderpath)
        # subfilenames = os.listdir(folderpath)
        # subfoldernames = [fn for fn in subfilenames if os.path.isdir(os.path.join(folderpath, fn))]
        # for subfoldername in subfoldernames:
        #     print(f"  Processing subfolder: {subfoldername}")
        #     subfolderpath = os.path.join(ROOT, foldername, subfoldername)
        #     # subfolderpath is orientation
        #     # if os.path.isdir(folderpath):
        #     sub_image_ids = parse_folder(subfolderpath)
        #     all_image_ids.extend(sub_image_ids)
        #     print(keyword_id, sub_image_ids, foldername, subfoldername)
        #     build_sql(subfolderpath, keyword_id, sub_image_ids, foldername, orientation=subfoldername)

# if WALK_FOLDERS:
#     # save __all__ image ids to csv
#     all_image_ids_str = ",".join(all_image_ids)
#     output_filepath = os.path.join(ROOT, "all_image_ids.txt")
#     with open(output_filepath, "w") as f:
#         f.write(all_image_ids_str)
#     print(f"Saved all image IDs to {output_filepath}")
