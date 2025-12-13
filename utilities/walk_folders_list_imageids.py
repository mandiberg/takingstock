'''
Walk ROOT folder. 
For each folder in ROOT, get file list of all jpgs. They will have a file structure like this:
X-180-180_Y-180-180_Z-180-180_cc332_pNone_t553_h0-22_00340_12721734.jpg
Extract the image_id from teh filename, which is the number after the last underscore and before the .jpg
Build a list of all image_ids in the folder.
join that list into a single string, separated by commas.
Save that list as a text file in the same folder, called image_ids.txt
'''

import os

def parse_folder(folderpath):
    image_ids = []
    for filename in os.listdir(folderpath):
        if filename.endswith(".jpg"):
            parts = filename.split("_")
            image_id_with_ext = parts[-1]
            image_id = image_id_with_ext.split(".")[0]
            image_ids.append(image_id)
    image_ids_str = ",".join(image_ids)
    output_filepath = os.path.join(folderpath, "image_ids.txt")
    with open(output_filepath, "w") as f:
        f.write(image_ids_str)
    print(f"Saved image IDs to {output_filepath}")
    return image_ids

def build_sql(folderpath, keyword_id, image_ids, object_id, orientation=None):
    print(f"Building SQL for object_id, orientation: {object_id}, {orientation}")
    output_filepath = os.path.join(folderpath, f"object{keyword_id}.sql")
    if orientation == None: orientation = "NULL"
    with open(output_filepath, "w") as f:
        f.write("USE Stock;\n")
        for image_id in image_ids:
            stmt = f"INSERT IGNORE INTO Images{keyword_id} (image_id, object_id, orientation) VALUES ({image_id},{object_id},{orientation});\n"
            f.write(stmt)
    print(f"Saved sql to {output_filepath}")
    
    # USE Stock;
    # INSERT IGNORE INTO IsNotDupeOf (image_id_i, image_id_j) VALUES (112511474,110166102);

WALK_FOLDERS = True
# ROOT= "/Volumes/OWC4/segment_images/phone_tests_nov4_money1000"
ROOT= "/Users/michaelmandiberg/Documents/projects-active/facemap_production/heft_keyword_fusion_clusters/clustercc332_pNone_t553_h0_FOR_SQLinput"
ROOT = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/dupe_sorting_flags_notsorting_reclusterfirst/clustercc77_pNone_t553_h2-2_1762728693.992975"
keyword_id = ROOT.split("_t")[1].split("_")[0]
all_image_ids = []
for foldername in os.listdir(ROOT):
    # foldername is 
    print(f"Processing folder: {foldername}")
    folderpath = os.path.join(ROOT, foldername)
    if os.path.isdir(folderpath):
        image_ids = parse_folder(folderpath)
        build_sql(folderpath, keyword_id, image_ids, foldername, orientation=None)
        subfilenames = os.listdir(folderpath)
        subfoldernames = [fn for fn in subfilenames if os.path.isdir(os.path.join(folderpath, fn))]
        for subfoldername in subfoldernames:
            print(f"  Processing subfolder: {subfoldername}")
            subfolderpath = os.path.join(ROOT, foldername, subfoldername)
            # subfolderpath is orientation
            # if os.path.isdir(folderpath):
            sub_image_ids = parse_folder(subfolderpath)
            all_image_ids.extend(sub_image_ids)
            print(keyword_id, sub_image_ids, foldername, subfoldername)
            build_sql(subfolderpath, keyword_id, sub_image_ids, foldername, orientation=subfoldername)

if WALK_FOLDERS:
    # save __all__ image ids to csv
    all_image_ids_str = ",".join(all_image_ids)
    output_filepath = os.path.join(ROOT, "all_image_ids.txt")
    with open(output_filepath, "w") as f:
        f.write(all_image_ids_str)
    print(f"Saved all image IDs to {output_filepath}")