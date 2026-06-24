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

WALK_FOLDERS = False
DEDUPE = EXCLUDE = False
LOOPING_ONLY = True
keyword_id = None

# ROOT= "/Volumes/OWC4/segment_images/phone_tests_nov4_money1000"
# ROOT= "/Users/michaelmandiberg/Documents/projects-active/facemap_production/excludes"
# ROOT = "/Volumes/LaCie/dedupe/" # for dedupe/exclue, just point to the parent folder holding both
ROOT = "/Volumes/LaCie/output_folder/_excludes"


def parse_folder(folderpath, exclude=False):
    image_ids = []
    cluster_id = p_id = None
    for filename in os.listdir(folderpath):
        if filename.endswith(".jpg"):
            if exclude:
                print(f"Parsing filename for exclude: {filename}")
                cluster_id = filename.split("_cc")[1].split("_")[0]
                p_id = filename.split("_p")[1].split("_")[0]
            parts = filename.split("_")
            image_id_with_ext = parts[-1]
            image_id = image_id_with_ext.split(".")[0]
            if cluster_id and p_id:
                image_ids.append((image_id, cluster_id, p_id))
            else:
                image_ids.append(image_id)
    image_ids_str = ",".join([id for id, _, _ in image_ids]) if image_ids and isinstance(image_ids[0], tuple) else ",".join(image_ids)
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
    
def build_dedupe_sql(folderpath, image_ids):
    print(f"Building dedupe SQL for {len(image_ids)} image_ids")
    output_filepath = os.path.join(folderpath, f"dedupe.sql")
    with open(output_filepath, "w") as f:
        f.write("USE Stock;\n")
        for image_id in image_ids:
            if isinstance(image_id, tuple):
                image_id = image_id[0]
            stmt = f"UPDATE Encodings SET is_dupe_of = 1 WHERE image_id = {image_id};\n"
            f.write(stmt)

def build_exclude_sql(folderpath, image_ids, looping=False):
    print(f"Building exclude SQL for {len(image_ids)} image_ids and is looping {looping}")
    output_filepath = os.path.join(folderpath, f"exclude.sql")
    with open(output_filepath, "w") as f:
        f.write("USE Stock;\n")
        for image_id, c_id, p_id in image_ids:
            if looping: 
                stmt = f"INSERT IGNORE INTO Exclude (image_id, c_id, p_id, looping_only) VALUES ({image_id}, {c_id}, {p_id}, True);\n"
            else:                
                stmt = f"INSERT IGNORE INTO Exclude (image_id, c_id, p_id) VALUES ({image_id}, {c_id}, {p_id});\n"
            f.write(stmt)


if "_t" in ROOT:
    keyword_id = ROOT.split("_t")[1].split("_")[0]
all_image_ids = []
for foldername in os.listdir(ROOT):
    # foldername is 
    if "dupe" in [foldername]:
        DEDUPE = True
    elif "exclude" in foldername:
        EXCLUDE = True
    print(f"Processing folder: {foldername}, DEDUPE: {DEDUPE}, EXCLUDE: {EXCLUDE}")
    folderpath = os.path.join(ROOT, foldername)
    if os.path.isdir(folderpath):
        image_ids = parse_folder(folderpath, EXCLUDE)
        if DEDUPE:
            print(f"Going to build dedupe SQL for {len(image_ids)} image IDs")
            build_dedupe_sql(folderpath, image_ids)
        elif EXCLUDE:
            print(f"Going to build exclude SQL for {len(image_ids)} image IDs")
            # image_ids is a list of tuples (image_id, cluster_id, p_id)
            build_exclude_sql(folderpath, image_ids, LOOPING_ONLY)
        else:
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
