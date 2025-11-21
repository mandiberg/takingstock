import os
import csv
import shutil
import sys
from pathlib import Path

# calculate_background.py will output a list of missing files. 
# create a csv file with the original path and the destination path
# this script will copy the files from the original path to the destination path

# WORKS WITH df_sorted format from make_video.py FALL 2025 !!

# if not USE_DF_SORTED then csv format should be <site_name_id>, <local_file_path> this is basic query:
'''
SELECT i.site_name_id, i.imagename
FROM Images i
JOIN ImagesKeywords ik on i.image_id = ik.image_id
JOIN SegmentOct20 s on i.image_id = s.image_id
WHERE ik.keyword_id = 4222
'''

ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/facemap/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)

# import file

from mp_db_io import DataIO
IS_SSD = False  # if True it will use the SSD path, if False it will use the RAID path
io = DataIO(IS_SSD)

# Define the path to the CSV file
# csv_file = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/test_orig/df_sorted_0_ct9422.csv'


CSV_FOLDER = os.path.join(io.ROOT_DBx, "NML_transition")
CSV_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/validating_sql_mongo/movethis" # for testing
USE_DF_SORTED = False  # if True it will use the df_sorted format from make_video.py, false expects output from SQL query above
USE_RAW_PATHS = False # this skips the site_name_id and joins the ORIGIN to the filename in the CSV directly
USE_HASH_FOLDERS = True  # if True it will create hash folders in the destination folder
IS_TEST = False
ORIGIN = "/Volumes/LaCie/images_adobe"
# DEST = os.path.join(io.ROOT_DBx, "NMLdeshard")
DEST = "/Volumes/OWC5/segment_images_SQLonly_stillmissing/"
if IS_TEST:
    # to run a smaller test, put a few files in the test folder
    DEST = DEST + "_test"
    CSV_FOLDER = CSV_FOLDER + "_test"
    print(f"Running in test mode. Using DEST: {DEST} and CSV_FOLDER: {CSV_FOLDER}")
START = 0

# make folders
if USE_DF_SORTED:
    print("Using df_sorted format")
    for path in io.folder_list[1:]:
        last_folder = os.path.basename(path)
        site_folder = os.path.join(DEST, last_folder)
        if last_folder == '':
            continue
        # else:
        #     print("site_folder", site_folder)
        #     print(f"Last folder: XX{last_folder}XX")

        if not os.path.exists(site_folder):
            os.makedirs(site_folder)
            print(f"Created folder: {site_folder}")
        io.make_hash_folders(site_folder)
else:
    io.make_hash_folders(DEST)

# if os.path.exists("/Volumes/RAID54/images_getty/5/54/975648904.jpg"):
#     print("file exists")
# quit()

i = 0

def move_file_pair(original_path, destination_path):
    # temp hack to check for existing files first (bc faster SSD read):
    if os.path.exists(destination_path):
        print(f" == File already exists, skipping: {destination_path}")
        return

    # Check if the file path exists
    if os.path.exists(original_path):
        # print(f" + Copying {original_path} to {destination_path}")
        
        try:
            if not os.path.exists(destination_path):
                print(f" ++ Copying file: {original_path} to {destination_path}")
                # Copy the file to the destination location, but leave the original file in place
                shutil.copy(original_path, destination_path)
            
                # Move the file to the destination location, and delete the original file
                # shutil.move(original_path, destination_path)
                # print(original_path, destination_path)
                
                # if i % 100 == 0:
                #     print(f"{i} Files copied successfully.")
                # i += 1

                # print(f"File copied successfully.")
            else:
                print(f"File already exists: {destination_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print(f" ><>< File does not exist: {original_path}")



def move_files_from_csv(csv_file, start=0):
    global i
    # Open the CSV file in read mode
    with open(csv_file, mode='r', newline='') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        # Iterate over each row in the CSV file
        for row in reader:
            if START > i:
                continue
            
            if USE_RAW_PATHS:
                filename = row[1]
                original_path = os.path.join(ORIGIN, filename)
                destination_path = os.path.join(DEST, filename)
            else:
                # # this is for moving is_face = NULL files
                # original_path = os.path.join(ORIGIN,row[1])
                # destination_path = os.path.join(DEST,row[1])

                # this is for moving make_video df_sorted files
                site_name_id = int(row[1]) if USE_DF_SORTED else int(row[0])
                site_root = io.folder_list[site_name_id]
                last_folder = os.path.basename(site_root)
                filepath = row[3] if USE_DF_SORTED else row[1]
                original_path = os.path.join(site_root,filepath)
                if USE_HASH_FOLDERS:
                    destination_path = os.path.join(DEST,last_folder, filepath)
                else:
                    destination_path = os.path.join(DEST, os.path.basename(filepath))
            move_file_pair(original_path, destination_path)
            i += 1

            if i % 100 == 0:
                print(f"{i} Files copied successfully.")
            i += 1

def main():
    global i
    csv_file_list = os.listdir(CSV_FOLDER)
    for csv_file in csv_file_list:
        if not csv_file.endswith('.csv'):
            continue
        csv_file_path = os.path.join(CSV_FOLDER, csv_file)
        print(f"Processing CSV file: {csv_file}")
        i = 0
        print("Starting to move files from CSV...")
        move_files_from_csv(csv_file_path, start=START)
        print(f"Total files moved: {i}")
        print("Finished moving files.")

if __name__ == '__main__':
    main()
