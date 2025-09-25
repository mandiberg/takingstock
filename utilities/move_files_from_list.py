import os
import csv
import shutil
import sys
from pathlib import Path

# calculate_background.py will output a list of missing files. 
# create a csv file with the original path and the destination path
# this script will copy the files from the original path to the destination path

# WORKS WITH df_sorted format from make_video.py FALL 2025 !!

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
CSV_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/heft_keyword_fusion_clusters/body3D_512_keywords_2_full/focus_clusters" # for testing
USE_DF_SORTED = True  # if True it will use the df_sorted format from make_video.py
IS_TEST = False
# ORIGIN = "/Volumes/OWC4/segment_images/images_adobe"
# DEST = os.path.join(io.ROOT_DBx, "NMLdeshard")
DEST = "/Volumes/OWC4/segment_images_focus_clusters"
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

            # # this is for moving is_face = NULL files
            # original_path = os.path.join(ORIGIN,row[1])
            # destination_path = os.path.join(DEST,row[1])

            # this is for moving make_video df_sorted files
            site_name_id = int(row[1]) if USE_DF_SORTED else int(row[0])
            site_root = io.folder_list[site_name_id]
            last_folder = os.path.basename(site_root)
            filepath = row[3] if USE_DF_SORTED else row[1]
            original_path = os.path.join(site_root,filepath)
            destination_path = os.path.join(DEST,last_folder, filepath)


            # Check if the file path exists
            if os.path.exists(original_path):
                print(f"Copying {original_path} to {destination_path}")
                
                try:
                    if not os.path.exists(destination_path):
                        # Copy the file to the destination location, but leave the original file in place
                        shutil.copy(original_path, destination_path)
                    
                        # Move the file to the destination location, and delete the original file
                        # shutil.move(original_path, destination_path)
                        # print(original_path, destination_path)
                        
                        if i % 100 == 0:
                            print(f"{i} Files copied successfully.")
                        i += 1

                        # print(f"File copied successfully.")
                    else:
                        print(f"File already exists: {destination_path}")
                except Exception as e:
                    print(f"An error occurred: {e}")
            else:
                print(f"File does not exist: {original_path}")


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
