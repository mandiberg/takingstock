import os
import csv
import shutil
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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
CSV_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/moving_objects_to_SSDs/move_this" # for testing
USE_DF_SORTED = False  # if True it will use the df_sorted format from make_video.py, false expects output from SQL query above
USE_RAW_PATHS = False # this skips the site_name_id and joins the ORIGIN to the filename in the CSV directly
USE_HASH_FOLDERS = True  # if True it will create hash folders in the destination folder
FROM_SSD_TO_SSD = False # overrides io settings to move from the ORIGIN_SSD to DEST 
ORIGIN_SSD = "/Volumes/SSD4_Green/segment_images_OWC4"
ORIGIN_SSD = "/Volumes/LaCie/segment_images_book_clock_bowl"
IS_TEST = False
OUTPUT_INTERVAL = 1000
PRINT_EACH_FILE = False
ORIGIN = "segment_images_ALL" # this needs to be path to segment_images/images_*
# DEST = os.path.join(io.ROOT_DBx, "NMLdeshard")
DEST = "/Volumes/SSD4_Green/segment_images_detected_63_67"
# DEST = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/segment_images_detected_63_67"  # for testing
if IS_TEST:
    # to run a smaller test, put a few files in the test folder
    DEST = DEST + "_test"
    CSV_FOLDER = CSV_FOLDER + "_test"
    print(f"Running in test mode. Using DEST: {DEST} and CSV_FOLDER: {CSV_FOLDER}")
START = 3200000
# START = 0  # for testing restart

# check for site folders
io.check_site_folders(DEST)

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

moved_count = 0
exists_count = 0
failed_count = 0
i = 0
counter_lock = Lock()
MAX_WORKERS = 8  # number of threads to use
MAX_IN_FLIGHT = MAX_WORKERS * 4  # throttle outstanding futures to cap memory

def move_file_pair(original_path, destination_path):
    # temp hack to check for existing files first (bc faster SSD read):
    if os.path.exists(destination_path):
        if PRINT_EACH_FILE:
            print(f" == File already exists, skipping: {destination_path}")
        return "exists"

    # Check if the file path exists
    if os.path.exists(original_path):
        try:
            if not os.path.exists(destination_path):
                if PRINT_EACH_FILE:
                    print(f" ++ Copying file: {original_path} to {destination_path}")
                # Copy the file to the destination location, but leave the original file in place
                shutil.copy(original_path, destination_path)
            else:
                if PRINT_EACH_FILE:
                    print(f"File already exists: {destination_path}")
                return "exists"
        except Exception as e:
            if PRINT_EACH_FILE:
                print(f"An error occurred: {e}")
            return "error"
    else:
        if PRINT_EACH_FILE:
            print(f" ><>< File does not exist: {original_path}")
        return "missing"

    return "moved"


def worker_task(row_data):
    """Worker function for thread pool"""
    # print("worker_task called")
    row, row_idx = row_data
    if USE_RAW_PATHS:
        filename = row[1]
        original_path = os.path.join(ORIGIN, filename)
        destination_path = os.path.join(DEST, filename)
    else:
        # if not isinstance(row[1], int): return "error"
        try:site_name_id = int(row[1]) if USE_DF_SORTED else int(row[0])
        except ValueError:
            if PRINT_EACH_FILE:
                print(f"Skipping invalid site_name_id line: {row[1] if USE_DF_SORTED else row[0]}")
            return "error"
        site_root = io.folder_list[site_name_id]
        last_folder = os.path.basename(site_root)
        filepath = row[3] if USE_DF_SORTED else row[1]
        original_path = os.path.join(site_root, filepath)
        # print(f"Row {row_idx}: Moving from {original_path} to {destination_path}")
        if USE_HASH_FOLDERS:
            if FROM_SSD_TO_SSD:
                original_path = os.path.join(ORIGIN_SSD, last_folder, filepath)
                destination_path = os.path.join(DEST, last_folder, filepath)
                # if not os.path.exists(destination_path):
                #     if "000" in destination_path:
                #         if PRINT_EACH_FILE:
                #             print(f" not exist on SSD: {destination_path}, using RAID path instead.")
                #     return None
            else:
                destination_path = os.path.join(DEST, last_folder, filepath)
        else:
            destination_path = os.path.join(DEST, os.path.basename(filepath))
    
    return move_file_pair(original_path, destination_path)


def move_files_from_csv(csv_file, start=0):
    global i, moved_count, exists_count, failed_count
    moved_count = 0
    exists_count = 0
    failed_count = 0
    interval_start_time = time.time()
    interval_moved = 0
    interval_exists = 0
    interval_failed = 0
    interval_processed = 0
    
    def handle_result(status):
        nonlocal interval_moved, interval_exists, interval_failed, interval_processed, interval_start_time
        global i, moved_count, exists_count, failed_count
        if status is None:
            return
        with counter_lock:
            if status == "moved":
                moved_count += 1
                interval_moved += 1
            elif status == "exists":
                exists_count += 1
                interval_exists += 1
            else:
                failed_count += 1
                interval_failed += 1

            i += 1
            interval_processed += 1

            if interval_processed >= OUTPUT_INTERVAL:
                interval_time = time.time() - interval_start_time
                per_file_sec = interval_time / max(1, interval_processed)
                projected_seconds = 100000 * per_file_sec
                proj_hours = int(projected_seconds // 3600)
                proj_minutes = int((projected_seconds % 3600) // 60)
                print(
                    f"Interval ({interval_processed} files): moved {interval_moved}, existed {interval_exists}, failed {interval_failed}; "
                    f"time {interval_time:.1f}s; projected 100000 files: {proj_hours}h {proj_minutes}m"
                )
                interval_start_time = time.time()
                interval_moved = interval_exists = interval_failed = 0
                interval_processed = 0

    submitted = 0
    futures = []
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for idx, row in enumerate(reader):
                if idx < start:
                    continue
                futures.append(executor.submit(worker_task, (row, idx)))
                submitted += 1

                if len(futures) >= MAX_IN_FLIGHT:
                    for future in as_completed(futures):
                        handle_result(future.result())
                    futures.clear()

            # drain any remaining futures
            for future in as_completed(futures):
                handle_result(future.result())

    print(f"Processed {submitted} files with up to {MAX_WORKERS} threads (max in-flight {MAX_IN_FLIGHT}).")

def main():
    global i
    csv_file_list = os.listdir(CSV_FOLDER)
    for csv_file in csv_file_list:
        if not csv_file.endswith('.csv'):
            continue
        csv_file_path = os.path.join(CSV_FOLDER, csv_file)
        print(f"Processing CSV file: {csv_file}")
        print(f"moving to {DEST}")
        i = 0
        print("Starting to move files from CSV... starting at ", START)
        move_files_from_csv(csv_file_path, start=START)
        print(f"Total files processed: {i}")
        print(f"  Moved: {moved_count}, existed: {exists_count}, failed: {failed_count}")
        print("Finished moving files.")

if __name__ == '__main__':
    main()
