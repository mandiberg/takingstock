import os
import csv
import shutil
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# This script reads a CSV with site_name_id and imagename
# It deletes the files from their original locations (io.folder_list)

# WORKS WITH df_sorted format from make_video.py FALL 2025 !!

# if not USE_DF_SORTED then csv format should be <site_name_id>, <local_file_path> this is basic query:
'''
SELECT i.site_name_id, i.imagename
FROM Images i
JOIN ImagesKeywords ik on i.image_id = ik.image_id
JOIN SegmentOct20 s on i.image_id = s.image_id
WHERE ik.keyword_id = 4222
'''

ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/takingstock/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)

# import file

from mp_db_io import DataIO
IS_SSD = False  # if True it will use the SSD path, if False it will use the RAID path
io = DataIO(IS_SSD)

# Define the path to the CSV file
# csv_file = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/test_orig/df_sorted_0_ct9422.csv'


CSV_FOLDER = os.path.join(Path.home(), "Documents/projects-active/takingstock_production/delete_files") # folder containing CSV files
DELETE_FOLDER = "/Volumes/LaCie/segment_images_ALL" # segment_images to delete from. 
USE_DF_SORTED = False  # if True it will use the df_sorted format from make_video.py, false expects output from SQL query above
IS_TEST = False
OUTPUT_INTERVAL = 1000
PRINT_EACH_FILE = False
if IS_TEST:
    CSV_FOLDER = CSV_FOLDER + "_test"
    print(f"Running in test mode. Using CSV_FOLDER: {CSV_FOLDER}")
START = 0

# if os.path.exists("/Volumes/RAID54/images_getty/5/54/975648904.jpg"):
#     print("file exists")
# quit()

deleted_count = 0
notfound_count = 0
error_count = 0
i = 0
counter_lock = Lock()
MAX_WORKERS = 8  # number of threads to use
MAX_IN_FLIGHT = MAX_WORKERS * 4  # throttle outstanding futures to cap memory

def delete_file(file_path):
    """Delete a single file. Returns status string."""
    if not os.path.exists(file_path):
        if PRINT_EACH_FILE:
            print(f" -- File not found: {file_path}")
        return "notfound"

    try:
        os.remove(file_path)
        if PRINT_EACH_FILE:
            print(f" ++ Deleted: {file_path}")
        return "deleted"
    except Exception as e:
        if PRINT_EACH_FILE:
            print(f"An error occurred deleting {file_path}: {e}")
        return "error"


def worker_task(row_data):
    """Worker function for thread pool: constructs file path and deletes it."""
    row, row_idx = row_data
    try:
        site_name_id = int(row[1]) if USE_DF_SORTED else int(row[0])
    except (ValueError, IndexError):
        if PRINT_EACH_FILE:
            print(f"Skipping invalid site_name_id in row: {row}")
        return "error"
    
    try:
        site_folder = os.path.basename(io.folder_list[site_name_id])
        site_root = os.path.join(DELETE_FOLDER, site_folder)
    except (IndexError, KeyError):
        if PRINT_EACH_FILE:
            print(f"Skipping invalid site_name_id {site_name_id}")
        return "error"
    
    filepath = row[3] if USE_DF_SORTED else row[1]
    file_path = os.path.join(site_root, filepath)
    # print(f"Row {row_idx}: Deleting file at {file_path}")
    return delete_file(file_path)


def delete_files_from_csv(csv_file, start=0):
    global i, deleted_count, notfound_count, error_count
    deleted_count = 0
    notfound_count = 0
    error_count = 0
    interval_start_time = time.time()
    interval_deleted = 0
    interval_notfound = 0
    interval_error = 0
    interval_processed = 0
    
    def handle_result(status):
        nonlocal interval_deleted, interval_notfound, interval_error, interval_processed, interval_start_time
        global i, deleted_count, notfound_count, error_count
        if status is None:
            return
        with counter_lock:
            if status == "deleted":
                deleted_count += 1
                interval_deleted += 1
            elif status == "notfound":
                notfound_count += 1
                interval_notfound += 1
            else:
                error_count += 1
                interval_error += 1

            i += 1
            interval_processed += 1

            if interval_processed >= OUTPUT_INTERVAL:
                interval_time = time.time() - interval_start_time
                per_file_sec = interval_time / max(1, interval_processed)
                projected_seconds = 100000 * per_file_sec
                proj_hours = int(projected_seconds // 3600)
                proj_minutes = int((projected_seconds % 3600) // 60)
                print(
                    f"Interval ({interval_processed} files): deleted {interval_deleted}, notfound {interval_notfound}, errors {interval_error}; "
                    f"time {interval_time:.1f}s; projected 100000 files: {proj_hours}h {proj_minutes}m"
                )
                interval_start_time = time.time()
                interval_deleted = interval_notfound = interval_error = 0
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
        i = 0
        print(f"Starting to delete files from {DELETE_FOLDER} using CSV (starting at row {START})...")
        delete_files_from_csv(csv_file_path, start=START)
        print(f"Total files processed: {i}")
        print(f"  Deleted: {deleted_count}, not found: {notfound_count}, errors: {error_count}")
        print("Finished deleting files.")

if __name__ == '__main__':
    main()
