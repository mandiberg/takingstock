import os
import time
from datetime import datetime

# Base directory where all folders are located
base_dir = "/Volumes/OWC4/segment_images"

# List of folders to process
folders = [
    "images_vcg_inpaint_1.3_2_2.7_2",
    "images_123rf_inpaint_1.3_2_2.7_2",
    "images_adobe_inpaint_1.3_2_2.7_2",
    "images_alamy_inpaint_1.3_2_2.7_2",
    "images_getty_inpaint_1.3_2_2.7_2",
    "images_india_inpaint_1.3_2_2.7_2",
    "images_picha_inpaint_1.3_2_2.7_2",
    "images_pixcy_inpaint_1.3_2_2.7_2",
    "images_bazzar_inpaint_1.3_2_2.7_2",
    "images_istock_inpaint_1.3_2_2.7_2",
    "images_pexels_inpaint_1.3_2_2.7_2",
    "images_unsplash_inpaint_1.3_2_2.7_2",
    "images_123rf_inpaint_1.3_1.85_2.4_1.85",
    "images_adobe_inpaint_1.3_1.85_2.4_1.85",
    "images_alamy_inpaint_1.3_1.85_2.4_1.85",
    "images_getty_inpaint_1.3_1.85_2.4_1.85",
    "images_india_inpaint_1.3_1.85_2.4_1.85",
    "images_picha_inpaint_1.3_1.85_2.4_1.85",
    "images_pixcy_inpaint_1.3_1.85_2.4_1.85",
    "images_bazzar_inpaint_1.3_1.85_2.4_1.85",
    "images_istock_inpaint_1.3_1.85_2.4_1.85",
    "images_shutterstock_inpaint_1.3_2_2.7_2",
    "images_shutterstock_inpaint_1.3_1.85_2.4_1.85",
]

# Define cutoff date: March 1, 2025
cutoff_timestamp = time.mktime(datetime(2025, 4, 11).timetuple())

# Define keywords to search in filenames
keywords = [
    "_corners",
    "5_extendcv2_corners",
    "4_premerge",
    "inpaint_file",
    "5_aftmerge",
    "6_blurmask",
    "_inpaint",
    "1_prepmask",
    "2_mask",
    "5_cornermask",
    "3_extendcv2",
]

# Function to check if a file is hidden
def is_hidden(filepath):
    return os.path.basename(filepath).startswith('.')

# Process each folder
for folder in folders:
    root_dir = os.path.join(base_dir, folder)
    if not os.path.exists(root_dir):
        print(f"Folder does not exist, skipping: {root_dir}")
        continue

    print(f"Processing folder: {root_dir}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)

            # Skip hidden/system files
            if is_hidden(filepath):
                continue

            try:
                file_mtime = os.path.getmtime(filepath)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

            should_delete = False

            # Check if created/modified after March 1, 2025
            if file_mtime > cutoff_timestamp:
                should_delete = True

            # Check if filename contains any of the keywords
            for keyword in keywords:
                if keyword in filename:
                    should_delete = True
                    break

            # Delete if matched
            if should_delete:
                try:
                    os.remove(filepath)
                    print(f"Deleted: {filepath}")
                except Exception as e:
                    print(f"Failed to delete {filepath}: {e}")
