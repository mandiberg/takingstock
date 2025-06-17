import os
import shutil
import re
from pathlib import Path
import pandas as pd

# Define the source and target folders
# where you are making your list of files you want to match to the target folder
source_folder = "/Volumes/OWC4/segment_images/cluster21_0_1747365897sorted"
# this is the folder where files get moved out of, into new folder_name
target_folder = "/Volumes/OWC4/segment_images/cluster21_0_1747505897fresh"
folder_name = "_x"

USE_DF_CSV = False  # Set to True if using metas.csv, False if using metas.json
df_enc_filepath = "/Volumes/OWC4/segment_images/cluster21_112_1747250587.244954/df_sorted.csv"

source_folder = os.path.join(source_folder, folder_name)
moved_folder = os.path.join(target_folder, folder_name)

# Create the "moved" subfolder if it doesn't exist
if not os.path.exists(moved_folder):
    os.makedirs(moved_folder)
    print(f"Created directory: {moved_folder}")

# Function to extract UIDs from source folder filenames
def extract_uids_from_source():
    uids = []
    
    if USE_DF_CSV:
        # Read the CSV file and extract UIDs from the "filename" column
        df = pd.read_csv(df_enc_filepath)
        uids = df["image_id"].tolist()
        print(f"Extracted {len(uids)} UIDs from CSV file")
    else:
        # Get all files from the source folder
        for filename in os.listdir(source_folder):
            if os.path.isfile(os.path.join(source_folder, filename)):
                # Remove the file extension
                base_name = os.path.splitext(filename)[0]
                
                # Split on the last underscore and get the UID
                parts = base_name.split('_')
                if len(parts) >= 2:
                    uid = parts[-1]  # Get the last part after splitting
                    uids.append(uid)
                    print(f"Extracted UID: {uid} from file: {filename}")
    
    return uids

# Function to move files from target folder that contain a matching UID
def move_matching_files(uids):
    files_moved = 0
    
    # Get all files from the target folder
    for filename in os.listdir(target_folder):
        file_path = os.path.join(target_folder, filename)
        
        # Skip if it's not a file or is the "moved" folder
        if not os.path.isfile(file_path) or filename == "moved":
            continue
        
        # Check if any UID is contained in the filename
        for uid in uids:

            if str(uid) in filename:
                # Move the file to the "moved" subfolder
                destination = os.path.join(moved_folder, filename)
                shutil.move(file_path, destination)
                print(f"Moved: {filename} to {moved_folder}")
                files_moved += 1
                break  # No need to check other UIDs once matched
    
    return files_moved

# Main execution
if __name__ == "__main__":
    if USE_DF_CSV:
        print(f"Using CSV file: {df_enc_filepath}")
    else:
        print(f"Source folder: {source_folder}")

    print(f"Target folder: {target_folder}")
    print(f"Moved folder: {moved_folder}")
    
    # Step 1: Extract UIDs from source folder filenames
    print("\nExtracting UIDs from source folder...")
    uids = extract_uids_from_source()
    print(f"Found {len(uids)} unique UIDs")
    
    # Step 2: Move matching files from target folder
    print("\nMoving matching files from target folder...")
    files_moved = move_matching_files(uids)
    
    print(f"\nProcess completed. Moved {files_moved} files.")