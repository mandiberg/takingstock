import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the source and destination root directories
source_root = "/Volumes/RAID18/images_getty"
dest_root = "/Volumes/OWC4/segment_images/images_getty"

# Function to copy a single file if it is empty in the destination
def replace_file_if_empty(dest_file_path):
    try:
        # Check if the file is empty (size == 0)
        if os.path.getsize(dest_file_path) == 0:
            # Calculate the corresponding source file path
            relative_path = os.path.relpath(dest_file_path, dest_root)  # Get relative path of the file
            source_file_path = os.path.join(source_root, relative_path)  # Construct the source file path

            # Check if the source file exists
            if os.path.exists(source_file_path):
                # Copy the source file to replace the empty destination file
                print(f"Copying {source_file_path} to {dest_file_path}")
                shutil.copy2(source_file_path, dest_file_path)  # copy2 to preserve metadata
            else:
                print(f"Source file not found: {source_file_path}")
    except Exception as e:
        print(f"Error processing {dest_file_path}: {e}")

# Function to recursively search for empty files in a threaded manner
def replace_empty_files_threaded(source_root, dest_root, max_workers=8):
    # Collect all the files to process
    files_to_check = []
    for root, dirs, files in os.walk(dest_root):
        for file in files:
            files_to_check.append(os.path.join(root, file))  # Full path of the destination file

    # Use ThreadPoolExecutor to handle the copying in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(replace_file_if_empty, file): file for file in files_to_check}
        
        for future in as_completed(futures):
            file = futures[future]
            try:
                future.result()  # Will raise an exception if occurred in the thread
            except Exception as e:
                print(f"Error processing file {file}: {e}")

# Run the function
if __name__ == "__main__":
    replace_empty_files_threaded(source_root, dest_root, max_workers=8)  # Adjust max_workers to match your CPU cores
