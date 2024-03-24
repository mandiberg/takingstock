import os
import csv
import shutil

# calculate_background.py will output a list of missing files. 
# create a csv file with the original path and the destination path
# this script will copy the files from the original path to the destination path

# Define the path to the CSV file
csv_file = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/process_documentation/process_2024_march/background_run_missing_images.csv'

# Open the CSV file in read mode
with open(csv_file, mode='r', newline='') as file:
    # Create a CSV reader object
    reader = csv.reader(file)
    
    # Iterate over each row in the CSV file
    for row in reader:
        # Extract the file paths from the first and second columns
        original_path = row[0]
        destination_path = row[1]
        
        # Check if the file path exists
        if os.path.exists(original_path):
            print(f"Copying {original_path} to {destination_path}")
            
            # Copy the file to the destination location
            shutil.copy(original_path, destination_path)
            
            print(f"File copied successfully.")
        else:
            print(f"File {original_path} does not exist.")
