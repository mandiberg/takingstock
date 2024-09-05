import os
import csv
import re
import json

# Directory where the CSV files are located
root_dir = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/Photo_Scraping/Getty_Scraping_Output_Metas"
output_file = "getty_metas_id_loc.jsonl"

# Function to extract the 5-digit location_id from the filename
def extract_location_id(filename):
    match = re.search(r'_(\d{5})_', filename)
    if match:
        return match.group(1)
    return None

# Function to extract the id from the contentUrl field
def extract_id_from_content_url(content_url):
    match = re.search(r'picture-id(\d+)', content_url)
    if match:
        return match.group(1)
    return None

# Open the output file for writing
with open(output_file, 'w', encoding='utf-8') as jsonl_file:
    # Walk through the directories
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(subdir, file)
                location_id = extract_location_id(file)

                if location_id:
                    # Read the CSV file
                    with open(file_path, newline='', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            content_url = row.get("contentUrl", "")
                            id_value = extract_id_from_content_url(content_url)
                            
                            if id_value:
                                # Create the JSON object
                                item = {
                                    "id": id_value,
                                    "location_id": location_id
                                }
                                # Write the JSON object to the jsonl file
                                jsonl_file.write(json.dumps(item) + "\n")

print(f"Data extraction complete. Output saved to {output_file}.")
