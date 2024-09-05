import os
import json

def process_jsonl_file(file_path, file_ids_set, output_file):
    with open(file_path, mode='r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            file_id = data.get('id')
            if file_id and file_id not in file_ids_set:
                output_file.write(line)
                file_ids_set.add(file_id)

def process_folder(folder_path, output_file_path):
    file_ids_set = set()  # Initialize the set to track IDs
    with open(output_file_path, mode='w', encoding='utf-8') as output_file:
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith('.jsonl'):  # Process only .jsonl files
                    file_path = os.path.join(root, file_name)
                    process_jsonl_file(file_path, file_ids_set, output_file)

if __name__ == "__main__":
    folder_path = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape"  # Root folder to search
    output_file_path = "output.jsonl"  # Output file path
    
    process_folder(folder_path, output_file_path)
    print(f"Processing complete. Unique entries saved to {output_file_path}.")
