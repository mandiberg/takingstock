import json
import csv
import os

# Specify the path to your jsonl file and CSV output file
# jsonl_file_path = '/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/INDIA-PB/items_cache.jsonl'
# csv_output_file_path = '/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/INDIA-PB/keys_by_count.csv'

FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/alamyCSV"

jsonl_file_path = os.path.join(FOLDER,'items_cache.jsonl')
csv_output_file_path = os.path.join(FOLDER,'jsonl_keys_by_count.csv')

# Create an empty dictionary to store keyword counts
keyword_counts = {}

# Open the jsonl file and iterate through lines
with open(jsonl_file_path, 'r') as file:
    for line in file:
        # Load the JSON object from the line
        json_object = json.loads(line)
        
        # Extract the 'keywords' value and split it into a list
        keywords = json_object.get('keywords', '').split('|')
        # print(keywords)
        # Update the counts in the dictionary
        for keyword in keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

# Sort the keyword counts by count in descending order
sorted_keyword_counts = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
print(len(sorted_keyword_counts))

# Write sorted keyword counts to a CSV file
with open(csv_output_file_path, 'w', newline='') as csvfile:
    fieldnames = ['Keyword', 'Count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    print("about to write keys")
    for keyword, count in sorted_keyword_counts:
        writer.writerow({'Keyword': keyword, 'Count': count})
    print("wrote keys")
