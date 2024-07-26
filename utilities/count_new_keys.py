import csv
import os

DATA_ROOT = "/Users/michaelmandiberg/Documents/GitHub/facemap/utilities/keys/"
# FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/alamyCSV"
# FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/INDIA-PB"
# FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/ImagesBazzar"
# FOLDER = "/Users/michaelmandiberg/Downloads/pixcy_v2"
# FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/PIXERF"
# FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/iwaria"
# FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/nappy_v3_w-data"
# FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/PICHA-STOCK"
FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/AFRIPICS"


RAW_KEYS = os.path.join(FOLDER, 'jsonl_keys_by_count.csv')
NEW_KEYS = os.path.join(FOLDER, 'jsonl_keys_by_count_new.csv')


def read_csv_keywords(filename,position=0):
    keywords = set()
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            keywords.add(row[position].lower())  # Add lowercase keyword
    return keywords

# Read keywords from the three CSVs to exclude
exclude_keywords = set()
exclude_keywords.update(read_csv_keywords(os.path.join(DATA_ROOT,'Keywords_202407221412.csv'),2))
exclude_keywords.update(read_csv_keywords(os.path.join(DATA_ROOT,'CSV_KEY2KEY_GETTY.csv')))
exclude_keywords.update(read_csv_keywords(os.path.join(DATA_ROOT,'CSV_KEY2LOC.csv')))
exclude_keywords.update(read_csv_keywords(os.path.join(DATA_ROOT,'CSV_GENDER_DICT.csv')))
exclude_keywords.update(read_csv_keywords(os.path.join(DATA_ROOT,'Location_202308041952.csv')))
exclude_keywords.update(read_csv_keywords(os.path.join(DATA_ROOT,'CSV_AGE_DICT.csv')))
exclude_keywords.update(read_csv_keywords(os.path.join(DATA_ROOT,'CSV_ETH_GETTY.csv')))
exclude_keywords.update(read_csv_keywords(os.path.join(DATA_ROOT,'CSV_AGE_DETAIL_DICT.csv')))
exclude_keywords.update(read_csv_keywords(os.path.join(DATA_ROOT,'CSV_GENDER_DICT_TNB.csv')))
exclude_keywords.update(read_csv_keywords(os.path.join(DATA_ROOT,'CSV_ETH_MULTI.csv')))

# Process jsonl_keys_by_count_trim.csv and write new CSV
with open(RAW_KEYS, 'r', encoding='utf-8') as infile, \
     open(NEW_KEYS, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Write header
    header = next(reader)
    writer.writerow(header)
    
    # Process rows
    for row in reader:
        keyword = row[0].lower()
        if keyword not in exclude_keywords:
            writer.writerow(row)

print("Processing complete. New file 'jsonl_keys_by_count_new.csv' has been created.")