import csv

RAW_KEYS = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/alamyCSV/jsonl_keys_by_count_trim.csv'
NEW_KEYS = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/alamyCSV/jsonl_keys_by_count_new.csv'

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
exclude_keywords.update(read_csv_keywords('/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/done/Keywords_202407151345.csv',2))
exclude_keywords.update(read_csv_keywords('/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/done/CSV_KEY2KEY_GETTY.csv'))
exclude_keywords.update(read_csv_keywords('/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/done/CSV_KEY2LOC.csv'))

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