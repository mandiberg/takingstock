import csv
import re
from collections import Counter

input_file = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/done/getty_noeth_toobig/CSV_NOKEYS.csv'
output_file = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/done/getty_noeth_toobig/CSV_NOKEYSunique.csv'

# Read the CSV file and extract the second column
keywords = []
with open(input_file, 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:  # Make sure the row has at least two columns
            keyword = row[1].strip()  # Remove leading/trailing whitespaces
            # Exclude keywords that are only numbers or start with "y\d"
            if not (keyword.isdigit() or re.match(r'^y\d', keyword)):
                keywords.append(keyword)

# Count the occurrences of each valid keyword
keyword_counts = Counter(keywords)

# Sort the keyword counts in descending order
sorted_keyword_counts = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)

# Write the sorted results to a new CSV file
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Count', 'Keyword'])  # Write the header row
    for keyword, count in sorted_keyword_counts:
        writer.writerow([count, keyword])
