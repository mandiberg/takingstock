import csv
import re
import os
from collections import Counter
import pandas as pd

FOLDER = "/Users/michaelmandiberg/Documents/GitHub/facemap/utilities/keys/"
KEYS = os.path.join(FOLDER,'ImagesKeywords_Keywords_SegmentOct20_202509121921.csv')
NOUNS = os.path.join(FOLDER,'objects1000.csv')
output_file = os.path.join(FOLDER,'CSV_NOKEYSunique.csv')

# read the nouns into a set for quick lookup
nouns_set = set()
with open(NOUNS, 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if row:
            noun = row[0].strip()  # Remove leading/trailing whitespaces
            nouns_set.add(noun) # Add noun to the set

# open the CSV file into a df
df = pd.read_csv(KEYS, dtype=str)

df_nouns = df[df['keyword_text'].isin(nouns_set)]
df_non_nouns = df[~df['keyword_text'].isin(nouns_set)]

# save both to the FOLDER
df_nouns.to_csv(os.path.join(FOLDER,'CSV_NOUNS.csv'), index=False)
df_non_nouns.to_csv(os.path.join(FOLDER,'CSV_NOT_NOUNS.csv'), index=False)