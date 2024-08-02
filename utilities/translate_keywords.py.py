import json
import csv
from collections import defaultdict
import concurrent.futures
import re
import os

def load_translations(csv_file):
    translations = defaultdict(str)
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            translations[row['Chinese']] = row['Fitted']
    return translations

def load_table_translations(csv_file, table_file):
    translations = {}
    table_mapping = {}

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                key = row[0].lower().strip().strip('"')
                try:
                    value = int(row[1].strip())
                    translations[key] = value
                except ValueError:
                    print(f"Warning: Skipping row due to non-integer value: {row}")

    with open(table_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    key = int(row[0].strip())
                    value = row[1].strip().strip('"')
                    table_mapping[key] = value
                except ValueError:
                    print(f"Warning: Skipping row due to non-integer key: {row}")

    return translations, table_mapping

def process_line(line, translations, age_translations, age_mapping, gender_translations, gender_mapping, 
                 ethnicity_translations, ethnicity_mapping, location_translations, location_mapping):
    data = json.loads(line)
    
    # Step 1: Regular translations
    keywords = data['keywords'].split('|')
    translated_keywords = [translations.get(kw, kw) for kw in keywords]
    data['keywords'] = '|'.join(translated_keywords)
    
    # Step 2: Table mapping translations
    keywords = data['keywords'].split('|')
    new_keywords = set(keywords)
    translated_keys = set()

    for table_name, table_trans, table_map in [
        ('age', age_translations, age_mapping),
        ('gender', gender_translations, gender_mapping),
        ('ethnicity', ethnicity_translations, ethnicity_mapping),
        ('location', location_translations, location_mapping)
    ]:
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in table_trans:
                translated_value = table_map.get(table_trans[kw_lower], kw)
                new_keywords.add(translated_value)
                translated_keys.add(kw)

    # Step 3: Remove original untranslated keys that were translated
    final_keywords = new_keywords - translated_keys

    data['keywords'] = '|'.join(final_keywords)
    

    # Process title
    original_title = data['title']
    separator = "***_***"
    
    data['title_cn'] = ""  # Initialize title_cn
    
    if separator in original_title:
        parts = original_title.split(separator)
        data['title_cn'] = parts[0].strip()
        if len(parts) > 1 and parts[1].strip():  # English title exists
            data['title'] = parts[1].strip()
        else:  # Only Chinese title
            data['title'] = ""
    elif any(ord(char) > 127 for char in original_title):  # Contains non-ASCII characters, assume Chinese
        data['title_cn'] = original_title
        data['title'] = ""
    else:  # Assume English
        data['title'] = original_title

    return json.dumps(data, ensure_ascii=False)


def translate_file(input_file, output_file, translations, age_translations, age_mapping, 
                   gender_translations, gender_mapping, ethnicity_translations, ethnicity_mapping, 
                   location_translations, location_mapping):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i, line in enumerate(infile):
                futures.append(executor.submit(process_line, line, translations, 
                                               age_translations, age_mapping,
                                               gender_translations, gender_mapping,
                                               ethnicity_translations, ethnicity_mapping,
                                               location_translations, location_mapping))
                if i % 1000 == 0:
                    print(f"Submitted {i} lines for processing")

            print(f"Total futures submitted: {len(futures)}")
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                outfile.write(result + '\n')
                completed += 1
                
                if completed % 1000 == 0:
                    print(f"Completed and wrote {completed} lines")

            print(f"Total lines processed and written: {completed}")

def main():
    KEYROOT = "/Users/michaelmandiberg/Documents/GitHub/facemap/utilities/keys/"
    JSONROOT = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/VCG2/"
    
    translations = load_translations(os.path.join(KEYROOT, 'CSV_KEY2KEY_VCG.csv'))
    
    age_translations, age_mapping = load_table_translations(
        os.path.join(KEYROOT, 'VCG_age.csv'), 
        os.path.join(KEYROOT, 'age_table.csv')
    )
    
    gender_translations, gender_mapping = load_table_translations(
        os.path.join(KEYROOT, 'CSV_GENDER_DICT.csv'), 
        os.path.join(KEYROOT, 'gender_table.csv')
    )
    
    ethnicity_translations, ethnicity_mapping = load_table_translations(
        os.path.join(KEYROOT, 'CSV_ETH_GETTY.csv'), 
        os.path.join(KEYROOT, 'ethnicity_table.csv')
    )
    
    location_translations, location_mapping = load_table_translations(
        os.path.join(KEYROOT, 'CSV_KEY2LOC.csv'), 
        os.path.join(KEYROOT, 'locations_table.csv')
    )
    
    input_file = os.path.join(JSONROOT, 'items_cache.jsonl')
    output_file = os.path.join(JSONROOT, 'items_cache_translated.jsonl')
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    translate_file(input_file, output_file, translations, 
                   age_translations, age_mapping,
                   gender_translations, gender_mapping,
                   ethnicity_translations, ethnicity_mapping,
                   location_translations, location_mapping)

if __name__ == '__main__':
    main()