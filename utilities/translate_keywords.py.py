import string
import json
import csv
from collections import defaultdict
import concurrent.futures
import re
import os

# def load_translations(csv_file):
#     translations = defaultdict(str)
#     with open(csv_file, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             translations[row['Chinese']] = row['Fitted']
#     return translations


def load_translations(csv_file):
    translations = defaultdict(str)
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:  # Ensure the row has at least two columns
                translations[row[0]] = row[1]
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

    # if no keywords get them from description
    if not data['keywords']:
        if data['title']:
            cleaned_words = []
            words = data['title'].split(' ')
            print("words:", words)
            for word in words:
                # strip all punctuation
                orig_word = word
                cleaned_word = word.strip(string.punctuation)
                cleaned_word = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', cleaned_word)
                cleaned_word = cleaned_word.replace('.', '')
                
                if cleaned_word != orig_word:
                    print(f"Warning: Stripped punctuation from word: {orig_word} -> {cleaned_word}")
                if cleaned_word:  # Only add non-empty words
                    cleaned_words.append(cleaned_word)
            data['keywords'] = '|'.join(cleaned_words)
            
            # If you need to join the words back into a string:
            # cleaned_title = ' '.join(cleaned_words)



        else:
            data['keywords'] = data.get("filters", {}).get("base", None).replace('person ', '')
            # print("Warning: Empty keywords field, using filters instead:", data['keywords'])

    # Step 1: Regular translations
    keywords = data['keywords'].split('|')
    
    translated_keywords = [translations.get(kw, kw) for kw in keywords]
    translated_keywords = [kw for kw in translated_keywords if kw is not None]
    if translated_keywords:
        if len(translated_keywords) == 0:
            print("Length of translated keywords:", len(translated_keywords))
            print("type of translated keywords:", type(translated_keywords))
            print("Warning: Empty translated keywords field:", data)
            data['keywords'] = ''
        else:
            data['keywords'] = '|'.join(translated_keywords)
    else:
        print("Warning: Empty keywords field:", data)
        data['keywords'] = ''
        

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
    

    # translate the location filter
    # location_filter = data.get("filters", {}).get("location", None)
    location_filter = data.get("location", None)
    location_filter = location_filter.lower() if location_filter else None
    if location_filter in location_translations:
        translated_location = location_mapping.get(location_translations[location_filter], location_filter)
        # data['filters']['location'] = translated_location
        data['location'] = translated_location


    # # Process title - VCG only
    # original_title = data['title']
    # separator = "***_***"
    
    # data['title_cn'] = ""  # Initialize title_cn
    
    # if separator in original_title:
    #     parts = original_title.split(separator)
    #     data['title_cn'] = parts[0].strip()
    #     if len(parts) > 1 and parts[1].strip():  # English title exists
    #         data['title'] = parts[1].strip()
    #     else:  # Only Chinese title
    #         data['title'] = ""
    # elif any(ord(char) > 127 for char in original_title):  # Contains non-ASCII characters, assume Chinese
    #     data['title_cn'] = original_title
    #     data['title'] = ""
    # else:  # Assume English
    #     data['title'] = original_title

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
    JSONROOT = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/alamyCSV/"
    # JSONROOT = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/nappy_v3_w-data"
    
    # translations = load_translations(os.path.join(KEYROOT, 'CSV_KEY2KEY_VCG.csv'))
    translations = load_translations(os.path.join(KEYROOT, 'CSV_KEY2KEY_ALAMY.csv'))
    translations2 = load_translations(os.path.join(KEYROOT, 'CSV_KEY2KEY_GETTY.csv'))
    translations.update(translations2)

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
    # input_file = os.path.join(JSONROOT, 'items_cache.notvia.jsonl')
    output_file = os.path.join(JSONROOT, 'items_cache_translated.jsonl')

    # # for second VCG2 translation
    # translations2 = load_translations(os.path.join(KEYROOT, 'CSV_KEY2KEY_VCG2.csv'))
    # translations.update(translations2)
    # input_file = os.path.join(JSONROOT, 'items_cache_translated_middle.jsonl')
    # output_file = os.path.join(JSONROOT, 'items_cache_translated.jsonl')


    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    translate_file(input_file, output_file, translations, 
                   age_translations, age_mapping,
                   gender_translations, gender_mapping,
                   ethnicity_translations, ethnicity_mapping,
                   location_translations, location_mapping)

if __name__ == '__main__':
    main()