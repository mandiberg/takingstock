import os
import json

def merge_files(input_folder, output_file):
    unique_ids = set()

    with open(output_file, 'w') as outfile:
        for filename in os.listdir(input_folder):
            if filename.endswith('.jsonl'):
                with open(os.path.join(input_folder, filename), 'r') as infile:
                    for line in infile:
                        data = json.loads(line)
                        if data['id'] not in unique_ids:
                            json.dump(data, outfile)
                            outfile.write('\n')
                            unique_ids.add(data['id'])
                        else:
                            print(data['id'])

if __name__ == "__main__":
    input_folder = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/iStock_ingest/"
    output_file = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/iStock_ingest/april15_iStock_output_deduped.jsonl"
    merge_files(input_folder, output_file)
