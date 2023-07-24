import json

def load_unique_ids(file_path):
    unique_ids = set()
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            unique_ids.add(data['id'])
    return unique_ids

def merge_unique_lines(file_a_path, file_b_path, output_file_path, output_nogender_file_path):
    unique_ids_a = load_unique_ids(file_a_path)

    with open(output_file_path, 'w') as outfile:
	    with open(output_nogender_file_path, 'w') as outfile_nogender:
	        with open(file_b_path, 'r') as file_b:
	            for line in file_b:
	                data = json.loads(line)
	                print(data)
	                print(data['id'])
	                print(data['title'])
	                print(data['filters']['gender'])
	                if data['id'] not in unique_ids_a:
	                	if data['filters']['gender'] == "-women -men -children":
		                    json.dump(data, outfile_nogender)   
		                    outfile_nogender.write('\n')
		                else:
		                    json.dump(data, outfile)
		                    outfile.write('\n')

if __name__ == "__main__":
    file_a_path = "/Users/michaelmandiberg/Downloads/adobe_downloaded.jsonl"
    file_b_path = "/Users/michaelmandiberg/Downloads/adobe_all_jsonl_files.jsonl"

    # file_a_path = "/Users/michaelmandiberg/Downloads/adobe_jsonl_process_delete_when_done/adobe_rebase_outputDONE/items_cache.adobeStockScraper_v2.3_pullskeylist_B.jsonl"
    # file_b_path = "/Users/michaelmandiberg/Downloads/adobe_jsonl_process_delete_when_done/3400DON/3400zeta.jsonl"

    output_file_path = "/Users/michaelmandiberg/Downloads/unique_lines_B.jsonl"
    output_nogender_file_path = "/Users/michaelmandiberg/Downloads/unique_lines_B_nogender.jsonl"

    merge_unique_lines(file_a_path, file_b_path, output_file_path, output_nogender_file_path)
