import os
import re

# THIS IS A DESTRUCTIVE OPERATION. DUPLICATE YOUR FILES FIRST, to keep an intact copy.
ROOT_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/heft_keyword_fusion_clusters/body3D_768_CSVs_test"  # Change this to your folder
TRIM_ON_SPLIT = True
SPLIT_PATTERN = r"^[0-9,A-Z,a-z]+,\d+,http"
NTH_PATTERN = 40 # Change this to the number of entries you want to keep
TRIM_LENGTH = 4200 * NTH_PATTERN  # about 4000-4200 lines per entry
# note that the make_video CSVs have multiline elements, so 80000 lines is about 20 entries

def trim_csv_files(root_folder, trim_length):
    for filename in os.listdir(root_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(root_folder, filename)
            with open(file_path, "r", encoding="utf-8") as infile:
                lines = infile.readlines()
            with open(file_path, "w", encoding="utf-8") as outfile:
                outfile.writelines(lines[:trim_length])
            print(f"Trimmed {filename} to {min(len(lines), trim_length)} lines.")

def split_csv_on_pattern(root_folder, split_pattern, nth_pattern, output_suffix="_split.csv"):
    pattern = re.compile(split_pattern)
    for filename in os.listdir(root_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(root_folder, filename)
            with open(file_path, "r", encoding="utf-8") as infile:
                lines = infile.readlines()
            count = 0
            split_index = None
            for i, line in enumerate(lines):
                if pattern.search(line):
                    count += 1
                    if count == nth_pattern:
                        out1 = file_path.replace('.csv', f'{output_suffix}')
                        split_index = i   # include this line in the first part
                        break
            if split_index is not None:
                # print(f"Split {filename} at {split_index} (after {nth_pattern} matches of pattern '{split_pattern}') ")
                part1 = lines[:split_index]
                # print(part1)
                part2 = lines[split_index:]
                out1 = file_path
                # out1 = file_path.replace('.csv', f'{output_suffix}')
                # out2 = file_path.replace('.csv', f'{output_suffix}2')
                with open(out1, "w", encoding="utf-8") as f1:
                    f1.writelines(part1)
                # with open(out2, "w", encoding="utf-8") as f2:
                #     f2.writelines(part2)
                print(f"Split {filename} at {split_index} (after {nth_pattern} matches of pattern '{split_pattern}') into {out1} ")
            else:
                print(f"Pattern '{split_pattern}' did not appear {nth_pattern} times in {filename}. No split performed.")

if __name__ == "__main__":
    if TRIM_ON_SPLIT: split_csv_on_pattern(ROOT_FOLDER, SPLIT_PATTERN, NTH_PATTERN)
    else: trim_csv_files(ROOT_FOLDER, TRIM_LENGTH)

