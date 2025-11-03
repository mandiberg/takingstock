import os
import shutil
import re

ONLY_DUPES = False  # Set to True to only include files with "dupes" in the filename
# BASE_FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/Michael-Tench/deduped"  # Change this to your folder
# DUPES_FILE = os.path.join(BASE_FOLDER, "dupes.sql")
# MODE = 1 # 0 is for is_dupe, 1 is for is_not_dupe

ROOT_FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/Michael-Tench/"
dedupe_this = os.path.join(ROOT_FOLDER, 'dedupe_this')
duped = os.path.join(ROOT_FOLDER, 'deduped')
removed_files = os.path.join(ROOT_FOLDER, 'removed_files')
dupes_file = os.path.join(duped, "dupes.sql")

def concatenate_sql_files(base_folder, merged_file):
	total_files = 0
	if ONLY_DUPES:
		base_folder = os.path.join(base_folder, "dupe")
	with open(merged_file, "w", encoding="utf-8") as outfile:
		outfile.write("Use Stock;\n")
		for root, dirs, files in os.walk(base_folder):
			print(len(dirs), "dirs in", root)
			print(len(files), "files in", root)
			for file in files:
				if file.endswith(".sql") and file != "merged.sql":
					file_path = os.path.join(root, file)
					with open(file_path, "r", encoding="utf-8") as infile:
						# outfile.write(f"-- Start of {file_path}\n")
						outfile.write(infile.read())
						total_files += 1
						# outfile.write(f"\n-- End of {file_path}\n\n")
	print(f"Merged all .sql files into {merged_file} with total {total_files} files.")

def compare_both_folders(dedupe_this, duped, removed_files):
    os.makedirs(removed_files, exist_ok=True)

	# Collect all .sql files in dedupe_this (with relative paths)
    dedupe_sql_files = set()
    for root, dirs, files in os.walk(dedupe_this):
        for file in files:
            if file.endswith('.sql'):
                rel_path = os.path.relpath(os.path.join(root, file), dedupe_this)
                dedupe_sql_files.add(rel_path)

	# Collect all .sql files in duped (with relative paths)
    deduped_sql_files = set()
    for root, dirs, files in os.walk(duped):
        for file in files:
            if file.endswith('.sql'):
                rel_path = os.path.relpath(os.path.join(root, file), duped)
                deduped_sql_files.add(rel_path)

	# Find files in dedupe_this not in duped
    to_copy = dedupe_sql_files - deduped_sql_files


    for rel_path in to_copy:
        file_name = os.path.basename(rel_path)
    # print(f"File {file_name} is in dedupe_this but not in duped.")
        src = os.path.join(dedupe_this, rel_path)
        dst = os.path.join(removed_files, file_name)
    # if file_name.endswith(".sql"):
    #     print(f"Would copy {src} to {dst}")
        shutil.copy2(src, dst)
        print(f"Copied {src} to {dst}")

def build_not_dupe_sql(removed_files):
    IS_NOT_DUPES_FILE = os.path.join(ROOT_FOLDER, "is_not_dupe_of.sql")
    total_files = 0
    with open(IS_NOT_DUPES_FILE, "w", encoding="utf-8") as outfile:
        outfile.write("USE Stock;\n")
        for root, dirs, files in os.walk(removed_files):
            for file in files:
                if file.endswith(".sql") and file != "merged.sql":
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as infile:
                        for line in infile:
                            # Match: UPDATE Encodings SET is_dupe_of = 6395958 WHERE image_id = 6378025;
                            m = re.match(r"UPDATE Encodings SET is_dupe_of = (\d+) WHERE image_id = (\d+);", line.strip())
                            if m:
                                image_id_i = m.group(1)
                                image_id_j = m.group(2)
                                outfile.write(f"INSERT IGNORE INTO IsNotDupeOf (image_id_i, image_id_j) VALUES ({image_id_i},{image_id_j});\n")
                        total_files += 1
    print(f"Merged all .sql files into {IS_NOT_DUPES_FILE} with total {total_files} files.")



if __name__ == "__main__":
	# if MODE == 0:
	concatenate_sql_files(duped, dupes_file)
	# elif MODE == 1:
	# this part deals with is_not_dupe_of
	compare_both_folders(dedupe_this, duped, removed_files)
	build_not_dupe_sql(removed_files)
	print("All .sql files have been concatenated.")