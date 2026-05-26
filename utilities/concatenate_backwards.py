import os
import shutil
import re

ONLY_DUPES = False  # Set to True to only include files with "dupes" in the filename
# BASE_FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/Michael-Tench/deduped"  # Change this to your folder
# DUPES_FILE = os.path.join(BASE_FOLDER, "dupes.sql")
# MODE = 1 # 0 is for is_dupe, 1 is for is_not_dupe

# ROOT_FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/Michael-Tench/"
ROOT_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/dupestuff"
dedupe_this = os.path.join(ROOT_FOLDER, 'dedupe_this') # this is where the unsorted files go -- they are needed for comparison
duped = os.path.join(ROOT_FOLDER, 'deduped') # this is where the tench sorted files go
removed_files = os.path.join(ROOT_FOLDER, 'removed_files') # files present in deduped → these are actually NOT dupes → is_not_dupe_of.sql
to_dupe = os.path.join(ROOT_FOLDER, 'to_dupe')           # files in dedupe_this NOT in deduped → these are actual dupes → dupes.sql
dupes_file = os.path.join(to_dupe, "dupes.sql")

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

def compare_both_folders(dedupe_this, duped, removed_files, to_dupe):
    os.makedirs(removed_files, exist_ok=True)
    os.makedirs(to_dupe, exist_ok=True)

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

	# REVERSED: files present in deduped → removed_files (actually NOT dupes → is_not_dupe_of)
    not_dupes = deduped_sql_files
    for rel_path in not_dupes:
        file_name = os.path.basename(rel_path)
        src = os.path.join(duped, rel_path)
        dst = os.path.join(removed_files, file_name)
        shutil.copy2(src, dst)
        print(f"[not-dupe] Copied {src} to {dst}")

	# REVERSED: files in dedupe_this NOT in deduped → to_dupe (actual dupes → dupes.sql)
    actual_dupes = dedupe_sql_files - deduped_sql_files
    for rel_path in actual_dupes:
        file_name = os.path.basename(rel_path)
        src = os.path.join(dedupe_this, rel_path)
        dst = os.path.join(to_dupe, file_name)
        shutil.copy2(src, dst)
        print(f"[dupe] Copied {src} to {dst}")

    print(f"compare_both_folders: {len(not_dupes)} not-dupes → removed_files, {len(actual_dupes)} actual dupes → to_dupe")

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
                                outfile.write(f"INSERT IGNORE INTO IsNotDupeOf (image_id_i, image_id_j, manual_check) VALUES ({image_id_i},{image_id_j}, 1);\n")
                        total_files += 1
    print(f"Merged all .sql files into {IS_NOT_DUPES_FILE} with total {total_files} files.")



if __name__ == "__main__":
	# Sort files into their reversed buckets first
	compare_both_folders(dedupe_this, duped, removed_files, to_dupe)
	# Merge actual dupes (from dedupe_this not in deduped) → dupes.sql
	concatenate_sql_files(to_dupe, dupes_file)
	# Build is_not_dupe_of.sql from files that were present in deduped (actually not-dupes)
	build_not_dupe_sql(removed_files)
	print("All .sql files have been concatenated.")