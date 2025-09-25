import os

ONLY_DUPES = False  # Set to True to only include files with "dupes" in the filename
BASE_FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/Michael-Tench/dedupe_this/dupe_sorting_body3D_1024"  # Change this to your folder
MERGED_FILE = os.path.join(BASE_FOLDER, "merged.sql")
def concatenate_sql_files(base_folder, merged_file):
	total_files = 0
	if ONLY_DUPES:
		base_folder = os.path.join(base_folder, "dupe")
	with open(merged_file, "w", encoding="utf-8") as outfile:
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

def move_folders_with_no_sql_files(base_folder):
	for root, dirs, files in os.walk(base_folder):
		# if there is a .jpg in the folder:
		if any(file.endswith(".jpg") for file in files):
			if not any(file.endswith(".sql") for file in files):
				print(f"Folder with no .sql files: {root}")

if __name__ == "__main__":
	move_folders_with_no_sql_files(BASE_FOLDER)
	# concatenate_sql_files(BASE_FOLDER, MERGED_FILE)

print("All .sql files have been concatenated.")