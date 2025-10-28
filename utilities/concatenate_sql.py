import os

ONLY_DUPES = False  # Set to True to only include files with "dupes" in the filename
BASE_FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/Michael-Tench/deduped"  # Change this to your folder
DUPES_FILE = os.path.join(BASE_FOLDER, "dupes.sql")
IS_NOT_DUPES_FILE = os.path.join(BASE_FOLDER, "is_not_dupe_of.sql")
# MODE = 1 # 0 is for is_dupe, 1 is for is_not_dupe

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

def move_folders_with_no_sql_files(base_folder, merged_file):
	### OOOPS this doesn't work because the jpg are site_name_ids not image_ids
	### USE utilities/compare_dupe_sorting_folders.py instead
	with open(merged_file, "w", encoding="utf-8") as outfile:
		outfile.write("Use Stock;\n")
		for root, dirs, files in os.walk(base_folder):
			# if there is a .jpg in the folder:
			jpgs = [file.replace(".jpg","") for file in files if file.endswith(".jpg")]
			# if any(file.endswith(".jpg") for file in files):
			if any(jpgs):
				if not any(file.endswith(".sql") for file in files):
					# print(f"Folder with no .sql files: {root}")
					table = "IsNotDupeOf"
					# sql = "INSERT IGNORE INTO `"+table+"` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
					col_list = ["image_id_i", "image_id_j"]
					image_ids = [int(jpg.split("-")[0]) for jpg in jpgs]
					# for jpg in jpgs:
					print(image_ids)
					sql = f"INSERT IGNORE INTO {table} (image_id_i, image_id_j) VALUES ({image_ids[0]}, {image_ids[1]});\n"
					# with open(file_path, "r", encoding="utf-8") as infile:
					# 	# outfile.write(f"-- Start of {file_path}\n")
					outfile.write(sql)
						# total_files += 1



if __name__ == "__main__":
	# if MODE == 0:
	concatenate_sql_files(BASE_FOLDER, DUPES_FILE)
	# elif MODE == 1:
	move_folders_with_no_sql_files(BASE_FOLDER, IS_NOT_DUPES_FILE)

print("All .sql files have been concatenated.")