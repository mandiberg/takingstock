import os

BASE_FOLDER = "/Volumes/OWC4/segment_images/dupe_sorting_ready"  # Change this to your folder
MERGED_FILE = os.path.join(BASE_FOLDER, "merged.sql")

def concatenate_sql_files(base_folder, merged_file):
	with open(merged_file, "w", encoding="utf-8") as outfile:
		for root, dirs, files in os.walk(base_folder):
			for file in files:
				if file.endswith(".sql") and file != "merged.sql":
					file_path = os.path.join(root, file)
					with open(file_path, "r", encoding="utf-8") as infile:
						# outfile.write(f"-- Start of {file_path}\n")
						outfile.write(infile.read())
						# outfile.write(f"\n-- End of {file_path}\n\n")
	print(f"Merged all .sql files into {merged_file}")

if __name__ == "__main__":
	concatenate_sql_files(BASE_FOLDER, MERGED_FILE)

print("All .sql files have been concatenated.")