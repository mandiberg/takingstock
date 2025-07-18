import os

folder = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/Michael-Tench/description_counts_bytopic/descriptions"

for filename in os.listdir(folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder, filename)
        with open(file_path, "r") as f:
            lines = f.readlines()
        # Keep only the first 1000 lines
        lines = lines[:1000]
        with open(file_path, "w") as f:
            f.writelines(lines)
print("All CSV files trimmed to 1000 lines.")
