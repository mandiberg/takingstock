import csv
import os

# Define the input and output file paths
ROOT = "/Volumes/Test36/CSVs_to_ingest/123rfCSVs"
INPUT_FILE = os.path.join(ROOT,"123rf.output.csv")
OUTPUT_FILE = os.path.join(ROOT,"unique_rows.csv")
header=None
counter = 0 
# Create a set to store unique rows
unique_rows = set()

# Open the input file and read its contents as a CSV
with open(INPUT_FILE, "r") as csvfile:
    reader = csv.reader(csvfile)

    # Loop over each row in the file
    for row in reader:
        if counter == 0:
            print(row)
        else:
            # Convert the row to a tuple to make it hashable
            row_tuple = tuple(row)

            # Add the row tuple to the set of unique rows
            unique_rows.add(row_tuple)
            counter += 1
        if (i % 10000) == 0:
            print(counter)

# Write the unique rows to the output file
with open(OUTPUT_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(unique_rows)