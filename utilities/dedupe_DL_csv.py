import csv

# Define the input and output file paths
INPUT_FILE = "output.csv"
OUTPUT_FILE = "unique_images.csv"
header=""

# Create a set to store unique rows
unique_rows = set()

# Open the input file and read its contents as a CSV
with open(INPUT_FILE, "r") as csvfile:
    reader = csv.reader(csvfile)

    # Loop over each row in the file
    for row in reader:
        if not header:
            header = tuple([row[10],row[11]])
            print(header)
        else:
            # Convert the row to a tuple to make it hashable
            row_tuple = tuple([row[10],row[11]])

            # Add the row tuple to the set of unique rows
            unique_rows.add(row_tuple)

# Write the unique rows to the output file
with open(OUTPUT_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(unique_rows)