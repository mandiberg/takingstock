import csv

# Define the input and output file paths
INPUT_FILE = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/iStock_ingest/first_8M_fix/Images_202308150838.csv"
DIFFERENT_FILE = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/iStock_ingest/first_8M_fix/8103000.csv"
OUTPUT_FILE = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/iStock_ingest/first_8M_fix/8103000_NOLOC.csv"

# Create a set to store unique rows
input_rows = set()
different_rows = set()
unique_rows = set()

db_noloc = []
borked_rows = []
# Open the input file and read its contents as a CSV
with open(INPUT_FILE, "r") as csvfile:
    reader = csv.reader(csvfile)

    # Loop over each row in the file
    for row in reader:
        # this gets rid of the UID from each query
        # row_without_first_column = row[1:]
        # row[0] = 0

        # # Convert the row to a tuple to make it hashable
        # row_tuple = tuple(row)
        db_noloc.append(row[0])
        # Add the value I want to the set of unique rows
        # input_rows.add(row[1])

db_noloc = set(db_noloc)

# Open the input file and read its contents as a CSV
with open(DIFFERENT_FILE, "r") as csvfile:
    reader = csv.reader(csvfile)
    # Write the unique rows to the output file
    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Loop over each row in the file
        for row in reader:

            if row[0] in db_noloc:
                writer.writerow(row)




def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list
 
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def find_noloc_rows(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
 
# Driver Code
# lst1 = [4, 9, 1, 17, 11, 26, 28, 54, 69]
# lst2 = [9, 9, 74, 21, 45, 11, 63, 28, 26]
# print(intersection(lst1, lst2))
# # Driver Code
# lst1 = [23, 15, 2, 14, 14, 16, 20 ,52]
# lst2 = [2, 48, 15, 12, 26, 32, 47, 54]
# union_list = intersection(db_noloc, borked_rows)

# union_rows = different_rows.union(input_rows)

