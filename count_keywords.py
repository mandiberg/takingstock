import csv

# modified from ChatGPT code

# Open the CSV file for reading
with open('/Users/michaelmandiberg/Downloads/Pexels_v2/CSV_NOKEYS.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    
    # Skip the first column of each row
    next(reader)
    
    # Initialize a dictionary to count occurrences of each value in the second column
    count_dict = {}
    
    # Iterate over each row in the CSV file
    for row in reader:
        value = row[1] # Get the value from the second column
        if value in count_dict:
            count_dict[value] += 1 # If the value is already in the dictionary, increment its count
        else:
            count_dict[value] = 1 # If the value isn't in the dictionary yet, add it with a count of 1

# Open a new CSV file for writing
with open('/Users/michaelmandiberg/Downloads/Pexels_v2/CSV_NOKEYSoutput_file.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write a header row to the new CSV file
    writer.writerow(['Value', 'Count'])
    
    # Iterate over the keys and values in the count dictionary and write them to the new CSV file
    for key, value in count_dict.items():
        if value > 50:
            writer.writerow([key, value])
