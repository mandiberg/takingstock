import os
import csv
import shutil

# calculate_background.py will output a list of missing files. 
# create a csv file with the original path and the destination path
# this script will copy the files from the original path to the destination path

# Define the path to the CSV file
csv_file = '/Volumes/2TB_Quadra_Bare 1/Encodings_Images_202404132159.istock.csv'

ORIGIN = "/Volumes/RAID54/images_istock"
DEST = "/Volumes/2TB_Quadra_Bare 1/images_istock"
START = 0

def make_hash_folders(path):
    def touch(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    alphabet = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9'  
    # alphabet = '0'  
    alphabet2 = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9'  
    # alphabet = 'A B C 0 1 2'   #short alphabet for testing purposes
    # alphabet2 = 'A B C 0 1 2'   #short alphabet for testing purposes
    alphabet = alphabet.split()
    alphabet2 = alphabet2.split()

    for letter in alphabet:
        # print (letter)
        pth = os.path.join(path,letter)
        touch(pth)
        for letter2 in alphabet2:
            # print (letter2)
            pth2 = os.path.join(path,letter,letter+letter2)
            touch(pth2)

# make folders
make_hash_folders(DEST)

# if os.path.exists("/Volumes/RAID54/images_getty/5/54/975648904.jpg"):
#     print("file exists")
# quit()

i = 0

# Open the CSV file in read mode
with open(csv_file, mode='r', newline='') as file:
    # Create a CSV reader object
    reader = csv.reader(file)
    # Iterate over each row in the CSV file
    for row in reader:
        # # Extract the file paths from the first and second columns
        # original_path = row[0]
        # destination_path = row[1]
        if START > i:
            continue

        # this is for moving is_face = NULL files
        original_path = os.path.join(ORIGIN,row[1])
        destination_path = os.path.join(DEST,row[1])


        # Check if the file path exists
        if os.path.exists(original_path):
            # print(f"Copying {original_path} to {destination_path}")
            
            try:
                if not os.path.exists(destination_path):
                    # Copy the file to the destination location, but leave the original file in place
                    # shutil.copy(original_path, destination_path)

                    # Move the file to the destination location, and delete the original file
                    shutil.move(original_path, destination_path)
                    # print(original_path, destination_path)
                    
                    if i % 100 == 0:
                        print(f"{i} Files copied successfully.")
                    i += 1

                    # print(f"File copied successfully.")
                else:
                    print(f"File already exists: {destination_path}")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print(f"File does not exist: {original_path}")
