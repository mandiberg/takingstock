import hashlib
import os
import shutil
import re
import threading
import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')

# import file

from mp_db_io import DataIO

io = DataIO()

# moves files from folder located at PATH to hash folders created at NEWPATH
# uses get_hash_folders to determine which folder to put in
# if ALL_IN_ONE_FOLDER = True it will look only in the PATH folder 
# if ALL_IN_ONE_FOLDER = False it will look inside all the folders recursively inside of PATH
# (does not leave original file in place)

testname = "woman-in-a-music-concert-picture-id505111652.jpg"
# PATH= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production/gettyimages") 
PATH = "/Volumes/RAID54/adobeStockScraper_v3/test1"
NEWPATH = "/Volumes/RAID54/adobeStockScraper_v3/dest1"

# PATH = "/Volumes/RAID54/adobeStockScraper_v3/images_doover_4round_july5"
# NEWPATH = "/Volumes/RAID54/adobeStockScraper_v3/images_adobe"

ALL_IN_ONE_FOLDER = False

# folder ="5GB_testimages"
COPY=True
# CSV="/Users/michaelmandiberg/Dropbox/facemap_dropbox/test_data/Images_202302101516_30K.csv"

def get_hash_folders(filename):
    m = hashlib.md5()
    m.update(filename.encode('utf-8'))
    d = m.hexdigest()
    # csvWriter1.writerow(["https://upload.wikimedia.org/wikipedia/commons/"+d[0]+'/'+d[0:2]+'/'+filename])
    return d[0], d[0:2]
# print(get_hash_folders(testname))

# def get_csv_files(CSV):
    
def get_dir_files(folder):
    # counter = 1

    # directory = folder
    directory = os.path.join(folder)
    # print(directory)

    meta_file_list = []
    try:
        os.chdir(directory)
        # print(len(os.listdir(directory)))
        for filename in os.listdir(directory):
        # for item in os.listdir(root):
            # print (counter)

            if not filename.startswith('.') and os.path.isfile(os.path.join(directory, filename)):
                meta_file_list.append(filename)

    except Exception as e:
        raise e
    return meta_file_list


def touch(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def make_hash_folders():
    # basepath = '/new_images_pexels'

    #setup alphabet list
    #long to crate full directory structure
    alphabet = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
    alphabet2 = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
    # alphabet = 'A B C 0 1 2'   #short alphabet for testing purposes
    # alphabet2 = 'A B C 0 1 2'   #short alphabet for testing purposes
    alphabet = alphabet.split()
    alphabet2 = alphabet2.split()

    #helper variable for determining what depth you are at
    # c_depth = alphabet

    #create depth 0
    for letter in alphabet:
        # print (letter)
        pth = os.path.join(NEWPATH,letter)
        touch(pth)
        for letter2 in alphabet2:
            # print (letter2)

            pth = os.path.join(NEWPATH,letter,letter+letter2)
            touch(pth)



#touch all new folders (once done once, can comment out)
make_hash_folders()

#loop through all existing folders
# basepath = '/images_pexels'

#setup alphabet list
#long to crate full directory structure
alphabet = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
# alphabet = '0'  
alphabet2 = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
# alphabet = 'A B C 0 1 2'   #short alphabet for testing purposes
# alphabet2 = 'A B C 0 1 2'   #short alphabet for testing purposes
alphabet = alphabet.split()
alphabet2 = alphabet2.split()

#helper variable for determining what depth you are at
# c_depth = alphabet

counter = 0


# for unsorted images, all in one folder



def process_files_in_folder(folder):
    print("going to get get_img_list")
    meta_file_list = io.get_img_list(folder, sort=False)
    print(len(meta_file_list))
    counter = 0
    for newfile in meta_file_list:
        # print(newfile)
        a, b = get_hash_folders(newfile)
        currentpathfile = os.path.join(folder, newfile)
        newpathfile = os.path.join(NEWPATH, a, b, newfile)

        # print(currentpathfile, newpathfile)

        shutil.move(currentpathfile, newpathfile)

        print("moved from: ", currentpathfile)
        print("moved to: ", newpathfile)
        print(counter)
        counter = counter + 1

if ALL_IN_ONE_FOLDER:
    process_files_in_folder(PATH)
else:
    for root, dirs, files in os.walk(PATH):
        for folder in dirs:
            process_files_in_folder(os.path.join(root, folder))




