import hashlib
import os
import shutil
import re
import threading
import queue
import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')

# import file

from mp_db_io import DataIO

io = DataIO()

### THIS IS THE MAIN FILE MOVING UTILITY - USE THIS TO MOVE FROM SSD TO RAID ###

# moves files from folder located at PATH to hash folders created at NEWPATH
# uses get_hash_folders to determine which folder to put in
# if ALL_IN_ONE_FOLDER = True it will look only in the PATH folder (for when all files are in one folder)
# if ALL_IN_ONE_FOLDER = False it will look inside all the folders recursively inside of PATH

# (does not leave original file in place)
# if False it will not delete the original file
MOVE_DELETE_ORIGINAL = False

# testname = "woman-in-a-music-concert-picture-id505111652.jpg"
# # PATH= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production/gettyimages") 

# PATH = "/Volumes/SSD4green/images_shutterstock2"
# NEWPATH = "/Volumes/RAID54/images_shutterstock"

PATH = "/Volumes/SSD4/images_getty_reDL_test_redundant"
NEWPATH = "/Volumes/RAID18/images_getty"

ALL_IN_ONE_FOLDER = False

# folder ="5GB_testimages"
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



# Mutex for thread synchronization
lock = threading.Lock()

# Event for signaling thread completion
threads_completed = threading.Event()

# Queue for distributing work among threads
work_queue = queue.Queue()

def threaded_process_files():
    while not work_queue.empty():
        currentpathfile, newfile, a, b = work_queue.get()
        newpathfile = os.path.join(NEWPATH, a, b, newfile)

        if os.path.exists(newpathfile):
            pass
            # print("file exists, skipping: ", newpathfile)
            # remove newpathfile from hard drive
            # os.remove(newpathfile)
            # print("could remove: ", currentpathfile)
        else:
            if MOVE_DELETE_ORIGINAL:
                shutil.move(currentpathfile, newpathfile)
                print("moved to: ", newpathfile)
            else:
                shutil.copy(currentpathfile,  newpathfile)
                print("copied (w/o deleting) to: ", newpathfile)


        with lock:
            global counter
            counter += 1

        work_queue.task_done()

def threaded_processing():
    thread_list = []
    for _ in range(num_threads):
        thread = threading.Thread(target=threaded_process_files)
        thread_list.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in thread_list:
        thread.join()

    # Set the event to signal that threads are completed
    threads_completed.set()

def add_queue(this_path):
    meta_file_list = get_dir_files(this_path)
    for newfile in sorted(meta_file_list):
        a, b = get_hash_folders(newfile)
        currentpathfile = os.path.join(this_path, newfile)
        work_queue.put((currentpathfile, newfile, a, b))


counter = 0
num_threads = 8  

if ALL_IN_ONE_FOLDER:
    print("looking on only one folder")
    add_queue(PATH)
else:
    print("going to walk folders")
    # Put work into the queue
    for root, dirs, files in os.walk(PATH):
        for folder in sorted(dirs):
            print("looking in ", folder)
            this_path = os.path.join(root, folder)
            add_queue(this_path)

print("going to start threading")
threaded_processing()

# Wait for threads to complete
threads_completed.wait()
