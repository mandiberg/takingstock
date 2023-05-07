import hashlib
import os
import shutil
from shutil import move
import re
import csv
from time import sleep

from threading import Thread

sig = '''
  _   _ _  _ _  _   _   ___ _  _ ___ _____ 
 | | | | \| | || | /_\ / __| || |_ _|_   _|
 | |_| | .` | __ |/ _ \\__ \ __ || |  | |  
  \___/|_|\_|_||_/_/ \_\___/_||_|___| |_|  
'''

# testname = "https://images.pexels.com/photos/9304005/pexels-photo-9304005.jpeg?auto=compress&cs=tinysrgb&w=1440"
# PATH= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production/gettyimages") 

#where the images are:
PATH = "/Volumes/Stock12/"
#where the images are going:
PATH2 = "/Volumes/Test36/"

COPY=True
UNIQUE_FILES_PATH="/Volumes/Test36/CSVs_to_ingest/pexelsCSVs/unique_images.csv"
IMAGES_THREAD_COUNTER = 0
IMAGES_FOLDER_NAME = 'images_pexels'
NEWIMAGES_FOLDER_NAME = 'images_pexels'
NUMBER_OF_THREADS_IMAGES_DOWNLOAD =15
OLDPATH = os.path.join(PATH, IMAGES_FOLDER_NAME)
NEWPATH = os.path.join(PATH2, NEWIMAGES_FOLDER_NAME)


# startpoint = 622000

#setup alphabet list
#long to crate full directory structure
alphabet = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
# alphabet = '0'  
alphabet2 = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 0'  
# alphabet = 'A B C 0 1 2'   #short alphabet for testing purposes
# alphabet2 = 'A B C 0 1 2'   #short alphabet for testing purposes
alphabet = alphabet.split()
alphabet2 = alphabet2.split()


def get_hash_folders(filename):
    m = hashlib.md5()
    m.update(filename.encode('utf-8'))
    d = m.hexdigest()
    # csvWriter1.writerow(["https://upload.wikimedia.org/wikipedia/commons/"+d[0]+'/'+d[0:2]+'/'+filename])
    return d[0], d[0:2]
# print(get_hash_folders(testname))
    
def touch(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def make_hash_folders(path):
    #create depth 0
    for letter in alphabet:
        # print (letter)
        pth = os.path.join(path,letter)
        touch(pth)
        for letter2 in alphabet2:
            # print (letter2)

            pth2 = os.path.join(path,letter,letter+letter2)
            touch(pth2)

def get_hash_folders(filename):
    m = hashlib.md5()
    m.update(filename.encode('utf-8'))
    d = m.hexdigest()
    return (os.path.join(d[0], d[0:2]), d)

def generate_local_image_filepath(image_name):
    file_name = image_name.split('?')[0]
    extension = file_name.split('.')[-1]
    hash_folder, image_hashed_filename = get_hash_folders(file_name)
    return os.path.join(
        IMAGES_FOLDER_NAME, hash_folder, '{}.{}'.format(image_hashed_filename, extension))

def generate_local_unhashed_image_filepath(image_name):
    file_name_path = image_name.split('?')[0]
    file_name = file_name_path.split('/')[-1]
    extension = file_name.split('.')[-1]
    hash_folder, image_hashed_filename = get_hash_folders(file_name)
    return os.path.join(NEWIMAGES_FOLDER_NAME, hash_folder,file_name)
        # IMAGES_FOLDER_NAME, hash_folder, '{}.{}'.format(file_name, extension))

def unhash_files():
    global IMAGES_THREAD_COUNTER

    def thread(image_hashpath, image_unhashed_path, retry=0):
        global IMAGES_THREAD_COUNTER
        try:
            #THIS IS WHERE I WILL MOVE THE STUFF
            # move(src,dest)
            move(image_hashpath, image_unhashed_path)
            print("moved")
        except:
            if retry < 5:
                thread(image_hashpath, image_unhashed_path, retry+1)
        if IMAGES_THREAD_COUNTER > 0:
            IMAGES_THREAD_COUNTER -= 1

    def read_csv(path):
        with open(path, "r") as f1:
            last_line = f1.readlines()[-1]
        return int(last_line)

    def write_log_csv(path,max_pages):
        headers = ["maxpages"]
        with open(path, 'a') as csvfile: 
            writer=csv.writer(csvfile, delimiter=',')
            writer.writerow([max_pages])

    # check to make sure the old files are actualy there
    try:
        if not os.path.exists(OLDPATH):
            print("[-] No folder here: ", OLDPATH)
            quit()
        #     os.mkdir(IMAGES_FOLDER_NAME)
        # # initialize hash folders
        if not os.path.exists(os.path.join(OLDPATH, "A")):
            print("[-] No folder here: ", OLDPATH)
            quit()
        #     make_hash_folders(IMAGES_FOLDER_NAME)
    except:
        print('[download_images_from_cache] unable to find folder\n', flush=True)

    # check to see if the new folder structure is in place, and if not make it. 
    try:
        if not os.path.exists(NEWPATH):
            os.mkdir(NEWPATH)
        # # initialize hash folders
        if not os.path.exists(os.path.join(NEWPATH, "A")):
            print("no subfolders")
            make_hash_folders(NEWPATH)
            print("just made new hash_folder")
    except:
        print('[download_images_from_cache] unable to create folder\n', flush=True)

    if not os.path.exists(UNIQUE_FILES_PATH):
        print('[-] cache `%s` not found.')
        exit(0)

    with open(UNIQUE_FILES_PATH, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Loop over each row in the file
        start_counter = 570000
        counter = start_counter
        alreadyDL = 0
        print("starting from start_counter: ",start_counter)

        for i in range(start_counter):
            next(reader)  # skip the line

        # print('starting to traverse the file, starting from: ',str(start_counter))
        for row in reader:
            # obj = json.loads(item)
            # print(counter)
            # while start_counter > counter:counter is:  
            #     # print("in while")
            #     counter += 1
            #     print("skipping, ",counter)
            #     continue

            if row[0] is None:
                continue
            # if startpoint > 0 and startpoint > counter:
            #     continue

            # this stores images in hashed folders, to ensure reasonable
            # number of files per folder
            image_url = row[0]
            # where the old images is (PATH)
            image_hashpath = os.path.join(PATH,generate_local_image_filepath(image_url))
            # where the new images goes (PATH2)
            image_unhashed_path = os.path.join(PATH2,generate_local_unhashed_image_filepath(image_url.replace('.jpeg','.jpg')))
            # print out to countout every 1000 batches
            if start_counter % 10 == 0:
                print("start_counter is: ",start_counter)
            start_counter += 1

            if counter % 1000 == 0 and counter > start_counter:
                print("counter is: ",counter)
                # write_log_csv(CSV_COUNTOUT_PATH,counter)

            if os.path.isfile(image_hashpath):
                counter += 1
                print("this file will be moved", str(counter), image_hashpath)
            else:
                alreadyDL += 1
                print("nobody there", str(alreadyDL), image_hashpath)
                continue
            if IMAGES_THREAD_COUNTER < NUMBER_OF_THREADS_IMAGES_DOWNLOAD:
                Thread(target=thread, args=[
                       image_hashpath, image_unhashed_path], daemon=True).start()
                IMAGES_THREAD_COUNTER += 1
            # print("IMAGES_THREAD_COUNTER ",str(IMAGES_THREAD_COUNTER))
            while IMAGES_THREAD_COUNTER >= NUMBER_OF_THREADS_IMAGES_DOWNLOAD:
                sleep(.1)
                print('[-] Processing batch #%s' %
                      (int(counter / NUMBER_OF_THREADS_IMAGES_DOWNLOAD)), end='\r')
                      # (int(counter / NUMBER_OF_THREADS_IMAGES_DOWNLOAD)),  (int(total_count / NUMBER_OF_THREADS_IMAGES_DOWNLOAD)), end='\r')
            counter += 1
    print('[-] All images have been downloaded successfully\n')




if __name__ == '__main__':
    print(sig)
    try:
        unhash_files()
    except KeyboardInterrupt as _:
        print('[-] User cancelled.\n', flush=True)
    except Exception as e:
        print('[__main__] %s' % str(e), flush=True)
