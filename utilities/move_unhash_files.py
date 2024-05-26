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
PATH = "/Volumes/RAID18/images_unsplash"
#where the images are going:
PATH2 = "/Volumes/RAID18/images_unsplash_un/"

COPY=True
UNIQUE_FILES_PATH="/Volumes/RAID54/process_CSVs_to_ingest/unsplashCSVs/unsplash.output.csv"
NEW_UNIQUE_FILES_PATH="/Volumes/RAID54/process_CSVs_to_ingest/unsplashCSVs/unsplash.output_new.csv"
IMAGES_THREAD_COUNTER = 0
IMAGES_FOLDER_NAME = 'images'
NEWIMAGES_FOLDER_NAME = 'new_images'
NUMBER_OF_THREADS_IMAGES_DOWNLOAD =15
OLDPATH = os.path.join(PATH, IMAGES_FOLDER_NAME)
NEWPATH = os.path.join(PATH2, NEWIMAGES_FOLDER_NAME)


startpoint = 340000

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

# def get_hash_folders(filename):
#     m = hashlib.md5()
#     m.update(filename.encode('utf-8'))
#     d = m.hexdigest()
#     return (os.path.join(d[0], d[0:2]), d)

def generate_local_image_filepath(image_name):
    file_name_path = image_name.split('?')[0]
    file_name = file_name_path.split('/')[-1]
    extension = file_name.split('.')[-1]
    hash_folder, image_hashed_filename = get_hash_folders(file_name)
    print(hash_folder, image_hashed_filename, file_name, extension)
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
            move(image_hashpath, image_unhashed_path)
            # print(image_hashpath, image_unhashed_path)
            # print("moved")
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

    # with open(CSV_OUTPUT_PATH, 'w', newline='', encoding="utf-8") as f:
    #     writer = csv.DictWriter(
    #         f, fieldnames=headers, delimiter=',', quoting=csv.QUOTE_ALL, doublequote=False)
    #     writer.writeheader()

    # with open(CSV_OUTPUT_PATH, 'a', encoding="utf-8", newline='') as output:
    #     with open(QUERIES_CACHE_PATH, 'r') as cache_file:
    #         writer = csv.writer(
    #             output, delimiter=',', quoting=csv.QUOTE_ALL, doublequote=False)
    #         for item in cache_file.readlines():
    #             parsed_obj = parser_item(item)
    #             writer.writerow(parsed_obj)

    ## this is restructured to write a fresh csv file with the updated filename, configd for pond5
    headers = ["id", "title", "keywords", "number_of_people", "orientation", "age",
               "gender", "ethnicity", "mood", "image_url", "image_filename"]
    with open(NEW_UNIQUE_FILES_PATH, 'w', newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=headers, delimiter=',', quoting=csv.QUOTE_ALL, doublequote=False)
        writer.writeheader()


    with open(NEW_UNIQUE_FILES_PATH, 'a', encoding="utf-8", newline='') as output:
        with open(UNIQUE_FILES_PATH, 'r') as csvfile:
            reader = csv.reader(csvfile)
            writer = csv.writer(output, delimiter=',', quoting=csv.QUOTE_ALL, doublequote=False)

            # Loop over each row in the file
            start_counter = 0
            counter = start_counter
            alreadyDL = 0
            print("starting from start_counter: ",start_counter)

            # for i in range(start_counter):
            #     next(reader)  # skip the line

            # print('starting to traverse the file, starting from: ',str(start_counter))
            for row in reader:
                # obj = json.loads(item)
                # print(counter)
                # while start_counter > counter:counter is:  
                #     # print("in while")
                #     counter += 1
                #     print("skipping, ",counter)
                #     continue

                if row[0] is None or startpoint > start_counter:
                    start_counter += 1
                    continue
                # if startpoint > 0 and startpoint > start_counter:
                #     continue

                # this stores images in hashed folders, to ensure reasonable
                # number of files per folder
                image_name = row[0]
                # if pulling an image_id, it needs a .jpg suffix:
                image_name = image_name+".jpg"
                # where the old images is (PATH)
                image_hashpath = os.path.join(PATH, row[11].replace('images/',''))
                # image_hashpath = os.path.join(PATH,generate_local_image_filepath(image_name))
                # where the new images goes (PATH2)
                # this pulls an image_name created from image_id and gets hash folders from it.
                f1, f2 = get_hash_folders(image_name)
                image_filename = os.path.join(f1,f2,image_name)
                image_unhashed_path = os.path.join(PATH2,NEWIMAGES_FOLDER_NAME, image_filename)

                # image_unhashed_path = os.path.join(PATH2,generate_local_unhashed_image_filepath(image_name.replace('.jpeg','.jpg')))
                # print out to countout every 1000 batches
                # print(image_hashpath, image_unhashed_path)
                # continue
                if start_counter % 200 == 0:
                    print("start_counter is: ",start_counter)
                start_counter += 1
                counter += 1

                if counter % 1000 == 0 and counter > start_counter:
                    print("counter is: ",counter)
                    # write_log_csv(CSV_COUNTOUT_PATH,counter)

                if os.path.isfile(image_hashpath):
                    row[10]=image_filename
                    writer.writerow(row)
                    # print("this file will be moved", str(counter), image_hashpath, image_unhashed_path)
                else:
                    alreadyDL += 1
                    print("nobody there", str(alreadyDL), row[10])
                    continue
                # continue # for testing
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
    print('[-] All images have been downloaded successfully\n')




if __name__ == '__main__':
    print(sig)
    try:
        unhash_files()
    except KeyboardInterrupt as _:
        print('[-] User cancelled.\n', flush=True)
    except Exception as e:
        print('[__main__] %s' % str(e), flush=True)
