import os
from sys import platform
import csv
import hashlib


class DataIO:
    """Store key database and file IO info for use across codebase"""

    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 0.3

        # platform specific file folder (mac for michael, win for satyam)
        if platform == "darwin":
            ####### Michael's OS X Credentials ########
            self.db = {
                "host":"localhost",
                "name":"stock",            
                "user":"root",
                "pass":"XFZ5dPJq2"
            }
            self.ROOT= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production") ## only on Mac
            self.ROOT36= "/Volumes/Test36" ## only on Mac
            self.NUMBER_OF_PROCESSES = 8
        elif platform == "win32":
            ######## Satyam's WIN Credentials #########
            self.db = {
                "host":"localhost",
                "name":"gettytest3",                 
                "user":"root",
                "pass":"SSJ2_mysql"
            }
            self.ROOT= os.path.join("D:/"+"Documents/projects-active/facemap_production") ## SD CARD
            self.NUMBER_OF_PROCESSES = 4


        self.folder_list = [
            "", #0, Empty, there is no site #0 -- starts count at 1
            os.path.join(self.ROOT,"gettyimages/newimages"), #1, Getty
            os.path.join(self.ROOT36,""),
            os.path.join(self.ROOT36,""),
            os.path.join(self.ROOT36,""),
            os.path.join(self.ROOT36,"images_pexels"), #5, Pexels
            os.path.join(self.ROOT36,""),
            os.path.join(self.ROOT36,""),
            os.path.join(self.ROOT36,"images_123rf"), #8, images_123rf
            os.path.join(self.ROOT36,""),
        ]

    def capitalize_directory(self,path):
        dirname, filename = os.path.split(path)
        parts = dirname.split('/')
        capitalized_parts = [part if i < len(parts) - 2 else part.upper() for i, part in enumerate(parts)]
        capitalized_dirname = '/'.join(capitalized_parts)
        return os.path.join(capitalized_dirname, filename)

    def get_counter(self,CSV_COUNTOUT_PATH):
        # read last completed file
        try:
            print("trying to get last saved")
            with open(CSV_COUNTOUT_PATH, "r") as f1:
                last_line = f1.readlines()[-1]
            # last_line = read_csv(CSV_COUNTOUT_PATH)
            # print(type(last_line))
            start_counter = int(last_line)
        except:
            start_counter = 0
            print('[download_images_from_cache] set max_element to 0 \n', flush=True)
            print("max_element,", start_counter)
        return start_counter


    def write_csv(self,path,value_list):

        # headers = header_list

        with open(path, 'a') as csvfile: 
        # with open('lat_lon', 'w') as csvfile:
            writer=csv.writer(csvfile, delimiter=',')
            writer.writerow(value_list)


    def get_hash_folders(self,filename):
        m = hashlib.md5()
        m.update(filename.encode('utf-8'))
        d = m.hexdigest()
        return d[0].upper(), d[0:2].upper()

    def touch(self,folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def make_hash_folders(self,path):
        #create depth 0
        for letter in alphabet:
            # print (letter)
            pth = os.path.join(path,letter)
            touch(pth)
            for letter2 in alphabet2:
                # print (letter2)

                pth2 = os.path.join(path,letter,letter+letter2)
                touch(pth2)
