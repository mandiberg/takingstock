import os
from sys import platform
import csv
import hashlib
import json
import ast


class DataIO:
    """Store key database and file IO info for use across codebase"""

    def __init__(self, IS_SSD=False):
        self.max_retries = 3
        self.retry_delay = 5
        # platform specific file folder (mac for michael, win for satyam)
        if platform == "darwin":
            ####### Michael's OS X Credentials ########
            # self.db = {
            #     "host":"localhost",
            #     "name":"stock",            
            #     "user":"root",
            #     "pass":"XFZ5dPJq2"
            # }

            # ####### Michael's MAMP Credentials ########
            # self.db = {
            #     "host":"localhost",
            #     "name":"stock",            
            #     "user":"root",
            #     "pass":"root",
            #     "unix_socket":"/Applications/MAMP/tmp/mysql/mysql.sock",
            #     "raise_on_warnings": True
            # }

            ####### Michael's MAMP Credentials ########
            self.db = {
                "host":"127.0.0.1",
                "name":"stock",            
                "user":"root",
                "pass":"mypassword",
                "unix_socket":"",
                "raise_on_warnings": True
            }

            self.dbmongo = {
                "host":"mongodb://localhost:27017/",
                "name":"stock",
                "collection":"encodings"
            }

            self.ROOT_PROD= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production/segment_images") ## only on Mac
            self.ROOT54= "/Volumes/RAID54" ## only on 
            # temp migration for
            # self.ROOT54= "/Volumes/6TB_mayday_2" ## only on 
            self.ROOT= self.ROOT_PROD ## defining ROOT though may be redefinied in main()
            self.ROOT4 = "/Volumes/SSD4"
            self.ROOT18 = "/Volumes/RAID18"
            self.NUMBER_OF_PROCESSES = 8
        elif platform == "win32":
            ######## Satyam's WIN Credentials #########
            self.db = {
                #"host":"localhost",
                "host":"127.0.0.1:3333",
                "name":"ministock",                 
                "user":"root",
                "unix_socket":"",
                "pass":"SSJ2_mysql"
            }

            self.dbmongo = {
                "host":"mongodb://localhost:27017/",
                "name":"test2",
                "collection":"encodings3"
            }

            # self.ROOT= "E:\\"+"work\\face_map\\Documents\\projects-active\\facemap_production\\" ## SSD
            self.ROOT= "E:/"+"work/face_map/Documents/projects-active/facemap_production/" ## SSD
            self.ROOT54= self.ROOT
            self.ROOT18= self.ROOT
            self.ROOT_PROD= self.ROOT
            self.NUMBER_OF_PROCESSES = 4

        if IS_SSD:
            self.folder_list = [
                "", #0, Empty, there is no site #0 -- starts count at 1
                os.path.join(self.ROOT_PROD,"images_getty"), #1, Getty
                # temp for testing
                # os.path.join(self.ROOT54,"gettyimages/testimages"), #1, Getty
                os.path.join(self.ROOT_PROD,"images_shutterstock"), #2, Shutterstock
                os.path.join(self.ROOT_PROD,"images_adobe"), #3, Adobe
                os.path.join(self.ROOT_PROD,"images_istock"), #4, iStock
                os.path.join(self.ROOT_PROD,"images_pexels"), #5, Pexels
                os.path.join(self.ROOT_PROD,"images_unsplash"),
                os.path.join(self.ROOT_PROD,"images_pond5"),
                os.path.join(self.ROOT_PROD,"images_123rf"), #8, images_123rf
                os.path.join(self.ROOT_PROD,"images_alamy"), #9 alamy
                os.path.join(self.ROOT_PROD,"images_vcg") # visual china group
            ]
        else:
            self.folder_list = [
                "", #0, Empty, there is no site #0 -- starts count at 1
                os.path.join(self.ROOT18,"images_getty"), #1, Getty
                # "/Volumes/SSD4/images_getty_reDL", #1, Getty TEMP
                # temp for testing
                # os.path.join(self.ROOT54,"gettyimages/testimages"), #1, Getty
                os.path.join(self.ROOT54,"images_shutterstock"),
                os.path.join(self.ROOT54,"images_adobe"), #3, Adobe
                os.path.join(self.ROOT18,"images_istock"), #4, iStock
                os.path.join(self.ROOT18,"images_pexels"), #5, Pexels
                os.path.join(self.ROOT18,"images_unsplash"),
                os.path.join(self.ROOT18,"images_pond5"),
                os.path.join(self.ROOT54,"images_123rf"), #8, images_123rf
                os.path.join(self.ROOT18,"images_alamy"), #9 alamy
                os.path.join(self.ROOT18,"images_vcg") # visual china group
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

    def get_csv_aslist(self,CSV_COUNTOUT_PATH):
        list_of_lines = []
        try:
            print("trying to get list of saved")
            with open(CSV_COUNTOUT_PATH, encoding="utf-8", newline="") as in_file:
                reader = csv.reader(in_file, delimiter=",")
                # next(reader)  # Header row

                for row in reader:
                    list_of_lines.append(row[0])
                    # yield row
        except:
            print('[get_csv_aslist] something is wrong here')
        return list_of_lines


    def write_csv(self,path,value_list):

        # headers = header_list

        with open(path, 'a') as csvfile: 
        # with open('lat_lon', 'w') as csvfile:
            writer=csv.writer(csvfile, delimiter=',')
            writer.writerow(value_list)

    def get_img_list(self, folder, sort=True):
        img_list=[]
        for count,file in enumerate(os.listdir(folder)):
            if not file.startswith('.') and os.path.isfile(os.path.join(folder, file)):
                filepath = os.path.join(folder, file)
                filepath=filepath.replace('\\' , '/')
                img_list.append(file)
        if sort is True:
            img_list.sort()
        print(len(img_list))
        print("got image list")
        return img_list    



    def get_folders(self,folder):
        subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
        return subfolders

    def get_hash_folders(self,filename):
        m = hashlib.md5()
        m.update(filename.encode('utf-8'))
        d = m.hexdigest()
        return d[0].upper(), d[0:2].upper()

    def touch(self,folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def make_hash_folders(self,path, as_list=False):
        #create depth 0
        alphabet = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9'  
        # alphabet = '0'  
        alphabet2 = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9'  
        # alphabet = 'A B C 0 1 2'   #short alphabet for testing purposes
        # alphabet2 = 'A B C 0 1 2'   #short alphabet for testing purposes
        alphabet = alphabet.split()
        alphabet2 = alphabet2.split()

        if as_list is False:
            for letter in alphabet:
                # print (letter)
                pth = os.path.join(path,letter)
                self.touch(pth)
                for letter2 in alphabet2:
                    # print (letter2)

                    pth2 = os.path.join(path,letter,letter+letter2)
                    self.touch(pth2)
        elif as_list is True:
            folder_paths = []
            for letter in alphabet:
                for letter2 in alphabet2:
                    path = os.path.join(letter,letter+letter2)
                    folder_paths.append(path)
            return folder_paths

    def unstring_json(self, json_string):
        eval_string = ast.literal_eval(json_string)
        if isinstance(eval_string, dict):
            return eval_string
        else:
            json_dict = json.loads(eval_string)
            return json_dict


