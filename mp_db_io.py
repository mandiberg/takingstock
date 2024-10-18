import os
from sys import platform
import csv
import hashlib
import json
import ast
import pandas as pd
import pickle
import numpy as np
import pymongo
from decimal import Decimal
from pathlib import Path

class DataIO:
    """Store key database and file IO info for use across codebase"""

    def __init__(self, IS_SSD=False):
        self.max_retries = 3
        self.retry_delay = 5
        self.query_face = True
        self.query_hands = True
        self.query_body = True
        self.query_head_pose = True
        # platform specific file folder (mac for michael, win for satyam)
        self.home = Path.home()
        if platform == "darwin":
            self.platform = "darwin"
            ####### Michael's OS X Credentials ########
            # self.db = {
            #     "host":"localhost",
            #     "name":"stock",            
            #     "user":"root",
            #     "pass":"XFZ5dPJq2"
            # }

            ####### Michael's MAMP Credentials ########
            self.db = {
                "host":"localhost",
                "name":"stock",            
                "user":"root",
                "pass":"root",
                "unix_socket":"/Applications/MAMP/tmp/mysql/mysql.sock",
                "raise_on_warnings": True
            }

            # ####### Michael's MAMP Credentials ########
            # self.db = {
            #     "host":"127.0.0.1",
            #     "name":"stock",            
            #     "user":"root",
            #     "pass":"mypassword",
            #     "unix_socket":"",
            #     "raise_on_warnings": True
            # }

            self.dbmongo = {
                "host":"mongodb://localhost:27017/",
                "name":"stock",
                "collection":"encodings"
            }
            self.mongo_client = pymongo.MongoClient(self.dbmongo['host'])
            self.mongo_db = self.mongo_client[self.dbmongo['name']]
            self.mongo_collection_face = self.mongo_db['encodings']
            self.mongo_collection_body = self.mongo_db["body_landmarks_norm"]
            self.mongo_collection_hands = self.mongo_db["hand_landmarks"]

            # self.ROOT_PROD= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production/segment_images") ## only on Mac
            # moved images to SSD
            self.ROOT_PROD=  "/Volumes/OWC4/segment_images" ## only on Mac
            self.ROOTSSD = os.path.join(self.home,"Documents/projects-active/facemap_production")
            self.ROOT54= "/Volumes/RAID54" ## only on 
            # temp migration for
            # self.ROOT54= "/Volumes/6TB_mayday_2" ## only on 
            self.ROOT= self.ROOT_PROD ## defining ROOT though may be redefinied in main()
            self.ROOT4 = "/Volumes/SSD4"
            self.ROOT18 = "/Volumes/RAID18"
            self.NUMBER_OF_PROCESSES = 8
        elif platform == "win32":
            self.platform = "win32"
            ######## Satyam's WIN Credentials #########
            ####DOCKER#################
            # self.db = {
            #     #"host":"localhost",
            #     "host":"127.0.0.1:3333",
            #     "name":"fullstock",                 
            #     "user":"root",
            #     "unix_socket":"",
            #     "pass":"SSJ2_mysql"
            # }

            # self.dbmongo = {
            #     # "host":"mongodb://127.0.0.1:27018/",
            #     "host":"mongodb://SJHA:SSJ2_mongo@127.0.0.1:27018/",
            #     "name":"fullstock",
            #     "collection":"tokens"
            # }
            #############################
            self.db = {
                "host":"127.0.0.1:3306",
                "name":"stock",                 
                "user":"root",
                "unix_socket":"",
                "pass":"SSJ2_mysql"
            }

            self.dbmongo = {
                "host":"mongodb://127.0.0.1:27017/",
                "name":"stock",
                "collection":"tokens"
            }

            # self.ROOT= "E:\\"+"work\\face_map\\Documents\\projects-active\\facemap_production\\" ## SSD
            self.ROOT= "E:/"+"work/face_map/Documents/projects-active/facemap_production/" ## SSD
            self.ROOT54= self.ROOT
            self.ROOT18= self.ROOT
            self.ROOT_PROD= self.ROOT
            self.ROOTSSD = self.ROOT
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
                os.path.join(self.ROOT_PROD,"images_vcg"), # 10 visual china group
                os.path.join(self.ROOT_PROD,"images_pixcy"), # 11	picxy
                os.path.join(self.ROOT_PROD,"images_pixerf"), # 12	pixerf
                os.path.join(self.ROOT_PROD,"images_bazzar"), # 13	imagesbazaar
                os.path.join(self.ROOT_PROD,"images_india"), # 14	indiapicturebudget
                os.path.join(self.ROOT_PROD,"images_iwaria"), # 15	iwaria
                os.path.join(self.ROOT_PROD,"images_nappy"), # 16	nappy
                os.path.join(self.ROOT_PROD,"images_picha"), # 17	picha
                os.path.join(self.ROOT_PROD,"") # 18	afripics
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
                os.path.join(self.ROOT18,"images_vcg"), # visual china group
                os.path.join(self.ROOT18,"images_pixcy"), # 11	picxy
                os.path.join(self.ROOT18,"images_pixerf"), # 12	pixerf
                os.path.join(self.ROOT18,"images_bazzar"), # 13	imagesbazaar
                os.path.join(self.ROOT18,"images_india"), # 14	indiapicturebudget
                os.path.join(self.ROOT18,"images_iwaria"), # 15	iwaria
                os.path.join(self.ROOT18,"images_nappy"), # 16	nappy
                os.path.join(self.ROOT18,"images_picha"), # 17	picha
                os.path.join(self.ROOT18,"") # 18	afripics
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
        img_list = []
        for count, file in enumerate(os.listdir(folder)):
            if not file.startswith('.') and os.path.isfile(os.path.join(folder, file)):
                if not file.endswith(('.csv', '.txt', '.json')):
                    filepath = os.path.join(folder, file)
                    filepath = filepath.replace('\\', '/')
                    img_list.append(file)
        if sort is True:
            img_list.sort()
        print(len(img_list))
        print("got image list")
        return img_list

    def get_existing_image_ids_from_wavs(self,folder):
        existing_files = self.get_img_list(folder)
        existing_image_ids = [int(f.split("_")[0]) for f in existing_files if f.endswith(".wav")]
        return existing_image_ids


    def get_folders(self, folder, sort="alphabetical"):
        sort = sort.lower()  # Convert to lowercase to make the comparison case-insensitive
        print("getting folders", sort)
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        if sort == "alphabetical":
            subfolders.sort()  # Sort alphabetically
            print("sorted alphabetically", subfolders)
        elif sort == "chronological":
            subfolders.sort(key=os.path.getmtime)
            print("sorted by date", subfolders)
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


    def oddify(self,x):
        x = int(x)
        if x % 2 == 0: return x+1
        else: return x


    def convert_decimals_to_float(self, arr):
        return [float(x) if isinstance(x, Decimal) else x for x in arr]


    def get_encodings_mongo(self,image_id):

        # print("self.query_face: ", self.query_face)
        # print("self.query_body: ", self.query_body)
        results_face = results_body = results_hands = None
        if image_id:
            if self.query_face: results_face = self.mongo_collection_face.find_one({"image_id": image_id})
            if self.query_body: results_body = self.mongo_collection_body.find_one({"image_id": image_id})
            if self.query_hands: results_hands = self.mongo_collection_hands.find_one({"image_id": image_id})
            # print("got results from mongo, types are: ", type(results_face), type(results_body))
            # print("results_face: ", results_face)
            # print("results_body: ", results_body)
            face_encodings68 = face_landmarks = body_landmarks = body_landmarks_normalized = None
            if results_body:
                body_landmarks_normalized = results_body["nlms"]
            if results_face:
                try:
                    face_encodings68 = results_face['face_encodings68']
                    face_landmarks = results_face['face_landmarks']
                    body_landmarks = results_face['body_landmarks']
                except KeyError as e:
                    print(f"Error loading face data for image {image_id}: {e}")
            # if results_hands:
            #     print("results_hands: ", results_hands)
                # left_hand = results_hands['left_hand']
                # right_hand = results_hands['right_hand']
                # print("got encodings from mongo, types are: ", type(face_encodings68), type(face_landmarks), type(body_landmarks))
        return pd.Series([face_encodings68, face_landmarks, body_landmarks, body_landmarks_normalized, results_hands])
        #     # else:
        #     #     return pd.Series([None, None, None])
        # else:
        #     return pd.Series([None, None, None, None, None])

    def unpickle_array(self,pickled_array):
        if pickled_array:
            try:
                # Attempt to unpickle using Protocol 3 in v3.7
                return pickle.loads(pickled_array, encoding='latin1')
            except TypeError:
                # If TypeError occurs, unpickle using specific protocl 3 in v3.11
                # return pickle.loads(pickled_array, encoding='latin1', fix_imports=True)
                try:
                    # Set the encoding argument to 'latin1' and protocol argument to 3
                    obj = pickle.loads(pickled_array, encoding='latin1', fix_imports=True, errors='strict', protocol=3)
                    return obj
                except pickle.UnpicklingError as e:
                    print(f"Error loading pickle data: {e}")
                    return None
        else:
            return None

    # should just be in sort pose
    # def get_landmarks_2d(self, Lms, selected_Lms, structure="dict"):
    #     # this is redundantly in sort_pose also
    #     Lms2d = {}
    #     Lms1d = []
    #     Lms1d3 = []
    #     for idx, lm in enumerate(Lms.landmark):
    #         if idx in selected_Lms:
    #             # print("idx", idx)
    #             # x, y = int(lm.x * img_w), int(lm.y * img_h)
    #             # print("lm.x, lm.y", lm.x, lm.y)
    #             if structure == "dict":
    #                 Lms2d[idx] =([lm.x, lm.y])
    #             elif structure == "list":
    #                 Lms1d.append(lm.x)
    #                 Lms1d.append(lm.y)
    #             elif structure == "list3":
    #                 Lms1d3.append(lm.x)
    #                 Lms1d3.append(lm.y)
    #                 Lms1d3.append(lm.visibility)
    #     # print("Lms2d", Lms2d)
    #     # print("Lms1d", Lms1d)

    #     if Lms1d:
    #         return Lms1d
    #     elif Lms1d3:
    #         return Lms1d3
    #     else:
    #         return Lms2d


    def unstring_json(self, json_string):
        eval_string = ast.literal_eval(json_string)
        if isinstance(eval_string, dict):
            return eval_string
        else:
            json_dict = json.loads(eval_string)
            return json_dict
    def make_float(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value

