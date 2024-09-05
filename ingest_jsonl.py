from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import csv
import json
import os
import hashlib
import cv2
import math
import pickle
import sys # can delete for production
import pathlib
import re
import traceback
from collections import Counter
from retrying import retry
from datetime import datetime

import numpy as np
import pandas as pd
from pyinflect import getInflection

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
# my ORM
from my_declarative_base import Base, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Images, ImagesEthnicity, ImagesKeywords

from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, String, VARCHAR, Float, ForeignKey, Date, update, insert, select, PrimaryKeyConstraint, UniqueConstraint
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.sql import compiler
from sqlalchemy.dialects import mysql

#mine
from mp_db_io import DataIO

sig = '''

_)                       |                     
 | __ \   _` |  _ \  __| __|   __|  __|\ \   / 
 | |   | (   |  __/\__ \ |    (   \__ \ \ \ /  
_|_|  _|\__, |\___|____/\__| \___|____/  \_/   
        |___/                                  
'''


######## Michael's Credentials ########
# platform specific credentials
io = DataIO()
db = io.db
# overriding DB for testing
# io.db["name"] = "stocktest"
io.db["name"] = "stock"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#######################################

'''
1   getty
2   shutterstock
3   adobe
4   istock
5   pexels
6   unsplash
7   pond5
8   123rf
9   alamy
10  visualchinagroup
11	picxy
12	pixerf
13	imagesbazaar
14	indiapicturebudget
15	iwaria
16	nappy
17	picha
18	afripics
'''

THIS_SITE = 1

SEARCH_KEYS_FOR_LOC = True
VERBOSE = False
TIMER = False
LOC_ONLY = True
CSV_TYPE = None

# where key, etc csvs stored
INGEST_ROOT = "/Users/michaelmandiberg/Documents/GitHub/facemap/utilities/keys/"
eth_dict_per_site = {}

if THIS_SITE == 1:
    # LOC 2
    # INGEST_ROOTG = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/done"
    # INGEST_FOLDER = os.path.join(INGEST_ROOTG, "unfinished_serbia")
    # JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache.jsonl")

    INGEST_FOLDER = "/Users/michaelmandiberg/Documents/GitHub/facemap/"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "getty_metas_id_loc.jsonl")

elif THIS_SITE == 2:
    # testing round 2
    INGEST_FOLDER = "/Volumes/RAID54/process_CSVs_to_ingest/shutterstockCSVs/Shutterstock_Caches-bup/less_1M_done"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache.jsonl")
    LOC_ONLY = False
elif THIS_SITE == 3:
    # LOC 2
    INGEST_FOLDER = "/Users/michaelmandiberg/Documents/GitHub/facemap/"
    # INGEST_FOLDER = "/Volumes/RAID18/scraping_process/process_adobe/output.jsonl"
    # also at /Volumes/RAID18/scraping_process/process_adobe/output.jsonl
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "output.jsonl")
    # JSONL_IN_PATH = "/Volumes/RAID18/scraping_process/process_adobe/adobe_all_jsonl_files/3400/3400alpha2.jsonl"
elif THIS_SITE == 4:
    # LOC 2
    INGEST_FOLDER = "/Volumes/RAID54/process_CSVs_to_ingest/iStock"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "april15_iStock_output.jsonl")
elif THIS_SITE == 5:
    # LOC 2 as CSV
    INGEST_FOLDER = "/Volumes/RAID54/process_CSVs_to_ingest/pexelsCSVs"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "pexels.output.csv")
    INPUT_TYPE = "csv"
elif THIS_SITE == 6:
    # done -- only 1488334?
    # LOC 2 (no keys, so no loc)
    io.db["name"] = "stock"
    INGEST_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/unsplashCSVs"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache.jsonl")
elif THIS_SITE == 7:
    # LOC 2
    INGEST_FOLDER = "/Volumes/RAID18/scraping_process/process_scraping process_deletewhendoneDL/Pond5"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache.jsonl")
elif THIS_SITE == 8:
    # LOC 2 as CSV
    CSV_TYPE = "123rf"
    INGEST_FOLDER = "/Volumes/RAID54/process_CSVs_to_ingest/123rfCSVs"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "123rf.output.csv")
elif THIS_SITE == 9:
    # DONE 6150898
    # LOC_ONLY'd
    io.db["name"] = "stock"
    INGEST_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/alamyCSV"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache_translated.jsonl")
    # JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache_funny1.jsonl")
elif THIS_SITE == 10:
    # DONE 504250
    # LOC 2
    io.db["name"] = "stock"
    INGEST_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/VCG2"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache_translated.jsonl")
elif THIS_SITE == 11:
    io.db["name"] = "stock"
    # DONE 426693
    # LOC 2
    INGEST_FOLDER = "/Users/michaelmandiberg/Downloads/pixcy_v2"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache_filtered.jsonl")
elif THIS_SITE == 12:
    # DONE 16854
    # LOC 2
    io.db["name"] = "stock"
    INGEST_FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/PIXERF"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache_translated.jsonl")
elif THIS_SITE == 13:
    # DONE 305058
    # LOC 2
    INGEST_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/ImagesBazzar"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache.jsonl")
    eth_dict_per_site = {'asian ethnicity':9, 'tika':9, 'alta':9, 'asian  ethnicity':9,'asian ethinicity':9,'asian farmer':9,'asian medicine':9,'asians':9,'asiascher':9,'asiatisch':9,'asiatische':9,'asiatischen':9,'asiatischer':9,'asiatisches':9,'asien':9}
elif THIS_SITE == 14:
    # DONE 49176
    # LOC 2
    INGEST_FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/INDIA-PB"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache.jsonl")
    eth_dict_per_site = {'asian ethnicity':9, 'tika':9, 'alta':9, 'asian  ethnicity':9,'asian ethinicity':9,'asian farmer':9,'asian medicine':9,'asians':9,'asiascher':9,'asiatisch':9,'asiatische':9,'asiatischen':9,'asiatischer':9,'asiatisches':9,'asien':9}
elif THIS_SITE == 15:
    # DONE 4385
    # LOC 2
    io.db["name"] = "stock"
    INGEST_FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/iwaria"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache_translated.jsonl")
elif THIS_SITE == 16:
    # DONE 2209
    # LOC 2
    io.db["name"] = "stock"
    INGEST_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/nappy_v3_w-data"
    # JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache_translated.jsonl")
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache.jsonl")
elif THIS_SITE == 17:
    # DONE 25663
    # LOC 2
    io.db["name"] = "stock"
    INGEST_FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/PICHA-STOCK"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache.jsonl")
elif THIS_SITE == 18:
    # DONE 60300
    # LOC 2
    io.db["name"] = "stock"
    INGEST_FOLDER = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/AFRIPICS"
    JSONL_IN_PATH = os.path.join(INGEST_FOLDER, "items_cache_translated.jsonl")


# key, etc csvs
KEYWORD_PATH = os.path.join(INGEST_ROOT, "Keywords_202408151415.csv")
LOCATION_PATH = os.path.join(INGEST_ROOT, "Location_202308041952.csv")
CSV_KEY2LOC_PATH = os.path.join(INGEST_ROOT, "CSV_KEY2LOC.csv")
CSV_KEY2KEY_GETTY_PATH = os.path.join(INGEST_ROOT, "CSV_KEY2KEY_GETTY.csv")
CSV_KEY2KEY_VCG_PATH = os.path.join(INGEST_ROOT, "CSV_KEY2KEY_VCG.csv")
CSV_ETH2_GETTY = os.path.join(INGEST_ROOT, "CSV_ETH_GETTY.csv")
CSV_AGE_DICT_PATH = os.path.join(INGEST_ROOT, "CSV_AGE_DICT.csv")
CSV_GENDER_DICT_PATH = os.path.join(INGEST_ROOT, "CSV_GENDER_DICT.csv")
CSV_GENDER_DICT_TNB_PATH = os.path.join(INGEST_ROOT, "CSV_GENDER_DICT_TNB.csv")
CSV_ETH_MULTI_PATH = os.path.join(INGEST_ROOT, "CSV_ETH_MULTI.csv")
CSV_AGE_DETAIL_DICT_PATH = os.path.join(INGEST_ROOT, "CSV_AGE_DETAIL_DICT.csv")

# output csvs
CSV_NOKEYS_PATH = os.path.join(INGEST_FOLDER, "CSV_NOKEYS.csv")
CSV_IMAGEKEYS_PATH = os.path.join(INGEST_FOLDER, "CSV_IMAGEKEYS.csv")
# NEWIMAGES_FOLDER_NAME = 'images_pexels'
CSV_COUNTOUT_PATH = os.path.join(INGEST_FOLDER, "countout.csv")
CSV_NOLOC_PATH = os.path.join(INGEST_FOLDER, "CSV_NOLOC.csv")
CSV_BLANKLOC_PATH = os.path.join(INGEST_FOLDER, "CSV_BLANKLOC_PATH.csv")
CSV_ETH2_PATH = os.path.join(INGEST_FOLDER, "CSV_ETH2.csv")

# open and read a csv file, and assign each row as an element in a list
def dict_from_csv(file_path):
    this_dict = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # print(row)
            # CSV has two columns: key and int value
            key = row[0].strip()
            try:
                value = int(row[1].strip())
            except:
                value = row[1].strip()
            this_dict[key] = value
    return this_dict


loc2loc = {"niue":"Niue Island", "east timor":"timor-leste"}
# key2key = {}

key2loc = dict_from_csv(CSV_KEY2LOC_PATH)
skip_keys = ["other", "easy resource", "suggestive filter", "unspecified", "diffrential focus", "internal term 1", "birds eye view", "bird's eye view", "los banos", "scoot", "angel fire"]
key2key = dict_from_csv(CSV_KEY2KEY_GETTY_PATH)
key2key_vcg = dict_from_csv(CSV_KEY2KEY_VCG_PATH)
age_dict = dict_from_csv(CSV_AGE_DICT_PATH)
gender_dict = dict_from_csv(CSV_GENDER_DICT_PATH)

gender_dict_shutter_secondary = {}
# gender_dict_both = {}
gender_dict_TNB = dict_from_csv(CSV_GENDER_DICT_TNB_PATH)

eth_dict = dict_from_csv(CSV_ETH2_GETTY)
pond5_spacedeth = ['american indian', 'asian woman', 'black woman', 'asian man', 'indian man', 'african man', 'african woman', 'black man', 'indian woman', 'hispanic woman', 'hispanic man', 'chinese man', 'latino man', 'japanese man', 'vietnam woman']
multi_dict = dict_from_csv(CSV_ETH_MULTI_PATH)

age_details_dict = dict_from_csv(CSV_AGE_DETAIL_DICT_PATH)

def lower_dict(this_dict):
    lower_dict = {k.lower(): v for k, v in this_dict.items()}
    return lower_dict

gender_dict_secondary = gender_dict = lower_dict({**gender_dict, **gender_dict_TNB})
# gender_dict_secondary = lower_dict({**gender_dict, **gender_dict_shutter_secondary})
eth_all_dict = eth_dict = lower_dict({**eth_dict, **eth_dict_per_site, **multi_dict})
eth_set = set(eth_dict.keys())
# key2key = lower_dict({**key2key, **key2key_getty})
key2key_set = set(key2key)
# eth_set = set(eth_all_dict.keys())
gender_all_dict = lower_dict({**gender_dict, **gender_dict_secondary})
gender_set = set(gender_all_dict.keys())
age_set = set(age_dict.keys())
# gender_set = set(gender_all_dict.keys())
multi_eth_list = [k for k,v in eth_all_dict.items() if v == 6]
TNB_list = [k.lower() for k,v in gender_all_dict.items() if v == 5 or v == 7]
all_dict = lower_dict({**eth_all_dict, **gender_all_dict, **key2loc, **age_dict, **key2key})
all_set = set(all_dict.keys())          
print("TNB_list", TNB_list)

# for searching descrption for eth keywords, get rid of ambiguous/polyvalent terms
eth_keys_dict = eth_dict


# engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
#                                 .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


PEXELS_HEADERS = ["id", "title", "keywords", "country", "number_of_people", "orientation", "age","gender", "ethnicity", "mood", "image_url", "image_filename"]
ONETWOTHREE_HEADERS = ["id","title","keywords","orientation","age","word","exclude","people","ethnicity","image_url","image_filename"]
KEYWORD_HEADERS = ["keyword_id","keyword_number","keyword_text","keytype","weight","parent_keyword_id","parent_keyword_text"]
IMG_KEYWORD_HEADERS = ["site_image_id","keyword_text"]
header_list = PEXELS_HEADERS

# get all inserted rows for this site
with engine.connect() as conn:

    select_stmt = select(Images.image_id, Images.site_image_id, Images.location_id).where(
        (Images.site_name_id == THIS_SITE)
    )
    # select all the rows
    results = conn.execute(select_stmt).fetchall()
    # make a dict of the image_id and site_image_id
    site_image_id_dict = {row[1]: row[0] for row in results}
    # make a set of all the site_image_ids
    # already_ingested = set([row[1] for row in results])
    if LOC_ONLY:
        already_ingested = set([row[1] for row in results if row[2] is not None])
    else:
        already_ingested = set(site_image_id_dict.keys())
if VERBOSE: print("already_ingested", len(already_ingested))
if already_ingested: print("site_image_id_dict first", site_image_id_dict[next(iter(site_image_id_dict))])

def read_csv(csv_file):
    with open(csv_file, encoding="utf-8", newline="") as in_file:
        reader = csv.reader(in_file, delimiter=",")
        next(reader)  # Header row

        for row in reader:
            yield row

def write_csv(path,value_list):
    # headers = header_list
    with open(path, 'a') as csvfile: 
        writer=csv.writer(csvfile, delimiter=',')
        writer.writerow(value_list)

def init_csv(path, headers):
    csvpath_isfile = pathlib.Path(path)
    if not csvpath_isfile.is_file():
        write_csv(path, headers)


def nan2none(this_dict):
    for key, value in this_dict.items():
        if isinstance(value, float) and pd.isnull(value):
            this_dict[key] = None
    return this_dict

def make_key_dict_col3(filepath):
    keys = read_csv(filepath)
    keys_dict = {}
    for row in keys:
        keys_dict[row[2].lower()] = row[0]

def make_key_dict_getty(filepath):
    keys = read_csv(filepath)
    keys_dict = {}
    for row in keys:
        keys_dict[row[1].lower()] = row[0]



    # do I need to pop header? or can I just leave it. 
    return keys_dict

def make_key_dict_col7(filepath):
    keys = read_csv(filepath)
    keys_dict = {}
    for row in keys:
        keys_dict[row[7].lower()] = row[0]

    # do I need to pop header? or can I just leave it. 
    return keys_dict

def make_key_dict(filepath, keytype=None):
    keys = read_csv(filepath)
    keys_dict = {}
    for row in keys:
        if keytype is None:
            keys_dict[row[2].lower()] = row[0]
        elif keytype and row[3].lower() == keytype:
            # print("gotta keytype")
            keys_dict[row[3].lower()] = row[0]
        else:
            # if not none, but not the specific keytype
            pass
            # print("nothing here")

    # do I need to pop header? or can I just leave it. 
    return keys_dict

def get_counter():
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

# returns one mode value from list, ignoring None values
def get_mode(data):
    filtered_data = [value for value in data if value is not None]
    
    # Check if the filtered_data is empty before proceeding
    if not filtered_data:
        return None  # or return any other default value you prefer
    
    mode_counts = Counter(filtered_data)
    mode_value, max_count = mode_counts.most_common(1)[0]
    return mode_value



# abstraction to run a list through unlock_key_plurals_etc
def unlock_key_list(site_image_id, keys_list, keys_dict):
    key_nos_list = []
    
    for key in keys_list:
        key_no = unlock_key_plurals_etc(site_image_id, key, keys_dict)
        # print(key_no)
        if key_no:
            key_nos_list.append(key_no)
    return key_nos_list


# takes a key and runs all permutations through the dict, and saves missing ones
# this is the kitchen sink function
def unlock_key_plurals_etc(site_id,key, this_dict):
    def parse_plurals(key, this_dict, this_dict_set):
        plur_key = getInflection(key, 'NNS')
        sing_key = getInflection(key, 'NN')
        gerund_key = getInflection(key, 'VBG')
        # print("inflected are: ", plur_key, sing_key, gerund_key)
        if plur_key in this_dict_set: return this_dict[plur_key]
        elif sing_key in this_dict_set: return this_dict[sing_key]
        elif gerund_key in this_dict_set: return this_dict[gerund_key]
        else: return None


        # if plur_key and key != plur_key:
        #     try:
        #         key_no = this_dict[plur_key[0]]
        #         # key = plur_key
        #         # if VERBOSE: print(key_no)
        #     except:
        #         pass# print(key)
        # elif sing_key and key != sing_key:
        #     try:
        #         key_no = this_dict[sing_key[0]]
        #         # key = plur_key
        #         # print(key_no)
        #     except:
        #         pass
        #         # print(key)
        
        # if gerund_key and key != gerund_key:
        #     try:
        #         key_no = this_dict[gerund_key[0]]
        #         # key = plur_key
        #         # print("gotta key_no, " ,key_no)
        #     except:
        #         pass
        #         # print(key)


    dict_name = None
    first_key = next(iter(this_dict))
    if VERBOSE: print("first_key", first_key)
    if first_key == "drug": 
        dict_name = "keywords"
        this_dict_set = keys_set
    elif first_key == "men": 
        dict_name = "gender"
        this_dict_set = gender_set
    elif first_key == "newborn":
        dict_name = "age"
        this_dict_set = age_set
    else:
        print("no dict_name", first_key)
        this_dict_set = set(this_dict.keys())

    # elif first_key == "drug": dict_name = "keywords"
    key_no = None
    key = key.lower()
    if VERBOSE: print("trying to unlock key, and keys_dict:", key, dict_name)

    if key in this_dict_set:
    # try:
        # if VERBOSE: print("trying basic keys_dict for this key:,", key)
        # print("this is the dict", this_dict)
        key_no = this_dict[key]
        if VERBOSE: print("this is the key and key_no", key, key_no)

    # except:
        # key2keyGB = {"colour":"color", "centre":"center", "organisation":"organization", "recognise":"recognize", "realise":"realize", "favour":"favor", "labour":"labor", "neighbour":"neighbor", "theatre":"theater", "cheque":"check", "defence":"defense", "licence":"license", "offence":"offense", "ogue":"og", "programme":"program", "dialogue":"dialog", "catalogue":"catalog", "humour":"humor", "fibre":"fiber"}
        # # catching british spellings
        # for gbkey in key2keyGB.keys():
        #     if gbkey in key:
        #         uskey = key.replace(gbkey,key2keyGB[gbkey])
        #         try:
        #             key_no = this_dict[uskey]
        #             return(key_no)
        #         except:
        #             pass
        # if "our" in key:
        #     ourkey = key.replace("our","or")
        #     try:
        #         key_no = this_dict[ourkey]
        #         return(key_no)
        #     except:
        #         pass
        # getting rid of getty dash specifier categories
    else:
        if key in key2key_set:
            if VERBOSE: print(dict_name, "trying k2k for this key:,", key)
            # key = key2key[key]
            try:
                key_no = this_dict[key2key[key]]
                if VERBOSE: print("this is the key_no", key_no)
            except:
                if VERBOSE: print("wrong this_dict, no result for", key_no, "dict_name is ", dict_name)
                key_no = None
        if " - " in key and pd.isnull(key_no):
            dashlesskey = key.replace(" - "," ")
            dashcommakey = key.replace(" - ",", ")
            dashsplitkey = key.split(" - ")
            if VERBOSE: print("trying dashlesskey", dashlesskey)
            if dashlesskey in this_dict_set: key_no = this_dict[dashlesskey]
            elif dashcommakey in this_dict_set: key_no = this_dict[dashcommakey]
            elif dashsplitkey[0] in this_dict_set: 
                if VERBOSE: print("trying dashsplitkey 0", dashsplitkey[0])
                # print("this is the set", this_dict_set)
                key_no = this_dict[dashsplitkey[0]]
                if VERBOSE: print("this is the key_no", key_no)
            elif dashsplitkey[1] in this_dict_set: 
                if VERBOSE: print("trying dashsplitkey 1", dashsplitkey[1])
                # print("this is the set", this_dict_set)
                key_no = this_dict[dashsplitkey[1]]
                if VERBOSE: print("this is the key_no", key_no)
            else: key_no = None         
        if pd.isnull(key_no):
            key_no = parse_plurals(key, this_dict, this_dict_set)

            # try:
            #     key_no = this_dict[dashlesskey]
            #     return(key_no)
            # except:
            #     try:
            #         key_no = this_dict[dashcommakey]
            #         return(key_no)
            #     except:
            #         try:
            #             key_no = this_dict[dashsplitkey[0]]
            #             return(key_no)
            #         except:
            #             pass
                
        # try:
        #     # if key in key2key_set:
        #     #     key = key2key[key]
        #     #     key_no = this_dict[key]
        #     #     return(key_no)
        # except:
        #     # try:
        #     # try with inflections
        #     # print(site_id)
        #     # print(key)
        #     key_no = parse_plurals(key, this_dict, this_dict_set)
        #     # plur_key = getInflection(key, 'NNS')
            # sing_key = getInflection(key, 'NN')
            # gerund_key = getInflection(key, 'VBG')
            # # print("inflected are: ", plur_key, sing_key, gerund_key)
            # if plur_key and key != plur_key:
            #     try:
            #         key_no = this_dict[plur_key[0]]
            #         # key = plur_key
            #         if VERBOSE: print(key_no)
            #     except:
            #         pass# print(key)
            # elif sing_key and key != sing_key:
            #     try:
            #         key_no = this_dict[sing_key[0]]
            #         # key = plur_key
            #         # print(key_no)
            #     except:
            #         pass
            #         # print(key)
            
            # if gerund_key and key != gerund_key:
            #     try:
            #         key_no = this_dict[gerund_key[0]]
            #         # key = plur_key
            #         # print("gotta key_no, " ,key_no)
            #     except:
            #         pass
            #         # print(key)

    if key_no is not None:
        # save and return key_no
        value_list = [site_id,key_no]
        write_csv(CSV_IMAGEKEYS_PATH,value_list)
        return key_no
    else:
        # if nothing worked, save key, but only if site_id > 10
        # for gender/age, it passes in site_name_id, not site_image_id
        # doesn't write if key is in key2loc -- already captured locations
        if VERBOSE: 
            # proving that key is not in key2loc, only verbose
            print("key_no is null", key)
            try:
                print("key2loc[key]", key2loc[key])
            except:
                print("key not in key2loc - thus not a location")
                # not isinstance(site_id, int) and
        if key and len(this_dict) > 5000 and not key.startswith('-') and not key.endswith('.jpg') and not key in all_set and not key in set(TNB_list+multi_eth_list):
            # print(type(site_id))
            # print(site_id)
            value_list = [site_id,key]
            if dict_name == "gender" or dict_name == "age":
                # print("could not unlock:", value_list, "not saving")
                pass # not saving
            else:
                if not LOC_ONLY: print("could not unlock:", value_list, "and saving")
                try: write_csv(CSV_NOKEYS_PATH,value_list)
                except: print("failed to write", value_list)
        # print(value_list)
        return None


# finds dict items in a string
# called by search_keys
def findall_dict(my_dict,description):
    # Create a regular expression pattern that matches complete words in the dictionary, ignoring case
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(key) for key in my_dict.keys()) + r')\b', re.IGNORECASE)

    # # this re is greedier, matching partial words, w/o boundaries
    # pattern = re.compile(r'\b(' + '|'.join(my_dict.keys()) + r')\b', re.IGNORECASE)


    # Use the regular expression pattern to search for matches in the given string
    matches = pattern.findall(description)

    # Print the values associated with the matching keys
    for match in matches:
        value = my_dict.get(match.lower())
        if value is not None:
            return(value)
            # gender = value
            # print(f'The color of {match} is {value}')

# only searches through keys with dict
# called in by the get_eth_dictonly << as of Mar 2024 this is not longer true, I believe
def search_keys(keys_list, this_dict, do_write_csv, multi=False):
    if VERBOSE: print("[search_keys]")
    results = []
    found_eth2 = False
    for key in keys_list:
        # found = findall_dict(this_dict,key)
        try:
            found = unlock_key_dict(key, this_dict)
        except:
            found = None
        # catches empty strings, and converts to None (was breaking the loop)
        if found == "": found = None
        if found is not None:
            found_eth2 = True
            results.append(found)
            if VERBOSE: print('search_keys found:', found,"from key:", key)
            if do_write_csv:
                write_csv(CSV_ETH2_PATH,[key,found])
    if found_eth2 and do_write_csv: 
        write_csv(CSV_ETH2_PATH,keys_list)
            #age needs to be int()
            # print(results)

        # try:
        #     found = unlock_key_dict(key, this_dict)
        # # this is searching the keys based on the dict, note unlocking dict with keys
        # # found = findall_dict(this_dict,key)
        #     if found is not None:
        #         results.append(found)
        #         print('found it in keywords:', found,"from key:", key)
        # #     #age needs to be int()
        # #     # print(results)
        # except OperationalError as e:
        #         print("exception on unlock_key_dict")
        #         print(e)


    if len(set(results)) == 1:
        try:
            one_result = int(results[0])
        except:
            if VERBOSE: print("one_result int failed with this key: ", results[0])
            one_result = None
        # print("found a GOOD result: ", one_result)
    else:
        one_result = None
        # print("failed search: ", one_result)

    # returns one or many, in a list
    if multi:
        results_list = list(set(results))
    else:
        results_list = [one_result]
    return results_list

# print(key_nos_list)

def get_key_no_dictonly(key_name, keys_list, this_dict, do_write_csv=False):
    # key_name = df['ethnicity'][ind]
    # print('isnan?')
    # print(np.isnan(key_name))
    key_no_list = []
    key_no = None
    # if key_name is not None or key_name is not np.isnan(key_name):
    # metis checks to see if it VCG, and if so, does alternate loop to deal with Metis mistagging
    if not pd.isnull(key_name) and (key_name.lower() not in ["metis","6"] and THIS_SITE == 10):
        try:
            if VERBOSE: print("trying with key_name ",key_name)
            # print("this_dict", this_dict)
            key_no = unlock_key_dict(key_name, this_dict)
            if VERBOSE: print("key_no with key_name ",key_no)
            key_no_list.append(key_no)

            # key_no = eth_dict[key_name.lower()]
        # need to key this into integer, like with keys
            # print("key_name ",key_name)
        except:
            if VERBOSE: print("eth_dict failed with this key: ", eth_name)
    else:
        key_no_list = search_keys(keys_list, this_dict, do_write_csv, True)
        if VERBOSE: print("searched keys and found key_no: ", key_no_list)
    if THIS_SITE == 10 and not LOC_ONLY:
        # adding more general BIPOC tag, as Metis seems to be very imprecisely used in VCG
        if not key_no_list and key_name.lower() == "metis": key_no_list.append(13)
        # handle the "6" in VCG data, actually nope, skipping.
        elif key_name == "6": pass # key_no_list.append(6)
    return(key_no_list)

def unlock_key_dict(key,this_dict,this_key2key=None):
    key_no = None
    key = key.lower()
    # print("len of this_dict", len(this_dict))
    # print("len of this_key2key", len(this_key2key))
    # print("this_key2key", this_key2key)
    try:
        try:
            key_no = this_dict[key]
            if VERBOSE: print(f"unlock_key_dict yields key_no {str(key_no)} for {key}")
            return(key_no)
        except:
            # try again without underscores
            key_no = this_dict[key.replace("_"," ")]
            if VERBOSE: print(f"unlock_key_dict without underscores yields key_no {str(key_no)} for {key}")
            return(key_no)            
    except:
        if this_key2key:
            try:
                if key in this_key2key.keys():
                    altkey = this_key2key[key.lower()]
                    if VERBOSE: print("altkey", altkey)
                    key_no = this_dict[altkey.lower()]
                    if VERBOSE: print("this is the key_no via loc2loc", key_no)
                else:
                    print("key not in this_key2key", key)
                    key_no = key2loc[key]
                    print("key in key2loc", key_no)

                return(key_no)
            except:
                try:
                    #hard coding the second location dict here
                    key_no = locations_dict_alt[key.lower()]
                except:
                    print("NEW KEY -------------------------> ", key)
                    write_csv(CSV_NOLOC_PATH,[key])
                    return(999999999)
        else:
            pass
            # print out for testing purposes
            # print("unlock_key_dict failed, and no key2key for this key: ", key)

# def get_location(df, ind, keys_list):
#     location = None
#     key = df['country'][ind]
#     if not pd.isnull(key):
#         try:
#             location = loc_dict[key]
#         except:
#             print('NEW KEY, NOT IN COUNTRY -------------------------> ', key)
#     # else:
#     #     print('NULL country: ', key)

#     return(location)


#currently only called for keys to desc in main show. not for gender.
def description_to_keys(description, site_id, this_dict="keys_dict"):
    if this_dict=="keys_dict":
        this_dict = keys_dict

    if description: description = description.replace(",","").replace("'s","").replace(".","")
    else: return None

    # print("description_to_keys")    
    key_nos_list =[]
    # description = df['title'][ind]
    # print(description)
    key_no = None
    desc_keys = description.split(" ")
    if VERBOSE: print("[description_to_keys] desc_keys ",desc_keys)
    # print(this_dict)
    for key in desc_keys:
        if VERBOSE: print("checking key ", key)
        if not pd.isnull(key) and len(key) > 2:
            key_no = unlock_key_plurals_etc(site_id,key,this_dict)
            # print("key_no passed through:")
            # print(key_no)
            if key_no:
                key_nos_list.append(key_no)
            # print("key_nos_list ",key_nos_list)
    return key_nos_list

# sorts out binary code from trans and non binary gender
# returns False if binary code, else returns gender_id
def get_TNB(description, keys_list):
    global is_code
    global gender_TNB
    global TNB_list
    global binary
    global Code_list 
    is_code = False
    gender_TNB = None
    binary = "binary"
    transpattern = r'\btrans\b'
    Code_list = ["number", "screen", "computer", "code", "coding", "program", "programming", "developer", "development", "software", "engineer", "engineering", "digital", "internet", "web", "online", "cyber"]
    def binary_code_in_desc(description):
        global is_code
        if any(word in description for word in Code_list):
            if VERBOSE: print("Binary Code Desc:", description)
            is_code = True
        elif any(word in keys_list for word in Code_list):
            if VERBOSE: print("Binary Code Keys:", description)
            is_code = True                
        else:
            is_code = False
        return is_code

    if any(word in description for word in TNB_list):
        gender_TNB = findall_dict(gender_dict_TNB,description)
        # gender_TNB = get_TNB(description)
        if VERBOSE: print("Binary TNB_list:", gender_TNB, description)
    elif re.search(transpattern, description):
        gender_TNB = 7
        if VERBOSE: print("Binary transpattern:", gender_TNB, description)
    elif binary in description:
        if not binary_code_in_desc(description):
            gender_TNB = findall_dict(gender_dict_TNB,description)
            if VERBOSE: print("Binary not Code:", gender_TNB, description)
    return gender_TNB


def get_gender_age_row(gender_string, age_string, description, keys_list, site_id):
    global gender_dict
    global age_dict
    global age_details_dict
    global skip_keys

    if LOC_ONLY:
        # skipping if reprocessing for location only
        return None, None, None
    
    def try_gender_age_key(gender, age, age_detail, this_string, extra_dict=False):
        global gender_dict
        global age_dict
        global age_details_dict
        if extra_dict:
                gender_dict = gender_dict_secondary
                # age_dict = age_dict_secondary
        if not pd.isnull(this_string):
            # print(f"looking for {this_string} in dict")
            #convertkeys
            try:
                gender = gender_dict[this_string.lower()]
                if gender == 3:
                    gender = 1
                    age = 7
                elif gender == 4:
                    gender = 8
                    age = 7
                elif gender == 9:
                    gender = 1
                    age = 5
                elif gender == 10:
                    gender = 8
                    age = 5
                if VERBOSE: print(f"first try, gender is {str(gender)} and age is {str(age)}")
                # gender_dict={"men":1, "none":2, "oldmen":3, "oldwomen":4, "nonbinary":5, "other":6, "trans":7, "women":8, "youngmen":9, "youngwomen":10}
            except:
                try:
                    age = age_dict[this_string.lower()]
                    if VERBOSE: print(f"second try age is {str(age)}")

                except:
                    # this isn't relevant for iStock
                    # commenting out
                    # print('NEW KEY, NOT AGE OR GENDER -------------------------> ', this_string)
                    pass
            try:
                age_detail = age_details_dict[this_string.lower()]
            except:
                # no age detail. it is only on some sites
                pass

        # else:
        #     print("string is None")
        return gender, age, age_detail

    def prioritize_age_gender(gender_list,age_list):
        if VERBOSE: print(f"prioritize_age_gender gender_list is {gender_list}")
        if (1 in gender_list and 8 in gender_list) or (3 in gender_list and 4 in gender_list) or (9 in gender_list and 10 in gender_list):
            gender = 11
            age = get_mode(age_list)
        elif gender_list.count(3) > 0:
            gender = 1
            age = 7
        elif gender_list.count(4) > 0:
            gender = 8
            age = 7
        elif gender_list.count(9) > 0:
            gender = 1
            age = 5
        elif gender_list.count(10) > 0:
            gender = 8
            age = 5
        else:
            gender = get_mode(gender_list)
            age = get_mode(age_list)
        return gender, age

    def get_gender_age_keywords(gender, age, age_detail, keys_list):
        global skip_keys
        gender_list=[]
        age_list=[]
        age_detail_list=[]
        for key in keys_list:
            # print("key is ", key)
            # reset variables
            if key in skip_keys:
                if VERBOSE: ("skip_keys for other")
                continue
            gender = None
            age= None
            age_detail= None
            gender, age, age_detail = try_gender_age_key(gender, age, age_detail, key, extra_dict=True)
            gender_list.append(gender)
            age_list.append(age)
            age_detail_list.append(age_detail)
        # print(gender_list)
        # print(age_list)
        # print(age_detail_list)
        gender, age = prioritize_age_gender(gender_list,age_list)
        age_detail = get_mode(age_detail_list)
        return gender, age, age_detail

    print("get_gender_age_row starting")
    global gender_dict
    gender = None
    age= None
    age_detail= None
    desc_age = None

    if description: description = description.replace(",","").replace("'s","").replace(".","")

    if VERBOSE: 
        print("gender_string, age_string",gender_string, age_string)
        print("types",type(gender_string), type(age_string))

    # this if/if structure is necessary because "" and isnull were not compatible
    # Get gender
    # why isn't this working right? 
    if gender_string != "":
        print("trying try_gender_age_key for", gender_string, age_string)
        gender, age, age_detail = try_gender_age_key(gender, age, age_detail, gender_string)
        if VERBOSE:
            print(gender)
            print(age)
            print(age_detail)

    else:
        print("trying get_gender_age_keywords for", gender_string, age_string)
        gender, age, age_detail = get_gender_age_keywords(gender, age, age_detail, keys_list)
        if VERBOSE:
            print(gender)
            print(age)
            print(age_detail)


        #try keys for gender
    # print("gender, age, after try key gender_string")
    # print(gender)
    # print(age)
    if pd.isnull(gender) and description: 
        if VERBOSE: print("gender is null")
        gender_results = description_to_keys(description, site_id, gender_dict)
        if gender_results and len(set(gender_results)) == 1:
            gender = gender_results[0]
        elif len(set(gender_results)) > 1: 
            for result in gender_results:
                if 8 and 10 in gender_results:
                    print("GOTTA TEEN F", gender_results)
                    gender = 8
                    desc_age = 5
                elif 1 and 9 in gender_results:
                    print("GOTTA TEEN M", gender_results)
                    gender = 1
                    desc_age = 5
                elif 8 and 4 in gender_results:
                    print("GOTTA OLD F", gender_results)
                    gender = 8
                    desc_age = 7
                elif 1 and 3 in gender_results:
                    print("GOTTA OLD M", gender_results)
                    gender = 1
                    desc_age = 7
                else:
                    print("too many genders", gender_results)
    # print("gender, age, after try description gender_string")
    # print(gender)
    # print(age)

    # Get age 
    if age_string != "":
        gender, age, age_detail = try_gender_age_key(gender, age, age_detail, age_string)
        #this shouldn't overwrite the existing gender if it is not None
    # print("gender, age, after try key age_string")
    # print(gender)
    # print(age)
    if pd.isnull(age) and desc_age:
        age = desc_age
    elif pd.isnull(age):
        print("age is still null, trying keywordsa again, but not for gender")
        _, age, age_detail = get_gender_age_keywords(gender, age, age_detail, keys_list)
        if VERBOSE:
            print(gender)
            print(age)
            print(age_detail)
        if pd.isnull(age) and description:
            print("age is really still null, trying description")

            age_results = description_to_keys(description, site_id, age_dict)
            if len(set(age_results)) == 1:
                age = age_results[0]

    # after everything, get gender for TNB from description and supercede existing gender string
    if description:
        gender_TNB = get_TNB(description.lower(), keys_list)    
        if VERBOSE: print("gender_TNB", gender_TNB)
        if gender_TNB: 
            gender = gender_TNB

    if VERBOSE:
        print("gender, age, after everything")
        print(gender)
        print(age)

    if not age or not gender:
        pass
        if VERBOSE: 
            print("MISSING AGE OR GENDER, IS IT IN THE KEYS?")

            print(keys_list)

    return gender, age, age_detail


'''
1   getty
2   shutterstock
3   adobe
4   istock
5   pexels
6   unsplash
7   pond5
8   123rf
9   alamy
10  visualchinagroup
'''

def get_hash_folders(filename):
    m = hashlib.md5()
    m.update(filename.encode('utf-8'))
    d = m.hexdigest()
    # print(d)
    # csvWriter1.writerow(["https://upload.wikimedia.org/wikipedia/commons/"+d[0]+'/'+d[0:2]+'/'+filename])
    return d[0].upper(), d[0:2].upper()

def generate_local_unhashed_image_filepath(image_name):
    file_name_path = image_name.split('?')[0]
    file_name = file_name_path.split('/')[-1].replace(".jpeg",".jpg")
    # extension = file_name.split('.')[-1]
    hash_folder, hash_subfolder = get_hash_folders(file_name)
    # print("hash_folder: ",hash_folder)
    # print("hash_subfolder: ", hash_subfolder)
    if VERBOSE: print (os.path.join(hash_folder, hash_subfolder,file_name))
    return os.path.join(hash_folder, hash_subfolder,file_name)
        # IMAGES_FOLDER_NAME, hash_folder, '{}.{}'.format(file_name, extension))

# I don't think there is a df? 
def structure_row_pexels(df, ind, keys_list): 
    site_id = 5
    gender_key, age_key = get_gender_age(df, ind, keys_list, site_id)
    if 'country' in df.columns:
        location_no = get_location(df, ind, keys_list)
    else:
        location_no = None
    image_row = {
        "site_image_id": df.loc[ind,'id'],
        "site_name_id": site_id,
        "description": df['title'][ind],
        "location_id": location_no,
        # "": df['number_of_people'][ind] # should always be one. If not one, toss it? 
        # "": df['orientation'][ind]
        "age_id": age_key,
        "gender_id": gender_key,  
        # "location_id":"0",
        # "": df['mood'][ind]
        "contentUrl": df['image_url'][ind],
        "imagename": generate_local_unhashed_image_filepath(df['image_url'][ind]) # need to refactor this from the contentURL using the hash function
    }
    return(nan2none(image_row))


def structure_row_123_asrow(row, ind, keys_list):
    site_id = 8
    gender = None
    age = row[4]
    description = row[1]
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, site_id)

    image_row = {
        "site_image_id": row[0],
        "site_name_id": site_id,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": row[9],
        "imagename": generate_local_unhashed_image_filepath(row[9])  # need to refactor this from the contentURL using the hash function
    }
    
    return nan2none(image_row)


def structure_row_adobe(row, ind, keys_list):
    site_id = 3
    gender = row[7]
    age = None
    description = row[1]
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, site_id)

    image_row = {
        "site_image_id": row[0],
        "site_name_id": site_id,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": row[8],
        "imagename": generate_local_unhashed_image_filepath(row[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }
    
    return nan2none(image_row)
def structure_row_loconly(item, ind, keys_list):
    if VERBOSE: print(item)
    gender = item.get("filters", {}).get("gender", None)
    age = item.get("filters", {}).get("age", None)
    shoot_location = item.get("filters", {}).get("location", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, THIS_SITE)
    filename = item["id"]+".jpg"
    location = item.get("country", None)
    image_row = {
        "site_image_id": item["id"],
        "site_name_id": THIS_SITE,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0],get_hash_folders(filename)[1],filename),  # hash filename from id+jpg
        "location_id": get_location(location, keys_list),        
        "author": item.get("author", None),     

        ## TK
        # "shoot_location": shoot_location,
        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }
    return nan2none(image_row)



def structure_row_istock(row, ind, keys_list):
    print(row[11])
    site_id = 4 #id for the site, not the image
    gender = row[7].replace("_"," ")
    age = row[6].replace("_"," ")
    country = row[3].replace("_"," ")
    description = row[1]
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, site_id)
    country_key = None
    print("row 3 is", type(country))
    if country and country != "":
        country_key = unlock_key_dict(country,locations_dict, loc2loc)
        if country_key == 999999999:
            write_csv(CSV_BLANKLOC_PATH,row)
            return(None)
    if not country or country == "":
        #tk search keys
        # get eth from keywords, using keys_list and eth_keys_dict
        if not LOC_ONLY: print("UNLOCKING SEARCH_KEYS_FOR_LOC <><><><><><><><>")
        # absence of search string ("None") triggers search_keys function
        loc_no_list = get_key_no_dictonly(None, keys_list, locations_dict, True)
        print(loc_no_list)
        country_key = get_mode(loc_no_list)
        # if not loc_no_list:
        #     loc_no_list = get_key_no_dictonly(None, keys_list, locations_dict_alt, True)
        #     if loc_no_list: 
        if country_key: print(f"SEARCH_KEYS_FOR_LOC found for location_id {country_key}")
    else:
        country_key = None

    # handle long URLs
    if len(row[10])>300: contentUrl = "contentUrl was greater than 300 characters search for site_image_id"
    else: contentUrl = row[10]

    image_row = {
        "location_id": country_key,        
        "site_image_id": row[0],
        "site_name_id": site_id,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": contentUrl,
        "imagename": generate_local_unhashed_image_filepath(row[11].replace("images/","").split("?")[0])  # need to refactor this from the contentURL using the hash function
    }
    
    return nan2none(image_row)

def structure_row_istock_loconly(item, ind, keys_list):
    if VERBOSE: print(item)
    gender = item.get("filters", {}).get("gender", None)
    age = item.get("filters", {}).get("age", None)
    shoot_location = item.get("filters", {}).get("location", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, THIS_SITE)
    filename = item["id"]+".jpg"
    location = item.get("country", None)
    image_row = {
        "site_image_id": item["id"],
        "site_name_id": THIS_SITE,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0],get_hash_folders(filename)[1],filename),  # hash filename from id+jpg
        "location_id": get_location(location, keys_list),        
        "author": item.get("author", None),     

        ## TK
        # "shoot_location": shoot_location,
        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }
    return nan2none(image_row)


def structure_row_shutterstock(row, ind, keys_list):
    # print(row[11])
    site_id = 2 #id for the site, not the image
    gender = row[6].replace("_"," ")
    age = row[5].replace("_"," ")
    country = row[3].replace("_"," ")
    description = row[1]
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, site_id)
    country_key = None
    print("row 3 is", type(country))
    if country and country != "":
        country_key = unlock_key_dict(country,locations_dict_AA, loc2loc)
        # print("country_key", str(country_key))
        if country_key == 999999999:
            write_csv(CSV_BLANKLOC_PATH,row)
            return(None)
    elif not country or country == "":
        #tk search keys
        # get eth from keywords, using keys_list and eth_keys_dict
        if not LOC_ONLY: print("UNLOCKING SEARCH_KEYS_FOR_LOC <><><><><><><><>")
        # absence of search string ("None") triggers search_keys function
        loc_no_list = get_key_no_dictonly(None, keys_list, locations_dict, True)
        print(loc_no_list)
        country_key = get_mode(loc_no_list)
        # if not loc_no_list:
        #     loc_no_list = get_key_no_dictonly(None, keys_list, locations_dict_alt, True)
        #     if loc_no_list: 
        if country_key: print(f"SEARCH_KEYS_FOR_LOC found for {country_key}")
    else:
        country_key = None

    # handle long URLs
    if len(row[8])>300: contentUrl = "contentUrl was greater than 300 characters search for site_image_id"
    else: contentUrl = row[8]

    image_row = {
        "location_id": country_key,        
        "site_image_id": row[0],
        "site_name_id": site_id,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": contentUrl,
        "imagename": generate_local_unhashed_image_filepath(row[9].replace("images/","").split("?")[0])  # need to refactor this from the contentURL using the hash function
    }
    # print(image_row)
    return nan2none(image_row)

def structure_row_shutterstock_loconly(item, ind, keys_list):
    if VERBOSE: print(item)
    gender = item.get("filters", {}).get("gender", None)
    age = item.get("filters", {}).get("age", None)
    shoot_location = item.get("filters", {}).get("location", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, THIS_SITE)
    filename = item["id"]+".jpg"
    location = item.get("filters", {}).get("country", None)
    if not location: location = item.get("country", None)

    image_row = {
        "site_image_id": item["id"],
        "site_name_id": THIS_SITE,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0],get_hash_folders(filename)[1],filename),  # hash filename from id+jpg
        "location_id": get_location(location, keys_list),        
        "author": item.get("author", None),     

        ## TK
        # "shoot_location": shoot_location,
        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }
    return nan2none(image_row)

def structure_row_getty(item, ind, keys_list):
    site_id = 1
    gender = item.get("gender_id", None)
    age = item.get("age_id", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, site_id)
    country_key = None
    if item.get("location_id") is not None:
        print("item location_id is", item.get("location_id"))
        if VERBOSE: print("we gotta location: ", item.get("location_id"))
        country_key = unlock_key_dict(item.get("location_id"),locations_dict_getty, loc2loc)
    if not item.get("location_id") or not country_key:
        if VERBOSE: print("nada location: ", item.get("location_id"))
        country_key = search_keys(keys_list, key2loc, True)
        if VERBOSE: ("search_keys key2loc for location found: ", country_key)

    image_row = {
        "site_image_id": item.get("id"),
        "site_name_id": site_id,
        "description": description[:140] if description else None,
        "caption": item.get("caption", None)[:140] if item.get("caption") else None,
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item.get("contentUrl", None),
        "imagename": item.get("imagename", None),
        "location_id": country_key,        
        "author": item.get("author", None),        
        "uploadDate": item.get("uploadDate", None),        
    }

        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function

    return nan2none(image_row)





def structure_row_unsplash(item, ind, keys_list):
    # I am tossing the unsplash gender age and ethnicity data, as it is garbage. Noise. 
    # unsplash is generaly garbage, and I should exclude it from the data analysis. 
    site_id = 6
    gender = None
    age = None
    description = item["title"]
    filename = item["id"]+".jpg"
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, site_id)
    country_key = None
    # if item["location_id"]:
    #     if VERBOSE: print("we gotta location: ", item["location_id"])
    #     country_key = unlock_key_dict(item["location_id"],locations_dict_getty, loc2loc)
    # if not item["location_id"] or not country_key:
    #     if VERBOSE: print("nada location: ", item["location_id"])
    #     country_key = search_keys(keys_list, key2loc, True)
    #     if VERBOSE: ("search_keys key2loc for location found: ", country_key)

    image_row = {
        "site_image_id": item["id"],
        "site_name_id": site_id,
        "description": description[:140],
        "caption": None,
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0],get_hash_folders(filename)[1],filename),  # hash filename from id+jpg
        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }

    return nan2none(image_row)

# def get_location(location, keys_list):
    
#     country_key = None
#     if location:
#         if VERBOSE: print("we gotta location: ", location)
#         country_key = unlock_key_dict(location,locations_dict_getty, loc2loc)
#     if not location or not country_key:
#         if VERBOSE: print("nada location: ", location)
#         country_key = search_keys(keys_list, key2loc, True)
#         if VERBOSE: ("search_keys key2loc for location found: ", country_key)
#     return country_key

def itter_location(country, keys_list):
    country_key = None
    if country: country = country.lstrip().rstrip()
    if country and country != "" and len(country) == 2:
        if country == "uk": country = "gb"
        country_key = unlock_key_dict(country,locations_dict_AA, loc2loc)
        # print("country_key", str(country_key))
    elif country and country != "":
        country_key = unlock_key_dict(country,locations_dict, loc2loc)
        if not country_key:
            country_key = unlock_key_dict(country,locations_dict_alt, loc2loc)
            if not country_key:
                print("country not found after AA reg and alt-------------> ", country)
    elif not country or country == "":
        #tk search keys
        # get eth from keywords, using keys_list and eth_keys_dict
        if not LOC_ONLY: print("UNLOCKING SEARCH_KEYS_FOR_LOC <><><><><><><><>")
        # absence of search string ("None") triggers search_keys function
        loc_no_list = get_key_no_dictonly(None, keys_list, locations_dict, True)
        if not LOC_ONLY: print(loc_no_list)
        if not loc_no_list:
            loc_no_list = get_key_no_dictonly(None, keys_list, locations_dict_alt, True)
        #     if loc_no_list: 
        country_key = get_mode(loc_no_list)
        if country_key: print(f"SEARCH_KEYS_FOR_LOC found for {country_key}")
    if country_key == 999999999:
        return(None)
    return country_key

def get_location(country, keys_list):
    if VERBOSE: print("get_location starting")
    country_key = None
    location_info = []
    if country:
        if VERBOSE: print("we gotta location: ", country)
        location_info = country.split(", ") if "," in country else []
        if len(location_info) > 1: 
            print(location_info)
            country_key = itter_location(location_info[-1], keys_list)
    if not country_key and len(location_info) > 1:
        if VERBOSE: print("we location_info: ", country)
        for loc in location_info:
            country_key = itter_location(loc, keys_list)
            if country_key:
                break
    if not country_key:
        if VERBOSE: print("itter country: ", country)
        country_key = itter_location(country, keys_list)
    if not country_key and len(location_info) > 1:
        for loc in location_info:
            print("loc", loc)
            try:
                country_key = loc2loc[loc.lstrip().rstrip()]
                if country_key:
                    return(country_key)
            except:
                print("failed loc2loc, onto next")
            if not country_key:
                try:
                    country_key = loc2loc[loc.lstrip().rstrip()]
                except:
                    print("failed final loc2loc")
    return country_key

def structure_row_pixcy(item, ind, keys_list):
    print("keys_list", keys_list)
    gender = item.get("gender", None)
    print("gender", gender)
    age = item.get("age", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, THIS_SITE)
    print("gender_key, age_key, age_detail_key", gender_key, age_key, age_detail_key)
    location = item.get("location", None)
    filename = item["id"]+".jpg"

    if item["location"]:
        if VERBOSE: print("we gotta location: ", item["location"])
        country_key = unlock_key_dict(item["location"],locations_dict_getty, loc2loc)
    if not item["location"] or not country_key:
        if VERBOSE: print("nada location: ", item["location"])
        country_key = search_keys(keys_list, key2loc, True)
        if VERBOSE: ("search_keys key2loc for location found: ", country_key)

    image_row = {
        "site_image_id": item["id"],
        "site_name_id": THIS_SITE,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0],get_hash_folders(filename)[1],filename),  # hash filename from id+jpg
        "location_id": get_location(location, keys_list),        
        "author": item["author"],        
        
        ## TK
        "release_name_id": get_release(item)

        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }

    return nan2none(image_row)


def structure_row_PIXERF(item, ind, keys_list):
    if VERBOSE: print(item)
    gender = item.get("gender", None)
    age = item.get("age", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, THIS_SITE)
    filename = item["id"]+".jpg"
    location = item.get("location", None)
    image_row = {
        "site_image_id": item["id"],
        "site_name_id": THIS_SITE,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0],get_hash_folders(filename)[1],filename),  # hash filename from id+jpg
        "location_id": get_location(location, keys_list),        
        "author": item.get("author", None)     

        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }

    return nan2none(image_row)

def structure_row_ImagesBazzar(item, ind, keys_list):
    if VERBOSE: print(item)
    gender = item.get("filters", {}).get("gender", None)
    age = item.get("filters", {}).get("age", None)
    shoot_location = item.get("filters", {}).get("location", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, THIS_SITE)
    filename = item["id"]+".jpg"
    location = item.get("location", None)
    image_row = {
        "site_image_id": item["id"],
        "site_name_id": THIS_SITE,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0],get_hash_folders(filename)[1],filename),  # hash filename from id+jpg
        "location_id": get_location(location, keys_list),        
        "author": item.get("author", None),     

        ## TK
        # "shoot_location": shoot_location,
        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }

    print("shoot_location", shoot_location)

    return nan2none(image_row), shoot_location

def get_release(item):
    release_dict = {'Yes': 1, 'No': 2, 'CCBYSA': 3, 'CCBY': 4, 'CCBYSANC': 5, 'CCBYNC': 6, 'Free': 7, 'N/A': 8}
    release_key = item.get("model_release", None)
    if release_key:
        release_value = release_dict.get(release_key.lower(), None)
    else:
        release_value = None
    return release_value

def structure_row_INDIAPB(item, ind, keys_list):
    if VERBOSE: print(item)
    gender = item.get("filters", {}).get("gender", None)
    age = item.get("filters", {}).get("age", None)
    # shoot_location = item.get("filters", {}).get("location", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, THIS_SITE)
    filename = item["id"]+".jpg"
    location = item.get("location", None)
    image_row = {
        "site_image_id": item["id"],
        "site_name_id": THIS_SITE,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0],get_hash_folders(filename)[1],filename),  # hash filename from id+jpg
        "location_id": get_location(location, keys_list),        
        "author": item.get("author", None),     

        ## TK
        "release_name_id": get_release(item)
        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }


    return nan2none(image_row)

def structure_row_iwaria(item, ind, keys_list):
    if VERBOSE: print(item)
    gender = item.get("filters", {}).get("gender", None)
    age = item.get("filters", {}).get("age", None)
    # shoot_location = item.get("filters", {}).get("location", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, THIS_SITE)
    filename = item["id"]+".jpg"
    location = item.get("location", None)
    if description: description = description[:140]
    image_row = {
        "site_image_id": item["id"],
        "site_name_id": THIS_SITE,
        "description": description,
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0],get_hash_folders(filename)[1],filename),  # hash filename from id+jpg
        "location_id": get_location(location, keys_list),        
        "author": item.get("author", None),     
        "uploadDate": item.get("uploadDate"),        

        ## TK
        "release_name_id": get_release(item)
        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }
    return nan2none(image_row)

def structure_row_nappy(item, ind, keys_list):
    if VERBOSE: print(item)
    gender = item.get("filters", {}).get("gender", None)
    age = item.get("filters", {}).get("age", None)
    # shoot_location = item.get("filters", {}).get("location", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, THIS_SITE)
    filename = item["id"]+".jpg"
    location = item.get("location", None)
    location_id = get_location(location, keys_list)
    print("location", location_id)
    image_row = {
        "site_image_id": item["id"],
        "site_name_id": THIS_SITE,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0],get_hash_folders(filename)[1],filename),  # hash filename from id+jpg
        "location_id": location_id,        
        "author": item.get("author", None),     
        "uploadDate": item.get("uploadDate"),        

        ## TK
        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }
    print("image_row", image_row)
    return nan2none(image_row)

def structure_row_PICHA(item, ind, keys_list):
    if VERBOSE: print(item)
    gender = item.get("filters", {}).get("gender", None)
    age = item.get("filters", {}).get("age", None)
    # shoot_location = item.get("filters", {}).get("location", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, THIS_SITE)
    filename = item["id"]+".jpg"
    location = item.get("location", None)
    image_row = {
        "site_image_id": item["id"],
        "site_name_id": THIS_SITE,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0],get_hash_folders(filename)[1],filename),  # hash filename from id+jpg
        "location_id": get_location(location, keys_list),        
        "author": item.get("author", None),     
        "uploadDate": item.get("uploadDate"),        

        ## TK
        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }
    return nan2none(image_row)
def structure_row_AFRIPICS(item, ind, keys_list):
    if VERBOSE: print(item)
    gender = item.get("filters", {}).get("gender", None)
    age = item.get("filters", {}).get("age", None)
    # shoot_location = item.get("filters", {}).get("location", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, THIS_SITE)
    filename = item["id"]+".jpg"
    location = item.get("location", None)
    image_row = {
        "site_image_id": item["id"],
        "site_name_id": THIS_SITE,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0],get_hash_folders(filename)[1],filename),  # hash filename from id+jpg
        "location_id": get_location(location, keys_list),        
        "author": item.get("author", None),     
        "uploadDate": item.get("uploadDate"),        

        ## TK
        "release_name_id": get_release(item)
        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }
    return nan2none(image_row)


def structure_row_alamy(item, ind, keys_list):
    if VERBOSE: print(item)
    gender = item.get("filters", {}).get("gender", None)
    if not gender: gender = item.get("gender", None)
    age = item.get("filters", {}).get("age", None)
    shoot_location = item.get("filters", {}).get("location", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, THIS_SITE)
    filename = item["id"]+".jpg"
    location = item.get("location", None)

    def parse_date(date_string):
        if date_string:
            try:
                # Try to parse "dd Month yyyy" format
                return datetime.strptime(date_string, '%d %B %Y')
            except ValueError:
                try:
                    # Try to parse "Month yyyy" format
                    return datetime.strptime(date_string, '%B %Y')
                except ValueError:
                    print(f"Warning: Unable to parse date '{date_string}'. Setting to None.")
                    return None
        return None

    date_string = item.get("date_taken", None)
    if date_string:
        parsed_date = parse_date(date_string)
        formatted_date = parsed_date.strftime('%Y-%m-%d') if parsed_date else None
    else:
        formatted_date = None
    image_row = {
        "site_image_id": item["id"],
        "site_name_id": THIS_SITE,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0], get_hash_folders(filename)[1], filename),
        "location_id": get_location(location, keys_list),        
        "author": item.get("author", None),     
        "uploadDate": formatted_date,    
        "release_name_id": get_release(item)
    }
    return nan2none(image_row), shoot_location

def structure_row_VCG(item, ind, keys_list):
    if VERBOSE: print(item)
    gender = item.get("filters", {}).get("gender", None)
    if not gender: gender = item.get("gender", None)
    age = item.get("filters", {}).get("age", None)
    shoot_location = item.get("filters", {}).get("location", None)
    description = item.get("title", None)
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, THIS_SITE)
    filename = item["id"]+".jpg"
    location = item.get("location", None)

    
    image_row = {
        "site_image_id": item["id"],
        "site_name_id": THIS_SITE,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": item["img"],
        "imagename": os.path.join(get_hash_folders(filename)[0],get_hash_folders(filename)[1],filename),  # hash filename from id+jpg
        "location_id": get_location(location, keys_list),        
        "author": item.get("author", None),     
        "uploadDate": item.get("date_taken", None),    
        # "shoot_location": shoot_location,

        ## TK
        "release_name_id": get_release(item)
        # "imagename": generate_local_unhashed_image_filepath(item[9].replace("images/",""))  # need to refactor this from the contentURL using the hash function
    }
    return nan2none(image_row), shoot_location



# Define a custom retry decorator
def custom_retry(func):
    @retry(
        wait_fixed=1000,  # 1 second wait
        stop_max_attempt_number=5,  # 5 retries
        retry_on_exception=lambda ex: isinstance(ex, OperationalError),
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Example usage of the custom_retry decorator
@custom_retry
def execute_query_with_retry(conn, query, parameters=None):
    try:
        if parameters:
            result = conn.execute(query, parameters)
        else:
            result = conn.execute(query)
        conn.commit()  # Commit the transaction
        return result
    except OperationalError as e:
        print(f"OperationalError occurred: {e}")
        raise e

def ingest_json():
    def construct_keys_list(raw_keys):
        def rlstrip(x):
            strip_patterns =[":", ";", "'", " ", "(", ")", "/", "#", ".", "{", "}", "@", "&", "!", "***_***"]
            for ptn in strip_patterns:
                x = x.rstrip(ptn).lstrip(ptn)
            # dealing with " and '
            x = x.rstrip('"').lstrip('"')
                
        keys_list = []
        split_patterns = ["|", ";", "#", ]
        # test raw_keys to see if it is a list
        if type(raw_keys) is not list:
            raw_keys = raw_keys.split("|")

        # splitting keys. try is for getty, except is currently set for pexels.
        try:
            # keys_list = row[column_keys].lower().split(separator_keys)
            # keys_list = [x.lower() for x in item["keywords"]]
            for x in raw_keys:
                x = x.lower()

                # print(skip_keys)
                if x not in skip_keys:
                    if x in key2key_set:
                        xkey = key2key[x.lower()]
                    else:
                        xkey = x.lower()
                    keys_list.append(xkey)
            # # adobe specific. remove for future sites
            # desc_list = row[3].replace(",","").lower().split("-")
            # keys_list = list(filter(None, keys_list + desc_list))

        except IndexError:
            print("keys failed")
        if VERBOSE: print("keys_list", keys_list)
        return keys_list

    def item_from_row(row, CSV_TYPE=None):
        if CSV_TYPE == "123rf":
            # Parse according to the new CSV structure (123rf.com)
            item = {
                "id": row["id"],
                "title": row["title"],
                "keywords": row["word"],  # Use "word" for keywords in the 123rf structure
                "img": row["image_url"],
                "filters": {
                    "number_of_people": row["people"] if row["people"] else "",
                    "country": "",  # This CSV structure doesn't have a country field
                    "gender": "",  # No gender field in this structure
                    "age": row["age"] if row["age"] else "",
                    "mood": row["exclude"] if row["exclude"] else "",  # Assuming "exclude" represents mood
                    "ethnicity": row["ethnicity"] if row["ethnicity"] else "",
                }
            }
        else:
            # Parse according to the original CSV structure
            item = {
                "id": row["id"],
                "title": row["title"],
                "keywords": row["keywords"].replace('|', '|'),  # Adjust if needed for keyword delimiter
                "img": row["image_url"],
                "filters": {
                    "number_of_people": row["number_of_people"] if row["number_of_people"] else "",
                    "country": row["country"] if row["country"] else "",
                    "gender": row["gender"] if row["gender"] else "",
                    "age": row["age"] if row["age"] else "",
                    "mood": row["mood"] if row["mood"] else "",
                    "ethnicity": row["ethnicity"] if row["ethnicity"] else "",
                }
            }
        return item
    # change this for each site ingested #
    # adobe
    # column_keys = 6 #where the keywords are
    # separator_keys = " " #for keywords, in the column listed above
    # column_site = 8 #not sure this is used
    # column_eth = None #ethnicity
    # search_desc_for_keys = True

    # csv only
    # # # istock
    # column_keys = 2 #where the keywords are
    # separator_keys = "|" #for keywords, in the column listed above
    # # column_site = 8 #not sure this is used
    # column_eth = 7 #ethnicity

    search_desc_for_keys = False
    if LOC_ONLY: search_desc_for_keys = True
    

    # with open(JSONL_IN_PATH) as file_obj:
    #     # reader = csv.reader((row.replace('\0', '').replace('\x00', '') for row in in_file), delimiter=",")

    # # for csv - PEXELS
    # with open(JSONL_IN_PATH, newline='', encoding='utf-8') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     print("CSV_TYPE", CSV_TYPE)

    # restore this
    with open(JSONL_IN_PATH, 'r') as cache_file:

        start_counter = get_counter()
        print("start_counter: ", start_counter)
        # start_counter = 358430 #temporary for testing, while adobe is ongoing
        if LOC_ONLY: start_counter = 0
        counter = 0
        new_loc_counter = 0
        ind = 0
        shoot_location = None

        # restore this
        for item in cache_file.readlines():
            item = json.loads(item)
        
        # # for csv - PEXELS/123
        # for row in reader:
        #     item = item_from_row(row, CSV_TYPE)
            

            if TIMER: start_timer = time.time()

            if counter < start_counter:
                counter += 1
                continue
            
            if LOC_ONLY:
                if item.get("id", None) in already_ingested:
                    counter += 1
                    if counter % 10000 == 0:
                        print("already ingested count ", counter)
                    # print("already ingested", item.get("id", None))
                    continue
                else:
                    print("new loc", item.get("id", None))
            key_nos_list = []
            keys_list = []
            if VERBOSE: print("getting keys from item[keywords]", item["keywords"])
            # construct and clean keys_list (skip_keys, key2key)
            keywords = item.get("keywords", None)
            ethnicity = item.get("ethnicity", None)
            if keywords:
                keys_list = construct_keys_list(item["keywords"])
                # get keywords
                key_nos_list = unlock_key_list(item["id"], keys_list, keys_dict)

            if THIS_SITE == 1:
            # image_row = structure_row_adobe(row, ind, keys_list)
                image_row = structure_row_getty(item, ind, keys_list)
            elif THIS_SITE == 2 and not LOC_ONLY:
                image_row = structure_row_shutterstock_loconly(item, ind, keys_list)
                ethnicity = item.get("filters", {}).get("ethnicity", None)
            elif THIS_SITE == 2 and LOC_ONLY:
                image_row = structure_row_shutterstock_loconly(item, ind, keys_list)
            elif THIS_SITE in [3,4,5,6,7,8] and LOC_ONLY:
                image_row = structure_row_loconly(item, ind, keys_list)
            elif THIS_SITE == 6:
                image_row = structure_row_unsplash(item, ind, keys_list)
                search_desc_for_keys = True
                # item["ethnicity"] = None
            elif THIS_SITE == 9:
                image_row, shoot_location = structure_row_alamy(item, ind, keys_list)
                ethnicity = item.get("filters", {}).get("ethnicity", None)

            elif THIS_SITE == 10:
                # en_keys = []
                # for key in keys_list:
                #     try:
                #         en_key = key2key_vcg[key]
                #         en_keys.append(en_key)
                #     except:
                #         print("going to reappend because no key2key for", key) # for gender, loc, eth
                #         en_keys.append(key)
                # keys_list = en_keys
                # print(keys_list)
                image_row, shoot_location = structure_row_VCG(item, ind, keys_list)
                ethnicity = item.get("filters", {}).get("ethnicity", None)
            elif THIS_SITE == 11:
                search_string = item.get("filters", {}).get("search", None)
                if search_string:
                    search_keys = search_string.split(" ")
                    keys_list = keys_list + search_keys
                image_row = structure_row_pixcy(item, ind, keys_list)
                search_desc_for_keys = True
                # item["ethnicity"] = None
            elif THIS_SITE == 12:
                image_row = structure_row_PIXERF(item, ind, keys_list)
                # item["ethnicity"] = None
            elif THIS_SITE == 13:
                # filters = item.get("filters", None) 
                people = item.get("filters", {}).get("people", None)
                if people == "Groups Or Crowd": 
                    print("skipping groups or crowd")
                    continue
                image_row, shoot_location = structure_row_ImagesBazzar(item, ind, keys_list)
            elif THIS_SITE == 14:
                image_row = structure_row_INDIAPB(item, ind, keys_list)
            elif THIS_SITE == 15:
                author = item.get("author", None)
                if " via " in author:
                    print("skipping unsplash or pixabay")
                    continue
                image_row = structure_row_iwaria(item, ind, keys_list)
            elif THIS_SITE == 16:
                image_row = structure_row_nappy(item, ind, keys_list)
            elif THIS_SITE == 17:
                image_row = structure_row_PICHA(item, ind, keys_list)
            elif THIS_SITE == 18:
                image_row = structure_row_AFRIPICS(item, ind, keys_list)

            if shoot_location: keys_list.append(shoot_location)

            if TIMER: 
                print("time to structure row", time.time()-start_timer)
                start_timer = time.time()

            # if the image row has problems, skip it (structure_row saved it to csv)
            if not image_row:
                continue
            
            if not keys_list or search_desc_for_keys == True and image_row['description']:
                desc_key_nos_list = description_to_keys(image_row['description'], image_row['site_image_id'])
                if key_nos_list: key_nos_list = set(key_nos_list + desc_key_nos_list)
                else: key_nos_list = desc_key_nos_list

            
                # >> need to revisit this for row/item <<
            # skipping ethnicity for LOC_ONLY reprocessing. 
            if not LOC_ONLY:
                if not pd.isnull(ethnicity) and len(ethnicity)>0:
                    print("have eth", ethnicity.lower(), "type is", type(ethnicity.lower()))
                    if VERBOSE: print("len of eth_dict", len(eth_dict))
                    eth_no_list = get_key_no_dictonly(ethnicity.lower(), keys_list, eth_dict)
                    if VERBOSE: print("eth_no_list after get_key_no_dictonly", eth_no_list)
                else:
                    # get eth from keywords, using keys_list and eth_keys_dict
                    print("UNLOCKING KEYS FOR eth_keys_dict <><><><><><><><>")
                    # absence of search string (ie "None") triggers search_keys function
                    eth_no_list = get_key_no_dictonly(None, keys_list, eth_keys_dict)
                    print(eth_no_list)
                    if not eth_no_list:
                        eth_no_list = get_key_no_dictonly(None, keys_list, eth_all_dict, True)
                        if eth_no_list: 
                            print(f"_secondary found for {eth_no_list}")
                        elif "descent" in keys_list:
                            print(f"descent in keys_list {keys_list}")

                if TIMER: 
                    print("time to description and ethnicity", time.time()-start_timer)
                    start_timer = time.time()

                # look for multi_dict, and add to eth_no_list
                if not 6 in eth_no_list and image_row['description'] and THIS_SITE != 10:
                    is_multi = False
                    key_soup = " ".join(keys_list)+" "+image_row['description']

                    for multi in multi_dict:
                        if multi in key_soup:
                            print("multi found", multi)
                            is_multi = True
                    if is_multi: eth_no_list.append(6)

                if 6 in eth_no_list: 
                    multi_eth_no_list = get_key_no_dictonly(None, keys_list, eth_keys_dict)
                    print("multi_eth_no_list", multi_eth_no_list)
                    #if there are any new values in multi_eth_no_list, add them to eth_no_list
                    for eth in multi_eth_no_list:
                        if eth not in eth_no_list:
                            eth_no_list.append(eth)
                print("finally, eth_no_list", eth_no_list)


                if TIMER: 
                    print("time to multi ethnicity", time.time()-start_timer)
                    start_timer = time.time()




            # STORE THE DATA
            if VERBOSE: print("connecting to DB", io.db)
            if VERBOSE: 
                print(image_row)
                print("key_nos_list", key_nos_list)
                if not LOC_ONLY: print("eth_no_list", eth_no_list)

    
            try:
                # check to see if site_image_id is in already_ingested
                if image_row['site_image_id'] in already_ingested:
                    print("this has been already ingested", image_row['site_image_id'])
                    insert_image_row = False
                    already_image_id = site_image_id_dict.get(image_row['site_image_id'], None)
                elif image_row['location_id'][0] is None:
                    # print("this has a NULL value", image_row['site_image_id'])
                    insert_image_row = False
                    already_image_id = site_image_id_dict.get(image_row['site_image_id'], None)
                else:
                    insert_image_row = True
                    # add to already_ingested
                    already_ingested.add(image_row['site_image_id'])
                    # if len(already_ingested) % 100 == 0:
                    #     print(len(already_ingested), " total ingested")



                with engine.connect() as conn:
                #     select_stmt = select(Images).where(
                #         (Images.site_name_id == image_row['site_name_id']) &
                #         (Images.site_image_id == image_row['site_image_id'])
                #     )
                #     row = conn.execute(select_stmt).fetchone()

                    if TIMER: 
                        print("time to check if already exists", time.time()-start_timer)
                        start_timer = time.time()

                    if insert_image_row and not LOC_ONLY:
                        insert_stmt = insert(Images).values(image_row)
                        if VERBOSE: print(str(insert_stmt))
                        dialect = mysql.dialect()
                        statement = str(insert_stmt.compile(dialect=dialect))
                        if VERBOSE: print(statement)
                        # continue

                        try:
                            result = execute_query_with_retry(conn, insert_stmt)  # Retry on OperationalError
                        except Exception as e:
                            print(f"Exception occurred: {e}")
                        # result = execute_query_with_retry(conn, insert_stmt)  # Retry on OperationalError
                        if TIMER: 
                            print("time to insert Images", time.time()-start_timer)
                            start_timer = time.time()

                        if key_nos_list and result.lastrowid:
                            # add image_id to site_image_id_dict
                            site_image_id_dict[image_row['site_image_id']] = result.lastrowid

                            keyrows = [{'image_id': result.lastrowid, 'keyword_id': keyword_id} for keyword_id in key_nos_list]
                            with engine.connect() as conn:
                                imageskeywords_insert_stmt = insert(ImagesKeywords).values(keyrows)
                                imageskeywords_insert_stmt = imageskeywords_insert_stmt.on_duplicate_key_update(
                                    keyword_id=imageskeywords_insert_stmt.inserted.keyword_id
                                )
                                execute_query_with_retry(conn, imageskeywords_insert_stmt)  # Retry on OperationalError
                        if TIMER: 
                            print("time to insert Keys", time.time()-start_timer)
                            start_timer = time.time()

                        if eth_no_list and result.lastrowid:
                            ethrows = [{'image_id': result.lastrowid, 'ethnicity_id': ethnicity_id} for ethnicity_id in eth_no_list if ethnicity_id is not None]
                            if ethrows:
                                with engine.connect() as conn:
                                    imagesethnicity_insert_stmt = insert(ImagesEthnicity).values(ethrows)
                                    imagesethnicity_insert_stmt = imagesethnicity_insert_stmt.on_duplicate_key_update(
                                        ethnicity_id=imagesethnicity_insert_stmt.inserted.ethnicity_id
                                    )
                                    execute_query_with_retry(conn, imagesethnicity_insert_stmt)  # Retry on OperationalError
                        if TIMER: 
                            print("time to insert eth", time.time()-start_timer)
                            start_timer = time.time()

                        print("last_inserted_id:", result.lastrowid)
                        print(" ")
                        if VERBOSE:
                            with engine.connect() as conn:
                                select_stmt = select(Images).where(
                                    (Images.image_id == result.lastrowid)
                                )
                                row = conn.execute(select_stmt).fetchone()
                                print(result.lastrowid, "row:", row)
                                print(" ")

                    elif LOC_ONLY:
                        if not insert_image_row: continue
                        # add any new loc_no_list values
                        location_id = image_row['location_id']
                        image_id = site_image_id_dict.get(image_row['site_image_id'], None)
                        if location_id and image_id:
                            print(' --- >>>> adding new LOCATION:', new_loc_counter, image_id, location_id)
                            new_loc_counter += 1
                            with engine.connect() as conn:
                                # Perform the update using SQLAlchemy Core
                                update_stmt = update(Images).where(Images.image_id == image_id).values(location_id=location_id)
                                execute_query_with_retry(conn, update_stmt)

                    else:
                        # if it already exists, add any new eth_no_list values
                        print('Row already exists:', ind, already_image_id)
                        with engine.connect() as conn:
                            select_stmt = select(ImagesEthnicity).where(
                                (ImagesEthnicity.image_id == already_image_id) 
                            )
                            eth_already = conn.execute(select_stmt).fetchall()
                            # print(eth_already)
                            if eth_already:
                                for eth in eth_already:
                                    if eth[1] in eth_no_list:
                                        # print("removing this eth", eth[1])
                                        #remove eth_already from eth_no_list
                                        eth_no_list.remove(eth[1])
                            else:
                                print("eth_already is None")
                                # print(eth_already)
                                # print(eth_no_list)
                                print(" ")

                            # if we still have any values in eth_no_list, insert them
                            if eth_no_list:
                                ethrows = [{'image_id': already_image_id, 'ethnicity_id': ethnicity_id} for ethnicity_id in eth_no_list if ethnicity_id is not None]
                                if ethrows:
                                    if VERBOSE: print("going to insert ", ethrows)
                                    with engine.connect() as conn:
                                        imagesethnicity_insert_stmt = insert(ImagesEthnicity).values(ethrows)
                                        imagesethnicity_insert_stmt = imagesethnicity_insert_stmt.on_duplicate_key_update(
                                            ethnicity_id=imagesethnicity_insert_stmt.inserted.ethnicity_id
                                        )
                                        execute_query_with_retry(conn, imagesethnicity_insert_stmt)  # Retry on OperationalError
                            else:
                                if VERBOSE: print("eth_no_list is empty, nothing to insert")
                                # print(eth_already)
                                # print(eth_no_list)
                                print(" ")



            except Exception as e:
                print(f"An error occurred while connecting to DB: {e}")

            finally:
                # Close the session
                session.close()

            if counter % 1000 == 0:
                print(counter, " images ingested")
                save_counter = [counter]
                write_csv(CSV_COUNTOUT_PATH, save_counter)
            if TIMER: 
                print("time to store", time.time()-start_timer)
                print(" ")
                start_timer = time.time()
            
            counter += 1
            ind += 1


if __name__ == '__main__':
    print(sig)
    try:
        init_csv(CSV_NOKEYS_PATH,IMG_KEYWORD_HEADERS)
        init_csv(CSV_IMAGEKEYS_PATH,IMG_KEYWORD_HEADERS)
        keys_dict = make_key_dict(KEYWORD_PATH)
        keys_set = set(keys_dict.keys())
        print("this many keys", len(keys_dict))
        locations_dict = make_key_dict(LOCATION_PATH)
        locations_dict_alt = make_key_dict_col3(LOCATION_PATH)
        locations_dict_getty = make_key_dict_getty(LOCATION_PATH)
        locations_dict_AA = make_key_dict_col7(LOCATION_PATH)
        print(locations_dict)
        print(locations_dict_alt)
        print(locations_dict_AA)
        print("this many locations", len(locations_dict))

        ingest_json()
        print("finished with ingest")
    except KeyboardInterrupt as _:
        print('[-] User cancelled.\n', flush=True)
    except Exception as e:
        print('[__main__] %s' % str(e), flush=True)
        print(traceback.format_exc())
        # or
        print(sys.exc_info()[2])
    





