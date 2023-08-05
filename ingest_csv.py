from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import csv
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

import numpy as np
import pandas as pd
from pyinflect import getInflection

from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, String, VARCHAR, ForeignKey, Date, update, insert, select, PrimaryKeyConstraint, UniqueConstraint
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.mysql import insert
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
io.db["name"] = "gettytest3"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#######################################

INGEST_ROOT = "/Users/michaelmandiberg/Documents/projects-active/facemap_production"
# INGEST_FOLDER = os.path.join(INGEST_ROOT, "adobe_csv_4ingest/")
# CSV_IN_PATH = os.path.join(INGEST_FOLDER, "unique_lines_B_nogender.csv")
INGEST_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/iStock_ingest/"
CSV_IN_PATH = os.path.join(INGEST_FOLDER, "april15_iStock_output_sample.jsonl.csv")
KEYWORD_PATH = os.path.join(INGEST_FOLDER, "Keywords_202305150950.csv")
LOCATION_PATH = os.path.join(INGEST_FOLDER, "Location_202308041952.csv")
CSV_NOKEYS_PATH = os.path.join(INGEST_FOLDER, "CSV_NOKEYS.csv")
CSV_IMAGEKEYS_PATH = os.path.join(INGEST_FOLDER, "CSV_IMAGEKEYS.csv")
# NEWIMAGES_FOLDER_NAME = 'images_pexels'
CSV_COUNTOUT_PATH = os.path.join(INGEST_FOLDER, "countout.csv")
CSV_NOLOC_PATH = os.path.join(INGEST_FOLDER, "CSV_NOLOC.csv")

# key2key = {"person":"people", "kid":"child","affection":"Affectionate", "baby":"Baby - Human Age", "beautiful":"Beautiful People", "pretty":"Beautiful People", "blur":"Blurred Motion", "casual":"Casual Clothing", "children":"Child", "kids":"Child", "couple":"Couple - Relationship", "adorable":"Cute", "room":"Domestic Room", "focus":"Focus - Concept", "happy":"Happiness", "at home":"Home Interior", "home":"Home Interior", "face":"Human Face", "hands":"Human Hand", "landscape":"Landscape - Scenery", "outfit":"Landscape - Scenery", "leisure":"Leisure Activity", "love":"Love - Emotion", "guy":"Men", "motherhood":"Mother", "parenthood":"Parent", "positive":"Positive Emotion", "recreation":"Recreational Pursuit", "little":"Small", "studio shoot":"Studio Shot", "together":"Togetherness", "vertical shot":"Vertical", "lady":"women", "young":"Young Adult"}
loc2loc = {"niue":"Niue Island"}
key2key = {"person":"people", "kid":"child","affection":"affectionate", "baby":"baby - human age", "beautiful":"beautiful people", "pretty":"beautiful people", "blur":"blurred motion", "casual":"casual clothing", "children":"child", "kids":"child", "couple":"couple - relationship", "adorable":"cute", "room":"domestic room", "focus":"focus - concept", "happy":"happiness", "at home":"home interior", "home":"home interior", "face":"human face", "hands":"human hand", "landscape":"landscape - scenery", "outfit":"landscape - scenery", "leisure":"leisure activity", "love":"love - emotion", "guy":"men", "motherhood":"mother", "parenthood":"parent", "positive":"positive emotion", "recreation":"recreational pursuit", "little":"small", "studio shoot":"studio shot", "together":"togetherness", "vertical shot":"vertical", "lady":"women", "young":"young adult", "light":"light - natural phenomenon", "trees":"tree", "disabled":"disability", "landline":"phone", "tradesman":"worker", "apprentice":"work", "arbeit":"work", "wheel-chair":"wheelchair", "treatments":"treatment", "transports":"transportation", "thoughtfully":"thoughtful", "technologies":"technology", "piscine":"swim", "astonished":"surprise", "surgeons":"surgeon", "sommer":"summer", "suffering":"suffer", "studentin":"student", "stressful":"stressed", "smoothies":"smoothie", "smilling":"smiling", "kleines":"small", "sleeps":"sleeping", "dealership":"sales", "salads":"salad", "ressources":"resources", "relaxes":"relaxed", "presentations":"presentation", "phones":"phone", "telefon":"phone", "telefoniert":"phone", "patients":"patient", "papier":"paper", "painful":"pain", "offended":"offend", "occupations":"occupation", "muscled":"muscles", "motivated":"motivation", "pinup":"model", "pin-up":"model", "meetings":"meeting", "massages":"massage", "kleiner":"little", "(lawyer)":"lawyer", "kitchens":"kitchen", "injections":"injection", "hospitals":"hospital", "zuhause":"home", "happily":"happy", "joyfully":"happy", "overjoyed":"happiness", "rejoices":"happiness", "handshaking":"handshake", "groups":"group", "full-length":"Full Length", "blumen":"flowers", "florists":"florist", "panic":"fear", "fell":"fall", "equipements":"equipement", "enthusiastic":"enthusiasm", "osteopathy":"doctor", "disgusted":"disgust", "schreibtisch":"desk", "dances":"dancing", "crowds":"crowd", "robber":"criminal", "copyspace":"Copy Space", "misunderstandings":"confusion", "confidently":"confidence", "concerts":"concert", "climbs":"climb", "celebrations":"celebration", "caught":"catch", "casually":"casual", "motorsports":"car", "banker":"Business Person", "supervisor":"boss", "executives":"boss", "bedrooms":"bedroom", "beautifull":"beautiful", "beaches":"beach", "bathrooms":"bathroom", "backgroud":"background", "attraktive":"attractive", "sportwear":"athletic", "sportliche":"athletic", "addicted":"addiction", "alcoholism":"addiction"}
gender_dict = {"men":1,"man":1,"male":1,"males":1,"his":1,"him":1,"businessman":1,"businessmen":1,"father":1, "men's":1, "himself":1, "homme":1, "hombre":1, "(man)":1, "-women men -children":1, "-women -men -children":2, "none":2, "oldmen":3, "grandfather":3,"oldwomen":4, "grandmother":4, "nonbinary":5, "other":6, "trans":7, 
        "women":8,"woman":8,"female":8,"females":8, "hers":8, "her":8, "businesswoman":8, "businesswomen":8, "mother":8, "frauen":8, "mujer":8, "haaren":8, "frau":8, "woman-doctor":8, "maiden":8, "hausfrau":8, "women -men -children":8, "youngmen":9, "boy":9, "boys":9, "jungen":9, "youngwomen":10,"girl":10, "girls":10, "ragazza":10, "schoolgirls":8,}
gender_dict_istock = {"Mid Adult Men":1, "Only Mid Adult Men":1, "One Mid Adult Man Only":1, "Only Men":1, "One Man Only":1, "Senior Men":3, "Only Senior Men":3, "One Senior Man Only":3, "Mature Men":3, "Only Mature Men":3, "One Mature Man Only":3, "Mature Women":4, "Only Mature Women":4, "One Mature Woman Only":4, "Senior Women":4, "Only Senior Women":4, "One Senior Woman Only":4, "Mid Adult Women":8, "Only Mid Adult Women":8, "One Mid Adult Woman Only":8, "Only Women":8, "One Woman Only":8, "Young Men":9, "Only Young Men":9, "One Young Man Only":9, "Teenage Boys":9, "Only Teenage Boys":9, "One Teenage Boy Only":9, "Only Boys":9, "One Boy Only":9, "Baby Boys":9, "Only Baby Boys":9, "One Baby Boy Only":9, "Young Women":10, "Only Young Women":10, "One Young Woman Only":10, "Teenage Girls":10, "Only Teenage Girls":10, "One Teenage Girl Only":10, "Only Girls":10, "One Girl Only":10, "Baby Girls":10, "Only Baby Girls":10, "One Baby Girl Only":10}
# gender2key = {"man":"men", "woman":"women"}
eth_dict = {"black":1, "african-american":1, "afro-american":1, "africanamerican":1, "african american":1, "african":1, "indigenous peoples of africa":1, "african ethnicity":1, "african-american ethnicity":1, "caucasian":2, "white people":2, "europeans":2, "eastasian":3,"east asian":3, "chinese":3, "japanese":3, "asian":3, "hispaniclatino":4, "latino":4, "latina":4, "latinx":4, "hispanic":4, "mexican":4, "middleeastern":5, "middle eastern":5, "arab":5, "mixedraceperson":6, "mixedrace":6, "mixed-race":6, "mixed race":6, "mixed ethnicity":6, "multiethnic":6, "multi ethnic":6, "multi-ethnic":6, "biracial":6, "nativeamericanfirstnations":7, "native american":7, "nativeamerican":7, "native-american":7, "indian american":7, "indianamerican":7, "indian-american":7, "first nations":7, "firstnations":7, "first-nations":7, "indigenous":7, "pacificislander":8, "pacific islander":8, "pacific-islander":8, "southasian":9, "south asian":9, "south-asian":9, "indian":9, "southeastasian":10, "southest asian":10, "southeast asian":10, "southeast-asian":10}
eth_dict_istock = {"Northern European Descent":2, "Scandinavian Descent":2, "Southern European Descent":2, "East Asian Ethnicity":3, "Japanese Ethnicity":3, "Chinese Ethnicity":3, "Southeast Asian Ethnicity":10, "South Asian Ethnicity":9, "West Asian Ethnicity":5, "North African Ethnicity":5, "African-American Ethnicity":1, "Latin American and Hispanic Ethnicity":4, "Cuban Ethnicity":4, "Puerto Rican Ethnicity":4, "Mexican Ethnicity":4, "Multiracial Group":6, "Multiracial Person":6}
# for searching descrption for eth keywords, get rid of ambiguous/polyvalent terms
eth_keys_dict = eth_dict
for k in ['black', 'african']: eth_keys_dict.pop(k)

# load Keywords_202304300930.csv as df, drop all but keytype Locations, create two dicts: string->ID & GettyID->ID  
# loc_dict = {"Canada":1989}

age_dict = {
    "newborn":1,
    "baby":1,
    "infant":2,
    "infants":2,
    "toddlers":3,
    "toddler":3,
    "child":3,
    "children":3,
    "childrens":3,
    "girls":3,
    "boys":3,
    "girl":3,
    "boy":3,
    "jeune":3,
    "junger":3,
    "kinder":3,
    "bambino":3,
    "bambina":3,
    "ragazza":3, 
    "schoolgirls":3, 
    "jungen":3,
    "-women -men children":3,
    "teen":4,
    "teens":4,
    "teenager":4,
    "teenagers":4,
    "young adult":5,
    "young man":5,
    "young woman":5,
    "young men":5,
    "young women":5,
    "young":5,
    "20s":5,
    "30s":5,
    "adult":6,
    "40s":6,
    "50s":6,
    "old":7,
    "60s":7,
    "70s":7,
    "70+":7,
    "seniorin":7
}
age_dict_istock = {"0-1 Months":1, "0-11 Months":1, "Babies Only":1, "2-5 Months":2, "6-11 Months":2, "Preschool Age":2, "12-17 Months":3, "2-3 Years":3, "4-5 Years":3, "6-7 Years":3, "8-9 Years":3, "10-11 Years":3, "12-13 Years":3, "12-23 Months":3, "Elementary Age":3, "Pre-Adolescent Child":3, "Children Only":3, "18-23 Months":3, "14-15 Years":4, "16-17 Years":4, "18-19 Years":4, "Teenagers Only":4, "20-24 Years":5, "25-29 Years":5, "30-34 Years":5, "20-29 Years":5, "30-39 Years":5, "35-39 Years":6, "40-44 Years":6, "45-49 Years":6, "50-54 Years":6, "55-59 Years":6, "Adults Only":6, "Mid Adult":6, "Mature Adult":6, "40-49 Years":6, "50-59 Years":6, "60-64 Years":7, "65-69 Years":7, "70-79 Years":7, "Senior Adult":7, "60-69 Years":7, "80-89 Years":7, "Over 100":7, "90 Plus Years":7}

age_details_dict = {
    'toddler': 2,
    '20s': 4,
    '30s': 5,
    '40s': 6,
    '50s': 7,
    '60s': 8,
    '70+': 9
}
age_detail_dict_istock = {"0-1 Months":1, "2-5 Months":1, "0-11 Months":1, "Babies Only":1, "12-17 Months":2, "2-3 Years":2, "6-11 Months":2, "12-23 Months":2, "18-23 Months":2, "12-13 Years":3, "14-15 Years":3, "16-17 Years":3, "18-19 Years":3, "20-24 Years":4, "25-29 Years":4, "20-29 Years":4, "30-34 Years":5, "35-39 Years":5, "30-39 Years":5, "40-44 Years":6, "45-49 Years":6, "40-49 Years":6, "50-54 Years":7, "55-59 Years":7, "50-59 Years":7, "60-64 Years":8, "65-69 Years":8, "60-69 Years":8, "70-79 Years":9, "80-89 Years":9, "Over 100":9, "90 Plus Years":9, "4-5 Years":10, "6-7 Years":10, "8-9 Years":11, "10-11 Years":11}

def lower_dict(this_dict):
    lower_dict = {k.lower(): v for k, v in this_dict.items()}
    return lower_dict

gender_dict = lower_dict({**gender_dict, **gender_dict_istock})
eth_dict = lower_dict({**eth_dict, **eth_dict_istock})
age_dict = lower_dict({**age_dict, **age_dict_istock})
age_details_dict = lower_dict({**age_details_dict, **age_detail_dict_istock})





# table_search ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id"
SELECT = "DISTINCT(i.image_id), i.gender_id, author, caption, contentUrl, description, imagename"
FROM ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings e ON i.image_id = e.image_id "
WHERE = "e.image_id IS NULL"
LIMIT = 10


# engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
#                                 .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

metadata = MetaData(engine)


images_table = Table('Images', metadata,
    Column('image_id', Integer, primary_key=True, autoincrement=True),
    Column('site_name_id', Integer, ForeignKey('Site.site_name_id')),
    Column('site_image_id', String(50), nullable=False),
    Column('age_id', Integer, ForeignKey('Age.age_id')),
    Column('age_detail_id', Integer, ForeignKey('AgeDetail.age_detail_id')),
    Column('gender_id', Integer, ForeignKey('Gender.gender_id')),
    Column('location_id', Integer, ForeignKey('Location.location_id')),
    Column('author', String(100)),
    Column('caption', String(150)),
    Column('contentUrl', String(200), nullable=False),
    Column('description', String(150)),
    Column('imagename', String(100)),
    Column('uploadDate', Date)
)

imageskeywords_table = Table('ImagesKeywords', metadata,
    Column('image_id', Integer, ForeignKey('Images.image_id')),
    Column('keyword_id', Integer, ForeignKey('Keywords.keyword_id')),
    PrimaryKeyConstraint('image_id', 'keyword_id'),
    UniqueConstraint('image_id', 'keyword_id', name='uq_image_keyword')
)

imagesethnicity_table = Table('ImagesEthnicity', metadata,
    Column('image_id', Integer, ForeignKey('Images.image_id')),
    Column('ethnicity_id', Integer, ForeignKey('Ethnicity.ethnicity_id')),
    PrimaryKeyConstraint('image_id', 'ethnicity_id'),
    UniqueConstraint('image_id', 'ethnicity_id', name='uq_image_keyword')
)

PEXELS_HEADERS = ["id", "title", "keywords", "country", "number_of_people", "orientation", "age","gender", "ethnicity", "mood", "image_url", "image_filename"]
ONETWOTHREE_HEADERS = ["id","title","keywords","orientation","age","word","exclude","people","ethnicity","image_url","image_filename"]
KEYWORD_HEADERS = ["keyword_id","keyword_number","keyword_text","keytype","weight","parent_keyword_id","parent_keyword_text"]
IMG_KEYWORD_HEADERS = ["site_image_id","keyword_text"]
header_list = PEXELS_HEADERS

def read_csv(csv_file):
    with open(csv_file, encoding="utf-8", newline="") as in_file:
        reader = csv.reader(in_file, delimiter=",")
        next(reader)  # Header row

        for row in reader:
            yield row

def write_csv(path,value_list):

    # headers = header_list

    with open(path, 'a') as csvfile: 
    # with open('lat_lon', 'w') as csvfile:
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

def make_key_dict(filepath, keytype=None):
    keys = read_csv(filepath)
    keys_dict = {}
    for row in keys:
        if keytype is None:
            keys_dict[row[2].lower()] = row[0]
        elif keytype and row[3].lower() == keytype:
            # print("gotta keytype")
            keys_dict[row[2].lower()] = row[0]
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




# abstraction to run a list through unlock_key_plurals_etc
def unlock_key_list(site_image_id, keys_list, keys_dict):
    key_nos_list = []
    
    for key in keys_list:
        key_no = unlock_key_plurals_etc(site_image_id.lower(), key, keys_dict)
        # print(key_no)
        if key_no:
            key_nos_list.append(key_no)
    return key_nos_list

# takes a key and runs all permutations through the dict, and saves missing ones
# this is the kitchen sink function
def unlock_key_plurals_etc(site_id,key, this_dict):
    key_no = None
    key = key.lower()
    try:
        # print("trying basic keys_dict for this key:")
        # print(key)
        # print("from dict this long")
        # print(len(this_dict))
        # # print(this_dict)

        # print(this_dict["office"])
        key_no = this_dict[key]
        # print("this is the key_no")
        # print(key_no)
        return(key_no)
    except:
        try:
            key = key2key[key]
            key_no = this_dict[key]
            return(key_no)
        except:
            # try:
            # try with inflections
            # print(site_id)
            # print(key)

            plur_key = getInflection(key, 'NNS')
            sing_key = getInflection(key, 'NN')
            gerund_key = getInflection(key, 'VBG')
            # print("inflected are: ", plur_key, sing_key, gerund_key)
            if plur_key and key != plur_key:
                try:
                    key_no = this_dict[plur_key[0]]
                    # key = plur_key
                    print(key_no)
                except:
                    pass# print(key)
            elif sing_key and key != sing_key:
                try:
                    key_no = this_dict[sing_key[0]]
                    # key = plur_key
                    # print(key_no)
                except:
                    pass
                    # print(key)
            
            if gerund_key and key != gerund_key:
                try:
                    key_no = this_dict[gerund_key[0]]
                    # key = plur_key
                    # print("gotta key_no, " ,key_no)
                except:
                    pass
                    # print(key)

            if pd.isnull(key_no):
                # if nothing worked, save key, but only if site_id > 10
                # for gender/age, it passes in site_name_id, not site_image_id
                if not isinstance(site_id, int) and not key.startswith('-'):
                    # print(type(site_id))
                    # print(site_id)
                    value_list = [site_id,key]
                    # print("value_list")
                    # print(value_list)
                    write_csv(CSV_NOKEYS_PATH,value_list)
                # print(value_list)
                return
            else:
                value_list = [site_id,key_no]
                write_csv(CSV_IMAGEKEYS_PATH,value_list)
                return key_no


# finds dict items in a string
# called by search_keys
def findall_dict(my_dict,description):
    # Create a regular expression pattern that matches complete words in the dictionary, ignoring case
    pattern = re.compile(r'\b(' + '|'.join(my_dict.keys()) + r')\b', re.IGNORECASE)

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
# called in by the get_eth_dictonly
def search_keys(keys_list, this_dict, multi=False):
    results = []
    for key in keys_list:
        found = findall_dict(this_dict,key)
        if found is not None:
            results.append(found)
            print('found it in keywords:', found,"from key:", key)
            #age needs to be int()
            # print(results)

    if len(set(results)) == 1:
        one_result = int(results[0])
        # print("found a GOOD result: ", one_result)
    else:
        one_result = 0
        # print("failed search: ", one_result)
    if multi:
        results_list = list(set(results))
    else:
        results_list = [one_result]
    return results_list

# print(key_nos_list)

def get_key_no_dictonly(eth_name, keys_list, this_dict, search_dict=None):
    # eth_name = df['ethnicity'][ind]
    # print('isnan?')
    # print(np.isnan(eth_name))
    key_no_list = []
    key_no = None
    # if eth_name is not None or eth_name is not np.isnan(eth_name):
    if not pd.isnull(eth_name):
        try:
            key_no = unlock_key_dict(eth_name, this_dict)
            # key_no = eth_dict[eth_name.lower()]
        # need to key this into integer, like with keys
            # print("eth_name ",eth_name)
        except:
            key_no = None
            print("eth_dict failed with this key: ", eth_name)
        key_no_list.append(key_no)
    elif search_dict:
        key_no_list = search_keys(keys_list, search_dict, True)
        print("searched keys and found key_no: ", key_no_list)
    return(key_no_list)

def unlock_key_dict(key,this_dict,this_key2key=None):
    key_no = None
    key = key.lower()
    try:
        print("trying basic this_dict for this key:")
        print(key)
        key_no = this_dict[key]
        print("this is the key_no")
        print(key_no)
        return(key_no)
    except:
        if this_key2key:
            try:
                altkey = this_key2key[key.lower()]
                print("altkey")
                print(altkey)
                key_no = this_dict[altkey.lower()]
                print("this is the key_no via loc2loc")
                print(key_no)
                return(key_no)
            except:
                print("NEW KEY -------------------------> ", key)
                write_csv(CSV_NOLOC_PATH,key)
        else:
            print("unlock_key_dict failed, and no key2key for this key: ", key)

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

    description = description.replace(",","").replace("'s","").replace(".","")

    # print("description_to_keys")    
    key_nos_list =[]
    # description = df['title'][ind]
    # print(description)
    key_no = None
    desc_keys = description.split(" ")
    # print("desc_keys ",desc_keys)
    for key in desc_keys:
        # print("checking key ", key)
        if not pd.isnull(key):
            key_no = unlock_key_plurals_etc(site_id,key,this_dict)
            # print("key_no passed through:")
            # print(key_no)
            if key_no:
                key_nos_list.append(key_no)
            # print("key_nos_list ",key_nos_list)
    return key_nos_list



def get_gender_age_row(gender_string, age_string, description, keys_list, site_id):
    def try_gender_age_key(gender, age, age_detail, this_string):
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
                # print(f"first try, gender is {str(gender)} and age is {str(age)}")
                # gender_dict={"men":1, "none":2, "oldmen":3, "oldwomen":4, "nonbinary":5, "other":6, "trans":7, "women":8, "youngmen":9, "youngwomen":10}
            except:
                try:
                    age = age_dict[this_string.lower()]
                    # print(f"second try age is {str(age)}")

                except:
                    # this isn't relevant for iStock
                    print('NEW KEY, NOT AGE OR GENDER -------------------------> ', this_string)
            try:
                age_detail = age_details_dict[this_string.lower()]
            except:
                # no age detail. it is only on some sites
                pass

        # else:
        #     print("string is None")
        return gender, age, age_detail

    def get_mode(data):
        filtered_data = [value for value in data if value is not None]
        
        # Check if the filtered_data is empty before proceeding
        if not filtered_data:
            return None  # or return any other default value you prefer
        
        mode_counts = Counter(filtered_data)
        mode_value, max_count = mode_counts.most_common(1)[0]
        return mode_value

    def prioritize_age_gender(gender_list,age_list):
        if gender_list.count(3) > 0:
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
        gender_list=[]
        age_list=[]
        age_detail_list=[]
        for key in keys_list:
            # print("key is ", key)
            # reset variables
            gender = None
            age= None
            age_detail= None
            gender, age, age_detail = try_gender_age_key(gender, age, age_detail, key)
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

    description = description.replace(",","").replace("'s","").replace(".","")

    print("gender_string, age_string",gender_string, age_string)
    # print("types",type(gender_string), type(age_string))

    # this if/if structure is necessary because "" and isnull were not compatible
    # Get gender
    if gender_string != "":
        gender, age, age_detail = try_gender_age_key(gender, age, age_detail, gender_string)
    else:
        gender, age, age_detail = get_gender_age_keywords(gender, age, age_detail, keys_list)
        print(gender)
        print(age)
        print(age_detail)


        #try keys for gender
    # print("gender, age, after try key gender_string")
    # print(gender)
    # print(age)
    if pd.isnull(gender): 
        # print("gender is null")
        gender_results = description_to_keys(description, site_id, gender_dict)
        if len(set(gender_results)) == 1:
            gender = gender_results[0]
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
    if pd.isnull(age):
        print("age is still null, trying keywordsa again, but not for gender")
        _, age, age_detail = get_gender_age_keywords(gender, age, age_detail, keys_list)
        print(gender)
        print(age)
        print(age_detail)
        if pd.isnull(age):
            print("age is really still null, trying description")

            age_results = description_to_keys(description, site_id, age_dict)
            if len(set(age_results)) == 1:
                age = age_results[0]


    print("gender, age, after everything")
    print(gender)
    print(age)

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
    print (os.path.join(hash_folder, hash_subfolder,file_name))
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


def structure_row_istock(row, ind, keys_list):
    site_id = 4 #id for the site, not the image
    gender = row[7]
    age = row[6]
    description = row[1]
    gender_key, age_key, age_detail_key = get_gender_age_row(gender, age, description, keys_list, site_id)
    country_key = unlock_key_dict(row[3],locations_dict, loc2loc)

    image_row = {
        "location_id": country_key,        
        "site_image_id": row[0],
        "site_name_id": site_id,
        "description": description[:140],
        "age_id": age_key,
        "gender_id": gender_key,
        "age_detail_id": age_detail_key,
        "contentUrl": row[10],
        "imagename": generate_local_unhashed_image_filepath(row[11].replace("images/","").split("?")[0])  # need to refactor this from the contentURL using the hash function
    }
    
    return nan2none(image_row)


def ingest_csv():

    # change this for each site ingested #
    # adobe
    # column_keys = 6 #where the keywords are
    # separator_keys = " " #for keywords, in the column listed above
    # column_site = 8 #not sure this is used
    # column_eth = None #ethnicity
    # search_desc_for_keys = True

    # # istock
    column_keys = 2 #where the keywords are
    separator_keys = "|" #for keywords, in the column listed above
    column_site = 8 #not sure this is used
    column_eth = None #ethnicity
    search_desc_for_keys = True


    with open(CSV_IN_PATH) as file_obj:
        reader_obj = csv.reader(file_obj)
        next(reader_obj)  # Skip header row
        start_counter = get_counter()
        print("start_counter: ", start_counter)
        # start_counter = 0 #temporary for testing, while adobe is ongoing
        counter = 0
        ind = 0
        
        for row in reader_obj:
            # print(row[1])
            
            if counter < start_counter:
                counter += 1
                continue
            
            # splitting keys. try is for getty, except is currently set for pexels.
            try:
                keys_list = row[column_keys].lower().split(separator_keys)

                # # adobe specific. remove for future sites
                # desc_list = row[3].replace(",","").lower().split("-")
                # keys_list = list(filter(None, keys_list + desc_list))

            except IndexError:
                print("keys failed")
            print("keys_list")
            print(keys_list)


            # image_row = structure_row_adobe(row, ind, keys_list)
            image_row = structure_row_istock(row, ind, keys_list)
            
            site_image_id = image_row['site_image_id']

            # get keywords
            key_nos_list = unlock_key_list(site_image_id, keys_list, keys_dict)

            if search_desc_for_keys == True:
                desc_key_nos_list = description_to_keys(image_row['description'], image_row['site_image_id'])
                key_nos_list = set(key_nos_list + desc_key_nos_list)
            

            if column_eth:
                # print(key_nos_list)
                eth_no_list = get_key_no_dictonly(row[column_eth].lower(), keys_list, eth_dict)
            else:
                # get eth from keywords, using keys_list and eth_keys_dict
                print("UNLOCKING KEYS FOR eth_keys_dict <><><><><><><><>")
                eth_no_list = get_key_no_dictonly(None, keys_list, eth_dict, eth_keys_dict)
                print(eth_no_list)

            # STORE THE DATA
            with engine.connect() as conn:
                select_stmt = select([images_table]).where(
                    (images_table.c.site_name_id == image_row['site_name_id']) &
                    (images_table.c.site_image_id == image_row['site_image_id'])
                )
                row = conn.execute(select_stmt).fetchone()
                
                if row is None:
                    insert_stmt = insert(images_table).values(image_row)
                    result = conn.execute(insert_stmt)
                    last_inserted_id = result.lastrowid

                    if key_nos_list and last_inserted_id:
                        keyrows = [{'image_id': last_inserted_id, 'keyword_id': keyword_id} for keyword_id in key_nos_list]
                        with engine.connect() as conn:
                            imageskeywords_insert_stmt = insert(imageskeywords_table).values(keyrows)
                            imageskeywords_insert_stmt = imageskeywords_insert_stmt.on_duplicate_key_update(
                                keyword_id=imageskeywords_insert_stmt.inserted.keyword_id
                            )
                            conn.execute(imageskeywords_insert_stmt)
                    
                    if eth_no_list and last_inserted_id:
                        ethrows = [{'image_id': last_inserted_id, 'ethnicity_id': ethnicity_id} for ethnicity_id in eth_no_list if ethnicity_id is not None]
                        if ethrows:
                            with engine.connect() as conn:
                                imagesethnicity_insert_stmt = insert(imagesethnicity_table).values(ethrows)
                                imagesethnicity_insert_stmt = imagesethnicity_insert_stmt.on_duplicate_key_update(
                                    ethnicity_id=imagesethnicity_insert_stmt.inserted.ethnicity_id
                                )
                                conn.execute(imagesethnicity_insert_stmt)
                    
                    print("last_inserted_id:", last_inserted_id)
                else:
                    print('Row already exists:', ind)
            
            if counter % 1000 == 0:
                save_counter = [counter]
                write_csv(CSV_COUNTOUT_PATH, save_counter)
            
            counter += 1
            ind += 1

'''
# I don't think this is doing anything right now, and I don't know how it is different from ingest_csv
def update_csv():

    # change this for each site ingested
    column_keys = 5
    separator_keys = " "
    # column_site = 8
    column_eth = None
    search_desc_for_keys = True


    with open(CSV_IN_PATH) as file_obj:
        reader_obj = csv.reader(file_obj)
        next(reader_obj)  # Skip header row
        start_counter = get_counter()
        counter = 0
        ind = 0
        
        for row in reader_obj:
            # print(row[1])
            
            if counter < start_counter:
                counter += 1
                continue
            if counter >4001000:
                quit()
            # splitting keys. try is for getty, except is currently set for pexels.
            try:
                keys_list = row[column_keys].lower().split(separator_keys)
            except IndexError:
                print("keys failed")
            # print(keys_list)

            image_row = structure_row_123_asrow(row, ind, keys_list)
            key_nos_list = []
            
            for key in keys_list:
                key_no = unlock_key_plurals_etc(image_row['site_image_id'].lower(), key, keys_dict)
                # print(key_no)
                if key_no:
                    key_nos_list.append(key_no)
            
            # print(key_nos_list)

            if search_desc_for_keys == True:
                desc_key_nos_list = description_to_keys(image_row['description'], image_row['site_image_id'])
                key_nos_list = set(key_nos_list + desc_key_nos_list)
            
            if column_eth:
                # print(key_nos_list)
                eth_no_list = get_eth(row[column_eth].lower(), keys_list)
                print("eth_no_list " , eth_no_list)
            else:
                eth_no_list = None


            # Define the maximum number of retries and the delay between retries
            max_retries = 3
            retry_delay = 20  # in seconds

            # Retry loop
            for retry in range(max_retries):
                try:
                    with engine.connect() as conn:
                        select_stmt = select([images_table]).where(
                            (images_table.c.site_name_id == image_row['site_name_id']) &
                            (images_table.c.site_image_id == image_row['site_image_id'])
                        )
                        row = conn.execute(select_stmt).fetchone()
                        
                        if row is None:

                            print("will insert")
                            # insert_stmt = insert(images_table).values(image_row)
                            # result = conn.execute(insert_stmt)
                            # last_inserted_id = result.lastrowid

                            # if key_nos_list and last_inserted_id:
                            #     keyrows = [{'image_id': last_inserted_id, 'keyword_id': keyword_id} for keyword_id in key_nos_list]
                            #     with engine.connect() as conn:
                            #         imageskeywords_insert_stmt = insert(imageskeywords_table).values(keyrows)
                            #         imageskeywords_insert_stmt = imageskeywords_insert_stmt.on_duplicate_key_update(
                            #             keyword_id=imageskeywords_insert_stmt.inserted.keyword_id
                            #         )
                            #         conn.execute(imageskeywords_insert_stmt)
                            
                            # if eth_no_list and last_inserted_id:
                            #     ethrows = [{'image_id': last_inserted_id, 'ethnicity_id': ethnicity_id} for ethnicity_id in eth_no_list if ethnicity_id is not None]
                            #     if ethrows:
                            #         with engine.connect() as conn:
                            #             imagesethnicity_insert_stmt = insert(imagesethnicity_table).values(ethrows)
                            #             imagesethnicity_insert_stmt = imagesethnicity_insert_stmt.on_duplicate_key_update(
                            #                 ethnicity_id=imagesethnicity_insert_stmt.inserted.ethnicity_id
                            #             )
                            #             conn.execute(imagesethnicity_insert_stmt)
                            
                            # print("last_inserted_id:", last_inserted_id)
                        else:
                            print('Row already exists:', ind)
            
                        break  # If the execution reaches here without exceptions, exit the loop

                except OperationalError as e:
                    print(f"Database connection error: {e}")

                    # Retry after a delay
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue

            else:
                # If the loop completes without a successful connection, handle the error
                print(f"Failed to connect to the database after {max_retries} attempts.")

            if counter % 1000 == 0:
                save_counter = [counter]
                write_csv(CSV_COUNTOUT_PATH, save_counter)
            
            counter += 1
            ind += 1


    # print("inserted")
'''

if __name__ == '__main__':
    print(sig)
    try:
        init_csv(CSV_NOKEYS_PATH,IMG_KEYWORD_HEADERS)
        init_csv(CSV_IMAGEKEYS_PATH,IMG_KEYWORD_HEADERS)
        keys_dict = make_key_dict(KEYWORD_PATH)
        print("this many keys", len(keys_dict))
        locations_dict = make_key_dict(LOCATION_PATH)
        print("this many locations", len(locations_dict))

        ingest_csv()
    except KeyboardInterrupt as _:
        print('[-] User cancelled.\n', flush=True)
    except Exception as e:
        print('[__main__] %s' % str(e), flush=True)
        print(traceback.format_exc())
        # or
        print(sys.exc_info()[2])





