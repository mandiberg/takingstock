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

import numpy as np
import pandas as pd
from pyinflect import getInflection

from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, String, VARCHAR, ForeignKey, Date, update, insert, select, PrimaryKeyConstraint, UniqueConstraint
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.mysql import insert

sig = '''

_)                       |                     
 | __ \   _` |  _ \  __| __|   __|  __|\ \   / 
 | |   | (   |  __/\__ \ |    (   \__ \ \ \ /  
_|_|  _|\__, |\___|____/\__| \___|____/  \_/   
        |___/                                  
'''


######## Michael's Credentials ########
db = {
    "host":"localhost",
    "name":"gettytest3",            
    "user":"root",
    "pass":"Fg!27Ejc!Mvr!GT"
}

ROOT= os.path.join(os.environ['HOME'], "Documents/projects-active/facemap_production") ## only on Mac
NUMBER_OF_PROCESSES = 8
#######################################


CSV_IN_PATH = "/Users/michaelmandiberg/Downloads/Pexels_v2/output.csv"
KEYWORD_PATH = "/Users/michaelmandiberg/Downloads/Pexels_v2/Keywords_202304300930.csv"
CSV_NOKEYS_PATH = "/Users/michaelmandiberg/Downloads/Pexels_v2/CSV_NOKEYS.csv"
CSV_IMAGEKEYS_PATH = "/Users/michaelmandiberg/Downloads/Pexels_v2/CSV_IMAGEKEYS.csv"

# key2key = {"person":"people", "kid":"child","affection":"Affectionate", "baby":"Baby - Human Age", "beautiful":"Beautiful People", "pretty":"Beautiful People", "blur":"Blurred Motion", "casual":"Casual Clothing", "children":"Child", "kids":"Child", "couple":"Couple - Relationship", "adorable":"Cute", "room":"Domestic Room", "focus":"Focus - Concept", "happy":"Happiness", "at home":"Home Interior", "home":"Home Interior", "face":"Human Face", "hands":"Human Hand", "landscape":"Landscape - Scenery", "outfit":"Landscape - Scenery", "leisure":"Leisure Activity", "love":"Love - Emotion", "guy":"Men", "motherhood":"Mother", "parenthood":"Parent", "positive":"Positive Emotion", "recreation":"Recreational Pursuit", "little":"Small", "studio shoot":"Studio Shot", "together":"Togetherness", "vertical shot":"Vertical", "lady":"women", "young":"Young Adult"}

key2key = {"person":"people", "kid":"child","affection":"affectionate", "baby":"baby - human age", "beautiful":"beautiful people", "pretty":"beautiful people", "blur":"blurred motion", "casual":"casual clothing", "children":"child", "kids":"child", "couple":"couple - relationship", "adorable":"cute", "room":"domestic room", "focus":"focus - concept", "happy":"happiness", "at home":"home interior", "home":"home interior", "face":"human face", "hands":"human hand", "landscape":"landscape - scenery", "outfit":"landscape - scenery", "leisure":"leisure activity", "love":"love - emotion", "guy":"men", "motherhood":"mother", "parenthood":"parent", "positive":"positive emotion", "recreation":"recreational pursuit", "little":"small", "studio shoot":"studio shot", "together":"togetherness", "vertical shot":"vertical", "lady":"women", "young":"young adult", "light":"light - natural phenomenon", "trees":"tree"}
gender_dict = {"men":1,"man":1,"father":1,"boy":1, "boys":1, "none":2, "oldmen":3, "grandfather":3,"oldwomen":4, "grandmother":4, "nonbinary":5, "other":6, "trans":7, 
        "women":8,"woman":8, "mother":8, "girl":8, "girls":8, "youngmen":9, "youngwomen":10}
# gender2key = {"man":"men", "woman":"women"}
eth_dict = {"Black":1, "African-American":1, "AfricanAmerican":1, "African American":1, "African":1, "caucasian":2, "white people":2, "europeans":2, "eastasian":3, "chinese":3, "japanese":3, "asian":3, "hispaniclatino":4, "latino":4, "hispanic":4, "mexican":4, "middleeastern":5, "middle eastern":5, "arab":5, "mixedraceperson":6, "mixedrace":6, "mixed race":6, "mixed ethnicity":6, "multiethnic":6, "multi ethnic":6, "multi-ethnic":6, "nativeamericanfirstnations":7, "native american":7, "nativeamerican":7, "native-american":7, "indian american":7, "indianamerican":7, "indian-american":7, "first nations":7, "firstnations":7, "first-nations":7, "indigenous":7, "pacificislander":8, "pacific islander":8, "pacific-islander":8, "southasian":9, "south asian":9, "south-asian":9, "indian":9, "southeastasian":10, "southeast asian":10, "southeast-asian":10}
# load Keywords_202304300930.csv as df, drop all but keytype Locations, create two dicts: string->ID & GettyID->ID  
loc_dict = {"Canada":1989}
age_dict = {
    "baby":1,
    "infant":2,
    "child":3,
    "teen":4,
    "teenager":4,
    "young":5,
    "adult":6,
    "old":7
}

# table_search ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id"
SELECT = "DISTINCT(i.image_id), i.gender_id, author, caption, contentUrl, description, imagename"
FROM ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings e ON i.image_id = e.image_id "
WHERE = "e.image_id IS NULL"
LIMIT = 10


engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
metadata = MetaData(engine)


images_table = Table('ImagesTest', metadata,
    Column('image_id', Integer, primary_key=True, autoincrement=True),
    Column('site_name_id', Integer, ForeignKey('Site.site_name_id')),
    Column('site_image_id', String(50), nullable=False),
    Column('age_id', Integer, ForeignKey('Age.age_id')),
    Column('gender_id', Integer, ForeignKey('Gender.gender_id')),
    Column('location_id', Integer, ForeignKey('Location.location_id')),
    Column('author', String(100)),
    Column('caption', String(150)),
    Column('contentUrl', String(200), nullable=False),
    Column('description', String(150)),
    Column('imagename', String(100)),
    Column('uploadDate', Date)
)

imageskeywords_table = Table('ImagesKeywordsTest', metadata,
    Column('image_id', Integer, ForeignKey('Images.image_id')),
    Column('keyword_id', Integer, ForeignKey('Keywords.keyword_id')),
    PrimaryKeyConstraint('image_id', 'keyword_id'),
    UniqueConstraint('image_id', 'keyword_id', name='uq_image_keyword')
)

imagesethnicity_table = Table('ImagesEthnicityTest', metadata,
    Column('image_id', Integer, ForeignKey('Images.image_id')),
    Column('ethnicity_id', Integer, ForeignKey('Ethnicity.ethnicity_id')),
    PrimaryKeyConstraint('image_id', 'ethnicity_id'),
    UniqueConstraint('image_id', 'ethnicity_id', name='uq_image_keyword')
)

PEXELS_HEADERS = ["id", "title", "keywords", "country", "number_of_people", "orientation", "age","gender", "ethnicity", "mood", "image_url", "image_filename"]
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
        if isinstance(value, float) and np.isnan(value):
            this_dict[key] = None
    return this_dict

def make_key_dict(keys):
    keys_dict = {}
    for row in keys:
        keys_dict[row[2].lower()] = row[0]
    # do I need to pop header? or can I just leave it. 
    return keys_dict


def unlock_key(site_id,key):
    key_no = 0
    try:
        key_no = keys_dict[key]
        return(key_no)
    except:
        try:
            key = key2key[key]
            key_no = keys_dict[key]
            return(key_no)
        except:
            # try:
            # try with inflections
            # print(site_id)

            plur_key = getInflection(key, 'NNS')
            sing_key = getInflection(key, 'NN')
            gerund_key = getInflection(key, 'VBG')
            # print("inflected are: ", plur_key, sing_key, gerund_key)
            if plur_key and key != plur_key:
                try:
                    key_no = keys_dict[plur_key[0]]
                    # key = plur_key
                    # print(key_no)
                except:
                    pass# print(key)
            elif sing_key and key != sing_key:
                try:
                    key_no = keys_dict[sing_key[0]]
                    # key = plur_key
                    # print(key_no)
                except:
                    pass
                    # print(key)
            
            if gerund_key and key != gerund_key:
                try:
                    key_no = keys_dict[gerund_key[0]]
                    # key = plur_key
                    # print(key_no)
                except:
                    pass
                    # print(key)

            if key_no == 0:
                # if nothing worked, save key
                value_list = [site_id,key]
                write_csv(CSV_NOKEYS_PATH,value_list)
                # print(value_list)
                return
            else:
                value_list = [site_id,key_no]
                write_csv(CSV_IMAGEKEYS_PATH,value_list)
                return key_no

def insertignore(dataframe,table):

     # creating column list for insertion
     cols = "`,`".join([str(i) for i in dataframe.columns.tolist()])

     # Insert DataFrame recrds one by one.
     for i,row in dataframe.iterrows():
         sql = "INSERT IGNORE INTO `"+table+"` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
         engine.connect().execute(sql, tuple(row))


def insertignore_df(dataframe,table_name, engine):

     # Convert the DataFrame to a SQL table using pandas' to_sql method
     with engine.connect() as connection:
         dataframe.to_sql(name=table_name, con=connection, if_exists='append', index=False)

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

def search_keys(keys_list, this_dict, multi=False):
    results = []
    for key in keys_list:
        found = findall_dict(this_dict,key)
        if found is not None:
            results.append(found)
            print('found it in keywords:', found,"from key:", key)
            #age needs to be int()
            print(results)

    if len(set(results)) == 1:
        one_result = int(results[0])
        print("found a GOOD result: ", one_result)
    else:
        one_result = 0
        print("failed search: ", one_result)
    if multi:
        results_list = list(set(results))
    else:
        results_list = [one_result]
    return results_list


def get_eth(eth_name, keys_list):
    # eth_name = df['ethnicity'][ind]
    # print('isnan?')
    # print(np.isnan(eth_name))
    eth_no_list = []
    eth_no = None
    # if eth_name is not None or eth_name is not np.isnan(eth_name):
    if not np.isnan(eth_name):
        try:
            eth_no = eth_dict[eth_name]
        # need to key this into integer, like with keys
            print("eth_name ",eth_name)
        except:
            eth_no = None
            print("eth_dict failed with this key: ", eth_name)
        eth_no_list.append(eth_no)
    else:
        eth_no_list = search_keys(keys_list, eth_dict, True)
        print("searched keys and found eth_no: ", eth_no)
    return(eth_no_list)

def insertignore_dict(dict_data,table_name):

     # # creating column list for insertion
     # # cols = "`,`".join([str(i) for i in dataframe.columns.tolist()])
     # cols = "`,`".join([str(i) for i in list(dict.keys())])
     # tup = tuple(list(dict.values()))

     # sql = "INSERT IGNORE INTO `"+table+"` (`" +cols + "`) VALUES (" + "%s,"*(len(tup)-1) + "%s)"
     # engine.connect().execute(sql, tup)

     # Create a SQLAlchemy Table object representing the target table
     target_table = Table(table_name, metadata, extend_existing=True, autoload_with=engine)

     # Insert the dictionary data into the table using SQLAlchemy's insert method
     with engine.connect() as connection:
         connection.execute(target_table.insert(), dict_data)


def get_location(df, ind, keys_list):
    location = None
    key = df['country'][ind]
    if not np.isnan(key):
        try:
            location = loc_dict[key]
        except:
            print('NEW KEY, NOT IN COUNTRY -------------------------> ', key)
    else:
        print('NULL country: ', key)

    return(location)


def get_gender_age(df, ind, keys_list):
    global gender_dict


    # gender_pattern = r'\b(' + '|'.join(gender_dict.keys()) + r')\b'
    gender =0
    age=0
    key = df['gender'][ind]
    description = df['title'][ind]
    if key:
        #convertkeys
        try:
            gender = gender_dict[key]
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
            # gender_dict={"men":1, "none":2, "oldmen":3, "oldwomen":4, "nonbinary":5, "other":6, "trans":7, "women":8, "youngmen":9, "youngwomen":10}
        except:
            try:
                age = age_dict[key]
            except:
                print('NEW KEY, NOT AN AGE OR GENDER -------------------------> ', key)
    else:
        print('NULL gender: ', key)

    #try to find gender or age in description
    if gender == 0:
        # print("looking for gender in description")
        try:
            gender = findall_dict(gender_dict,description)
        except:
            print("no gender, going keyword hunting")
            try:
                gender = search_keys(keys_list, gender_dict)[0]
            except:
                print('no gender found: ', description)

    if age == 0:
        print("looking for age in description")
        try:
            age = findall_dict(age_dict,description)
        except:
            try:
                age = search_keys(keys_list, age_dict)[0]
            except:
                print('no age found: ', description)

    print("gender, age: ")
    print(gender)
    print (age)
    return gender, age
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

def structure_row_pexels(df, ind, keys_list): 
    gender_key, age_key = get_gender_age(df, ind, keys_list)
    location_no = get_location(df, ind, keys_list)
    image_row = {
        "site_image_id": df.loc[ind,'id'],
        "site_name_id": 5,
        "description": df['title'][ind],
        "location_id": location_no,
        # "": df['number_of_people'][ind] # should always be one. If not one, toss it? 
        # "": df['orientation'][ind]
        "age_id": age_key,
        "gender_id": gender_key,  
        # "location_id":"0",
        # "": df['mood'][ind]
        "contentUrl": df['image_url'][ind],
        "imagename": df['image_filename'][ind] # need to refactor this from the contentURL using the hash function
    }
    return(nan2none(image_row))


# make keyword dict
# gerund_key = getInflection(key, 'VBG')

# inftest = getInflection('watch', tag='VBG')
# print(inftest[0])
# quit()

def ingest_it():
    # print(keys_dict["cute"])

    df = pd.read_csv(CSV_IN_PATH)
    df = df.drop_duplicates()
    print(df)

    ind = 0
    # print(len(df.index))

    while (ind < len(df.index)):
        # make keywords list 
        keys_list = df['keywords'][ind].lower().split("|")

        # PEXELS_HEADERS = ["id", "title", "keywords", "country", "number_of_people", "orientation", "age","gender", "ethnicity", "mood", "image_url", "image_filename"]
        image_row = structure_row_pexels(df, ind, keys_list)
        # print(image_row)

        # turn keywords into keyword_id
        key_nos_list =[]
        for key in keys_list:
            key_no = unlock_key(image_row['site_image_id'],key)
            if key_no:
                key_nos_list.append(key_no)
        print(key_nos_list)
        ind += 1

        # ethnicity
        eth_no_list = get_eth(df['ethnicity'][ind], keys_list)

        # reparse the filename. should be able to bring over this code from the scraper_download.py

        with engine.connect() as conn:
            # Check if a row with the same data already exists
            select_stmt = select([images_table]).where(
                (images_table.c.site_name_id == image_row['site_name_id']) &
                (images_table.c.site_image_id == image_row['site_image_id'])
            )

            row = conn.execute(select_stmt).fetchone()

            if row is None:
                # Row does not exist, insert it
                insert_stmt = insert(images_table).values(image_row).prefix_with('IGNORE')
                result = conn.execute(insert_stmt)
                last_inserted_id = result.lastrowid

                rows = []
                for keyword_id in key_nos_list:
                    rows.append({'image_id': last_inserted_id, 'keyword_id': keyword_id})

                with engine.connect() as conn:
                    imageskeywords_insert_stmt = insert(imageskeywords_table).values(rows)
                    imageskeywords_insert_stmt = imageskeywords_insert_stmt.on_duplicate_key_update(keyword_id=imageskeywords_insert_stmt.inserted.keyword_id)
                    conn.execute(imageskeywords_insert_stmt)

    # imagesethnicity_table
            else:
                # Row already exists, do not insert
                print('Row already exists')



            # # retrieve the image_id of the inserted row
            # last_inserted_id = result.inserted_primary_key[0]
            # print(last_inserted_id)


    # insertignore_df(df, "images", engine)

    # print("inserted")

if __name__ == '__main__':
    print(sig)
    try:
        init_csv(CSV_NOKEYS_PATH,IMG_KEYWORD_HEADERS)
        init_csv(CSV_IMAGEKEYS_PATH,IMG_KEYWORD_HEADERS)
        keys = read_csv(KEYWORD_PATH)
        keys_dict = make_key_dict(keys)
        print("this many keys", len(keys_dict))

        ingest_it()
    except KeyboardInterrupt as _:
        print('[-] User cancelled.\n', flush=True)
    except Exception as e:
        print('[__main__] %s' % str(e), flush=True)





