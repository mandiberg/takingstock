import cv2
import pandas as pd
import os
import time
import sys
import pickle
import hashlib
import base64
import json
import ast

#linear sort imports non-class
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import matplotlib.pyplot as plt
import imutils
from imutils import face_utils
import shutil
from sys import platform
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
# my ORM
from my_declarative_base import Base, Clusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, update, Float, ForeignKey
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool


#mine
from mp_pose_est import SelectPose
from mp_sort_pose import SortPose
from mp_db_io import DataIO


VIDEO = False
CYCLECOUNT = 1
# ROOT="/Users/michaelmandiberg/Documents/projects-active/facemap_production/"

# keep this live, even if not SSD
# SegmentTable_name = 'May25segment123side_to_side'
# SegmentTable_name = 'July15segment123straight'
SegmentTable_name = 'SegmentAug30Straightahead'  #actually straight ahead smile

# SATYAM, this is MM specific
# for when I'm using files on my SSD vs RAID
IS_SSD = True
#IS_MOVE is in move_toSSD_files.py

# This is for when you only have the segment table. RW SQL query
IS_SEGONLY= True

# this is for controlling if it is using
# all clusters,
IS_CLUSTER = False
# number of clusters to analyze -- this is also declared in Clustering_SQL. Move to IO?
N_CLUSTERS = 44
# this is for IS_ONE_CLUSTER to only run on a specific CLUSTER_NO
IS_ONE_CLUSTER = True
CLUSTER_NO = 117

# this controls whether it is using the linear or angle process
IS_ANGLE_SORT = False

# this control whether sorting by topics
IS_TOPICS = False
N_TOPICS = 75

# I/O utils
io = DataIO(IS_SSD)
db = io.db
# overriding DB for testing
# io.db["name"] = "ministock"

NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES

# if IS_SSD:
#     io.ROOT = io.ROOT_PROD 
# else:
#     io.ROOT = io.ROOT36


if IS_SEGONLY is not True:
    print("production run. IS_SSD is", IS_SSD)

    # # # # # # # # # # # #
    # # for production  # #
    # # # # # # # # # # # #

    # SAVE_SEGMENT controls whether the result will be saved to the db as a new table
    SAVE_SEGMENT = False
    SELECT = "DISTINCT(i.image_id), i.site_name_id, i.contentUrl, i.imagename, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings68, i.site_image_id"

    # don't need keywords if SegmentTable_name
    # this is for MM segment table
    # FROM =f"Images i LEFT JOIN Encodings e ON i.image_id = e.image_id INNER JOIN {SegmentTable_name} seg ON i.site_image_id = seg.site_image_id"
    # WHERE = "e.is_face IS TRUE AND e.face_encodings IS NOT NULL AND e.bbox IS NOT NULL AND i.site_name_id = 8 AND i.age_id NOT IN (1,2,3,4)"

    # this is for gettytest3 table
    FROM ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings e ON i.image_id = e.image_id JOIN ImagesClusters ic ON i.image_id = ic.image_id"
    WHERE = "e.is_face IS TRUE AND e.bbox IS NOT NULL AND i.site_name_id = 1 AND k.keyword_text LIKE 'smil%'"
    LIMIT = 1000


elif IS_SEGONLY:

    SAVE_SEGMENT = False

    # no JOIN just Segment table

    if IS_CLUSTER:
        SELECT = "DISTINCT(i.image_id), i.site_name_id, i.contentUrl, i.imagename, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings68, i.site_image_id"
        FROM =f"Images i LEFT JOIN Encodings e ON i.image_id = e.image_id INNER JOIN {SegmentTable_name} seg ON i.site_image_id = seg.site_image_id JOIN ImagesClusters ic ON i.image_id = ic.image_id"
        # WHERE = "e.is_face IS TRUE AND e.face_encodings IS NOT NULL AND e.bbox IS NOT NULL AND i.site_name_id = 8 AND i.age_id NOT IN (1,2,3,4)"
        WHERE = "i.site_name_id != 1"
        # WHERE = "mouth_gap < 2 AND age_id NOT IN (1,2,3,4) AND image_id < 40647710"
        # WHERE = "mouth_gap < 2 AND age_id NOT IN (1,2,3,4) AND s.image_id < 40647710 AND k.keyword_text LIKE 'work%'"
        LIMIT = 1000


    if IS_TOPICS:
        SELECT = "DISTINCT(s.image_id), s.site_name_id, s.contentUrl, s.imagename, s.face_x, s.face_y, s.face_z, s.mouth_gap, s.face_landmarks, s.bbox, s.face_encodings68, s.site_image_id"
        FROM = f"{SegmentTable_name} s JOIN ImagesTopics it ON s.image_id = it.image_id"
        # FROM =f"Images i LEFT JOIN Encodings e ON s.image_id = s.image_id INNER JOIN {SegmentTable_name} seg ON s.site_image_id = seg.site_image_id JOIN ImagesClusters ic ON s.image_id = ic.image_id"
        # WHERE = "s.is_face IS TRUE AND s.face_encodings IS NOT NULL AND s.bbox IS NOT NULL AND s.site_name_id = 8 AND s.age_id NOT IN (1,2,3,4)"
        WHERE = "s.site_name_id != 1 AND age_id NOT IN (1,2,3,4) "
        # WHERE = "mouth_gap < 2 AND age_id NOT IN (1,2,3,4) AND image_id < 40647710"
        # WHERE = "mouth_gap < 2 AND age_id NOT IN (1,2,3,4) AND s.image_id < 40647710 AND k.keyword_text LIKE 'work%'"
        LIMIT = 10000


    else:
        SELECT = "*" 
        FROM = f"{SegmentTable_name} AS seg JOIN ImagesClusters AS ic ON seg.image_id = ic.image_id"
        # FROM = f"{SegmentTable_name} AS seg"
        # FROM = f"{SegmentTable_name} s JOIN ImagesKeywords ik ON s.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id"
        # WHERE = "seg.age_id NOT IN (1,2,3,4) and seg.mouth_gap > 15"
        WHERE = "seg.age_id NOT IN (1,2,3,4) and seg.site_name_id !=1"
        # WHERE = "age_id NOT IN (1,2,3,4) AND k.keyword_text LIKE 'happ%' "
        # WHERE = "mouth_gap < 2 AND age_id NOT IN (1,2,3,4) AND image_id < 40647710 AND gender_id = 1"
        LIMIT = 100



    '''
    this is the old way, with a JOIN
    # don't need keywords if SegmentTable_name
    # this is for MM segment table
    SELECT = "DISTINCT(i.image_id), i.site_name_id, i.contentUrl, i.imagename, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings68, i.site_image_id" 
    FROM =f"Images i LEFT JOIN Encodings e ON i.image_id = e.image_id INNER JOIN {SegmentTable_name} seg ON i.site_image_id = seg.site_image_id"
    if IS_CLUSTER is True or IS_ONE_CLUSTER is True:
        FROM += " JOIN ImagesClusters ic ON i.image_id = ic.image_id"
    # WHERE = "e.face_encodings68 IS NOT NULL AND i.site_name_id = 8 AND i.age_id NOT IN (1,2,3,4) AND e.mouth_gap > 10"
    # WHERE = "e.face_encodings68 IS NOT NULL"
    WHERE = "e.mouth_gap > 15 AND i.age_id NOT IN (1,2,3,4)"
    '''

    # this is for gettytest3 table
    # SELECT = "DISTINCT(image_id), site_name_id, contentUrl, imagename, face_x, face_y, face_z, mouth_gap, face_landmarks, bbox, face_encodings, site_image_id"
    # FROM = SegmentTable_name
    # FROM = f"{SegmentTable_name} st JOIN ImagesClusters ic ON st.image_id = ic.image_id JOIN Clusters c ON ic.cluster_no = c.cluster_no"
    # "Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings e ON i.image_id = e.image_id JOIN ImagesClusters ic ON i.image_id = ic.image_id"
    # WHERE = "bbox IS NOT NULL"
    # AND i.site_name_id = 1 AND k.keyword_text LIKE 'smil%'"


motion = {
    "side_to_side": False,
    "forward_smile": True,
    "laugh": False,
    "forward_nosmile":  False,
    "static_pose":  False,
    "simple": False,
}

EXPAND = False

# face_height_output is how large each face will be. default is 750
face_height_output = 500
# face_height_output = 256

# define ratios, in relationship to nose
# units are ratio of faceheight
# top, right, bottom, left
# image_edge_multiplier = [1, 1, 1, 1]
# image_edge_multiplier = [1.5,1.5,1.5,1.5]
# image_edge_multiplier = [1.5, 2, 1.5, 2]
image_edge_multiplier = [1.2, 1.2, 1.6, 1.2]


# construct my own objects
sort = SortPose(motion, face_height_output, image_edge_multiplier,EXPAND)

start_img_name = "median"
start_site_image_id = None
# start_img_name = "start_site_image_id"
# start_site_image_id = "3/3B/193146471-photo-portrait-of-funky-young-lady-fooling-show-fingers-claws-growl-tiger-wear-stylish-striped"
# start_site_image_id = "0/02/159079944-hopeful-happy-young-woman-looking-amazed-winning-prize-standing-white-background.jpg"
# start_site_image_id = "0/08/158083627-man-in-white-t-shirt-gesturing-with-his-hands-studio-cropped.jpg"

# no gap
# start_site_image_id = "5/58/95516714-happy-well-dressed-man-holding-a-gift-on-white-background.jpg"


# start_site_image_id = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/images_123rf/E/E8/95447708-portrait-of-happy-smiling-beautiful-young-woman-touching-skin-or-applying-cream-isolated-over-white.jpg"
# 274243    Portrait of funny afro guy  76865   {"top": 380, "left": 749, "right": 1204, "bottom": 835}
d = None


# override io.db for testing mode
# db['name'] = "123test"

if db['unix_socket']:
    # for MM's MAMP config
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# to create new SegmentTable with variable as name
class SegmentTable(Base):
    __tablename__ = SegmentTable_name

    image_id = Column(Integer, primary_key=True)
    site_name_id = Column(Integer)
    contentUrl = Column(String(300), nullable=False)
    imagename = Column(String(200))
    face_x = Column(DECIMAL(6, 3))
    face_y = Column(DECIMAL(6, 3))
    face_z = Column(DECIMAL(6, 3))
    mouth_gap = Column(DECIMAL(6, 3))
    face_landmarks = Column(BLOB)
    bbox = Column(JSON)
    face_encodings = Column(BLOB)
    face_encodings68 = Column(BLOB)
    site_image_id = Column(String(50), nullable=False)

# construct mediapipe objects
mp_drawing = mp.solutions.drawing_utils

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,min_detection_confidence=0.5)


###################
# SQL  FUNCTIONS  #
###################

def selectSQL(cluster_no=None):
    print(f"cluster_no is")
    print(cluster_no)
    if cluster_no is not None:
        if IS_CLUSTER or IS_ONE_CLUSTER:
            cluster =f"AND ic.cluster_id = {str(cluster_no)}"
        elif IS_TOPICS:
            cluster =f"AND it.topic_id = {str(cluster_no)}"
    else:
        cluster=""
    print(f"cluster SELECT is {cluster}")
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} {cluster} LIMIT {str(LIMIT)};"
    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    return(resultsjson)


def select_cluster_median(cluster_no):
    cluster_selectsql = f"SELECT c.cluster_median FROM Clusters c WHERE cluster_id={cluster_no};"
    result = engine.connect().execute(text(cluster_selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    cluster_median = (resultsjson[0]['cluster_median'])
    return(cluster_median)




def save_segment_DB(df_segment):
    #save the df to a table
    # Assuming you have your DataFrame named 'df' containing the query results
    for _, row in df_segment.iterrows():
        instance = SegmentTable(
            image_id=row['image_id'],
            site_name_id=row['site_name_id'],
            contentUrl=row['contentUrl'],
            imagename=row['imagename'],
            face_x=row['face_x'],
            face_y=row['face_y'],
            face_z=row['face_z'],
            mouth_gap=row['mouth_gap'],
            face_landmarks=pickle.dumps(row['face_landmarks'], protocol=3),
            bbox=row['bbox'],
            face_encodings=pickle.dumps(row['face_encodings'], protocol=3),
            site_image_id=row['site_image_id']
        )
        session.add(instance)
    session.commit()



###################
# SORT FUNCTIONS  #
###################


# need to pass through start_img_enc rather than start_img_name
# for linear it is in the df_enc, but for itter, the start_img_name is in prev df_enc
# takes a dataframe of images and encodings and returns a df sorted by distance
def sort_by_face_dist(df_enc, df_128_enc):
    

    this_start = sort.counter_dict["start_img_name"]
    face_distances=[]

    # this prob should be a df.iterrows
    print("df_enc.index")
    print(df_enc.index)
    print(len(df_enc.index))
    print(sort.counter_dict)
    FIRST_ROUND = True
    if sort.CUTOFF < len(df_enc.index):
        itters = sort.CUTOFF
    else: 
        itters = len(df_enc.index)
    for i in range(itters):
        # find the image
        print(df_enc)
        # this is the site_name_id for this_start, needed to test mse
        print("this_start", this_start)
        print("starting sort round ",str(i))
        
        ## Get the starting encodings 
        if this_start != "median" and this_start != "start_site_image_id" and i == 0:
            # this is the first round. set encodings to the passed through encodings
            # IF NO START IMAGE SPECIFIED (this line works for no clusters)
            print("attempting set enc1 from pass through")
            enc1 = sort.counter_dict["last_image_enc"]
            # enc1 = df_enc.loc[this_start]['face_encodings']
            print(enc1)
            print("set enc1 from pass through")
        else:
            #this is the first??? round, set via df
            print("trying get_start_enc()")
            enc1, df_128_enc = sort.get_start_enc(this_start, df_128_enc)
            # # test to see if get_start_enc was successful
            # # if not, retain previous enc1. or shoudl it reassign median? 
            # if enc1_temp is not None:
            #     enc1 = enc1_temp
            print("set enc1 from get_start_enc()")

        ## Find closest
        try:
            # closest_dict is now a dict with 1 or more items
            # this_start is a filepath, which serves as df index
            # it is now a dict of key=distance value=filepath
            print("going to get closest")

            # TK
            # need to send the df_enc with the same two keys through to get_closest
            # dist, closest_dict, df_128_enc = sort.get_closest_df(FIRST_ROUND, enc1,df_enc, df_128_enc, sorttype="128d")
            dist, closest_dict, df_128_enc = sort.get_closest_df(FIRST_ROUND, enc1,df_enc, df_128_enc, sorttype="planar")
            # dist, closest_dict, df_128_enc = sort.get_closest_df(enc1,df_enc, df_128_enc)
            FIRST_ROUND = False


            print("got closest")
            print(closest_dict)

            # Break out of the loop if greater than MAXDIST
            # I think this will be graceful with cluster iteration
            print("dist")
            print(dist)
            print("sort.MAXDIST")
            print(sort.MAXDIST)
            if dist > sort.MAXDIST:
                print("should breakout")
                break

        except Exception as e:
            print(str(e))


     
        # Iterate through the results and append
        dkeys = list(closest_dict.keys())
        dkeys.sort()
        images_to_drop =[]
        for dkey in dkeys:


            ## Collect values and append to face_distances
            this_start = closest_dict[dkey]
            print("this_start assigned as ", this_start)
            face_landmarks=None
            bbox=None

            print("THIS: closest_dict[dkey],")
            print(closest_dict[dkey])

            try:
                print("dkey, df_enc.loc[closest_dict[dkey]]")
                print(dkey)
                print(closest_dict[dkey])
                print(df_enc.loc[closest_dict[dkey]])
                site_name_id = df_enc.loc[closest_dict[dkey]]['site_name_id']
                face_landmarks = df_enc.loc[closest_dict[dkey]]['face_landmarks']
                bbox = df_enc.loc[closest_dict[dkey]]['bbox']
                print("assigned bbox", bbox)
            except:
                print("won't assign landmarks/bbox")
            print("site_name_id is the following")

            # for some reason, site_name_id is not an int. trying to test if int.
            # print(type(site_name_id))
            # if not pd.is_int(site_name_id): continue
            print(site_name_id)
            print("site_specific_root_folder", io.folder_list[site_name_id])
            site_specific_root_folder = io.folder_list[site_name_id]
            print("site_specific_root_folder")
            print(site_specific_root_folder)
            # save the image -- this prob will be to append to list, and return list? 
            # save_sorted(i, folder, start_img_name, dist)
            this_dist=[dkey, site_specific_root_folder, this_start, site_name_id, face_landmarks, bbox]
            face_distances.append(this_dist)
            images_to_drop.append(this_start)

        # remove the last image this_start, then drop them from df_128_enc
        # the this_start will be dropped in the get_start_enc method
        print("lenght of images to drop before and after removing this_start")
        print(len(images_to_drop))
        images_to_drop.remove(this_start)
        print(len(images_to_drop))
        for dropimage in images_to_drop:
            print("going to remove this image enc", dropimage)
            df_128_enc=df_128_enc.drop(dropimage)

        #debuggin
        print(f"sorted round {str(i)} which is actually round  {str(i+len(dkeys)-1)}")
        print(len(df_128_enc.index))
        print(dist)
        print (start_img_name)

    ## When loop is complete, create df
    df = pd.DataFrame(face_distances, columns =['dist', 'folder', 'filename','site_name_id','face_landmarks', 'bbox'])
    print(df)

    ## Set a start_img_name for next round --> for clusters
    try:
        last_file = face_distances[-1][2]
        print("last_file ",last_file)
    except:
        last_file = this_start
        print("last_file is this_start",last_file)
    sort.counter_dict["start_img_name"] = last_file

    # df = df.sort_values(by=['dist']) # this was sorting based on delta distance, not sequential distance
    # print(df)
    return df



def cycling_order(CYCLECOUNT, sort):
    img_array = []
    cycle = 0 
    # metamedian = get_metamedian(angle_list)
    metamedian = sort.metamedian
    d = sort.d

    print("CYCLE to test: ",cycle)

    while cycle < CYCLECOUNT:
        print("CYCLE: ",cycle)
        for angle in sort.angle_list:
            print("angle: ",str(angle))
            # # print(d[angle].iloc[(d[angle][SECOND_SORT]-metamedian).abs().argsort()[:2]])
            # # print(d[angle].size)
            try:
                # I don't remember exactly how this segments the data...!!!
                # [:CYCLECOUNT] gets the first [:0] value on first cycle?
                # or does it limit the total number of values to the number of cycles?
                print(d[angle])
                
                #this is a way of finding the image with closest second sort (Y)
                #mystery value is the image to be matched? 
                print("second sort, metamedian ",d[angle][sort.SECOND_SORT],sort.metamedian)
                mysteryvalue = (d[angle][sort.SECOND_SORT]-sort.metamedian)
                print('mysteryvalue ',mysteryvalue)
                #is mystery value a df?
                #this is finding the 
                mysterykey = mysteryvalue.abs().argsort()[:CYCLECOUNT]
                print('mysterykey: ',mysterykey)
                closest = d[angle].iloc[mysterykey]
                closest_file = closest.iloc[cycle]['imagename']
                closest_mouth = closest.iloc[cycle]['mouth_gap']
                print('closest: ')
                print(closest_file)
                img = cv2.imread(closest_file)
                height, width, layers = img.shape
                size = (width, height)
                img_array.append(img)
            except:
                print('failed cycle angle:')
                # print('failed:',row['imagename'])
        print('finished a cycle')
        sort.angle_list.reverse()
        cycle = cycle +1
        # print(angle_list)
    return img_array, size


def prep_encodings(df_segment):
    # format the encodings for sorting by distance
    # df_enc will be the df with bbox, site_name_id, etc, keyed to filename
    # df_128_enc will be 128 colums of encodings, keyed to filename
    # print("prep_encodings df_segment", df_segment)
    col1="imagename"
    col2="face_encodings68"
    col3="site_name_id"
    col4="face_landmarks"
    col5="bbox"
    df_enc=pd.DataFrame(columns=[col1, col2, col3, col4, col5])
    df_enc = pd.DataFrame({col1: df_segment['imagename'], col2: df_segment['face_encodings68'].apply(lambda x: np.array(x)), 
                col3: df_segment['site_name_id'], col4: df_segment['face_landmarks'], col5: df_segment['bbox'] })
    df_enc.set_index(col1, inplace=True)

    # Create column names for the 128 encoding columns
    encoding_cols = [f"encoding{i}" for i in range(128)]

    # Create a new DataFrame with the expanded encoding columns
    df_expanded = df_enc.apply(lambda row: pd.Series(row[col2], index=encoding_cols), axis=1)

    # Concatenate the expanded DataFrame with the original DataFrame
    df_final = pd.concat([df_enc, df_expanded], axis=1)

    # make a separate df that just has the encodings
    df_128_enc = df_final.drop([col2, col3, col4, col5], axis=1)

    print(df_128_enc)
    return df_enc, df_128_enc

def compare_images(last_image, img, face_landmarks, bbox):
    is_face = None

    #crop image here:
    if sort.EXPAND:
        cropped_image = sort.expand_image(img, face_landmarks, bbox)
    else:
        cropped_image = sort.crop_image(img, face_landmarks, bbox)
    print("cropped_image type: ",type(cropped_image))

    # this code takes image i, and blends it with the subsequent image
    # next step is to test to see if mp can recognize a face in the image
    # if no face, a bad blend, try again with i+2, etc. 
    if cropped_image is not None:
        print("have a cropped image trying to save", cropped_image.shape)
        try:
            print("last_image is ", type(last_image))
        except:
            print("couldn't test last_image")
        try:
            if not sort.counter_dict["first_run"]:
                print("testing is_face")
                is_face = sort.test_pair(last_image, cropped_image)
                if is_face:
                    # print("same person, testing mse")
                    is_face = sort.unique_face(last_image,cropped_image)
                    # print ("mse ",mse)
                else:
                    print("failed is_face test")
            else:
                print("first round, skipping the pair test")
        except:
            print("last_image try failed")
        # if is_face or first_run and sort.resize_factor < sort.resize_max:
        if is_face or sort.counter_dict["first_run"]:
            sort.counter_dict["first_run"] = False
            last_image = cropped_image
            sort.counter_dict["good_count"] += 1
        else: 
            print("pair do not make a face, skipping")
            sort.counter_dict["isnot_face_count"] += 1
    elif cropped_image is None and sort.counter_dict["first_run"]:
        print("first run, but bad first image")
        last_image = cropped_image
        sort.counter_dict["cropfail_count"] += 1

    else:
        print("no image here, trying next")
        sort.counter_dict["cropfail_count"] += 1
    return cropped_image


def print_counters():
    print("good_count")
    print(sort.counter_dict["good_count"])
    print("isnot_face_count")
    print(sort.counter_dict["isnot_face_count"])
    print("cropfail_count")
    print(sort.counter_dict["cropfail_count"])
    print("sort.negmargin_count")
    print(sort.negmargin_count)
    print("sort.toosmall_count")
    print(sort.toosmall_count)
    print("failed_dist_count")
    print(sort.counter_dict["failed_dist_count"])
    print("total count")
    print(sort.counter_dict["counter"])


def const_imgfilename(filename, df, imgfileprefix):
    print("filename", filename)
    UID = filename.split('-id')[-1].split("/")[-1].replace(".jpg","")
    print("UID ",UID)
    counter_str = str(sort.counter_dict["counter"]).zfill(len(str(df.size)))  # Add leading zeros to the counter
    imgfilename = imgfileprefix+"_"+str(counter_str)+"_"+UID+".jpg"
    print("imgfilename ",imgfilename)
    return imgfilename

def linear_test_df(df,cluster_no, itter=None):
    #itter is a cap, to stop the process after a certain number of rounds
    print('writing images')
    imgfileprefix = f"X{str(sort.XLOW)}-{str(sort.XHIGH)}_Y{str(sort.YLOW)}-{str(sort.YHIGH)}_Z{str(sort.ZLOW)}-{str(sort.ZHIGH)}_ct{str(df.size)}"
    print(imgfileprefix)
    good = 0
    img_list = []
    try:
        for index, row in df.iterrows():
            print('in loop, index is', str(index))
            print(row)
            imgfilename = const_imgfilename(row['filename'], df, imgfileprefix)
            outpath = os.path.join(sort.counter_dict["outfolder"],imgfilename)
            open_path = os.path.join(io.ROOT,row['folder'],row['filename'])
            # print(outpath, open_path)
            try:
                img = cv2.imread(open_path)
            except:
                print("couldn't read image")
                continue
            if row['dist'] < sort.MAXDIST:
                # compare_images to make sure they are face and not the same
                # last_image is cv2 np.array
                cropped_image = compare_images(sort.counter_dict["last_image"], img, row['face_landmarks'], row['bbox'])
                if cropped_image is not None:
                    img_list.append((outpath, cropped_image))
                    # this is done in compare function
                    # sort.counter_dict["good_count"] += 1
                    good += 1
                    # print("row['filename']")
                    # print(row['filename'])
                    sort.counter_dict["start_img_name"] = row['filename']
                    # print(sort.counter_dict["last_image"])
                    print("saved: ",outpath)
                    sort.counter_dict["counter"] += 1
                    if itter and good > itter:
                        print("breaking after this many itters,", str(good), str(itter))
                        continue
                    sort.counter_dict["last_image"] = img_list[-1][1]  #last pair in list, second item in pair
                else:
                    print("cropped_image is None")
            else:
                sort.counter_dict["failed_dist_count"] += 1
                print("MAXDIST too big:" , str(sort.MAXDIST))
        # print("sort.counter_dict with last_image???")
        # print(sort.counter_dict)

    except Exception as e:
        print(str(e))

    return img_list
    
def write_images(img_list):
    for path_img in img_list:
        cv2.imwrite(path_img[0],path_img[1])


def process_iterr_angles(start_img_name, df_segment, cluster_no, sort):
    #cycling patch
    img_list = []
    cycle = 0 
    metamedian = sort.metamedian
    d = sort.d

    print("CYCLE to test: ",cycle, start_img_name)
    while cycle < CYCLECOUNT:
        print("CYCLE: ",cycle)
        for angle in sort.angle_list:
            print("angle: ",str(angle))
            # print(d[angle].iloc[(d[angle][sort.SECOND_SORT]-metamedian).abs().argsort()[:2]])
            if(d[angle].size) != 0:
                try:
                    print("sort.counter_dict[start_img_name] before sort_by_face_dist")
                    print(sort.counter_dict["start_img_name"] )
                    if sort.counter_dict["start_img_name"] != "median":
                        try:
                            last_row = df_segment.loc[sort.counter_dict["start_img_name"]]
                            print("last_row")
                            print(last_row)
                        except Exception as e:
                            print(str(e))
                    df_enc, df_128_enc = prep_encodings(d[angle])
                    # # get dataframe sorted by distance
                    df_sorted = sort_by_face_dist(df_enc, df_128_enc)
                    # print("df_sorted")
                    # print(df_sorted)
                    # print("sort.counter_dict before linear_test_df")
                    # print(sort.counter_dict)
                    if sort.counter_dict["last_image"] is None:
                        try:
                            sort.counter_dict["last_image"] = cv2.imread(sort.counter_dict["start_img_name"])
                        except:
                            print("failed to open sort.counter_dict[start_img_name]")
                    else:
                        print("sort.counter_dict has a last_image")
                    # write_images(df_sorted, cluster_no)
                    # print("df_sorted before linear_test_df")

                    # print(type(df_sorted.size))
                    # print(df_sorted.size)
                    # print(df_sorted)
                    # print("sort.counter_dict after linear_test_df")
                    # print(sort.counter_dict)
                    # print("img_list")
                    # print(img_list[0])
                    # print(len(img_list))
                    # # only write the first, closest one
                    # # in the future, prob want to assign each image list to
                    # # a list/df keyed by angle, so can iterate through it? 
                    if angle > 15 and motion['forward_smile'] == True:
                        img_list = linear_test_df(df_sorted,cluster_no)
                        write_images(img_list)
                    else:
                        print("sending in an itter cap")
                        img_list = linear_test_df(df_sorted,cluster_no, 1)
                        cv2.imwrite(img_list[0][0],img_list[0][1])
                    


                except:
                    print('failed cycle angle:')
                    # print('failed:',row['imagename'])
            else:
                print("skipping empty angle")
        print('finished a cycle')
        sort.angle_list.reverse()
        cycle = cycle +1
        # # print(angle_list)

    print_counters()


def process_linear(start_img_name, df_segment, cluster_no, sort):
    # linear sort by encoding distance
    print("processing linear")
    # preps the encodings for sort
    # sort.set_counters(io.ROOT,cluster_no, start_img_name)  

    df_enc, df_128_enc = prep_encodings(df_segment)

    # # get dataframe sorted by distance
    df_sorted = sort_by_face_dist(df_enc, df_128_enc)

    # test to see if they make good faces
    img_list = linear_test_df(df_sorted,cluster_no)
    write_images(img_list)
    print_counters()


###################
#  MY MAIN CODE   #
###################

def main():
    # these are used in cleaning up fresh df from SQL
    def unpickle_array(pickled_array):
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
    def unstring_json(json_string):
        eval_string = ast.literal_eval(json_string)
        if isinstance(eval_string, dict):
            return eval_string
        else:
            json_dict = json.loads(eval_string)
            return json_dict
    def make_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    def decode_64_array(encoded):
        decoded = base64.b64decode(encoded).decode('utf-8')
        return decoded
    def newname(contentUrl):
        file_name_path = contentUrl.split('?')[0]
        file_name = file_name_path.split('/')[-1]
        extension = file_name.split('.')[-1]
        if file_name.endswith(".jpeg"):
            file_name = file_name.replace(".jpeg",".jpg")
        elif file_name.endswith(".png") or file_name.endswith(".webm"):
            pass
        elif not file_name.endswith(".jpg"):
            file_name += ".jpg"    
        hash_folder1, hash_folder2 = io.get_hash_folders(file_name)
        newname = os.path.join(hash_folder1, hash_folder2, file_name)
        return newname
        # file_name = file_name_path.split('/')[-1]
    print("in main, making SQL query")


    ###################
    #  MAP THE IMGS   #
    ###################

    # this is the key function, which is called for each cluster
    # or only once if no clusters
    def map_images(resultsjson, cluster_no=None):
        # print(df_sql)

        # read the csv and construct dataframe
        try:
            df = pd.json_normalize(resultsjson)
            print(df)
        except:
            print('you forgot to change the filename DUH')
        if not df.empty:

            # Apply the unpickling function to the 'face_encodings' column
            # TK this is where I will change face_encodings to the variations
            df['face_encodings68'] = df['face_encodings68'].apply(unpickle_array)
            df['face_landmarks'] = df['face_landmarks'].apply(unpickle_array)
            df['bbox'] = df['bbox'].apply(lambda x: unstring_json(x))

            # this may be a big problem
            # turn URL into local hashpath (still needs local root folder)
            # df['imagename'] = df['contentUrl'].apply(newname)
            # make decimals into float
            columns_to_convert = ['face_x', 'face_y', 'face_z', 'mouth_gap']
            df[columns_to_convert] = df[columns_to_convert].applymap(make_float)

            ### SEGMENT THE DATA ###

            # make the segment based on settings
            print("going to segment")
            df_segment = sort.make_segment(df)
            print("made segment")


            # this is to save files from a segment to the SSD
            print("will I save segment? ", SAVE_SEGMENT)
            if SAVE_SEGMENT:
                Base.metadata.create_all(engine)
                print(df_segment.size)
                save_segment_DB(df_segment)
                print("saved segment to ", SegmentTable_name)
                quit()

            ### Set counter_dict ###

            sort.set_counters(io.ROOT,cluster_no, start_img_name,start_site_image_id)

            print("set sort.counter_dict:" )
            print(sort.counter_dict)


            ### Get cluster_median encodings for cluster_no ###

            if cluster_no is not None and cluster_no !=0:
                # skips cluster 0 for pulling median because it was returning NULL
                # cluster_median = select_cluster_median(cluster_no)
                # image_id = insert_dict['image_id']
                # can I filter this by site_id? would that make it faster or slower? 

                results = session.query(Clusters).filter(Clusters.cluster_id==cluster_no).first()


                # results = session.query(Clusters).filter(Clusters.cluster_id==cluster_no).first()
                print(results)
                cluster_median = unpickle_array(results.cluster_median)
                # start_img_name = cluster_median
                sort.counter_dict["last_image_enc"]=cluster_median


            ### SORT THE LIST OF SELECTED IMAGES ###
            ###    THESE ARE THE VARIATIONS      ###

            if motion["side_to_side"] is True and IS_ANGLE_SORT is False:
                # this is old, hasn't been refactored.
                img_list, size = cycling_order(CYCLECOUNT, sort)
                # size = sort.get_cv2size(ROOT, img_list[0])
            elif IS_ANGLE_SORT is True:
                # get list of all angles in segment
                angle_list = sort.createList(df_segment)

                # sort segment by angle list
                # creates sort.d attribute: a dataframe organized (indexed?) by angle list
                sort.get_divisor(df_segment)

                # # is this used anywhere? 
                # angle_list_pop = angle_list.pop()

                # get median for first sort
                median = sort.get_median()

                # get metamedian for second sort, creates sort.metamedian attribute
                sort.get_metamedian()
                # print(df_segment)

                process_iterr_angles(start_img_name,df_segment, cluster_no, sort)
            else:
                # hard coding override to just start from median
                # sort.counter_dict["start_img_name"] = "median"

                process_linear(start_img_name,df_segment, cluster_no, sort)
        elif df.empty and IS_CLUSTER:
            print('dataframe empty, but IS_CLUSTER so continuing to next cluster_no')

        else: 
            print('dataframe empty, and not IS_CLUSTER so probably bad path or bad SQL')
            sys.exit()



    ###          THE MAIN PART OF MAIN()           ###
    ### QUERY SQL BASED ON CLUSTERS AND MAP_IMAGES ###

    #creating my objects
    start = time.time()

    # to loop or not to loop that is the cluster
    if IS_CLUSTER:
        print(f"IS_CLUSTER is {IS_CLUSTER} with {N_CLUSTERS}")
        for cluster_no in range(N_CLUSTERS):
            print(f"SELECTing cluster {cluster_no} of {N_CLUSTERS}")
            resultsjson = selectSQL(cluster_no)
            print(f"resultsjson contains {len(resultsjson)} images")
            map_images(resultsjson, cluster_no)
    elif IS_TOPICS:
        print(f"IS_TOPICS is {IS_TOPICS} with {N_TOPICS}")
        for cluster_no in range(N_TOPICS):
            print(f"SELECTing cluster {cluster_no} of {N_TOPICS}")
            resultsjson = selectSQL(cluster_no)
            print(f"resultsjson contains {len(resultsjson)} images")
            map_images(resultsjson, cluster_no)
    elif IS_ONE_CLUSTER:
        print(f"SELECTing cluster {CLUSTER_NO}")
        resultsjson = selectSQL(CLUSTER_NO)
        print(f"resultsjson contains {len(resultsjson)} images")
        map_images(resultsjson, CLUSTER_NO)
    else:
        print("doing regular linear")
        resultsjson = selectSQL() 
        map_images(resultsjson)

    print("got results, count is: ",len(resultsjson))


if __name__ == '__main__':
    main()

