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
from my_declarative_base import Base, SegmentTable, Clusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Images

from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, update, Float
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
SegmentTable_name = 'SegmentOct20'

# SATYAM, this is MM specific
# for when I'm using files on my SSD vs RAID
IS_SSD = True
#IS_MOVE is in move_toSSD_files.py

# This is for when you only have the segment table. RW SQL query
IS_SEGONLY= True

HSV_CONTROL = True # defining so it doesn't break below, if commented out
# This tells it to pull luminosity. Comment out if not using
if HSV_CONTROL:
    HSV_BOUNDS = {
        "LUM_MIN": 0,
        "LUM_MAX": 40,
        "SAT_MIN": 0,
        "SAT_MAX": 1000,
        "HUE_MIN": 0,
        "HUE_MAX": 360
    }
else:
    HSV_BOUNDS = {
        "LUM_MIN": 0,
        "LUM_MAX": 100,
        "SAT_MIN": 0,
        "SAT_MAX": 1000,
        "HUE_MIN": 0,
        "HUE_MAX": 360
    }
HSV_WEIGHTS = {
    # converts everything to a 0-1 scale
    "LUM": .01,
    "SAT": 1,
    "HUE": 0.002777777778
}

# this is for controlling if it is using
# all clusters,
IS_CLUSTER = False
# number of clusters to analyze -- this is also declared in Clustering_SQL. Move to IO?
N_CLUSTERS = 128
# this is for IS_ONE_CLUSTER to only run on a specific CLUSTER_NO
IS_ONE_CLUSTER = False
CLUSTER_NO = 63

# cut the kids
NO_KIDS = True
VERBOSE = True
# this controls whether it is using the linear or angle process
IS_ANGLE_SORT = False

# this control whether sorting by topics
IS_TOPICS = False
N_TOPICS = 30

IS_ONE_TOPIC = True
TOPIC_NO = [7]
# 7 is isolated, 84 is business, 27 babies, 16 pointing
# 37 is doctor << 35 covid
# 45 is hands
# 34 holding tablet
# 63 feeling frustrated
# 85 is hands to face
# 71 shout
# 10 is surprise
# 25 is yoga << planar, 85 planar, 77 fingers crossed

SORT_TYPE = "128d"
# SORT_TYPE ="planar"
# SORT_TYPE = "planar_body"

ONE_SHOT = False # take all files, based off the very first sort order.
JUMP_SHOT = True # jump to random file if can't find a run

# I/O utils
io = DataIO(IS_SSD)
db = io.db
# overriding DB for testing
io.db["name"] = "stock"
METAS_FILE = "metas.csv"

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

    # don't need keywords if SegmentTable
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
    SELECT = "DISTINCT(s.image_id), s.site_name_id, s.contentUrl, s.imagename, s.face_x, s.face_y, s.face_z, s.mouth_gap, s.face_landmarks, s.bbox, s.face_encodings68, s.site_image_id, s.body_landmarks"
    
    FROM =f"{SegmentTable_name} s "

    # this is the standard segment topics/clusters query for November 2023
    WHERE = "s.site_name_id != 1 AND face_encodings68 IS NOT NULL AND face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2"

    # HIGHER
    # WHERE = "s.site_name_id != 1 AND face_encodings68 IS NOT NULL AND face_x > -27 AND face_x < -23 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2"

    # WHERE += " AND mouth_gap > 15 "
    # WHERE += " AND s.age_id NOT IN (1,2,3,4) "
    # WHERE += " AND s.age_id > 4 "

    ## To add keywords to search
    # FROM += " JOIN ImagesKeywords ik ON s.image_id = ik.image_id JOIN Keywords k ON ik.keyword_id = k.keyword_id "
    # WHERE += " AND k.keyword_text LIKE 'shout%' "

    if IS_CLUSTER or IS_ONE_CLUSTER:
        FROM += " JOIN ImagesClusters ic ON s.image_id = ic.image_id "
    if IS_TOPICS or IS_ONE_TOPIC:
        FROM += " JOIN ImagesTopics it ON s.image_id = it.image_id "
        WHERE += " AND it.topic_score > .3"
        SELECT += ", it.topic_score" # add description here, after resegmenting
    if NO_KIDS:
        WHERE += " AND s.age_id NOT IN (1,2,3) "
    if HSV_CONTROL:
        FROM += " JOIN ImagesBackground ibg ON s.image_id = ibg.image_id "
        # WHERE += " AND ibg.lum > .3"
        SELECT += ", ibg.lum, ibg.lum_bb, ibg.hue, ibg.hue_bb, ibg.sat, ibg.sat_bb " # add description here, after resegmenting
        
    # # join to keywords
    # FROM += " JOIN ImagesKeywords ik ON s.image_id = ik.image_id JOIN Keywords k ON ik.keyword_id = k.keyword_id "
    # WHERE += " AND k.keyword_text LIKE 'surpris%' "

    # WHERE = "s.site_name_id != 1"
    LIMIT = 2000




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
# image_edge_multiplier = [1, 1, 1, 1] # just face
# image_edge_multiplier = [1.5,1.5,2,1.5] # bigger portrait
# image_edge_multiplier = [1.4,2.6,1.9,2.6] # wider for hands
# image_edge_multiplier = [1.2,2.3,1.7,2.3] # medium for hands
image_edge_multiplier = [1.2, 1.2, 1.6, 1.2] # standard portrait


# construct my own objects
sort = SortPose(motion, face_height_output, image_edge_multiplier,EXPAND, ONE_SHOT, JUMP_SHOT, HSV_BOUNDS, VERBOSE)

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

# construct mediapipe objects
mp_drawing = mp.solutions.drawing_utils

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,min_detection_confidence=0.5)


###################
# SQL  FUNCTIONS  #
###################

def selectSQL(cluster_no=None, topic_no=None):
    # print(f"cluster_no is")
    # print(cluster_no)
    if cluster_no is not None or topic_no is not None:
        cluster = " "
        if IS_CLUSTER or IS_ONE_CLUSTER:
            cluster +=f"AND ic.cluster_id = {str(cluster_no)} "
        if IS_TOPICS or IS_ONE_TOPIC:
            # cluster +=f"AND it.topic_id = {str(topic_no)} "
            if isinstance(topic_no, list):
                # Convert the list into a comma-separated string
                topic_ids = ', '.join(map(str, topic_no))
                # Use the IN operator to check if topic_id is in the list of values
                cluster += f"AND it.topic_id IN ({topic_ids}) "
            else:
                # If topic_no is not a list, simply check for equality
                cluster += f"AND it.topic_id = {str(topic_no)} "            
    else:
        cluster=""
    # print(f"cluster SELECT is {cluster}")
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
def sort_by_face_dist(df_enc, df_128_enc, df_33_lms):
    

    this_start = sort.counter_dict["start_img_name"]
    face_distances=[]

    # this prob should be a df.iterrows
    print("df_enc.index")
    print(df_enc.index)
    print(len(df_enc.index))
    # print(sort.counter_dict)
    FIRST_ROUND = True
    if sort.CUTOFF < len(df_enc.index):
        itters = sort.CUTOFF
    else: 
        itters = len(df_enc.index)
    for i in range(itters):
        # find the image
        # print(df_enc)
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
            # print(enc1)
            print("set enc1 from pass through")
        else:
            #this is the first??? round, set via df
            print("trying get_start_enc()")
            enc1, df_128_enc, df_33_lms = sort.get_start_enc(this_start, df_128_enc, df_33_lms, SORT_TYPE)
            # # test to see if get_start_enc was successful
            # # if not, retain previous enc1. or shoudl it reassign median? 
            # if enc1_temp is not None:
            #     enc1 = enc1_temp
            print("set enc1 from get_start_enc() ")

        ## Find closest
        try:
            # closest_dict is now a dict with 1 or more items
            # this_start is a filepath, which serves as df index
            # it is now a dict of key=distance value=filepath
            print("going to get closest")

            # NEED TO GET IT TO DROP FROM df_33_lms in get_closest_df
            # need to send the df_enc with the same two keys through to get_closest
            dist, closest_dict, df_128_enc, df_33_lms = sort.get_closest_df(FIRST_ROUND, enc1,df_enc, df_128_enc, df_33_lms, sorttype=SORT_TYPE)
            # dist, closest_dict, df_128_enc = sort.get_closest_df(FIRST_ROUND, enc1,df_enc, df_128_enc, sorttype="planar")
            # dist, closest_dict, df_128_enc = sort.get_closest_df(enc1,df_enc, df_128_enc)
            FIRST_ROUND = False


            print("got closest")
            # print(closest_dict)

            # Break out of the loop if greater than MAXDIST
            # I think this will be graceful with cluster iteration
            print("dist")
            # print(dist)
            if dist > sort.MAXDIST and sort.SHOT_CLOCK != 0:
                print("should breakout")
                break

        except Exception as e:
            print("exception on going to get closest")
            print(str(e))


     
        # Iterate through the results and append
        dkeys = list(closest_dict.keys())
        dkeys.sort()
        images_to_drop =[]
        print("length of dkeys for closest_dict is ", len(dkeys))
        for dkey in dkeys:


            ## Collect values and append to face_distances
            this_start = closest_dict[dkey]
            print("this_start assigned as ", this_start)
            face_landmarks=None
            bbox=None

            # print("THIS: closest_dict[dkey],")
            # print(closest_dict[dkey])

            try:
                # print("dkey, df_enc.loc[closest_dict[dkey]]")
                # print(dkey)
                # print(closest_dict[dkey])
                # print(df_enc.loc[closest_dict[dkey]])
                site_name_id = df_enc.loc[closest_dict[dkey]]['site_name_id']
                face_landmarks = df_enc.loc[closest_dict[dkey]]['face_landmarks']
                bbox = df_enc.loc[closest_dict[dkey]]['bbox']
                # print("assigned bbox", bbox)
            except:
                print("won't assign landmarks/bbox")
            # print("site_name_id is the following")

            # for some reason, site_name_id is not an int. trying to test if int.
            # print(type(site_name_id))
            # if not pd.is_int(site_name_id): continue
            # print(site_name_id)
            # print("site_specific_root_folder", io.folder_list[site_name_id])
            site_specific_root_folder = io.folder_list[site_name_id]
            # print("site_specific_root_folder")
            # print(site_specific_root_folder)
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
            try:
                df_128_enc=df_128_enc.drop(dropimage)
            except Exception as e:
                print(str(e))

        #debuggin
        print(f"sorted round {str(i)} which is actually round  {str(i+len(dkeys)-1)}")
        print(len(df_128_enc.index))
        if len(df_128_enc.index) < 2:
            break
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
    def create_hsv_list(row):
        if row['hue_bb'] >= 0:
            # print("create_hsv_list bb", [row['hue_bb'], row['sat_bb'], row['lum_bb']])
            return [row['hue_bb']*HSV_WEIGHTS["HUE"], row['sat_bb']*HSV_WEIGHTS["SAT"], row['lum_bb']*HSV_WEIGHTS["LUM"]]
        else:
            return [row['hue']*HSV_WEIGHTS["HUE"], row['sat']*HSV_WEIGHTS["SAT"], row['lum']*HSV_WEIGHTS["LUM"]]    
    # format the encodings for sorting by distance
    # df_enc will be the df with bbox, site_name_id, etc, keyed to filename
    # df_128_enc will be 128 colums of encodings, keyed to filename
    # print("prep_encodings df_segment", df_segment)
    col1="imagename"
    col2="face_encodings68"
    col3="site_name_id"
    col4="face_landmarks"
    col5="bbox"
    col6="body_landmarks"
    col7="hsv"
    # df_enc=pd.DataFrame(columns=[col1, col2, col3, col4, col5])
    df_enc=pd.DataFrame(columns=[col1, col2, col3, col4, col5, col6, col7])
    df_enc = pd.DataFrame({
                col1: df_segment['imagename'], col2: df_segment['face_encodings68'].apply(lambda x: np.array(x)), 
                # col3: df_segment['site_name_id'], col4: df_segment['face_landmarks'], col5: df_segment['bbox']})
                col3: df_segment['site_name_id'], col4: df_segment['face_landmarks'], col5: df_segment['bbox'], col6: df_segment['body_landmarks'], 
                col7: df_segment.apply(lambda row: create_hsv_list(row), axis=1),
                })
    df_enc.set_index(col1, inplace=True)
    if VERBOSE: print(df_enc)
    # Create column names for the 128 encoding columns
    encoding_cols = [f"encoding{i}" for i in range(128)]
    lms_cols = [f"lm{i}" for i in range(33)]

    # Create a new DataFrame with the expanded encoding columns
    enc_expanded = df_enc.apply(lambda row: pd.Series(row[col2], index=encoding_cols), axis=1)
    # lms_expanded = df_enc.apply(lambda row: pd.Series(row[col6], index=lms_cols), axis=1)

    # Concatenate the expanded DataFrame with the original DataFrame
    enc_concat = pd.concat([df_enc, enc_expanded], axis=1)
    # lms_concat = pd.concat([df_enc, lms_expanded], axis=1)

    # make a separate df that just has the encodings
    df_128_enc = enc_concat.drop([col2, col3, col4, col5, col6, col7], axis=1)
    # df_33_lms = lms_concat.drop([col2, col3, col4, col5, col6], axis=1)
    df_33_lms = df_enc.drop([col2, col3, col4, col5, col7], axis=1)

    if VERBOSE: print("df_33_lms", df_33_lms)
    return df_enc, df_128_enc, df_33_lms

def compare_images(last_image, img, face_landmarks, bbox):
    is_face = None
    face_diff = 100 # declaring full value, for first round
    #crop image here:
    if sort.EXPAND:
        cropped_image = sort.expand_image(img, face_landmarks, bbox)
    else:
        cropped_image = sort.crop_image(img, face_landmarks, bbox)
    # print("cropped_image type: ",type(cropped_image))

    # this code takes image i, and blends it with the subsequent image
    # next step is to test to see if mp can recognize a face in the image
    # if no face, a bad blend, try again with i+2, etc. 
    if cropped_image is not None:
        if VERBOSE: print("have a cropped image trying to save", cropped_image.shape)
        # try:
        #     print("last_image is ", type(last_image))
        # except:
        #     print("couldn't test last_image")
        try:
            if not sort.counter_dict["first_run"]:
                if VERBOSE:  print("testing is_face")
                if SORT_TYPE == "planar_body":
                    # skipping test_pair for body, b/c it is meant for face
                    is_face = True
                else:
                    is_face = sort.test_pair(last_image, cropped_image)
                    if is_face:
                        if VERBOSE: print("testing mse to see if same image")
                        face_diff = sort.unique_face(last_image,cropped_image)
                        if VERBOSE: print ("mse ", mse)
                    else:
                        print("failed is_face test")
                        # use cv2 to place last_image and cropped_image side by side in a new image


                        height = max(last_image.shape[0], cropped_image.shape[0])
                        last_image = cv2.resize(last_image, (last_image.shape[1], height))
                        cropped_image = cv2.resize(cropped_image, (cropped_image.shape[1], height))

                        # Concatenate images horizontally
                        combined_image = cv2.hconcat([last_image, cropped_image])
                        outpath_notface = os.path.join(sort.counter_dict["outfolder"],"notface",sort.counter_dict['last_description'][:30]+".jpg")
                        sort.not_make_face.append(outpath_notfacecombined_image)
                        # Save the new image

            else:
                print("first round, skipping the pair test")
        except:
            print("last_image try failed")
        # if is_face or first_run and sort.resize_factor < sort.resize_max:
        if face_diff > sort.FACE_DIST or sort.counter_dict["first_run"]:
            sort.counter_dict["first_run"] = False
            last_image = cropped_image
            sort.counter_dict["good_count"] += 1
        else: 
            print("pair do not make a face, skipping")
            sort.counter_dict["isnot_face_count"] += 1
            return None
    elif cropped_image is None and sort.counter_dict["first_run"]:
        print("first run, but bad first image")
        last_image = cropped_image
        sort.counter_dict["cropfail_count"] += 1
    else:
        print("no image here, trying next")
        sort.counter_dict["cropfail_count"] += 1
    return cropped_image, face_diff


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
    # print("filename", filename)
    UID = filename.split('-id')[-1].split("/")[-1].replace(".jpg","")
    # print("UID ",UID)
    counter_str = str(sort.counter_dict["counter"]).zfill(len(str(df.size)))  # Add leading zeros to the counter
    imgfilename = imgfileprefix+"_"+str(counter_str)+"_"+UID+".jpg"
    print("imgfilename ",imgfilename)
    return imgfilename

def linear_test_df(df_sorted,df_segment,cluster_no, itter=None):
    def save_image_metas(row):
        # print("row")
        print("save_image_metas")
        parent_row = df_segment[df_segment['imagename'] == row['filename']]
        image_id = parent_row['image_id'].values[0]
        # use image_id to retrieve description from mysql database 
        # this is temporary until I resegment the images with description in the segment
        try:
            description = session.query(Images.description).filter(Images.image_id == image_id).first()
        except Exception as e:
            print(str(e))
        # description = parent_row['description'].values[0]
        topic_score = parent_row['topic_score'].values[0]
        metas = [image_id, description[0], topic_score]
        metas_path = os.path.join(sort.counter_dict["outfolder"],METAS_FILE)
        io.write_csv(metas_path, metas)
        # print(image_id, description[0], topic_score)
        # return([image_id, description[0], topic_score])
    #itter is a cap, to stop the process after a certain number of rounds
    print('writing images')
    imgfileprefix = f"X{str(sort.XLOW)}-{str(sort.XHIGH)}_Y{str(sort.YLOW)}-{str(sort.YHIGH)}_Z{str(sort.ZLOW)}-{str(sort.ZHIGH)}_ct{str(df_sorted.size)}"
    print(imgfileprefix)
    good = 0
    img_list = []
    metas_list = []
    for index, row in df_sorted.iterrows():
        # parent_row = df_segment[df_segment['imagename'] == row['filename']]
        # print("parent_row")
        # print(parent_row)

        print('-- linear_test_df [-] in loop, index is', str(index))
        print(row)
        # select the row in df_segment where the imagename == row['filename']
        try:
            imgfilename = const_imgfilename(row['filename'], df_sorted, imgfileprefix)
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
                cropped_image, face_diff = compare_images(sort.counter_dict["last_image"], img, row['face_landmarks'], row['bbox'])


                ###
                # I'm trying to compare descriptions here but it isn't working
                # first_run isn't working. 
                ###

                try:
                    parent_row = df_segment[df_segment['imagename'] == row['filename']]
                    image_id = parent_row['image_id'].values[0]                        
                    description = session.query(Images.description).filter(Images.image_id == image_id).first()
                except Exception as e:
                    print(str(e))
                temp_first_run = sort.counter_dict["first_run"]
                print("temp_first_run")
                print(temp_first_run)
                if sort.counter_dict["first_run"]:
                    sort.counter_dict["last_description"] = description
                    print("first run, setting last_description")
                elif face_diff < 30:
                    print("face_diff too small")
                    # temp, until resegmenting
                    print(description[0])
                    print(sort.counter_dict["last_description"])
                    if description[0] == sort.counter_dict["last_description"]:
                        print("same description!!!")


                if cropped_image is not None:
                    img_list.append((outpath, cropped_image))
                    # this is done in compare function
                    # sort.counter_dict["good_count"] += 1
                    good += 1
                    # print("going to save image metas")
                    save_image_metas(row)
                    # metas_list.append(save_image_metas(row))
                    # parent_row = df_segment[df_segment['imagename'] == row['filename']]
                    # print("parent_row")
                    # print(parent_row)

                    
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
        print("metas_list")
        print(metas_list)
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

    df_enc, df_128_enc, df_33_lms = prep_encodings(df_segment)

    # # get dataframe sorted by distance
    df_sorted = sort_by_face_dist(df_enc, df_128_enc, df_33_lms)

    # test to see if they make good faces
    img_list = linear_test_df(df_sorted,df_segment,cluster_no)
    write_images(img_list)
    write_images(sort.not_make_face)
    print_counters()


###################
#  MY MAIN CODE   #
###################

def main():
    # these are used in cleaning up fresh df from SQL
    def unpickle_array(pickled_array):
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
            df['face_encodings68'] = df['face_encodings68'].apply(unpickle_array)
            df['face_landmarks'] = df['face_landmarks'].apply(unpickle_array)
            df['body_landmarks'] = df['body_landmarks'].apply(unpickle_array)
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
            # need to make sure has HSV here
            df_segment = sort.make_segment(df)
            print("made segment")


            # this is to save files from a segment to the SSD
            print("will I save segment? ", SAVE_SEGMENT)
            if SAVE_SEGMENT:
                Base.metadata.create_all(engine)
                print(df_segment.size)
                save_segment_DB(df_segment)
                print("saved segment to segmentTable")
                quit()

            ### Set counter_dict ###

            sort.set_counters(io.ROOT,cluster_no, start_img_name,start_site_image_id)

            print("set sort.counter_dict:" )
            print(sort.counter_dict)


            ### Get cluster_median encodings for cluster_no ###

            if cluster_no is not None and cluster_no !=0 and IS_CLUSTER:
                # skips cluster 0 for pulling median because it was returning NULL
                # cluster_median = select_cluster_median(cluster_no)
                # image_id = insert_dict['image_id']
                # can I filter this by site_id? would that make it faster or slower? 

                # temp fix
                results = session.query(Clusters).filter(Clusters.cluster_id==cluster_no).first()


                print(results)
                cluster_median = unpickle_array(results.cluster_median)
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
    if IS_CLUSTER and not IS_ONE_TOPIC:
        print(f"IS_CLUSTER is {IS_CLUSTER} with {N_CLUSTERS}")
        for cluster_no in range(N_CLUSTERS):
            print(f"SELECTing cluster {cluster_no} of {N_CLUSTERS}")
            resultsjson = selectSQL(cluster_no, None)
            print(f"resultsjson contains {len(resultsjson)} images")
            map_images(resultsjson, cluster_no)
    elif IS_CLUSTER and IS_ONE_TOPIC:
        print(f"IS_CLUSTER is {IS_CLUSTER} with {N_CLUSTERS}, and topic {TOPIC_NO}")
        for cluster_no in range(N_CLUSTERS):
            print(f"SELECTing cluster {cluster_no} of {N_CLUSTERS}")
            resultsjson = selectSQL(cluster_no, TOPIC_NO)
            print(f"resultsjson contains {len(resultsjson)} images")
            map_images(resultsjson, cluster_no)
    elif IS_TOPICS:
        print(f"IS_TOPICS is {IS_TOPICS} with {N_TOPICS}")
        for topic_no in range(N_TOPICS):
            print(f"SELECTing cluster {topic_no} of {N_TOPICS}")
            resultsjson = selectSQL(None, topic_no)
            print(f"resultsjson contains {len(resultsjson)} images")
            map_images(resultsjson, topic_no)
    elif IS_ONE_CLUSTER:
        print(f"SELECTing cluster {CLUSTER_NO}")
        resultsjson = selectSQL(CLUSTER_NO, None)
        print(f"resultsjson contains {len(resultsjson)} images")
        map_images(resultsjson, CLUSTER_NO)
    elif IS_ONE_TOPIC:
        print(f"SELECTing topic {TOPIC_NO}")
        resultsjson = selectSQL(None, TOPIC_NO)
        print(f"resultsjson contains {len(resultsjson)} images")
        map_images(resultsjson, TOPIC_NO) # passing in TOPIC_NO to use in saving folder name
    else:
        print("doing regular linear")
        resultsjson = selectSQL() 
        map_images(resultsjson)

    print("got results, count is: ",len(resultsjson))


if __name__ == '__main__':
    main()

