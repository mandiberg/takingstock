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
from my_declarative_base import Base, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

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
# SegmentTable_name = 'May25segment123side_to_side'
SegmentTable_name = 'May25segment123updown_laugh'
# SegmentTable_name = 'May25segment123straight_lessrange'  #actually straight ahead smile

# SATYAM, this is MM specific
# for when I'm using files on my SSD vs RAID
IS_MOVE = False
IS_SSD = True
IS_CLUSTER = False
IS_ONE_CLUSTER = True
IS_ANGLE_SORT = False
# number of clusters to analyze -- this is also declared in Clustering_SQL. Move to IO?
N_CLUSTERS = 128
# this is for IS_ONE_CLUSTER to only run on a specific cluster
CLUSTER_NO = 11

# I/O utils
io = DataIO(IS_SSD)
db = io.db
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES

# if IS_SSD:
#     io.ROOT = io.ROOT_PROD 
# else:
#     io.ROOT = io.ROOT36


if not IS_MOVE:
    print("production run. IS_SSD is", IS_SSD)

    # # # # # # # # # # # #
    # # for production  # #
    # # # # # # # # # # # #

    SAVE_SEGMENT = False
    SELECT = "DISTINCT(i.image_id), i.site_name_id, i.contentUrl, i.imagename, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings, i.site_image_id"

    # don't need keywords if SegmentTable_name
    # this is for MM segment table
    # FROM =f"Images i LEFT JOIN Encodings e ON i.image_id = e.image_id INNER JOIN {SegmentTable_name} seg ON i.site_image_id = seg.site_image_id"
    # WHERE = "e.is_face IS TRUE AND e.face_encodings IS NOT NULL AND e.bbox IS NOT NULL AND i.site_name_id = 8 AND i.age_id NOT IN (1,2,3,4)"

    # this is for gettytest3 table
    FROM ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings e ON i.image_id = e.image_id JOIN ImagesClusters ic ON i.image_id = ic.image_id"
    WHERE = "e.is_face IS TRUE AND e.bbox IS NOT NULL AND i.site_name_id = 8 AND k.keyword_text LIKE 'smil%'"

elif IS_MOVE:
    print("moving to SSD")

    # # # # # # # # # # # #
    # # for move to SSD # #
    # # # # # # # # # # # #

    SAVE_SEGMENT = True
    SELECT = "DISTINCT(i.image_id), i.site_name_id, i.contentUrl, i.imagename, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings, i.site_image_id"
    FROM ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings e ON i.image_id = e.image_id"
    # # for smiling images
    WHERE = "e.is_face IS TRUE AND e.face_encodings IS NOT NULL AND e.bbox IS NOT NULL AND i.site_name_id = 8 AND k.keyword_text LIKE 'smil%'"


    # # for laugh images
    # WHERE = "e.is_face IS TRUE AND e.face_encodings IS NOT NULL AND e.bbox IS NOT NULL AND i.site_name_id = 8 AND k.keyword_text LIKE 'laugh%'"
    # SegmentTable_name = 'SegmentForward123laugh'

    # yelling, screaming, shouting, yells, laugh; x is -4 to 30, y ~ 0, z ~ 0
    # regular rotation left to right, which should include the straight ahead? 


LIMIT = 10000

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
# base_image_size = 750
face_height_output = 400

# define ratios, in relationship to nose
# units are ratio of faceheight
# top, right, bottom, left
# image_edge_multiplier = [1, 1, 1, 1]
# image_edge_multiplier = [1.5, 2, 1.5, 2]
image_edge_multiplier = [1.2, 1.2, 1.6, 1.2]


# construct my own objects
sort = SortPose(motion, face_height_output, image_edge_multiplier,EXPAND)

start_img = "median"
# start_img = "start_site_image_id"
start_site_image_id = "e/ea/portrait-of-funny-afro-guy-picture-id1402424532.jpg"
# start_site_image_id = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/images_123rf/E/E8/95447708-portrait-of-happy-smiling-beautiful-young-woman-touching-skin-or-applying-cream-isolated-over-white.jpg"
# 274243    Portrait of funny afro guy  76865   {"top": 380, "left": 749, "right": 1204, "bottom": 835}
enc_persist = None


# override io.db for testing mode
# db['name'] = "123test"

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
    body_landmarks = Column(BLOB)
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
        cluster =f"AND ic.cluster_id = {str(cluster_no)}"
    else:
        cluster=""
    print(f"cluster SELECT is {cluster}")
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} {cluster} LIMIT {str(LIMIT)};"
    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    return(resultsjson)



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
            face_landmarks=pickle.dumps(row['face_landmarks']),
            bbox=row['bbox'],
            face_encodings=pickle.dumps(row['face_encodings']),
            site_image_id=row['site_image_id']
        )
        session.add(instance)
    session.commit()



###################
# SORT FUNCTIONS  #
###################


# takes a dataframe of images and encodings and returns a df sorted by distance
def sort_by_face_dist(start_img,df_enc, df_128_enc):
    face_distances=[]
    # this prob should be a df.iterrows
    for i in range(len(df_enc.index)-2):
        # find the image
        print(df_enc)
        # this is the site_name_id for start_img, needed to test mse
        if start_img is "median":
            # setting to zero for first one, as not relevant
            site_name_id = 0
        else: 
            site_name_id = df_enc.loc[start_img]['site_name_id']
        # the hardcoded #1 needs to be replaced with site_name_id, which needs to be readded to the df
        print("starting sort round ",str(i))
        dist, start_img, df_128_enc = sort.get_closest_df(start_img,df_128_enc,site_name_id)
        # dist[0], dist_dict[dist[0]]
        thisimage=None
        face_landmarks=None
        bbox=None
        try:
            site_name_id = df_enc.loc[start_img]['site_name_id']
            face_landmarks = df_enc.loc[start_img]['face_landmarks']
            bbox = df_enc.loc[start_img]['bbox']
            print("assigned bbox", bbox)
        except:
            print("won't assign landmarks/bbox")
        site_specific_root_folder = io.folder_list[site_name_id]
        # save the image -- this prob will be to append to list, and return list? 
        # save_sorted(i, folder, start_img, dist)
        this_dist=[dist, site_specific_root_folder, start_img, site_name_id, face_landmarks, bbox]
        face_distances.append(this_dist)

        #debuggin
        print("sorted round ",str(i))
        print(len(df_128_enc.index))
        print(dist)
        print (start_img)
        for row in face_distances:
            print(str(row[0]), row[2])
    df = pd.DataFrame(face_distances, columns =['dist', 'folder', 'filename','site_name_id','face_landmarks', 'bbox'])
    print(df)
    # df = df.sort_values(by=['dist']) # this was sorting based on delta distance, not sequential distance
    # print(df)
    return df


# not currently in use
def simple_order(segment):
    img_array = []
    delta_array = []
    # size = []
    #simple ordering
    rotation = segment.sort_values(by=sort.SORT)
    print("rotation: ")
    print(rotation)

    # for num, name in enumerate(presidents, start=1):
    i = 0
    for index, row in rotation.iterrows():
        # print(index, row['x'], row['y'], row['imagename'])
        delta_array.append(row['mouth_gap'])
        try:
            print(row['imagename'])
            #this is pointin to the wrong location
            #/Users/michaelmandiberg/Documents/projects-active/facemap_production/gettyimages_output_feb/crop_1_3.9980554263116233_-3.8588402232564545_2.2552074063078456_0.494_portrait-of-young-woman-covering-eye-with-compass-morocco-picture-id1227557214.jpg
            #original files are actually here:
            #/Users/michaelmandiberg/Documents/projects-active/facemap_production/gettyimages/newimages/F/F0/portrait-of-young-woman-covering-eye-with-compass-morocco-picture-id1227557214.jpg
            #doesn't seem to be picking up the cropped image.

            newimage = cv2.imread(row['imagename'])
            height, width, layers = img.shape
            size = (width, height)
            # test to see if this is actually an face, to get rid of blank ones/bad ones
            # this may not be necessary
            if sort.is_face(img):
                # if not the first image
                print('is_face')
                if i>0:
                    print('i is greater than 0')
                    # blend this image with the last image
                    # blend = cv2.addWeighted(img, 0.5, img, 0.5, 0.0)
                    # # blend = cv2.addWeighted(img, 0.5, img_array[i-1], 0.5, 0.0)
                    # blended_face = sort.is_face(blend)
                    # print('is_face ',blended_face)
                    # if blended image has a detectable face, append the img
                    if blend_is_face(oldimage, newimage):
                        img_array.append(img)
                        print('simple_order is a face! adding it')
                    else:
                        print('skipping this one')
                # for the first one, just add the image
                # this may need to be refactored in case the first one is bad?
                else:
                    img_array.append(img)
            else:
                print('skipping this one: ',row['imagename'])

            i+=1
            oldimage = newimage

        except:
            print('failed:',row['imagename'])
    # print("delta_array")
    # print(delta_array)
    return img_array, size



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
    col1="imagename"
    col2="face_encodings"
    col3="site_name_id"
    col4="face_landmarks"
    col5="bbox"
    df_enc=pd.DataFrame(columns=[col1, col2, col3, col4, col5])
    df_enc = pd.DataFrame({col1: df_segment['imagename'], col2: df_segment['face_encodings'].apply(lambda x: np.array(x)), 
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

def compare_images(last_image, img, face_landmarks, bbox, first_run, good_count, isnot_face_count, cropfail_count):
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
        print(cropped_image.shape)
        print("have a cropped image trying to save")
        try:
            print(type(last_image))
        except:
            print("couldn't test last_image")
        try:
            if not first_run:
                print("testing is_face")
                is_face = sort.test_pair(last_image, cropped_image)
                if is_face:
                    print("same person, testing mse")
                    is_face = sort.unique_face(last_image,cropped_image)
                    print ("mse ",mse)
            else:
                print("first round, skipping the pair test")
        except:
            print("last_image try failed")
        # if is_face or first_run and sort.resize_factor < sort.resize_max:
        if is_face or first_run:
            first_run = False
            last_image = cropped_image
            good_count += 1
        else: 
            print("pair do not make a face, skipping")
            isnot_face_count += 1
    else:
        print("no image here, trying next")
        cropfail_count += 1
    return last_image, cropped_image, first_run, good_count, isnot_face_count, cropfail_count

def write_images(df,cluster_no):
    print('writing images')
    # imgfileprefix = f"faceimg_crop{str(sort.MINCROP)}_X{str(sort.XLOW)}toX{str(sort.XHIGH)}_Y{str(sort.YLOW)}toY{str(sort.YHIGH)}_Z{str(sort.ZLOW)}toZ{str(sort.ZHIGH)}_maxResize{str(sort.MAXRESIZE)}_ct{str(df.size)}"
    imgfileprefix = f"X{str(sort.XLOW)}-{str(sort.XHIGH)}_Y{str(sort.YLOW)}-{str(sort.YHIGH)}_Z{str(sort.ZLOW)}-{str(sort.ZHIGH)}_ct{str(df.size)}"
    print(imgfileprefix)
    outfolder = os.path.join(io.ROOT,"cluster"+str(cluster_no)+"_"+str(time.time()))
    if not os.path.exists(outfolder):      
        os.mkdir(outfolder)

    try:
        counter = 1
        good_count = 0
        isnot_face_count = 0
        cropfail_count = 0
        sort.negmargin_count = 0
        sort.toosmall_count = 0 
        last_image = None
        first_run = True
        for index, row in df.iterrows():
            print('in loop, index is', str(index))
            UID = row['filename'].split('-id')[-1].split("/")[-1].replace(".jpg","")
            print("UID ",UID)
            counter_str = str(counter).zfill(len(str(df.size)))  # Add leading zeros to the counter
            imgfilename = imgfileprefix+"_"+str(counter_str)+"_"+UID+".jpg"
            print("imgfilename ",imgfilename)
            outpath = os.path.join(outfolder,imgfilename)
            print("outpath ",outpath)

            # folder is specific to each file's site_name_id
            # this is how it was, and seems hardcoded to Test36
            # open_path = os.path.join(ROOT,row['folder'],row['filename'])

            # here I'm using the actual root. Root gets pulled from io, then passed back to sort pose.
            # but the folder is fused to the root somewhere... in makevideo? it needs to be found and pulled off there. 
            open_path = os.path.join(io.ROOT,row['folder'].replace("/Volumes/Test36/",""),row['filename'])
            img = cv2.imread(open_path)
            if row['dist'] < sort.MAXDIST:
                # compare_images to make sure they are face and not the same
                last_image, cropped_image, first_run, good_count, isnot_face_count, cropfail_count = compare_images(last_image, img, row['face_landmarks'], row['bbox'], first_run, good_count, isnot_face_count, cropfail_count)
                if cropped_image is not None:
                    cv2.imwrite(outpath, cropped_image)
                    print("saved: ",outpath)
            else:
                print("MAXDIST too big:" , str(sort.MAXDIST))


            counter += 1

        print("good_count")
        print(good_count)
        print("isnot_face_count")
        print(isnot_face_count)
        print("cropfail_count")
        print(cropfail_count)
        print("sort.negmargin_count")
        print(sort.negmargin_count)
        print("sort.toosmall_count")
        print(sort.toosmall_count)
        print("total count")
        print(counter)

        print('wrote files')
    except Exception as e:
        print(str(e))

def write_images_by_angle(df,cluster_no, sort):
    print('writing images')
    # imgfileprefix = f"faceimg_crop{str(sort.MINCROP)}_X{str(sort.XLOW)}toX{str(sort.XHIGH)}_Y{str(sort.YLOW)}toY{str(sort.YHIGH)}_Z{str(sort.ZLOW)}toZ{str(sort.ZHIGH)}_maxResize{str(sort.MAXRESIZE)}_ct{str(df.size)}"
    imgfileprefix = f"X{str(sort.XLOW)}-{str(sort.XHIGH)}_Y{str(sort.YLOW)}-{str(sort.YHIGH)}_Z{str(sort.ZLOW)}-{str(sort.ZHIGH)}_ct{str(df.size)}"
    print(imgfileprefix)
    outfolder = os.path.join(io.ROOT,"cluster"+str(cluster_no)+"_"+str(time.time()))
    if not os.path.exists(outfolder):      
        os.mkdir(outfolder)


    #cycling patch
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
            # print(d[angle].iloc[(d[angle][sort.SECOND_SORT]-metamedian).abs().argsort()[:2]])
            if(d[angle].size) != 0:
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
                    continue



                    img = cv2.imread(closest_file)
                    sort.preview_img(img)
                    height, width, layers = img.shape
                    size = (width, height)
                    img_array.append(img)
                except:
                    print('failed cycle angle:')
                    # print('failed:',row['imagename'])
            else:
                print("skipping empty angle")
        print('finished a cycle')
        sort.angle_list.reverse()
        cycle = cycle +1
        # # print(angle_list)
    quit()
    try:
        counter = 1
        good_count = 0
        isnot_face_count = 0
        cropfail_count = 0
        sort.negmargin_count = 0
        sort.toosmall_count = 0 
        last_image = None
        is_face = None
        first_run = True
        for index, row in df.iterrows():
            print('in loop, index is', str(index))
            UID = row['filename'].split('-id')[-1].split("/")[-1].replace(".jpg","")
            print("UID ",UID)
            counter_str = str(counter).zfill(len(str(df.size)))  # Add leading zeros to the counter
            imgfilename = imgfileprefix+"_"+str(counter_str)+"_"+UID+".jpg"
            print("imgfilename ",imgfilename)
            outpath = os.path.join(outfolder,imgfilename)
            print("outpath ",outpath)

            # folder is specific to each file's site_name_id
            # this is how it was, and seems hardcoded to Test36
            # open_path = os.path.join(ROOT,row['folder'],row['filename'])

            # here I'm using the actual root. Root gets pulled from io, then passed back to sort pose.
            # but the folder is fused to the root somewhere... in makevideo? it needs to be found and pulled off there. 
            open_path = os.path.join(io.ROOT,row['folder'].replace("/Volumes/Test36/",""),row['filename'])
            img = cv2.imread(open_path)

            #crop image here:
            if sort.EXPAND:
                cropped_image = sort.expand_image(img, row['face_landmarks'], row['bbox'])
            else:
                cropped_image = sort.crop_image(img, row['face_landmarks'], row['bbox'])
            print("cropped_image type: ",type(cropped_image))

            # this code takes image i, and blends it with the subsequent image
            # next step is to test to see if mp can recognize a face in the image
            # if no face, a bad blend, try again with i+2, etc. 
            if cropped_image is not None:
                print(cropped_image.shape)
                print("have a cropped image trying to save")
                try:
                    print(type(last_image))
                except:
                    print("couldn't test last_image")
                try:
                    if not first_run:
                        print("testing is_face")
                        is_face = sort.test_pair(last_image, cropped_image)
                        if is_face and row['dist'] < sort.MAXDIST:
                            print("same person, testing mse")
                            is_face = sort.unique_face(last_image,cropped_image)
                            print ("mse ",mse)
                    else:
                        print("first round, skipping the pair test")
                except:
                    print("last_image try failed")
                # if is_face or first_run and sort.resize_factor < sort.resize_max:
                if is_face or first_run:
                    first_run = False
                    cv2.imwrite(outpath, cropped_image)
                    last_image = cropped_image
                    print("saved: ",outpath)
                    good_count += 1
                else: 
                    print("pair do not make a face, skipping")
                    isnot_face_count += 1
            else:
                print("no image here, trying next")
                cropfail_count += 1
            counter += 1

        print("good_count")
        print(good_count)
        print("isnot_face_count")
        print(isnot_face_count)
        print("cropfail_count")
        print(cropfail_count)
        print("sort.negmargin_count")
        print(sort.negmargin_count)
        print("sort.toosmall_count")
        print(sort.toosmall_count)
        print("total count")
        print(counter)

        print('wrote files')
    except Exception as e:
        print(str(e))


###################
#  MY MAIN CODE   #
###################

def main():
    # these are used in cleaning up fresh df from SQL
    def unpickle_array(pickled_array):
        return pickle.loads(pickled_array)
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
        if df.empty:
            print('dataframe empty, probably bad path or bad SQL')
            sys.exit()

        # Apply the unpickling function to the 'face_encodings' column
        df['face_encodings'] = df['face_encodings'].apply(unpickle_array)
        df['face_landmarks'] = df['face_landmarks'].apply(unpickle_array)
        df['bbox'] = df['bbox'].apply(lambda x: unstring_json(x))
        # turn URL into local hashpath (still needs local root folder)
        df['imagename'] = df['contentUrl'].apply(newname)
        # make decimals into float
        columns_to_convert = ['face_x', 'face_y', 'face_z', 'mouth_gap']
        df[columns_to_convert] = df[columns_to_convert].applymap(make_float)

        ### SEGMENT THE DATA ###

        # make the segment based on settings
        df_segment = sort.make_segment(df)

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

        # this is to save files from a segment to the SSD
        print("will I save segment? ", SAVE_SEGMENT)
        if SAVE_SEGMENT:
            Base.metadata.create_all(engine)
            print(df_segment.size)
            save_segment_DB(df_segment)
            print("saved segment to ", SegmentTable_name)
            quit()


        ### SORT THE LIST OF SELECTED IMAGES ###

        # img_array is actual bitmap data? 
        if motion["side_to_side"] is True:
            img_list, size = cycling_order(CYCLECOUNT, sort)
            # size = sort.get_cv2size(ROOT, img_list[0])
        elif IS_ANGLE_SORT is True:
            write_images_by_angle(df_segment, cluster_no, sort)
        else:
            # simple sort by encoding distance
            # preps the encodings for sort
            df_enc, df_128_enc = prep_encodings(df_segment)

            # # get dataframe sorted by distance
            df_sorted = sort_by_face_dist(start_img,df_enc, df_128_enc)
            print("df_sorted")
            print(df_sorted)
            write_images(df_sorted, cluster_no)

            # img_list = df_sorted['filename'].tolist()
            # # the hardcoded #1 needs to be replaced with site_name_id, which needs to be readded to the df
            # site_specific_root_folder = io.folder_list[1]
            # size = sort.get_cv2size(site_specific_root_folder, img_list[0])
            # # print(img_list)



            # img_array, size = sort.simplest_order(segment) 

        # print("img_array: ",img_array)

        ### WRITE THE IMAGES TO VIDEO/FILES ###
        # turning off for now

        # if VIDEO == True:
        #     #save individual as video
        #     # need to rework to accept df and calc size internally
        #     sort.write_video(io.ROOT, img_list, df_segment, size)

        # else:
        #     #save individual as images
        #     sort.write_images(io.ROOT, df_sorted, cluster_no)


    ### this is the start of the real action ###

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
    if IS_ONE_CLUSTER:
        print(f"SELECTing cluster {CLUSTER_NO}")
        resultsjson = selectSQL(CLUSTER_NO)
        print(f"resultsjson contains {len(resultsjson)} images")
        map_images(resultsjson, CLUSTER_NO)
    else:
        resultsjson = selectSQL() 
        map_images(resultsjson)

    print("got results, count is: ",len(resultsjson))


if __name__ == '__main__':
    main()

