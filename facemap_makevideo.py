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
CYCLECOUNT = 2
# ROOT="/Users/michaelmandiberg/Documents/projects-active/facemap_production/"

# keep this live, even if not SSD
# SegmentTable_name = 'May25segment123side_to_side'
# SegmentTable_name = 'May25segment123updown_laugh'
SegmentTable_name = 'May25segment123straight_lessrange'  #actually straight ahead smile

# SATYAM, this is MM specific
# for when I'm using files on my SSD vs RAID
IS_MOVE = False
IS_SSD = True
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
    # FROM ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings e ON i.image_id = e.image_id"
    # FROM =f"Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings6 e ON i.image_id = e.image_id INNER JOIN {SegmentTable_name} seg ON i.site_image_id = seg.site_image_id"

    # don't need keywords if SegmentTable_name
    FROM =f"Images i LEFT JOIN Encodings e ON i.image_id = e.image_id INNER JOIN {SegmentTable_name} seg ON i.site_image_id = seg.site_image_id"
    WHERE = "e.is_face IS TRUE AND e.face_encodings IS NOT NULL AND e.bbox IS NOT NULL AND i.site_name_id = 8 AND i.age_id NOT IN (1,2,3,4)"

    # WHERE = "i.site_image_id LIKE '1402424532'"
    # WHERE = "i.site_image_id IN (1402424532)"
    # WHERE = "i.site_image_id IN (1311507298, 1402424532, 168449643, 1182617710)"

    # WHERE = "e.is_face IS TRUE AND e.bbox IS NOT NULL AND i.site_name_id = 5 AND k.keyword_text LIKE 'smil%'"
    # WHERE = "e.image_id IS NULL "

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


LIMIT = 300


motion = {
    "side_to_side": False,
    "forward_smile": True,
    "forward_nosmile":  False,
    "static_pose":  False,
    "simple": False,
}


# face_height_output is how large each face will be. default is 750
base_image_size = 750

# define ratios, in relationship to nose
# units are ratio of faceheight
# top, right, bottom, left
image_edge_multiplier = [1, 1, 1, 1]
# image_edge_multiplier = [1.2, 1.2, 1.6, 1.2]


# construct my own objects
sort = SortPose(motion, base_image_size, image_edge_multiplier)

start_img = "median"
# start_img = "start_site_image_id"
start_site_image_id = "e/ea/portrait-of-funny-afro-guy-picture-id1402424532.jpg"
# 274243    Portrait of funny afro guy  76865   {"top": 380, "left": 749, "right": 1204, "bottom": 835}



# override io.db for testing mode
# db['name'] = "123test"

engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

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


# create new SegmentTable


mp_drawing = mp.solutions.drawing_utils

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,min_detection_confidence=0.5)


# I/O utils

def selectSQL():
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)};"
    # print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    return(resultsjson)

def get_hash_folders(filename):
    m = hashlib.md5()
    m.update(filename.encode('utf-8'))
    d = m.hexdigest()
    return d[0].upper(), d[0:2].upper()

def make_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value



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



# ### Linear sorting

def get_img_list(folder):
    img_list=[]
    for file in os.listdir(folder):
        if not file.startswith('.') and os.path.isfile(os.path.join(folder, file)):
            filepath = os.path.join(folder, file)
            filepath=filepath.replace('\\' , '/')
            img_list.append(file)
    return img_list        
    print("got image list")
    
def save_sorted(counter, folder, image, dist):
    sorted_name = "linear_sort_"+str(counter)+"_"+str(round(dist, 2))+".jpg"
    sortfolder="sorted2"
    newfolder = os.path.join(folder,sortfolder)
    print(newfolder)
    old_name=os.path.join(folder,image)
    new_name=os.path.join(newfolder,sorted_name)
    print(old_name)
    print(new_name)
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)
    shutil.copy(old_name, new_name)
    print('saved, ',sorted_name)


#compare image bitmaps 



#get distance beetween encodings

def get_d(enc1, enc2):
    enc1=np.array(enc1)
    print("enc1")
    print(enc1[0])
    enc2=np.array(enc2)
    print("enc2")
    print(enc2[0])
    d=np.linalg.norm(enc1 - enc2, axis=0)
    print("d")
    print(d)
    return d

def get_closest_df(start_img, df_enc,site_name_id):
    first = True
    if start_img == "median":
        enc1 = df_enc.median().to_list()
        print("in median")

        # print(enc1)

    elif start_img == "start_site_image_id":
        print("start_site_image_id (this is what we are comparing to)")
        print(start_site_image_id)
        enc1 = df_enc.loc[start_site_image_id].to_list()
        # print(enc1)
        
    else:
#         enc1 = get 2-129 from df via stimg key
        print("start_img key is (this is what we are comparing to):")
        print(start_img)
        enc1 = df_enc.loc[start_img].to_list()
        df_enc=df_enc.drop(start_img)
        first = False
        # print("in new img",len(df_enc.index))
        # print(enc1)
    
#     img_list.remove(start_img)
#     enc1=enc_dict[start_img]
    
    dist=[]
    dist_dict={}
    
    # print("df_enc for ", )
    # print(df_enc)
    
    for index, row in df_enc.iterrows():
#         print(row['c1'], row['c2'])
#     for img in img_list:
        enc2 = row
        print("testing this", index, "against the start img",start_img)
        if (enc1 is not None) and (enc2 is not None):
            # mse = False
            d = get_d(enc1, enc2)
            print ("d is", str(d), "for", index)
            dist.append(d)
            dist_dict[d]=index
    dist.sort()
    print ("the winner is: ", str(dist[0]), dist_dict[dist[0]])
#     print(len(dist))
    return dist[0], dist_dict[dist[0]], df_enc



# test if new and old make a face
def is_face(image):
    # For static images:
    # I think this list is not used
    IMAGE_FILES = []
    with mp_face_detection.FaceDetection(model_selection=1, 
                                        min_detection_confidence=0.6
                                        ) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        if not results.detections:
            is_face = False
        else:
            is_face = True
        return is_face

# in class now...?
# # test if new and old make a face, calls is_face
# def test_pair(last_file, new_file):
#     try:
#         img = cv2.imread(new_file)
#         height, width, layers = img.shape
#         size = (width, height)
#         print('loaded img 1')
        
#         # I think this should be "last_file"
#         last_img = cv2.imread(new_file)
#         last_height, last_width, last_layers = last_img.shape
#         last_size = (last_width, last_height)
#         print('loaded img 2')
        
#         # test to see if this is actually an face, to get rid of blank ones/bad ones
#         if is_face(img):
#             print('new file is face')
#             # if not the first image
# #             if i>0:
#             # blend this image with the last image
#             blend = cv2.addWeighted(img, 0.5, last_img, 0.5, 0.0)
#             print('blended faces')
#             blended_face = is_face(blend)
#             print('is_face ',blended_face)
#             # if blended image has a detectable face, append the img
#             if blended_face:
# #                     img_array.append(img)
#                 print('test_pair is a face! adding it')
#                 return True
#             else:
#                 print('skipping this one')
#                 return False
#             # for the first one, just add the image
#             # this may need to be refactored in case the first one is bad?
# #             else:
# #                 print('this is maybe the first round?')
# #                 img_array.append(img)
#         else:
#             print('new_file is not face: ',new_file)
#             return False

# #         i+=1

#     except:
#         print('failed:',new_file)
#         return False





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
        dist, start_img, df_128_enc = get_closest_df(start_img,df_128_enc,site_name_id)
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



###################
#  MY MAIN CODE   #
###################

def main():
    def unpickle_array(pickled_array):
        return pickle.loads(pickled_array)
    def unstring_json(json_string):
        eval_string = ast.literal_eval(json_string)
        json_dict = json.loads(eval_string)
        return json_dict

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
        hash_folder1, hash_folder2 = get_hash_folders(file_name)
        newname = os.path.join(hash_folder1, hash_folder2, file_name)
        return newname
        # file_name = file_name_path.split('/')[-1]
    print("in main, making SQL query")



    #creating my objects
    start = time.time()

    resultsjson = selectSQL()
    print("got results, count is: ",len(resultsjson))


    # print(df_sql)

    # read the csv and construct dataframe
    try:
        df = pd.json_normalize(resultsjson)
        print(df)


    except:
        print('you forgot to change the filename DUH')
    if df.empty:
        print('dataframe empty, probably bad path')
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

    # print("raw df from DB")
    # print(df['face_encodings'])


# turning this off for debugging
    # ### PROCESS THE DATA ###

    # # make the segment based on settings
    # df_segment = sort.make_segment(df)

    # # get list of all angles in segment
    # angle_list = sort.createList(df_segment)

    # # sort segment by angle list
    # # creates sort.d attribute: a dataframe organized (indexed?) by angle list
    # sort.get_divisor(df_segment)

    # # # is this used anywhere? 
    # # angle_list_pop = angle_list.pop()

    # # get median for first sort
    # median = sort.get_median()

    # # get metamedian for second sort, creates sort.metamedian attribute
    # sort.get_metamedian()


# adding this for debugging

    print("about to segment")
    # make the segment based on settings
    df_segment = sort.make_segment(df)
    print(df_segment)

    # duplicate_site_ids = df_segment[df_segment.duplicated(['site_image_id'])]['site_image_id']
    # print("duplicate_site_ids")
    # print(duplicate_site_ids)

    print("will I save segment? ", SAVE_SEGMENT)
    if SAVE_SEGMENT:
        Base.metadata.create_all(engine)
        print(df_segment.size)
        save_segment_DB(df_segment)
        print("saved segment to ", SegmentTable_name)
        quit()

    # df_segment = df

    # # OLD format the encodings for sorting by distance
    col1="imagename"
    col2="face_encodings"
    col3="site_name_id"
    col4="face_landmarks"
    col5="bbox"
    df_enc=pd.DataFrame(columns=[col1, col2, col3, col4, col4])
    df_enc = pd.DataFrame({col1: df_segment['imagename'], col2: df_segment['face_encodings'].apply(lambda x: np.array(x)), 
                col3: df_segment['site_name_id'], col4: df_segment['face_landmarks'], col5: df_segment['bbox'] })
    df_enc.set_index(col1, inplace=True)


    print(df_enc)

    # Create column names for the 128 encoding columns
    encoding_cols = [f"encoding{i}" for i in range(128)]

    # Create a new DataFrame with the expanded encoding columns
    df_expanded = df_enc.apply(lambda row: pd.Series(row[col2], index=encoding_cols), axis=1)

    # Concatenate the expanded DataFrame with the original DataFrame
    df_final = pd.concat([df_enc, df_expanded], axis=1)

    # Optionally, drop the original 'face_encodings' column
    # df_final.drop(col2, axis=1, inplace=True)
    df_128_enc = df_final.drop([col2, col3, col4, col5], axis=1)

    print(df_128_enc)

    # start_img = "f/f4/young-woman-laughing-picture-id1001121288.jpg"
    # enc1 = df_enc.loc[start_img].to_list()
    # print(enc1)


    # for index, row in df_128_enc.iterrows():
    #     enc2 = row
    #     print("this is the enc2 row passing in", enc2)

    ### BUILD THE LIST OF SELECTED IMAGES ###

    # img_array is actual bitmap data? 
    if motion["side_to_side"] is True:
        img_list, size = cycling_order(CYCLECOUNT, sort)
        # size = sort.get_cv2size(ROOT, img_list[0])
    else:
    # dont neet to pass SECOND_SORT, because it is already there

        # img_list, size = simple_order(segment)


        # not being used currently
        # save_sorted(i, folder, start_img, dist)

        # # get dataframe sorted by distance

        df_sorted = sort_by_face_dist(start_img,df_enc, df_128_enc)
        print("df_sorted")
        print(df_sorted)
        # img_list = df_sorted['filename'].tolist()
        # # the hardcoded #1 needs to be replaced with site_name_id, which needs to be readded to the df
        # site_specific_root_folder = io.folder_list[1]
        # size = sort.get_cv2size(site_specific_root_folder, img_list[0])
        # # print(img_list)



        # img_array, size = sort.simplest_order(segment) 

    # print("img_array: ",img_array)
    ### WRITE THE IMAGES TO VIDEO/FILES ###

    if VIDEO == True:
        #save individual as video
        # need to rework to accept df and calc size internally
        sort.write_video(io.ROOT, img_list, df_segment, size)

    else:
        #save individual as images
        sort.write_images(io.ROOT, df_sorted)


if __name__ == '__main__':
    main()

