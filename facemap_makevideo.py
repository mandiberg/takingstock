import cv2
import pandas as pd
import os
import time
import sys
import pickle
import hashlib


#linear sort imports non-class
import numpy as np
import mediapipe as mp
import face_recognition_models
import dlib
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import matplotlib.pyplot as plt
import imutils
from imutils import face_utils
import face_recognition
import shutil
from sys import platform
from pathlib import Path

from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool


#mine
from mp_sort_pose import SortPose
from mp_db_io import DataIO

VIDEO = False
CYCLECOUNT = 2
# ROOT="/Users/michaelmandiberg/Documents/projects-active/facemap_production/"
MAPDATA_FILE = "allmaps_62607.csv"

SELECT = "i.image_id, i.site_name_id, i.contentUrl, i.imagename, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_encodings"
# SELECT = "DISTINCT(i.image_id), i.gender_id, author, caption, contentUrl, description, imagename"
FROM ="Images i JOIN ImagesKeywords ik ON i.image_id = ik.image_id JOIN Keywords k on ik.keyword_id = k.keyword_id LEFT JOIN Encodings e ON i.image_id = e.image_id "
WHERE = "e.face_x IS NOT NULL AND i.site_name_id = 1 AND k.keyword_text LIKE 'smil%'"
# WHERE = "e.image_id IS NULL "
LIMIT = 5000


motion = {
    "side_to_side": False,
    "forward_smile": False,
    "forward_nosmile":  True,
    "static_pose":  False,
    "simple": False,
}


# construct my own objects
sort = SortPose(motion)
io = DataIO()
db = io.db
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES

engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
metadata = MetaData(engine)

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
    return d[0], d[0:2]

def make_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


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


#get distance beetween encodings

def get_d(enc1, enc2):
    enc1=np.array(enc1)
    # enc2=np.array(enc2)
    print("enc1 size ")
    print(enc1.shape)
    print("enc2 size ")
    print(enc2.shape)
    d=np.linalg.norm(enc1 - enc2, axis=0)
    print("get_d: ",d)
    return d

def get_closest_df_v2(start_img, df_enc):
    if start_img == "median":
        # print("df_enc: ", df_enc)
        encodings_array = np.array(df_enc['encoding'].tolist())

        # Calculate the median along axis 0
        enc1 = np.median(encodings_array, axis=0)
        # print("median_encoding ",enc1)

        # enc1 = df_enc.median()
        # print("in median: ", enc1)
    else:
#         enc1 = get 2-129 from df via stimg key
        enc1 = df_enc.loc[start_img]['encoding'].tolist()
        print("in else")
        print(enc1)
        print(enc1[0])
        df_enc=df_enc.drop(start_img)
#         print("in new img",len(df_enc.index))
    
#     img_list.remove(start_img)
#     enc1=enc_dict[start_img]
    
    dist=[]
    dist_dict={}
    for index, row in df_enc.iterrows():
#         print(row['c1'], row['c2'])
#     for img in img_list:

        enc2 = row['encoding']
        if (enc1 is not None) and (enc2 is not None):
            d = get_d(enc1, enc2)
            print(d)
            if len(enc1) >1:
                dist.append(d)
                dist_dict[d]=index
            else:
                print("seems like an 128 array OOPS")
    dist.sort()
#     print(len(dist))
    return dist[0], dist_dict[dist[0]], df_enc



def get_closest_df(folder, start_img, df_enc):
    if start_img == "median":
        enc1 = df_enc.median().to_list()
#         print("in median")
    else:
#         enc1 = get 2-129 from df via stimg key
        enc1 = df_enc.loc[start_img].to_list()
        df_enc=df_enc.drop(start_img)
#         print("in new img",len(df_enc.index))
    
    dist=[]
    dist_dict={}
    for index, row in df_enc.iterrows():
        enc2 = row
        if enc1 and enc2:
            d = get_d(enc1, enc2)
            dist.append(d)
            dist_dict[d]=index
    dist.sort()
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


# test if new and old make a face
def test_pair(last_file, new_file):
    try:
        img = cv2.imread(new_file)
        height, width, layers = img.shape
        size = (width, height)
        print('loaded img 1')
        
        last_img = cv2.imread(new_file)
        last_height, last_width, last_layers = last_img.shape
        last_size = (last_width, last_height)
        print('loaded img 2')
        
        # test to see if this is actually an face, to get rid of blank ones/bad ones
        if is_face(img):
            print('new file is face')
            # if not the first image
#             if i>0:
            # blend this image with the last image
            blend = cv2.addWeighted(img, 0.5, last_img, 0.5, 0.0)
            print('blended faces')
            blended_face = is_face(blend)
            print('is_face ',blended_face)
            # if blended image has a detectable face, append the img
            if blended_face:
#                     img_array.append(img)
                print('is a face! adding it')
                return True
            else:
                print('skipping this one')
                return False
            # for the first one, just add the image
            # this may need to be refactored in case the first one is bad?
#             else:
#                 print('this is maybe the first round?')
#                 img_array.append(img)
        else:
            print('new_file is not face: ',new_file)
            return False

#         i+=1

    except:
        print('failed:',new_file)
        return False





###################
# SORT FUNCTIONS  #
###################


# takes a dataframe of images and encodings and returns a df sorted by distance
def sort_by_face_dist(start_img,df_enc):
    face_distances=[]
    for i in range(len(df_enc.index)-2):
        # find the image
        print(df_enc)
        # the hardcoded #1 needs to be replaced with site_name_id, which needs to be readded to the df
        site_specific_root_folder = io.folder_list[1]
        print("starting sort round ",str(i))
        dist, start_img, df_enc = get_closest_df_v2(start_img,df_enc)
        # save the image -- this prob will be to append to list, and return list? 
        # save_sorted(i, folder, start_img, dist)
        this_dist=[dist, site_specific_root_folder, start_img]
        face_distances.append(this_dist)

        #debuggin
        print("sorted round ",str(i))
        print(len(df_enc.index))
        print(dist)
        print (start_img)
    df = pd.DataFrame(face_distances, columns =['dist', 'folder', 'filename'])
    df = df.sort_values(by=['dist'])
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
                        print('is a face! adding it')
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
    def newname(contentUrl):
        file_name_path = contentUrl.split('?')[0]
        file_name = file_name_path.split('/')[-1]
        extension = file_name.split('.')[-1]
        if not file_name.endswith(".jpg"):
            file_name += ".jpg"    
        hash_folder1, hash_folder2 = get_hash_folders(file_name)
        newname = os.path.join(hash_folder1, hash_folder2, file_name)
        return newname
        # file_name = file_name_path.split('/')[-1]
    print("main")



    #creating my objects
    start = time.time()

    resultsjson = selectSQL()
    print("got results, count is: ",len(resultsjson))


    # print(df_sql)

    # read the csv and construct dataframe
    try:
        # df = pd.read_csv(os.path.join(ROOT,MAPDATA_FILE))
        df = pd.json_normalize(resultsjson)



    except:
        print('you forgot to change the filename DUH')
    if df.empty:
        print('dataframe empty, probably bad path')
        sys.exit()

    # Apply the unpickling function to the 'face_encodings' column
    df['face_encodings'] = df['face_encodings'].apply(unpickle_array)
    # turn URL into local hashpath (still needs local root folder)
    df['imagename'] = df['contentUrl'].apply(newname)
    # make decimals into float
    columns_to_convert = ['face_x', 'face_y', 'face_z', 'mouth_gap']
    df[columns_to_convert] = df[columns_to_convert].applymap(make_float)

    print(df)

    ### PROCESS THE DATA ###

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

    col1="file_name"
    col2="encoding"

    df_enc=pd.DataFrame(columns=[col1, col2])

    
    df_enc = pd.DataFrame({col1: df_segment['imagename'], col2: df_segment['face_encodings'].apply(lambda x: np.array(x))})
    df_enc.set_index(col1, inplace=True)



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
        start_img = "median"
        #ROOT is where it is borking
        df_sorted = sort_by_face_dist(start_img,df_enc)
        print("df_sorted")
        print(df_sorted)
        img_list = df_sorted['filename'].tolist()
        # the hardcoded #1 needs to be replaced with site_name_id, which needs to be readded to the df
        site_specific_root_folder = io.folder_list[1]
        size = sort.get_cv2size(site_specific_root_folder, img_list[0])
        # print(img_list)



        # img_array, size = sort.simplest_order(segment) 

    # print("img_array: ",img_array)
    ### WRITE THE IMAGES TO VIDEO/FILES ###

    if VIDEO == True:
        #save individual as video
        sort.write_video(ROOT, img_list, df_segment, size)

    else:
        #save individual as images
        
        sort.write_images(ROOT, img_list)


if __name__ == '__main__':
    main()

