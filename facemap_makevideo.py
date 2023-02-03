import cv2
import pandas as pd
import os
import time
import sys

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

#mine
from mp_sort_pose import SortPose

VIDEO = False
CYCLECOUNT = 2
ROOT="/Users/michaelmandiberg/Documents/projects-active/facemap_production/"
MAPDATA_FILE = "allmaps_64910.csv"

motion = {
    "side_to_side": False,
    "forward_smile": False,
    "forward_nosmile":  True,
    "static_pose":  False,
    "simple": False,
}

#creating my objects
start = time.time()
sort = SortPose(motion)

# read the csv and construct dataframe
try:
    df = pd.read_csv(os.path.join(ROOT,MAPDATA_FILE))
except:
    print('you forgot to change the filename DUH')
if df.empty:
    print('dataframe empty, probably bad path')
    sys.exit()

### PROCESS THE DATA ###

# make the segment based on settings
segment = sort.make_segment(df)

# get list of all angles in segment
angle_list = sort.createList(segment)

# sort segment by angle list
# d is a dataframe organized (indexed?) by angle list
d = sort.get_d(segment)

# is this used anywhere? 
angle_list_pop = angle_list.pop()

# get median for first sort
median = sort.get_median()

# get metamedian for second sort, creates sort.metamedian attribute
sort.get_metamedian()

### BUILD THE LIST OF SELECTED IMAGES ###

if motion["side_to_side"] is True:

    img_array, size = sort.cycling_order(CYCLECOUNT)
else:
# dont neet to pass SECOND_SORT, because it is already there
    img_array, size = sort.simple_order(segment) 

    # img_array, size = sort.simplest_order(segment) 


### WRITE THE IMAGES TO VIDEO/FILES ###

if VIDEO == True:
    #save individual as video
    sort.write_video(ROOT, img_array, segment, size)

else:
    #save individual as images
    outfolder = os.path.join(ROOT,"images"+str(time.time()))
    sort.write_images(outfolder, img_array, segment, size)