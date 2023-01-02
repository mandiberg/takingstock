import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd

import os
import math
import time
import sys

#mine
from mp_sort_pose import SortPose

VIDEO = True
CYCLECOUNT = 1
ROOT="/Users/michaelmandiberg/Documents/projects-active/facemap_production/"

motion = {
    "side_to_side": False,
    "forward_smile": False,
    "forward_nosmile":  False,
    "static_pose":  False,
    "simple": True,
}

#declariing path and image before function, but will reassign in the main loop

# folder ="sourceimages"
# FOLDER ="/Users/michaelmandiberg/Dropbox/Photo Scraping/facemesh/facemeshes_commons/"
MAPDATA_FILE = "allmaps_64910.csv"
# size = (750, 750) #placeholder 

#Do These matter?
FOLDER = os.path.join(ROOT,"5GB_testimages_output")
outputfolderRGB = os.path.join(ROOT,"face_mesh_outputsRGB")
outputfolderBW = os.path.join(ROOT,"face_mesh_outputsBW")
outputfolderMEH = os.path.join(ROOT,"face_mesh_outputsMEH")

#creating my objects
start = time.time()
sort = SortPose(motion)

XLOW = sort.XLOW
XHIGH = sort.XHIGH
YLOW = sort.YLOW
YHIGH = sort.YHIGH
ZLOW = sort.ZLOW
ZHIGH = sort.ZHIGH
MINCROP = sort.MINCROP
MAXRESIZE = sort.MAXRESIZE
MAXMOUTHGAP = sort.MAXMOUTHGAP
FRAMERATE = sort.FRAMERATE
SORT = sort.SORT
SECOND_SORT = sort.SECOND_SORT
ROUND = sort.ROUND

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

# borks when I put it in class
def cycling_order(angle_list, CYCLECOUNT, SECOND_SORT):
    img_array = []
    cycle = 0 
    # metamedian = get_metamedian(angle_list)
    metamedian = sort.metamedian

    while cycle < CYCLECOUNT:
        print("CYCLE: ",cycle)
        for angle in angle_list:
            print("angle: ",str(angle))
            print(d[angle].iloc[(d[angle][SECOND_SORT]-metamedian).abs().argsort()[:2]])
            print(d[angle].size)
            try:
                closest = d[angle].iloc[(d[angle][SECOND_SORT]-metamedian).abs().argsort()[:CYCLECOUNT]]
                closest_file = closest.iloc[cycle]['newname']
                closest_mouth = closest.iloc[cycle]['mouth_gap']
                # print('closest: ')
                print(closest_mouth)
                img = cv2.imread(closest_file)
                height, width, layers = img.shape
                size = (width, height)
                img_array.append(img)
            except:
                print('failed:')
                # print('failed:',row['newname'])
        # print('finished a cycle')
        angle_list.reverse()
        cycle = cycle +1
        # print(angle_list)
        return img_array, size

img_array, size = cycling_order(angle_list, CYCLECOUNT, SECOND_SORT)
# img_array = simple_order(segment, SECOND_SORT)


### WRITE THE IMAGES TO VIDEO/FILES ###

videofile = f"facevid_crop{str(MINCROP)}_X{str(XLOW)}toX{str(XHIGH)}_Y{str(YLOW)}toY{str(YHIGH)}_Z{str(ZLOW)}toZ{str(ZHIGH)}_maxResize{str(MAXRESIZE)}_ct{str(len(segment))}_rate{(str(FRAMERATE))}.mp4"
imgfileprefix = f"faceimg_crop{str(MINCROP)}_X{str(XLOW)}toX{str(XHIGH)}_Y{str(YLOW)}toY{str(YHIGH)}_Z{str(ZLOW)}toZ{str(ZHIGH)}_maxResize{str(MAXRESIZE)}_ct{str(len(segment))}"

if VIDEO == True:
    #realizing that I didn't test to see if this was wrorking beforehand... 
    # sort.write_video(ROOT, img_array, segment, size)
    try:
        out = cv2.VideoWriter(os.path.join(ROOT,videofile), cv2.VideoWriter_fourcc(*'mp4v'), FRAMERATE, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        print('wrote:',videofile)
    except:
        print('failed VIDEO, probably because segmented df until empty')

else:
    #save individual as images
    outfolder = os.path.join(ROOT,"images"+str(time.time()))
    sort.write_images(outfolder, img_array, segment, size)