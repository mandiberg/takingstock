import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd

import os
import math
import time
import sys

XLOW = -20
XHIGH = 10
YLOW = -2
YHIGH = 2
ZLOW = -2
ZHIGH = 2
MINCROP = 1
MAXRESIZE = .5
FRAMERATE = 15
SORT = 'mouth_gap'

#creating my objects

start = time.time()

#regular 31.8s
#concurrent 

#declariing path and image before function, but will reassign in the main loop
ROOT="/Users/michaelmandiberg/Documents/projects-active/facemap_production/"

# folder ="sourceimages"
# FOLDER ="/Users/michaelmandiberg/Dropbox/Photo Scraping/facemesh/facemeshes_commons/"
MAPDATA_FILE = "allmaps_65814.csv"
# size = (750, 750) #placeholder 


# file = "auto-service-workerowner-picture-id931914734.jpg"
# path = "sourceimages/auto-service-workerowner-picture-id931914734.jpg"
# image = cv2.imread(os.path.join(root,folder, file))  # read any image containing a face
# dfallmaps = pd.DataFrame(columns=['name', 'cropX', 'x', 'y', 'z', 'resize', 'newname', 'mouth_gap']) 

# def touch(folder):
#     if not os.path.exists(folder):
#         os.makedirs(folder)


FOLDER = os.path.join(ROOT,"5GB_testimages_output")

outputfolderRGB = os.path.join(ROOT,"face_mesh_outputsRGB")
outputfolderBW = os.path.join(ROOT,"face_mesh_outputsBW")
outputfolderMEH = os.path.join(ROOT,"face_mesh_outputsMEH")

try:
    df = pd.read_csv(os.path.join(ROOT,MAPDATA_FILE))
except:
    print('you forgot to change the filename DUH')

if df.empty:
    print('dataframe empty, probably bad path')
    sys.exit()

print(df)

# display(df)

segment = df.loc[((df['y'] < YHIGH) & (df['y'] > YLOW))]
segment = segment.loc[((segment['x'] < XHIGH) & (segment['x'] > XLOW))]
segment = segment.loc[((segment['z'] < ZHIGH) & (segment['z'] > ZLOW))]
# segment = segment.loc[((segment['z'] > Zneg))]
# segment = segment.loc[((segment['z'] < Zpos))]
# segment = segment.loc[segment['color'] >= True]
segment = segment.loc[segment['cropX'] >= MINCROP]
segment = segment.loc[segment['resize'] < MAXRESIZE]

print(segment)
rotation = segment.sort_values(by=SORT)

print(rotation)

img_array = []
videofile = f"facevid_crop{str(MINCROP)}_X{str(XLOW)}toX{str(XHIGH)}_Y{str(YLOW)}toY{str(YHIGH)}_Z{str(ZLOW)}toZ{str(ZHIGH)}_maxResize{str(MAXRESIZE)}_ct{str(len(rotation))}_rate{(str(FRAMERATE))}.mp4"

def simple_order(rotation):
    for index, row in rotation.iterrows():
        print(row['x'], row['y'], row['newname'])
        print(row['newname'])
    # filenames = glob.glob('image-*.png')
    # filenames.sort()
    # for filename in filenames:
    #     print(filename)
        try:
            img = cv2.imread(row['newname'])
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        except:
            print('failed:',row['newname'])
    return img_array

img_array = simple_order(rotation)

# self._name = name + '.mp4'
# self._cap = VideoCapture(0)
# self._fourcc = VideoWriter_fourcc(*'MP4V')
# self._out = VideoWriter(self._name, self._fourcc, 20.0, (640,480))
try:
    out = cv2.VideoWriter(os.path.join(ROOT,videofile), cv2.VideoWriter_fourcc(*'mp4v'), FRAMERATE, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('wrote:',videofile)
except:
    print('failed, probably because segmented df until empty')