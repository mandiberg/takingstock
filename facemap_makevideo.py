import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd

import os
import math
import time
import sys
import statistics

XLOW = -20
XHIGH = 1
YLOW = -30
YHIGH = 30
ZLOW = -3
ZHIGH = -2
MINCROP = 1
MAXRESIZE = .5
FRAMERATE = 15
SORT = 'y'
SECOND_SORT = 'x'
# SORT = 'mouth_gap'
startAngle=YLOW
endAngle=YHIGH

#creating my objects

start = time.time()

#regular 31.8s
#concurrent 

#declariing path and image before function, but will reassign in the main loop
ROOT="/Users/michaelmandiberg/Documents/projects-active/facemap_production/"

# folder ="sourceimages"
# FOLDER ="/Users/michaelmandiberg/Dropbox/Photo Scraping/facemesh/facemeshes_commons/"
MAPDATA_FILE = "allmaps_62607.csv"
# size = (750, 750) #placeholder 


# file = "auto-service-workerowner-picture-id931914734.jpg"
# path = "sourceimages/auto-service-workerowner-picture-id931914734.jpg"
# image = cv2.imread(os.path.join(root,folder, file))  # read any image containing a face
# dfallmaps = pd.DataFrame(columns=['name', 'cropX', 'x', 'y', 'z', 'resize', 'newname', 'mouth_gap']) 

# def touch(folder):
#     if not os.path.exists(folder):
#         os.makedirs(folder)


#Do These matter?
FOLDER = os.path.join(ROOT,"5GB_testimages_output")
outputfolderRGB = os.path.join(ROOT,"face_mesh_outputsRGB")
outputfolderBW = os.path.join(ROOT,"face_mesh_outputsBW")
outputfolderMEH = os.path.join(ROOT,"face_mesh_outputsMEH")


# Python3 Program to Create list
# with integers within given range

def createList(r1, r2):
    # Testing if range r1 and r2
    # are equal
    if (r1 == r2):
        return r1
    else:
        # Create empty list
        res = []
        # loop to append successors to
        # list until r2 is reached.
        while(r1 < r2+1 ):
            res.append(r1)
            r1 += 1
        return res


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
print(segment.size)
segment = segment.loc[((segment['x'] < XHIGH) & (segment['x'] > XLOW))]
print(segment.size)
segment = segment.loc[((segment['z'] < ZHIGH) & (segment['z'] > ZLOW))]
print(segment.size)
# segment = segment.loc[((segment['z'] > Zneg))]
# segment = segment.loc[((segment['z'] < Zpos))]
# segment = segment.loc[segment['color'] >= True]
segment = segment.loc[segment['cropX'] >= MINCROP]
print(segment.size)
# segment = segment.loc[segment['resize'] < MAXRESIZE]
print(segment.size)

print(segment)


#simple ordering
rotation = segment.sort_values(by=SORT)
print(rotation)

# #complex ordering
# angle = startAngle
# counter = 0

# Driver Code
r1, r2 = startAngle, endAngle
print(createList(r1, r2))

angle_list = createList(r1, r2)


d = {}
for angle in angle_list:
    d[angle] = segment.loc[((segment['y'] > angle) & (segment['y'] < angle+1))]

print('manual test of -30')
print(d[-30].size)




# while (angle < endAngle):
#     segment+str(counter) = segment.loc[((segment['y'] < segment) & (segment['y'] > segment+1))]
#     segment = segment+1

# print(segment0.size)



angle_list_pop = angle_list.pop()

img_array = []
videofile = f"facevid_crop{str(MINCROP)}_X{str(XLOW)}toX{str(XHIGH)}_Y{str(YLOW)}toY{str(YHIGH)}_Z{str(ZLOW)}toZ{str(ZHIGH)}_maxResize{str(MAXRESIZE)}_ct{str(len(rotation))}_rate{(str(FRAMERATE))}.mp4"

median = d[0][SECOND_SORT].median()
print("starting from this median: ",median)

medians = []
for angle in angle_list:
    print(angle)
    print (d[angle].size)
    # print(d[angle].iloc[1]['newname'])
    this_median = d[angle]['x'].median()
    medians.append(this_median)

print("all medians: ",medians)
print("median of all medians: ",statistics.median(medians))
metamedian = statistics.mean(medians)
print("mean of all medians: ",metamedian)

# #old structure
# for index, row in rotation.iterrows():
#     print(row['x'], row['y'], row['newname'])
#     print(row['newname'])


# filenames = glob.glob('image-*.png')
# filenames.sort()
# for filename in filenames:
#     print(filename)

for angle in angle_list:
    print('closest: ')
    # print(d[angle].iloc[(d[angle][SECOND_SORT]-metamedian).abs().argsort()[:2]])
    try:
        closest = d[angle].iloc[(d[angle][SECOND_SORT]-metamedian).abs().argsort()[:1]]
        # cprint(closest)
        closest_file = closest.iloc[0]['newname']
        first = d[angle].iloc[1]['newname']
        print(first)
        img = cv2.imread(closest_file)
        # img = cv2.imread(closest)
        #old version
        # img = cv2.imread(row['newname'])
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    except:
        print('failed:')
        # print('failed:',row['newname'])

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
