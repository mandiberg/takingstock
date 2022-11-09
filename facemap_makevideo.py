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

VIDEO = False
side_to_side = False
forward_smile = False
forward_nosmile = False
static_pose = True
CYCLECOUNT = 4

if side_to_side == True:
    XLOW = -20
    XHIGH = 1
    YLOW = -30
    YHIGH = 30
    ZLOW = -1
    ZHIGH = 1
    MINCROP = 1
    MAXRESIZE = .5
    MAXMOUTHGAP = 4
    FRAMERATE = 15
    SORT = 'y'
    SECOND_SORT = 'x'
    # SORT = 'mouth_gap'
    ROUND = 0
elif forward_smile == True:
    XLOW = -20
    XHIGH = 1
    YLOW = -4
    YHIGH = 4
    ZLOW = -3
    ZHIGH = 3
    MINCROP = 1
    MAXRESIZE = .5
    FRAMERATE = 15
    SECOND_SORT = 'x'
    SORT = 'mouth_gap'
    ROUND = 1
elif forward_nosmile == True:
    XLOW = -20
    XHIGH = 1
    YLOW = -4
    YHIGH = 4
    ZLOW = -3
    ZHIGH = 3
    MINCROP = 1
    MAXRESIZE = .5
    FRAMERATE = 15
    SECOND_SORT = 'x'
    MAXMOUTHGAP = 2
    SORT = 'mouth_gap'
    ROUND = 1
elif static_pose == True:
    XLOW = -20
    XHIGH = 1
    YLOW = -4
    YHIGH = 4
    ZLOW = -3
    ZHIGH = 3
    MINCROP = 1
    MAXRESIZE = .5
    FRAMERATE = 15
    SECOND_SORT = 'mouth_gap'
    MAXMOUTHGAP = 10
    SORT = 'x'
    ROUND = 1


divisor = eval(f"1e{ROUND}")


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

def createList(r1, r2, ROUND):

    # divides angles based off of ROUND    
    divisor = eval(f"1e{ROUND}")

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
            res.append(round(r1,ROUND))
            r1 += 1/divisor
            print(r1 )
        return res


try:
    df = pd.read_csv(os.path.join(ROOT,MAPDATA_FILE))
except:
    print('you forgot to change the filename DUH')

if df.empty:
    print('dataframe empty, probably bad path')
    sys.exit()

# print(df)

# display(df)

segment = df.loc[((df['y'] < YHIGH) & (df['y'] > YLOW))]
print(segment.size)
segment = segment.loc[((segment['x'] < XHIGH) & (segment['x'] > XLOW))]
print(segment.size)
segment = segment.loc[((segment['z'] < ZHIGH) & (segment['z'] > ZLOW))]
print(segment.size)
segment = segment.loc[segment['cropX'] >= MINCROP]
print(segment.size)
segment = segment.loc[segment['mouth_gap'] >= MAXMOUTHGAP]
# segment = segment.loc[segment['mouth_gap'] <= MAXMOUTHGAP]
print(segment.size)
# segment = segment.loc[segment['resize'] < MAXRESIZE]

#simple ordering
rotation = segment.sort_values(by=SORT)
# print(rotation)

# #complex ordering
# angle = startAngle
# counter = 0

startAngle = segment[SORT].min()
endAngle = segment[SORT].max()
print("startAngle, endAngle")
print(startAngle, endAngle)
angle_list = createList(startAngle, endAngle, ROUND)


d = {}
for angle in angle_list:
    # print(angle)
    d[angle] = segment.loc[((segment[SORT] > angle) & (segment[SORT] < angle+(1/divisor)))]
    # print(d[angle].size)

# print('manual test of -30')
# print(d[-30].size)




# while (angle < endAngle):
#     segment+str(counter) = segment.loc[((segment['y'] < segment) & (segment['y'] > segment+1))]
#     segment = segment+1

# print(segment0.size)



angle_list_pop = angle_list.pop()

img_array = []
videofile = f"facevid_crop{str(MINCROP)}_X{str(XLOW)}toX{str(XHIGH)}_Y{str(YLOW)}toY{str(YHIGH)}_Z{str(ZLOW)}toZ{str(ZHIGH)}_maxResize{str(MAXRESIZE)}_ct{str(len(rotation))}_rate{(str(FRAMERATE))}.mp4"
imgfileprefix = f"faceimg_crop{str(MINCROP)}_X{str(XLOW)}toX{str(XHIGH)}_Y{str(YLOW)}toY{str(YHIGH)}_Z{str(ZLOW)}toZ{str(ZHIGH)}_maxResize{str(MAXRESIZE)}_ct{str(len(rotation))}"

median = d[0][SECOND_SORT].median()
print("starting from this median: ",median)

def get_metamedian(angle_list):
    medians = []
    for angle in angle_list:
        print(angle)
        print (d[angle].size)
        try:
            print(d[angle].iloc[1]['newname'])
            this_median = d[angle]['x'].median()
            medians.append(this_median)
        except:
            print("empty set, moving on")
    print("all medians: ",medians)
    print("median of all medians: ",statistics.median(medians))
    metamedian = statistics.mean(medians)
    print("mean of all medians: ",metamedian)
    return metamedian

metamedian = get_metamedian(angle_list)

# #old structure
# for index, row in rotation.iterrows():
#     print(row['x'], row['y'], row['newname'])
#     print(row['newname'])


# filenames = glob.glob('image-*.png')
# filenames.sort()
# for filename in filenames:
#     print(filename)

# angle_list = angle_list[:-1]
# angle_list_pop = del angle_list[-1]
# print(record)

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


cycle = 0 
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

# self._name = name + '.mp4'
# self._cap = VideoCapture(0)
# self._fourcc = VideoWriter_fourcc(*'MP4V')
# self._out = VideoWriter(self._name, self._fourcc, 20.0, (640,480))


if VIDEO == True:
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
    if not os.path.exists(outfolder):      
        os.mkdir(outfolder)

    try:
        counter = 1
        # out = cv2.VideoWriter(os.path.join(ROOT,videofile), cv2.VideoWriter_fourcc(*'mp4v'), FRAMERATE, size)
        for i in range(len(img_array)):
            print('in loop')
            imgfilename = imgfileprefix+"_"+str(counter)+".jpg"
            print(imgfilename)
            outpath = os.path.join(outfolder,imgfilename)
            cv2.imwrite(outpath, img_array[i])
            print(outpath)
            # out.write(img_array[i])
            counter += 1
        # out.release()
        # print('wrote:',videofile)
    except:
        print('failed IMAGES, probably because segmented df until empty')
