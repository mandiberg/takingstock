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

#mine
from mp_sort_pose import SortPose

VIDEO = False
CYCLECOUNT = 1

motion = {
    "side_to_side": False,
    "forward_smile": False,
    "forward_nosmile":  False,
    "static_pose":  False,
    "simple": True,
}

#declariing path and image before function, but will reassign in the main loop
ROOT="/Users/michaelmandiberg/Documents/projects-active/facemap_production/"

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

# print(rotation)

# #complex ordering
# angle = startAngle
# counter = 0

startAngle = segment[SORT].min()
endAngle = segment[SORT].max()
print("startAngle, endAngle")
print(startAngle, endAngle)
angle_list = sort.createList(startAngle, endAngle)



# moving to class
d = sort.get_d(segment)

# divisor = eval(f"1e{ROUND}")
# dd = {}
# for angle in angle_list:
#     print(angle)
#     dd[angle] = segment.loc[((segment[SORT] > angle) & (segment[SORT] < angle+(1/divisor)))]
#     # print(d[angle].size)

# # print("original d")
# # print (dd)

# print("classy d")
# print(d)

# # print('manual test of -30')
# # print(d[-30].size)





# while (angle < endAngle):
#     segment+str(counter) = segment.loc[((segment['y'] < segment) & (segment['y'] > segment+1))]
#     segment = segment+1

# print(segment0.size)



angle_list_pop = angle_list.pop()

videofile = f"facevid_crop{str(MINCROP)}_X{str(XLOW)}toX{str(XHIGH)}_Y{str(YLOW)}toY{str(YHIGH)}_Z{str(ZLOW)}toZ{str(ZHIGH)}_maxResize{str(MAXRESIZE)}_ct{str(len(segment))}_rate{(str(FRAMERATE))}.mp4"
imgfileprefix = f"faceimg_crop{str(MINCROP)}_X{str(XLOW)}toX{str(XHIGH)}_Y{str(YLOW)}toY{str(YHIGH)}_Z{str(ZLOW)}toZ{str(ZHIGH)}_maxResize{str(MAXRESIZE)}_ct{str(len(segment))}"

# print(d[1])


angle_list_median = round(statistics.median(angle_list))
print('angle_list_median: ',angle_list_median)


print('angle_list_median][SECOND_SORT',d[angle_list_median])

if not d[angle_list_median][SECOND_SORT].empty:
    median = d[angle_list_median][SECOND_SORT].median()
else:
    newmedian = angle_list_median+1
    while newmedian < max(angle_list):
        if d[newmedian][SECOND_SORT].empty:
            newmedian += 1
        else:
            median = d[newmedian][SECOND_SORT].median()
            print('good newmedian is: ',newmedian)
            print('good new median is: ', median)
            print(d[newmedian][SECOND_SORT].size)
            break



print("starting from this median: ",median)

### moved to class
def get_metamedian(angle_list):
    print('anglelist: ',angle_list)
    medians = []
    for angle in angle_list:
        print('angle: ',angle)
        print ('d angle size: ',d[angle].size)
        try:
            print("not empty set!")
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



def simple_order(segment, this_sort):
    img_array = []
    delta_array = []
    #simple ordering
    rotation = segment.sort_values(by=this_sort)

    for index, row in rotation.iterrows():
        # print(row['x'], row['y'], row['newname'])
        delta_array.append(row['mouth_gap'])
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
    print("delta_array")
    print(delta_array)
    return img_array


def cycling_order(angle_list, CYCLECOUNT, SECOND_SORT):
    img_array = []
    cycle = 0 
    # metamedian = get_metamedian(angle_list)
    metamedian = sort.get_metamedian()

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
        return img_array

# self._name = name + '.mp4'
# self._cap = VideoCapture(0)
# self._fourcc = VideoWriter_fourcc(*'MP4V')
# self._out = VideoWriter(self._name, self._fourcc, 20.0, (640,480))

img_array = cycling_order(angle_list, CYCLECOUNT, SECOND_SORT)
# img_array = simple_order(segment, SECOND_SORT)




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
            # this code takes image i, and blends it with the subsequent image
            # next step is to test to see if mp can recognize a face in the image
            # if no face, a bad blend, try again with i+2, etc. 
            # except it would need to do that with the sub-array, so move above? 
            blend = cv2.addWeighted(img_array[i], 0.5, img_array[(i+1)], 0.5, 0.0)
            cv2.imwrite(outpath, blend)
            print(outpath)
            # out.write(img_array[i])
            counter += 1
        # out.release()
        # print('wrote:',videofile)
    except:
        print('failed IMAGES, probably because segmented df until empty')
