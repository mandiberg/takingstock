import cv2
import os
import sys

import tkinter as tk
from tkinter import filedialog
import numpy as np

#mine
from mp_sort_pose import SortPose

FRAMERATE = 15

def get_img_list(folder):
    img_list=[]
    for file in os.listdir(folder):
        if not file.startswith('.') and os.path.isfile(os.path.join(folder, file)):
            filepath = os.path.join(folder, file)
            filepath=filepath.replace('\\' , '/')
            img_list.append(file)
    return img_list        
    print("got image list")

def get_cv2size(ROOT, filename):
    img = cv2.imread(os.path.join(ROOT,filename))
    size = (img.shape[0], img.shape[1])
    return size


def write_video(img_array,ROOT, FRAMERATE):
    foldername = os.path.basename(os.path.normpath(ROOT))
    videofile = f"facevid_{str(foldername)}_{str(FRAMERATE)}FPS.mp4"
    # img = cv2.imread(os.path.join(ROOT,img_array[0]))
    # size = img.shape
    size = get_cv2size(ROOT,img_array[0])
    savefile = os.path.join(ROOT,videofile)
    print("foldername ",foldername)
    print("videofile ",videofile)
    try:
        fc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(videofile, fc, FRAMERATE, size)

        # out = cv2.VideoWriter(os.path.join(ROOT,videofile), cv2.VideoWriter_fourcc(*'mp4v'), FRAMERATE, size)
        for i in range(len(img_array)):
            print(img_array[i])
            img = cv2.imread(os.path.join(ROOT,img_array[i]))
            out.write(img)
        out.release()
        print('wrote:',videofile)
    except:
        print('failed VIDEO')

def selectDir():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askdirectory()
    return file_path


ROOT = selectDir()

if not ROOT:
    print('No Folder Selected', 'Please select a valid Folder')
else :
    list_of_files= get_img_list(ROOT)
    print(list_of_files)
    list_of_files.sort()
    print(list_of_files)
    
    write_video(list_of_files, ROOT, FRAMERATE)
