import statistics
import os
import cv2
import pandas as pd
import mediapipe as mp
import hashlib
import time
import json
import random
import numpy as np
import sys
from collections import Counter
from simple_lama_inpainting import SimpleLama
from sklearn.neighbors import NearestNeighbors
import ast
import re


class SortPose:
    # """Sort image files based on head pose"""

    def __init__(self, motion, face_height_output, image_edge_multiplier, EXPAND, ONE_SHOT, JUMP_SHOT, HSV_CONTROL=None, VERBOSE=True,INPAINT=False):

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.get_bg_segment=mp.solutions.selfie_segmentation.SelfieSegmentation()  
              

        #maximum allowable distance between encodings
        self.MAXDIST = 1.4
        self.MINDIST = .45
        self.MINBODYDIST = .05
        self.CUTOFF = 100
        self.FACE_DIST = 15

        self.INPAINT=INPAINT
        if self.INPAINT:self.INPAINT_MODEL=SimpleLama()
        # self.MAX_IMAGE_EDGE_MULTIPLIER=[1.5,2.6,2,2.6] #maximum of the elements
        self.MAX_IMAGE_EDGE_MULTIPLIER = image_edge_multiplier #testing

        self.knn = NearestNeighbors(metric='euclidean', algorithm='ball_tree')

        # if edge_multiplier_name:self.edge_multiplier_name=edge_multiplier_name
        # maximum allowable scale up
        self.resize_max = 5.99
        self.image_edge_multiplier = image_edge_multiplier
        self.face_height_output = face_height_output
        # takes base image size and multiplies by avg of multiplier
        self.output_dims = (int(face_height_output*(image_edge_multiplier[1]+image_edge_multiplier[3])/2),int(face_height_output*(image_edge_multiplier[0]+image_edge_multiplier[2])/2))
        self.EXPAND = EXPAND
        self.EXPAND_SIZE = (2000,2000)
        # self.EXPAND_SIZE = (4000,3000)
        self.BGCOLOR = [255,255,255]
        # self.BGCOLOR = [0,0,0]
        self.ONE_SHOT = ONE_SHOT
        self.JUMP_SHOT = JUMP_SHOT
        self.SHOT_CLOCK = 0
        self.SHOT_CLOCK_MAX = 10
        self.BODY_LMS = [0, 13, 14, 15, 16, 19, 20]
        self.VERBOSE = VERBOSE

        # place to save bad images
        self.not_make_face = []
        self.same_img = []
        
        # luminosity parameters
        self.HSV_DELTA_MAX = .5
        if HSV_CONTROL:
            self.LUM_MIN = HSV_CONTROL['LUM_MIN']
            self.LUM_MAX = HSV_CONTROL['LUM_MAX']
            self.SAT_MIN = HSV_CONTROL['SAT_MIN']
            self.SAT_MAX = HSV_CONTROL['SAT_MAX']
            self.HUE_MIN = HSV_CONTROL['HUE_MIN']
            self.HUE_MAX = HSV_CONTROL['HUE_MAX']
            self.HSV_WEIGHT = HSV_CONTROL['HSV_WEIGHT']
            self.d128_WEIGHT = HSV_CONTROL['d128_WEIGHT']
            self.LUM_WEIGHT = HSV_CONTROL['LUM_WEIGHT']
        # set some defaults, looking forward
        self.XLOW = -20
        self.XHIGH = 1
        self.YLOW = -4
        self.YHIGH = 4
        self.ZLOW = -3
        self.ZHIGH = 3
        self.MINCROP = 1
        self.MAXRESIZE = .5
        self.MINMOUTHGAP = 0
        self.MAXMOUTHGAP = 4
        self.FRAMERATE = 15
        self.SORT = 'face_y'
        self.SECOND_SORT = 'face_x'
        self.ROUND = 0

        if motion['side_to_side'] == True:
            self.XLOW = -20
            self.XHIGH = 1
            self.YLOW = -30
            self.YHIGH = 30
            self.ZLOW = -1
            self.ZHIGH = 1
            self.MINCROP = 1
            self.MAXRESIZE = .5
            self.MAXMOUTHGAP = 4
            self.FRAMERATE = 15
            self.SORT = 'face_y'
            self.SECOND_SORT = 'face_x'
            # self.SORT = 'mouth_gap'
            self.ROUND = 0
        elif motion['forward_smile'] == True:
            # self.XLOW = -33
            # self.XHIGH = -27
            self.XLOW = -33
            self.XHIGH = -27
            self.YLOW = -2
            self.YHIGH = 2
            self.ZLOW = -2
            self.ZHIGH = 2
            self.MINCROP = 1
            self.MAXRESIZE = .5
            self.FRAMERATE = 15
            self.SECOND_SORT = 'face_x'
            self.MINMOUTHGAP = 0
            self.MAXMOUTHGAP = 40
            self.SORT = 'mouth_gap'
            self.ROUND = 1
        elif motion['laugh'] == True:
            self.XLOW = 5
            self.XHIGH = 40
            self.YLOW = -4
            self.YHIGH = 4
            self.ZLOW = -3
            self.ZHIGH = 3
            self.MINCROP = 1
            self.MAXRESIZE = .5
            self.FRAMERATE = 15
            self.SECOND_SORT = 'face_x'
            self.MAXMOUTHGAP = 20
            self.SORT = 'mouth_gap'
            self.ROUND = 1
        elif motion['forward_nosmile'] == True:
            self.XLOW = -15
            self.XHIGH = 5
            self.YLOW = -4
            self.YHIGH = 4
            self.ZLOW = -3
            self.ZHIGH = 3
            self.MINCROP = 1
            self.MAXRESIZE = .3
            self.FRAMERATE = 15
            self.SECOND_SORT = 'face_y'
            self.MAXMOUTHGAP = 2
            self.SORT = 'face_x'
            self.ROUND = 1
        elif motion['static_pose'] == True:
            self.XLOW = -20
            self.XHIGH = 1
            self.YLOW = -4
            self.YHIGH = 4
            self.ZLOW = -3
            self.ZHIGH = 3
            self.MINCROP = 1
            self.MAXRESIZE = .5
            self.FRAMERATE = 15
            self.SECOND_SORT = 'mouth_gap'
            self.MAXMOUTHGAP = 10
            self.SORT = 'face_x'
            self.ROUND = 1
        elif motion['simple'] == True:
            self.XLOW = -20
            self.XHIGH = 1
            self.YLOW = -4
            self.YHIGH = 4
            self.ZLOW = -3
            self.ZHIGH = 3
            self.MINCROP = 1
            self.MAXRESIZE = .5
            self.FRAMERATE = 15
            self.SECOND_SORT = 'mouth_gap'
            self.MAXMOUTHGAP = 10
            self.SORT = 'face_x'
            self.ROUND = 1

    def set_counters(self,ROOT,cluster_no,start_img_name,start_site_image_id):
        self.negmargin_count = 0
        self.toosmall_count = 0 
        self.outfolder = os.path.join(ROOT,"cluster"+str(cluster_no)+"_"+str(time.time()))
        if not os.path.exists(self.outfolder):      
            os.mkdir(self.outfolder)
        self.counter_dict = {
            "counter": 1,
            "good_count": 0,
            "isnot_face_count": 0,
            "cropfail_count":  0,
            "failed_dist_count": 0,
            "inpaint_count":0,
            "outfolder":  self.outfolder,
            "first_run":  True,
            "start_img_name":start_img_name,
            "start_site_image_id":start_site_image_id,
            "last_image":None,
            "last_description":None,
            "last_image_enc":None,
            "last_image_hsv":None,
            "last_image_lum":None

        }


    def make_segment(self, df):

        segment = df.loc[((df['face_y'] < self.YHIGH) & (df['face_y'] > self.YLOW))]
        print(segment.size)
        segment = segment.loc[((segment['face_x'] < self.XHIGH) & (segment['face_x'] > self.XLOW))]
        print(segment.size)
        segment = segment.loc[((segment['face_z'] < self.ZHIGH) & (segment['face_z'] > self.ZLOW))]
        print(segment.size)

        if self.LUM_MIN:
            segment = segment.loc[((segment['lum'] < self.LUM_MAX) & (segment['lum'] > self.LUM_MIN))]
        if self.SAT_MIN:
            segment = segment.loc[((segment['sat'] < self.SAT_MAX) & (segment['sat'] > self.SAT_MIN))]
        if self.HUE_MIN:
            segment = segment.loc[((segment['hue'] < self.HUE_MAX) & (segment['hue'] > self.HUE_MIN))]

        
        # removing cropX for now. Need to add that back into the data
        # segment = segment.loc[segment['cropX'] >= self.MINCROP]
        # print(segment.size)

        # COMMENTING OUT MOUTHGAP as it is functioning as a minimum. Needs refactoring
        segment = segment.loc[segment['mouth_gap'] >= self.MINMOUTHGAP]
        segment = segment.loc[segment['mouth_gap'] <= self.MAXMOUTHGAP]
        # segment = segment.loc[segment['mouth_gap'] <= MAXMOUTHGAP]
        print(segment.size)
        # segment = segment.loc[segment['resize'] < MAXRESIZE]
        print(segment)
        return segment


    def createList(self,segment):

        r1 = segment[self.SORT].min()
        r2 = segment[self.SORT].max()
        print("startAngle, endAngle")
        print(r1, r2)

        # divides angles based off of ROUND    
        divisor = eval(f"1e{self.ROUND}")

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
                res.append(round(r1,self.ROUND))
                r1 += 1/divisor
                # print(r1 )
            self.angle_list = res
            return res

    def get_divisor(self, segment):

        divisor = eval(f"1e{self.ROUND}")
        self.d = {}
        for angle in self.angle_list:
            # print(angle)
            self.d[angle] = segment.loc[((segment[self.SORT] > angle) & (segment[self.SORT] < angle+(1/divisor)))]
            # print(self.d[angle].size)
        

    def get_median(self):

        median = None
        print(self.angle_list)
        angle_list_median = round(statistics.median(self.angle_list))
        print('angle_list_median: ',angle_list_median)

        print(self.d)
        # this is an empty set
        print('angle_list_median[SECOND_SORT]',self.d[angle_list_median])

        if not self.d[angle_list_median][self.SECOND_SORT].empty:
            median = self.d[angle_list_median][self.SECOND_SORT].median()
        else:
            newmedian = angle_list_median+1
            while newmedian < max(self.angle_list):
                if self.d[newmedian][self.SECOND_SORT].empty:
                    newmedian += 1
                else:
                    median = self.d[newmedian][self.SECOND_SORT].median()
                    print('good newmedian is: ',newmedian)
                    print('good new median is: ', median)
                    print(self.d[newmedian][self.SECOND_SORT].size)
                    break
        if not median:
            print("median is none --------- this is a problem")
            median = None

        return median
        print("starting from this median: ",median)

    def get_metamedian(self):
        medians = []
        print('anglelist: ',self.angle_list)

        for angle in self.angle_list:
            # print('angle: ',angle)
            # print ('d angle size: ',self.d[angle].size)
            try:
                print("not empty set!")
                print(self.d[angle].iloc[1]['imagename'])
                this_median = self.d[angle]['face_x'].median()
                medians.append(this_median)
            except:
                print("empty set, moving on")
        print("all medians: ",medians)
        print("median of all medians: ",statistics.median(medians))
        self.metamedian = statistics.mean(medians)
        print("mean of all medians: ",self.metamedian)
        return self.metamedian

    def is_face(self, image):
        # For static images:
        IMAGE_FILES = []
        with self.mp_face_detection.FaceDetection(model_selection=1, 
                                            min_detection_confidence=0.75
                                            ) as face_detection:
            # image = cv2.imread(file)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Draw face detections of each face.
            if not results.detections:
                is_face = False
            else:
                is_face = True
        print("returning is_face ", is_face)
        return is_face

    # def blend_is_face(self, oldimage, newimage):
    #     blend = cv2.addWeighted(oldimage, 0.5, newimage, 0.5, 0.0)
    #     # blend = cv2.addWeighted(img, 0.5, img_array[i-1], 0.5, 0.0)
    #     blended_face = sort.is_face(blend)
    #     return blended_face

    # def get_hash_folders(self,filename):
    #     m = hashlib.md5()
    #     m.update(filename.encode('utf-8'))
    #     d = m.hexdigest()
    #     # csvWriter1.writerow(["https://upload.wikimedia.org/wikipedia/commons/"+d[0]+'/'+d[0:2]+'/'+filename])
    #     return d[0], d[0:2]

    #get distance beetween encodings
    def get_d(self, enc1, enc2):
        enc1=np.array(enc1)
        # print("enc1")
        # print(enc1)
        # # this is currently an np.array, not 128d list
        # print(enc1[0])
        enc2=np.array(enc2)
        # print("enc2")
        # print(enc2)
        d=np.linalg.norm(enc1 - enc2, axis=0)
        return d


    def simplest_order(self, segment):
        img_array = []
        delta_array = []
        #simple ordering, second sort, because this is the...?
        rotation = segment.sort_values(by=self.SORT)

        i = 0
        for index, row in rotation.iterrows():
            print(row['face_x'], row['face_y'], row['imagename'])

            #I don't know what this does or why
            delta_array.append(row['mouth_gap'])

            try:
                img = cv2.imread(row['imagename'])
                height, width, layers = img.shape
                size = (width, height)
                # test to see if this is actually an face, to get rid of blank ones/bad ones
                # this may not be necessary
                img_array.append(img)

                i+=1

            except:
                print('failed:',row['imagename'])
        # print("delta_array")
        # print(delta_array)
        return img_array, size        


    def get_cv2size(self, site_specific_root_folder, filename_or_imagedata):
        #IF filename_or_imagedata IS STRING:
        img = cv2.imread(os.path.join(site_specific_root_folder,filename_or_imagedata))
        #ELIF filename_or_imagedata IS NDARRAY:
        #img = filename_or_imagedata
        size = (img.shape[0], img.shape[1])
        return size

    # this doesn't seem to work
    # but it might be relevant later
    def cv2_safeopen_size(self, ROOT, filename_or_imagedata):
        print('attempting safeopen')
        if (len(filename_or_imagedata.split('/'))>1):
            # has path
            print('about to try')
            try:
                img = cv2.imread(filename_or_imagedata)
            except:
                print('could not read image path ',filename_or_imagedata)

        elif (len(filename_or_imagedata.split('.'))==1):
            # is filename
            print('about to try')
            try:
                img = cv2.imread(os.path.join(ROOT,filename_or_imagedata))
            except:
                print('could not read image ',filename_or_imagedata)

        else:
            # is imagedata
            img = filename_or_imagedata
            print('was image data')

        size = (img.shape[0], img.shape[1])
        return img, size

    # # test if new and old make a face
    # def is_face(self,image):
    #     # For static images:
    #     # I think this list is not used
    #     IMAGE_FILES = []
    #     with mp_face_detection.FaceDetection(model_selection=1, 
    #                                         min_detection_confidence=0.6
    #                                         ) as face_detection:
    #         results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #         # Draw face detections of each face.
    #         if not results.detections:
    #             is_face = False
    #         else:
    #             is_face = True
    #         return is_face



    # test if new and old make a face, calls is_face
    def test_pair(self, last_file, new_file):
        print(img.shape,"@@@@@@@@@@@@@@@@@@@@")
        print(last_img.shape,"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        try:
            img = new_file
            height, width, layers = img.shape
            size = (width, height)
            # print('loaded img 1')

            last_img = last_file
            last_height, last_width, last_layers = last_img.shape
            last_size = (last_width, last_height)
            # print('loaded img 2')

            # Check if dimensions match
            if size != last_size:
                print('Image dimensions do not match. Skipping blending.')
                return False

            # code for face detection and blending
            if self.is_face(img):
                # print('new file is face')
                blend = cv2.addWeighted(img, 0.5, last_img, 0.5, 0.0)
                # foopath = os.path.join("/Users/michaelmandiberg/Documents/projects-active/facemap_production/blends", "foobar_"+str(random.random())+".jpg")
                # cv2.imwrite(foopath, blend)
                # print('blended faces')
                blended_face = self.is_face(blend)
                # print('blended is_face', blended_face)
                if blended_face:
                    if self.VERBOSE: print('test_pair: is_face True! adding it')
                    return True
                else:
                    print('test_pair: skipping this one')
                    return False
            else:
                print('test_pair: new_file is not a face:')
                return False

        except Exception as e:
            print('failed:', new_file)
            print('Error:', str(e))
            return False

    def preview_img(self,img):
        cv2.imshow("difference", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def unique_face(self,img1,img2):
        # convert the images to grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # define the function to compute MSE between two images
        def mse(img1, img2):
            h, w = img1.shape
            try:
                diff = cv2.subtract(img1, img2)
                err = np.sum(diff**2)
                mse = err/(float(h*w))
            except Exception as e:
                print('failed mse')
                print('Error:', str(e))

            return mse, diff

        error, diff = mse(img1, img2)
        
        # # i don't know what number to use
        # if error == 0:
        #     print(f"unique_face: {error} Fail, images identical")
        #     return False
        # elif error < 15:
        #     print(f"unique_face: {error} Fail, images less than 15 diff")
        #     # preview_img(diff)
        #     # preview_img(img1)
        #     # preview_img(img2)
        #     return False
        # elif error < 25:
        #     print(f"unique_face: {error} Fail, images 15-25 diff")
        #     preview_img(diff)
        #     preview_img(img1)
        #     preview_img(img2)
        #     return False
        # else:
        #     return True

        return error



    def point(self,coords):
        newpoint = (int(coords[0]), int(coords[1]))
        return newpoint

    def dist(self,p, q):
        """ 
        Return euclidean distance between points p and q
        assuming both to have the same number of dimensions
        """
        # sum of squared difference between coordinates
        s_sq_difference = 0
        for p_i,q_i in zip(p,q):
            s_sq_difference += (p_i - q_i)**2
        
        # take sq root of sum of squared difference
        distance = s_sq_difference**0.5
        return distance    

    def get_face_2d_point(self, point):
        # print("get_face_2d_point")

        # print(self.bbox)
        # print(type(self.bbox))
        # print(self.bbox['left'])
        # set bbox dimensions
        img_h = self.h
        img_w = self.w
        bbox_x = self.bbox['left']
        bbox_y = self.bbox['top']
        bbox_w = self.bbox['right'] - self.bbox['left']
        bbox_h = self.bbox['bottom'] - self.bbox['top']
        for idx, lm in enumerate(self.faceLms.landmark):
            if idx == point:
                # print("found point:")
                # print(idx)
                # pointXY = (lm.x * img_w, lm.y * img_h)
                pointXY = (lm.x * bbox_w + bbox_x, lm.y * bbox_h + bbox_y)
                # print(pointXY)
                # pointXYonly = (lm.x, lm.y)
                # print(pointXYonly)
        return pointXY


    def get_faceheight_data(self):
        # print("get_faceheight_data")
        top_2d = self.get_face_2d_point(10)
        # print(top_2d)
        bottom_2d = self.get_face_2d_point(152)
        # print(bottom_2d)
        self.ptop = (int(top_2d[0]), int(top_2d[1]))
        self.pbot = (int(bottom_2d[0]), int(bottom_2d[1]))
        # height = int(pbot[1]-ptop[1])
        # print(self.ptop)
        # print(self.pbot)
        self.face_height = self.dist(self.point(self.pbot), self.point(self.ptop))
        # print("face_height", str(self.face_height))
        # return ptop, pbot, face_height



    def get_crop_data_scalable(self):

        # p1 is tip of nose
        p1 = (int(self.nose_2d[0]), int(self.nose_2d[1]))

        toobig = False  # Default value

        print("checkig boundaries")
        print(self.w, self.h)
        print(p1)
        print(self.face_height)

        topcrop = int(p1[1]-self.face_height*self.image_edge_multiplier[0])
        rightcrop = int(p1[0]+self.face_height*self.image_edge_multiplier[1])
        botcrop = int(p1[1]+self.face_height*self.image_edge_multiplier[2])
        leftcrop = int(p1[0]-self.face_height*self.image_edge_multiplier[3])
        self.simple_crop = [topcrop, rightcrop, botcrop, leftcrop]
        print("crop top, right, bot, left")
        print(self.simple_crop)
        if topcrop >= 0 and self.w-rightcrop >= 0 and self.h-botcrop>= 0 and leftcrop>= 0:
            print("all positive")
            toobig = False
        else:
            print("one is negative")
            toobig = True
            self.negmargin_count += 1
        return toobig

    def get_image_face_data(self,image, faceLms, bbox):
        self.image = image
        self.size = (self.image.shape[0], self.image.shape[1])
        self.h = self.image.shape[0]
        self.w = self.image.shape[1]
        self.faceLms = faceLms
        self.bbox = (bbox)

        print("get_image_face_data [-] size is", self.size)
        #I'm not sure the diff between nose_2d and p1. May be redundant.
        #it would prob be better to do this with a dict and a loop
        # Instead of hard-coding the index 1, you can use a variable or constant for the point index
        nose_point_index = 1
        self.nose_2d = self.get_face_2d_point(nose_point_index)

        try:
            # get self.face_height
            self.get_faceheight_data()
        except:
            print("couldn't get_faceheight_data")

            # this is the in progress neck rotation stuff
            # self.get_crop_data(sinY)

    def expand_image(self,image, faceLms, bbox, sinY=0):
        self.get_image_face_data(image, faceLms, bbox)    
        try:
            # print(type(self.image))
            borderType = cv2.BORDER_CONSTANT

            # scale image to match face heights
            resize = self.face_height_output/self.face_height
            if resize < 15:
                print("expand_image [-] resize", str(resize))
                # image.shape is height[0] and width[1]
                resize_dims = (int(self.image.shape[1]*resize),int(self.image.shape[0]*resize))
                # resize_nose.shape is  width[0] and height[1]
                resize_nose = (int(self.nose_2d[0]*resize),int(self.nose_2d[1]*resize))
                # print("resize_dims")
                # print(resize_dims)
                # print("resize_nose")
                # print(resize_nose)
                # this wants width and height
                resized_image = cv2.resize(self.image, resize_dims, interpolation=cv2.INTER_LINEAR)
                # self.preview_img(resized_image)

                # calculate boder size by comparing scaled image dimensions to EXPAND_SIZE
                # nose as center
                # set top, bottom, left, right
                top_border = int(self.EXPAND_SIZE[1]/2 - resize_nose[1])
                bottom_border = int(self.EXPAND_SIZE[1]/2 - (resize_dims[1]-resize_nose[1]))
                left_border = int(self.EXPAND_SIZE[0]/2 - resize_nose[0])
                right_border = int(self.EXPAND_SIZE[0]/2 - (resize_dims[0]-resize_nose[0]))

                # print([top_border, bottom_border, left_border, right_border])
                # print([top_border, resize_dims[0]/2-right_border, resize_dims[1]/2-bottom_border, left_border])
                # print([top_border, self.EXPAND_SIZE[0]/2-right_border, self.EXPAND_SIZE[1]/2-bottom_border, left_border])

                # expand image with borders
                if top_border >= 0 and right_border >= 0 and self.EXPAND_SIZE[0]/2-right_border >= 0 and bottom_border >= 0 and self.EXPAND_SIZE [1]/2-bottom_border>= 0 and left_border>= 0:
                # if topcrop >= 0 and self.w-rightcrop >= 0 and self.h-botcrop>= 0 and leftcrop>= 0:
                    print("crop is good")
                    new_image = cv2.copyMakeBorder(resized_image, top_border, bottom_border, left_border, right_border, borderType, None, self.BGCOLOR)
                else:
                    print("crop failed")
                    new_image = None
                    self.negmargin_count += 1
                # self.preview_img(new_image)
                # quit()
            else:
                new_image = None
                print("failed expand loop")


        except Exception as e:
            print("not expand_image loop failed")
            print('Error:', str(e))
            sys.exit(1)
        # quit() 
        return new_image  

    def get_extension_pixels(self,image):
        if self.VERBOSE: print("calculating extension pixels")
        p1 = (int(self.nose_2d[0]), int(self.nose_2d[1]))
        topcrop = int(p1[1]-self.face_height*self.MAX_IMAGE_EDGE_MULTIPLIER[0])
        rightcrop = int(p1[0]+self.face_height*self.MAX_IMAGE_EDGE_MULTIPLIER[1])
        botcrop = int(p1[1]+self.face_height*self.MAX_IMAGE_EDGE_MULTIPLIER[2])
        leftcrop = int(p1[0]-self.face_height*self.MAX_IMAGE_EDGE_MULTIPLIER[3])
        ext_crop = np.array([topcrop, self.w-rightcrop, self.h-botcrop, leftcrop])
        ext_crop=np.abs(np.minimum(np.zeros(4),ext_crop)).astype(int)
        # simple_crop=np.array([np.maximum(0,topcrop),np.minimum(self.w,rightcrop),np.minimum(self.h,botcrop),np.maximum(0,leftcrop)]).astype(int)
        # simple_crop={"top":simple_crop[0],"right":simple_crop[1],"bottom":simple_crop[2],"left":simple_crop[3]}
        extension_pixels={"top":ext_crop[0],"right":ext_crop[1],"bottom":ext_crop[2],"left":ext_crop[3]}
        if self.VERBOSE:
            print("extension pixels",extension_pixels)
        #     print("simple crop",simple_crop)
        # simple_crop_image=image[simple_crop["top"]:simple_crop["bottom"],simple_crop["left"]:simple_crop["right"],:]
        if self.VERBOSE: print("extension pixels calculated")
        return extension_pixels

    def prepare_mask(self,image,extension_pixels,blur_radius=100):
        if self.VERBOSE:print("starting mask preparation")
        height, width = image.shape[:2]
        top, bottom, left, right = extension_pixels["top"], extension_pixels["bottom"], extension_pixels["left"],extension_pixels["right"] 
        extended_img = np.zeros((height + top+bottom, width+left+right, 3), dtype=np.uint8)
        extended_img[top:height+top, left:width+left,:] = image
        mask = np.zeros_like(extended_img[:, :, 0])
        mask[:top,:] = 255
        mask[:,:left] = 255
        mask[(height+top):,:] = 255
        mask[:,(width+left):] = 255

        # mask blur
        if blur_radius % 2 == 0:blur_radius += 1
        mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
        if self.VERBOSE:print("mask preparation done")
        return extended_img,mask
    
    def extend_lama(self,extended_img, mask,downsampling_scale=1):
        if self.VERBOSE: print("doing lama generative fill")
        n_height,n_width=extended_img.shape[:2]
        extended_img = cv2.resize(extended_img, (n_width//downsampling_scale, n_height//downsampling_scale), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask, (n_width//downsampling_scale, n_height//downsampling_scale), interpolation = cv2.INTER_AREA)
        inpaint = self.INPAINT_MODEL(extended_img, mask)
        inpaint=np.array(inpaint,dtype=np.uint8)
        inpaint = cv2.resize(inpaint, (n_width,n_height), interpolation = cv2.INTER_LANCZOS4)
        if self.VERBOSE: print("generative fill done")

        return inpaint

    # def inpaint(self,image):
    #     extension_pixels=self.get_extension_pixels(image)
    #     extended_img,mask=self.prepare_mask(image,extension_pixels)
    #     inpaint_image=self.extend_lama(extended_img, mask,scale=4)
    #     return inpaint_image

    def crop_image(self,image, faceLms, bbox, sinY=0,SAVE=False):

        self.get_image_face_data(image, faceLms, bbox)    
        # check for crop, and if not exist, then get
        # if not hasattr(self, 'crop'): 
        try:
            toobig = self.get_crop_data_scalable()
        except:
            print("couldn't get crop data")
            toobig = True


        if not toobig:
            if self.VERBOSE: print("crop_image: going to crop because too big is ", toobig)
            # print (self.padding_points)
            #set main points for drawing/cropping

            #moved this back up so it would NOT     draw map on both sets of images
            try:
                if self.VERBOSE: print(type(self.image))
                # image_arr = numpy.array(self.image)
                # print(type(image_arr))
                cropped_actualsize_image = self.image[self.simple_crop[0]:self.simple_crop[2], self.simple_crop[3]:self.simple_crop[1]]
                if self.VERBOSE: print("cropped_actualsize_image.shape", cropped_actualsize_image.shape)
                resize = self.output_dims[0]/cropped_actualsize_image.shape[0] 
                if self.VERBOSE: print("resize", resize)
                if resize > self.resize_max:
                    if self.VERBOSE: print("toosmall")
                    self.toosmall_count += 1
                    return None
                if self.VERBOSE: print("about to resize")
                # crop[0] is top, and clockwise from there. Right is 1, Bottom is 2, Left is 3. 
                if self.VERBOSE: print("output dims", self.output_dims)
                cropped_image = cv2.resize(cropped_actualsize_image, (self.output_dims), interpolation=cv2.INTER_LINEAR)
                if self.VERBOSE: print("image actually cropped")
            except:
                cropped_image = None
                print("not cropped_image loop", self.h, self.w)
        else:
            cropped_image = np.array([-1])
            print("crop_image: cropped_image is None because too big is ", toobig)
            # resize = None
        return cropped_image

    def get_bg_hue_lum(self,image,bbox=None,faceLms=None):
        # expects image in RGB format
        print("in get_bg_hue_lum")
        hue = sat = val = lum = lum_torso = None
        if bbox:
            try:
                if type(bbox)==str:
                    bbox=json.loads(bbox)
                    if self.VERBOSE: print("bbox type", type(bbox))
                #sample_img=sample_img[bbox['top']:bbox['bottom'],bbox['left']:bbox['right'],:]
                # passing in bbox as a str
                image = self.crop_image(image, faceLms, bbox)
                if image is None: return -1,-1,-1,-1,-1 ## if TOO_BIG==true, checking if cropped image is empty
            except:
                if self.VERBOSE: print("FAILED CROPPING, bad bbox",bbox)
                return -2,-2,-2,-2,-2
            print("bbox['bottom'], ", bbox['bottom'])
        print("[get_bg_hue_lum] about to go for segemntation")
        result = self.get_bg_segment.process(image[:,:,::-1]) #convert RBG to BGR then process with mp
        print("[get_bg_hue_lum] got result")
        mask=np.repeat((1-result.segmentation_mask)[:, :, np.newaxis], 3, axis=2) 
        print("[get_bg_hue_lum] made mask")
        mask_torso=np.repeat((result.segmentation_mask)[:, :, np.newaxis], 3, axis=2) 
        print("[get_bg_hue_lum] made torso mask")

        print("[get_bg_hue_lum] doing some stuff")
        masked_img=mask*image[:,:,::-1]/255 ##RGB format
        masked_img_torso=mask_torso*image[:,:,::-1]/255 ##RGB format

        print("[get_bg_hue_lum] about to make bk px mask")
        # Identify black pixels where R=0, G=0, B=0
        black_pixels_mask = np.all(masked_img == [0, 0, 0], axis=-1)
        black_pixels_mask_torso = np.all(masked_img_torso == [0, 0, 0], axis=-1)

        # Filter out black pixels and compute the mean color of the remaining pixels
        mean_color = np.mean(masked_img[~black_pixels_mask], axis=0)[np.newaxis,np.newaxis,:] # ~ means negate/remove
        self.hue = cv2.cvtColor(mean_color, cv2.COLOR_RGB2HSV)[0,0,0]
        self.sat = cv2.cvtColor(mean_color, cv2.COLOR_RGB2HSV)[0,0,1]
        self.val = cv2.cvtColor(mean_color, cv2.COLOR_RGB2HSV)[0,0,2]
        self.lum = cv2.cvtColor(mean_color, cv2.COLOR_RGB2LAB)[0,0,0]
        print("hue, sat, val, lum", self.hue, self.sat, self.val, self.lum)
        if self.VERBOSE: print("NOTmasked_img_torso size", masked_img_torso.shape, black_pixels_mask_torso.shape)
        if bbox:
            # SJ something is broken in here. It returns an all black image which produces a lum of 100
            masked_img_torso = masked_img_torso[bbox['bottom']:]
            black_pixels_mask_torso = black_pixels_mask_torso[bbox['bottom']:]
        # else:
        #     print("YIKES! no bbox. Here's a hacky hack to crop to the bottom 20%")
        #     bottom_fraction = masked_img_torso.shape[0] // 5
        #     masked_img_torso = masked_img_torso[-bottom_fraction:]
        #     black_pixels_mask_torso = black_pixels_mask_torso[-bottom_fraction:]

        if self.VERBOSE: print("masked_img_torso size", masked_img_torso.shape, black_pixels_mask_torso.shape)
        mean_color = np.mean(masked_img_torso[~black_pixels_mask_torso], axis=0)[np.newaxis,np.newaxis,:] # ~ is negate
        self.lum_torso=cv2.cvtColor(mean_color, cv2.COLOR_RGB2LAB)[0,0,0]

        if self.VERBOSE: print("HSV, lum", hue,sat,val,lum, lum_torso)
        return self.hue,self.sat,self.val,self.lum,self.lum_torso
    
    def get_start_enc(self, start_img, df_128_enc, df_33_lms, SORT_TYPE):
        print("get_start_enc")
        '''
        if start_img == "median" and not self.CLNO:
            get median

        elif start_img == "median" and self.CLNO:
            get cluster median

        '''
        if start_img == "median":
            # dfmode = df_128_enc.round(2).mode(dropna=True)

            # Step 1: Round all values to 2 decimal points
            df_rounded = df_128_enc.round(3)

            # Step 2: Flatten the DataFrame into a single array
            flattened_array = df_rounded.values

            # Step 3: Convert the flattened array into tuples for hashing
            hashable_rows = [tuple(row) for row in flattened_array]

            # Step 4: Find the mode using the most_common function from the collections module
            counter = Counter(hashable_rows)
            most_common_row = counter.most_common(1)[0][0]

            print("Most common face embedding:")
            print(most_common_row)
            enc1 = most_common_row
            # print(dfmode)
            # enc1 = dfmode.iloc[0].to_list()
            # enc1 = df_128_enc.median().to_list()
            print("in median")

        elif start_img == "start_site_image_id":
            print("start_site_image_id (this is what we are comparing to)")
            # print(start_site_image_id)
            print(self.counter_dict["start_site_image_id"])
            enc1 = df_128_enc.loc[self.counter_dict["start_site_image_id"]].to_list()
        elif SORT_TYPE == "planar_body":
            # print("get_start_enc planar_body start_img key is (this is what we are comparing to):")
            # print(start_img)
            try:
                # enc1 = df_33_lms.loc[start_img].to_list()
                # TK 
                enc1 = self.get_landmarks_2d_dict(df_33_lms.loc[start_img, "body_landmarks"], self.BODY_LMS)

                # print("get_start_enc planar_body", enc1)
            except:
                print("Returning enc1 = median << KeyError for ", start_img)
                # enc1 = None
                enc1 = df_33_lms.median().to_list()
                print(enc1)

        else:
            # enc1 = get 2-129 from df via string key
            # print("start_img key is (this is what we are comparing to):")
            # print(start_img)
            try:
                print("buggggggggy start_img", start_img)
                print(df_128_enc)
                print(df_128_enc.loc[start_img])                
                enc1 = df_128_enc.loc[start_img].to_list()
                # print(enc1)
            except:
                print("Returning enc1 = median << KeyError for ", start_img)
                # enc1 = None
                enc1 = df_128_enc.median().to_list()
                print(enc1)

            try:
                df_128_enc=df_128_enc.drop(start_img)
                print("dropped ",start_img)
            except:
                print("couldn't drop the start_img")
        return enc1, df_128_enc, df_33_lms


    def get_face_2d_dict(self, faceLms):
        face_2d = {}
        for idx, lm in enumerate(faceLms.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199 or idx == 10 or idx == 152:
                # x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d[idx] =([lm.x, lm.y])

                # Get the 2D Coordinates
                # face_2d.append([x, y])

        # Convert it to the NumPy array
        # image points
        # self.face_2d = np.array(face_2d, dtype=np.float64)

        return face_2d

    def get_landmarks_2d_dict(self, Lms, selected_Lms):
        Lms2d = {}
        for idx, lm in enumerate(Lms.landmark):
            if idx in selected_Lms:
                # x, y = int(lm.x * img_w), int(lm.y * img_h)
                Lms2d[idx] =([lm.x, lm.y])

                # Get the 2D Coordinates
                # body_2d.append([x, y])

        # Convert it to the NumPy array
        # image points
        # self.body_2d = np.array(body_2d, dtype=np.float64)

        return Lms2d

    def get_planar_d(self,last_dict,this_dict):
        d_list = []
        for point in last_dict:
            # d=np.linalg.norm(enc1 - enc2, axis=0)
            # print(last_dict[point])
            # print(type(last_dict[point]))
            # this_d = np.linalg.norm(last_dict[point],this_dict[point])

            enc1=np.array(last_dict[point])
            enc2=np.array(this_dict[point])
            # print("enc2.shape")
            # print(enc2.shape)

            d=np.linalg.norm(enc1 - enc2)

            d_list.append(d)
        # print(d_list)
        d = statistics.mean(d_list)
        # print(last_dict[1][0])
        # print(this_dict[1][0])
        # print(d)
        return d

    def normalize_hsv(self, hsv1, hsv2):
        # hue is between 0-360 degrees (which is scaled to 0-1 here)
        # hue is a circle, so we need to go both ways around the circle and pick the smaller one
        dist_reg = abs(hsv1[0] - hsv2[0])
        dist_offset = abs((hsv1[0]) + (1-hsv2[0]))
        hsv2[0] = min(dist_reg, dist_offset)
        # this is where I will curve the sat data
        return hsv2

    def weight_hue(self, hsv):        
        def min_max_scale(h):
            if h <= 0.3:
                return 0
            else:
                return (h - 0.3) / (1 - 0.3)

        h, s, v = hsv
        s = min_max_scale(s)
        v = min_max_scale(v)
        return h, s, v

        # cube root, not using for now    
        # if h>0:
        #     w_s = s**(1./3.)
        # else:
        #     w_s = -((-s)**(1./3.))


    # def sort_dHSV(self, dist_dict, df_enc, HSVonly=False):
    #     # print("sort_dHSV, HSVonly is", HSVonly)
    #     hsv_dist_dict = {}
    #     for item in dist_dict:
    #         # print("item", item)
    #         # for testing
    #         # if not self.counter_dict["last_image_hsv"]:
    #         #     self.counter_dict["last_image_hsv"] = [.1,.1,.85]
    #         if not self.counter_dict["last_image_hsv"]:
    #             # assign 128d to all vars for first run, ignoring hsv and lum, and just sort on 128d
    #             # if self.VERBOSE: print("assigned first run hsv_dist values", item)
    #             hsv_dist = item
    #             lum_dist = item
    #         elif abs(self.counter_dict["last_image_hsv"][2] - df_enc.loc[dist_dict[item], "hsv"][2]) > self.HSV_DELTA_MAX:
    #             # skipping this one, too great a value shift, will produce flicker
    #             # if self.VERBOSE: print("skipping this one, too great a color shift:", dist_dict[item])
    #             continue
    #         elif abs(self.counter_dict["last_image_lum"][1] - df_enc.loc[dist_dict[item], "lum"][1]) > self.HSV_DELTA_MAX:
    #             # if self.VERBOSE: print("skipping this one, too great a value shift:", dist_dict[item])
    #             # skipping this one, too great a value shift, will produce flicker
    #             continue
    #         else:
    #             # print("in circle else")
    #             hsv_converted = self.normalize_hsv(self.counter_dict["last_image_hsv"], df_enc.loc[dist_dict[item], "hsv"])
    #             hsv_dist = self.get_d(self.counter_dict["last_image_hsv"], hsv_converted)
    #             lum_dist = self.get_d(self.counter_dict["last_image_lum"], df_enc.loc[dist_dict[item], "lum"])
    #         # if self.VERBOSE: print("hsv, lum dist, item", hsv_dist, lum_dist, item)
    #         if HSVonly:
    #             # print("HSVonly, sorting results")
    #             # finds closest hsv distance bg and torso
    #             hsv_dist_dict[item] = [hsv_dist*self.HSV_WEIGHT,lum_dist*self.LUM_WEIGHT]

    #             # sort the hsv_dist_dict dictionary based on the sum of both values in the list
    #             sorted_keys_dHSV = [k for k, v in sorted(hsv_dist_dict.items(), key=lambda item: item[1][0] + item[1][1])]

    #         else:
    #             # finds closest 3D distanct for 128d, hsv, and lum
    #             hsv_dist_dict[item] = [item*self.d128_WEIGHT,hsv_dist*self.HSV_WEIGHT,lum_dist*self.LUM_WEIGHT]

    #             # sort the hsv_dist_dict dictionary based on the sum of all three values in the list
    #             sorted_keys_dHSV = [k for k, v in sorted(hsv_dist_dict.items(), key=lambda item: item[1][0] + item[1][1] + item[1][2])]

    #     # if self.VERBOSE: print("sorted_keys_dHSV", sorted_keys_dHSV)
    #     self.counter_dict["last_image_hsv"] =df_enc.loc[dist_dict[sorted_keys_dHSV[0]], "hsv"]
    #     self.counter_dict["last_image_lum"] =df_enc.loc[dist_dict[sorted_keys_dHSV[0]], "lum"]
    #     # TK this needs to be handled separately for RUNS and for planar

    #     return sorted_keys_dHSV


    # April 20 something version, may contain bugs
    def sort_dHSV(self, dist_dict, df_enc, HSVonly=False):
        if self.VERBOSE: print(f"in sort_dHSV, HSVonly is {HSVonly} counter_dict is {self.counter_dict}")

        # Sort the dictionary by keys
        sorted_dist_dict = dict(sorted(dist_dict.items()))

        # Move items beyond the first 200 into a new dictionary
        extra_dist_dict = {k: v for i, (k, v) in enumerate(sorted_dist_dict.items()) if i >= 200}

        # Remove the items beyond the first 200 from the original dictionary
        dist_dict_200 = {k: v for i, (k, v) in enumerate(sorted_dist_dict.items()) if i < 200}

        print("Sorted Dictionary (first 200 items):")
        print(len(dist_dict_200))

        print("\nExtra Dictionary (beyond first 200 items):")
        print(len(extra_dist_dict))

        hsv_dist_dict = {}
        # if dist_dict is greater than 200


        # only going through dist_dict_200, but leaving dist_dict as the ref below
        for item in dist_dict_200:
            print("item", item)
            # for testing
            # if not self.counter_dict["last_image_hsv"]:
            #     self.counter_dict["last_image_hsv"] = [.1,.1,.85]
            try:
                if not self.counter_dict["last_image_hsv"]:
                    # assign 128d to all vars for first run, ignoring hsv and lum, and just sort on 128d
                    # if self.VERBOSE: print("assigned first run hsv_dist values", item)
                    hsv_dist = item
                    lum_dist = item
                elif abs(self.counter_dict["last_image_hsv"][2] - df_enc.loc[dist_dict[item], "hsv"][2]) > self.HSV_DELTA_MAX:
                    # skipping this one, too great a value shift, will produce flicker
                    # if self.VERBOSE: print("skipping this one, too great a color shift:", dist_dict[item])
                    continue
                elif abs(self.counter_dict["last_image_lum"][1] - df_enc.loc[dist_dict[item], "lum"][1]) > self.HSV_DELTA_MAX:
                    # if self.VERBOSE: print("skipping this one, too great a value shift:", dist_dict[item])
                    # skipping this one, too great a value shift, will produce flicker
                    continue
                else:
                    # if self.VERBOSE: print("in circle else")
                    hsv_converted = self.normalize_hsv(self.counter_dict["last_image_hsv"], df_enc.loc[dist_dict[item], "hsv"])
                    hsv_dist = self.get_d(self.counter_dict["last_image_hsv"], hsv_converted)
                    lum_dist = self.get_d(self.counter_dict["last_image_lum"], df_enc.loc[dist_dict[item], "lum"])
            except:
                print(f"error in sort_dHSV, skipping {dist_dict[item]}")
                continue
            if self.VERBOSE: print("hsv, lum dist, item", hsv_dist, lum_dist, item)
            if HSVonly:
                print("HSVonly, sorting results")
                # finds closest hsv distance bg and torso
                hsv_dist_dict[item] = [hsv_dist*self.HSV_WEIGHT,lum_dist*self.LUM_WEIGHT]

                # sort the hsv_dist_dict dictionary based on the sum of both values in the list
                sorted_keys_dHSV = [k for k, v in sorted(hsv_dist_dict.items(), key=lambda item: item[1][0] + item[1][1])]

            else:
                # finds closest 3D distanct for 128d, hsv, and lum
                hsv_dist_dict[item] = [item*self.d128_WEIGHT,hsv_dist*self.HSV_WEIGHT,lum_dist*self.LUM_WEIGHT]
                # if self.VERBOSE: print("hsv_dist_dict[item]", hsv_dist_dict[item], "item", item)    
                # sort the hsv_dist_dict dictionary based on the sum of all three values in the list
                sorted_keys_dHSV = [k for k, v in sorted(hsv_dist_dict.items(), key=lambda item: item[1][0] + item[1][1] + item[1][2])]
                # print(f"len(sorted_keys_dHSV) {len(sorted_keys_dHSV)} out of {len(dist_dict)}")

        if self.VERBOSE: 
            print("done with dHSV sort loop, going to store keys")
            # print("sorted_keys_dHSV", sorted_keys_dHSV)
        self.counter_dict["last_image_hsv"] =df_enc.loc[dist_dict[sorted_keys_dHSV[0]], "hsv"]
        self.counter_dict["last_image_lum"] =df_enc.loc[dist_dict[sorted_keys_dHSV[0]], "lum"]
        # TK this needs to be handled separately for RUNS and for planar
        sorted_keys_dHSV_to_return = sorted_keys_dHSV+list(extra_dist_dict.keys())
        print("len(sorted_keys_dHSV_to_return)", len(sorted_keys_dHSV_to_return))
        return sorted_keys_dHSV_to_return





    # def jump(self, dist, dist_dict):
    #     if self.VERBOSE: print("jumping!")
    #     print("dist", len(dist), type(dist))
    #     print("dist_dict", len(dist_dict), type(dist_dict)) 
    #     random_d = int(len(dist)*random.random())
    #     print("random_d", random_d)
    #     random_d_in_run = dist[random_d]
    #     print("random_d_in_run", random_d_in_run)
    #     # create a dict with dist_dict[random_d_in_run] as the value for key random_d_in_run
    #     random_imagename = dist_dict[random_d_in_run]
    #     print("random_imagename", random_imagename)

    #     dist_single_dict = {random_d: dist_dict[random_d_in_run]}
    #     print("dist_single_dict", dist_single_dict)
    #     self.SHOT_CLOCK = 0 # reset the shot clock
    #     return random_d_in_run, dist_single_dict

    def get_closest_df_NN(self, enc1, df_enc, df_128_enc, df_33_lms, sorttype="128d"):
        dist=[]
        dist_dict={}
        dist_run_dict={}
        enc2_dict={}
        print("sorttype", sorttype)
        # print("enc1 right before index row itter", enc1)
        print(f"get_closest_df, sorttype is {sorttype} FIRST_ROUND is {FIRST_ROUND}")
        if sorttype == "128d" or (sorttype == "planar" and FIRST_ROUND is True) or (sorttype == "planar_body" and FIRST_ROUND is True):
            
            print(df_128_enc)
            # NN sort df_128_enc for closest to enc1
            # with distance saved somewhere
            # need to be able to cut it off at MINDIST and return that in some form.

            for index, row in df_128_enc.iterrows():
                enc2 = row
                # print("testing this", index, "against the start img",start_img)
                if (enc1 is not None) and (enc2 is not None):
                    # print("getting d with enc1", enc1)
                    # print("getting d with enc2", enc1)
                    d = self.get_d(enc1, enc2)
                    if d > 0:
                        # print ("d is", str(d), "for", index)
                        dist.append(d)
                        dist_dict[d]=index
                        enc2_dict[d]=enc2

                        if sorttype == "128d":
                            if d < self.MINDIST:
                                dist_run_dict[d]=index

                else:
                    print("128d: missing enc1 or enc2")
                    continue
                # FIRST_ROUND = False

        # elif sorttype == "planar":
        #     for index, row in df_128_enc.iterrows():
        #         # print("planar: index")
        #         last_face_2d_dict = self.get_face_2d_dict(df_enc.loc[self.counter_dict["last_image"], "face_landmarks"])
        #         this_face_2d_dict = self.get_face_2d_dict(df_enc.loc[index, "face_landmarks"])
        #         d = self.get_planar_d(last_face_2d_dict,this_face_2d_dict)
        #         print ("planar d is", str(d), "for", index)
        #         dist.append(d)
        #         dist_dict[d]=index


        # elif sorttype == "planar_body":
        #     try:
        #         df_128_enc=df_128_enc.drop(self.counter_dict["last_image"])
        #     except:
        #         print("last_image already dropped")

        #     for index, row in df_128_enc.iterrows():
        #         # print("planar_body [-] getting 2D")
        #         # print(index)
        #         if df_enc.loc[index, "body_landmarks"] is not None:
        #             last_body_2d_dict = self.get_landmarks_2d_dict(df_enc.loc[self.counter_dict["last_image"], "body_landmarks"], self.BODY_LMS)
        #             this_body_2d_dict = self.get_landmarks_2d_dict(df_enc.loc[index, "body_landmarks"], self.BODY_LMS)
        #             # print("planar_body [-] got last_body_2d_dict and this_body_2d_dict")
        #             # print(last_body_2d_dict,this_body_2d_dict)
        #             d = self.get_planar_d(last_body_2d_dict,this_body_2d_dict)
        #             # print ("planar_body d is", str(d), "for", index)
        #             dist.append(d) 
        #             dist_dict[d]=index

        #             # TK this isn't working. needs debugging.
        #             if d < self.MINBODYDIST:
        #                 dist_run_dict[d]=index

        #         else:
        #             try:
        #                 df_128_enc=df_128_enc.drop(index)
        #                 print("dropped: ", index)
        #             except Exception as e:
        #                 print(str(e))


        # print("made it through iterrows")
        if len(dist_run_dict) > 2:
            k = list(dist_run_dict.keys())
            print("WE HAVE A RUN!!!!! ---------- ", str(len(k)))
            # print(dist_run_dict)
            # print(k)
            self.SHOT_CLOCK = 0 # reset the shot clock

            # sort_dHSV returns a list of 128d dists sorted by the sum of 128d and HSVd
            # this code uses that list to sort the dist_run_dict
            # making a new dist_run which is a small subset of dist
            if self.VERBOSE: print("dist_run_dict", dist_run_dict, len(dist_run_dict))
            dist_run = self.sort_dHSV(dist_run_dict, df_enc)
            dist_run_dict_temp = {}
            for d in dist_run:
                dist_run_dict_temp[d] = dist_run_dict[d]
            dist_run_dict = dist_run_dict_temp

            possible_last_d_dict = {}
            for _ in range(100):
                # random_d in second half of ALL results, further away from the start
                half_dist = len(dist)//2
                random_d = half_dist+int(half_dist*random.random())
                random_d_in_run = dist[random_d]
                possible_last_d_dict[random_d_in_run] = dist_dict[random_d_in_run]

            # print("possible_last_d_dict", possible_last_d_dict, len(possible_last_d_dict))
            sorted_last_d_list = self.sort_dHSV(possible_last_d_dict, df_enc, HSVonly=True)
            # print("sorted_last_d_dict", sorted_last_d_dict)
            
            # add on the last d in the run + imagename
            last_d_in_run = sorted_last_d_list[0]
            dist_run_dict[last_d_in_run] = possible_last_d_dict[last_d_in_run]

            dist_single_dict = {random_d: dist_dict[random_d_in_run]}
            self.SHOT_CLOCK = 0 # reset the shot clock

            # last_d_in_run = max(k)
            # print("last_d_in_run", str(last_d_in_run))
            # print(enc2_dict[last_d_in_run])

            # TK this will need refactoring for planar_body
            if sorttype == "planar":
                self.counter_dict["last_image_enc"]=enc2_dict[last_d_in_run]

            # adding the run to the good_count, minus the one added in compare_images
            self.counter_dict["good_count"] += len(dist_run_dict)-1

            # switch to returning dist_run_dict
            if self.VERBOSE: print("dist_run_dict", dist_run_dict, len(dist_run_dict))
            return last_d_in_run, dist_run_dict, df_128_enc, df_33_lms

   
        # '''
        else:
            # print("going to find a winner")
            # dist.sort()
            
            # sort_dHSV returns a list of 128d dists sorted by the sum of 128d and HSVd
            dist = self.sort_dHSV(dist_dict, df_enc)

            print("length of dist -- how many enc are left in the mix")
            # print(dist)
            print(len(dist))
            print(type(len(dist)))
            self.SHOT_CLOCK += 1 # reset the shot clock
            print(f"self.SHOT_CLOCK is {self.SHOT_CLOCK}")
            if dist[0] > self.MAXDIST:
                print("TOO GREAT A DISTANCE ---> going to break the loop and keep the last values")
                dist_single_dict = {1: "null"}
                return 1, dist_single_dict, df_128_enc, df_33_lms
            elif self.JUMP_SHOT and self.SHOT_CLOCK > self.SHOT_CLOCK_MAX:
                print("SHOT_CLOCK IS UP ---> going to pick a random new start")
                # random_d_in_run, dist_single_dict = self.jump(dist, dist_dict)
                random_d = int(len(dist)*random.random())
                random_d_in_run = dist[random_d]

                dist_single_dict = {random_d: dist_dict[random_d_in_run]}
                self.SHOT_CLOCK = 0 # reset the shot clock
                return random_d_in_run, dist_single_dict, df_128_enc, df_33_lms

            elif self.ONE_SHOT and FIRST_ROUND is False:
                # for d in dist:
                #     dist_run_dict[d]
                # k = list(dist_dict.keys())
                last_d_in_run = max(dist)

                print("WE HAVE ONE_SHOT!!!!! ---------- ", str(len(dist)))
                # print(dist_dict)
                if sorttype == "128d":
                    self.counter_dict["last_image_enc"]=enc2_dict[last_d_in_run]
                elif sorttype == "planar":
                    pass

                # adding the run to the good_count, minus the one added in compare_images
                # self.counter_dict["good_count"] += len(dist)-1

                # switch to returning dist_run_dict
                return last_d_in_run, dist_dict, df_128_enc, df_33_lms

            try:
                print ("the winner is: ", str(dist[0]), dist_dict[dist[0]])
            #     print(len(dist))
                self.counter_dict["last_image"]=dist_dict[dist[0]]
                if sorttype == "128d":
                    self.counter_dict["last_image_enc"]=enc2_dict[dist[0]]
                elif sorttype == "planar":
                    print("not setting enc2_dict")
                elif sorttype == "planar_body":
                    print("not setting enc2_dict")
                # make dict of one and to return it
                # dist_single_dict[dist[0]] = dist_dict[dist[0]]
                dist_single_dict = {dist[0]: dist_dict[dist[0]]}
                return dist[0], dist_single_dict, df_128_enc, df_33_lms
            except:
                print("NOTHING HERE")
                dist_single_dict = {1: "null"}
                return 1, dist_single_dict, df_128_enc, df_33_lms

    def fix_128_string(self, encoding_str):
        # Use regex to replace one or more spaces with a single comma
        fixed_str = re.sub(r'\s+', ',', encoding_str)
        # Fix for edge cases
        fixed_str = fixed_str.replace('[,', '[').replace(',]', ']')
        return fixed_str

    # def eval_df_128_enc(self, df_enc):
    #     # Apply the fix_formatting function to the 'face_encodings68' column
    #     df_enc['face_encodings68'] = df_enc['face_encodings68'].apply(self.fix_128_string)
    #     # Convert the 'face_encodings68' from string to list of floats
    #     df_enc['face_encodings68'] = df_enc['face_encodings68'].apply(ast.literal_eval)
    #     return df_enc
    
    def sort_df_KNN(self, df_enc, enc1, sorttype="128d"):
        if sorttype == "128d": sortcol = 'face_encodings68'
        elif sorttype == "planar": sortcol = 'face_landmarks'
        elif sorttype == "planar_body": sortcol = 'body_landmarks'

        start_time = time.time()            
        # Extract the face encodings from the dataframe
        encodings_array = df_enc[sortcol].to_numpy().tolist()

        # Convert to numpy array
        # encodings_array = np.array(encodings)

        print("time to convert to numpy array", time.time() - start_time)
        start_time = time.time()            

        self.knn.fit(encodings_array)
        print("time to fit array", time.time() - start_time)
        start_time = time.time()            

        # Find the distances and indices of the neighbors
        distances, indices = self.knn.kneighbors([enc1], n_neighbors=len(df_enc))

        # Flatten the indices and distances
        indices = indices.flatten()
        distances = distances.flatten()

        # Add distances to the dataframe
        df_enc['distance_to_enc1'] = distances
        print("time to find indices and flatten", time.time() - start_time)
        start_time = time.time()            

        # Sort the dataframe based on the distances
        df_enc = df_enc.sort_values(by='distance_to_enc1')

        # Display the sorted dataframe
        print(df_enc.head())

        return df_enc

    def get_closest_df(self, FIRST_ROUND, enc1, df_enc, df_128_enc, df_33_lms, sorttype="128d"):
        dist=[]
        dist_dict={}
        dist_run_dict={}
        enc2_dict={}
        # print("df_128_enc")
        # print("enc1 right before index row itter", enc1)
        print(f"get_closest_df, sorttype is {sorttype} FIRST_ROUND is {FIRST_ROUND}")
        if sorttype == "128d" or (sorttype == "planar" and FIRST_ROUND is True) or (sorttype == "planar_body" and FIRST_ROUND is True):
            # print(df_enc['face_encodings68'].tolist())
            # df_enc.to_csv("df_enc.csv")
            print("starting regular brute force")
            #start timer
            start_time = time.time()

            for index, row in df_128_enc.iterrows():
                enc2 = row
                # print("testing this round ", index)
                if (enc1 is not None) and (enc2 is not None):
                    # print("getting d with enc1", enc1)
                    # print("getting d with enc2", enc1)
                    d = self.get_d(enc1, enc2)
                    if d > 0:
                        # print ("d is", str(d), "for", index)
                        dist.append(d)
                        dist_dict[d]=index
                        enc2_dict[d]=enc2

                        if sorttype == "128d":
                            if d < self.MINDIST:
                                dist_run_dict[d]=index

                else:
                    print("128d: missing enc1 or enc2")
                    continue
                # FIRST_ROUND = False
            print("--- %s seconds ---" % (time.time() - start_time))
            print("about to do it KNN style")
            start_time = time.time()            
            # df_enc = self.eval_df_128_enc(df_enc)
            # force 128d on first round, (for planar and planar_body)
            df_sorted = self.sort_df_KNN(df_enc, enc1, "128d")
            print(df_sorted)
            print("--- %s seconds ---" % (time.time() - start_time))
            # quit()
        elif sorttype == "planar":
            for index, row in df_128_enc.iterrows():
                # print("planar: index")
                last_face_2d_dict = self.get_face_2d_dict(df_enc.loc[self.counter_dict["last_image"], "face_landmarks"])
                this_face_2d_dict = self.get_face_2d_dict(df_enc.loc[index, "face_landmarks"])
                d = self.get_planar_d(last_face_2d_dict,this_face_2d_dict)
                print ("planar d is", str(d), "for", index)
                dist.append(d)
                dist_dict[d]=index


        elif sorttype == "planar_body":
            try:
                df_128_enc=df_128_enc.drop(self.counter_dict["last_image"])
            except:
                print("last_image already dropped")

            for index, row in df_128_enc.iterrows():
                # print("planar_body [-] getting 2D")
                # print(index)
                if df_enc.loc[index, "body_landmarks"] is not None:
                    last_body_2d_dict = self.get_landmarks_2d_dict(df_enc.loc[self.counter_dict["last_image"], "body_landmarks"], self.BODY_LMS)
                    this_body_2d_dict = self.get_landmarks_2d_dict(df_enc.loc[index, "body_landmarks"], self.BODY_LMS)
                    # print("planar_body [-] got last_body_2d_dict and this_body_2d_dict")
                    # print(last_body_2d_dict,this_body_2d_dict)
                    d = self.get_planar_d(last_body_2d_dict,this_body_2d_dict)
                    # print ("planar_body d is", str(d), "for", index)
                    dist.append(d) 
                    dist_dict[d]=index

                    # TK this isn't working. needs debugging.
                    if d < self.MINBODYDIST:
                        dist_run_dict[d]=index

                else:
                    try:
                        df_128_enc=df_128_enc.drop(index)
                        print("dropped: ", index)
                    except Exception as e:
                        print(str(e))


        # print("made it through iterrows")
        if len(dist_run_dict) > 2:
            k = list(dist_run_dict.keys())
            print("WE HAVE A RUN!!!!! ---------- ", str(len(k)))
            # print(dist_run_dict)
            # print(k)
            self.SHOT_CLOCK = 0 # reset the shot clock

            # sort_dHSV returns a list of 128d dists sorted by the sum of 128d and HSVd
            # this code uses that list to sort the dist_run_dict
            # making a new dist_run which is a small subset of dist
            if self.VERBOSE: print("dist_run_dict", dist_run_dict, len(dist_run_dict))
            dist_run = self.sort_dHSV(dist_run_dict, df_enc)
            dist_run_dict_temp = {}
            for d in dist_run:
                dist_run_dict_temp[d] = dist_run_dict[d]
            dist_run_dict = dist_run_dict_temp

            possible_last_d_dict = {}
            for _ in range(100):
                # random_d in second half of ALL results, further away from the start
                half_dist = len(dist)//2
                random_d = half_dist+int(half_dist*random.random())
                random_d_in_run = dist[random_d]
                possible_last_d_dict[random_d_in_run] = dist_dict[random_d_in_run]

            # print("possible_last_d_dict", possible_last_d_dict, len(possible_last_d_dict))
            sorted_last_d_list = self.sort_dHSV(possible_last_d_dict, df_enc, HSVonly=True)
            # print("sorted_last_d_dict", sorted_last_d_dict)
            
            # add on the last d in the run + imagename
            last_d_in_run = sorted_last_d_list[0]
            dist_run_dict[last_d_in_run] = possible_last_d_dict[last_d_in_run]

            dist_single_dict = {random_d: dist_dict[random_d_in_run]}
            self.SHOT_CLOCK = 0 # reset the shot clock

            # last_d_in_run = max(k)
            # print("last_d_in_run", str(last_d_in_run))
            # print(enc2_dict[last_d_in_run])

            # TK this will need refactoring for planar_body
            if sorttype == "planar":
                self.counter_dict["last_image_enc"]=enc2_dict[last_d_in_run]

            # adding the run to the good_count, minus the one added in compare_images
            self.counter_dict["good_count"] += len(dist_run_dict)-1

            # switch to returning dist_run_dict
            if self.VERBOSE: print("dist_run_dict", dist_run_dict, len(dist_run_dict))
            return last_d_in_run, dist_run_dict, df_128_enc, df_33_lms

        # '''
        #     # print(dist_run_dict)
        #     # print(k)
        #     self.SHOT_CLOCK = 0 # reset the shot clock

        #     # sort_dHSV returns a list of 128d dists sorted by the sum of 128d and HSVd
        #     # this code uses that list to sort the dist_run_dict
        #     dist = self.sort_dHSV(dist_run_dict, df_enc)
        #     dist_run_dict_temp = {}
        #     for d in dist:
        #         dist_run_dict_temp[d] = dist_run_dict[d]
        #     dist_run_dict = dist_run_dict_temp

        #     # moving k here, as this is after items > HSV_DELTA_MAX are dropped 
        #     k = list(dist_run_dict.keys())
        #     print("WE HAVE A RUN!!!!! ---------- ", str(len(k)))
        #     last_d_in_run = max(k)

        #     # print("last_d_in_run", str(last_d_in_run))
        #     # print(enc2_dict[last_d_in_run])
        #     if sorttype == "planar":
        #         self.counter_dict["last_image_enc"]=enc2_dict[last_d_in_run]
        #     # adding the run to the good_count, minus the one added in compare_images
        #     self.counter_dict["good_count"] += len(k)-1
        #     # switch to returning dist_run_dict
        #     # if VERBOSE: print("dist_run_dict", dist_run_dict)            
            
        #     # at the end of the run, jump somewhere random to break it up
        #     # remove items in dist_run_dict from dist_dict
        #     print("dist_run_dict", len(dist_run_dict), type(dist_run_dict))
        #     print("dist_dict", len(dist_dict), type(dist_dict))

        #     dist_dict_trim = {k: v for k, v in dist_dict.items() if k not in dist_run_dict}
        #     print("dist_dict_trim", len(dist_dict_trim), type(dist_dict_trim))

        #     # for d in dist_run_dict:
        #     #     del dist_dict[d]
            
        #     # make a random jump
        #     # last_d_in_run, dist_single_dict = self.jump(dist, dist_dict_trim)
        #     print(last_d_in_run, dist_single_dict)

            
        #     # random_d = int(len(dist)*random.random())
        #     # random_d_in_run = dist[random_d]

        #     # dist_single_dict = {random_d: dist_dict[random_d_in_run]}
        #     # self.SHOT_CLOCK = 0 # reset the shot clock

        #     # add dist_single_dict to dist_run_dict at the end
        #     print("jumping to random_d_in_run", dist_single_dict)
        #     dist_run_dict[last_d_in_run] = dist_single_dict[random_d_in_run]

        #     return last_d_in_run, dist_run_dict, df_128_enc, df_33_lms
        # '''
        else:
            # print("going to find a winner")
            # dist.sort()
            
            # sort_dHSV returns a list of 128d dists sorted by the sum of 128d and HSVd
            dist = self.sort_dHSV(dist_dict, df_enc)

            print("length of dist -- how many enc are left in the mix")
            # print(dist)
            print(len(dist))
            print(type(len(dist)))
            self.SHOT_CLOCK += 1 # reset the shot clock
            print(f"self.SHOT_CLOCK is {self.SHOT_CLOCK}")
            if dist[0] > self.MAXDIST:
                print("TOO GREAT A DISTANCE ---> going to break the loop and keep the last values")
                dist_single_dict = {1: "null"}
                return 1, dist_single_dict, df_128_enc, df_33_lms
            elif self.JUMP_SHOT and self.SHOT_CLOCK > self.SHOT_CLOCK_MAX:
                print("SHOT_CLOCK IS UP ---> going to pick a random new start")
                # random_d_in_run, dist_single_dict = self.jump(dist, dist_dict)
                random_d = int(len(dist)*random.random())
                random_d_in_run = dist[random_d]

                dist_single_dict = {random_d: dist_dict[random_d_in_run]}
                self.SHOT_CLOCK = 0 # reset the shot clock
                return random_d_in_run, dist_single_dict, df_128_enc, df_33_lms

            elif self.ONE_SHOT and FIRST_ROUND is False:
                # for d in dist:
                #     dist_run_dict[d]
                # k = list(dist_dict.keys())
                last_d_in_run = max(dist)

                print("WE HAVE ONE_SHOT!!!!! ---------- ", str(len(dist)))
                # print(dist_dict)
                if sorttype == "128d":
                    self.counter_dict["last_image_enc"]=enc2_dict[last_d_in_run]
                elif sorttype == "planar":
                    pass

                # adding the run to the good_count, minus the one added in compare_images
                # self.counter_dict["good_count"] += len(dist)-1

                # switch to returning dist_run_dict
                return last_d_in_run, dist_dict, df_128_enc, df_33_lms

            try:
                print ("the winner is: ", str(dist[0]), dist_dict[dist[0]])
            #     print(len(dist))
                self.counter_dict["last_image"]=dist_dict[dist[0]]
                if sorttype == "128d":
                    self.counter_dict["last_image_enc"]=enc2_dict[dist[0]]
                elif sorttype == "planar":
                    print("not setting enc2_dict")
                elif sorttype == "planar_body":
                    print("not setting enc2_dict")
                # make dict of one and to return it
                # dist_single_dict[dist[0]] = dist_dict[dist[0]]
                dist_single_dict = {dist[0]: dist_dict[dist[0]]}
                return dist[0], dist_single_dict, df_128_enc, df_33_lms
            except:
                print("NOTHING HERE")
                dist_single_dict = {1: "null"}
                return 1, dist_single_dict, df_128_enc, df_33_lms






    def write_video(self, ROOT, img_array, segment, size):
        videofile = f"facevid_crop{str(self.MINCROP)}_X{str(self.XLOW)}toX{str(self.XHIGH)}_Y{str(self.YLOW)}toY{str(self.YHIGH)}_Z{str(self.ZLOW)}toZ{str(self.ZHIGH)}_maxResize{str(self.MAXRESIZE)}_ct{str(len(segment))}_rate{(str(self.FRAMERATE))}.mp4"
        size = self.get_cv2size(ROOT, img_array[0])
        try:
            out = cv2.VideoWriter(os.path.join(ROOT,videofile), cv2.VideoWriter_fourcc(*'mp4v'), self.FRAMERATE, size)
            for i in range(len(img_array)):
                print(img_array[i])
                img = cv2.imread(img_array[i])
                print('read file')
                out.write(img)
            out.release()
            # print('wrote:',videofile)
        except:
            print('failed VIDEO, probably because segmented df until empty')


