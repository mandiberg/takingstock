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


class SortPose:
    # """Sort image files based on head pose"""

    def __init__(self, motion, face_height_output, image_edge_multiplier, EXPAND):

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        #maximum allowable distance between encodings
        self.MAXDIST = 0.6

        # maximum allowable scale up
        self.resize_max = 1.6
        self.image_edge_multiplier = image_edge_multiplier
        self.face_height_output = face_height_output
        # takes base image size and multiplies by avg of multiplier
        self.output_dims = (int(face_height_output*(image_edge_multiplier[1]+image_edge_multiplier[3])/2),int(face_height_output*(image_edge_multiplier[0]+image_edge_multiplier[2])/2))
        self.EXPAND = EXPAND

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
            self.XLOW = -20
            self.XHIGH = 1
            self.YLOW = -4
            self.YHIGH = 4
            self.ZLOW = -3
            self.ZHIGH = 3
            self.MINCROP = 1
            self.MAXRESIZE = .5
            self.FRAMERATE = 15
            self.SECOND_SORT = 'face_x'
            self.MAXMOUTHGAP = 2
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


    def make_segment(self, df):

        segment = df.loc[((df['face_y'] < self.YHIGH) & (df['face_y'] > self.YLOW))]
        print(segment.size)
        segment = segment.loc[((segment['face_x'] < self.XHIGH) & (segment['face_x'] > self.XLOW))]
        print(segment.size)
        segment = segment.loc[((segment['face_z'] < self.ZHIGH) & (segment['face_z'] > self.ZLOW))]
        print(segment.size)
        # removing cropX for now. Need to add that back into the data
        # segment = segment.loc[segment['cropX'] >= self.MINCROP]
        # print(segment.size)

        # COMMENTING OUT MOUTHGAP as it is functioning as a minimum. Needs refactoring
        segment = segment.loc[segment['mouth_gap'] >= self.MAXMOUTHGAP]
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
        print("enc1")
        print(enc1)
        print(enc1[0])
        enc2=np.array(enc2)
        print("enc2")
        print(enc2[0])
        d=np.linalg.norm(enc1 - enc2, axis=0)
        print("d")
        print(d)
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
        try:
            img = new_file
            height, width, layers = img.shape
            size = (width, height)
            print('loaded img 1')

            last_img = last_file
            last_height, last_width, last_layers = last_img.shape
            last_size = (last_width, last_height)
            print('loaded img 2')

            # Check if dimensions match
            if size != last_size:
                print('Image dimensions do not match. Skipping blending.')
                return False

            # code for face detection and blending
            if self.is_face(img):
                print('new file is face')
                blend = cv2.addWeighted(img, 0.5, last_img, 0.5, 0.0)
                # foopath = os.path.join("/Users/michaelmandiberg/Documents/projects-active/facemap_production/blends", "foobar_"+str(random.random())+".jpg")
                # cv2.imwrite(foopath, blend)
                # print('blended faces')
                blended_face = self.is_face(blend)
                print('blended is_face', blended_face)
                if blended_face:
                    print('is a face! adding it')
                    return True
                else:
                    print('skipping this one')
                    return False
            else:
                print('new_file is not a face:', new_file)
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
        
        print("Image matching Error between the two images:",error)
        # i don't know what number to use
        if error == 0:
            return False
        elif error < 15:
            # preview_img(diff)
            # preview_img(img1)
            # preview_img(img2)
            return False
        else:
            return True





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
        print("get_faceheight_data")
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
        print("got face_height")
        print(self.face_height)
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

        print("attempting cropped_image")
        print(self.size)
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
        print("going to expand now")
        try:
            print(type(self.image))
            borderType = cv2.BORDER_CONSTANT
            self.EXPAND_SIZE = (5000,5000)
            # value = [255,255,255]
            value = [0,0,0]

            # scale image to match face heights
            resize = self.face_height_output/self.face_height
            if resize < 4:
                print("resize")
                print(resize)
                # image.shape is height[0] and width[1]
                resize_dims = (int(self.image.shape[1]*resize),int(self.image.shape[0]*resize))
                # resize_nose.shape is  width[0] and height[1]
                resize_nose = (int(self.nose_2d[0]*resize),int(self.nose_2d[1]*resize))
                print("resize_dims")
                print(resize_dims)
                print("resize_nose")
                print(resize_nose)
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

                print([top_border, bottom_border, left_border, right_border])
                print([top_border, resize_dims[0]/2-right_border, resize_dims[1]/2-bottom_border, left_border])
                print([top_border, self.EXPAND_SIZE[0]/2-right_border, self.EXPAND_SIZE[1]/2-bottom_border, left_border])

                # expand image with borders
                if top_border >= 0 and right_border >= 0 and self.EXPAND_SIZE[0]/2-right_border >= 0 and bottom_border >= 0 and self.EXPAND_SIZE [1]/2-bottom_border>= 0 and left_border>= 0:
                # if topcrop >= 0 and self.w-rightcrop >= 0 and self.h-botcrop>= 0 and leftcrop>= 0:
                    print("all positive")
                    new_image = cv2.copyMakeBorder(resized_image, top_border, bottom_border, left_border, right_border, borderType, None, value)
                else:
                    print("one is negative")
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

    def crop_image(self,image, faceLms, bbox, sinY=0):

        self.get_image_face_data(image, faceLms, bbox)    
        # check for crop, and if not exist, then get
        # if not hasattr(self, 'crop'): 
        try:
            toobig = self.get_crop_data_scalable()
        except:
            print("couldn't get crop data")
            toobig = True

        if not toobig:
            print("going to crop because too big is ", toobig)
            # print (self.padding_points)
            #set main points for drawing/cropping

            #moved this back up so it would NOT     draw map on both sets of images
            try:
                print(type(self.image))
                # image_arr = numpy.array(self.image)
                # print(type(image_arr))
                cropped_actualsize_image = self.image[self.simple_crop[0]:self.simple_crop[2], self.simple_crop[3]:self.simple_crop[1]]
                print("cropped_actualsize_image.shape")
                print(cropped_actualsize_image.shape)
                resize = self.output_dims[0]/cropped_actualsize_image.shape[0] 
                print(resize)
                if resize > self.resize_max:
                    print("toosmall")
                    self.toosmall_count += 1
                    return None
                print("about to resize")
                # crop[0] is top, and clockwise from there. Right is 1, Bottom is 2, Left is 3. 
                print(self.output_dims)
                cropped_image = cv2.resize(cropped_actualsize_image, (self.output_dims), interpolation=cv2.INTER_LINEAR)
                print("image actually cropped")
            except:
                cropped_image = None
                print("not cropped_image loop")

                print(self.h, self.w)

        else:
            cropped_image = None
            # resize = None
        return cropped_image


    def get_start_enc(self, start_img, df_128_enc):
        print("get_start_enc")
        if start_img == "median":
            enc1 = df_128_enc.median().to_list()
            print("in median")

        elif start_img == "start_site_image_id":
            print("start_site_image_id (this is what we are comparing to)")
            print(start_site_image_id)
            enc1 = df_128_enc.loc[start_site_image_id].to_list()
        else:
    #         enc1 = get 2-129 from df via stimg key
            print("start_img key is (this is what we are comparing to):")
            print(start_img)
            enc1 = df_128_enc.loc[start_img].to_list()
            try:
                df_128_enc=df_128_enc.drop(start_img)
            except:
                print("couldn't drop the start_img")
        return enc1, df_128_enc

    def get_closest_df(self, enc1, df_128_enc):
        print("get_closest_df")
        dist=[]
        dist_dict={}
        
        for index, row in df_128_enc.iterrows():
    #         print(row['c1'], row['c2'])
    #     for img in img_list:
            enc2 = row
            # print("testing this", index, "against the start img",start_img)
            if (enc1 is not None) and (enc2 is not None):
                d = self.get_d(enc1, enc2)
                print ("d is", str(d), "for", index)
                dist.append(d)
                dist_dict[d]=index
        dist.sort()
        print("debug index")
        print(dist)
        print(len(dist))
        print ("the winner is: ", str(dist[0]), dist_dict[dist[0]])
    #     print(len(dist))
        return dist[0], dist_dict[dist[0]], df_128_enc



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


