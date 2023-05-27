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


class SortPose:
    # """Sort image files based on head pose"""

    def __init__(self, motion):

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

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


    def unique_face(self,img1,img2):
        def preview_img(img):
            cv2.imshow("difference", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


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
            preview_img(diff)
            preview_img(img1)
            preview_img(img2)
            return False
        else:
            return True

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

    def write_images(self, ROOT, df_sorted):
        # site_name_id = df_enc.loc[start_img]['site_name_id']

        print('writing images')
        imgfileprefix = f"faceimg_crop{str(self.MINCROP)}_X{str(self.XLOW)}toX{str(self.XHIGH)}_Y{str(self.YLOW)}toY{str(self.YHIGH)}_Z{str(self.ZLOW)}toZ{str(self.ZHIGH)}_maxResize{str(self.MAXRESIZE)}_ct{str(df_sorted.size)}"
        print(imgfileprefix)
        outfolder = os.path.join(ROOT,"images"+str(time.time()))
        if not os.path.exists(outfolder):      
            os.mkdir(outfolder)

        try:
            # couldn't I use i here? 
            counter = 1
            last_image = None
            is_face = None
            first_run = True
            # print(df_sorted)
            # out = cv2.VideoWriter(os.path.join(ROOT,videofile), cv2.VideoWriter_fourcc(*'mp4v'), FRAMERATE, size)
            for index, row in df_sorted.iterrows():
                print('in loop, index is', str(index))
                UID = row['filename'].split('-id')[-1].split("/")[-1].replace(".jpg","")
                print("UID ",UID)
                counter_str = str(counter).zfill(len(str(df_sorted.size)))  # Add leading zeros to the counter
                imgfilename = imgfileprefix+"_"+str(counter_str)+"_"+UID+".jpg"
                print("imgfilename ",imgfilename)
                outpath = os.path.join(outfolder,imgfilename)
                print("outpath ",outpath)

                # folder is specific to each file's site_name_id

                # this is how it was, and seems hardcoded to Test36
                # open_path = os.path.join(ROOT,row['folder'],row['filename'])

                # here I'm using the actual root. Root gets pulled from io, then passed back to sort pose.
                # but the folder is fused to the root somewhere... in makevideo? it needs to be found and pulled off there. 
                open_path = os.path.join(ROOT,row['folder'].replace("/Volumes/Test36/",""),row['filename'])
                print(ROOT,row['folder'],row['filename'])

                print("open_path ",open_path)

                # this code takes image i, and blends it with the subsequent image
                # next step is to test to see if mp can recognize a face in the image
                # if no face, a bad blend, try again with i+2, etc. 
                # except it would need to do that with the sub-array, so move above? 
                # blend = cv2.addWeighted(img_array[i], 0.5, img_array[(i+1)], 0.5, 0.0)
                # cv2.imwrite(outpath, blend)
                img = cv2.imread(open_path)

                #crop image here:

                cropped_image = self.crop_image(img, row['face_landmarks'], row['bbox'])
                print("cropped_image type: ",type(cropped_image))
                if cropped_image is not None:
                    print(cropped_image.shape)
                    print("have a cropped image trying to save")
                    try:
                        print(type(last_image))
                    except:
                        print("couldn't test last_image")
                    try:
                        if not first_run:
                            print("testing is_face")
                            is_face = self.test_pair(last_image, cropped_image)
                            if is_face and row['dist'] < 0.35:
                                print("same person, testing mse")
                                # print(start_img,index,site_name_id)
                                # mse = self.test_mse(ROOT,last_image,cropped_image,site_name_id)
                                is_face = self.unique_face(last_image,cropped_image)
                                print ("mse ",mse)
                        else:
                            print("first round, skipping the pair test")
                    except:
                        print("last_image try failed")
                    if is_face or first_run:
                        first_run = False
                        cv2.imwrite(outpath, cropped_image)
                        last_image = cropped_image
                        print("saved: ",outpath)
                    else: 
                        print("pair do not make a face, skipping")
                else:
                    print("no image here, trying next")
                # print(outpath)
                # out.write(img_array[i])
                counter += 1

            # i think this is left over from video???
            # out.release()
            print('wrote files')
        except Exception as e:
            print(str(e))




##############################

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
                print("found point:")
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

    def get_crop_data_simple(self):
        print("get_crop_data_simple")
        #it would prob be better to do this with a dict and a loop
        # nose_2d = self.get_face_2d_point(self.faceLms,1)
        # print("self.sinY: ",self.sinY)
        #set main points for drawing/cropping
        #p1 is tip of nose
        p1 = (int(self.nose_2d[0]), int(self.nose_2d[1]))
        # print(p1)

        toobig = False
        # print(p1[1])
        # print(self.face_height)
        # print(self.h)
        if p1[1]>(self.face_height*1) and (self.h-p1[1])>(self.face_height*1):
            if p1[0]>(self.face_height*1) and (self.w-p1[0])>(self.face_height*1):
                self.crop_multiplier = 1
            else:
                print('face too wiiiiiiiide')
                self.crop_multiplier = .25
                toobig=True

        else:
            self.crop_multiplier = .25
            print('face too biiiiigggggg')
            toobig=True

        if not toobig:
            print(self.crop_multiplier)
            # self.h - p1[1]
            top_overlap = p1[1]-self.face_height

            print(top_overlap)
            if top_overlap > 0:
                #set crop
                crop_size = self.face_height*self.crop_multiplier
                leftcrop = int(p1[0]-crop_size)
                rightcrop = int(p1[0]+crop_size)
                topcrop = int(p1[1]-crop_size)
                botcrop = int(p1[1]+crop_size)
                self.simple_crop = [topcrop, rightcrop, botcrop, leftcrop]
                print("crop top, right, bot, left")
                print(self.simple_crop)
            else:
                print("top_overlap is negative")
        return toobig


    def crop_image(self,cropped_image, faceLms, bbox, sinY=0):
        self.image = cropped_image
        self.size = (self.image.shape[0], self.image.shape[1])
        self.h = self.image.shape[0]
        self.w = self.image.shape[1]
        self.faceLms = faceLms
        print("about to load_ json")
        # self.bbox = json.loads(bbox)
        self.bbox = (bbox)
        print(self.bbox)
        print(type(self.bbox))
        print(self.bbox['left'])

        print("attempting cropped_image")
        #I'm not sure the diff between nose_2d and p1. May be redundant.
        #it would prob be better to do this with a dict and a loop
        # Instead of hard-coding the index 1, you can use a variable or constant for the point index
        nose_point_index = 1
        self.nose_2d = self.get_face_2d_point(nose_point_index)
        print(self.nose_2d)

        try:
            # get self.face_height
            self.get_faceheight_data()
        except:
            print("couldn't get_faceheight_data")
    
        # check for crop, and if not exist, then get
        # if not hasattr(self, 'crop'): 
        try:
            toobig = self.get_crop_data_simple()
        except:
            print("couldn't get crop data")
            # this is the in progress neck rotation stuff
            # self.get_crop_data(sinY)

        if not toobig:
            print("not too big, this big: ", toobig)
            # print (self.padding_points)
            #set main points for drawing/cropping

            # getting rid of this. I think it is for adding padding
            #p1 is tip of nose
            # p1 = (int(self.nose_2d[0]), int(self.nose_2d[1]))

            # # print(crop_multiplier)
            # self.h - p1[1]
            # top_overlap = p1[1]-self.face_height

            # #adding this in to padd image
            # try:
            #     padded_image = self.add_margin(cropped_image, self.padding_points)
            # except:
            #     padded_image = False

            basesize = 750
            resize_factor = basesize/self.face_height
            newsize = basesize*resize_factor
            # newsize = (basesize*self.simple_crop[0],basesize*self.simple_crop[1])
            # print(newsize)
            # resize = np.round(newsize[0]/(self.face_height*2.5), 3)
            # print(resize)

            #moved this back up so it would NOT     draw map on both sets of images
            try:
                print(type(self.image))
                # image_arr = numpy.array(self.image)
                # print(type(image_arr))
                cropped_actualsize_image = self.image[self.simple_crop[0]:self.simple_crop[2], self.simple_crop[3]:self.simple_crop[1]]
                print(cropped_actualsize_image.shape)
                desired_width = 750  # Specify the desired width
                desired_height = 750  # Specify the desired height
                # crop[0] is top, and clockwise from there. Right is 1, Bottom is 2, Left is 3. 
                cropped_image = cv2.resize(cropped_actualsize_image, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
                print("image actually cropped")
            except:
                cropped_image = None
                print("not cropped_image loop")

                print(self.h, self.w)
        else:
            cropped_image = None
            # resize = None
        return cropped_image
