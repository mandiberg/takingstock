import statistics
import os
import cv2
import pandas as pd
import mediapipe as mp
import hashlib
import time


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
            self.MAXMOUTHGAP = 40
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

        angle_list_median = round(statistics.median(self.angle_list))
        print('angle_list_median: ',angle_list_median)
        print('angle_list_median][SECOND_SORT]',self.d[angle_list_median])

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
                print(self.d[angle].iloc[1]['newname'])
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
                                            min_detection_confidence=0.6
                                            ) as face_detection:
            # image = cv2.imread(file)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                is_face = False
            else:
                is_face = True
            # annotated_image = image.copy()
            # for detection in results.detections:
            #     is_face = True
            #     print('Nose tip:')
            #     print(mp_face_detection.get_key_point(
            #       detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            #     mp_drawing.draw_detection(annotated_image, detection)
            # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

        return is_face

    def blend_is_face(self, oldimage, newimage):
        blend = cv2.addWeighted(oldimage, 0.5, newimage, 0.5, 0.0)
        # blend = cv2.addWeighted(img, 0.5, img_array[i-1], 0.5, 0.0)
        blended_face = sort.is_face(blend)
        return blended_face



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
            print(row['face_x'], row['face_y'], row['newname'])

            #I don't know what this does or why
            delta_array.append(row['mouth_gap'])

            try:
                img = cv2.imread(row['newname'])
                height, width, layers = img.shape
                size = (width, height)
                # test to see if this is actually an face, to get rid of blank ones/bad ones
                # this may not be necessary
                img_array.append(img)

                i+=1

            except:
                print('failed:',row['newname'])
        # print("delta_array")
        # print(delta_array)
        return img_array, size        








    def get_cv2size(self, ROOT, filename_or_imagedata):
        #IF filename_or_imagedata IS STRING:
        img = cv2.imread(os.path.join(ROOT,filename_or_imagedata))
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

    def write_images(self, ROOT, img_array, segment):
        # print('writing images')
        imgfileprefix = f"faceimg_crop{str(self.MINCROP)}_X{str(self.XLOW)}toX{str(self.XHIGH)}_Y{str(self.YLOW)}toY{str(self.YHIGH)}_Z{str(self.ZLOW)}toZ{str(self.ZHIGH)}_maxResize{str(self.MAXRESIZE)}_ct{str(len(segment))}"
        # print(imgfileprefix)
        outfolder = os.path.join(ROOT,"images"+str(time.time()))
        if not os.path.exists(outfolder):      
            os.mkdir(outfolder)

        try:
            # couldn't I use i here? 
            counter = 1
            # print(img_array)
            # out = cv2.VideoWriter(os.path.join(ROOT,videofile), cv2.VideoWriter_fourcc(*'mp4v'), FRAMERATE, size)
            for i in range(len(img_array)):
                # print('in loop')
                UID = img_array[i].split('-id')[-1]
                imgfilename = imgfileprefix+"_"+UID+"_"+str(counter)+".jpg"
                outpath = os.path.join(outfolder,imgfilename)
                # print(img_array[i])

                # this code takes image i, and blends it with the subsequent image
                # next step is to test to see if mp can recognize a face in the image
                # if no face, a bad blend, try again with i+2, etc. 
                # except it would need to do that with the sub-array, so move above? 
                # blend = cv2.addWeighted(img_array[i], 0.5, img_array[(i+1)], 0.5, 0.0)
                # cv2.imwrite(outpath, blend)
                img = cv2.imread(img_array[i])
                cv2.imwrite(outpath, img)
                print("saved: ",imgfilename)

                # print(outpath)
                # out.write(img_array[i])
                counter += 1
            # out.release()
            # print('wrote:',videofile)
        except:
            print('failed IMAGES, probably because segmented df until empty')



