import statistics
import os
import cv2
import pandas as pd
import mediapipe as mp
import hashlib


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
            self.SORT = 'y'
            self.SECOND_SORT = 'x'
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
            self.SECOND_SORT = 'x'
            self.MAXMOUTHGAP = 40
            self.SORT = 'mouth_gap'
            self.ROUND = 1
        elif motion['forward_nosmile'] == True:
            self.XLOW = -20
            self.XHIGH = 1
            self.YLOW = -4
            self.YHIGH = 4
            self.ZLOW = -3
            self.ZHIGH = 3
            self.MINCROP = 1
            self.MAXRESIZE = .5
            self.FRAMERATE = 15
            self.SECOND_SORT = 'y'
            self.MAXMOUTHGAP = 2
            self.SORT = 'x'
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
            self.SORT = 'x'
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
            self.SORT = 'x'
            self.ROUND = 1


    def make_segment(self, df):

        segment = df.loc[((df['y'] < self.YHIGH) & (df['y'] > self.YLOW))]
        print(segment.size)
        segment = segment.loc[((segment['x'] < self.XHIGH) & (segment['x'] > self.XLOW))]
        print(segment.size)
        segment = segment.loc[((segment['z'] < self.ZHIGH) & (segment['z'] > self.ZLOW))]
        print(segment.size)
        segment = segment.loc[segment['cropX'] >= self.MINCROP]
        print(segment.size)
        segment = segment.loc[segment['mouth_gap'] >= self.MAXMOUTHGAP]
        # segment = segment.loc[segment['mouth_gap'] <= MAXMOUTHGAP]
        print(segment.size)
        # segment = segment.loc[segment['resize'] < MAXRESIZE]
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

    def get_d(self, segment):

        divisor = eval(f"1e{self.ROUND}")
        self.d = {}
        for angle in self.angle_list:
            # print(angle)
            self.d[angle] = segment.loc[((segment[self.SORT] > angle) & (segment[self.SORT] < angle+(1/divisor)))]
            # print(self.d[angle].size)
        return self.d

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
                this_median = self.d[angle]['x'].median()
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

    def get_hash_folders(self,filename):
        m = hashlib.md5()
        m.update(filename.encode('utf-8'))
        d = m.hexdigest()
        # csvWriter1.writerow(["https://upload.wikimedia.org/wikipedia/commons/"+d[0]+'/'+d[0:2]+'/'+filename])
        return d[0], d[0:2]

    #not currently in use. not sure what the diff is between this and cycling order. 
    # will need to be refactored into a class method if I use in the future. 
    def simple_order(self, segment):
        img_array = []
        delta_array = []
        # size = []
        #simple ordering
        rotation = segment.sort_values(by=self.SORT)
        print("rotation: ")
        print(rotation)

        # for num, name in enumerate(presidents, start=1):
        i = 0
        for index, row in rotation.iterrows():
            print(index, row['x'], row['y'], row['newname'])
            delta_array.append(row['mouth_gap'])
            # datapath = row['newname'].split("gettyimages_output/")
            # # filepath = os.path.join(str(datapath[0]),self.get_hash_folders(datapath[1]),str(datapath[1]))
            # a, b = self.get_hash_folders(datapath[1][0])
            # filepath = str(datapath[0])+str(a)+"/"+str(b)+"/"+str(datapath[1])
            # print(filepath)
            try:
                img = cv2.imread(row['newname'])
                height, width, layers = img.shape
                size = (width, height)
                # test to see if this is actually an face, to get rid of blank ones/bad ones
                # this may not be necessary
                if self.is_face(img):
                    # if not the first image
                    if i>0:
                        print("in loop")
                        # blend this image with the last image
                        blend = cv2.addWeighted(img, 0.5, img, 0.5, 0.0)
                        # blend = cv2.addWeighted(img, 0.5, img_array[i-1], 0.5, 0.0)
                        print("blended")
                        blended_face = self.is_face(blend)
                        print("blended_face")

                        # print('is_face ',blended_face)
                        # if blended image has a detectable face, append the img
                        if blended_face:
                            img_array.append(img)
                            print('is a face! adding it')
                        else:
                            print('skipping this one')
                    # for the first one, just add the image
                    # this may need to be refactored in case the first one is bad?
                    else:
                        img_array.append(img)
                else:
                    print('skipping this one: ',row['newname'])

                i+=1

            except:
                print('failed:',row['newname'])
        print("delta_array")
        print(delta_array)
        return img_array, size


    def simplest_order(self, segment):
        img_array = []
        delta_array = []
        #simple ordering, second sort, because this is the...?
        rotation = segment.sort_values(by=self.SORT)

        i = 0
        for index, row in rotation.iterrows():
            print(row['x'], row['y'], row['newname'])

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


    def cycling_order(self, CYCLECOUNT):
        img_array = []
        cycle = 0 
        # metamedian = get_metamedian(angle_list)
        metamedian = self.metamedian
        d = self.d

        print("CYCLE to test: ",cycle)

        while cycle < CYCLECOUNT:
            print("CYCLE: ",cycle)
            for angle in self.angle_list:
                print("angle: ",str(angle))
                # # print(d[angle].iloc[(d[angle][SECOND_SORT]-metamedian).abs().argsort()[:2]])
                # # print(d[angle].size)
                try:
                    # I don't remember exactly how this segments the data...!!!
                    # [:CYCLECOUNT] gets the first [:0] value on first cycle?
                    # or does it limit the total number of values to the number of cycles?
                    mysteryvalue = (d[angle][self.SECOND_SORT]-self.metamedian)
                    print('mysteryvalue ',mysteryvalue)
                    mysterykey = mysteryvalue.abs().argsort()[:CYCLECOUNT]
                    print('mysterykey: ',mysterykey)
                    closest = d[angle].iloc[mysterykey]
                    closest_file = closest.iloc[cycle]['newname']
                    closest_mouth = closest.iloc[cycle]['mouth_gap']
                    print('closest: ')
                    print(closest_file)
                    img = cv2.imread(closest_file)
                    height, width, layers = img.shape
                    size = (width, height)
                    img_array.append(img)
                except:
                    print('failed cycle angle:')
                    # print('failed:',row['newname'])
            print('finished a cycle')
            self.angle_list.reverse()
            cycle = cycle +1
            # print(angle_list)
        return img_array, size






    def write_video(self, ROOT, img_array, segment, size):
        videofile = f"facevid_crop{str(self.MINCROP)}_X{str(self.XLOW)}toX{str(self.XHIGH)}_Y{str(self.YLOW)}toY{str(self.YHIGH)}_Z{str(self.ZLOW)}toZ{str(self.ZHIGH)}_maxResize{str(self.MAXRESIZE)}_ct{str(len(segment))}_rate{(str(self.FRAMERATE))}.mp4"

        try:
            out = cv2.VideoWriter(os.path.join(ROOT,videofile), cv2.VideoWriter_fourcc(*'mp4v'), self.FRAMERATE, size)
            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()
            print('wrote:',videofile)
        except:
            print('failed VIDEO, probably because segmented df until empty')

    def write_images(self, outfolder, img_array, segment, size):
        imgfileprefix = f"faceimg_crop{str(self.MINCROP)}_X{str(self.XLOW)}toX{str(self.XHIGH)}_Y{str(self.YLOW)}toY{str(self.YHIGH)}_Z{str(self.ZLOW)}toZ{str(self.ZHIGH)}_maxResize{str(self.MAXRESIZE)}_ct{str(len(segment))}"
        
        if not os.path.exists(outfolder):      
            os.mkdir(outfolder)

        try:
            # couldn't I use i here? 
            counter = 1
            # out = cv2.VideoWriter(os.path.join(ROOT,videofile), cv2.VideoWriter_fourcc(*'mp4v'), FRAMERATE, size)
            for i in range(len(img_array)):
                # print('in loop')
                imgfilename = imgfileprefix+"_"+str(counter)+".jpg"
                outpath = os.path.join(outfolder,imgfilename)
                # this code takes image i, and blends it with the subsequent image
                # next step is to test to see if mp can recognize a face in the image
                # if no face, a bad blend, try again with i+2, etc. 
                # except it would need to do that with the sub-array, so move above? 
                # blend = cv2.addWeighted(img_array[i], 0.5, img_array[(i+1)], 0.5, 0.0)
                # cv2.imwrite(outpath, blend)

                # here is the original noblend write:
                cv2.imwrite(outpath, img_array[i])
                print("saved: ",imgfilename)

                # print(outpath)
                # out.write(img_array[i])
                counter += 1
            # out.release()
            # print('wrote:',videofile)
        except:
            print('failed IMAGES, probably because segmented df until empty')



