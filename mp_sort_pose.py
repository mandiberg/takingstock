import statistics


class SortPose:
    # """Sort image files based on head pose"""

    def __init__(self, motion):
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
            self.SECOND_SORT = 'x'
            self.MAXMOUTHGAP = 2
            self.SORT = 'mouth_gap'
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
            print('angle: ',angle)
            print ('d angle size: ',self.d[angle].size)
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

    #not currently in use. not sure what the diff is between this and cycling order. 
    # will need to be refactored into a class method if I use in the future. 
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


