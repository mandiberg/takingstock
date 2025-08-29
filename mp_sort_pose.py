import math
import shutil
import statistics
import os
from turtle import distance
import cv2
import pandas as pd
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import hashlib
import time
import json
import random
import numpy as np
import sys
from collections import Counter
from simple_lama_inpainting import SimpleLama
from sklearn.neighbors import NearestNeighbors
import re
import random
from cv2 import dnn_superres
import pymongo
import pickle
import traceback
from skimage.metrics import structural_similarity as ssim


from deap import base, creator, tools, algorithms


class SortPose:
    # """Sort image files based on head pose"""

    def __init__(self, motion, face_height_output, image_edge_multiplier, EXPAND=False, ONE_SHOT=False, JUMP_SHOT=False, HSV_CONTROL=None, VERBOSE=True,INPAINT=False, SORT_TYPE="128d", OBJ_CLS_ID = None,UPSCALE_MODEL_PATH=None, use_3D=False,TSP_SORT=False):

        # need to refactor this for GPU
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.get_bg_segment=mp.solutions.selfie_segmentation.SelfieSegmentation()  
              
        self.VERBOSE = VERBOSE

        #maximum allowable distance between encodings (this accounts for dHSV)
        self.MAXDIST = 1.8
        self.MAXFACEDIST = .8
        self.MINFACEDIST = .5 #TK
        self.MAXBODYDIST = 1.5
        self.MINBODYDIST = .15
        self.FACE_DUPE_DIST = .06
        self.BODY_DUPE_DIST = .04
        self.HSV_DELTA_MAX = .5
        self.HSVMULTIPLIER = 3
        self.NORM_BODY_MULTIPLIER = 4
        self.BRUTEFORCE = False
        self.use_3D = use_3D
        # print("init use_3D",self.use_3D)
        self.CUTOFF = 20 # DOES factor if ONE_SHOT
        self.ORIGIN = 0
        self.this_nose_bridge_dist = self.NOSE_BRIDGE_DIST = None # to be set in first loop, and sort.this_nose_bridge_dist each time

        self.CHECK_DESC_DIST = 30

        self.SORT_TYPE = SORT_TYPE
        self.TSP_SORT = TSP_SORT

        if self.SORT_TYPE == "128d":
            self.MIND = self.MINFACEDIST * 1.5
            self.MAXD = self.MAXFACEDIST * 1.3
            self.MULTIPLIER = self.HSVMULTIPLIER
            self.DUPED = self.FACE_DUPE_DIST
            self.HSV_DELTA_MAX = self.HSV_DELTA_MAX * 1.5
        elif self.SORT_TYPE == "planar": 
            self.MIND = self.MINBODYDIST * 1.5
            self.MAXD = self.MAXBODYDIST
            self.MULTIPLIER = self.HSVMULTIPLIER * (self.MINBODYDIST / self.MINFACEDIST)
            self.DUPED = self.BODY_DUPE_DIST
        elif self.SORT_TYPE in ["planar_body", "planar_hands", "fingertips_positions", "body3D"]: 
            self.MIND = self.MINBODYDIST
            self.MAXD = self.MAXBODYDIST * 4
            self.MULTIPLIER = self.HSVMULTIPLIER * (self.MINBODYDIST / self.MINFACEDIST)
            self.DUPED = self.BODY_DUPE_DIST
            self.FACE_DIST_TEST = .02
            self.CHECK_DESC_DIST = 45
        elif self.SORT_TYPE == "planar_hands_USE_ALL":
            # designed to take everything
            self.MIND = 1000
            self.MAXD = 1000
            self.MULTIPLIER = 1
            self.DUPED = 1
            self.FACE_DIST_TEST = -1
            self.CHECK_DESC_DIST = -1
            self.HSV_DELTA_MAX = 1000            
            self.FACE_DUPE_DIST = -1
            self.BODY_DUPE_DIST = -1
    

        self.INPAINT=INPAINT
        if self.INPAINT:self.INPAINT_MODEL=SimpleLama()
        # self.MAX_IMAGE_EDGE_MULTIPLIER=[1.5,2.6,2,2.6] #maximum of the elements

        self.knn = NearestNeighbors(metric='euclidean', algorithm='ball_tree')

        # if edge_multiplier_name:self.edge_multiplier_name=edge_multiplier_name
        # maximum allowable scale up
        self.resize_max = 5.99
        self.resize_increment = 345
        self.USE_INCREMENTAL_RESIZE = True
        self.image_edge_multiplier = image_edge_multiplier
        self.face_height_output = face_height_output
        self.EXPAND = EXPAND
        self.EXPAND_SIZE = (25000,25000) # full body
        # self.EXPAND_SIZE = (10000,10000) # regular
        # self.EXPAND_SIZE = (6400,6400)
        if self.EXPAND_SIZE[0] == 25000:
            self.EXISTING_CROPABLE_DIM = 8288
        else:
            self.EXISTING_CROPABLE_DIM = None
            print(f"   XXX ERROR XXX NEED TO SET EXISTING_CROPABLE_DIM for {self.EXPAND_SIZE[0]}")
        self.TARGET_SIZE = (384, 384)  # Size to temporarily resize images for SSIM comparison
        self.BGCOLOR = [0,0,0]
        self.ONE_SHOT = ONE_SHOT
        self.JUMP_SHOT = JUMP_SHOT
        self.SHOT_CLOCK = 0
        self.SHOT_CLOCK_MAX = 10
        # self.BODY_LMS = list(range(13, 23)) # 0 is nose, 13-22 are left and right hands and elbows
        # this was set to 13,23 as of July 2025, prior to assigning clusters on BodyPoses3D
        self.BODY_LMS = list(range(33)) # 0 is nose, 13-22 are left and right hands and elbows
        
        ## clustering parameters
        self.query_face = True # set to true. Clusturing code will set some to false
        self.query_hands = True
        self.query_body = True
        self.query_head_pose = True
        self.cluster_medians = None
        self.hands_medians = None
        # # self.BODY_LMS = [0,15,16,19,20,21,22] # 0 is nose, 13-22 are left and right hands and elbows
        # self.POINTERS = [16,20,15,19] # 0 is nose, 13-22 are left and right hands and elbows
        # self.THUMBS = [16,22,15,21] # 0 is nose, 13-22 are left and right hands and elbows
        # self.BODY_LMS = [20,19] # 0 is nose, 13-22 are left and right hands and elbows
        # self.SUBSET_LANDMARKS = [13,14,19,20] # this should match what is in Clustering
        self.ELBOW_HAND = [i for i in range(13,22)]
        self.HAND_LMS = self.make_subset_landmarks(15,22)
        # self.RIGHT_HAND_LMS = self.make_subset_landmarks(16,18,20,22)
        # forearm
        self.FOREARM_LMS = self.make_subset_landmarks(13,16)
        self.FINGER_LMS = self.make_subset_landmarks(19,20)
        self.THUMB_POINTER_LMS = self.make_subset_landmarks(19,22)
        self.WRIST_LMS = self.make_subset_landmarks(15,16)
        self.ANKLES_LMS = self.make_subset_landmarks(27,28)
        self.HAND_LMS_POINTER = self.make_subset_landmarks(8,8)
        # adding pointer finger tip
        # self.SUBSET_LANDMARKS.extend(self.FINGER_LMS) # this should match what is in Clustering
        # only use wrist and finger

        # Hands Positions/Gestures
        self.HANDS_POSITIONS_LMS = self.make_subset_landmarks(0,20)

        if self.SORT_TYPE == "fingertips_positions":
            # just fingertips
            self.CLUSTER_TYPE = "fingertips_positions" # fingertips_positions
            self.SUBSET_LANDMARKS = self.HAND_LMS_POINTER
            self.SORT_TYPE = "planar_hands"
        elif "hands" in self.SORT_TYPE:
            # catches any hands
            self.CLUSTER_TYPE = "planar_hands"
            self.SUBSET_LANDMARKS = self.HAND_LMS
        else:
            self.CLUSTER_TYPE = "BodyPoses" # defaults
            self.SUBSET_LANDMARKS = self.BODY_LMS
            # TBD for DEFAULT LMS SUBSET

        # print("final set of subset landmarks", self.SUBSET_LANDMARKS)
        # self.SUBSET_LANDMARKS = self.choose_hand(self.HAND_LMS,"right")
        

        self.OBJ_CLS_ID = OBJ_CLS_ID

        # self.BODY_LMS = [15]
        #____________________TSP SORT________________
        if self.TSP_SORT==True:
            if self.VERBOSE:print("using travelling salesman sorting")
            self.SKIP_FRAC = 35/100  # fraction of entries that can be skipped to optimize for distance
            self.POP_SIZE =200 # larger size leads to more accurate results but more time taken
            self.SEED=42 # Seed for reproducible sort
        #____________________TSP SORT________________

        # place to save bad images
        self.not_make_face = []
        self.same_img = []
        # for testing shoulders for image background
        self.SHOULDER_THRESH = 0.75
        self.nose_2d = None
        self.nose_3d = None
        #UPSCALING PARAMS
        if UPSCALE_MODEL_PATH:
            self.upscale_model= self.set_upscale_model(UPSCALE_MODEL_PATH)
        # luminosity parameters
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
        elif motion['forward_wider'] == True:
            print("setting XYZ for forward_wider")
            # self.XLOW = -33
            # self.XHIGH = -27
            self.XLOW = -40
            self.XHIGH = -20
            self.YLOW = -5
            self.YHIGH = 5
            self.ZLOW = -5
            self.ZHIGH = 5
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
        elif motion['use_all'] == True:
            self.XLOW = -180
            self.XHIGH = 180
            self.YLOW = -180
            self.YHIGH = 180
            self.ZLOW = -180
            self.ZHIGH = 180
            self.MINCROP = 1
            self.MAXRESIZE = .5
            self.FRAMERATE = 15
            self.SECOND_SORT = 'mouth_gap'
            self.MAXMOUTHGAP = 1000
            self.SORT = 'face_x'
            self.ROUND = 1

    def set_output_dims(self):
        if self.image_edge_multiplier == [1.3,1.85,2.4,1.85]:
            print("setting face_height_output to 1.925/1.85")
            self.face_height_output = self.face_height_output*(1.925/1.85)
            self.output_dims = (1920,1920)
        else:
            # self.face_height_output = face_height_output
            # takes base image size and multiplies by avg of multiplier
            self.output_dims = (int(self.face_height_output*(self.image_edge_multiplier[1]+self.image_edge_multiplier[3])/2),int(self.face_height_output*(self.image_edge_multiplier[0]+self.image_edge_multiplier[2])/2))
        self.MAX_IMAGE_EDGE_MULTIPLIER = self.image_edge_multiplier #testing
        print("output_dims",self.output_dims)

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
            "last_image_id":None,
            "last_description":None,
            "last_image_enc":None,
            "last_image_hsv":None,
            "last_image_lum":None,
            "cluster_no":cluster_no

        }

    # def ensure_lms_list(self, lms):
    #     # convert NormalizedLandmarkList to list of tuples
    #     if isinstance(lms, landmark_pb2.NormalizedLandmarkList):
    #         return [(lm.x, lm.y, lm.z) for lm in lms.landmark]

    def compute_skip_threshold(self):
        """
        Given a symmetric distance matrix and a desired number of skips,
        return the distance threshold above which exactly `max_skip` pairwise
        distances lie.

        Parameters
        ----------
        dist_matrix : np.ndarray, shape (n, n)
            Symmetric matrix of pairwise distances, with zeros on the diagonal.
        max_skip : int
            The number of distances you want to skip (the top `max_skip` largest).
        
        Returns
        -------
        float
            The distance cutoff so that exactly `max_skip` distances are larger.
        """
        # 1) Extract upper-triangle distances (excluding diagonal)
        triu_idxs = np.triu_indices_from(self.dist_matrix, k=1)
        dists = self.dist_matrix[triu_idxs]
        
        # 2) Sort distances ascending
        dists_sorted = np.sort(dists)
        total = len(dists_sorted)
        
        # 3) The cutoff is at the (total - max_skip)th value
        #    If max_skip >= total, return max distance
        if self.MAX_SKIP >= total:
            return dists_sorted[-1]
        cutoff_index = total - self.MAX_SKIP
        if self.VERBOSE:print("cutoff_index",cutoff_index)
        return float(dists_sorted[cutoff_index])
    
    def evaluate_tsp(self,ind):
        route = list(ind)
        skip_cnt = 0
        for _ in range(int(self.MAX_SKIP)):
            overheads = []
            for j in range(1, len(route)-1):
                p, c, n = route[j-1], route[j], route[j+1]
                incl = self.dist_matrix[p,c] + self.dist_matrix[c,n]
                skip = self.dist_matrix[p,n]
                overheads.append((incl-skip, j))
            if not overheads:
                break
            max_oh, idx = max(overheads, key=lambda x: x[0])
            if max_oh > self.SKIP_THRESHOLD:
                route.pop(idx)
                skip_cnt += 1
            else:
                break
        total = sum(self.dist_matrix[route[i], route[i+1]] for i in range(len(route)-1))
        return (total + skip_cnt * self.SKIP_PENALTY,)
    
    def cxFixedEndsSimpleSwap(self,ind1, ind2):
        """Swap a random slice only within ind[1:-1], keeping endpoints fixed."""
        a, b = sorted(random.sample(range(1, self.N_POINTS-1), 2))
        ind1[a:b], ind2[a:b] = ind2[a:b], ind1[a:b]
        return ind1, ind2

    def mutFixedEndsShuffle(self,ind, indpb):
        """Shuffle only inside ind[1:-1], keeping endpoints fixed."""
        sub = ind[1:-1]
        tools.mutShuffleIndexes(sub, indpb=indpb)
        ind[1:-1] = sub
        return (ind,)

    def set_TSP_sort(self,df,START_IDX=None,END_IDX=None):
        #____SETTING UP DISTANCE MATRIX_______
        df_array = df.values
        df_diffs = df_array[:, None, :] - df_array[None, :, :]
        self.dist_matrix = np.sqrt((df_diffs**2).sum(axis=2))

        self.N_POINTS=self.dist_matrix.shape[0]
        if self.VERBOSE:print("Number of Points",self.N_POINTS)
        if START_IDX == None: START_IDX=0
        if END_IDX == None: END_IDX=self.N_POINTS-1
        #______CALC OPTIMAL SKIP DISTANCE____________
        self.MAX_SKIP=int(np.floor(self.SKIP_FRAC*self.N_POINTS))
        self.SKIP_THRESHOLD = self.compute_skip_threshold()
        self.SKIP_PENALTY = self.SKIP_THRESHOLD  # penalty per skipped point (tweak as needed)
        print(f"Optimal SKIP_THRESHOLD for skipping {self.MAX_SKIP} distances: {self.SKIP_THRESHOLD:.3f}")

        # ─── DEAP SETUP ───────────────────────────────────────────────────────────
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("evaluate", self.evaluate_tsp)
        self.toolbox.register("mate",    self.cxFixedEndsSimpleSwap)
        self.toolbox.register("mutate",  self.mutFixedEndsShuffle, indpb=0.05)
        self.toolbox.register("select",  tools.selTournament, tournsize=3)
        # Individual: full route with fixed endpoints
        self.toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            lambda: [START_IDX]
                    + random.sample([i for i in range(self.N_POINTS) if i not in (START_IDX, END_IDX)], self.N_POINTS-2)
                    + [END_IDX]
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        #___________SETTING UP SORTING___________________
        random.seed(42)
        self.pop = self.toolbox.population(n=200)
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("min", np.min)
        self.stats.register("avg", np.mean)
        return

    def do_TSP_SORT(self,raw_df):
        self.pop, log = algorithms.eaSimple(self.pop, self.toolbox,
                                cxpb=0.7, mutpb=0.2,
                                ngen=40, stats=self.stats,
                                halloffame=self.hof, verbose=self.VERBOSE)

        best = self.hof[0]
        if self.VERBOSE:
            print("Best route:", best)
            print("Best fitness:", self.evaluate_tsp(best)[0])
        sorted_df = raw_df.iloc[best].reset_index(drop=True)

        return sorted_df

    def set_cluster_medians(self,cluster_medians):
        self.cluster_medians = cluster_medians

    def make_segment(self, df):

        print(df.size)
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


    def check_body_landmarks_visibility_for_duplicate(self, df_sorted, index):
        #function to check body landmarks visibility for dupes, ie if one of two images has legs and the other doesn't, cant be dupe. first pass for dupe detection
        print("checking body landmark visibility for duplicates at index", index)
        current_row = df_sorted.iloc[index]
        visibility_enc_1 = current_row.get('body_landmarks_normalized_visible_array', None)
        visibility_enc_1 = [round(value) for value in visibility_enc_1]
        count_of_comparisons = 10  # Number of rows to compare with the current row
        df_slice = df_sorted.iloc[index-1:index+count_of_comparisons]
        visibility_diffs = []
        for i, row in df_slice.iterrows():
            if i != index:
                visibility_enc_2 = row.get('body_landmarks_normalized_visible_array', None)
                visibility_enc_2 = [round(value) for value in visibility_enc_2]
                if (visibility_enc_1 is not None) and (visibility_enc_2 is not None):
                    visibility_diff = sum(abs(a - b) for a, b in zip(visibility_enc_1, visibility_enc_2))
                   
                    #highest diffs so far have been 4, going to have it ping for any higher
                    if visibility_diff > 4:
                        print("!!!HIGH VIS DIFF!!!", visibility_diff)
                    visibility_diffs.append(visibility_diff)
        return visibility_diffs

    def trim_bottom(self, img, site_name_id):
        # if self.VERBOSE: print("trimming bottom")
        if site_name_id == 2: trim = 100
        elif site_name_id == 9: trim = 90
        img = img[0:img.shape[0]-trim, 0:img.shape[1]]
        return img


    ###########################################################################
    ################### TENCH CODE FOR DEDUPE ANALYSIS ########################
    # CLAUDE LINK https://claude.ai/chat/15ff9f07-2387-4018-8005-49e44a9e1ad5 #
    ###########################################################################

    def calculate_ssim(self, image1, image2):
        image1 = cv2.resize(image1, self.TARGET_SIZE)
        image2 = cv2.resize(image2, self.TARGET_SIZE)
        """Calculate the SSIM between two images."""
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # Convert photo 1 to grayscale
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)  # Convert photo 2 to grayscale
        score, _ = ssim(gray1, gray2, full=True)
        return score  # Return the SSIM score


    def remove_duplicates(self, folder_list, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to remove duplicates from the dataframe.
        Returns sorted_df with all duplicates removed (keeping first occurrence).
        """
        to_remove = set()
        n_rows = len(df)

        # df = self.split_landmarks_to_columns_or_list(df, first_col="left_hand_world_landmarks", second_col="right_hand_world_landmarks", structure="list")

        if self.VERBOSE: print(f"Processing {n_rows} images for duplicate detection...")
        look_closer_dict = {}
        AUTO_CHECK = .2
        THRESH1 = 2
        SITE_NAME_ID_THRESH = 100000
        THRESHOLD = 0.85  # Similarity threshold to use for SSIM comparison .85 and above are all dupes. .75 has false positives
        # self.TARGET_SIZE = (128, 128)  # Size to temporarily resize images for comparison
        self.TARGET_SIZE = (384, 384)  # Size to temporarily resize images for comparison

        threshold_dict = {
            'wrist_ankle_landmarks_normalized_array': [.8, "less"],
            'face_xyz': [15, "less"],
            'bbox_array': [80, "less"],
            'face_encodings68': [.5, "less"],
            'hand_landmarks': [1, "less"],
            'body_landmarks_normalized_array': [1, "less"]

        }

        def norm_dist(this_dist, threshold):
            return this_dist / threshold

        def construct_filepath(key, df):
            filename = df.iloc[key]['imagename']
            site_name_id = df.iloc[key]['site_name_id']
            return os.path.join(folder_list[site_name_id], filename)

        def open_and_crop(key,df):
            img = cv2.imread(construct_filepath(key, df))
            dfrow = df.iloc[key]
            if dfrow["site_name_id"] in [2,9]: 
                img = self.trim_bottom(img, dfrow["site_name_id"])
            return img

        def sort_on_score(ssim_score, this_dist_list):
            sort_folder =""
            if ssim_score >= 0.99:
                sort_folder += "exactSSIM"
            elif ssim_score > 0.95:
                sort_folder += "ultraSSIM"
            elif ssim_score > 0.85:
                sort_folder += "highSSIM"
            elif ssim_score > 0.75:
                sort_folder += "mediumSSIM"
            elif ssim_score > 0.65:
                sort_folder += "ambiguousSSIM"
            elif ssim_score == 0:
                sort_folder += "noneSSIM"
            else:
                sort_folder += "lowSSIM"

            if sum(this_dist_list[:3]) == 0:
                sort_folder += "exactMETA"                
            elif sum(this_dist_list[:3]) < .5:
                sort_folder += "ultraMETA"                
            elif sum(this_dist_list[:3]) < 1:
                sort_folder += "highMETA"                
            elif sum(this_dist_list[:3]) < 2:
                sort_folder += "mediumMETA"                

            if this_dist_list[-3] > 15:
                sort_folder += "highHANDS"

            if this_dist_list[-1] > 1:
                sort_folder += "sameSHOOT"
            return sort_folder
        
        def save_dupes_look_closer(i, j, ssim_score, this_dist_list, df):
            # Treat True as 1 and False as 0 for summing
            sum_this_dist_list = round(sum(this_dist_list[:3]), 2)
            rounded_dist_list = [f"{x:.2f}" if isinstance(x, (float, int)) else str(x) for x in this_dist_list]
            foldername = f"{sum_this_dist_list}_{ssim_score:.2f}_{i}_{j}" + "_" + "_".join(rounded_dist_list)
            for key in [i,j]:
                filepath = construct_filepath(key, df)
                sort_folder = sort_on_score(ssim_score, this_dist_list)
                save_folder = os.path.join(self.outfolder, sort_folder, foldername)
                # print(f"Saving duplicate look closer images to {save_folder} for {filepath} from site {site_name_id}")
                # save the images to the folder
                os.makedirs(save_folder, exist_ok=True)
                shutil.copy(filepath, save_folder)

        # Pass 1: Fast screening, if it sum(norm_dist) < threshold, add it into the look_closer list to be checked in pass 2
        for i in range(n_rows):
            for j in range(i + 1, n_rows):
                this_dist_list = []
                for key, (threshold, operator) in threshold_dict.items():
                    if key not in ['wrist_ankle_landmarks_normalized_array', 'face_xyz', 'bbox_array']: continue
                    # calculate distance and add to list
                    this_dist = self.dupe_comparison(i, j, key, df)
                    normed_dist = norm_dist(this_dist, threshold)
                    this_dist_list.append(normed_dist)
                
                if sum(this_dist_list) <= THRESH1 or any(d < AUTO_CHECK for d in this_dist_list):
                    # if the normed dist is low enough, look closer
                    look_closer_dict[(i,j)] = this_dist_list
                    if self.VERBOSE: 
                        print(f"dupe_detection_pass_1 triggered at {i},{j}. with NORMED distances {this_dist_list}")
                else:
                    #print(f"tossing {i},{j}. with n_d {this_dist_list}")
                    
                    continue
                      # If only one of the checks passed, skip to next pair
        #pass 2
        keys_to_remove = []
        for (i,j), this_dist_list in look_closer_dict.items():
            key = "face_encodings68"
            threshold, _ = threshold_dict[key]
            this_dist = self.dupe_comparison(i, j, key, df)
            normed_dist = norm_dist(this_dist, threshold)
            if normed_dist < 1:
                this_dist_list.append(normed_dist)
                # update look_closer_dict
                look_closer_dict[(i,j)] = this_dist_list
                if self.VERBOSE: print(f"{normed_dist}   >>  so FACEMATCH {i},{j}. this_dist_list {this_dist_list}")
            else:
                if self.VERBOSE: print(f"{normed_dist} noface {i},{j}. this_dist_list {this_dist_list}")
                # mark for removal after iteration
                keys_to_remove.append((i,j))
                continue
        for key in keys_to_remove:
            look_closer_dict.pop(key, None)
        
        #pass 3 hand gesture, description, site name id
        for (i,j), this_dist_list in look_closer_dict.items():
            # calculate full body lms distance
            key = "body_landmarks_normalized_array"
            threshold, operator = threshold_dict[key]
            body_dist = self.dupe_comparison(i, j, key, df)
            normed_body_dist = norm_dist(body_dist, threshold)
            # replace the exising hands/ankles with full body lms
            this_dist_list[0] = normed_body_dist

            # calculate hand dist
            key = "hand_landmarks"
            threshold, operator = threshold_dict[key]
            hand_dist = self.dupe_comparison(i, j, key, df)
            normed_hand_dist = norm_dist(hand_dist, threshold)

            desc_dist = self.dupe_comparison_metadata(i, j, 'description', df)
            # 0 means diff sites, >0 means difference between ids for given site -- closer means same shoot
            # .5 means same site but ids contain strings, so can't compare
            site_name_id_dist = self.dupe_comparison_metadata(i, j, 'site_name_id', df)
            # print(f"site_name_id_dist for {i},{j} is {site_name_id_dist}")
            # normed_site_name_id_dist = norm_dist(site_name_id_dist, SITE_NAME_ID_THRESH)
            # if normed_site_name_id_dist < 5: normed_site_name_id_dist = 5

            # inverse relationship between distance and likelihood of duplicates
            if site_name_id_dist > 0:
                normed_site_name_id_dist = SITE_NAME_ID_THRESH / site_name_id_dist
            else:
                normed_site_name_id_dist = site_name_id_dist

            this_dist_list.extend([normed_hand_dist, desc_dist, normed_site_name_id_dist])
            look_closer_dict[(i,j)] = this_dist_list

        # print(f"  >>> look_closer_dict is {look_closer_dict}")
        if self.VERBOSE: print("--------------------------------dupe_detection analysis--------------------------------")


        for (i,j), this_dist_list in look_closer_dict.items():
            # if self.VERBOSE: print(f"  >>> closest_look {i},{j} with distances {this_dist_list}")
            image_i = open_and_crop(i, df)
            image_j = open_and_crop(j, df)
            if image_i is not None and image_j is not None:
 
                
                ssim_score = self.calculate_ssim(image_i, image_j)


                if round(image_i.shape[0]/image_i.shape[1], 2) == round(image_j.shape[0]/image_j.shape[1], 2):
                    image_resize_j = cv2.resize(image_j, (image_i.shape[1], image_i.shape[0]))
                    same_shape = True
                else:
                    image_resize_j = image_j
                    same_shape = False


                # image_resize_j = image_j
                 # prep for image background object
                get_background_mp = mp.solutions.selfie_segmentation
                get_bg_segment = get_background_mp.SelfieSegmentation()
                
                segmentation_mask_i = self.get_segmentation_mask(get_bg_segment, image_i, None, None)
                # selfie_bbox_i = self.get_selfie_bbox(segmentation_mask_i)
                # hue_i, sat_i, val_i, lum_i, lum_torso_i = self.get_bg_hue_lum(image_i, segmentation_mask_i, selfie_bbox_i)
                segmentation_mask_j = self.get_segmentation_mask(get_bg_segment, image_resize_j, None, None)
                # selfie_bbox_j = self.get_selfie_bbox(segmentation_mask_j)
                # hue_j, sat_j, val_j, lum_j, lum_torso_j = self.get_bg_hue_lum(image_resize_j, segmentation_mask_j, selfie_bbox_j)

                 #resize to match mask_i
                # segmentation_mask_j = cv2.resize(segmentation_mask_j, (segmentation_mask_i.shape[1], segmentation_mask_i.shape[0]))
                
                segmentation_mask_i = (segmentation_mask_i > 0.1).astype(np.uint8) * 255
                segmentation_mask_j = (segmentation_mask_j > 0.1).astype(np.uint8) * 255

                segmentation_mask_i = cv2.cvtColor(segmentation_mask_i, cv2.COLOR_GRAY2BGR)
                segmentation_mask_j = cv2.cvtColor(segmentation_mask_j, cv2.COLOR_GRAY2BGR)


                selfie_i = cv2.bitwise_and(image_i, segmentation_mask_i)
                selfie_j = cv2.bitwise_and(image_resize_j, segmentation_mask_j)
               
                bbox_i = df.at[i,"bbox"]
                bbox_j = df.at[j, "bbox"]
                print(f'image {i},{j} sizes: {image_i.size}, {image_resize_j.size}')
                bbox_i["bottom_extend"] = bbox_i["top"]+ ((bbox_i["bottom"]- bbox_i["top"])*2)
                bbox_j["bottom_extend"] = bbox_j["top"]+ ((bbox_j["bottom"]- bbox_j["top"])*2)

                selfie_i = selfie_i[bbox_i["top"]:bbox_i["bottom_extend"], bbox_i["left"]:bbox_i["right"]]
                if same_shape:
                    selfie_j = selfie_j[bbox_i["top"]:bbox_i["bottom_extend"], bbox_i["left"]:bbox_i["right"]]
                else:
                    print(f'error comparing segmentation masks, images {i},{j} different sizes, {selfie_i.shape}, {selfie_j.shape}')
                    selfie_j = selfie_j[bbox_j["top"]:bbox_j["bottom_extend"], bbox_j["left"]:bbox_j["right"]]
                    selfie_j = cv2.resize(selfie_j, (selfie_i.shape[1], selfie_i.shape[0]))
                    

                # selfie_j = cv2.resize(selfie_j, (selfie_i.shape[1], selfie_i.shape[0]))

                # cv2.imshow(f'segmentation_mask_i', selfie_i)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # cv2.imshow(f'segmentation_mask_j:', selfie_j)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                # Create diff mask
               
                ssim_score = self.calculate_ssim(selfie_i, selfie_j)


                # Display the mask_diff mask
                # selfie_mse_score, mask_diff = self.mse(selfie_i, selfie_j)
                # cv2.imshow(f'mask mask_diff at {i},{j}, scoring: {selfie_mse_score}', mask_diff)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()


            else:
                print(f"SSIM detect image {i} or image {j} is None")
                ssim_score = 0
                selfie_ssim_score = 0
            
            # save i and j to a shared folder
            if self.VERBOSE: save_dupes_look_closer(i, j, ssim_score, this_dist_list, df)

        # ADD all good j's to to_remove HERE !!!

        # Create sorted_df with all duplicates removed
        sorted_df = df.drop(df.index[list(to_remove)]).reset_index(drop=True)
        return sorted_df
    

    def dupe_comparison(self, row1, row2, column, df):
        """Compare two different rows in df for dupe detection"""
        enc1 = df.iloc[row1].get(column, None)
        enc2 = df.iloc[row2].get(column, None)

        if column == 'hand_landmarks':
            if type(enc1) == str:
                enc1 = json.loads(enc1)
            if type(enc2) == str:
                enc2 = json.loads(enc2)
            # print(f'hand landmark debugging after json load enc1, type: {type(enc1)}', enc1)
            # print(f'hand landmark debugging after json load enc2, type: {type(enc2)}', enc2)

        if enc1 is None or enc2 is None:
            if self.VERBOSE: print(f'remove_duplicates_{column} unable to process: {row1}, {row2}')
            return False
        
        # print(f"hand results: {df.iloc[row1].get('hand_results', None)}")
        # print(f"hand results: {df.iloc[row2].get('hand_results', None)}")
        try:
            distance = self.get_d(enc1, enc2)
            # print(f'dupe_detection col:{column}, row: {row1},{row2}, dist:{distance}')
            return distance
        except:
            # print(f'dupe_detect_{column}_type: {type(enc1)}, val{enc1}')
            return None

    def dupe_comparison_metadata(self, row1, row2, column, df):
        enc1 = df.iloc[row1].get(column, None)
        enc2 = df.iloc[row2].get(column, None)
        score = 1 # default to negative outcome
        if column == 'description':
            if enc1 is None or enc2 is None:
                if self.VERBOSE: print(f'remove_duplicates_{column} one {column} is None: {row1}, {row2}')
                if enc1 == enc2: score = .5 # halfway for a None None match
            elif enc1 == enc2: 
                score = 0
        elif column == 'site_name_id':
             if enc1 == enc2: 
                    try:
                        
                        id1 = df.iloc[row1].get('site_image_id', None)
                        id2 = df.iloc[row2].get('site_image_id', None)
                        # print(f"comparing site_image_id for {row1}, {row2}: {id1}, {id2}")
                        # assign diff between ids to score 
                        score = abs(id1 - id2)
                    except:
                        score = .5
             else:
                # different sites, likely different images
                score = 0 # default to negative outcome
        return score

    def check_metadata_for_duplicate(self, df_sorted, index):
        face_embeddings_distance, body_landmarks_distance, same_description, same_site_name_id = None, None, False, None
        print("checking metadata for duplicate at index", index)
        # Check for duplicate metadata in the DataFrame
        df_sorted['description'] = df_sorted['description'].apply(lambda x: x if pd.notna(x) else None)
        current_row = df_sorted.iloc[index]
        current_image_id = current_row.get('image_id', None)
        enc1 = current_row.get('face_encodings68', None)
    

        body_lms1 = current_row.get('body_landmarks_normalized_array', None)
        print("going to get ankles")
        wrist_ankle_lms1 = current_row.get('wrist_ankle_landmarks_normalized_array', None)
        print("got anklkes",wrist_ankle_lms1 )
        if current_row['description'] is not None: current_description = current_row['description'][:100] 
        else: current_description = "No description"
        if self.VERBOSE: print("this is the current row", current_row)
        # print("this is the current row", current_row)
        count_of_comparisons = 10  # Number of rows to compare with the current row
        df_slice = df_sorted.iloc[index-1:index+count_of_comparisons]
        #Nan to None for 'description' column
        # Iterate over a slice of df_sorted: 10 rows starting from 'index'
        # print("this is the slice of df_sorted", df_slice)
        for i, row in df_slice.iterrows():
            if i != index:
                row_image_id = row.get('image_id', None)
                enc2 = row.get('face_encodings68', None)
                body_lms2 = row.get('body_landmarks_normalized_array', None)
                wrist_ankle_lms2 = row.get('wrist_ankle_landmarks_normalized_array', None)
                if self.VERBOSE: print(f"comparing index {index} {current_image_id} to slice index {i} {row_image_id}")
                # print("row description is", row['description'])

                if row['description'] is not None: row_description = row['description'][:100]  # Limit to first 100 characters for display
                else: row_description = "No description"
                # print(current_description, row_description)
                # print(current_row['site_name_id'], row['site_name_id'])
               
                # Test dimension of each row
                if (enc1 is not None) and (enc2 is not None):
                    face_embeddings_distance = self.get_d(enc1, enc2)
                    if face_embeddings_distance < .4: print(f"face_embeddings_distance for index {index} and slice index {i}: {face_embeddings_distance}")
                if (body_lms1 is not None) and (body_lms2 is not None):
                    body_landmarks_distance = self.get_d(body_lms1, body_lms2)
                    wrist_ankle_landmarks_distance = self.get_d(wrist_ankle_lms1, wrist_ankle_lms2)
                    print(f"wrist ankle dist {wrist_ankle_landmarks_distance}")
                    if body_landmarks_distance < .2: print(f"body_landmarks_distance for index {index} and slice index {i}: {body_landmarks_distance}")
                if (current_description == row_description) and row_description != "No description": 
                    print(f"Duplicate found slice index {i}: {current_description} is a duplicate of {row_description}")
                    same_description = True

                if (current_row['site_name_id'] == row['site_name_id']): 
                    # this is hacky -- I'm checking to see how far apart the site_image_id is
                    # if they are from the same shoot and same upload, they will be close together
                    # these close together ones are likely to be subtly DIFFERENT images from the same shoot
                    # if they were uploaded at different times, they will be far apart
                    # 1000 is a random guess. It probabaly will be bigger. Maybe 10000 or more. 
                    if abs(current_row['site_image_id'] - row['site_image_id']) < 1000:
                        print(f"FALSE: Same site_image_id slice index {i}: {current_image_id} & {row_image_id} but far apart site_image_id {current_row['site_image_id']} & {row['site_image_id']}")
                        same_site_name_id = False
                    else:
                        if self.VERBOSE: print(f"SAME site_name_id slice index {i}: {current_image_id} & {row_image_id}")
                        same_site_name_id = True
                else:
                    if self.VERBOSE: print(f"Different site_name_id slice index {i}: {current_image_id} & {row_image_id}")
                    same_site_name_id = False
        return face_embeddings_distance, body_landmarks_distance, same_description, same_site_name_id


###########################################################################
################### END TENCH CODE FOR DUPE DETECTION #####################
###########################################################################

    def mse(self, img1, img2):
        h, w, _ = img1.shape
        try:
            diff = cv2.subtract(img1, img2)
            err = np.sum(diff**2)
            mse = err/(float(h*w))
        except Exception as e:
            print('failed mse')
            print('Error:', str(e))

        return mse, diff
    
    def unique_face(self,img1,img2):
        # convert the images to grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        if self.VERBOSE: print("img1 shape", img1.shape)
        if self.VERBOSE: print("img2 shape", img2.shape)
        # check if the images have the same dimensions
        if img1.shape != img2.shape:
            # find the most common dimension and assign that to all other dimensions
            common_dim = statistics.mode([img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1]])
            if img1.shape != (common_dim, common_dim): img1 = cv2.resize(img1, (common_dim, common_dim))
            if img2.shape != (common_dim, common_dim): img2 = cv2.resize(img2, (common_dim, common_dim))
        # define the function to compute MSE between two images
    

        error, diff = self.mse(img1, img2)
        
        return error

    def split_landmarks_to_columns_or_list(self, df, first_col="left_hand_world_landmarks", second_col="right_hand_world_landmarks", structure="cols"):
        # converts world landmarks for hand and body to list for knn. 
        # legacy code to convert hands to cols. not sure if it is used anymore.
        if "left_hand" in first_col and "right_hand" in second_col:
            destination_col = "hand_landmarks"
        elif first_col=="body_landmarks_3D" and second_col is None:
            destination_col = "body_landmarks_array"
        else: 
            print("split_landmarks_to_columns_or_list:", first_col, second_col, "not supported")
        # Extract and flatten landmarks for left and right hands
        first_landmarks = df[first_col].apply(self.extract_landmarks)
        if destination_col == "hand_landmarks": second_landmarks = df[second_col].apply(self.extract_landmarks)
        print("split_landmarks_to_columns_or_list first_landmarks", first_landmarks)
        if structure == "cols":
            if self.CLUSTER_TYPE == "fingertips_positions":
                col_num = len(self.SUBSET_LANDMARKS)
            else:
                col_num = 63
            # col_num = 
            # Create new columns for each dimension (21 points * 3 = 63 columns for each hand)
            first_landmark_cols = pd.DataFrame(first_landmarks.tolist(), columns=[f'left_dim_{i+1}' for i in range(col_num)])
            first_landmark_cols = pd.DataFrame(second_landmarks.tolist(), columns=[f'right_dim_{i+1}' for i in range(col_num)])
            
            # Concatenate the original DataFrame with the new columns
            df = pd.concat([df, first_landmark_cols, first_landmark_cols], axis=1)
        if structure == "list":
            # combine the left and right landmarks into a single list
            if not first_landmarks.all(): 
                print("first_landmarks is None")
            if destination_col == "hand_landmarks": 
                if not second_landmarks.all(): 
                    print("second_landmarks is None")
                landmarks_list = first_landmarks + second_landmarks
            else:
                landmarks_list = first_landmarks
            df[destination_col] = landmarks_list
        
        return df


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

        # set bbox dimensions
        img_h = self.h
        img_w = self.w
        bbox_x = self.bbox['left']
        bbox_y = self.bbox['top']
        bbox_w = self.bbox['right'] - self.bbox['left']
        bbox_h = self.bbox['bottom'] - self.bbox['top']
        # print("bboxxxxxxxxxxxxxxxxx",bbox_x,bbox_y,bbox_w,bbox_h)
        for idx, lm in enumerate(self.faceLms.landmark):
            # print("landmark",idx)
            if idx == point:
                # print("found point:")
                # print(idx)
                if self.VERBOSE: print("unprojected face lms",lm.x,lm.y)
                pointXY = (lm.x * img_w, lm.y * img_h)
                pointXY = (lm.x * bbox_w + bbox_x, lm.y * bbox_h + bbox_y)
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
        # print("face_top",self.ptop,"face_bottom",self.pbot)
        # print(self.pbot)
        self.face_height = self.dist(self.point(self.pbot), self.point(self.ptop))
        if self.VERBOSE: print("face_height", str(self.face_height))
        # return ptop, pbot, face_height



    def get_crop_data_scalable(self):
        # self.nose_2d accounts for self.NOSE_BRIDGE_DIST
        p1 = (int(self.nose_2d[0]), int(self.nose_2d[1]))
        
        toobig = False  # Default value
        width,height=self.w,self.h
        if self.VERBOSE: print("checkig boundaries")
        if self.VERBOSE: print("width",width,"height", height)
        if self.VERBOSE: print("nose_2d",p1)
        if self.VERBOSE: print("face_height",self.face_height)

        if not self.image_edge_multiplier[1] == self.image_edge_multiplier[3]:
            if self.VERBOSE: print("self.image_edge_multiplier left and right are not symmetrical breaking out", self.image_edge_multiplier[1], self.image_edge_multiplier[3])
            return
        topcrop = int(p1[1]-self.face_height*self.image_edge_multiplier[0])
        rightcrop = int(p1[0]+self.face_height*self.image_edge_multiplier[1])
        botcrop = int(p1[1]+self.face_height*self.image_edge_multiplier[2])
        leftcrop = int(p1[0]-self.face_height*self.image_edge_multiplier[3])
        self.simple_crop = [topcrop, rightcrop, botcrop, leftcrop]
        if self.VERBOSE: print("crop top, right, bot, left")
        if self.VERBOSE: print(self.simple_crop)

        # if topcrop >= 0 and width-rightcrop >= 0 and height-botcrop>= 0 and leftcrop>= 0:
        if any([topcrop < 0, width-rightcrop < 0, height-botcrop < 0, leftcrop < 0]):
            if self.VERBOSE: print("one is negative: topcrop",topcrop,"rightcrop",width-rightcrop,"botcrop",height-botcrop,"leftcrop",leftcrop)
            if self.VERBOSE: print("width",width,"height", height)
            toobig = True
            self.negmargin_count += 1
        else:
            if self.VERBOSE: print("all positive")
            toobig = False

        return toobig

    def calc_nose_bridge_dist(self, face_landmarks):
        """
        Calculate the vertical distance between landmark 1 and 6
        for a mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList.
        """
        if self.VERBOSE: print("calc_nose_bridge_dist, face_landmarks type", type(face_landmarks))
        nose_bridge_dist = None
        try:
            # Handle mediapipe NormalizedLandmarkList
            if hasattr(face_landmarks, "landmark") and len(face_landmarks.landmark) > 6:
                y1 = face_landmarks.landmark[1].y
                y6 = face_landmarks.landmark[6].y
                nose_bridge_dist = abs(y6 - y1)
                if self.VERBOSE: print("nose_bridge_dist is", nose_bridge_dist)
            # Fallback for list/tuple/dict (legacy)
            elif isinstance(face_landmarks, (list, tuple)) and len(face_landmarks) > 6:
                def get_y(lm):
                    if isinstance(lm, dict):
                        return lm.get('y', None)
                    elif isinstance(lm, (list, tuple)) and len(lm) > 1:
                        return lm[1]
                    else:
                        return None
                y1 = get_y(face_landmarks[1])
                y6 = get_y(face_landmarks[6])
                if y1 is not None and y6 is not None:
                    nose_bridge_dist = abs(y6 - y1)
                    if self.VERBOSE: print("nose_bridge_dist is", nose_bridge_dist)
        except Exception as e:
            print("Could not set nose_bridge_dist:", e)
            nose_bridge_dist = None
        return nose_bridge_dist

    def get_image_face_data(self,image, faceLms, bbox):

        if self.VERBOSE: print("faceLms type", type(faceLms))
        if self.VERBOSE: print("bbox type", type(bbox))
        self.image = image
        self.h = self.image.shape[0]
        self.w = self.image.shape[1]

        self.size = (self.image.shape[0], self.image.shape[1])        # if shape is not None:
        self.faceLms = faceLms
        self.bbox = (bbox)

        # print("get_image_face_data [-] size is", self.size)
        #I'm not sure the diff between nose_2d and p1. May be redundant.
        #it would prob be better to do this with a dict and a loop
        # Instead of hard-coding the index 1, you can use a variable or constant for the point index
        
        if self.this_nose_bridge_dist is None:
            try:
                # self.NOSE_BRIDGE_DIST
                self.nose_2d = self.get_face_2d_point(1)
                if self.VERBOSE: print("nose_2d",self.nose_2d)
                if self.ORIGIN == 6:
                    self.this_nose_bridge_dist = self.calc_nose_bridge_dist(faceLms)
                    if self.VERBOSE: print("NOSE_BRIDGE_DIST, self.this_nose_bridge_dist", self.NOSE_BRIDGE_DIST, self.this_nose_bridge_dist)
                    nose_delta = self.NOSE_BRIDGE_DIST - self.this_nose_bridge_dist
                    if self.VERBOSE: print("nose_delta", nose_delta)
                    self.nose_2d = (math.floor(self.nose_2d[0]), math.floor(self.nose_2d[1] + (nose_delta*self.face_height)))
                # cv2.circle(self.image, tuple(map(int, self.nose_2d)), 5, (0, 0,0), 5)
                if self.VERBOSE: print("get_image_face_data new nose_2d",self.nose_2d)
            except:
                print("couldn't get nose_2d via faceLms")
        else:
            pass
            if self.VERBOSE: print("self.this_nose_bridge_dist already set, skipping")


        try:
            if faceLms is not None and faceLms.landmark:
                if self.VERBOSE: print("get_image_face_data - faceLms is not None, using faceLms")
                # get self.face_height
                self.get_faceheight_data()
            else:
                print("get_image_face_data - NO faceLms, using bbox")
                # calculate face height based on bbox dimensions
                # TK I dont think this is accurate....????
                self.face_height = (self.bbox['bottom'] - self.bbox['top'])/2
        except:
            print(traceback.format_exc())
            print("couldn't get_faceheight_data")

            # this is the in progress neck rotation stuff
            # self.get_crop_data(sinY)

        
    def crop_whitespace(self, img):
        crop_dict = {
            8288: 3448,
            16576: 6898,
            24866: 10344,
            33154: 13792
        }
        cropped_img = None
        height, width, _ = img.shape
        if width not in crop_dict:
            print(f"Width {width} not found in crop_dict, returning original image.")
            return img
        crop_width = crop_dict.get(width, 0)
        # crop the image to the crop_width keeping the image centered
        CROP_LEFT = (width - crop_width) // 2
        CROP_RIGHT = width - CROP_LEFT - crop_width
        CROP_TOP = (height - crop_width) // 2
        CROP_BOTTOM = height - CROP_TOP - crop_width
        # Define the cropped area
        cropped_img = img[CROP_TOP:height - CROP_BOTTOM, CROP_LEFT:width - CROP_RIGHT]
        return cropped_img



    def expand_image(self,image, faceLms, bbox, sinY=0):
        # print("expand_image going to get self.face_height", str(self.face_height))
        self.get_image_face_data(image, faceLms, bbox)    
        # print("expand_image just got self.face_height", str(self.face_height))
        try:
            # print(type(self.image))
            borderType = cv2.BORDER_CONSTANT


            if self.USE_INCREMENTAL_RESIZE:
                resize_factor = math.ceil(self.face_height/self.resize_increment)
                if self.VERBOSE: print("expand_image resize_factor", str(resize_factor))
                face_incremental_output_size = resize_factor*self.resize_increment
                if self.VERBOSE: print("expand_image face_incremental_output_size", str(face_incremental_output_size))
                resize = face_incremental_output_size/self.face_height
                if self.VERBOSE: print("expand_image resize", str(resize))
            else:
                # scale image to match face heights
                resize_factor = None
                resize = self.face_height_output/self.face_height
                face_incremental_output_size = None
            if self.VERBOSE: print("expand_image resize", str(resize))
            if resize < 15:
                if self.VERBOSE: print("expand_image [-] resize", str(resize))
                # image.shape is height[0] and width[1]
                resize_dims = (int(self.image.shape[1]*resize),int(self.image.shape[0]*resize))
                # resize_nose.shape is  width[0] and height[1]
                resize_nose = (int(self.nose_2d[0]*resize),int(self.nose_2d[1]*resize))
                # resize_nose is scaled up nose position, but not expanded/cropped
                if self.VERBOSE: print(f"resize_factor {resize_factor} resize_dims {resize_dims}")
                # print("resize_nose", resize_nose)
                # this wants width and height

                # OLD WAY
                # resized_image = cv2.resize(self.image, resize_dims, interpolation=cv2.INTER_LINEAR)

                # NEW WAY
                upsized_image = self.upscale_model.upsample(self.image)
                resized_image = cv2.resize(upsized_image, (resize_dims))
                # for non-incremental, resized_image and resize_dims are scaled, but not expanded/cropped
                # resize dims is the size of the image before expanding/cropping

                if face_incremental_output_size:
                    image_incremental_output_ratio = face_incremental_output_size/self.face_height_output
                    if self.VERBOSE: print("face_incremental_output_size self.EXPAND_SIZE, face_incremental_output_size, self.face_height_output", self.EXPAND_SIZE, face_incremental_output_size, self.face_height_output)
                    this_expand_size = (int(self.EXPAND_SIZE[0]*image_incremental_output_ratio),int(self.EXPAND_SIZE[1]*image_incremental_output_ratio))
                    if self.VERBOSE: print("this_expand_size", image_incremental_output_ratio, this_expand_size)
                else:
                    this_expand_size = (self.EXPAND_SIZE[0],self.EXPAND_SIZE[1])
                    # for non-incremental, this_expand_size is now 2500


                # calculate boder size by comparing scaled image dimensions to EXPAND_SIZE
                # resize dims is the size of the image before expanding/cropping
                # resize_nose is scaled up nose position, but not expanded/cropped
                final_height_unit_from_nose = int(this_expand_size[1]/2) # 12500 default (or another set increment if incremental)
                final_width_unit_from_nose = int(this_expand_size[0]/2)
                existing_height = int(resize_dims[1]) # each image will be different
                existing_width = int(resize_dims[0])
                existing_pixels_above_nose = int(resize_nose[1])
                existing_pixels_below_nose = int(resize_dims[1]-existing_pixels_above_nose)
                existing_pixels_left_of_nose = int(resize_nose[0])
                existing_pixels_right_of_nose = int(resize_dims[0]-existing_pixels_left_of_nose)


                top_border = int(final_height_unit_from_nose - existing_pixels_above_nose)
                bottom_border = int(final_height_unit_from_nose - (existing_height-existing_pixels_above_nose))
                left_border = int(final_width_unit_from_nose - existing_pixels_left_of_nose)
                right_border = int(final_width_unit_from_nose - (existing_width-existing_pixels_left_of_nose))

                border_list = [top_border, bottom_border, left_border, right_border]
                existing_list = [existing_pixels_above_nose, existing_pixels_below_nose, existing_pixels_left_of_nose, existing_pixels_right_of_nose]
                if self.VERBOSE: print("borders to expand to", border_list)
                if self.VERBOSE: print("existing pixels", existing_list)
                # if all borders are positive, then we can expand the image
                expand_list =[]
                crop_list = []
                if all(x >= 0 for x in border_list):
                    expand_list = border_list
                    if self.VERBOSE: print("expand is good")
                # top_border = int(this_expand_size[1]/2 - resize_nose[1])
                # bottom_border = int(this_expand_size[1]/2 - (resize_dims[1]-resize_nose[1]))
                # left_border = int(this_expand_size[0]/2 - resize_nose[0])
                # right_border = int(this_expand_size[0]/2 - (resize_dims[0]-resize_nose[0]))

                # print([top_border, bottom_border, left_border, right_border])
                # print([top_border, resize_dims[0]/2-right_border, resize_dims[1]/2-bottom_border, left_border])
                # print([top_border, this_expand_size[0]/2-right_border, this_expand_size[1]/2-bottom_border, left_border])

                # OOOOOLD expand image with borders
                # if top_border >= 0 and right_border >= 0 and this_expand_size[0]/2-right_border >= 0 and bottom_border >= 0 and this_expand_size [1]/2-bottom_border>= 0 and left_border>= 0:
                # if topcrop >= 0 and self.w-rightcrop >= 0 and self.h-botcrop>= 0 and leftcrop>= 0:
                else:
                    for i, border in enumerate(border_list):
                        if border > 0:
                            expand_list.append(border)
                            crop_list.append(0)
                        else:
                            expand_list.append(0)
                            crop_list.append(abs(border))
                    if self.VERBOSE: print("expand failed, at least one is crop")
                    new_image = None
                    self.negmargin_count += 1                # self.preview_img(new_image)
                # quit()
                if self.VERBOSE: print("going to expand image with borders", expand_list)
                # Use border_list for the border sizes
                new_image = cv2.copyMakeBorder(
                    resized_image,
                    expand_list[0],  # top
                    expand_list[1],  # bottom
                    expand_list[2],  # left
                    expand_list[3],  # right
                    borderType,
                    None,
                    self.BGCOLOR
                )
                if self.VERBOSE: print("new_image shape after expanding:", new_image.shape)
                if any(x != 0 for x in crop_list):
                    if self.VERBOSE: print("some borders are negative, cropping")
                    # crop the image to the crop_list keeping the image centered
                    # start at top and left with the extra pixels the image extends beyond the output dimensions
                    # then go from there the distance of the output dimensions (final_height_unit_from_nose * 2, etc)
                    CROP_TOP = crop_list[0] 
                    CROP_BOTTOM = crop_list[0] + this_expand_size[1]
                    CROP_LEFT = crop_list[2]
                    CROP_RIGHT = crop_list[2] + this_expand_size[0] 
                    # CROP_LEFT = (new_image.shape[1] - crop_list[2]) // 2
                    # CROP_RIGHT = new_image.shape[1] - CROP_LEFT - crop_list[2]
                    # CROP_TOP = (new_image.shape[0] - crop_list[0]) // 2
                    # CROP_BOTTOM = new_image.shape[0] - CROP_TOP - crop_list[0]
                    if self.VERBOSE: print("CROP_LEFT, CROP_RIGHT, CROP_TOP, CROP_BOTTOM", CROP_LEFT, CROP_RIGHT, CROP_TOP, CROP_BOTTOM)
                    # Define the cropped area
                    new_image = new_image[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT]
                    if self.VERBOSE: print("cropped image to", crop_list)
            else:
                new_image = None
                if self.VERBOSE: print("failed expand loop, with resize", str(resize), "too big, skipping")


        except Exception as e:
            print("not expand_image loop failed")
            print('Error:', str(e))
            sys.exit(1)

        # add cropdown here

        # quit() 
        return new_image

    def get_extension_pixels(self,image):
        
        if self.VERBOSE: print("calculating extension pixels")
        p1 = (int(self.nose_2d[0]), int(self.nose_2d[1]))
        print("get_extension_pixels nose_2d",self.nose_2d)
        topcrop = int(p1[1]-self.face_height*self.MAX_IMAGE_EDGE_MULTIPLIER[0])
        rightcrop = int(p1[0]+self.face_height*self.MAX_IMAGE_EDGE_MULTIPLIER[1])
        botcrop = int(p1[1]+self.face_height*self.MAX_IMAGE_EDGE_MULTIPLIER[2])
        leftcrop = int(p1[0]-self.face_height*self.MAX_IMAGE_EDGE_MULTIPLIER[3])
        ext_crop = np.array([topcrop, self.w-rightcrop, self.h-botcrop, leftcrop])
        ext_crop=np.abs(np.minimum(np.zeros(4),ext_crop)).astype(int)
        # simple_crop=np.array([np.maximum(0,topcrop),np.minimum(self.w,rightcrop),np.minimum(self.h,botcrop),np.maximum(0,leftcrop)]).astype(int)
        # simple_crop={"top":simple_crop[0],"right":simple_crop[1],"bottom":simple_crop[2],"left":simple_crop[3]}
        extension_pixels={"top":ext_crop[0],"right":ext_crop[1],"bottom":ext_crop[2],"left":ext_crop[3]}
        #     print("simple crop",simple_crop)
        # simple_crop_image=image[simple_crop["top"]:simple_crop["bottom"],simple_crop["left"]:simple_crop["right"],:]
        if self.VERBOSE: print("extension pixels calculated")
        return extension_pixels

    def define_corners(self,extension_pixels,shape):
        # access results via corners["top_left"][0][0]
        # cornermask[corners["top_left"][0][0]:corners["top_left"][0][1],corners["top_left"][1][0]:corners["top_left"][1][1]] = 255
        top, bottom, left, right = extension_pixels["top"], extension_pixels["bottom"], extension_pixels["left"],extension_pixels["right"] 
        height, width = shape[:2]
        corners = {}
        # top left corner
        if top and not left: corners["top_left"] = [0,top],[0,top]
        elif top and left: corners["top_left"] = [0,top],[0,left]
        elif not top and left: corners["top_left"] = [0,left],[0,left]
        else: corners["top_left"] = [0,0],[0,0]
        # top right corner
        if top and not right: corners["top_right"] = [0,top],[width+left-top,width+left]
        elif top and right: corners["top_right"] = [0,top],[width+left,width+left+right]
        elif not top and right: corners["top_right"] = [0,right],[width+left,width+left+right]
        else: corners["top_right"] = [0,0],[width+left,width+left]
        # bottom left corner
        if bottom and not left: corners["bottom_left"] = [height+top,height+top+bottom],[0,bottom]
        elif bottom and left: corners["bottom_left"] = [height+top,height+top+bottom],[0,left]
        elif not bottom and left: corners["bottom_left"] = [height+top-left,height+top],[0,left]
        else: corners["bottom_left"] = [height+top,height+top],[0,0]
        # bottom right corner
        if bottom and not right: corners["bottom_right"] = [height+top,height+top+bottom],[width+left-bottom,width+left]
        elif bottom and right: corners["bottom_right"] = [height+top,height+top+bottom],[width+left,width+left+right]
        elif not bottom and right: corners["bottom_right"] = [height+top-right,height+top],[width+left,width+left+right]
        else: corners["bottom_right"] = [height+top,height+top],[width+left,width+left]

        return corners

    def test_consistency(self,img, area, threshold=20):
        # takes area matrix, and segments the image
        grid = img[area[0][0]:area[0][1],area[1][0]:area[1][1], :]
        
        pixels = grid.reshape(-1, 3) # Reshape the strip to a 2D array of pixels
        # mean_color = np.mean(pixels, axis=0)
        std_dev = np.std(pixels, axis=0)
        overall_std_dev = np.mean(std_dev)
        is_consistent = overall_std_dev < threshold
        # if not is_consistent:
            
        #     cv2.imshow("grid",grid)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        return is_consistent

    def prepare_mask(self,image,extension_pixels,color):
        if self.VERBOSE:print("starting mask preparation")
        height, width = image.shape[:2]
        top, bottom, left, right = extension_pixels["top"], extension_pixels["bottom"], extension_pixels["left"],extension_pixels["right"] 
        if color == "white":
            print("color is white")
            extended_img = np.ones((height + top+bottom, width+left+right, 3), dtype=np.uint8) * 255

        if color == "black":
            print("color is black")
            extended_img = np.zeros((height + top+bottom, width+left+right, 3), dtype=np.uint8)
        extended_img[top:height+top, left:width+left,:] = image

        # main mask
        #     mask = np.zeros_like(extended_img[:, :, 0])
        #     mask = np.ones_like(extended_img[:, :, 0]) * 255
        # else:
        mask = np.zeros_like(extended_img[:, :, 0])
        mask[:top,:] = 255
        mask[:,:left] = 255
        mask[(height+top):,:] = 255
        mask[:,(width+left):] = 255
        if self.VERBOSE:print("mask preparation done")

        # corner mask for second CV2 inpaint
        cornermask = np.zeros_like(extended_img[:, :, 0])
        corners = self.define_corners(extension_pixels,image.shape)
        cornermask[corners["top_left"][0][0]:corners["top_left"][0][1],corners["top_left"][1][0]:corners["top_left"][1][1]] = 255
        cornermask[corners["top_right"][0][0]:corners["top_right"][0][1],corners["top_right"][1][0]:corners["top_right"][1][1]] = 255
        cornermask[corners["bottom_left"][0][0]:corners["bottom_left"][0][1],corners["bottom_left"][1][0]:corners["bottom_left"][1][1]] = 255
        cornermask[corners["bottom_right"][0][0]:corners["bottom_right"][0][1],corners["bottom_right"][1][0]:corners["bottom_right"][1][1]] = 255

        return extended_img,mask, cornermask
    
    def extend_lama(self,extended_img, mask,downsampling_scale=1):
        if self.VERBOSE: print("doing lama generative fill")
        def kludge(dimension, dim):
            if dim == "w": factor = 1.034
            if dim == "h": factor = 1.027
            new_dim = int(dimension*factor)
            return new_dim
        n_height,n_width=extended_img.shape[:2]
        extended_img = cv2.resize(extended_img, (n_width//downsampling_scale, n_height//downsampling_scale), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask, (n_width//downsampling_scale, n_height//downsampling_scale), interpolation = cv2.INTER_AREA)
        inpaint = self.INPAINT_MODEL(extended_img, mask)
        inpaint=np.array(inpaint,dtype=np.uint8)
        inpaint = cv2.resize(inpaint, (kludge(n_width,"w"),kludge(n_height,"h")), interpolation = cv2.INTER_LANCZOS4)
        if self.VERBOSE: print("generative fill done")

        return inpaint

    def set_upscale_model(self,UPSCALE_MODEL_PATH):
        print("model_path",UPSCALE_MODEL_PATH)
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(UPSCALE_MODEL_PATH)
        sr.setModel("fsrcnn", 4) 
        return sr

    def crop_image(self,image, faceLms, bbox, sinY=0,SAVE=False):
        print("crop_image, going to get_image_face_data")
        self.get_image_face_data(image, faceLms, bbox) 
        is_inpaint = False
        cropped_image = None
        # check for crop, and if not exist, then get
        # if not hasattr(self, 'crop'): 
        try:
            toobig = self.get_crop_data_scalable()
        except:
            print("couldn't get crop data")
            toobig = True


        if not toobig:
            if self.VERBOSE: print("crop_image is GOOD: going to crop")
            # print (self.padding_points)
            #set main points for drawing/cropping

            #moved this back up so it would NOT     draw map on both sets of images
            try:
                if self.VERBOSE: print(type(self.image))
                # image_arr = numpy.array(self.image)
                # print(type(image_arr))
                cropped_actualsize_image = self.image[self.simple_crop[0]:self.simple_crop[2], self.simple_crop[3]:self.simple_crop[1]]
                # self.preview_img(cropped_actualsize_image)
                if self.VERBOSE: print("cropped_actualsize_image.shape", cropped_actualsize_image.shape)
                resize = self.output_dims[0]/cropped_actualsize_image.shape[0] 
                if self.VERBOSE: print("resize", resize)
                if resize > self.resize_max:
                    if self.VERBOSE: print("toosmall, returning None ")
                    self.toosmall_count += 1
                    return None, is_inpaint
                if self.VERBOSE: print("about to resize")
                # crop[0] is top, and clockwise from there. Right is 1, Bottom is 2, Left is 3. 
                if self.VERBOSE: print("output dims", self.output_dims)
                #####UPSCALING#######
                
                upsized_image = self.upscale_model.upsample(cropped_actualsize_image)
                cropped_image = cv2.resize(upsized_image, (self.output_dims))
                # print("UPSCALING DONEEEEEEEEEEEEEEEEEEEEEEE")
                ####################
                # cropped_image = cv2.resize(cropped_actualsize_image, (self.output_dims), interpolation=cv2.INTER_LINEAR)
                if self.VERBOSE: print("image actually cropped")
            except:
                cropped_image = None
                print("not cropped_image loop", self.h, self.w)
        else:
            # cropped_image = np.array([-1])
            print("crop_image: cropped_image is None because too big is ", toobig)
            cropped_image = None
            is_inpaint = True
            # resize = None
        return cropped_image, is_inpaint

    def get_bg_hue_lum(self,image,segmentation_mask,bbox):
        if type(bbox)==str:
            bbox=json.loads(bbox)
   
        if self.VERBOSE: print("[get_bg_hue_lum] bbox is",bbox)
        # expects image in RGB format

        if isinstance(bbox, str):
            # catching any str bbox that slipped through
            try:
                bbox = json.loads(bbox)
            except json.JSONDecodeError:
                print("Error: bbox is a string but not a valid JSON")
                return None  # or handle this error appropriately

        
        mask=np.repeat((1-segmentation_mask)[:, :, np.newaxis], 3, axis=2) 
        if self.VERBOSE: print("[get_bg_hue_lum] made mask")
        mask_torso=np.repeat((segmentation_mask)[:, :, np.newaxis], 3, axis=2) 
        if self.VERBOSE: print("[get_bg_hue_lum] made torso mask")

        if self.VERBOSE: print("[get_bg_hue_lum] doing some stuff")
        masked_img=mask*image[:,:,::-1]/255 ##RGB format
        masked_img_torso=mask_torso*image[:,:,::-1]/255 ##RGB format

        if self.VERBOSE: print("[get_bg_hue_lum] about to make bk px mask")
        # Identify black pixels where R=0, G=0, B=0
        black_pixels_mask = np.all(masked_img == [0, 0, 0], axis=-1)
        black_pixels_mask_torso = np.all(masked_img_torso == [0, 0, 0], axis=-1)

        # Filter out black pixels and compute the mean color of the remaining pixels
        mean_color = np.mean(masked_img[~black_pixels_mask], axis=0)[np.newaxis,np.newaxis,:] # ~ means negate/remove
        self.hue = cv2.cvtColor(mean_color, cv2.COLOR_RGB2HSV)[0,0,0]
        self.sat = cv2.cvtColor(mean_color, cv2.COLOR_RGB2HSV)[0,0,1]
        self.val = cv2.cvtColor(mean_color, cv2.COLOR_RGB2HSV)[0,0,2]
        self.lum = cv2.cvtColor(mean_color, cv2.COLOR_RGB2LAB)[0,0,0]
        if self.VERBOSE: print("hue, sat, val, lum", self.hue, self.sat, self.val, self.lum)
        if self.VERBOSE: print("NOTmasked_img_torso size", masked_img_torso.shape, black_pixels_mask_torso.shape)
        if bbox :
            # SJ something is broken in here. It returns an all black image which produces a lum of 100
            masked_img_torso = masked_img_torso[bbox["bottom"]:]
            black_pixels_mask_torso = black_pixels_mask_torso[bbox["bottom"]:]
        # else:
        #     print("YIKES! no bbox. Here's a hacky hack to crop to the bottom 20%")
        #     bottom_fraction = masked_img_torso.shape[0] // 5
        #     masked_img_torso = masked_img_torso[-bottom_fraction:]
        #     black_pixels_mask_torso = black_pixels_mask_torso[-bottom_fraction:]

        if self.VERBOSE: print("masked_img_torso size", masked_img_torso.shape, black_pixels_mask_torso.shape)
        mean_color = np.mean(masked_img_torso[~black_pixels_mask_torso], axis=0)[np.newaxis,np.newaxis,:] # ~ is negate
        self.lum_torso=cv2.cvtColor(mean_color, cv2.COLOR_RGB2LAB)[0,0,0]

        if self.VERBOSE: print("HSV, lum", self.hue,self.sat,self.val,self.lum, self.lum_torso)
        return self.hue,self.sat,self.val,self.lum,self.lum_torso
    

    def most_common_row(self, flattened_array):
        # Convert the flattened array into tuples for hashing
        hashable_rows = [tuple(row) for row in flattened_array]

        # Find the mode using the most_common function from the collections module
        counter = Counter(hashable_rows)
        # most_common_row = counter.most_common(1)[0][0]
        most_common = counter.most_common(1)
        if most_common and most_common[0][1] > 1:
            most_common_row = most_common[0][0]
        else:
            most_common_row = None  # or handle the no-mode case however you like
        print("Most common row in list:", most_common_row)
        return most_common_row
    
    def safe_round(self,x, decimals=1):
        if x is None:
            return None
        try:
            return np.round(x, decimals)
        except:
            return None

    def json_to_list(self,row):  
        if type(row) == dict:
            bbox = row    
        else:
            bbox = row["bbox_"+str(self.OBJ_CLS_ID)]
        if bbox: return [bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]]
        else: return None

    def smart_round_df(self, df_enc, col_to_round, round_down):
        df_rounded = pd.DataFrame()
        digits = int(math.log10(len(df_enc)))
        if round_down > (digits-1):
            return None
        if digits < 3: digits = digits-round_down
        else: digits = 1
        print("digits", digits)
        # drop all rows where all values in the row are 0.0
        df_enc = df_enc[df_enc[col_to_round].apply(lambda x: any(abs(val) != 0.0 for val in x))]
        print("df_enc", df_enc.head())
        # Round each value in the target column to the number of digits in the df
        # 300 rows = -2 decimal places (hundreds)
        # 3000 rows = -3 decimal places (thousands)       
        df_rounded[col_to_round] = df_enc[col_to_round].apply(lambda x: self.safe_round(x, digits))
            # drop all rows where bbox_col is None
        df_rounded = df_rounded.dropna(subset=[col_to_round])
        print("df_rounded", df_rounded.head())
        return df_rounded

    def get_start_obj_bbox(self, start_img, df_enc):
        enc1 = None
        if start_img == "median":
            print("[get_start_obj_bbox] in median", df_enc)
            sort_column = "obj_bbox_list"
            # df_rounded = self.smart_round_df(df_enc, bbox_col)
            round_down = 1
            while enc1 == None:
                # this should loop through and round the values down until it finds a mode
                df_rounded = self.smart_round_df(df_enc, sort_column, round_down)
                enc1, round_down = self.test_smart_round_df(df_rounded, df_enc, sort_column, round_down)

            # digits = (int(math.log10(len(df_enc)))+1)*-1
            # df_enc = df_enc[df_enc[bbox_col].apply(lambda x: any(val != 0.0 for val in x))]
            # df_rounded = pd.DataFrame()
            # # Round each value in the face_encodings68 column to -2 decimal places (hundreds)       
            # df_rounded[bbox_col] = df_enc[bbox_col].apply(lambda x: self.safe_round(x, digits))


            # # Convert the face_encodings68 column to a list of lists
            # flattened_array = df_rounded[bbox_col].tolist()        
            # print("flattened_array", flattened_array)    
            # try:
            #     enc1 = self.most_common_row(flattened_array)
            # except:
            #     enc1 = random.choice(flattened_array)
            print("get_start_obj_bbox", enc1)
            return enc1
        elif start_img == "start_bbox":
            print("starting from start_bbox")
            print(self.counter_dict["start_site_image_id"])
            enc1 = self.counter_dict["start_site_image_id"]
            # enc1 = df_enc.loc[self.counter_dict["start_site_image_id"]].to_list()
            return enc1
        elif start_img == "start_image_id":
            print("starting from start_bbox")
            # print(self.counter_dict["start_site_image_id"])
            sort_column = "bbox_"+str(self.OBJ_CLS_ID)
            enc1_image_id = self.counter_dict["start_site_image_id"]
            print("enc1_image_id", enc1_image_id)
            enc1 = df_enc.loc[df_enc['image_id'] == enc1_image_id, sort_column].values[0]            
            # enc1 = self.counter_dict["start_site_image_id"]
            print("enc1 phone bbox set from sort_column", enc1)
            # enc1 = df_enc.loc[self.counter_dict["start_site_image_id"]].to_list()
            return enc1
        else:
            print("[get_start_obj_bbox] - not median")
            return None

    def test_smart_round_df(self,df_rounded, df_enc, sort_column, round_down):
        # if it doesn't find a mode, it will return None
        # then it will pick a random value from the flattened_array
        if df_rounded is not None:
            flattened_array = df_rounded[sort_column].tolist()
            enc1 = self.most_common_row(flattened_array)
            print("get_start_enc_NN most_common_row", enc1)
            round_down += 1
        elif df_rounded is None or len(df_rounded) == 1:
            # pick a random enc2 from flattened_array
            print("get_start_enc_NN - no df_rounded")
            flattened_array = df_enc[sort_column].tolist()
            enc1 = random.choice(flattened_array)
        else:
            print("get_start_enc_NN - no df_rounded")
            enc1 = None
        return enc1, round_down

    def get_start_enc_NN(self, start_img, df_enc):
        print("get_start_enc, for self.SORT_TYPE", self.SORT_TYPE)
        print("first row of df_enc", df_enc.iloc[0])
        enc1 = None
        if self.SORT_TYPE == "128d": sort_column = "face_encodings68"
        elif self.SORT_TYPE == "planar_body": sort_column = "body_landmarks_array"
        elif self.SORT_TYPE == "body3D": sort_column = "body_landmarks_array"
        elif self.SORT_TYPE == "planar_hands": sort_column = "hand_landmarks" # hand_landmarks are left and right hands flat list of 126 values
        print("sort_column", sort_column)
        print("sort_column head", df_enc[sort_column].head())
        # print("lengtth of first value", len(df_enc[sort_column].iloc[0]))
        if start_img == "median" or start_img == "start_bbox":
            # when I want to start from start_bbox, I pass it a median 128d enc
            print("in median")
            print("df_enc", df_enc)
            # Round each value in the face_encodings68 column to 2 decimal places            
            # df_enc['face_encodings68'] = df_enc['face_encodings68'].apply(self.safe_round)
            # df_enc[sort_column] = df_enc[sort_column].apply(lambda x: np.round(x, 1))

            # # Convert the face_encodings68 column to a list of lists
            # flattened_array = df_enc[sort_column].tolist()     
            # # round each value in flattened_array to 2 decimal places
            # flattened_array = [np.round(x, 1) for x in flattened_array]
            round_down = 1
            while enc1 == None:
                # this should loop through and round the values down until it finds a mode
                df_rounded = self.smart_round_df(df_enc, sort_column, round_down)
                enc1, round_down = self.test_smart_round_df(df_rounded, df_enc, sort_column, round_down)
            # print(dfmode)
            # enc1 = dfmode.iloc[0].to_list()
            # enc1 = df_128_enc.median().to_list()

        elif start_img == "start_image_id":
            print("start_image_id (this is what we are comparing to)")
            # set enc1 = df_enc value in the self.SORT_TYPE column, for the row where column image_id = self.counter_dict["start_site_image_id"]
            # enc1 = df_enc.loc[df_enc['image_id'] == self.counter_dict["start_site_image_id"], sort_column].to_list()
            # enc1 = df_enc.loc[df_enc['image_id'] == self.counter_dict["start_site_image_id"], sort_column].to_list()
            enc1_image_id = self.counter_dict["start_site_image_id"]
            print("enc1_image_id", enc1_image_id)
            try:
                enc1 = df_enc.loc[df_enc['image_id'] == enc1_image_id, sort_column].values[0]
                print("enc1 set from sort_column", sort_column, enc1)
                # if self.SORT_TYPE == "planar_body":
                #     enc1 = self.get_landmarks_2d(enc1, self.SUBSET_LANDMARKS, "list")
            except:
                print("get_start_enc_NN - no enc1_image_id")
                quit()
                # # pick a random enc2 from flattened_array
                # flattened_array = df_enc[sort_column].tolist()
                # enc1 = random.choice(flattened_array)
# TK needs to be refactored NN June 8

        elif start_img == "start_site_image_id":
            print("start_site_image_id (this is what we are comparing to)")
            # print(start_site_image_id)
            print(self.counter_dict["start_site_image_id"])
            enc1 = df_128_enc.loc[self.counter_dict["start_site_image_id"]].to_list()
        elif start_img == "start_face_encodings":
            print("starting from start_face_encodings")
            print("self.counter_dict", self.counter_dict)
            enc1 = self.counter_dict["start_site_image_id"]
            print("start_face_encodings", enc1)
        elif self.SORT_TYPE == "planar_body":
            # THIS MAY NOT BE OPERANT
            # print("get_start_enc planar_body start_img key is (this is what we are comparing to):")
            # print(start_img)
            try:
                # enc1 = df_33_lms.loc[start_img].to_list()
                # TK 
                enc1 = self.get_landmarks_2d(df_33_lms.loc[start_img, "body_landmarks"], self.BODY_LMS)

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
        return enc1




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

    def choose_hand(self, landmark_list, hand):
        right = []
        left = []
        for idx, lm in enumerate(landmark_list):
            print("idx", idx)
            if idx % 2 == 0:
                print("even appended right", idx)
                right.append(lm)
            else:
                left.append(lm)
        if hand == "right":
            return right
        elif hand == "left":
            return left

    def make_subset_landmarks(self, first, last, dim=2):
        # takes number of lms and multiplies them by dim
        # add 1 to last to make it inclusive of final landmark
        if self.VERBOSE: print("am I using 3D?", self.use_3D)
        if self.use_3D == True: dim = 3
        subset = list(range(first*dim, (last+1)*dim))
        if self.VERBOSE: print(f"first {first} last {last} subset {subset}")
        return subset
                
    def get_landmarks_2d(self, Lms, selected_Lms, structure="dict"):
        def append_lms(idx, x,y,z, structure, Lms2d, Lms1d, Lms1d3):
            if structure == "dict":
                Lms2d[idx] =([x, y])
            elif structure == "list":
                Lms1d.append(x)
                Lms1d.append(y)
            elif structure == "list3":
                Lms1d3.append(x)
                Lms1d3.append(y)
                Lms1d3.append(z)
            # return Lms2d, Lms1d, Lms1d3

        # print("get_landmarks_2d", selected_Lms)
        # this is redundantly in io also
        Lms2d = {}
        Lms1d = []
        Lms1d3 = []

        if self.VERBOSE: print(type(Lms))
        if type(Lms) is list and len(Lms) > 0 and isinstance(Lms[0], (list, tuple)) and len(Lms[0]) == 3:
            print("lms is a list of xyz lists")
            for idx, lm in enumerate(Lms):
                # even_lms = only even values from selected_Lms 
                # converting back from xy values, to just the vertex number
                even_lms = [lm/2 for lm in selected_Lms if lm % 2 == 0]
                if idx in even_lms:                
                    x, y, z = Lms[idx]
                    append_lms(idx, x,y,z, structure, Lms2d, Lms1d, Lms1d3)
        if type(Lms) is list and len(Lms)==33:
            # handle erronous new API 3D body landmarks, 
            if self.VERBOSE: print("lms is a list of 33 landmarks")
            if self.VERBOSE: print("Lms", Lms)
            for idx, lm in enumerate(Lms):
                if idx in selected_Lms:
                    # If lm is a Landmark object, access attributes directly
                    if hasattr(lm, "x") and hasattr(lm, "y"):
                        x = lm.x
                        y = lm.y
                        z = getattr(lm, "z", 0.0)
                        visibility = getattr(lm, "visibility", 0.0)
                    else:
                        # fallback for tuple/list
                        x = lm[0]
                        y = lm[1]
                        z = lm[2] if len(lm) > 2 else 0.0
                        visibility = lm[3] if len(lm) > 3 else 0.0
                    if structure == "dict":
                        Lms2d[idx] =([x, y])
                    elif structure == "list":
                        Lms1d.append(x)
                        Lms1d.append(y)
                    elif structure == "list3":
                        Lms1d3.append(x)
                        Lms1d3.append(y)
                        Lms1d3.append(z)
                    elif structure == "list3D":
                        Lms1d3.append(x)
                        Lms1d3.append(y)
                        Lms1d3.append(z)
                        Lms1d3.append(visibility)
        elif hasattr(Lms, 'landmark') and len(Lms.landmark) > 0:
            for idx, lm in enumerate(Lms.landmark):
                if idx in selected_Lms:
                    # print("idx", idx)
                    # x, y = int(lm.x * img_w), int(lm.y * img_h)
                    # print("lm.x, lm.y", lm.x, lm.y)
                    if structure == "dict":
                        Lms2d[idx] =([lm.x, lm.y])
                    elif structure == "list":
                        Lms1d.append(lm.x)
                        Lms1d.append(lm.y)
                    elif structure == "list3":
                        Lms1d3.append(lm.x)
                        Lms1d3.append(lm.y)
                        Lms1d3.append(lm.visibility)
                    elif structure == "list3D":
                        Lms1d3.append(lm.x)
                        Lms1d3.append(lm.y)
                        Lms1d3.append(lm.z)
                        Lms1d3.append(lm.visibility)
        else:
            print("get_landmarks_2d: Lms is not a list or has no landmark attribute")
            print("Lms", Lms)
            print("selected_Lms", selected_Lms)
            return None
        # print("Lms2d", Lms2d)
        # print("Lms1d", Lms1d)
        # print("Lms1d3", Lms1d3)

        if Lms1d:
            return Lms1d
        elif Lms1d3:
            return Lms1d3
        else:
            return Lms2d


    def normalize_hsv(self, hsv1, df):
        def scale_hue(hsv1,hsv2):
            # hue is between 0-360 degrees (which is scaled to 0-1 here)
            # hue is a circle, so we need to go both ways around the circle and pick the smaller one
            dist_reg = abs(hsv1[0] - hsv2[0])
            dist_offset = abs((hsv1[0]) + (1-hsv2[0]))
            hsv2[0] = min(dist_reg, dist_offset)
            # this is where I will curve the sat data

            return hsv2
        
        # scale hue for each row in the hsv column and assign to a new hsvv column
        # combine the HSV values with the lum and lum_torso values 
        df["hsvll"] = df["hsv"].apply(lambda x: scale_hue(hsv1, x)) + df["lum"]

        return df

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



    def get_enc1(self, df, FIRST_ROUND=False, hsv_sort = False):
        enc1 = None
        obj_bbox1 = None
        if FIRST_ROUND:
            this_start = self.counter_dict["start_img_name"]
            ## Get the starting encodings (if not passed through)

            # this catches cluster_no, but the value is all 33 lms
            # if self.counter_dict["cluster_no"]:
            #     # this is the first round for a cluster, so set from median
            #     enc1 = self.cluster_medians[self.counter_dict["cluster_no"]]
            #     print("set enc1 from cluster median")
            #     print("enc1", enc1)
                
            if this_start not in ["median", "start_site_image_id", "start_image_id", "start_face_encodings", "start_bbox"]:
            # elif this_start not in ["median", "start_site_image_id", "start_face_encodings", "start_bbox"]:
                # this is the first round for clusters/itter where last_image_enc is true
                # set encodings to the passed through encodings
                # IF NO START IMAGE SPECIFIED (this line works for no clusters)
                print("attempting set enc1 from pass through")
                enc1 = self.counter_dict["last_image_enc"]
                print("set enc1 from pass through")
            elif hsv_sort == True:
                print("setting enc1 to HSV")
                # set enc1 to median
                enc1 = [0,0,1,1,.5]
            else:
                #this is the first??? round, set via df, defaults to face_encodings68
                print(f"trying get_start_enc() from {this_start} which corresponds to {self.counter_dict}")
                enc1 = self.get_start_enc_NN(this_start, df)
                if self.OBJ_CLS_ID > 0: 
                    print("setting obj_bbox1 from this_start", this_start)
                    obj_bbox1 = self.get_start_obj_bbox(this_start, df)
                    print("returned obj_bbox1 from this_start", obj_bbox1)
                print(f"set enc1 from get_start_enc() to {enc1}")
        else: 
            # if not first round -- is this for round 2+ ?
            # or is this when it is a cluster/topics pass through? 
            # 
            # print the last row of the dataframe
            # print("setting enc1 -- last row of the dataframe", df.iloc[-1])
            print("debugging enc1 setting", df.iloc[-1])
            # setting enc1
            if hsv_sort == True: enc1 = df.iloc[-1]["hsvll"]
            elif self.SORT_TYPE == "128d": enc1 = df.iloc[-1]["face_encodings68"]
            elif self.SORT_TYPE == "planar": enc1 = df.iloc[-1]["face_landmarks"]
            # elif self.SORT_TYPE == "planar_hands" and "hand_landmarks" in df.columns: enc1 = df.iloc[-1]["hand_landmarks"]
            elif self.SORT_TYPE == "planar_hands" and "hand_landmarks" in df.columns: enc1 = df.iloc[-1]["hand_landmarks"]
            elif self.SORT_TYPE == "planar_body" and "body_landmarks_array" in df.columns: enc1 = df.iloc[-1]["body_landmarks_array"]
            elif self.SORT_TYPE == "planar_body": enc1 = df.iloc[-1]["body_landmarks_normalized"]
            # setting obj_bbox1
            if self.SORT_TYPE == "planar_body" and "obj_bbox_list" in df.columns: obj_bbox1 = df.iloc[-1]["obj_bbox_list"]
        # print("returning enc1, obj_bbox1", enc1, obj_bbox1)
        return enc1, obj_bbox1


    def brute_force(self, df_enc, enc1):
        print("starting regular brute_force")
        for index, row in df_enc.iterrows():
            enc2 = row['face_encodings68']
            if (enc1 is not None) and (enc2 is not None):
                # print("getting d with enc1", enc1)
                # print("getting d with enc2", enc1)
                d = self.get_d(enc1, enc2)
                df_enc.loc[index, 'dist_enc1'] = d
            else:
                print("128d: missing enc1 or enc2")
                continue
        return df_enc
    

    def test_landmarks_vis(self, row):
        visible_LH = False
        visible_RH = False
        BOOL = False
        if isinstance(row, pd.Series):
            # If the input is a DataFrame row (pd.Series)
            lms = row['body_landmarks']
        else:
            # If the input is the landmarks object itself
            lms = row
            BOOL = True
        if isinstance(lms, landmark_pb2.NormalizedLandmarkList):
            # If the input is a NormalizedLandmarkList object
            
            left_hand = [15, 17, 19, 21]
            right_hand = [16, 18, 20, 22]
            for idx, lm in enumerate(lms.landmark):
                if idx in left_hand:
                    if lm.visibility > 0.5:
                        visible_LH = True
                elif idx in right_hand:
                    if lm.visibility > 0.5:
                        visible_RH = True
        elif isinstance(lms, np.ndarray):
            # TK prob not working correctly lms order
            left_hand = list(range(4))
            right_hand = list(range(4,8))
            print("left_hand, right_hand", left_hand, right_hand)
            # If the input is a NumPy array
            # print("lms is a numpy array,", lms)
            for idx in left_hand:
                if lms[idx] > 0.5:
                    visible_LH = True
            for idx in right_hand:
                if lms[idx] > 0.5:
                    visible_RH = True
        if BOOL:
            return [int(visible_LH), int(visible_RH)]
        # print("returning false, no hands from this row", row)
        else:
            return visible_LH, visible_RH

    def get_hand_angles(self, enc1):
        # calculate the angle of the vector between landmarks 16 and 20
        LW = np.array([enc1[0], enc1[1]])
        LF = np.array([enc1[2], enc1[3]])
        RW = np.array([enc1[4], enc1[5]])
        RF = np.array([enc1[6], enc1[7]])
        # LW = enc1[0,1]
        # LF = enc1[2,3]
        # RW = enc1[4,5]
        # RF = enc1[6,7]
        
        def get_angle(p1, p2):
            vector = p1 - p2
            angle_radians = np.arctan2(vector[1], vector[0])
            # angle_degrees = np.degrees(angle_radians)
            return angle_radians

        angle_LH = get_angle(LW, LF)
        angle_RH = get_angle(RW, RF)
        return [angle_LH, angle_RH]

    def weight_face_pose(self,row):
        row['face_x'] = (row['face_x'] - -28) * 0.25
        row['face_y'] = row['face_y'] * 0.25
        row['face_z'] = row['face_z']  * 0.25
        row['mouth_gap'] = row['mouth_gap']  * 0.25
        return row


    def prep_enc(self, enc1, structure="dict"):
        # if self.VERBOSE: print("prep_enc enc1", enc1)
        # if self.VERBOSE: print("prep_enc enc1", type(enc1))
        # if self.VERBOSE: print("self.SUBSET_LANDMARKS", self.SUBSET_LANDMARKS)
        if enc1 is list or enc1 is tuple or enc1 == "list":
            landmarks = self.SUBSET_LANDMARKS
        else:
            if self.VERBOSE: print("prep_enc enc1 is NormalizedLandmarkList, so devolving SUBSET")
            # devolve the x y landmarks back to index
            landmarks = []
            # take the even landmarks and divide by 2
            for lm in self.SUBSET_LANDMARKS:
                if structure == "wrists_and_ankles" and lm not in [15,16,27,28]:
                    continue
                landmarks.append(lm)


                # July 2025 WTF is going on with dividin g by 2?
                # if lm % 2 == 0:
                #     landmarks.append(int(lm / 2))

        # july 2025 was set to pointers
        # pointers = self.get_landmarks_2d(enc1, landmarks, structure)

        if structure == "visible":
            if self.VERBOSE: print("structure is visible")
            values = self.get_landmarks_2d(enc1, landmarks, "list3D")
            # extract only the visible ones
            visible_values = values[3::4]
            values = visible_values
        else:
            #redefining structure so that it will return values from get_landmarks
            if structure == "wrists_and_ankles": structure = "list"
            values = self.get_landmarks_2d(enc1, landmarks, structure)

        # pointers = self.get_landmarks_2d(enc1, self.SUBSET_LANDMARKS, structure)
        # print("prep_enc enc after get_landmarks_2d", type(enc1))

        # pointers = self.get_landmarks_2d(enc1, self.POINTERS, structure)
        # thumbs = self.get_landmarks_2d(enc1, self.THUMBS, structure)
        # # body = self.get_landmarks_2d(enc1, self.BODY_LMS, structure)
        # body = self.get_landmarks_2d(enc1, list(range(33)), structure)
        # # print("prep_enc enc after get_landmarks_2d", pointers, thumbs, body)
        # # enc1_np = np.array(enc1_list)

        # # calculate the angle of the vector between landmarks 16 and 20
        # angles_pointers = self.get_hand_angles(np.array(pointers))
        # angles_thumbs = self.get_hand_angles(np.array(thumbs))

        # # check if hands are visible
        # visibility = self.test_landmarks_vis(pointers)

        # print("types", type(angles_pointers), type(angles_thumbs), type(body), type(visibility))
        # print("enc1++ angles", angles_pointers, angles_thumbs, body, visibility)

        # print("enc1++ pointers", pointers, angles_pointers)
        # enc_angles_list = angles_pointers + angles_thumbs + body + visibility
        # enc_angles_list =  values 
        enc1 = np.array(values)
        if self.VERBOSE: print("enc1++ final np array", enc1)
        return enc1
    


    def sort_df_KNN(self, df_enc, enc1, knn_sort="128d"):
        if self.VERBOSE: print("df_enc at the start of sort_df_KNN")
        if self.VERBOSE: print(df_enc)

        output_cols = 'dist_enc1'
        if knn_sort == "128d":
            sortcol = 'face_encodings68'
        elif knn_sort == "planar":
            sortcol = 'face_landmarks'
        elif knn_sort == "planar_hands":
            sortcol = 'hand_landmarks'
            # sortcol = 'hand_landmarks'
        elif knn_sort == "planar_body":
            sortcol = 'body_landmarks_array'
            sourcecol = 'body_landmarks_normalized'
            # if type(enc1) is not list:
            #     enc1 = self.prep_enc(enc1, structure="list") # switching to 3d
            if self.VERBOSE: print("body_landmarks elif")
            # test to see if df_enc contains the sortcol column
            if sortcol not in df_enc.columns:
                if self.VERBOSE: print("sortcol not in df_enc.columns - body_landmarks enc1 pre prep_enc", enc1)
                # # if enc1 is not a numpy array, convert it to a list
                # # create enc list with x/y position and angles and visibility
                # if self.VERBOSE: print("enc1 before prep_enc", enc1)
                # if self.VERBOSE: print("enc1 after prep_enc", enc1)
                # # apply prep_enc to the sortcol column 
                # if self.VERBOSE: print("applying prep_enc to the sortcol column")
                # if self.VERBOSE: print("df_enc[sourcecol]", df_enc[sourcecol])
                # # do I need to reduce the number of landmarks I'm tracking at this point? 
                # # moving to pre_enc in makevid
                # # df_enc[sortcol] = df_enc[sourcecol].apply(lambda x: self.prep_enc(x, structure="list")) # swittching to 3d
                # if self.VERBOSE: print("df_enc[sortcol]", df_enc[sortcol])
        elif knn_sort == "body3D":
            sortcol = 'body_landmarks_array'
            # sourcecol = 'body_landmarks_3D'
        elif knn_sort == "HSV":
            if self.VERBOSE: print("knn_sort is HSV")
            sortcol = 'hsvll'
            output_cols = 'dist_HSV'
            if self.VERBOSE: print(type(enc1))
            if self.VERBOSE: print(enc1)
            # if self.VERBOSE: print(df_enc.loc[0])
            if self.VERBOSE: print(type(df_enc.head(1)['lum'].values[0]))
            if self.VERBOSE: print(df_enc.head(1)['lum'].values[0])
        elif knn_sort == "obj":
            # overriding for object detection
            sortcol = 'obj_bbox_list'
            output_cols = 'dist_obj'
            # enc1 = obj_bbox1

        # if self.OBJ_CLS_ID > 0:
        #     # overriding for object detection
        #     sortcol = 'obj_bbox_list'
        #     enc1 = obj_bbox1

        #create output column -- do i need to do this?
        df_enc[output_cols] = np.nan
        # df_enc[output_cols] = pd.Series(dtype='float64')
        # print(">>>>>       df_enc.index")
        # print(df_enc.index)
        # print(len(df_enc.index))
        # # print(df_enc[sortcol])
        # # print(df_enc[output_cols])
        # # print the count of NaN values in the sortcol column
        # print(">>>>>       NaN count in sortcol", df_enc[sortcol].isnull().sum())
        # # print("NaN in sortcol", df_enc[sortcol].isnull())
        print("sort_df_KNN, knn_sort is", knn_sort)
        # Extract the face encodings from the dataframe
        print("df_enc[sortcol]", df_enc[sortcol])

        encodings_array = df_enc[sortcol].to_numpy().tolist()
        # print("encodings_array to_numpy().tolist", (encodings_array))

        if knn_sort == "obj":
            print("encodings_array length", len(encodings_array))
            print("[sort_df_KNN] enc1", enc1)

        def contains_nan(arr):
            # Check for NaN in a nested structure
            for x in arr:
                # If the element is a NumPy array, check if any element is NaN
                if isinstance(x, np.ndarray):
                    if np.isnan(x).any():
                        return True
                # If the element is a list or scalar, use pd.isnull to check for NaN
                elif isinstance(x, list):
                    if any(pd.isnull(el) for el in x):
                        return True
                else:
                    if pd.isnull(x):
                        return True
            return False

        # Check if the encodings_array contains NaN values
        if contains_nan(encodings_array):
            print("encodings_array contains NaN values")
            
            # Convert encodings_array to a NumPy array
            encodings_array = np.array(encodings_array)

            # Create a boolean mask for non-NaN rows in encodings_array
            non_nan_mask = ~np.isnan(encodings_array).any(axis=1)
            
            # Remove rows with NaN values from encodings_array
            encodings_array = encodings_array[non_nan_mask]
            print("Cleaned encodings_array:", encodings_array)
            print("Cleaned encodings_array shape:", encodings_array.shape)

            # Remove the same rows from df_enc based on the non_nan_mask
            df_enc = df_enc[non_nan_mask].reset_index(drop=True)
            print("Cleaned df_enc:", df_enc.shape)

        self.knn.fit(encodings_array)

        
        # # Ensure n_neighbors does not exceed the number of samples
        # n_neighbors = min(len(df_enc_clean), len(encodings_array))

        # # Query the KNN model with the cleaned df_enc
        # distances, indices = self.knn.kneighbors([enc1], n_neighbors=n_neighbors)

        # # Find the distances and indices of the neighbors
        distances, indices = self.knn.kneighbors([enc1], n_neighbors=len(df_enc))


        # zip the indices and distances and print
        # print("indices and distances", list(zip(indices, distances)))
        # print("indices and distances", list(zip(indices.flatten(), distances.flatten())))        
        # Flatten the indices and distances
        indices = indices.flatten()
        distances = distances.flatten()

        # Mapping from position to actual index
        position_to_index = {pos: idx for pos, idx in enumerate(df_enc.index)}

        # Create a dictionary with actual indices as keys
        id_dict = {position_to_index[idx]: dist for idx, dist in zip(indices, distances)}


        # # Create a dictionary with valid indices as keys
        # id_dict = {idx: dist for idx, dist in zip(indices, distances)}
        # print("id_dict", id_dict)
        # print("len(id_dict)", len(id_dict))
        # # print count of NaN values the id_dict
        # print("NaN count in id_dict", pd.Series(id_dict).isnull().sum())

        # the id_dict is a dictionary with the index as the key and the distance as the value
        # it is correct
        # the code below is NOT correctly assigning the values to the df_enc


        # Update the 'dist_enc1' column with the distances for valid indices
        for idx in df_enc.index:
            if idx in id_dict:
                df_enc.at[idx, output_cols] = id_dict[idx]
            else:
                print(f"Index {idx} not found in id_dict")
                

        print("df_enc after adding distance")
        # print(df_enc.index)
        # print(df_enc[sortcol])
        # print(df_enc[output_cols])
        # print("<<<<<       NaN count in sortcol AFTER KNN", df_enc[sortcol].isnull().sum())

        # print(df_enc['dist_enc1'].dtype)

        # df_enc['dist_enc1'] = df_enc['dist_enc1'].astype(float)

        # def safe_float_convert(x):
        #     try:
        #         return float(x)
        #     except:
        #         return np.nan

        # df_enc['dist_enc1'] = df_enc['dist_enc1'].apply(safe_float_convert)

        # df_enc = df_enc.sort_values(by='dist_enc1', na_position='last')
        # df_enc = df_enc.sort_values(by='dist_enc1', na_position='last').reset_index(drop=True)
        # problematic_rows = df_enc[pd.to_numeric(df_enc['dist_enc1'], errors='coerce').isna()]
        # print("Problematic rows:")
        # print(problematic_rows)

        # df_enc = df_enc.dropna(subset=['dist_enc1'])
        # df_enc = df_enc.sort_values(by='dist_enc1').reset_index(drop=True)


        df_enc = df_enc.sort_values(by=output_cols)
        # print("df_enc after sorting")
        # print(df_enc.index)
        # print(df_enc[sortcol])
        # print(df_enc[output_cols])
        return df_enc

    def draw_point(self,image,landmarks_2d,index):
        #it would prob be better to do this with a dict and a loop
        # nose_2d = self.get_face_2d_point(faceLms,1)
        print("landmarks_2d", landmarks_2d)
        img_h, img_w = image.shape[:2]
        img_h2, img_w2 = int(img_h/2), int(img_w/2)
        points = zip(landmarks_2d[1::2],landmarks_2d[::2])
        # cv2.circle(image,(img_w2,img_h2),14,(0,0,255),-1)

        for point in points:
            y, x = int(point[0]*(img_h)), int(point[1]*(img_w))
            # x, y = int(point[0]*(img_w-300)), int(point[1]*(img_h-300))
            cv2.circle(image,(x,y),4,(255,0,0),-1)
        return image
    
    def get_closest_df_NN(self, df_enc, df_sorted,start_image_id=None, end_image_id=None):
  
        def mask_df(df, column, limit, type="lessthan"):
            # Create the mask based on the condition
            if type == "lessthan":
                flashmask = df[column] < limit
            elif type == "greaterthan":
                flashmask = df[column] > limit

            # Check if all values are False, meaning no rows satisfy the condition
            if not flashmask.any():
                print(f"Warning: No rows in '{column}' satisfy the '{type}' condition with limit {limit}.")
                return pd.DataFrame(columns=df.columns)  # Return an empty DataFrame with the same columns

            # Apply the mask and reset the index
            df = df[flashmask].reset_index(drop=True)

            return df
        
        def de_dupe(df_dist_hsv, df_sorted, column, is_run = False):
            # remove duplicates (where dist is less than BODY_DUPE_DIST)
            def json_to_list(json):
                return [v for k, v in json.items()]
            
            df_dist_hsv = mask_df(df_dist_hsv, column, self.DUPED, "greaterthan")
            df_close_ones = mask_df(df_dist_hsv, column, self.MIND, "lessthan")
            last_image = df_sorted.iloc[-1].to_dict()
            dupe_index = []
            hsvll_dist = face_dist = bbox_dist = 1 # so it doesn't trigger the dupe_score
            # print("de_duping from", last_image['image_id'], last_image['dist_enc1'], last_image['description'], last_image['bbox'])
            for index, row in df_close_ones.iterrows():
                
                # print("de_duping aginst", row['image_id'], row['dist_enc1'], row['description'], last_image['bbox'])
                # print(last_image['bbox'].items(), row['bbox'].items())
                hsvll_dist = self.get_d(last_image['hsvll'], row['hsvll'])
                face_dist = self.get_d(last_image['face_encodings68'], row['face_encodings68'])
                # should also test body_landmarks, (for 128d sort), and then bump dupe_score threshold to 3
                bbox_dist = self.get_d(json_to_list(last_image['bbox']), json_to_list(row['bbox']))
                # print("hsvll_dist", hsvll_dist, "face_dist", face_dist, "bbox_dist", bbox_dist)

                # tally up the dupe_score
                dupe_score = 0
                if row['description'] == last_image['description']: dupe_score += 1
                if hsvll_dist < .1 :  dupe_score += 1
                if face_dist < .4 : dupe_score += 1
                if bbox_dist < 5 : dupe_score += 1
                # print("dupe_score", dupe_score)

                if dupe_score > 2:
                    if self.VERBOSE: print("de_duping score", dupe_score, last_image['image_id'], "is a duplicate of", row['image_id'])
                    # add the index of the duplicate to the list of indexes to drop
                    dupe_index.append(index)
                elif is_run:
                    last_image = row

            df_dist_hsv = df_dist_hsv.drop(dupe_index).reset_index(drop=True)

            return df_dist_hsv

        if self.VERBOSE: print("debugging df_enc", df_enc.columns)
        if self.VERBOSE: print("debugging df_sorted", df_sorted.columns)
        if len(df_sorted) == 0: 
            FIRST_ROUND = True
            enc1, obj_bbox1 = self.get_enc1(df_enc, FIRST_ROUND)
            if self.VERBOSE: print("first round enc1, obj_bbox1", enc1, obj_bbox1)
            # drop all rows where obj_bbox_list is None
            if self.OBJ_CLS_ID > 0: df_enc = df_enc.dropna(subset=["obj_bbox_list"])
        else: 
            FIRST_ROUND = False
            enc1, obj_bbox1 = self.get_enc1(df_sorted, FIRST_ROUND)
            if self.VERBOSE: print("LATER round enc1, obj_bbox1", enc1, obj_bbox1)

        if self.VERBOSE: print(f"get_closest_df_NN, self.SORT_TYPE is {self.SORT_TYPE} FIRST_ROUND is {FIRST_ROUND}")
        # define self.SORT_TYPE for KNN
        if self.SORT_TYPE == "128d" or (self.SORT_TYPE == "planar" and FIRST_ROUND) or (self.SORT_TYPE == "planar_body" and len(enc1) > 66): 
            knn_sort = "128d"      
        elif self.SORT_TYPE == "planar": 
            knn_sort = "planar"
        elif self.SORT_TYPE == "planar_body": 
            if self.CLUSTER_TYPE == "HandsPositions":
                knn_sort = "planar_hands"
            else:
                knn_sort = "planar_body"
        elif self.SORT_TYPE == "body3D":
            knn_sort = "body3D"
        elif self.SORT_TYPE == "planar_hands": 
            knn_sort = "planar_hands"
        
        if self.VERBOSE: print("get_closest_df_NN - pre KNN - enc1", enc1)
        # sort KNN (always for planar) or BRUTEFORCE (optional only for 128d)
        if self.BRUTEFORCE and knn_sort == "128d": df_dist_enc = self.brute_force(df_enc, enc1)
        elif start_image_id is not None: self.sort_df_TSP(df_enc, start_image_id, end_image_id)
        else: df_dist_enc = self.sort_df_KNN(df_enc, enc1, knn_sort)
        if self.VERBOSE: print("df_shuffled", df_dist_enc[['image_id','dist_enc1']].sort_values(by='dist_enc1'))
        
        if self.VERBOSE: print("self.OBJ_CLS_ID", self.OBJ_CLS_ID)
        # sort KNN for OBJ_CLS_ID
        if self.OBJ_CLS_ID > 0: 
            if self.VERBOSE: print("get_closest_df_NN - pre KNN - obj_bbox1", obj_bbox1)
            if type(obj_bbox1) is dict:
                # turn the obj_bbox1 json dict into a list
                obj_bbox1 = self.json_to_list(obj_bbox1)
            df_dist_enc = self.sort_df_KNN(df_enc, obj_bbox1, "obj")
            if self.VERBOSE: print("df_shuffled obj", df_dist_enc[['image_id','dist_obj']].sort_values(by='dist_obj')) 

        # set HSV start enc and add HSV dist
        if not self.ONE_SHOT:
            if not 'dist_HSV' in df_sorted.columns:  
                if self.VERBOSE: print("not dist_HSV")
                enc1, obj_bbox1 = self.get_enc1(df_enc, FIRST_ROUND=True, hsv_sort=True)
            else: 
                if self.VERBOSE: print("else is dist_HSV")
                enc1, obj_bbox1 = self.get_enc1(df_sorted, FIRST_ROUND=False, hsv_sort=True)
            if self.VERBOSE: print("enc1", enc1)
            if self.VERBOSE: print("df_dist_enc before normalize_hsv", df_dist_enc)
            if self.VERBOSE: print("columns", df_dist_enc.columns)
            if self.VERBOSE: print("first row", df_dist_enc.iloc[0])
            df_dist_hsv = self.normalize_hsv(enc1, df_dist_enc)
            if self.VERBOSE: print("df_dist_enc after normalize_hsv, before sort", df_dist_enc)
            if self.VERBOSE: print("first row", df_dist_enc.iloc[0])
            df_dist_hsv = self.sort_df_KNN(df_dist_hsv, enc1, "HSV")
            if self.VERBOSE: print("columns", df_dist_enc.columns)
            if self.VERBOSE: print("df_shuffled HSV", df_dist_hsv[['image_id','dist_enc1','dist_HSV']].head())
        else:
            # skip HSV if ONE_SHOT (which is for still images, so N/A)
            # assign 0 to dist_HSV for first round
            df_dist_hsv = df_dist_enc

        if len(df_dist_hsv) > 0:
            if not FIRST_ROUND:
                # remove duplicates (where dist is less than BODY_DUPE_DIST)
                df_dist_hsv = de_dupe(df_dist_hsv, df_sorted, 'dist_enc1')

                # assign backto main df_enc to permanently rm dupes. 
                df_enc = df_dist_hsv
            
            if not self.ONE_SHOT:
                if self.VERBOSE: print("df_enc if not self.ONE_SHOT", df_enc)
                # np.set_printoptions(threshold = np.inf)
                # enc_values_list = df_enc['dist_enc1'].values
                # print("dist_enc1 values:", enc_values_list)
                # temporarily removes items for this round
                df_dist_noflash = mask_df(df_dist_hsv, 'dist_HSV', self.HSV_DELTA_MAX, "lessthan")
                if self.VERBOSE: print("df_dist_noflash if not self.ONE_SHOT",df_dist_noflash)
                # print all values in the dist_enc1 column
                # print("dist_enc1 values:", df_dist_noflash['dist_enc1'].values)
                # replacing dist_HSV with dist_enc1 here, Sept 27
                df_dist_close = mask_df(df_dist_noflash, 'dist_enc1', self.MAXD, "lessthan")
                if self.VERBOSE: print("df_dist_close if not self.ONE_SHOT",df_dist_close)

                if df_dist_close.empty:
                    if self.VERBOSE: print("No rows in the DataFrame met the filtering criteria, returning empty df and the untouched df_sorted.")
                    return df_dist_close, df_sorted
                else:
                    if self.VERBOSE: print("Filtered DataFrame:", df_dist_close)
                # implementing these masks for now
                    df_shuffled = df_dist_close
                
                # sort df_shuffled by the sum of dist_enc1 and dist_HSV
                df_shuffled['sum_dist'] = df_dist_noflash['dist_enc1'] + self.MULTIPLIER * df_shuffled['dist_HSV']
                # print("df_shuffled columns", df_shuffled.columns)

                # of OBJ sort kludge but throws errors for non object, non ONE SHOT. If so, use above
                # df_shuffled['sum_dist'] = df_shuffled['dist_obj'] 

                df_shuffled = df_shuffled.sort_values(by='sum_dist').reset_index(drop=True)
                if self.VERBOSE: print("df_shuffled pre_run", df_shuffled[['image_id','dist_enc1','dist_HSV','sum_dist']])
            else:
                # if ONE_SHOT, skip the mask, and keep all data
                df_shuffled = df_dist_hsv

            try: runmask = df_shuffled['sum_dist'] < self.MIND
            except: runmask = None
            if self.VERBOSE: print("runmask", runmask)

            # if self.ONE_SHOT and not runmask:

            if self.ONE_SHOT:
                if self.VERBOSE: print("ONE_SHOT going to assign all and try to drop everything")
                df_run = df_shuffled
                # drop all rows from df_shuffled
                df_enc = df_enc.drop(df_run.index).reset_index(drop=True)
                if self.VERBOSE: print("df_run", df_run)
                if self.VERBOSE: print("df_enc", len(df_enc))

            elif runmask.any():
                num_true_values = runmask.sum()
                if self.VERBOSE: print("we have a run ---->>>>", num_true_values)
                self.SHOT_CLOCK = 0 # reset the shot clock
                # if there is a run < MINFACEDIST
                df_run = df_shuffled[runmask]

                # need to dedupe run if not first round
                if not FIRST_ROUND: df_run = de_dupe(df_run, df_sorted, 'dist_enc1', is_run=True)

                # locate the index of df_enc where image_id = image_id in df_run
                index_names = df_enc[df_enc['image_id'].isin(df_run['image_id'])].index

                # remove the run from df_enc where image_id = image_id in df_run
                df_enc = df_enc.drop(index_names).reset_index(drop=True)

            elif len(df_shuffled) > 0 and (self.JUMP_SHOT is True and self.SHOT_CLOCK < self.SHOT_CLOCK_MAX):

                # df_run = first row of df_shuffled based on image_id
                df_run = df_shuffled.iloc[[0]]  # Select the first row
                # increment the shot clock
                self.SHOT_CLOCK += 1
                if self.VERBOSE: print(f"NO run, shot clock is {self.SHOT_CLOCK} <<<< ", df_run)

                # Locate the rows in df_enc with matching image_id
                index_names = df_enc[df_enc['image_id'].isin(df_run['image_id'])].index

                # Drop rows with matching image_id from df_enc
                if len(index_names) > 0:
                    df_enc = df_enc.drop(index_names).reset_index(drop=True)
                    if self.VERBOSE: print("df_run", df_run)
                    if self.VERBOSE: print("df_enc", len(df_enc))
                else:
                    if self.VERBOSE: print("No matching image_id found in df_enc for df_run")
            

            elif self.JUMP_SHOT is True and self.SHOT_CLOCK >= self.SHOT_CLOCK_MAX:
                    if self.VERBOSE: print("SHOT_CLOCK has reached the maximum value, resetting to 0.")
                    self.SHOT_CLOCK = 0

                    #jump somewhere else
                    if self.VERBOSE: print("JUMPING AROUND ^^^^ ", df_shuffled)
                    random_index = random.randint(0, len(df_shuffled) - 1)                
                    if self.VERBOSE: print("JUMPING TO ---> ", random_index)
                    df_run = df_shuffled.iloc[[random_index]]
                    if self.VERBOSE: print("df_run new start point", df_run)

                    df_enc = df_enc.drop(df_run.index).reset_index(drop=True)


            else:
                if self.VERBOSE: print("df_shuffled is empty")
                return df_enc, df_sorted

            if self.VERBOSE: print("df_run", df_run)
            # if self.VERBOSE: print("df_run", df_run[['image_id','dist_enc1','dist_HSV','sum_dist']])


            df_sorted = pd.concat([df_sorted, df_run])
            if self.VERBOSE: print("df_sorted containing all good items", len(df_sorted))
            
            if self.VERBOSE: print("df_enc", len(df_enc))
        else:
            if self.VERBOSE: print("df_shuffled is empty")

        return df_enc, df_sorted




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


#####################################################
# BODY BACKGROUND OBJECT DETECTION STUFF            #
#####################################################

    def normalize_hand_landmarks(self, results, nose_pos, face_height, shape):
        height, width = shape[:2]
        translated_landmark_dict = {}

        for hand_side in ['left_hand', 'right_hand']:
            if hand_side in results and 'image_landmarks' in results[hand_side]:
                hand_landmarks = results[hand_side]['image_landmarks']  # Get the image landmarks list

                translated_image_landmarks = []
                # print("nose_pos", nose_pos)
                for hand_landmark in hand_landmarks:
                    # Translate the landmark coordinates
                    translated_landmark = [
                        # in the hand_landmark, 0 is x and 1 is y
                        # this is the extant math:
                        # (nose_pos["x"] - hand_landmark[0] * width) / face_height,  # x
                        # (nose_pos["y"] - hand_landmark[1] * height) / face_height,  # y
                        # I changed to this math. This gives a negative number when hand is to the left of the nose
                        # and a positive number when the hand is below the nose
                        (hand_landmark[0] * width - nose_pos["x"]) / face_height,  # x
                        (hand_landmark[1] * height - nose_pos["y"]) / face_height,  # y
                        (0 - hand_landmark[2] * height) / face_height,  # z
                    ]

                    # Add the translated landmark to the list
                    translated_image_landmarks.append(translated_landmark)

                # Store the list of translated landmarks in the dictionary
                translated_landmark_dict[hand_side] = {
                    "image_landmarks": translated_image_landmarks,
                }

        return translated_landmark_dict


    def normalize_landmarks(self,landmarks,nose_pos,face_height,shape):
        height,width = shape[:2]
        translated_landmarks = landmark_pb2.NormalizedLandmarkList()
        for landmark in landmarks.landmark:
            # print("normalize_landmarks", nose_pos["x"], landmark.x, width, face_height)
            translated_landmark = landmark_pb2.NormalizedLandmark()
            translated_landmark.x = (nose_pos["x"]-landmark.x*width )/face_height
            translated_landmark.y = (nose_pos["y"]-landmark.y*height)/face_height
            translated_landmark.visibility = landmark.visibility
            translated_landmarks.landmark.append(translated_landmark)

        return translated_landmarks

    def project_normalized_landmarks(self,landmarks,nose_pos,face_height,shape):
        # height,width = shape[:2]
        projected_landmarks = landmark_pb2.NormalizedLandmarkList()
        i=0
        for landmark in landmarks.landmark:
            # print(nose_pos)
            #  nose_pos[x]-landmark.x*face_height
            projected_landmark = landmark_pb2.NormalizedLandmark()
            projected_landmark.x = int(nose_pos["x"]-landmark.x*face_height)
            projected_landmark.y = int(nose_pos["y"]-landmark.y*face_height)
            # projected_landmark.x=projected_landmark.x*height/width
            # projected_landmark.y=projected_landmark.x*width/height   
            projected_landmark.visibility = landmark.visibility
            projected_landmarks.landmark.append(projected_landmark)

        return projected_landmarks


    def convert_bbox_to_face_height(self,bbox):
        if type(bbox)==str:
            bbox=json.loads(bbox)
        # if VERBOSE:
        #     print("bbox",bbox)
        face_height=bbox["top"]-bbox["bottom"]
        return face_height





    def set_nose_pixel_pos(self,body_landmarks,shape):
        # body_landmarks = self.ensure_lms_list(body_landmarks)  # Ensure body_landmarks is a list
        # if body_landmarks is pickled, unpickle it
        if type(body_landmarks)==bytes:
            body_landmarks=pickle.loads(body_landmarks)
        if self.VERBOSE: print("set_nose_pixel_pos body_landmarks")
        height,width = shape[:2]
        if self.VERBOSE: print("set_nose_pixel_pos bodylms height, width", height, width)
        nose_pixel_pos ={
            "x":0,
            "y":0,
            "visibility":0
        }
        # nose_pixel_pos <- 864, 442 (stay as a separate variable)
        # nose_normalized_pos 0,0
        # nose_pos=body_landmarks.landmark[NOSE_ID]
        if self.VERBOSE: print("unprojected bodylms: ", body_landmarks.landmark[0].x, body_landmarks.landmark[0].y)
        nose_pixel_pos["x"]+=body_landmarks.landmark[0].x*width
        nose_pixel_pos["y"]+=body_landmarks.landmark[0].y*height
        if self.VERBOSE: print ("set_nose_pixel_pos bodylms nose_pixel_pos", nose_pixel_pos)
        self.nose_2d = nose_pixel_pos # this could be a problem
        if self.VERBOSE: print("set_nose_pixel_pos nose_pixel_pos",nose_pixel_pos)
        if self.VERBOSE: print("set_nose_pixel_pos self.nose_2d",self.nose_2d)
        nose_pixel_pos["visibility"]+=body_landmarks.landmark[0].visibility
        # nose_3d has visibility
        self.nose_3d = nose_pixel_pos
        return nose_pixel_pos
    
    def normalize_phone_bbox(self,phone_bbox,nose_pos,face_height,shape):
        height,width = shape[:2]
        if self.VERBOSE: print("phone_bbox",phone_bbox)
        if self.VERBOSE: print("type phone_bbox",type(phone_bbox))

        n_phone_bbox=phone_bbox
        n_phone_bbox["right"] =(nose_pos["x"]-n_phone_bbox["right"] )/face_height
        n_phone_bbox["left"]  =(nose_pos["x"]-n_phone_bbox["left"]  )/face_height
        n_phone_bbox["top"]   =(nose_pos["y"]-n_phone_bbox["top"]   )/face_height
        n_phone_bbox["bottom"]=(nose_pos["y"]-n_phone_bbox["bottom"])/face_height
        if self.VERBOSE: print("type phone_bbox",type(n_phone_bbox["right"]))

        return n_phone_bbox


    def insert_n_landmarks(self,bboxnormed_collection, image_id, n_landmarks):
        start = time.time()
        nlms_dict = {"image_id": image_id, "nlms": pickle.dumps(n_landmarks)}
        result = bboxnormed_collection.update_one(
            {"image_id": image_id},  # filter
            {"$set": nlms_dict},     # update
            upsert=True              # insert if not exists
        )
        if result.upserted_id:
            pass
            # print("Inserted new document with id:", result.upserted_id)
        else:
            print("Updated existing document")
        # print("Time to insert:", time.time()-start)
        return
    
    def update_hand_landmarks_in_mongo(self,mongo_hand_collection, image_id, hand_landmarks_norm):
        update_data = {}
        
            
                    # Check if left_hand exists and update accordingly
        if 'left_hand' in hand_landmarks_norm:            
            update_data['left_hand.hand_landmarks_norm'] = hand_landmarks_norm['left_hand']['image_landmarks']
        
        # Check if right_hand exists and update accordingly
        if 'right_hand' in hand_landmarks_norm:
            update_data['right_hand.hand_landmarks_norm'] = hand_landmarks_norm['right_hand']['image_landmarks']
        
        # Perform the MongoDB update operation
        if update_data:
            # print("update_data", update_data)
            mongo_hand_collection.update_one(
                {'image_id': image_id},  # Assuming you have a document_id to update
                {'$set': update_data},
                upsert=True
            )
        print("Hand landmarks updated successfully.", image_id)


    def insert_hand_landmarks_norm(self,mongo_hand_collection, image_id, hand_landmarks_norm):
        # start = time.time()
        nlms_dict = {"image_id": image_id, "nlms": pickle.dumps(hand_landmarks_norm)}
        result = mongo_hand_collection.update_one(
            {"image_id": image_id},  # filter
            {"$set": nlms_dict},     # update
            upsert=True              # insert if not exists
        )
        if result.upserted_id:
            pass
            # print("Inserted new document with id:", result.upserted_id)
        else:
            print("Updated existing document")
        # print("Time to insert:", time.time()-start)
        return

    def return_bbox(self, model, image, OBJ_CLS_LIST):
        result = model(image,classes=[OBJ_CLS_LIST])[0]
        bbox_dict={}
        bbox_count=np.zeros(len(OBJ_CLS_LIST))
        for i,OBJ_CLS_ID in enumerate(OBJ_CLS_LIST):
            for box in result.boxes:
                if int(box.cls[0].item())==OBJ_CLS_ID:
                    bbox = box.xyxy[0].tolist()    #the coordinates of the box as an array [x1,y1,x2,y2]
                    bbox = {"left":round(bbox[0]),"top":round(bbox[1]),"right":round(bbox[2]),"bottom":round(bbox[3])}
                    bbox=json.dumps(bbox)
                    # bbox=json.dumps(bbox, indent = 4) 
                    conf = round(box.conf[0].item(), 2)                
                    bbox_count[i]+=1 
                    bbox_dict[OBJ_CLS_ID]={"bbox": bbox, "conf": conf}

        for i,OBJ_CLS_ID in enumerate(OBJ_CLS_LIST):
            if bbox_count[i]>1: # checking to see it there are more than one objects of a class and removing 
                bbox_dict.pop(OBJ_CLS_ID)
                bbox_dict[OBJ_CLS_ID]={"bbox": None, "conf": -1} ##setting to default
            if bbox_count[i]==0:
                bbox_dict[OBJ_CLS_ID]={"bbox": None, "conf": -1} ##setting to default
        return bbox_dict

    def parse_bbox_dict(self, session, target_image_id, PhoneBbox, OBJ_CLS_LIST, bbox_dict):
        # I don't think it likes me sending PhoneBbox as a class
        # for calc face pose i'm moving this back to function
        for OBJ_CLS_ID in OBJ_CLS_LIST:
            bbox_n_key = "bbox_{0}_norm".format(OBJ_CLS_ID)
            print(bbox_dict)
            if bbox_dict[OBJ_CLS_ID]["bbox"]:
                PhoneBbox_entry = (
                    session.query(PhoneBbox)
                    .filter(PhoneBbox.image_id == target_image_id)
                    .first()
                )

                if PhoneBbox_entry:
                    setattr(PhoneBbox_entry, "bbox_{0}".format(OBJ_CLS_ID), bbox_dict[OBJ_CLS_ID]["bbox"])
                    setattr(PhoneBbox_entry, "conf_{0}".format(OBJ_CLS_ID), bbox_dict[OBJ_CLS_ID]["conf"])
                    setattr(PhoneBbox_entry, bbox_n_key, bbox_dict[bbox_n_key])
                    print("image_id:", PhoneBbox_entry.target_image_id)
                    #session.commit()
                    print(f"Bbox {OBJ_CLS_ID} for image_id {target_image_id} updated successfully.")
                else:
                    print(f"Bbox {OBJ_CLS_ID} for image_id {target_image_id} not found.")
            else:
                print(f"No bbox for {OBJ_CLS_ID} in image_id {target_image_id}")
        
        return session
    
#### ImagesBackground Stuff

    def get_segmentation_mask(self,get_bg_segment,img,bbox=None,face_landmarks=None):
        if self.VERBOSE: print("[get_bg_hue_lum] about to go for segemntation")

        if bbox:
            try:
                if self.VERBOSE: print("get_segmentation_mask: bbox type", type(bbox))
                if type(bbox)==str:
                    bbox=json.loads(bbox)
                    if self.VERBOSE: print("bbox type", type(bbox))
                #sample_img=sample_img[bbox['top']:bbox['bottom'],bbox['left']:bbox['right'],:]
                # passing in bbox as a str
                img, is_inpaint  = self.crop_image(img, face_landmarks, bbox)
                if img is None: return -1,-1,-1,-1,-1 ## if TOO_BIG==true, checking if cropped image is empty
            except:
                print(traceback.format_exc())
                if self.VERBOSE: print("FAILED CROPPING, bad bbox",bbox)
                return -2,-2,-2,-2,-2
            print("bbox['bottom'], ", bbox['bottom'])

        result = get_bg_segment.process(img[:,:,::-1]) #convert RBG to BGR then process with mp
        if self.VERBOSE: print("[get_bg_hue_lum] got result")
        return result.segmentation_mask


    def get_selfie_bbox(self, segmentation_mask):
        bbox=None
        scaled_mask = (segmentation_mask * 255).astype(np.uint8)
        # Apply a binary threshold to get a binary image
        _, binary = cv2.threshold(scaled_mask, 127, 255, cv2.THRESH_BINARY)
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Assume the largest contour is the shape
            contour = max(contours, key=cv2.contourArea)
            # Get the bounding box of the shape
            x, y, w, h = cv2.boundingRect(contour)
            # Draw the bounding box for visualization
            bbox={"top":y,"right":scaled_mask.shape[1] - (x + w),"bottom":scaled_mask.shape[0] - (y + h),"left":x}
        else:
            print("No contours were found")
        if bbox is None: print("bbox is empty, figure out what happened")
        else:
            if self.VERBOSE:print("bbox=",bbox)
        return bbox

    def test_shoulders(self,segmentation_mask):
        # print("segmentation_mask", segmentation_mask)
        # print("segmentation_mask[-1,0]", segmentation_mask[-1,0])
        # print("segmentation_mask[-1,-1]", segmentation_mask[-1,-1])
        # print("type segmentation_mask[-1,0]", type(segmentation_mask))
        left_shoulder=segmentation_mask[-1,0]
        right_shoulder=segmentation_mask[-1,-1]
        if left_shoulder<=self.SHOULDER_THRESH:
            is_left_shoulder=False
            # print("no left shoulder")
        else:
            # print("left shoulder present")
            is_left_shoulder=True

        if right_shoulder<=self.SHOULDER_THRESH:
            is_right_shoulder=False
            # print("no right shoulder")
        else:
            # print("right shoulder present")
            is_right_shoulder=True
        return is_left_shoulder,is_right_shoulder


#### HANDS STUFF

    def prep_cluster_medians(self, results):
        # store the results in a dictionary where the key is the cluster_id
        if results:
            cluster_medians = {}
            for i, row in enumerate(results, start=1):
                cluster_median = pickle.loads(row.cluster_median)
                cluster_medians[i] = cluster_median
                # print("cluster_medians", i, cluster_median)
                N_CLUSTERS = i # will be the last cluster_id which is count of clusters
        else:
            cluster_medians = None
            N_CLUSTERS = 0
        return cluster_medians, N_CLUSTERS
    
    def prep_hand_landmarks(self, hand_results):  
        left_hand_landmarks = left_hand_world_landmarks = left_hand_landmarks_norm = right_hand_landmarks = right_hand_world_landmarks = hand_landmarks = []
        if hand_results:
            if 'left_hand' in hand_results:
                left_hand_landmarks = hand_results['left_hand'].get('image_landmarks', [])
                left_hand_world_landmarks = hand_results['left_hand'].get('world_landmarks', [])
                left_hand_landmarks_norm = hand_results['left_hand'].get('hand_landmarks_norm', [])
            if 'right_hand' in hand_results:
                right_hand_landmarks = hand_results['right_hand'].get('image_landmarks', [])
                right_hand_world_landmarks = hand_results['right_hand'].get('world_landmarks', [])
                hand_landmarks = hand_results['right_hand'].get('hand_landmarks_norm', [])
        return left_hand_landmarks, left_hand_world_landmarks, left_hand_landmarks_norm, right_hand_landmarks, right_hand_world_landmarks, hand_landmarks

    def extract_landmarks(self, landmarks):
        # If no landmarks, return 63 zeros (21 points * 3 dimensions)
        # print("extract_landmarks self.SUBSET_LANDMARKS", self.SUBSET_LANDMARKS)
        if not landmarks:
            # print(f"extract_landmarks no landmarks {landmarks}")
            if self.CLUSTER_TYPE == "fingertips_positions":
                return [0.0] * len(self.SUBSET_LANDMARKS)
            else:
                # print(f"going to return 0s", ([0.0] * 63))

                return [0.0] * 63
                
        # print("landmarks", landmarks)
        # print("type landmarks", type(landmarks))
        # print("len landmarks", len(landmarks))
        # print("landmarks[0]", landmarks[0])


        # pointers = self.get_landmarks_2d(enc1, landmarks, structure)

        # Flatten the list of (x, y, z) for each landmark
        if self.VERBOSE: print("type of landmarks", type(landmarks))
        if type(landmarks) is list:
            if len(landmarks) > 0 and isinstance(landmarks[0], (list, tuple)):
                if self.VERBOSE: print("landmarks[0] type", type(landmarks[0]))
                flat_landmarks = [coord for point in landmarks for coord in point]
            else:
                print("landmarks is a list but not a list of lists, handling the erronous new AP stuff. ")
                # flat_landmarks = None
                # handle erronous new API 3D body landmarks, 
                flat_landmarks = []
                for idx, lm in enumerate(landmarks):
                    # If lm is a Landmark object, access attributes directly
                    if hasattr(lm, "x") and hasattr(lm, "y"):
                        x = lm.x
                        y = lm.y
                        z = getattr(lm, "z", 0.0)
                        visibility = getattr(lm, "visibility", 0.0)
                    else:
                        # fallback for tuple/list
                        x = lm[0]
                        y = lm[1]
                        z = lm[2] if len(lm) > 2 else 0.0
                        visibility = lm[3] if len(lm) > 3 else 0.0
                    flat_landmarks.extend([lm.x, lm.y, getattr(lm, 'z', 0.0), lm.visibility])
        else:
            # Handle mediapipe NormalizedLandmarkList
            flat_landmarks = []
            for lm in landmarks.landmark:
                flat_landmarks.extend([lm.x, lm.y, getattr(lm, 'z', 0.0), lm.visibility])
        # print("flat_landmarks", flat_landmarks)

        # assign the subset of landmarks to the flat_landmarks_subset
        if self.CLUSTER_TYPE == "fingertips_positions":
            flat_landmarks = [flat_landmarks[i] for i in self.SUBSET_LANDMARKS]
            # print("flat_landmarks", flat_landmarks)
        # print("flat_landmarks_subset", flat_landmarks) 
        return flat_landmarks


# FUSION STUFF

    def find_sorted_zero_indices(self, topic_no,min_value):
        folder_path='utilities/data/october_fusion_clusters'

        # Construct the file name and path
        file_name = 'topic_' + str(topic_no[0]) + '.csv'
        file_path = os.path.join(folder_path, file_name)
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path, header=None)
        # Check if the first column of the first row is a string


        if df.iloc[0, 0]=="ihp_cluster":
            # Drop the first column and the first row
            df = df.iloc[1:, 1:].reset_index(drop=True)
            # then convert all values to int
            df = df.astype(float)
            df = df.astype(int)
        if isinstance(df.iloc[0, 0], str):
            print("first column is a string,", print(df.iloc[0, 0]))
        
        # Convert the DataFrame to a NumPy array
        gesture_array = df.to_numpy()

        # Optionally, you can check the shape of the array
        # print("Shape of the array:", gesture_array.shape)
        # print(gesture_array)  # Print the array to verify its contents

        # Find the indices where elements are zero
        zero_indices = np.argwhere(gesture_array >min_value)
        
        # Convert the list of zero indices to a NumPy array
        zero_indices_array = np.array(zero_indices)

        # Sort first by axis 0 (rows), then by axis 1 (columns)
        sorted_zero_indices = zero_indices_array[np.lexsort((zero_indices_array[:,1], zero_indices_array[:,0]))]

        # Convert back to a list (if required)
        sorted_zero_indices_list = sorted_zero_indices.tolist()

        # Return the sorted list of zero indices
        return sorted_zero_indices_list


# Mass Build file management stuff

    def trim_top_crop(self, image, amount=.25):
        height, width = image.shape[:2]
        output_trim = int(self.EXISTING_CROPABLE_DIM * amount)
        if height == (self.EXISTING_CROPABLE_DIM-output_trim) and width == self.EXISTING_CROPABLE_DIM:
            print(f"Already cropped:")
            return None
        elif height == self.EXISTING_CROPABLE_DIM and width == self.EXISTING_CROPABLE_DIM:
            # remove output_trim pixels from the top, no changes to the width
            cropped_img = image[output_trim:, :width]
            return cropped_img
        else:
            print(f"Image dimensions do not match expected: {height}x{width}")
            return None
