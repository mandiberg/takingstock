from sqlalchemy import create_engine, select, text, bindparam, Column, Integer, Float, ForeignKey, BLOB
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from my_declarative_base import Base, Images, Detections
import pickle
import numpy as np
import json

class ToolsClustering:
    """Store key clustering info for use across codebase"""

    def __init__(self, CLUSTER_TYPE, session=None, VERBOSE=False):
        self.VERBOSE = VERBOSE
        self.CLUSTER_TYPE = CLUSTER_TYPE
        self.CLUSTER_MEDIANS = None
        self.session = session
        # Object-hand relationship constants
        self.TOUCH_THRESHOLD = 0.5  # face height units
        self.CLASS_ID_WEIGHT = 10  # multiplier to give more weight to class_id in clustering (since it's categorical and we want it to separate well)
        self.OVERLAP_IOU_THRESHOLD = 0.5
        self.HIGH_CONFIDENCE_THRESHOLD = 0.9
        self.CONFIDENCE_DIFF_THRESHOLD = 0.3
        self.MIN_DETECTION_CONFIDENCE = 0.4
        self.DEFAULT_HAND_POSITION = [0.0, 8.0, 0.0]
        self.USE_WHITELIST = True
        all_class_ids = set(range(0, 104))

        nonsense_class_ids_dict = {
            # Hand: exclude things that are basically never hand-held in your dataset.
            'hand': {
                # 1, 2, 3, 4, 5, 6, 7, 8,             # vehicles
                9, 10, 11, 12, 13,                  # street fixtures / public infrastructure
                # 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # animals
                # 56, 57, 58, 59, 60, 61, 62,         # furniture / room fixtures
                # 68, 69, 70, 71, 72,                 # appliances / kitchen fixtures
            },

            # Left eye: be much stricter here; large/background classes are usually nonsense in the eye zone.
            'left_eye': {
                1, 2, 3, 4, 5, 6, 7, 8,             # vehicles
                9, 10, 11, 12, 13,                  # street fixtures / public infrastructure
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # animals
                24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,  # bags, luggage, and sports gear
                42, 43, 44, 45,                     # utensils / bowl
                46, 47, 48, 49, 50, 51, 52, 53, 54, 55,  # food items
                56, 57, 58, 59, 60, 61, 62,         # furniture / room fixtures / tv
                68, 69, 70, 71, 72,                 # appliances / kitchen fixtures
                74, 75, 77, 78,                     # decor / stuffed object / hair appliance
            },

            # Right eye: same logic as left eye; keep this strict because weird background hits show up here easily.
            'right_eye': {
                1, 2, 3, 4, 5, 6, 7, 8,             # vehicles
                9, 10, 11, 12, 13,                  # street fixtures / public infrastructure
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # animals
                24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,  # bags, luggage, and sports gear
                42, 43, 44, 45,                     # utensils / bowl
                46, 47, 48, 49, 50, 51, 52, 53, 54, 55,  # food items
                56, 57, 58, 59, 60, 61, 62,         # furniture / room fixtures / tv
                68, 69, 70, 71, 72,                 # appliances / kitchen fixtures
                74, 75, 77, 78,                     # decor / stuffed object / hair appliance
            },

            # Top face: exclude large scene/background classes, but keep small handheld occluders plausible.
            'top_face': {
                1, 2, 3, 4, 5, 6, 7, 8,             # vehicles
                9, 10, 11, 12, 13,                  # street fixtures / public infrastructure
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # animals
                30, 31, 36, 37,                     # long outdoor sports gear / boards
                56, 57, 58, 59, 60, 61, 62,         # furniture / room fixtures / tv
                68, 69, 70, 71, 72,                 # appliances / kitchen fixtures
            },

            # Mouth: exclude large/background classes, but keep food, drink, and small handheld occluders available.
            'mouth': {
                1, 2, 3, 4, 5, 6, 7, 8,             # vehicles
                9, 10, 11, 12, 13,                  # street fixtures / public infrastructure
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # animals
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38,  # sports gear and long outdoor equipment
                56, 57, 58, 59, 60, 61, 62,         # furniture / room fixtures / tv
                68, 69, 70, 71, 72,                 # appliances / kitchen fixtures
                74, 75, 77,                         # decor / stuffed object
            },

            # Shoulder: keep backpack, handbag, tie, laptop, phone, and book plausible; drop most other background/object classes.
            'shoulder': {
                1, 2, 3, 4, 5, 6, 7, 8,             # vehicles
                9, 10, 11, 12, 13,                  # street fixtures / public infrastructure
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # animals
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38,  # sports gear and long outdoor equipment
                # 39, 40, 41, 42, 43, 44, 45,         # drinkware / utensils / bowl
                # 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,  # food items
                56, 57, 58, 59, 60, 61, 62,         # furniture / room fixtures / tv
                64, 65, 66,                         # desktop peripherals
                68, 69, 70, 71, 72,                 # appliances / kitchen fixtures
                74, 75, 76, 77, 78, 79,             # decor / scissors / stuffed object / bathroom items
            },

            # Waist: tuned to keep seating/support objects plausible (chair/couch/bed), while excluding obvious nonsense.
            'waist': {
                1, 2, 3, 4, 5, 6, 7, 8,             # vehicles
                9, 10, 11, 12, 13,                  # street fixtures / public infrastructure
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # animals
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38,  # sports gear and long outdoor equipment
                39, 40, 41, 42, 43, 44, 45,         # drinkware / utensils / bowl
                46, 47, 48, 49, 50, 51, 52, 53, 54, 55,  # food items
                58, 61, 62,                         # plant / toilet / tv
                64, 65, 66,                         # desktop peripherals
                68, 69, 70, 71, 72,                 # appliances / kitchen fixtures
                74, 75, 76, 77, 78, 79,             # decor / scissors / stuffed object / bathroom items
            },

            # Feet: broad and intentionally permissive to allow occasional meaningful lower-body objects (e.g., skis).
            'feet': {
                1, 2, 3, 4, 5, 6, 7, 8,             # vehicles
                9, 10, 11, 12, 13,                  # street fixtures / public infrastructure
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # animals
                42, 43, 44,                         # utensils
                46, 47, 48, 49, 50, 51, 52, 53, 54, 55,  # food items
                68, 69, 70, 71, 72,                 # appliances / kitchen fixtures
                74, 75, 77, 78,                     # decor / stuffed object / hair appliance
            },
        }

        self.WHITELIST_BY_SLOT = {
            slot_name: all_class_ids - nonsense_class_ids
            for slot_name, nonsense_class_ids in nonsense_class_ids_dict.items()
        }
        self._whitelist_slots = tuple(self.WHITELIST_BY_SLOT.keys())
        self._whitelist_reject_counts = {slot: 0 for slot in self._whitelist_slots}
        # Face object constraints to avoid large background objects
        self.MAX_FACE_WIDTH = 2.0  # max width of left+right to be considered face object
        self.MAX_FACE_VERT_EXTENSION = 0.75  # max how far object can extend into opposite zone
        self.EYE_ZONE_TOP = -0.65
        self.EYE_ZONE_BOTTOM = 0.15
        self.LEFT_EYE_X_MIN = -0.9
        self.LEFT_EYE_X_MAX = -0.05
        self.RIGHT_EYE_X_MIN = 0.05
        self.RIGHT_EYE_X_MAX = 0.9
        self.FULL_FACE_MASK_TOP_MAX = -0.15
        self.FULL_FACE_MASK_BOTTOM_MIN = 0.15
        self.WAIST_ZONE_TOP = 0.75
        self.WAIST_ZONE_BOTTOM = 2.25
        self.WAIST_X_MIN = -1.40
        self.WAIST_X_MAX = 1.40
        self.FEET_ZONE_TOP = 2.10
        self.FEET_ZONE_BOTTOM = 6.00
        self.FEET_X_MIN = -2.00
        self.FEET_X_MAX = 2.00
        
        # Feature standardization settings for ObjectFusion
        self.USE_FEATURE_STANDARDIZATION = True  # Use StandardScaler to normalize all features to similar scale
        self.FEATURE_WEIGHTS = {
            'face_angle': 0.3,      # pitch, yaw, roll - LOW weight (prevent face-angle mega-clusters)
            'class_id': 5.0,        # class_id - VERY HIGH weight at RAW SCALE (0-107) to force object-type separation
            'confidence': 0.2,      # detection confidence - very low weight
            'bbox': 1.0,            # bbox coordinates - reduced to standard weight
            'has_object': 3.0,      # binary indicator - high weight but lower than class_id
        }
        # Store fitted scaler for inverse transform during median calculation
        self.feature_scaler = None
        self.CLUSTER_DATA = {
            "BodyPoses": {"data_column": "mongo_body_landmarks", "is_feet": 1, "mongo_hand_landmarks": None},
            "BodyPoses3D": {"data_column": "mongo_body_landmarks_3D", "is_feet": 1, "mongo_hand_landmarks": None}, # changed this for testing
            "ArmsPoses3D": {"data_column": "mongo_body_landmarks_3D", "is_feet": None, "mongo_hand_landmarks": 1},
            "ObjectFusion": {"data_column": "mongo_body_landmarks_norm", "is_feet": None, "mongo_hand_landmarks": None},
            "HandsGestures": {"data_column": "mongo_hand_landmarks", "is_feet": None, "mongo_hand_landmarks": 1},
            "HandsPositions": {"data_column": "mongo_hand_landmarks_norm", "is_feet": None, "mongo_hand_landmarks": 1},
            "FingertipsPositions": {"data_column": "mongo_hand_landmarks_norm", "is_feet": None, "mongo_hand_landmarks": 1},
            "HSV": {"data_column": ["hue", "sat", "val"], "is_feet": None},
}

    def get_cluster_medians(self, session, Clusters, USE_SUBSET_MEDIANS=False, SUBSET_LANDMARKS=None):
        print("getting cluster medians")
        # Create a SQLAlchemy select statement
        select_query = select(Clusters.cluster_id, Clusters.cluster_median)

        # Execute the query using your SQLAlchemy session
        results = session.execute(select_query)
        median_dict = {}

        # Process the results as needed
        # print(f'Found {len(results)} clusters with medians.')
        # print("results is a sqlalchemy.engine.result.ChunkedIteratorResult object at 0x3233d5400. this is how many: ",len(results))
        for row in results:
            # print(row)
            cluster_id, cluster_median_pickle = row
            # print("cluster_id: ",cluster_id)

            # import pprint
            # pp = pprint.PrettyPrinter(indent=4)
            # print the pickle using pprint
            # pp.pprint(cluster_median_pickle)

            cluster_median = pickle.loads(cluster_median_pickle)
            # print(f"cluster_median {cluster_id}: ", cluster_median)
            # Check the type and content of the deserialized object
            if not isinstance(cluster_median, np.ndarray):
                print(f"Deserialized object is not a numpy array, it's of type: {type(cluster_median)}")

            if USE_SUBSET_MEDIANS:
                # handles body lms subsets
                # print("handling body lms subsets in get_cluster_medians for these subset landmarks: ",sort.SUBSET_LANDMARKS)
                subset_cluster_median = []
                for i in range(len(cluster_median)):
                    # print("i: ",i)
                    if i in SUBSET_LANDMARKS:
                        # print("adding i: ",i)
                        subset_cluster_median.append(cluster_median[i])
                cluster_median = subset_cluster_median
                print(f"subset cluster median {cluster_id}: ",cluster_median)
            median_dict[cluster_id] = cluster_median
        # print("median dict: ",median_dict)
        self.CLUSTER_MEDIANS = median_dict
        return median_dict 

    def get_meta_cluster_dict(self, session, ClustersMetaClusters):
        print("getting cluster medians")
        # Create a SQLAlchemy select statement
        select_query = select(ClustersMetaClusters.cluster_id, ClustersMetaClusters.meta_cluster_id)

        # Execute the query using your SQLAlchemy session
        results = session.execute(select_query)
        meta_cluster_dict = {}

        # Process the results as needed
        # print(f'Found {len(results)} clusters with medians.')
        # print("results is a sqlalchemy.engine.result.ChunkedIteratorResult object at 0x3233d5400. this is how many: ",len(results))
        for row in results:
            # print(row)
            cluster_id, meta_cluster_id = row
            meta_cluster_dict[cluster_id] = meta_cluster_id

        return meta_cluster_dict 


    def set_table_cluster_type(self, META):
        self.META = META
            # set cluster_table_type for ArmsPoses3D, so it pulls from BodyPoses3D table
        # this allows the ArmsPoses3D value to set the Dict and subset landmarks.
        if self.CLUSTER_TYPE == "ArmsPoses3D":
            table_cluster_type = self.CLUSTER_TYPE
        else:
            if self.META:table_cluster_type = "BodyPoses3D"
            else:table_cluster_type = self.CLUSTER_TYPE
        return table_cluster_type
    
    def construct_table_classes(self, table_cluster_type):
        # handle the table objects based on cl.CLUSTER_TYPE
        ClustersTable_name = table_cluster_type
        ImagesClustersTable_name = "Images"+table_cluster_type
        MetaClustersTable_name = "Meta"+table_cluster_type
        ClustersMetaClustersTable_name = "Clusters"+MetaClustersTable_name

        print("ClustersTable_name: ",ClustersTable_name, " ImagesClustersTable_name: ",ImagesClustersTable_name, " MetaClustersTable_name: ",MetaClustersTable_name, " ClustersMetaClustersTable_name: ",ClustersMetaClustersTable_name)

        class Clusters(Base):
            # this doubles as MetaClusters
            __tablename__ = ClustersTable_name

            cluster_id = Column(Integer, primary_key=True, autoincrement=True)
            cluster_median = Column(BLOB)

        class ImagesClusters(Base):
            __tablename__ = ImagesClustersTable_name

            image_id = Column(Integer, ForeignKey(Images.image_id, ondelete="CASCADE"), primary_key=True)
            cluster_id = Column(Integer, ForeignKey(f'{ClustersTable_name}.cluster_id', ondelete="CASCADE"))
            cluster_dist = Column(Float)

        class MetaClusters(Base):
            __tablename__ = MetaClustersTable_name

            cluster_id = Column(Integer, primary_key=True, autoincrement=True)
            cluster_median = Column(BLOB)

        class ClustersMetaClusters(Base):
            __tablename__ = ClustersMetaClustersTable_name
            # cluster_id is pkey, and meta_cluster_id will appear multiple times for multiple clusters
            cluster_id = Column(Integer, ForeignKey(f'{ClustersTable_name}.cluster_id', ondelete="CASCADE"), primary_key=True)
            meta_cluster_id = Column(Integer, ForeignKey(f'{MetaClustersTable_name}.cluster_id', ondelete="CASCADE"))
            cluster_dist = Column(Float)
        return Clusters, ImagesClusters, MetaClusters, ClustersMetaClusters
    
    def set_cluster_metacluster(self, Clusters, ImagesClusters, MetaClusters, ClustersMetaClusters):
        if self.META:
            this_Cluster = MetaClusters
            this_CrosswalkClusters = ClustersMetaClusters
        else:
            this_Cluster = Clusters
            this_CrosswalkClusters = ImagesClusters
        return this_Cluster, this_CrosswalkClusters

    def prep_pose_clusters_enc(self, enc1, median_dict=None):
        # print("prepping pose clusters for enc1: ", enc1, " with median_dict: ", median_dict, " and CLUSTER_MEDIANS: ", self.CLUSTER_MEDIANS)
        if median_dict is None and self.CLUSTER_MEDIANS is not None:
            median_dict = self.CLUSTER_MEDIANS
        # print("current image enc1", enc1)  
        enc1 = np.array(enc1)
        
        this_dist_dict = {}
        for cluster_id in median_dict:
            enc2 = median_dict[cluster_id]
            # print("cluster_id enc2: ", cluster_id,enc2)
            this_dist_dict[cluster_id] = np.linalg.norm(enc1 - enc2, axis=0)
        
        cluster_id, cluster_dist = min(this_dist_dict.items(), key=lambda x: x[1])

        # print(cluster_id)
        return cluster_id, cluster_dist

    # ==================== OBJECT-HAND RELATIONSHIP METHODS ====================

    def parse_bbox_norm(self, bbox_norm):
        """Parse bbox_norm from various formats to dict."""
        if bbox_norm is None:
            return None
        if isinstance(bbox_norm, dict):
            return bbox_norm
        if isinstance(bbox_norm, str):
            try:
                # Handle double-encoded JSON
                parsed = json.loads(bbox_norm)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                return parsed
            except:
                return None
        return None

    def detection_to_list(self, detection_dict):
        """Convert detection dict to 6-element list: [class_id, conf, top, left, right, bottom]."""
        if detection_dict is None:
            return None
        return [
            detection_dict['class_id'],
            detection_dict['conf'],
            detection_dict['top'],
            detection_dict['left'],
            detection_dict['right'],
            detection_dict['bottom']
        ]

    def detection_to_payload(self, detection_dict):
        """Convert detection dict to a compact payload for DataFrame/DB use."""
        if detection_dict is None:
            return None

        return {
            'detection_id': int(detection_dict['detection_id']),
            'class_id': float(detection_dict['class_id']),
            'conf': float(detection_dict['conf']),
            'top': float(detection_dict['top']),
            'left': float(detection_dict['left']),
            'right': float(detection_dict['right']),
            'bottom': float(detection_dict['bottom']),
        }

    def _normalize_class_id(self, class_id_value):
        """Normalize class_id value to int if possible."""
        if class_id_value is None:
            return None
        try:
            return int(float(class_id_value))
        except (TypeError, ValueError):
            return None

    def _get_detection_class_id(self, detection_dict):
        """Get stable class_id for filtering, preferring raw class ID if available."""
        class_id_value = detection_dict.get('class_id_raw', detection_dict.get('class_id'))
        return self._normalize_class_id(class_id_value)

    def _passes_slot_whitelist(self, detection_dict, slot_name):
        """Check whitelist eligibility for a given slot."""
        if not self.USE_WHITELIST:
            return True

        allowed_class_ids = self.WHITELIST_BY_SLOT.get(slot_name)
        if allowed_class_ids is None:
            return True

        class_id_value = self._get_detection_class_id(detection_dict)
        if class_id_value is None:
            return False
        return class_id_value in allowed_class_ids

    def _record_whitelist_reject(self, slot_name):
        """Increment whitelist reject counter for one slot."""
        if not self.USE_WHITELIST:
            return
        if slot_name in self._whitelist_reject_counts:
            self._whitelist_reject_counts[slot_name] += 1

    def reset_whitelist_reject_counts(self):
        """Reset per-batch whitelist reject counters."""
        self._whitelist_reject_counts = {slot: 0 for slot in self._whitelist_slots}

    def get_whitelist_reject_counts(self):
        """Return a copy of current whitelist reject counters."""
        return dict(self._whitelist_reject_counts)

    def calc_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bboxes."""
        x_left = max(bbox1['left'], bbox2['left'])
        x_right = min(bbox1['right'], bbox2['right'])
        y_top = max(bbox1['top'], bbox2['top'])
        y_bottom = min(bbox1['bottom'], bbox2['bottom'])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (bbox1['right'] - bbox1['left']) * (bbox1['bottom'] - bbox1['top'])
        area2 = (bbox2['right'] - bbox2['left']) * (bbox2['bottom'] - bbox2['top'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def point_to_bbox_distance(self, point, bbox):
        """
        Calculate minimum distance from a point (x, y) to bbox edges.
        Returns 0 if point is inside bbox, otherwise returns distance to nearest edge.
        """
        px, py = point[0], point[1]  # x, y from knuckle coords
        
        # Check if inside bbox
        if bbox['left'] <= px <= bbox['right'] and bbox['top'] <= py <= bbox['bottom']:
            return 0.0
        
        # Calculate distance to each edge
        dx = max(bbox['left'] - px, 0, px - bbox['right'])
        dy = max(bbox['top'] - py, 0, py - bbox['bottom'])
        
        return (dx**2 + dy**2)**0.5

    def is_touching_hand(self, knuckle_pos, bbox):
        """Check if knuckle is within TOUCH_THRESHOLD of bbox."""
        if knuckle_pos == self.DEFAULT_HAND_POSITION:
            return False
        dist = self.point_to_bbox_distance(knuckle_pos, bbox)
        return dist <= self.TOUCH_THRESHOLD

    def is_top_face_object(self, bbox):
        """Check if object is on top of face (above nose, spans both sides)."""
        # Top of bbox is above nose (in this coord system, negative y is above)
        # AND bbox spans both sides of nose (has both positive and negative x)
        # AND object is not too wide (max width constraint)
        # AND object doesn't extend too far into bottom zone
        width = bbox['right'] - bbox['left']
        extends_into_bottom = max(0, bbox['bottom'])  # how far does it go into positive y
        
        return (bbox['top'] < 0 and 
                bbox['left'] < 0 and 
                bbox['right'] > 0 and
                width <= self.MAX_FACE_WIDTH and
                extends_into_bottom <= self.MAX_FACE_VERT_EXTENSION)

    def is_bottom_face_object(self, bbox):
        """Check if object is on bottom of face (below nose, spans both sides)."""
        # Bottom of bbox is below nose (positive y)
        # AND bbox spans both sides of nose
        # AND object is not too wide (max width constraint)
        # AND object doesn't extend too far into top zone
        width = bbox['right'] - bbox['left']
        extends_into_top = max(0, -bbox['top'])  # how far does it go into negative y
        
        return (bbox['bottom'] > 0 and 
                bbox['left'] < 0 and 
                bbox['right'] > 0 and
                width <= self.MAX_FACE_WIDTH and
                extends_into_top <= self.MAX_FACE_VERT_EXTENSION)

    def is_mouth_object(self, bbox):
        """Check if object is on mouth area with zero top-zone tolerance."""
        width = bbox['right'] - bbox['left']
        extends_into_top = max(0, -bbox['top'])

        return (bbox['bottom'] > 0 and
                bbox['left'] < 0 and
                bbox['right'] > 0 and
                width <= self.MAX_FACE_WIDTH and
                extends_into_top <= 0.0)

    def _bbox_intersects_rect(self, bbox, x_min, x_max, y_top, y_bottom):
        """Check whether bbox intersects a rectangular zone."""
        return not (
            bbox['right'] < x_min
            or bbox['left'] > x_max
            or bbox['bottom'] < y_top
            or bbox['top'] > y_bottom
        )

    def is_full_face_mask_object(self, bbox):
        """Check if bbox covers both eyes+mouth region (full-face sheet mask style)."""
        width = bbox['right'] - bbox['left']
        return (
            bbox['left'] < -0.15
            and bbox['right'] > 0.15
            and bbox['top'] <= self.FULL_FACE_MASK_TOP_MAX
            and bbox['bottom'] >= self.FULL_FACE_MASK_BOTTOM_MIN
            and width <= self.MAX_FACE_WIDTH
        )

    def is_left_eye_object(self, bbox):
        """Check if bbox intersects the left-eye zone."""
        return self._bbox_intersects_rect(
            bbox,
            self.LEFT_EYE_X_MIN,
            self.LEFT_EYE_X_MAX,
            self.EYE_ZONE_TOP,
            self.EYE_ZONE_BOTTOM,
        )

    def is_right_eye_object(self, bbox):
        """Check if bbox intersects the right-eye zone."""
        return self._bbox_intersects_rect(
            bbox,
            self.RIGHT_EYE_X_MIN,
            self.RIGHT_EYE_X_MAX,
            self.EYE_ZONE_TOP,
            self.EYE_ZONE_BOTTOM,
        )

    def is_waist_object(self, bbox):
        """Check if bbox intersects a broad waist/seat interaction zone."""
        return self._bbox_intersects_rect(
            bbox,
            self.WAIST_X_MIN,
            self.WAIST_X_MAX,
            self.WAIST_ZONE_TOP,
            self.WAIST_ZONE_BOTTOM,
        )

    def is_feet_object(self, bbox):
        """Check if bbox intersects a broad lower-body/feet zone."""
        return self._bbox_intersects_rect(
            bbox,
            self.FEET_X_MIN,
            self.FEET_X_MAX,
            self.FEET_ZONE_TOP,
            self.FEET_ZONE_BOTTOM,
        )

    def _extract_xy_from_landmark(self, landmark):
        """Extract (x, y) from a landmark in object/dict/list form."""
        if landmark is None:
            return None

        if hasattr(landmark, 'x') and hasattr(landmark, 'y'):
            return [float(landmark.x), float(landmark.y)]

        if isinstance(landmark, dict):
            if 'x' in landmark and 'y' in landmark:
                return [float(landmark['x']), float(landmark['y'])]
            return None

        if isinstance(landmark, (list, tuple, np.ndarray)) and len(landmark) >= 2:
            return [float(landmark[0]), float(landmark[1])]

        return None

    def extract_shoulder_points(self, body_landmarks_normalized):
        """Extract left/right shoulder points (landmarks 11/12) as [x, y]."""
        if body_landmarks_normalized is None:
            return None, None

        try:
            landmarks = None
            if hasattr(body_landmarks_normalized, 'landmark'):
                landmarks = body_landmarks_normalized.landmark
            elif isinstance(body_landmarks_normalized, (list, tuple, np.ndarray)):
                landmarks = body_landmarks_normalized

            if landmarks is None or len(landmarks) <= 12:
                return None, None

            left_shoulder = self._extract_xy_from_landmark(landmarks[11])
            right_shoulder = self._extract_xy_from_landmark(landmarks[12])
            return left_shoulder, right_shoulder
        except Exception:
            return None, None

    def is_shoulder_object(self, bbox, left_shoulder, right_shoulder):
        """
        Check if object crosses the shoulder band.
        Shoulder band is the line from lm11 to lm12, extended 1.0 unit lower.
        """
        if left_shoulder is None or right_shoulder is None:
            return False

        x1, y1 = left_shoulder
        x2, y2 = right_shoulder

        shoulder_x_min = min(x1, x2)
        shoulder_x_max = max(x1, x2)

        if bbox['right'] < shoulder_x_min or bbox['left'] > shoulder_x_max:
            return False

        overlap_left = max(bbox['left'], shoulder_x_min)
        overlap_right = min(bbox['right'], shoulder_x_max)
        if overlap_right < overlap_left:
            return False

        if abs(x2 - x1) < 1e-9:
            shoulder_y_min = min(y1, y2)
            shoulder_y_max = max(y1, y2)
        else:
            def y_on_shoulder_line(x_val):
                t = (x_val - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)

            y_left = y_on_shoulder_line(overlap_left)
            y_right = y_on_shoulder_line(overlap_right)
            y_mid = y_on_shoulder_line((overlap_left + overlap_right) / 2.0)
            shoulder_y_min = min(y_left, y_right, y_mid)
            shoulder_y_max = max(y_left, y_right, y_mid)

        band_top = shoulder_y_min
        band_bottom = shoulder_y_max + 1.0

        return not (bbox['bottom'] < band_top or bbox['top'] > band_bottom)

    def resolve_overlapping_detections(self, detections):
        """
        Resolve overlapping detections by keeping the best one based on confidence rules.
        Returns filtered list of detections.
        """
        if len(detections) <= 1:
            return detections
        
        filtered = []
        used_indices = set()
        
        for i, det1 in enumerate(detections):
            if i in used_indices:
                continue
                
            best_det = det1
            
            for j, det2 in enumerate(detections):
                if j <= i or j in used_indices:
                    continue
                    
                iou = self.calc_iou(det1['bbox'], det2['bbox'])
                
                if iou >= self.OVERLAP_IOU_THRESHOLD:
                    # Overlapping detections - resolve based on confidence
                    conf1, conf2 = det1['conf'], det2['conf']
                    conf_diff = abs(conf1 - conf2)
                    
                    conf_diff_threshold_scaled_to_iou = self.CONFIDENCE_DIFF_THRESHOLD - (self.CONFIDENCE_DIFF_THRESHOLD * iou)
                    if conf1 >= self.HIGH_CONFIDENCE_THRESHOLD or conf2 >= self.HIGH_CONFIDENCE_THRESHOLD or conf_diff >= conf_diff_threshold_scaled_to_iou:
                        winner = det1 if conf1 >= conf2 else det2
                        loser = det2 if conf1 >= conf2 else det1
                        if self.VERBOSE:
                            print(f"  ✅ OVERLAP RESOLVED: Chose class {winner['class_id']} (conf={winner['conf']:.2f}) over class {loser['class_id']} (conf={loser['conf']:.2f}), IoU={iou:.2f}")
                        best_det = winner
                        used_indices.add(j)
                    
                    if det1['class_id'] == det2['class_id']:
                        # same class...
                        if iou >= self.OVERLAP_IOU_THRESHOLD*1.5:
                            # take the union of the boxes
                            new_bbox = {
                                'top': min(det1['bbox']['top'], det2['bbox']['top']),
                                'left': min(det1['bbox']['left'], det2['bbox']['left']),
                                'right': max(det1['bbox']['right'], det2['bbox']['right']),
                                'bottom': max(det1['bbox']['bottom'], det2['bbox']['bottom']),
                            }
                            best_det['bbox'] = new_bbox
                            print(f"  🔄 MERGED SAME CLASS OVERLAP: class {det1['class_id']} (conf={det1['conf']:.2f}) and class {det2['class_id']} (conf={det2['conf']:.2f}), IoU={iou:.2f} - merged bbox")
                        else:
                            # keep higher confidence
                            winner = det1 if conf1 >= conf2 else det2
                            loser = det2 if conf1 >= conf2 else det1
                            print(f"  ⚠️ SAME CLASS OVERLAP RESOLVED: Chose class {winner['class_id']} (conf={winner['conf']:.2f}) over class {loser['class_id']} (conf={loser['conf']:.2f}), IoU={iou:.2f}, conf_diff={conf_diff:.2f}")
                            best_det = winner
                        used_indices.add(j)

                    elif conf1 >= self.MIN_DETECTION_CONFIDENCE*1.5 or conf2 >= self.MIN_DETECTION_CONFIDENCE*1.5:
                        # both moderate confidence, keep higher
                        winner = det1 if conf1 >= conf2 else det2
                        loser = det2 if conf1 >= conf2 else det1
                        if self.VERBOSE:
                            print(f"  ❌ HIGH CONF RESOLVED: Chose class {winner['class_id']} (conf={winner['conf']:.2f}) over class {loser['class_id']} (conf={loser['conf']:.2f}), IoU={iou:.2f}")
                        best_det = winner
                        used_indices.add(j)

                    else:
                        # Cannot determine - alert and discard both
                        if self.VERBOSE:
                            print(f"  🚨 OVERLAP UNRESOLVED - DISCARDING: classes {det1['class_id']} (conf={conf1:.2f}) and {det2['class_id']} (conf={conf2:.2f}), IoU={iou:.2f} - keeping both")
            

            filtered.append(best_det)
            used_indices.add(i)
        
        return filtered

    def weight_detection_for_clustering(self, detections):
        """
        Apply weighting to detections features based on class_id for clustering.
        Note: If USE_FEATURE_STANDARDIZATION=True, this weight is applied BEFORE standardization,
        then features are standardized, then FEATURE_WEIGHTS are applied after.
        If USE_FEATURE_STANDARDIZATION=False, only CLASS_ID_WEIGHT is used (legacy behavior).
        """
        if not self.USE_FEATURE_STANDARDIZATION:
            # Legacy behavior: simple multiplication
            for det in detections:
                det['class_id'] *= self.CLASS_ID_WEIGHT
        else:
            # New behavior: CLASS_ID_WEIGHT is incorporated into FEATURE_WEIGHTS['class_id']
            # Don't multiply here - let standardization handle it
            pass
        return detections
    
    def classify_object_hand_relationships(self, detections, left_knuckle, right_knuckle, left_shoulder=None, right_shoulder=None):
        """
        Classify each detection based on its relationship to hands and face.
        Returns dict with keys: left_hand_object, right_hand_object,
                               top_face_object, left_eye_object, right_eye_object,
                               mouth_object, shoulder_object, waist_object, feet_object
        Each value is the detection dict or None.
        """

        # weight the class_ids for better clustering separation of classes
        detections = self.weight_detection_for_clustering(detections)
        
        results = {
            'left_hand_object': None,
            'right_hand_object': None,
            'top_face_object': None,
            'left_eye_object': None,
            'right_eye_object': None,
            'mouth_object': None,
            'shoulder_object': None,
            'waist_object': None,
            'feet_object': None,
        }
        
        if not detections:
            return results
        
        # First, resolve overlapping detections
        detections = self.resolve_overlapping_detections(detections)

        # 1. Find best object for each hand independently (same object may be assigned to both)
        def best_object_for_hand(knuckle):
            if knuckle == self.DEFAULT_HAND_POSITION:
                return None

            touching_candidates = []
            nearby_candidates = []

            for det in detections:
                if not self._passes_slot_whitelist(det, 'hand'):
                    self._record_whitelist_reject('hand')
                    continue
                dist = self.point_to_bbox_distance(knuckle, det['bbox'])
                if dist <= self.TOUCH_THRESHOLD:
                    touching_candidates.append((dist, det))
                elif dist <= self.TOUCH_THRESHOLD * 2:
                    nearby_candidates.append((dist, det))

            if touching_candidates:
                touching_candidates.sort(key=lambda item: item[0])
                return touching_candidates[0][1]

            if nearby_candidates:
                nearby_candidates.sort(key=lambda item: item[0])
                return nearby_candidates[0][1]

            return None

        results['left_hand_object'] = best_object_for_hand(left_knuckle)
        results['right_hand_object'] = best_object_for_hand(right_knuckle)

        # Hand-assigned objects are excluded from all other zones
        hand_detection_ids = set()
        for hand_key in ['left_hand_object', 'right_hand_object']:
            hand_det = results[hand_key]
            if hand_det is not None:
                hand_detection_ids.add(hand_det['detection_id'])

        non_hand_detections = [
            det for det in detections if det['detection_id'] not in hand_detection_ids
        ]

        # 2. Eye assignments (same object may map to both eyes, e.g., eyeglasses)
        for det in non_hand_detections:
            bbox = det['bbox']

            left_eye_allowed = self._passes_slot_whitelist(det, 'left_eye')
            right_eye_allowed = self._passes_slot_whitelist(det, 'right_eye')

            if not left_eye_allowed:
                self._record_whitelist_reject('left_eye')
            if not right_eye_allowed:
                self._record_whitelist_reject('right_eye')

            if not left_eye_allowed and not right_eye_allowed:
                continue

            if self.is_full_face_mask_object(bbox):
                continue

            if left_eye_allowed and self.is_left_eye_object(bbox):
                if results['left_eye_object'] is None:
                    results['left_eye_object'] = det
                elif det['conf'] > results['left_eye_object']['conf']:
                    results['left_eye_object'] = det

            if right_eye_allowed and self.is_right_eye_object(bbox):
                if results['right_eye_object'] is None:
                    results['right_eye_object'] = det
                elif det['conf'] > results['right_eye_object']['conf']:
                    results['right_eye_object'] = det

        # 3. Top-face / mouth / shoulder / waist / feet assignments
        for det in non_hand_detections:
            bbox = det['bbox']

            top_face_allowed = self._passes_slot_whitelist(det, 'top_face')
            mouth_allowed = self._passes_slot_whitelist(det, 'mouth')
            shoulder_allowed = self._passes_slot_whitelist(det, 'shoulder')
            waist_allowed = self._passes_slot_whitelist(det, 'waist')
            feet_allowed = self._passes_slot_whitelist(det, 'feet')

            if not top_face_allowed:
                self._record_whitelist_reject('top_face')
            if not mouth_allowed:
                self._record_whitelist_reject('mouth')
            if not shoulder_allowed:
                self._record_whitelist_reject('shoulder')
            if not waist_allowed:
                self._record_whitelist_reject('waist')
            if not feet_allowed:
                self._record_whitelist_reject('feet')

            if top_face_allowed and self.is_top_face_object(bbox):
                if results['top_face_object'] is None:
                    results['top_face_object'] = det
                elif bbox['top'] < results['top_face_object']['bbox']['top']:
                    results['top_face_object'] = det

            if mouth_allowed and self.is_mouth_object(bbox):
                if results['mouth_object'] is None:
                    results['mouth_object'] = det
                elif bbox['top'] < results['mouth_object']['bbox']['top']:
                    results['mouth_object'] = det

            if shoulder_allowed and self.is_shoulder_object(bbox, left_shoulder, right_shoulder):
                if results['shoulder_object'] is None:
                    results['shoulder_object'] = det
                elif det['conf'] > results['shoulder_object']['conf']:
                    results['shoulder_object'] = det

            if waist_allowed and self.is_waist_object(bbox):
                if results['waist_object'] is None:
                    results['waist_object'] = det
                elif det['conf'] > results['waist_object']['conf']:
                    results['waist_object'] = det

            if feet_allowed and self.is_feet_object(bbox):
                if results['feet_object'] is None:
                    results['feet_object'] = det
                elif det['conf'] > results['feet_object']['conf']:
                    results['feet_object'] = det
        
        return results

    def query_and_classify_detections(self, image_id, left_knuckle, right_knuckle, left_shoulder=None, right_shoulder=None):
        """
        Query detections for an image and classify their relationship to hands/face.
        Returns dict with 9 keys, each containing a detection payload dict or None.
        """
        if self.session is None:
            raise ValueError("Session not initialized. Pass session to ToolsClustering.__init__()")
        
        # Query detections
        detection_results = self.session.query(Detections).filter_by(image_id=image_id).\
            filter(Detections.conf > self.MIN_DETECTION_CONFIDENCE).all()
        
        debug = self.VERBOSE  # print full math when VERBOSE is on
        if debug:
            print(f"\n{'='*60}")
            print(f"[DEBUG] query_and_classify_detections image_id={image_id}")
            # print(f"  left_knuckle={left_knuckle}  right_knuckle={right_knuckle}")
            # print(f"  left_shoulder={left_shoulder}  right_shoulder={right_shoulder}")
            # print(f"  Raw detection_results count: {len(detection_results)}")
            # for d in detection_results:
            #     print(f"    detection_id={d.detection_id} class_id={d.class_id} conf={d.conf:.3f} bbox_norm={d.bbox_norm}")

        if not detection_results:
            if debug: print(f"  → No detections above MIN_DETECTION_CONFIDENCE={self.MIN_DETECTION_CONFIDENCE}; returning all None")
            return {
                'left_hand_object': None,
                'right_hand_object': None,
                'top_face_object': None,
                'left_eye_object': None,
                'right_eye_object': None,
                'mouth_object': None,
                'shoulder_object': None,
                'waist_object': None,
                'feet_object': None,
            }
        
        # Parse detections into standardized format
        detections = []
        for d in detection_results:
            bbox = self.parse_bbox_norm(d.bbox_norm)
            if bbox is None:
                # if debug: print(f"  ✗ detection_id={d.detection_id} — bbox_norm parse failed, skipping")
                continue
            detections.append({
                'detection_id': d.detection_id,
                'class_id': d.class_id,
                'class_id_raw': d.class_id,
                'conf': d.conf,
                'bbox': bbox,
                'top': bbox['top'],
                'left': bbox['left'],
                'right': bbox['right'],
                'bottom': bbox['bottom']
            })
            if debug:
                print(f"  ✓ detection_id={d.detection_id} parsed: class_id={d.class_id} conf={d.conf:.3f}")
                print(f"    bbox → top={bbox['top']} left={bbox['left']} right={bbox['right']} bottom={bbox['bottom']}")
                # Check each classifier
                print(f"    is_top_face_object:    {self.is_top_face_object(bbox)}")
                print(f"      (top<0={bbox['top']<0}, left<0={bbox['left']<0}, right>0={bbox['right']>0}, width={bbox['right']-bbox['left']:.4f}<=MAX={self.MAX_FACE_WIDTH}, extends_into_bottom={max(0,bbox['bottom']):.4f}<=MAX_VERT={self.MAX_FACE_VERT_EXTENSION})")
                print(f"    is_bottom_face_object: {self.is_bottom_face_object(bbox)}")
                print(f"    is_left_eye_object:    {self.is_left_eye_object(bbox)}")
                print(f"    is_right_eye_object:   {self.is_right_eye_object(bbox)}")
                print(f"    is_full_face_mask:     {self.is_full_face_mask_object(bbox)}")
                print(f"    is_mouth_object:       {self.is_mouth_object(bbox)}")
                print(f"    is_shoulder_object:    {self.is_shoulder_object(bbox, left_shoulder, right_shoulder)}")
                print(f"    is_waist_object:       {self.is_waist_object(bbox)}")
                print(f"    is_feet_object:        {self.is_feet_object(bbox)}")
                dist_left  = self.point_to_bbox_distance(left_knuckle, bbox)  if left_knuckle  != self.DEFAULT_HAND_POSITION else None
                dist_right = self.point_to_bbox_distance(right_knuckle, bbox) if right_knuckle != self.DEFAULT_HAND_POSITION else None
                print(f"    dist left_knuckle→bbox:  {dist_left}  (TOUCH_THRESHOLD={self.TOUCH_THRESHOLD})")
                print(f"    dist right_knuckle→bbox: {dist_right}")

        if debug:
            print(f"  Parsed {len(detections)} valid detections (of {len(detection_results)} raw)")

        # Classify relationships using class method
        classified = self.classify_object_hand_relationships(
            detections,
            left_knuckle,
            right_knuckle,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
        )
        if debug:
            print(f"  Classification results:")
            for k, v in classified.items():
                print(f"    {k}: {v}")
        
        # Convert to compact payload dicts for df storage / DB persistence
        result = {}
        for key, det in classified.items():
            if det is None:
                result[key] = None
            else:
                result[key] = self.detection_to_payload(det)
        
        return result

    def process_detections_for_df(self, df):
        """
        Process all detections for a dataframe and add object classification columns.
        Expects df to have: image_id, left_pointer_knuckle_norm, right_pointer_knuckle_norm
        """
        self.reset_whitelist_reject_counts()

        # Initialize new columns
        df['left_hand_object'] = None
        df['right_hand_object'] = None
        df['top_face_object'] = None
        df['left_eye_object'] = None
        df['right_eye_object'] = None
        df['mouth_object'] = None
        df['shoulder_object'] = None
        df['waist_object'] = None
        df['feet_object'] = None
        
        for idx, row in df.iterrows():
            image_id = row['image_id']
            left_knuckle = row.get('left_pointer_knuckle_norm', self.DEFAULT_HAND_POSITION)
            right_knuckle = row.get('right_pointer_knuckle_norm', self.DEFAULT_HAND_POSITION)
            
            # Handle case where knuckle data might be None or string
            if left_knuckle is None or (isinstance(left_knuckle, list) and len(left_knuckle) == 0):
                left_knuckle = self.DEFAULT_HAND_POSITION
            if right_knuckle is None or (isinstance(right_knuckle, list) and len(right_knuckle) == 0):
                right_knuckle = self.DEFAULT_HAND_POSITION

            left_shoulder, right_shoulder = self.extract_shoulder_points(row.get('body_landmarks_normalized'))
            
            # Query and classify
            classifications = self.query_and_classify_detections(
                image_id,
                left_knuckle,
                right_knuckle,
                left_shoulder=left_shoulder,
                right_shoulder=right_shoulder,
            )
            
            # Assign to df
            df.at[idx, 'left_hand_object'] = classifications['left_hand_object']
            df.at[idx, 'right_hand_object'] = classifications['right_hand_object']
            df.at[idx, 'top_face_object'] = classifications['top_face_object']
            df.at[idx, 'left_eye_object'] = classifications['left_eye_object']
            df.at[idx, 'right_eye_object'] = classifications['right_eye_object']
            df.at[idx, 'mouth_object'] = classifications['mouth_object']
            df.at[idx, 'shoulder_object'] = classifications['shoulder_object']
            df.at[idx, 'waist_object'] = classifications['waist_object']
            df.at[idx, 'feet_object'] = classifications['feet_object']
        
        return df

    def _build_detection_payload_from_sql_fields(self, detection_id, class_id, conf, bbox_norm):
        """Build detection payload dict from SQL selected fields."""
        if detection_id is None or class_id is None or conf is None or bbox_norm is None:
            return None

        bbox = self.parse_bbox_norm(bbox_norm)
        if bbox is None:
            return None

        required_keys = ['top', 'left', 'right', 'bottom']
        if not all(k in bbox for k in required_keys):
            return None

        return {
            'detection_id': int(detection_id),
            'class_id': float(class_id),
            'conf': float(conf),
            'top': float(bbox['top']),
            'left': float(bbox['left']),
            'right': float(bbox['right']),
            'bottom': float(bbox['bottom']),
        }

    def get_precomputed_detections_by_image_ids(self, image_ids):
        """
        Read precomputed detection assignments from ImagesDetections and join to Detections.
        Returns dict keyed by image_id with values containing the 9 detection payload fields.
        """
        if self.session is None:
            raise ValueError("Session not initialized. Pass session to ToolsClustering.__init__()")

        if not image_ids:
            return {}

        sql = text("""
            SELECT
                idet.image_id,

                lh.detection_id AS left_hand_detection_id,
                lh.class_id AS left_hand_class_id,
                lh.conf AS left_hand_conf,
                lh.bbox_norm AS left_hand_bbox_norm,

                rh.detection_id AS right_hand_detection_id,
                rh.class_id AS right_hand_class_id,
                rh.conf AS right_hand_conf,
                rh.bbox_norm AS right_hand_bbox_norm,

                tf.detection_id AS top_face_detection_id,
                tf.class_id AS top_face_class_id,
                tf.conf AS top_face_conf,
                tf.bbox_norm AS top_face_bbox_norm,

                le.detection_id AS left_eye_detection_id,
                le.class_id AS left_eye_class_id,
                le.conf AS left_eye_conf,
                le.bbox_norm AS left_eye_bbox_norm,

                re.detection_id AS right_eye_detection_id,
                re.class_id AS right_eye_class_id,
                re.conf AS right_eye_conf,
                re.bbox_norm AS right_eye_bbox_norm,

                mo.detection_id AS mouth_detection_id,
                mo.class_id AS mouth_class_id,
                mo.conf AS mouth_conf,
                mo.bbox_norm AS mouth_bbox_norm,

                so.detection_id AS shoulder_detection_id,
                so.class_id AS shoulder_class_id,
                so.conf AS shoulder_conf,
                so.bbox_norm AS shoulder_bbox_norm,

                wa.detection_id AS waist_detection_id,
                wa.class_id AS waist_class_id,
                wa.conf AS waist_conf,
                wa.bbox_norm AS waist_bbox_norm,

                fe.detection_id AS feet_detection_id,
                fe.class_id AS feet_class_id,
                fe.conf AS feet_conf,
                fe.bbox_norm AS feet_bbox_norm
            FROM ImagesDetections idet
            LEFT JOIN Detections lh ON idet.left_hand_object_id = lh.detection_id
            LEFT JOIN Detections rh ON idet.right_hand_object_id = rh.detection_id
            LEFT JOIN Detections tf ON idet.top_face_object_id = tf.detection_id
            LEFT JOIN Detections le ON idet.left_eye_object_id = le.detection_id
            LEFT JOIN Detections re ON idet.right_eye_object_id = re.detection_id
            LEFT JOIN Detections mo ON idet.mouth_object_id = mo.detection_id
            LEFT JOIN Detections so ON idet.shoulder_object_id = so.detection_id
            LEFT JOIN Detections wa ON idet.waist_object_id = wa.detection_id
            LEFT JOIN Detections fe ON idet.feet_object_id = fe.detection_id
            WHERE idet.image_id IN :image_ids
        """).bindparams(bindparam("image_ids", expanding=True))

        rows = self.session.execute(sql, {"image_ids": list(image_ids)}).mappings().all()

        result = {}
        for row in rows:
            image_id = row['image_id']
            result[image_id] = {
                'left_hand_object': self._build_detection_payload_from_sql_fields(
                    row['left_hand_detection_id'], row['left_hand_class_id'], row['left_hand_conf'], row['left_hand_bbox_norm']
                ),
                'right_hand_object': self._build_detection_payload_from_sql_fields(
                    row['right_hand_detection_id'], row['right_hand_class_id'], row['right_hand_conf'], row['right_hand_bbox_norm']
                ),
                'top_face_object': self._build_detection_payload_from_sql_fields(
                    row['top_face_detection_id'], row['top_face_class_id'], row['top_face_conf'], row['top_face_bbox_norm']
                ),
                'left_eye_object': self._build_detection_payload_from_sql_fields(
                    row['left_eye_detection_id'], row['left_eye_class_id'], row['left_eye_conf'], row['left_eye_bbox_norm']
                ),
                'right_eye_object': self._build_detection_payload_from_sql_fields(
                    row['right_eye_detection_id'], row['right_eye_class_id'], row['right_eye_conf'], row['right_eye_bbox_norm']
                ),
                'mouth_object': self._build_detection_payload_from_sql_fields(
                    row['mouth_detection_id'], row['mouth_class_id'], row['mouth_conf'], row['mouth_bbox_norm']
                ),
                'shoulder_object': self._build_detection_payload_from_sql_fields(
                    row['shoulder_detection_id'], row['shoulder_class_id'], row['shoulder_conf'], row['shoulder_bbox_norm']
                ),
                'waist_object': self._build_detection_payload_from_sql_fields(
                    row['waist_detection_id'], row['waist_class_id'], row['waist_conf'], row['waist_bbox_norm']
                ),
                'feet_object': self._build_detection_payload_from_sql_fields(
                    row['feet_detection_id'], row['feet_class_id'], row['feet_conf'], row['feet_bbox_norm']
                ),
            }

        return result

    def hydrate_detections_from_precomputed_table(self, df):
        """
        Fill detection classification columns from ImagesDetections + Detections.
        Returns (updated_df, missing_image_ids) where missing_image_ids are not found
        in ImagesDetections at all.
        """
        required_cols = [
            'left_hand_object', 'right_hand_object',
            'top_face_object', 'left_eye_object', 'right_eye_object',
            'mouth_object', 'shoulder_object', 'waist_object', 'feet_object'
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        image_ids = [int(x) for x in df['image_id'].dropna().unique().tolist()]
        precomputed = self.get_precomputed_detections_by_image_ids(image_ids)

        for idx, row in df.iterrows():
            image_id = row.get('image_id')
            if image_id is None:
                continue
            payload = precomputed.get(int(image_id))
            if payload is None:
                continue

            df.at[idx, 'left_hand_object'] = payload['left_hand_object']
            df.at[idx, 'right_hand_object'] = payload['right_hand_object']
            df.at[idx, 'top_face_object'] = payload['top_face_object']
            df.at[idx, 'left_eye_object'] = payload['left_eye_object']
            df.at[idx, 'right_eye_object'] = payload['right_eye_object']
            df.at[idx, 'mouth_object'] = payload['mouth_object']
            df.at[idx, 'shoulder_object'] = payload['shoulder_object']
            df.at[idx, 'waist_object'] = payload['waist_object']
            df.at[idx, 'feet_object'] = payload['feet_object']

        missing_image_ids = sorted(list(set(image_ids) - set(precomputed.keys())))
        return df, missing_image_ids

    def flatten_object_detections(self, detection_payload):
        """
        Flatten a detection payload dict into features.
        Returns a 6-element list, or [0,0,0,0,0,0] if None.
        """
        if detection_payload is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if isinstance(detection_payload, dict):
            return [
                float(detection_payload.get('class_id', 0.0)),
                float(detection_payload.get('conf', 0.0)),
                float(detection_payload.get('top', 0.0)),
                float(detection_payload.get('left', 0.0)),
                float(detection_payload.get('right', 0.0)),
                float(detection_payload.get('bottom', 0.0)),
            ]
        return list(detection_payload)

    def prepare_features_for_knn(self, df):
        """
        Flatten all columns into a single feature vector for KNN clustering.
        Handles numeric columns and object detection lists.
        Returns df with all features flattened into separate columns (columnar format).
        """
        import pandas as pd
        
        # Numeric columns to include
        numeric_cols = ['pitch', 'yaw', 'roll']
        
        # Detection columns (6 values each: class_id, conf, top, left, right, bottom)
        detection_cols = ['left_hand_object', 'right_hand_object', 'top_face_object',
                  'left_eye_object', 'right_eye_object', 'mouth_object', 'shoulder_object',
                  'waist_object', 'feet_object']
        detection_fields = ['class_id', 'conf', 'top', 'left', 'right', 'bottom']
        
        # Create feature dict by concatenating all values
        features_dict = {}
        
        # Add image_id if it exists
        if 'image_id' in df.columns:
            features_dict['image_id'] = df['image_id']
        else:
            print("Warning: 'image_id' column not found in DataFrame. It will be missing from the features.")

        # Add numeric columns
        for col in numeric_cols:
            if col in df.columns:
                features_dict[col] = df[col].apply(lambda x: float(x) if pd.notna(x) else 0.0)
        
        # Add detection features with descriptive column names
        # ALSO add binary "has_object" indicators to prevent all-zero clustering
        for det_col in detection_cols:
            if det_col in df.columns:
                # Add binary indicator: 1.0 if object present, 0.0 if not
                has_obj_col = f"{det_col}_has_object"
                features_dict[has_obj_col] = df[det_col].apply(
                    lambda x: 1.0 if x is not None else 0.0
                )
                
                # Add standard detection fields
                for i, field in enumerate(detection_fields):
                    col_name = f"{det_col}_{field}"
                    features_dict[col_name] = df[det_col].apply(
                        lambda x: self.flatten_object_detections(x)[i] if x is not None else 0.0
                    )
        
        result_df = pd.DataFrame(features_dict)
        print("prepare_features_for_knn result_df columns: ", result_df.columns)
        return result_df
    
    def prepare_features_for_knn_v2(self, df, fit_scaler=False):
        """
        Enhanced version with optional StandardScaler and per-feature-group weighting.
        
        Args:
            df: DataFrame with ObjectFusion features
            fit_scaler: If True, fit new scaler on this data. If False, use existing scaler.
                       Set True for training data, False for new data assignment.
        
        Returns:
            DataFrame with standardized and weighted features, suitable for K-means.
        """
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        
        # First, get base features using original method
        features_df = self.prepare_features_for_knn(df)
        
        if not self.USE_FEATURE_STANDARDIZATION:
            return features_df
        
        # Separate image_id before scaling
        image_id_col = None
        if 'image_id' in features_df.columns:
            image_id_col = features_df['image_id'].copy()
            features_df = features_df.drop(columns=['image_id'])
        
        # Group columns by feature type for weighted scaling
        face_angle_cols = ['pitch', 'yaw', 'roll']
        class_id_cols = [col for col in features_df.columns if col.endswith('_class_id')]
        confidence_cols = [col for col in features_df.columns if col.endswith('_conf')]
        bbox_cols = [col for col in features_df.columns if col.endswith(('_top', '_left', '_right', '_bottom'))]
        has_object_cols = [col for col in features_df.columns if col.endswith('_has_object')]
        
        # CRITICAL: Extract class_id columns BEFORE standardization (they're categorical, not continuous)
        class_id_values = features_df[class_id_cols].copy()
        
        # Remove class_id from features to be standardized
        features_to_scale = features_df.drop(columns=class_id_cols)
        
        # Standardize only continuous features (mean=0, std=1)
        if fit_scaler or self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            scaled_features = self.feature_scaler.fit_transform(features_to_scale)
            if self.VERBOSE:
                print("Fitted new StandardScaler on features (excluding class_id)")
                print(f"  Feature means: {self.feature_scaler.mean_[:5]}...")
                print(f"  Feature stds: {self.feature_scaler.scale_[:5]}...")
        else:
            scaled_features = self.feature_scaler.transform(features_to_scale)
        
        # Convert back to DataFrame to apply per-group weights
        scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale.columns, index=features_df.index)
        
        # Re-insert class_id columns at their ORIGINAL scale (not standardized)
        for col in class_id_cols:
            scaled_df[col] = class_id_values[col]
        
        # Apply feature-group-specific weights
        for col in scaled_df.columns:
            if col in face_angle_cols:
                scaled_df[col] *= self.FEATURE_WEIGHTS['face_angle']
            elif col in class_id_cols:
                scaled_df[col] *= self.FEATURE_WEIGHTS['class_id']
            elif col in confidence_cols:
                scaled_df[col] *= self.FEATURE_WEIGHTS['confidence']
            elif col in has_object_cols:
                scaled_df[col] *= self.FEATURE_WEIGHTS['has_object']
            elif col in bbox_cols:
                scaled_df[col] *= self.FEATURE_WEIGHTS['bbox']
        
        # Re-add image_id if it existed
        if image_id_col is not None:
            scaled_df.insert(0, 'image_id', image_id_col)
        
        if self.VERBOSE:
            print("Feature standardization and weighting applied:")
            print(f"  Face angles: {self.FEATURE_WEIGHTS['face_angle']}x")
            print(f"  Class IDs: {self.FEATURE_WEIGHTS['class_id']}x")
            print(f"  Has Object indicators: {self.FEATURE_WEIGHTS['has_object']}x")
            print(f"  Confidence: {self.FEATURE_WEIGHTS['confidence']}x")
            print(f"  BBox coords: {self.FEATURE_WEIGHTS['bbox']}x")
            print(f"  Final feature range: [{scaled_df.min().min():.2f}, {scaled_df.max().max():.2f}]")
        
        return scaled_df

    def construct_fusion_list(self, row):
        """
        Construct fusion list from a dataframe row for ObjectFusion sorting.
        Returns list format: [pitch, yaw, roll, left_hand(6), right_hand(6), top_face(6),
                              left_eye(6), right_eye(6), mouth(6), shoulder(6),
                              waist(6), feet(6)]
        Total: 57 elements (3 + 9*6)
        """
        fusion_list = row['pitch_yaw_roll_list'].copy()  # Start with [pitch, yaw, roll]
        
        # Add detection data (each is 6 elements: class_id, conf, top, left, right, bottom)
        for col in ['left_hand_object', 'right_hand_object', 'top_face_object',
                    'left_eye_object', 'right_eye_object', 'mouth_object', 'shoulder_object',
                    'waist_object', 'feet_object']:
            detection = row[col]
            if detection is None:
                fusion_list.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 6 zeros for None
            else:
                fusion_list.extend(self.flatten_object_detections(detection))
        
        return fusion_list
