from sqlalchemy import create_engine, select, Column, Integer, Float, ForeignKey, BLOB
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from my_declarative_base import Base, Images
import pickle
import numpy as np
import json

class ToolsClustering:
    """Store key clustering info for use across codebase"""

    def __init__(self, CLUSTER_TYPE, VERBOSE=False):
        self.VERBOSE = VERBOSE
        self.CLUSTER_TYPE = CLUSTER_TYPE
        self.CLUSTER_MEDIANS = None
        # Object-hand relationship constants
        self.TOUCH_THRESHOLD = 0.25  # face height units
        self.OVERLAP_IOU_THRESHOLD = 0.5
        self.HIGH_CONFIDENCE_THRESHOLD = 0.9
        self.CONFIDENCE_DIFF_THRESHOLD = 0.3
        self.MIN_DETECTION_CONFIDENCE = 0.4
        self.DEFAULT_HAND_POSITION = [0.0, 8.0, 0.0]
        # Face object constraints to avoid large background objects
        self.MAX_FACE_WIDTH = 2.0  # max width of left+right to be considered face object
        self.MAX_FACE_VERT_EXTENSION = 0.75  # max how far object can extend into opposite zone
        self.CLUSTER_DATA = {
            "BodyPoses": {"data_column": "mongo_body_landmarks", "is_feet": 1, "mongo_hand_landmarks": None},
            "BodyPoses3D": {"data_column": "mongo_body_landmarks_3D", "is_feet": 1, "mongo_hand_landmarks": None}, # changed this for testing
            "ArmsPoses3D": {"data_column": "mongo_body_landmarks_3D", "is_feet": None, "mongo_hand_landmarks": 1},
            "ObjectFusion": {"data_column": "mongo_hand_landmarks_norm", "is_feet": None, "mongo_hand_landmarks": None},
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

    def classify_object_hand_relationships(self, detections, left_knuckle, right_knuckle):
        """
        Classify each detection based on its relationship to hands and face.
        Returns dict with keys: both_hands_object, left_hand_object, right_hand_object, 
                               top_face_object, bottom_face_object
        Each value is the detection dict or None.
        """
        results = {
            'both_hands_object': None,
            'left_hand_object': None,
            'right_hand_object': None,
            'top_face_object': None,
            'bottom_face_object': None
        }
        
        if not detections:
            return results
        
        # First, resolve overlapping detections
        detections = self.resolve_overlapping_detections(detections)
        
        # Track which detections have been assigned
        assigned = set()
        
        # 1. Check for both_hands_object first (highest priority for hand-held objects)
        for det in detections:
            bbox = det['bbox']
            left_touching = self.is_touching_hand(left_knuckle, bbox)
            right_touching = self.is_touching_hand(right_knuckle, bbox)
            
            if left_touching and right_touching:
                if results['both_hands_object'] is None:
                    results['both_hands_object'] = det
                    assigned.add(det['detection_id'])
                else:
                    # Multiple both-hands objects - pick closest to midpoint of hands
                    existing_dist = self.point_to_bbox_distance(
                        [(left_knuckle[0] + right_knuckle[0])/2, (left_knuckle[1] + right_knuckle[1])/2],
                        results['both_hands_object']['bbox']
                    )
                    new_dist = self.point_to_bbox_distance(
                        [(left_knuckle[0] + right_knuckle[0])/2, (left_knuckle[1] + right_knuckle[1])/2],
                        bbox
                    )
                    if new_dist < existing_dist:
                        assigned.discard(results['both_hands_object']['detection_id'])
                        results['both_hands_object'] = det
                        assigned.add(det['detection_id'])
        
        # 2. Check for top_face_object and bottom_face_object
        for det in detections:
            if det['detection_id'] in assigned:
                continue
            bbox = det['bbox']
            
            if self.is_top_face_object(bbox):
                if results['top_face_object'] is None:
                    results['top_face_object'] = det
                    assigned.add(det['detection_id'])
                # Keep the one that's most "on top" (most negative top value)
                elif bbox['top'] < results['top_face_object']['bbox']['top']:
                    assigned.discard(results['top_face_object']['detection_id'])
                    results['top_face_object'] = det
                    assigned.add(det['detection_id'])
            
            if self.is_bottom_face_object(bbox):
                if results['bottom_face_object'] is None:
                    results['bottom_face_object'] = det
                    assigned.add(det['detection_id'])
                # Keep the one that's most "on bottom" (largest bottom value)
                elif bbox['bottom'] > results['bottom_face_object']['bbox']['bottom']:
                    assigned.discard(results['bottom_face_object']['detection_id'])
                    results['bottom_face_object'] = det
                    assigned.add(det['detection_id'])
        
        # 3. Find closest unassigned object to each hand
        unassigned = [d for d in detections if d['detection_id'] not in assigned]
        
        # Left hand - find closest object
        if left_knuckle != self.DEFAULT_HAND_POSITION:
            best_left = None
            best_left_dist = float('inf')
            for det in unassigned:
                dist = self.point_to_bbox_distance(left_knuckle, det['bbox'])
                if dist < best_left_dist:
                    best_left_dist = dist
                    best_left = det
            
            if best_left is not None and best_left_dist <= self.TOUCH_THRESHOLD * 2:
                results['left_hand_object'] = best_left
                assigned.add(best_left['detection_id'])
                unassigned = [d for d in unassigned if d['detection_id'] != best_left['detection_id']]
        
        # Right hand - find closest from remaining unassigned
        if right_knuckle != self.DEFAULT_HAND_POSITION:
            best_right = None
            best_right_dist = float('inf')
            for det in unassigned:
                dist = self.point_to_bbox_distance(right_knuckle, det['bbox'])
                if dist < best_right_dist:
                    best_right_dist = dist
                    best_right = det
            
            if best_right is not None and best_right_dist <= self.TOUCH_THRESHOLD * 2:
                results['right_hand_object'] = best_right
                assigned.add(best_right['detection_id'])
        
        # Sanity check: both_hands should not coexist with single-hand assignment for same object
        if results['both_hands_object'] is not None:
            if results['left_hand_object'] is not None and results['left_hand_object']['detection_id'] == results['both_hands_object']['detection_id']:
                print(f"  🚨 DEBUG ALERT: both_hands_object and left_hand_object are the same! det_id={results['both_hands_object']['detection_id']}")
            if results['right_hand_object'] is not None and results['right_hand_object']['detection_id'] == results['both_hands_object']['detection_id']:
                print(f"  🚨 DEBUG ALERT: both_hands_object and right_hand_object are the same! det_id={results['both_hands_object']['detection_id']}")
        
        return results

