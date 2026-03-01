"""
Analyze distribution of detections across positions and distance ranges.
Shows what we're capturing vs excluding in classify_object_hand_relationships.
"""

import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from my_declarative_base import Base, Images, Detections, Encodings
from tools_clustering import ToolsClustering
import json
from collections import defaultdict

import os
from pathlib import Path

# importing project-specific models
import sys
ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/takingstock/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)


# Setup database connection (using existing config pattern)
from mp_db_io import DataIO
from mp_sort_pose import SortPose

io = DataIO()
db = io.db

# Create minimal config for SortPose (required to use prep_knuckle_landmarks)
motion = {"side_to_side": False, "forward_smile": True, "laugh": False, "forward_nosmile": False, "static_pose": False, "simple": False}
cfg = {
    'motion': motion,
    'face_height_output': 500,
    'SORT_TYPE': 'ObjectFusion',
}
sort = SortPose(config=cfg)

if db['unix_socket']:
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), pool_pre_ping=True, pool_recycle=600)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(
        host=db['host'], db=db['name'], user=db['user'], pw=db['pass']
    ), pool_pre_ping=True, pool_recycle=600)

Session = sessionmaker(bind=engine)
session = Session()

def analyze_detection_positions(skip_every=10):
    """
    Sample detections and classify them, tracking:
    - How many fit into each of the 5 positions
    - How many are excluded (background)
    - Distance distribution of excluded objects
    
    Args:
        skip_every: Process every Nth image (e.g., 10 = sample 10% of data for speed)
                   Processes entire Detections table
    """
    
    
    cl = ToolsClustering("ObjectFusion", session=session, VERBOSE=False)
    
    # Track results
    position_counts = {
        'both_hands_object': 0,
        'left_hand_object': 0,
        'right_hand_object': 0,
        'top_face_object': 0,
        'bottom_face_object': 0,
        'excluded_background': 0,
    }
    
    distance_buckets = defaultdict(int)  # distance -> count
    excluded_distances = []  # List of all excluded distances
    
    # Query images with detections - match Clustering_SQL.py filtering:
    # FROM SegmentBig_isface + JOIN Encodings + INNER JOIN Detections + WHERE is_dupe_of IS NULL
    query_result = session.execute(text("""
        SELECT DISTINCT s.image_id
        FROM SegmentBig_isface s 
        JOIN Encodings e ON s.image_id = e.image_id 
        INNER JOIN Detections h ON h.image_id = s.image_id
        WHERE e.is_dupe_of IS NULL
        ORDER BY s.image_id
    """))
    image_ids = [row[0] for row in query_result.all()]
    total_images = len(image_ids)
    sampled_images = total_images // skip_every
    print(f"Found {total_images:,} distinct images matching Clustering_SQL.py filters")
    print(f"Processing every {skip_every}th image = {sampled_images:,} images to analyze...")
    
    for idx, image_id in enumerate(image_ids):
        # Skip pattern - only process every Nth image
        if idx % skip_every != 0:
            continue
            
        if (idx // skip_every) % 5000 == 0 and idx > 0:
            print(f"  Processed {idx // skip_every:,}/{sampled_images:,} images (at idx {idx:,})...")
        
        # Get hand landmarks from MongoDB
        try:
            encodings = io.get_encodings_mongo(image_id)
            hand_results = encodings['hand_results'] if isinstance(encodings, pd.Series) else encodings[5]
            
            if hand_results is None or pd.isna(hand_results):
                left_knuckle = cl.DEFAULT_HAND_POSITION
                right_knuckle = cl.DEFAULT_HAND_POSITION
            else:
                left_knuckle, right_knuckle = sort.prep_knuckle_landmarks(hand_results)
                if not left_knuckle or len(left_knuckle) == 0:
                    left_knuckle = cl.DEFAULT_HAND_POSITION
                if not right_knuckle or len(right_knuckle) == 0:
                    right_knuckle = cl.DEFAULT_HAND_POSITION
        except Exception as e:
            # print(f"Error fetching encodings for image_id {image_id}: {e}")
            left_knuckle = cl.DEFAULT_HAND_POSITION
            right_knuckle = cl.DEFAULT_HAND_POSITION
        
        # Get all detections for this image
        det_result = session.execute(text("SELECT detection_id, class_id, conf, bbox_norm FROM Detections WHERE image_id = :image_id AND conf > :min_conf"), {"image_id": image_id, "min_conf": cl.MIN_DETECTION_CONFIDENCE}).fetchall()
        
        if not det_result:
            continue
        
        # Parse detections
        detections = []
        for detection_id, class_id, conf, bbox_norm_str in det_result:
            bbox = cl.parse_bbox_norm(bbox_norm_str)
            if bbox is None:
                continue
            detections.append({
                'detection_id': detection_id,
                'class_id': class_id,
                'conf': conf,
                'bbox': bbox,
                'top': bbox['top'],
                'left': bbox['left'],
                'right': bbox['right'],
                'bottom': bbox['bottom']
            })
        
        if not detections:
            continue
        
        # Classify using current method
        classified = cl.classify_object_hand_relationships(detections, left_knuckle, right_knuckle)
        assigned_ids = set()
        
        # Count assigned positions
        for position, det in classified.items():
            if det is not None:
                position_counts[position] += 1
                assigned_ids.add(det['detection_id'])
        
        # Count excluded (background) detections
        excluded_count = len(detections) - len(assigned_ids)
        position_counts['excluded_background'] += excluded_count
        
        # Calculate distances for excluded detections
        for det in detections:
            if det['detection_id'] not in assigned_ids:
                # Distance to closest hand
                left_dist = cl.point_to_bbox_distance(left_knuckle, det['bbox'])
                right_dist = cl.point_to_bbox_distance(right_knuckle, det['bbox'])
                min_hand_dist = min(left_dist, right_dist)
                excluded_distances.append(min_hand_dist)
                
                # Bucket by distance
                bucket = int(min_hand_dist * 10) / 10  # Round to nearest 0.1
                distance_buckets[bucket] += 1
    
    # Print results
    print("\n" + "="*60)
    print("DETECTION POSITION DISTRIBUTION")
    print("="*60)
    
    total_detections = sum(position_counts.values())
    for position, count in position_counts.items():
        pct = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"{position:25s}: {count:6d} ({pct:5.1f}%)")
    
    print(f"\nTotal detections analyzed: {total_detections}")
    print(f"Current TOUCH_THRESHOLD: {cl.TOUCH_THRESHOLD}")
    
    if excluded_distances:
        print("\n" + "="*60)
        print("EXCLUDED (BACKGROUND) DETECTIONS - DISTANCE ANALYSIS")
        print("="*60)
        
        excluded_distances = np.array(excluded_distances)
        print(f"Count: {len(excluded_distances)}")
        print(f"Min distance: {excluded_distances.min():.3f} (face height units)")
        print(f"Max distance: {excluded_distances.max():.3f}")
        print(f"Mean distance: {excluded_distances.mean():.3f}")
        print(f"Median distance: {np.median(excluded_distances):.3f}")
        print(f"Std dev: {np.std(excluded_distances):.3f}")
        
        print("\nDistance distribution (face height units):")
        print("Distance Range          | Count | Cumulative %")
        print("-" * 50)
        
        sorted_buckets = sorted(distance_buckets.keys())
        cumulative = 0
        for bucket in sorted_buckets:
            count = distance_buckets[bucket]
            cumulative += count
            cum_pct = (cumulative / len(excluded_distances) * 100)
            pct = (count / len(excluded_distances) * 100)
            print(f"{bucket:5.1f} - {bucket+0.1:5.1f}          | {count:5d} | {cum_pct:5.1f}%")
        
        print("\n" + "="*60)
        print("THRESHOLD ANALYSIS")
        print("="*60)
        print("If we increase TOUCH_THRESHOLD to capture N% of excluded objects:")
        print()
        
        thresholds_to_test = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
        for threshold in thresholds_to_test:
            count_captured = sum(1 for d in excluded_distances if d <= threshold)
            pct = (count_captured / len(excluded_distances) * 100)
            total_with_this_threshold = position_counts['both_hands_object'] + \
                                        position_counts['left_hand_object'] + \
                                        position_counts['right_hand_object'] + \
                                        position_counts['top_face_object'] + \
                                        position_counts['bottom_face_object'] + \
                                        count_captured
            total_all = total_detections
            new_coverage = (total_with_this_threshold / total_all * 100)
            print(f"  TOUCH_THRESHOLD = {threshold:.2f}: {count_captured:6d} more objects ({pct:5.1f}% of excluded), total coverage {new_coverage:5.1f}%")
    
    session.close()
    engine.dispose()

if __name__ == "__main__":
    # Process every 10th image for 10x speedup
    # Analyzes ENTIRE Detections table, not limited
    analyze_detection_positions(skip_every=10)
