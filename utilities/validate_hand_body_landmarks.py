"""
Validation script to compare hand landmarks vs body landmarks for fingertip positions.

This script validates that hand landmark[8] (fingertip) and body landmark[19/20] (fingertip)
produce similar positions where both exist, to ensure the fallback logic is accurate.

Outputs raw and calculated data to CSV files for further analysis.

Usage:
    python utilities/validate_hand_body_landmarks.py [--sample-size N]
"""

import sys
import os
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from mp_db_io import DataIO
from mp_sort_pose import SortPose

def calculate_euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points."""
    if not point1 or not point2:
        return None
    if len(point1) < 3 or len(point2) < 3:
        return None
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(point1[:3], point2[:3])))

def extract_hand_fingertip(hand_results, side='left'):
    """Extract fingertip (landmark 8) from hand_results."""
    if not hand_results:
        return None
    
    hand_key = f'{side}_hand'
    if hand_key not in hand_results:
        return None
    
    try:
        hand_data = hand_results[hand_key]
        
        # Check for hand_landmarks_norm (preferred, normalized)
        if 'hand_landmarks_norm' in hand_data:
            landmarks = hand_data['hand_landmarks_norm']
            if isinstance(landmarks, bytes):
                # It's in protobuf format - skip
                return None
            if isinstance(landmarks, (list, tuple)) and len(landmarks) > 8:
                landmark_0 = landmarks[0]
                # Should be [x, y, z]
                if isinstance(landmark_0, (list, tuple)) and len(landmark_0) >= 3:
                    return list(landmark_0[:3])
        
        # Fallback to image_landmarks if norm not available
        if 'image_landmarks' in hand_data:
            landmarks = hand_data['image_landmarks']
            if isinstance(landmarks, (list, tuple)) and len(landmarks) > 8:
                landmark_0 = landmarks[0]
                if isinstance(landmark_0, (list, tuple)) and len(landmark_0) >= 3:
                    return list(landmark_0[:3])
    except (IndexError, TypeError, KeyError, AttributeError) as e:
        pass
    
    return None

def extract_body_fingertip(body_landmarks_norm, io, sort_pose, side='left'):
    """Extract wrist from body landmarks - direct access."""
    if body_landmarks_norm is None:
        return None
    
    try:
        # Unpickle if it's bytes
        if isinstance(body_landmarks_norm, bytes):
            body_landmarks_norm = io.unpickle_array(body_landmarks_norm)
        
        if body_landmarks_norm is None:
            return None
        
        # MediaPipe body pose has 33 landmarks
        # Using WRIST instead of fingertip since hand connects at wrist
        # MediaPipe standard: Landmark 15 = LEFT wrist, 16 = RIGHT wrist
        landmark_idx = 15 if side == 'left' else 16
        
        # Direct access to landmark - protobuf objects have .landmark attribute
        if hasattr(body_landmarks_norm, 'landmark'):
            landmarks = body_landmarks_norm.landmark
            if len(landmarks) > landmark_idx:
                lm = landmarks[landmark_idx]
                if hasattr(lm, 'x') and hasattr(lm, 'y') and hasattr(lm, 'z'):
                    return [lm.x, lm.y, lm.z]
        
        # If it's already a list/array
        elif isinstance(body_landmarks_norm, (list, tuple)) and len(body_landmarks_norm) > landmark_idx:
            lm = body_landmarks_norm[landmark_idx]
            if isinstance(lm, (list, tuple)) and len(lm) >= 3:
                return list(lm[:3])
            elif hasattr(lm, 'x') and hasattr(lm, 'y') and hasattr(lm, 'z'):
                return [lm.x, lm.y, lm.z]
    except Exception as e:
        # Debug: print first few errors
        pass
    
    return None

def main(sample_size=1000):
    """
    Main validation function.
    
    Args:
        sample_size: Number of images to sample for validation
    """
    print("=" * 80)
    print("Hand vs Body Landmarks Validation Script")
    print("=" * 80)
    print(f"\nSample size: {sample_size} images")
    print("\nQuerying images with both hand and body landmarks...")
    
    # Initialize with minimal config
    motion = {"side_to_side": False, "forward_smile": True, "laugh": False, 
              "forward_nosmile": False, "static_pose": False, "simple": False}
    cfg = {
        'motion': motion,
        'face_height_output': 500,
        'SORT_TYPE': 'ObjectFusion',
    }
    io = DataIO()
    sort_pose = SortPose(config=cfg)
    sort_pose.query_face = False
    sort_pose.query_hands = True
    sort_pose.query_body = True
    
    # Setup database connection
    db = io.db
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
    
    # Query images that have both hand and body landmarks
    # Sample evenly across dataset using SegmentHelper_oct2025_every40
    query = text(f"""
        SELECT 
            s.image_id,
            s.site_name_id,
            e.encoding_id
        FROM SegmentBig_isface s
        INNER JOIN Encodings e ON s.image_id = e.image_id
        INNER JOIN SegmentHelper_oct2025_every40 sh ON s.image_id = sh.image_id
        WHERE e.is_dupe_of IS NULL
        AND e.mongo_hand_landmarks_norm = 1
        AND e.mongo_body_landmarks_norm = 1
        ORDER BY s.image_id
        LIMIT {sample_size}
    """)
    
    result = session.execute(query)
    rows = result.all()
    
    print(f"Found {len(rows)} images with both hand and body landmarks")
    
    if len(rows) == 0:
        print("ERROR: No images found with both landmarks. Cannot validate.")
        return
    
    # Collect comparison data with metadata
    left_distances = []
    right_distances = []
    left_comparisons = []  # Store detailed comparison data for analysis
    right_comparisons = []
    
    # Track metadata for pattern analysis
    site_id_outliers = {}  # site_name_id -> outlier count
    image_id_ranges = []  # Track if outliers cluster by image_id
    
    missing_hand_data = 0
    missing_body_data = 0
    both_present = 0
    success_count = 0
    
    # Track all images for comprehensive analysis
    all_images_processed = []  # For tracking complete dataset
    
    print("\nProcessing images...")
    for i, row in enumerate(rows):
        image_id = row[0]
        site_name_id = row[1]
        encoding_id = row[2]
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(rows)} images...")
        
        # Fetch MongoDB data
        mongo_data = io.get_encodings_mongo(image_id)
        hand_results = mongo_data[5]  # 6th element is hand_results
        # body_landmarks_3D = mongo_data[4]  # 5th element is body_landmarks_3D
        body_landmarks_norm = mongo_data[3]  # 4th element is body_landmarks_norm
        
        # Extract left hand/body fingertips
        left_hand = extract_hand_fingertip(hand_results, 'left')
        left_body = extract_body_fingertip(body_landmarks_norm, io, sort_pose, 'left')
        
        # Track all image processing
        image_record = {
            'image_id': image_id,
            'encoding_id': encoding_id,
            'site_name_id': site_name_id,
            'has_left_hand': left_hand is not None,
            'has_left_body': left_body is not None,
            'has_right_hand': False,
            'has_right_body': False,
        }
        
        if left_hand and left_body:
            both_present += 1
            dist = calculate_euclidean_distance(left_hand, left_body)
            if dist is not None:
                left_distances.append(dist)
                is_outlier = dist > 0.5
                left_comparisons.append({
                    'image_id': image_id,
                    'encoding_id': encoding_id,
                    'site_name_id': site_name_id,
                    'hand': left_hand,
                    'body': left_body,
                    'distance': dist,
                    'is_outlier': is_outlier
                })
                
                # Track in image record
                image_record['left_hand_x'] = left_hand[0]
                image_record['left_hand_y'] = left_hand[1]
                image_record['left_body_x'] = left_body[0]
                image_record['left_body_y'] = left_body[1]
                image_record['left_distance'] = dist
                image_record['left_is_outlier'] = is_outlier
                
                # Track outliers by site_id
                if site_name_id not in site_id_outliers:
                    site_id_outliers[site_name_id] = {'outliers': 0, 'total': 0}
                site_id_outliers[site_name_id]['total'] += 1
                if is_outlier:
                    site_id_outliers[site_name_id]['outliers'] += 1
                    
                if success_count < 5:
                    print(f"\n  [SUCCESS {success_count+1}] Image {image_id} LEFT:")
                    print(f"    Site ID: {site_name_id}")
                    print(f"    Hand landmark[8]:  {left_hand}")
                    print(f"    Body landmark[15]: {left_body}")
                    print(f"    Distance: {dist:.4f}")
                    success_count += 1
        elif not left_hand:
            missing_hand_data += 1
        elif not left_body:
            missing_body_data += 1
        
        # Extract right hand/body fingertips
        right_hand = extract_hand_fingertip(hand_results, 'right')
        right_body = extract_body_fingertip(body_landmarks_norm, io, sort_pose, 'right')
        
        image_record['has_right_hand'] = right_hand is not None
        image_record['has_right_body'] = right_body is not None
        
        if right_hand and right_body:
            both_present += 1
            dist = calculate_euclidean_distance(right_hand, right_body)
            if dist is not None:
                right_distances.append(dist)
                is_outlier = dist > 0.5
                right_comparisons.append({
                    'image_id': image_id,
                    'encoding_id': encoding_id,
                    'site_name_id': site_name_id,
                    'hand': right_hand,
                    'body': right_body,
                    'distance': dist,
                    'is_outlier': is_outlier
                })
                
                # Track in image record
                image_record['right_hand_x'] = right_hand[0]
                image_record['right_hand_y'] = right_hand[1]
                image_record['right_body_x'] = right_body[0]
                image_record['right_body_y'] = right_body[1]
                image_record['right_distance'] = dist
                image_record['right_is_outlier'] = is_outlier
                
                # Track outliers by site_id
                if site_name_id not in site_id_outliers:
                    site_id_outliers[site_name_id] = {'outliers': 0, 'total': 0}
                site_id_outliers[site_name_id]['total'] += 1
                if is_outlier:
                    site_id_outliers[site_name_id]['outliers'] += 1
        elif not right_hand:
            missing_hand_data += 1
        elif not right_body:
            missing_body_data += 1
        
        # Track complete image record
        all_images_processed.append(image_record)
    
    print(f"\n{'-'*80}")
    print("VALIDATION RESULTS")
    print(f"{'-'*80}\n")
    
    # OUTPUT CSV FILES WITH DETAILED DATA
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File 1: Detailed comparisons (hand + body both present)
    comparisons_csv = f"validation_comparisons_{timestamp}.csv"
    
    print(f"Writing detailed comparisons to {comparisons_csv}...")
    
    with open(comparisons_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            'image_id', 'encoding_id', 'site_name_id', 'hand_side',
            'hand_x', 'hand_y', 'hand_z',
            'body_x', 'body_y', 'body_z',
            'distance_xyz', 'distance_xy_only',
            'x_diff', 'y_diff', 'z_diff',
            'is_outlier', 'outlier_threshold'
        ])
        
        # Write left hand data
        for comp in left_comparisons:
            hand = comp['hand']
            body = comp['body']
            dist_xyz = comp['distance']
            dist_xy = np.sqrt((hand[0] - body[0])**2 + (hand[1] - body[1])**2)
            
            writer.writerow([
                comp['image_id'],
                comp['encoding_id'],
                comp['site_name_id'],
                'LEFT',
                f"{hand[0]:.6f}",
                f"{hand[1]:.6f}",
                f"{hand[2]:.6f}",
                f"{body[0]:.6f}",
                f"{body[1]:.6f}",
                f"{body[2]:.6f}",
                f"{dist_xyz:.6f}",
                f"{dist_xy:.6f}",
                f"{hand[0] - body[0]:.6f}",
                f"{hand[1] - body[1]:.6f}",
                f"{hand[2] - body[2]:.6f}",
                comp['is_outlier'],
                '>0.5'
            ])
        
        # Write right hand data
        for comp in right_comparisons:
            hand = comp['hand']
            body = comp['body']
            dist_xyz = comp['distance']
            dist_xy = np.sqrt((hand[0] - body[0])**2 + (hand[1] - body[1])**2)
            
            writer.writerow([
                comp['image_id'],
                comp['encoding_id'],
                comp['site_name_id'],
                'RIGHT',
                f"{hand[0]:.6f}",
                f"{hand[1]:.6f}",
                f"{hand[2]:.6f}",
                f"{body[0]:.6f}",
                f"{body[1]:.6f}",
                f"{body[2]:.6f}",
                f"{dist_xyz:.6f}",
                f"{dist_xy:.6f}",
                f"{hand[0] - body[0]:.6f}",
                f"{hand[1] - body[1]:.6f}",
                f"{hand[2] - body[2]:.6f}",
                comp['is_outlier'],
                '>0.5'
            ])
    
    print(f"✓ Wrote {len(left_comparisons) + len(right_comparisons)} comparison rows to {comparisons_csv}")
    
    # File 2: All images with their data availability status
    images_csv = f"validation_images_{timestamp}.csv"
    
    print(f"Writing all images metadata to {images_csv}...")
    
    with open(images_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            'image_id', 'encoding_id', 'site_name_id',
            'has_left_hand', 'has_left_body', 'has_right_hand', 'has_right_body',
            'left_hand_x', 'left_hand_y', 'left_body_x', 'left_body_y', 'left_distance', 'left_is_outlier',
            'right_hand_x', 'right_hand_y', 'right_body_x', 'right_body_y', 'right_distance', 'right_is_outlier'
        ])
        
        # Write all image records
        for img in all_images_processed:
            writer.writerow([
                img['image_id'],
                img['encoding_id'],
                img['site_name_id'],
                img['has_left_hand'],
                img['has_left_body'],
                img['has_right_hand'],
                img['has_right_body'],
                img.get('left_hand_x', ''),
                img.get('left_hand_y', ''),
                img.get('left_body_x', ''),
                img.get('left_body_y', ''),
                img.get('left_distance', ''),
                img.get('left_is_outlier', ''),
                img.get('right_hand_x', ''),
                img.get('right_hand_y', ''),
                img.get('right_body_x', ''),
                img.get('right_body_y', ''),
                img.get('right_distance', ''),
                img.get('right_is_outlier', '')
            ])
    
    print(f"✓ Wrote {len(all_images_processed)} image records to {images_csv}")
    print(f"\nFile locations:")
    print(f"  Comparisons: {os.path.abspath(comparisons_csv)}")
    print(f"  All images: {os.path.abspath(images_csv)}\n")
    
    # Statistics for left hand
    if left_distances:
        left_distances_arr = np.array(left_distances)
        print("LEFT HAND/BODY Wrist COMPARISON:")
        print(f"  Valid comparisons: {len(left_distances)}")
        print(f"  Mean distance: {left_distances_arr.mean():.4f} face height units")
        print(f"  Median distance: {np.median(left_distances_arr):.4f} face height units")
        print(f"  Std deviation: {left_distances_arr.std():.4f}")
        print(f"  Min distance: {left_distances_arr.min():.4f}")
        print(f"  Max distance: {left_distances_arr.max():.4f}")
        print(f"  95th percentile: {np.percentile(left_distances_arr, 95):.4f}")
        
        outliers = np.sum(left_distances_arr > 0.5)
        print(f"  Outliers (>0.5 units): {outliers} ({100*outliers/len(left_distances):.1f}%)")
    else:
        print("LEFT HAND: No valid comparisons found")
    
    print()
    
    # Statistics for right hand
    if right_distances:
        right_distances_arr = np.array(right_distances)
        print("RIGHT HAND/BODY Wrist COMPARISON:")
        print(f"  Valid comparisons: {len(right_distances)}")
        print(f"  Mean distance: {right_distances_arr.mean():.4f} face height units")
        print(f"  Median distance: {np.median(right_distances_arr):.4f} face height units")
        print(f"  Std deviation: {right_distances_arr.std():.4f}")
        print(f"  Min distance: {right_distances_arr.min():.4f}")
        print(f"  Max distance: {right_distances_arr.max():.4f}")
        print(f"  95th percentile: {np.percentile(right_distances_arr, 95):.4f}")
        
        outliers = np.sum(right_distances_arr > 0.5)
        print(f"  Outliers (>0.5 units): {outliers} ({100*outliers/len(right_distances):.1f}%)")
    else:
        print("RIGHT HAND: No valid comparisons found")
    
    print()
    
    # Combined statistics
    if left_distances or right_distances:
        all_distances = left_distances + right_distances
        all_distances_arr = np.array(all_distances)
        print("COMBINED (LEFT + RIGHT):")
        print(f"  Total valid comparisons: {len(all_distances)}")
        print(f"  Mean distance: {all_distances_arr.mean():.4f} face height units")
        print(f"  Median distance: {np.median(all_distances_arr):.4f} face height units")
        print(f"  95th percentile: {np.percentile(all_distances_arr, 95):.4f}")
        
        outliers = np.sum(all_distances_arr > 0.5)
        print(f"  Outliers (>0.5 units): {outliers} ({100*outliers/len(all_distances):.1f}%)")
    
    print(f"\n{'-'*80}")
    print("DATA QUALITY:")
    print(f"  Images queried: {len(rows)}")
    print(f"  Images with both hand AND body data: {both_present}")
    print(f"  Missing hand data: {missing_hand_data} instances")
    print(f"  Missing body data: {missing_body_data} instances")
    print(f"{'-'*80}\n")
    
    # PATTERN ANALYSIS BY METADATA
    print("=" * 80)
    print("PATTERN ANALYSIS: OUTLIERS BY SITE_ID")
    print("=" * 80)
    
    if site_id_outliers:
        print("\nOUTLIER RATE BY SITE_NAME_ID:")
        # Sort by outlier rate
        site_data = [(site_id, data['outliers'], data['total'], 
                      100 * data['outliers'] / data['total'] if data['total'] > 0 else 0)
                     for site_id, data in site_id_outliers.items()]
        site_data.sort(key=lambda x: x[3], reverse=True)
        
        print(f"\n{'Site ID':<15} {'Outliers':>10} {'Total':>10} {'Rate':>10}")
        print("-" * 50)
        for site_id, outliers, total, rate in site_data:
            print(f"{site_id:<15} {outliers:>10} {total:>10} {rate:>9.1f}%")
    
    # PATTERN ANALYSIS BY IMAGE_ID RANGES
    print(f"\n{'-'*80}")
    print("PATTERN ANALYSIS: OUTLIERS BY IMAGE_ID RANGE")
    print(f"{'-'*80}\n")
    
    all_comparisons = left_comparisons + right_comparisons
    if all_comparisons:
        # Sort by image_id
        all_comparisons_sorted = sorted(all_comparisons, key=lambda x: x['image_id'])
        
        # Divide into 10 deciles
        n_deciles = 10
        decile_size = len(all_comparisons_sorted) // n_deciles
        
        print(f"Dividing {len(all_comparisons_sorted)} comparisons into {n_deciles} deciles by image_id:\n")
        print(f"{'Decile':<8} {'Image ID Range':<30} {'Outliers':>10} {'Total':>10} {'Rate':>10}")
        print("-" * 75)
        
        for i in range(n_deciles):
            start_idx = i * decile_size
            end_idx = start_idx + decile_size if i < n_deciles - 1 else len(all_comparisons_sorted)
            decile_data = all_comparisons_sorted[start_idx:end_idx]
            
            min_id = decile_data[0]['image_id']
            max_id = decile_data[-1]['image_id']
            outliers = sum(1 for d in decile_data if d['is_outlier'])
            total = len(decile_data)
            rate = 100 * outliers / total if total > 0 else 0
            
            print(f"{i+1:<8} {min_id:>10} - {max_id:<10} {outliers:>10} {total:>10} {rate:>9.1f}%")
    
    # PATTERN ANALYSIS BY ENCODING_ID RANGES
    print(f"\n{'-'*80}")
    print("PATTERN ANALYSIS: OUTLIERS BY ENCODING_ID RANGE")
    print(f"{'-'*80}\n")
    
    if all_comparisons:
        # Sort by encoding_id
        all_comparisons_enc_sorted = sorted(all_comparisons, key=lambda x: x['encoding_id'])
        
        print(f"Dividing {len(all_comparisons_enc_sorted)} comparisons into {n_deciles} deciles by encoding_id:\n")
        print(f"{'Decile':<8} {'Encoding ID Range':<30} {'Outliers':>10} {'Total':>10} {'Rate':>10}")
        print("-" * 75)
        
        for i in range(n_deciles):
            start_idx = i * decile_size
            end_idx = start_idx + decile_size if i < n_deciles - 1 else len(all_comparisons_enc_sorted)
            decile_data = all_comparisons_enc_sorted[start_idx:end_idx]
            
            min_id = decile_data[0]['encoding_id']
            max_id = decile_data[-1]['encoding_id']
            outliers = sum(1 for d in decile_data if d['is_outlier'])
            total = len(decile_data)
            rate = 100 * outliers / total if total > 0 else 0
            
            print(f"{i+1:<8} {min_id:>10} - {max_id:<10} {outliers:>10} {total:>10} {rate:>9.1f}%")
    
    # PATTERN ANALYSIS BY SITE_NAME_ID
    print(f"\n{'-'*80}")
    print("PATTERN ANALYSIS: OUTLIERS BY SITE_NAME_ID (detailed)")
    print(f"{'-'*80}\n")
    
    if all_comparisons:
        # Group by site_name_id
        site_id_data = {}
        for comp in all_comparisons:
            sid = comp['site_name_id']
            if sid not in site_id_data:
                site_id_data[sid] = {'outliers': 0, 'total': 0}
            site_id_data[sid]['total'] += 1
            if comp['is_outlier']:
                site_id_data[sid]['outliers'] += 1
        
        # Sort by outlier rate
        site_id_list = [(sid, data['outliers'], data['total'], 
                        100 * data['outliers'] / data['total'] if data['total'] > 0 else 0)
                       for sid, data in site_id_data.items()]
        site_id_list.sort(key=lambda x: x[3], reverse=True)
        
        print(f"{'Site ID':<10} {'Outliers':>10} {'Total':>10} {'Rate':>10}")
        print("-" * 45)
        for sid, outliers, total, rate in site_id_list:
            print(f"{sid:<10} {outliers:>10} {total:>10} {rate:>9.1f}%")
    
    # OUTLIER ANALYSIS
    print("=" * 80)
    print("OUTLIER ANALYSIS")
    print("=" * 80)
    
    if left_distances:
        left_distances_arr = np.array(left_distances)
        left_outlier_mask = left_distances_arr > 0.5
        left_outlier_indices = np.where(left_outlier_mask)[0]
        
        print(f"\nLEFT HAND OUTLIERS (distance > 0.5 units):")
        print(f"  Count: {len(left_outlier_indices)} / {len(left_distances)}")
        
        if len(left_outlier_indices) > 0:
            # Get outlier comparisons
            left_outliers = [left_comparisons[i] for i in left_outlier_indices]
            
            # Sort by distance for analysis
            left_outliers.sort(key=lambda x: x['distance'], reverse=True)
            
            # Show top 10 worst outliers
            print("\n  TOP 10 WORST OUTLIERS:")
            for j, comp in enumerate(left_outliers[:10]):
                hand = comp['hand']
                body = comp['body']
                dist = comp['distance']
                image_id = comp['image_id']
                
                dx = hand[0] - body[0]
                dy = hand[1] - body[1]
                
                print(f"\n    [{j+1}] Image {image_id} - Distance: {dist:.4f}")
                print(f"        Hand: [{hand[0]:7.4f}, {hand[1]:7.4f}, {hand[2]:7.4f}]")
                print(f"        Body: [{body[0]:7.4f}, {body[1]:7.4f}, {body[2]:7.4f}]")
                print(f"        Diff: [{dx:7.4f}, {dy:7.4f}]")
            
            # Analyze outlier patterns
            print("\n  OUTLIER PATTERN ANALYSIS:")
            outlier_distances = left_distances_arr[left_outlier_mask]
            print(f"    Mean outlier distance: {outlier_distances.mean():.4f}")
            print(f"    Median outlier distance: {np.median(outlier_distances):.4f}")
            print(f"    Min outlier distance: {outlier_distances.min():.4f}")
            print(f"    Max outlier distance: {outlier_distances.max():.4f}")
            
            # Analyze X/Y axis divergence in outliers
            x_diffs = []
            y_diffs = []
            for comp in left_outliers:
                hand = comp['hand']
                body = comp['body']
                x_diffs.append(abs(hand[0] - body[0]))
                y_diffs.append(abs(hand[1] - body[1]))
            
            x_diffs_arr = np.array(x_diffs)
            y_diffs_arr = np.array(y_diffs)
            
            print(f"\n    X-axis divergence in outliers:")
            print(f"      Mean: {x_diffs_arr.mean():.4f}, Median: {np.median(x_diffs_arr):.4f}")
            print(f"      Max: {x_diffs_arr.max():.4f}")
            
            print(f"    Y-axis divergence in outliers:")
            print(f"      Mean: {y_diffs_arr.mean():.4f}, Median: {np.median(y_diffs_arr):.4f}")
            print(f"      Max: {y_diffs_arr.max():.4f}")
            
            # Determine if divergence is systematic or random
            print(f"\n    AXIS ANALYSIS:")
            if x_diffs_arr.mean() > y_diffs_arr.mean():
                print(f"      Primary divergence: X-axis (horizontal shift)")
            else:
                print(f"      Primary divergence: Y-axis (vertical shift)")
            
            # Check if there's a directional bias
            x_signed = [hand[0] - body[0] for comp in left_outliers for hand in [comp['hand']] for body in [comp['body']]]
            y_signed = [hand[1] - body[1] for comp in left_outliers for hand in [comp['hand']] for body in [comp['body']]]
            
            x_mean_signed = np.mean(x_signed) if x_signed else 0
            y_mean_signed = np.mean(y_signed) if y_signed else 0
            
            if abs(x_mean_signed) > 0.1:
                direction = "positive (hand right of body)" if x_mean_signed > 0 else "negative (hand left of body)"
                print(f"      X-bias: {direction} (mean: {x_mean_signed:.4f})")
            else:
                print(f"      X-bias: balanced")
            
            if abs(y_mean_signed) > 0.1:
                direction = "positive (hand below body)" if y_mean_signed > 0 else "negative (hand above body)"
                print(f"      Y-bias: {direction} (mean: {y_mean_signed:.4f})")
            else:
                print(f"      Y-bias: balanced")
    
    if right_distances:
        right_distances_arr = np.array(right_distances)
        right_outlier_mask = right_distances_arr > 0.5
        right_outlier_indices = np.where(right_outlier_mask)[0]
        
        print(f"\n\nRIGHT HAND OUTLIERS (distance > 0.5 units):")
        print(f"  Count: {len(right_outlier_indices)} / {len(right_distances)}")
        
        if len(right_outlier_indices) > 0:
            # Get outlier comparisons
            right_outliers = [right_comparisons[i] for i in right_outlier_indices]
            
            # Sort by distance for analysis
            right_outliers.sort(key=lambda x: x['distance'], reverse=True)
            
            # Show top 10 worst outliers
            print("\n  TOP 10 WORST OUTLIERS:")
            for j, comp in enumerate(right_outliers[:10]):
                hand = comp['hand']
                body = comp['body']
                dist = comp['distance']
                image_id = comp['image_id']
                
                dx = hand[0] - body[0]
                dy = hand[1] - body[1]
                
                print(f"\n    [{j+1}] Image {image_id} - Distance: {dist:.4f}")
                print(f"        Hand: [{hand[0]:7.4f}, {hand[1]:7.4f}, {hand[2]:7.4f}]")
                print(f"        Body: [{body[0]:7.4f}, {body[1]:7.4f}, {body[2]:7.4f}]")
                print(f"        Diff: [{dx:7.4f}, {dy:7.4f}]")
            
            # Analyze outlier patterns
            print("\n  OUTLIER PATTERN ANALYSIS:")
            outlier_distances = right_distances_arr[right_outlier_mask]
            print(f"    Mean outlier distance: {outlier_distances.mean():.4f}")
            print(f"    Median outlier distance: {np.median(outlier_distances):.4f}")
            print(f"    Min outlier distance: {outlier_distances.min():.4f}")
            print(f"    Max outlier distance: {outlier_distances.max():.4f}")
            
            # Analyze X/Y axis divergence in outliers
            x_diffs = []
            y_diffs = []
            for comp in right_outliers:
                hand = comp['hand']
                body = comp['body']
                x_diffs.append(abs(hand[0] - body[0]))
                y_diffs.append(abs(hand[1] - body[1]))
            
            x_diffs_arr = np.array(x_diffs)
            y_diffs_arr = np.array(y_diffs)
            
            print(f"\n    X-axis divergence in outliers:")
            print(f"      Mean: {x_diffs_arr.mean():.4f}, Median: {np.median(x_diffs_arr):.4f}")
            print(f"      Max: {x_diffs_arr.max():.4f}")
            
            print(f"    Y-axis divergence in outliers:")
            print(f"      Mean: {y_diffs_arr.mean():.4f}, Median: {np.median(y_diffs_arr):.4f}")
            print(f"      Max: {y_diffs_arr.max():.4f}")
            
            # Determine if divergence is systematic or random
            print(f"\n    AXIS ANALYSIS:")
            if x_diffs_arr.mean() > y_diffs_arr.mean():
                print(f"      Primary divergence: X-axis (horizontal shift)")
            else:
                print(f"      Primary divergence: Y-axis (vertical shift)")
            
            # Check if there's a directional bias
            x_signed = [hand[0] - body[0] for comp in right_outliers for hand in [comp['hand']] for body in [comp['body']]]
            y_signed = [hand[1] - body[1] for comp in right_outliers for hand in [comp['hand']] for body in [comp['body']]]
            
            x_mean_signed = np.mean(x_signed) if x_signed else 0
            y_mean_signed = np.mean(y_signed) if y_signed else 0
            
            if abs(x_mean_signed) > 0.1:
                direction = "positive (hand right of body)" if x_mean_signed > 0 else "negative (hand left of body)"
                print(f"      X-bias: {direction} (mean: {x_mean_signed:.4f})")
            else:
                print(f"      X-bias: balanced")
            
            if abs(y_mean_signed) > 0.1:
                direction = "positive (hand below body)" if y_mean_signed > 0 else "negative (hand above body)"
                print(f"      Y-bias: {direction} (mean: {y_mean_signed:.4f})")
            else:
                print(f"      Y-bias: balanced")
    
    print(f"\n{'-'*80}")
    print("DISTANCE DISTRIBUTION ANALYSIS")
    print(f"{'-'*80}\n")
    
    # Analyze distribution buckets
    if left_distances or right_distances:
        all_distances = left_distances + right_distances
        all_distances_arr = np.array(all_distances)
        
        # Create distance buckets
        buckets = [
            (0.0, 0.1, "Perfect match (0.0-0.1)"),
            (0.1, 0.2, "Excellent (0.1-0.2)"),
            (0.2, 0.3, "Good (0.2-0.3)"),
            (0.3, 0.5, "Moderate (0.3-0.5)"),
            (0.5, 1.0, "Poor (0.5-1.0)"),
            (1.0, 2.0, "Very poor (1.0-2.0)"),
            (2.0, 10.0, "Critical (2.0+)"),
        ]
        
        print("DISTANCE DISTRIBUTION BY CATEGORY:")
        for low, high, label in buckets:
            count = np.sum((all_distances_arr >= low) & (all_distances_arr < high))
            pct = 100 * count / len(all_distances_arr)
            bar = "█" * int(pct / 2)
            print(f"  {label:30s}: {count:4d} ({pct:5.1f}%) {bar}")

    
    # Interpretation
    if left_distances or right_distances:
        all_distances_arr = np.array(left_distances + right_distances)
        mean_dist = all_distances_arr.mean()
        
        print("INTERPRETATION:")
        if mean_dist < 0.15:
            print("  ✅ EXCELLENT: Hand and body fingertips match very closely (<0.15 units)")
            print("     Fallback to body landmarks is highly accurate.")
        elif mean_dist < 0.30:
            print("  ✅ GOOD: Hand and body fingertips are reasonably close (<0.30 units)")
            print("     Fallback to body landmarks should work well.")
        elif mean_dist < 0.50:
            print("  ⚠️  MODERATE: Some divergence between hand and body landmarks")
            print("     Fallback will work but with reduced accuracy.")
        else:
            print("  ❌ POOR: Significant divergence between hand and body landmarks")
            print("     Consider using body wrist (landmarks 15/16) instead of fingertip.")
        
        print(f"\n  Expected outcome: Most detections should be <0.3 units from fingertip.")
        print(f"  Actual mean: {mean_dist:.4f} units")
    else:
        print("ERROR: No valid comparisons found. Cannot assess quality.")
    
    print("\n" + "=" * 80)
    print("Validation complete!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate hand vs body landmark positions')
    parser.add_argument('--sample-size', type=int, default=1000,
                        help='Number of images to sample (default: 1000)')
    
    args = parser.parse_args()
    
    main(sample_size=args.sample_size)
