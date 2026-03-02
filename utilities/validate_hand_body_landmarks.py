"""
Validation script to compare hand landmarks vs body landmarks for fingertip positions.

This script validates that hand landmark[8] (fingertip) and body landmark[19/20] (fingertip)
produce similar positions where both exist, to ensure the fallback logic is accurate.

Usage:
    python utilities/validate_hand_body_landmarks.py [--sample-size N]
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

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
        # Testing swapped: Landmark 16 = right wrist, 15 = left wrist
        landmark_idx = 16 if side == 'left' else 15
        
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
    query = text(f"""
        SELECT DISTINCT s.image_id
        FROM SegmentBig_isface s
        INNER JOIN Encodings e ON s.image_id = e.image_id
        WHERE e.is_dupe_of IS NULL
        AND e.mongo_hand_landmarks_norm = 1
        AND e.mongo_body_landmarks_norm = 1
        ORDER BY RAND()
        LIMIT {sample_size}
    """)
    
    result = session.execute(query)
    image_ids = [row[0] for row in result.all()]
    
    print(f"Found {len(image_ids)} images with both hand and body landmarks")
    
    if len(image_ids) == 0:
        print("ERROR: No images found with both landmarks. Cannot validate.")
        return
    
    # Collect comparison data
    left_distances = []
    right_distances = []
    
    missing_hand_data = 0
    missing_body_data = 0
    both_present = 0
    success_count = 0
    
    print("\nProcessing images...")
    for i, image_id in enumerate(image_ids):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_ids)} images...")
        
        # Fetch MongoDB data
        mongo_data = io.get_encodings_mongo(image_id)
        hand_results = mongo_data[5]  # 6th element is hand_results
        # body_landmarks_3D = mongo_data[4]  # 5th element is body_landmarks_3D
        body_landmarks_norm = mongo_data[3]  # 4th element is body_landmarks_norm
        
        # Extract left hand/body fingertips
        left_hand = extract_hand_fingertip(hand_results, 'left')
        left_body = extract_body_fingertip(body_landmarks_norm, io, sort_pose, 'left')
        
        if left_hand and left_body:
            both_present += 1
            dist = calculate_euclidean_distance(left_hand, left_body)
            if dist is not None:
                left_distances.append(dist)
                if success_count < 5:
                    print(f"\n  [SUCCESS {success_count+1}] Image {image_id} LEFT:")
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
        
        if right_hand and right_body:
            both_present += 1
            dist = calculate_euclidean_distance(right_hand, right_body)
            if dist is not None:
                right_distances.append(dist)
        elif not right_hand:
            missing_hand_data += 1
        elif not right_body:
            missing_body_data += 1
    
    print(f"\n{'-'*80}")
    print("VALIDATION RESULTS")
    print(f"{'-'*80}\n")
    
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
    print(f"  Images queried: {len(image_ids)}")
    print(f"  Images with both hand AND body data: {both_present}")
    print(f"  Missing hand data: {missing_hand_data} instances")
    print(f"  Missing body data: {missing_body_data} instances")
    print(f"{'-'*80}\n")
    
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
