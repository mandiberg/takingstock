#!/usr/bin/env python3
"""
Test script to verify deterministic grouped-by-row ranking for ArmsPoses3D_ObjectFusion mode.
Loads the generated CSV and calls find_sorted_suitable_indices to inspect pair selection order.
"""
import sys
import os
import json
import pandas as pd

# Add facemap module to path
if sys.platform == "darwin":
    sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
elif sys.platform == "win32":
    sys.path.insert(1, 'C:/Users/jhash/Documents/GitHub/facemap2/')

from mp_sort_pose import SortPose

# Configuration for test
CSV_FOLDER = "utilities/data/heft_ArmsPoses3D_768_ObjectFusion_768"
CSV_FILE = "ArmsPoses3D_67.csv"
MANIFEST_FILE = "fusion_manifest.json"
TOPIC_NO = 67
MIN_VALUE = 5  # Capture pairs with count >= 5
SORT_TYPE = "object_fusion"

def test_pair_selection():
    """Load CSV and test pair selection."""
    csv_path = os.path.join(CSV_FOLDER, CSV_FILE)
    manifest_path = os.path.join(CSV_FOLDER, MANIFEST_FILE)
    
    print(f"Testing pair selection for {CSV_FILE}")
    print(f"CSV Path: {csv_path}")
    print(f"Manifest Path: {manifest_path}")
    print()
    
    # Load manifest to check metadata
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        print(f"Manifest loaded: cluster_type={manifest.get('cluster_type')}")
        print(f"Available HSV bins: {manifest['files'][CSV_FILE].get('available_hsv_bins', [])}")
        print()
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"CSV shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()[:10]}... (showing first 10)")
    print()
    
    # Show data summary
    print("=== Data Summary ===")
    print(f"Total cells: {df.shape[0] * (df.shape[1] - 1)}")  # -1 for arms_cluster_id
    
    # Count non-zero cells (excluding arms_cluster_id and total)
    non_zero_sum = 0
    non_zero_count = 0
    for col in df.columns:
        if col not in ['arms_cluster_id', 'total']:
            non_zero_mask = df[col] > 0
            non_zero_count += non_zero_mask.sum()
            non_zero_sum += df.loc[non_zero_mask, col].sum()
    
    print(f"Non-zero cells: {non_zero_count}")
    print(f"Total image count: {non_zero_sum}")
    print()
    
    # Show rows with data
    rows_with_data = df[df['total'] > 0]['arms_cluster_id'].tolist()
    print(f"Rows with data (arms_cluster_id): {rows_with_data}")
    print()
    
    # Initialize sorter with proper motion dict structure (all keys set to False)
    print("=== Testing find_sorted_suitable_indices ===")
    motion_config = {
        'side_to_side': False, 
        'nod': False, 
        'forward_smile': False,
        'forward_wider': False,
        'laugh': False,
        'forward_nosmile': False,
        'static_pose': True,  # Use static_pose as default
        'simple': False,
        'use_all': False
    }
    cl = SortPose(motion=motion_config, face_height_output='face_height', SORT_TYPE=SORT_TYPE, VERBOSE=True)
    
    try:
        selected_pairs = cl.find_sorted_suitable_indices(
            topic_no=TOPIC_NO,
            min_value=MIN_VALUE,
            folder_path=CSV_FOLDER,
            hsv_cluster_groups=None,  # Not using HSV for this matrix
            manifest_file=MANIFEST_FILE
        )
        
        print()
        print(f"Successfully retrieved {len(selected_pairs)} pairs with count >= {MIN_VALUE}")
        print()
        
        # Display the pairs in order
        print("=== Selected Pairs (in selection order) ===")
        print(f"{'#':<4} {'Row':<6} {'Col':<8} {'Count':<8}")
        print("-" * 30)
        for i, (row, col) in enumerate(selected_pairs[:50]):  # Show first 50 pairs
            # Get the count value from the CSV
            col_name = f"object_cluster_{col}"
            if col_name in df.columns:
                count = df.loc[df['arms_cluster_id'] == row, col_name].values[0]
            else:
                count = "?"
            print(f"{i+1:<4} {row:<6} {col:<8} {count:<8}")
        
        if len(selected_pairs) > 50:
            print(f"... and {len(selected_pairs) - 50} more pairs")
        
        print()
        print("=== Verification ===")
        # Verify row-major ordering
        rows_in_selection = [pair[0] for pair in selected_pairs]
        unique_rows = []
        for r in rows_in_selection:
            if not unique_rows or unique_rows[-1] != r:
                unique_rows.append(r)
        
        is_row_major = unique_rows == sorted(unique_rows)
        print(f"Rows appear in ascending order: {is_row_major}")
        print(f"Unique rows in selection order: {unique_rows}")
        
        if is_row_major:
            print("✓ PASS: Deterministic grouped-by-row ranking verified!")
        else:
            print("✗ FAIL: Rows not in ascending order")
        
        return selected_pairs
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_pair_selection()
