import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from my_declarative_base import Base
import json

import os
from pathlib import Path

# importing project-specific models
import sys
ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/takingstock/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)


# Setup database connection (using your existing config pattern)
from mp_db_io import DataIO
io = DataIO()
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

# Query cluster medians and sizes
query = text("SELECT c.cluster_id, c.cluster_median, COUNT(ic.image_id) as cluster_size FROM ObjectFusion c LEFT JOIN ImagesObjectFusion ic ON c.cluster_id = ic.cluster_id GROUP BY c.cluster_id ORDER BY cluster_size DESC LIMIT 200;")

results = session.execute(query).fetchall()
session.close()

# Parse and analyze the medians
print("Analyzing cluster medians...")
print("=" * 100)

analysis_data = []

for cluster_id, cluster_median_blob, cluster_size in results:
    try:
        # Unpickle the median
        cluster_median = pickle.loads(cluster_median_blob)
        
        # Extract features from the median array
        # Based on prepare_features_for_knn_v2, the order should be:
        # [pitch, yaw, roll, (5 positions × 6 fields each), (5 has_object indicators)]
        # = 3 + 30 + 5 = 38 total features (before standardization)
        
        if len(cluster_median) < 3:
            continue
        
        pitch = cluster_median[0] if len(cluster_median) > 0 else 0
        yaw = cluster_median[1] if len(cluster_median) > 1 else 0
        roll = cluster_median[2] if len(cluster_median) > 2 else 0
        
        # Extract class_id values for each detection position (indices: 3, 9, 15, 21, 27)
        # Each position has 6 values: [class_id, conf, top, left, right, bottom]
        class_ids = []
        if len(cluster_median) >= 33:  # 3 angles + 5*6 features
            class_ids = [
                cluster_median[3],      # both_hands_object class_id
                cluster_median[9],      # left_hand_object class_id
                cluster_median[15],     # right_hand_object class_id
                cluster_median[21],     # top_face_object class_id
                cluster_median[27],     # bottom_face_object class_id
            ]
        
        # Extract has_object values (if they exist, indices 33-37)
        has_objects = []
        if len(cluster_median) >= 38:
            has_objects = [
                cluster_median[33],     # both_hands has_object
                cluster_median[34],     # left_hand has_object
                cluster_median[35],     # right_hand has_object
                cluster_median[36],     # top_face has_object
                cluster_median[37],     # bottom_face has_object
            ]
        
        # Count non-zero class_ids (indicates how many positions have objects)
        num_objects = sum(1 for cid in class_ids if cid > 0.5)  # threshold for class_id presence
        
        # Check if custom objects (class_id > 79)
        has_custom = any(cid > 79 for cid in class_ids)
        
        analysis_data.append({
            'cluster_id': cluster_id,
            'cluster_size': cluster_size,
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'both_hands_class_id': class_ids[0] if len(class_ids) > 0 else 0,
            'left_hand_class_id': class_ids[1] if len(class_ids) > 1 else 0,
            'right_hand_class_id': class_ids[2] if len(class_ids) > 2 else 0,
            'top_face_class_id': class_ids[3] if len(class_ids) > 3 else 0,
            'bottom_face_class_id': class_ids[4] if len(class_ids) > 4 else 0,
            'num_objects_in_median': num_objects,
            'has_custom_object': has_custom,
            'median_length': len(cluster_median),
            'raw_median': cluster_median
        })
    except Exception as e:
        print(f"Error processing cluster {cluster_id}: {e}")
        continue

# Convert to DataFrame for analysis
df = pd.DataFrame(analysis_data)

print("\n=== OVERALL STATISTICS ===")
print(f"Total clusters analyzed: {len(df)}")
print(f"Total images: {df['cluster_size'].sum()}")
print(f"\nCluster size distribution:")
print(df['cluster_size'].describe())

print("\n=== MEGA-CLUSTERS (TOP 5) ===")
mega = df.head(5)
for idx, row in mega.iterrows():
    print(f"\nCluster {row['cluster_id']}: {row['cluster_size']:,} images")
    print(f"  Face angles: pitch={row['pitch']:.2f}, yaw={row['yaw']:.2f}, roll={row['roll']:.2f}")
    print(f"  Objects in median: {row['num_objects_in_median']}")
    print(f"  Class IDs: both={row['both_hands_class_id']:.0f}, left={row['left_hand_class_id']:.0f}, right={row['right_hand_class_id']:.0f}, top_face={row['top_face_class_id']:.0f}, bottom_face={row['bottom_face_class_id']:.0f}")
    print(f"  Has custom object (>79): {row['has_custom_object']}")

print("\n=== CLUSTERS BY OBJECT COUNT ===")
object_count_summary = df.groupby('num_objects_in_median').agg({
    'cluster_id': 'count',
    'cluster_size': ['sum', 'mean', 'max'],
    'has_custom_object': 'sum'
}).round(0)
object_count_summary.columns = ['num_clusters', 'total_images', 'avg_cluster_size', 'max_cluster_size', 'clusters_with_custom']
print(object_count_summary)

print("\n=== CLUSTERS WITH CUSTOM OBJECTS (>79) VS COCO ===")
if len(df[df['has_custom_object'] == True]) > 0 and len(df[df['has_custom_object'] == False]) > 0:
    custom_summary = df.groupby('has_custom_object').agg({
        'cluster_id': 'count',
        'cluster_size': ['sum', 'mean', 'max'],
        'num_objects_in_median': 'mean'
    }).round(2)
    custom_summary.columns = ['num_clusters', 'total_images', 'avg_cluster_size', 'max_cluster_size', 'avg_objects_per_cluster']
    print(custom_summary)
else:
    print("Note: All clusters in this sample have COCO objects only (no custom objects >79)")

print("\n=== TOP 20 CLUSTERS (excluding mega-clusters) ===")
good_clusters = df[df['cluster_size'] < 100000].head(20)
for idx, row in good_clusters.iterrows():
    print(f"Cluster {row['cluster_id']}: {row['cluster_size']:,} images | objects={row['num_objects_in_median']} | classes: both={row['both_hands_class_id']:.0f} left={row['left_hand_class_id']:.0f} right={row['right_hand_class_id']:.0f} | custom={row['has_custom_object']}")

print("\n=== ANALYZING MEGA-CLUSTER MEDIANS ===")
print("The two mega-clusters have medians with these object counts:")
for idx, row in df.head(2).iterrows():
    print(f"Cluster {row['cluster_id']} ({row['cluster_size']:,} images): {row['num_objects_in_median']} objects in median")
    print(f"  This suggests images in this cluster typically have {row['num_objects_in_median']} detections on average")

# Export full analysis to CSV for inspection
df.to_csv('cluster_medians_analysis.csv', index=False)
print("\n✓ Full analysis exported to cluster_medians_analysis.csv")
