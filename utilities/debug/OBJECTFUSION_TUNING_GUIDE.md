# ObjectFusion Clustering Tuning Guide

## Problem Summary
You're clustering images based on:
- Face angles (pitch, yaw, roll): -90° to +90°
- Hand positions: normalized to face height (~-2 to +2)
- Object class_id: 0-107 (COCO + custom classes)
- Object bounding boxes: normalized coordinates

**Current Issues:**
1. Imbalanced cluster sizes (max 300k, median 1.5k)
2. Large clusters have high variance in class_id
3. Objects getting mixed within clusters despite consistent hands/angles

## Solution Implemented: Feature Standardization

I've added `StandardScaler` from scikit-learn to normalize all features to mean=0, std=1, then apply per-feature-group weights.

### New Parameters in `tools_clustering.py`

```python
self.USE_FEATURE_STANDARDIZATION = True  # Enable/disable standardization

self.FEATURE_WEIGHTS = {
    'face_angle': 1.5,      # pitch, yaw, roll
    'class_id': 3.0,        # Highest weight for object separation
    'confidence': 0.5,      # Low weight - mainly for tie-breaking
    'bbox': 1.0,            # Standard weight for object positions
}
```

### How It Works

**Before (problematic):**
```
pitch: -60 to +60     (range ~120)
yaw: -60 to +60       (range ~120)  
roll: -90 to +90      (range ~180)
class_id: 0 to 1070   (range ~1070, after CLASS_ID_WEIGHT × 10)
bbox_top: -2 to +2    (range ~4)
hand_x: -2 to +2      (range ~4)
```
K-means dominated by class_id (10x larger range), then face angles, with hands barely mattering.

**After (standardized):**
```
All features → StandardScaler → mean=0, std=1
Then multiply by FEATURE_WEIGHTS:
  - face_angle × 1.5
  - class_id × 3.0
  - confidence × 0.5
  - bbox × 1.0
```
Now class_id has 3x influence of bbox, 2x influence of face_angle, while all features contribute meaningfully.

## Tuning Steps

### 1. Test Current Settings (Recommended First)

Run with defaults:
```python
cl.USE_FEATURE_STANDARDIZATION = True  # Already set in code
cl.FEATURE_WEIGHTS = {
    'face_angle': 1.5,
    'class_id': 3.0,
    'confidence': 0.5,
    'bbox': 1.0,
}
```

Expected improvements:
- More balanced cluster sizes
- Better separation by class_id within similar poses
- Hands and bbox contributing more to clustering

### 2. Adjust Cluster Count

```python
# In Clustering_SQL.py, line ~133
N_CLUSTERS = 512  # Current

# Try these if needed:
N_CLUSTERS = 768   # More granular separation
N_CLUSTERS = 1024  # Very fine-grained
N_CLUSTERS = 384   # Fewer, larger clusters
```

**Rule of thumb:** With standardization, you may need 25-50% more clusters to achieve similar granularity because features are more balanced.

### 3. Adjust Feature Weights

If class_id still mixing too much:
```python
self.FEATURE_WEIGHTS = {
    'face_angle': 1.5,
    'class_id': 4.0,      # Increase from 3.0 → 4.0
    'confidence': 0.5,
    'bbox': 1.0,
}
```

If hands/bbox not mattering enough:
```python
self.FEATURE_WEIGHTS = {
    'face_angle': 1.5,
    'class_id': 3.0,
    'confidence': 0.5,
    'bbox': 1.5,          # Increase from 1.0 → 1.5
}
```

If face angle too dominant:
```python
self.FEATURE_WEIGHTS = {
    'face_angle': 1.0,    # Decrease from 1.5 → 1.0
    'class_id': 3.0,
    'confidence': 0.5,
    'bbox': 1.0,
}
```

### 4. Disable Standardization (Fallback)

If standardization makes things worse:
```python
# In tools_clustering.py, line ~30
self.USE_FEATURE_STANDARDIZATION = False
```

This reverts to old behavior (CLASS_ID_WEIGHT × 10).

## Diagnostic Queries

### Check cluster balance:
```sql
SELECT 
    COUNT(*) as num_clusters,
    MIN(size) as min_size,
    MAX(size) as max_size,
    AVG(size) as avg_size,
    STDDEV(size) as stddev_size,
    SUM(CASE WHEN size > 10000 THEN 1 ELSE 0 END) as clusters_over_10k
FROM (
    SELECT cluster_id, COUNT(*) as size
    FROM ImagesObjectFusion
    GROUP BY cluster_id
) AS cluster_sizes;
```

### Check class_id diversity in large clusters:
```sql
WITH cluster_sizes AS (
    SELECT cluster_id, COUNT(*) as size
    FROM ImagesObjectFusion
    GROUP BY cluster_id
    HAVING COUNT(*) > 10000
),
cluster_classes AS (
    SELECT 
        i.cluster_id,
        COUNT(DISTINCT i.left_hand_object) as distinct_left,
        COUNT(DISTINCT i.right_hand_object) as distinct_right,
        COUNT(DISTINCT i.top_face_object) as distinct_top_face,
        COUNT(DISTINCT i.bottom_face_object) as distinct_bottom_face
    FROM ImagesObjectFusion i
    INNER JOIN cluster_sizes cs ON i.cluster_id = cs.cluster_id
    GROUP BY i.cluster_id
)
SELECT 
    cs.cluster_id,
    cs.size,
    cc.distinct_left + cc.distinct_right + cc.distinct_top_face + cc.distinct_bottom_face as total_distinct_objects
FROM cluster_sizes cs
JOIN cluster_classes cc ON cs.cluster_id = cc.cluster_id
ORDER BY cs.size DESC;
```

### Examine a specific large cluster:
```sql
SELECT 
    cluster_id,
    COUNT(*) as count,
    left_hand_object,
    right_hand_object,
    top_face_object,
    AVG(pitch) as avg_pitch,
    AVG(yaw) as avg_yaw
FROM ImagesObjectFusion
WHERE cluster_id = <LARGE_CLUSTER_ID>
GROUP BY cluster_id, left_hand_object, right_hand_object, top_face_object
ORDER BY count DESC
LIMIT 20;
```

## Expected Results

### Good clustering indicators:
✅ Max cluster size < 50k (down from 300k)
✅ Cluster size std deviation < 3x median
✅ Large clusters have low class_id diversity (1-3 distinct objects per position)
✅ Similar poses + same object = same cluster
✅ Similar poses + different objects = different clusters

### Red flags:
⚠️ Many clusters with size < 10 (over-clustering)
⚠️ Large clusters with 10+ distinct objects per position
⚠️ Identical poses+objects in different clusters (under-clustering)

## Iterative Process

1. **Run with defaults** (face_angle=1.5, class_id=3.0, bbox=1.0)
2. **Check cluster size distribution** using SQL queries above
3. **If max cluster > 50k**: Increase `class_id` weight to 4.0 or 5.0
4. **If clusters too small**: Decrease `N_CLUSTERS` or reduce `class_id` weight
5. **If hands still inconsistent**: Increase `bbox` weight to 1.5
6. **Repeat** until satisfied

## Advanced: Per-Position Weighting

If you need different weights for different object positions (e.g., "face objects more important than hand objects"), you could modify `prepare_features_for_knn_v2` to weight individual detection positions:

```python
# Add to FEATURE_WEIGHTS:
self.FEATURE_WEIGHTS = {
    'face_angle': 1.5,
    'class_id': 3.0,
    'confidence': 0.5,
    'bbox': 1.0,
    # Per-position multipliers:
    'top_face_multiplier': 1.2,      # Face objects more important
    'bottom_face_multiplier': 1.2,
    'both_hands_multiplier': 1.0,
    'left_hand_multiplier': 0.8,     # Single hand less important
    'right_hand_multiplier': 0.8,
}
```

Then update the column weighting loop to check position prefixes.

## Notes

- Standardization requires scikit-learn (already in your environment)
- Scaler is fitted once on MODE 0 training data, reused for MODE 1 assignment
- Medians are calculated in standardized space for consistency
- The scaler is stored in `cl.feature_scaler` for the session

## Rollback

If anything breaks, disable standardization:
```python
cl.USE_FEATURE_STANDARDIZATION = False
```

And revert to commit before these changes.
