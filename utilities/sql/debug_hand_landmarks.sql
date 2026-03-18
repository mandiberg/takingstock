-- Debug: Check for MongoDB hand_results availability

-- 1. Check how many images in Encodings have hand landmarks flag
SELECT 
    'Encodings hand landmark flags' as check_name,
    COUNT(DISTINCT image_id) as total_images,
    COUNT(DISTINCT CASE WHEN mongo_hand_landmarks = 1 THEN image_id END) as has_hand_landmarks_flag,
    COUNT(DISTINCT CASE WHEN mongo_hand_landmarks_norm = 1 THEN image_id END) as has_hand_landmarks_norm_flag
FROM Encodings
WHERE is_dupe_of IS NULL;

-- 2. Check the overlap: filtered images with detections AND hand landmarks
SELECT 
    'Images usable for ObjectFusion with hand landmarks' as check_name,
    COUNT(DISTINCT s.image_id) as image_count
FROM SegmentBig_isface s 
JOIN Encodings e ON s.image_id = e.image_id 
INNER JOIN Detections d ON d.image_id = s.image_id
WHERE e.is_dupe_of IS NULL
  AND d.bbox_norm IS NOT NULL
  AND e.mongo_hand_landmarks_norm = 1;

-- 3. Alternative: maybe checking for pitch/yaw/roll (face landmarks)
SELECT 
    'Images with face pose data (pitch/yaw/roll)' as check_name,
    COUNT(DISTINCT s.image_id) as image_count
FROM SegmentBig_isface s 
JOIN Encodings e ON s.image_id = e.image_id 
INNER JOIN Detections d ON d.image_id = s.image_id
WHERE e.is_dupe_of IS NULL
  AND d.bbox_norm IS NOT NULL
  AND e.pitch IS NOT NULL 
  AND e.yaw IS NOT NULL 
  AND e.roll IS NOT NULL;

-- 4. Check breakdown of what's missing
SELECT 
    'Breakdown of missing requirements' as check_name,
    COUNT(DISTINCT s.image_id) as total_filtered,
    COUNT(DISTINCT CASE WHEN e.pitch IS NULL OR e.yaw IS NULL OR e.roll IS NULL THEN s.image_id END) as missing_face_pose,
    COUNT(DISTINCT CASE WHEN e.mongo_hand_landmarks_norm IS NULL OR e.mongo_hand_landmarks_norm = 0 THEN s.image_id END) as missing_hand_landmarks,
    COUNT(DISTINCT CASE WHEN d.bbox_norm IS NULL THEN s.image_id END) as missing_bbox_norm
FROM SegmentBig_isface s 
JOIN Encodings e ON s.image_id = e.image_id 
INNER JOIN Detections d ON d.image_id = s.image_id
WHERE e.is_dupe_of IS NULL;
