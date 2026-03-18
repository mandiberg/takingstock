-- Debug: Check bbox normalization status in Detections table

-- 1. Count detections with and without bbox_norm
SELECT 
    'Detections bbox_norm status' as check_name,
    COUNT(*) as total_detections,
    COUNT(bbox_norm) as has_bbox_norm,
    COUNT(*) - COUNT(bbox_norm) as missing_bbox_norm,
    ROUND((COUNT(bbox_norm) / COUNT(*)) * 100, 2) as pct_normalized
FROM Detections;

-- 2. Count distinct images with at least one normalized detection
SELECT 
    'Images with normalized detections' as check_name,
    COUNT(DISTINCT image_id) as image_count
FROM Detections
WHERE bbox_norm IS NOT NULL;

-- 3. Count distinct images with NO normalized detections (all detections unnormalized)
SELECT 
    'Images with ONLY unnormalized detections' as check_name,
    COUNT(DISTINCT image_id) as image_count
FROM Detections d1
WHERE NOT EXISTS (
    SELECT 1 FROM Detections d2 
    WHERE d2.image_id = d1.image_id 
    AND d2.bbox_norm IS NOT NULL
);

-- 4. Full breakdown by image
SELECT 
    'Image breakdown by bbox_norm status' as check_name,
    (SELECT COUNT(DISTINCT image_id) FROM Detections) as total_images_with_detections,
    (SELECT COUNT(DISTINCT image_id) FROM Detections WHERE bbox_norm IS NOT NULL) as images_with_at_least_one_normalized,
    (SELECT COUNT(DISTINCT image_id) FROM Detections d1 
     WHERE NOT EXISTS (
         SELECT 1 FROM Detections d2 
         WHERE d2.image_id = d1.image_id 
         AND d2.bbox_norm IS NOT NULL
     )) as images_with_zero_normalized;

-- 5. Check the Clustering_SQL filtered set - how many have bbox_norm?
SELECT 
    'Clustering_SQL filtered images with bbox_norm' as check_name,
    COUNT(DISTINCT d.image_id) as total_filtered,
    COUNT(DISTINCT CASE WHEN d.bbox_norm IS NOT NULL THEN d.image_id END) as has_normalized,
    COUNT(DISTINCT CASE WHEN d.bbox_norm IS NULL THEN d.image_id END) as missing_normalized
FROM SegmentBig_isface s 
JOIN Encodings e ON s.image_id = e.image_id 
INNER JOIN Detections d ON d.image_id = s.image_id
WHERE e.is_dupe_of IS NULL;

-- 6. The critical query: images in filtered set with at least one normalized bbox
SELECT 
    'Images actually usable for ObjectFusion clustering' as check_name,
    COUNT(DISTINCT d.image_id) as image_count
FROM SegmentBig_isface s 
JOIN Encodings e ON s.image_id = e.image_id 
INNER JOIN Detections d ON d.image_id = s.image_id
WHERE e.is_dupe_of IS NULL
  AND d.bbox_norm IS NOT NULL;
