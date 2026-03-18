-- Debug: Compare image counts between analyze_detection_positions.py and Clustering_SQL.py

-- 1. analyze_detection_positions.py query
-- Simple: just all distinct image_ids in Detections table
SELECT 'ANALYZE_DETECT_POSITIONS - All Detections' as query_name, 
       COUNT(DISTINCT image_id) as image_count
FROM Detections;

-- 2. Clustering_SQL.py MODE 0 ObjectFusion query (simplified)
-- Filters by is_dupe_of IS NULL from SegmentBig_isface
SELECT 'CLUSTERING_SQL - SegmentBig_isface filtered' as query_name,
       COUNT(DISTINCT s.image_id) as image_count
FROM SegmentBig_isface s
WHERE s.is_dupe_of IS NULL;

-- 3. Check SegmentTable_name configuration
SELECT 'SegmentTable_name breakdown' as query_name, 
       'SegmentBig_isface' as table_name,
       COUNT(DISTINCT image_id) as total_count,
       COUNT(DISTINCT CASE WHEN is_dupe_of IS NULL THEN image_id END) as non_dupe_count,
       COUNT(DISTINCT CASE WHEN is_dupe_of IS NOT NULL THEN image_id END) as dupe_count
FROM SegmentBig_isface;

-- 4. Check if Detections has rows not in SegmentBig_isface
SELECT 'Detections not in SegmentBig_isface' as query_name,
       COUNT(DISTINCT d.image_id) as count_in_detections_only
FROM Detections d
LEFT JOIN SegmentBig_isface s ON d.image_id = s.image_id
WHERE s.image_id IS NULL;

-- 5. Check how many distinct image_ids are in Detections vs Encodings
SELECT 'Image count comparison' as comparison,
       (SELECT COUNT(DISTINCT image_id) FROM Detections) as detections_distinct,
       (SELECT COUNT(DISTINCT image_id) FROM SegmentBig_isface) as segmentbig_total,
       (SELECT COUNT(DISTINCT image_id) FROM SegmentBig_isface WHERE is_dupe_of IS NULL) as segmentbig_non_dupe;

-- 6. Check if there are images with detections but no segment record
SELECT 'Coverage check' as check_name,
       (SELECT COUNT(DISTINCT image_id) FROM Detections) as total_detection_images,
       (SELECT COUNT(DISTINCT d.image_id) 
        FROM Detections d 
        JOIN SegmentBig_isface s ON d.image_id = s.image_id 
        WHERE s.is_dupe_of IS NULL) as detection_images_in_segment_nondupe,
       ((SELECT COUNT(DISTINCT image_id) FROM Detections) -
        (SELECT COUNT(DISTINCT d.image_id) 
         FROM Detections d 
         JOIN SegmentBig_isface s ON d.image_id = s.image_id 
         WHERE s.is_dupe_of IS NULL)) as detection_images_not_in_segment_nondupe;
