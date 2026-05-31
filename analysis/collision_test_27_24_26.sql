-- Collision analysis: images where tie (27) co-occurs with handbag (26) or backpack (24)
-- All counts are per-image (one row per image_id regardless of detection count)

USE Stock;

-- 1. Baseline: how many images have each class at all
SELECT 'tie (27) total images'            AS label, COUNT(DISTINCT image_id) AS image_count FROM Detections WHERE class_id = 27
UNION ALL
SELECT 'handbag (26) total images',                 COUNT(DISTINCT image_id)               FROM Detections WHERE class_id = 26
UNION ALL
SELECT 'backpack (24) total images',                COUNT(DISTINCT image_id)               FROM Detections WHERE class_id = 24
UNION ALL
;

'''
tie (27) total images	4310455
handbag (26) total images	2080169
backpack (24) total images	691891
'''

-- 2. Direct collision: tie + handbag in same image
SELECT 'tie(27) AND handbag(26)',
    COUNT(DISTINCT d27.image_id)
FROM Detections d27
INNER JOIN Detections d26 ON d26.image_id = d27.image_id AND d26.class_id = 26
WHERE d27.class_id = 27
;
-- tie(27) AND handbag(26)	123616

-- 3. Direct collision: tie + backpack in same image
SELECT 'tie(27) AND backpack(24)',
    COUNT(DISTINCT d27.image_id)
FROM Detections d27
INNER JOIN Detections d24 ON d24.image_id = d27.image_id AND d24.class_id = 24
WHERE d27.class_id = 27
;
-- tie(27) AND backpack(24)	23631

-- 4. Triple collision: tie + handbag + backpack all in same image
SELECT 'tie(27) AND handbag(26) AND backpack(24)',
    COUNT(DISTINCT d27.image_id)
FROM Detections d27
INNER JOIN Detections d26 ON d26.image_id = d27.image_id AND d26.class_id = 26
INNER JOIN Detections d24 ON d24.image_id = d27.image_id AND d24.class_id = 24
WHERE d27.class_id = 27
;

-- tie(27) AND handbag(26) AND backpack(24)	4532


-- 5. tie + either bag (union of 2+3, no double-count)
SELECT 'tie(27) AND (handbag(26) OR backpack(24))',
    COUNT(DISTINCT d27.image_id)
FROM Detections d27
INNER JOIN Detections dbag ON dbag.image_id = d27.image_id AND dbag.class_id IN (24, 26)
WHERE d27.class_id = 27
;
-- tie(27) AND (handbag(26) OR backpack(24))	142719

-- 6. Mouth collision: tie (27) + covid mask (110) in same image
--    (relevant if tie routes to mouth slot)
SELECT 'tie(27) AND covid_mask(110) [mouth slot collision]',
    COUNT(DISTINCT d27.image_id)
FROM Detections d27
INNER JOIN Detections d110 ON d110.image_id = d27.image_id AND d110.class_id = 110
WHERE d27.class_id = 27
;
-- tie(27) AND covid_mask(110) [mouth slot collision]	32365

-- 7. Shoulder slot: tie (27) + handbag (26) already assigned to shoulder in ImagesDetections
--    (shows actual current collision rate in the processed data)
SELECT 'tie(27) images where shoulder_object already assigned to handbag(26)',
    COUNT(DISTINCT idet.image_id)
FROM ImagesDetections idet
INNER JOIN Detections d_shoulder ON d_shoulder.detection_id = idet.shoulder_object_id AND d_shoulder.class_id = 26
INNER JOIN Detections d_tie ON d_tie.image_id = idet.image_id AND d_tie.class_id = 27
;
-- tie(27) images where shoulder_object already assigned to handbag(26)	1403

-- 8. Breakdown: of all tie images, what fraction also have a bag/mask (collision risk summary)
SELECT
    COUNT(DISTINCT d27.image_id)                                          AS tie_images_total,
    COUNT(DISTINCT d_bag.image_id)                                        AS tie_AND_bag,
    COUNT(DISTINCT d_mask.image_id)                                       AS tie_AND_mask110,
    ROUND(100.0 * COUNT(DISTINCT d_bag.image_id)  / COUNT(DISTINCT d27.image_id), 1) AS pct_tie_also_bag,
    ROUND(100.0 * COUNT(DISTINCT d_mask.image_id) / COUNT(DISTINCT d27.image_id), 1) AS pct_tie_also_mask
FROM Detections d27
LEFT JOIN Detections d_bag  ON d_bag.image_id  = d27.image_id AND d_bag.class_id  IN (24, 26)
LEFT JOIN Detections d_mask ON d_mask.image_id = d27.image_id AND d_mask.class_id = 110
WHERE d27.class_id = 27
;
-- 4310735	142728	32365	3.3	0.8
