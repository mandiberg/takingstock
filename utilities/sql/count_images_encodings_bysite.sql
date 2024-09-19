-- attempted to run Mar23, and quit after 300s

-- ran successfully on april 14. 

Use Stock;

SELECT
    i.site_name_id,
    COUNT(i.image_id) AS image_count,
    COUNT(e.encoding_id) AS encoding_count,
    SUM(e.is_face) AS is_face_count,
FROM
    Images i
LEFT JOIN
    Encodings e ON i.image_id = e.image_id
GROUP BY
    i.site_name_id
ORDER BY
    i.site_name_id;

   
   
SELECT
    COUNT(i.image_id) AS image_count,
    COUNT(e.encoding_id) AS encoding_count
FROM
    Images i
LEFT JOIN
    Encodings e ON i.image_id = e.image_id
WHERE i.site_name_id = 6
;

   
   
-- segment only
   
SELECT
    so.site_name_id,
    COUNT(so.image_id) AS image_count,
    SUM(CASE WHEN so.mongo_face_landmarks  IS NOT NULL THEN 1 ELSE 0 END) AS face_encodings68_not_null_count,
    SUM(CASE WHEN so.location_id  IS NOT NULL THEN 1 ELSE 0 END) AS location_id_count,
    SUM(CASE WHEN so.mongo_body_landmarks  IS NOT NULL THEN 1 ELSE 0 END) AS body_landmarks_not_null_count,
    SUM(CASE WHEN so.mongo_body_landmarks_norm  IS NOT NULL THEN 1 ELSE 0 END) AS mongo_body_landmarks_norm_count,
    SUM(CASE WHEN so.mongo_hand_landmarks  IS NOT NULL THEN 1 ELSE 0 END) AS hand_landmarks_not_null_count,
    SUM(CASE WHEN so.mongo_hand_landmarks_norm  IS NOT NULL THEN 1 ELSE 0 END) AS hand_body_landmarks_norm_count
FROM
    SegmentOct20 so 
GROUP BY
    so.site_name_id
ORDER BY
    so.site_name_id;

   
-- segment only with helper sub
   
SELECT
    so.site_name_id,
    COUNT(so.image_id) AS image_count,
    SUM(CASE WHEN so.mongo_face_landmarks IS NOT NULL THEN 1 ELSE 0 END) AS face_encodings68_not_null_count,
    SUM(CASE WHEN so.mongo_body_landmarks IS NOT NULL THEN 1 ELSE 0 END) AS body_landmarks_not_null_count
FROM
    SegmentOct20 so 
INNER JOIN
SegmentHelperApril4_topic17 shat ON so.image_id = shat.image_id 
GROUP BY
    so.site_name_id
ORDER BY
    so.site_name_id;

   
   
-- count ages per site
   
SELECT 
    site_name_id,
    SUM(CASE WHEN age_id IS NULL THEN image_count ELSE 0 END) AS 'NULL',
    SUM(CASE WHEN age_id = 1 THEN image_count ELSE 0 END) AS '1',
    SUM(CASE WHEN age_id = 2 THEN image_count ELSE 0 END) AS '2',
    SUM(CASE WHEN age_id = 3 THEN image_count ELSE 0 END) AS '3',
    SUM(CASE WHEN age_id = 4 THEN image_count ELSE 0 END) AS '4',
    SUM(CASE WHEN age_id = 5 THEN image_count ELSE 0 END) AS '5',
    SUM(CASE WHEN age_id = 6 THEN image_count ELSE 0 END) AS '6',
    SUM(CASE WHEN age_id = 7 THEN image_count ELSE 0 END) AS '7'
FROM 
(
    SELECT 
        site_name_id,
        age_id,
        COUNT(*) AS image_count
    FROM 
        Images
    JOIN Encodings ON Encodings.image_id = Images.image_id 
    WHERE Encodings.is_face = 1
    GROUP BY 
        site_name_id, age_id
) AS counts
GROUP BY 
    site_name_id
ORDER BY 
    site_name_id;
   
   
   
-- count of poses
   
SELECT
    ip.cluster_id,
    COUNT(so.image_id) AS image_count,
    SUM(CASE WHEN pb.bbox_67  IS NOT NULL THEN 1 ELSE 0 END) AS phone bbox
FROM
    SegmentOct20 so 
INNER JOIN ImagesPoses ip ON ip.image_id = so.image_id
JOIN PhoneBbox pb ON pb.image_id = so.image_id 
GROUP BY
    ip.cluster_id
ORDER BY
    ip.cluster_id;
   
   
-- count of fusion poses
   
SELECT
    ih.cluster_id,
    COUNT(so.image_id) AS image_count,
    SUM(CASE WHEN pb.bbox_67  IS NOT NULL THEN 1 ELSE 0 END) AS phone_bbox
FROM
    SegmentOct20 so 
INNER JOIN ImagesHandsPositions ihp ON ihp.image_id = so.image_id
INNER JOIN ImagesHandsGestures ihg ON ihg.image_id = so.image_id
JOIN PhoneBbox pb ON pb.image_id = so.image_id 
WHERE ihp.cluster_id = 113
GROUP BY
    ihg.cluster_id
ORDER BY
    ihg.cluster_id;
   

   
-- matrix of fusion poses
   
   
SELECT 
    ihp.cluster_id AS ihp_cluster,  -- Row: ImagesHandsPoses cluster_id
    SUM(CASE WHEN ih.cluster_id = 0 THEN 1 ELSE 0 END) AS ih_0,  -- Column for ih.cluster_id = 0
    SUM(CASE WHEN ih.cluster_id = 1 THEN 1 ELSE 0 END) AS ih_1,  -- Column for ih.cluster_id = 1
    SUM(CASE WHEN ih.cluster_id = 2 THEN 1 ELSE 0 END) AS ih_2,  -- Column for ih.cluster_id = 2
    SUM(CASE WHEN ih.cluster_id = 3 THEN 1 ELSE 0 END) AS ih_3  -- Column for ih.cluster_id = 3
    -- Add more CASE statements for additional clusters if necessary
    -- SUM(CASE WHEN ih.cluster_id = 75 THEN 1 ELSE 0 END) AS ih_75 -- Example cluster for ihp
FROM 
    SegmentOct20 so
JOIN 
    ImagesHandsPoses ihp ON ihp.image_id = so.image_id
JOIN 
    ImagesHands ih ON ih.image_id = so.image_id
GROUP BY 
    ihp.cluster_id  -- Group by ImagesHandsPoses cluster_id to create rows
ORDER BY 
    ihp_cluster;

   
   

   
   
 SELECT DISTINCT(s.image_id), s.site_name_id, s.contentUrl, s.imagename, s.description, s.face_x, s.face_y, s.face_z, s.mouth_gap, s.bbox, s.site_image_id, 
 ibg.lum, ibg.lum_bb, ibg.hue, ibg.hue_bb, ibg.sat, ibg.sat_bb, ibg.val, ibg.val_bb, ibg.lum_torso,ibg.lum_torso_bb  
FROM SegmentOct20 s  JOIN ImagesHandsPoses ihp ON s.image_id = ihp.image_id  
JOIN ImagesHands ih ON s.image_id = ih.image_id  JOIN ImagesBackground ibg ON s.image_id = ibg.image_id  
WHERE  s.is_dupe_of IS NULL  AND s.face_x > -50  AND s.age_id NOT IN (1,2,3)  AND s.mongo_body_landmarks = 1     
AND ihp.cluster_id = 15  AND ih.cluster_id = 80  LIMIT 1000;


SELECT COUNT(*) 
FROM SegmentOct20 s  JOIN ImagesHandsPoses ihp ON s.image_id = ihp.image_id  
JOIN ImagesHands ih ON s.image_id = ih.image_id    
WHERE   ihp.cluster_id = 15  AND ih.cluster_id = 80  ;

   
   
   
   
   
   
   
   
   
   
   
   -- stuff
   
       SUM(CASE WHEN so.mongo_face_landmarks  IS NOT NULL THEN 1 ELSE 0 END) AS face_encodings68_not_null_count,
    SUM(CASE WHEN so.mongo_body_landmarks  IS NOT NULL THEN 1 ELSE 0 END) AS body_landmarks_not_null_count,
    SUM(CASE WHEN so.mongo_body_landmarks_norm  IS NOT NULL THEN 1 ELSE 0 END) AS mongo_body_landmarks_norm_count

   
   
   
   
   
   


