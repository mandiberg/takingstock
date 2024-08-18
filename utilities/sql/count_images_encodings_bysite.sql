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

   
   
   
   
-- segment only
   
SELECT
    so.site_name_id,
    COUNT(so.image_id) AS image_count,
    SUM(CASE WHEN so.mongo_face_landmarks  IS NOT NULL THEN 1 ELSE 0 END) AS face_encodings68_not_null_count,
    SUM(CASE WHEN so.mongo_body_landmarks  IS NOT NULL THEN 1 ELSE 0 END) AS body_landmarks_not_null_count
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
