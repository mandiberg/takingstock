

USE stock;
SET GLOBAL innodb_buffer_pool_size=8053063680;



       

SELECT e.image_id, i.imagename
FROM Encodings e 
JOIN Images i ON e.image_id = i.image_id
WHERE i.site_name_id = 9 and e.is_face IS NULL
;


SELECT e.encoding_id 
FROM Encodings e 
WHERE e.is_face = 0
and e.is_body  = 1
and e.encoding_id > 100000000
and e.encoding_id < 130000000
LIMIT 1
;
 

-- 4287381

-- Identify duplicate image_id entries in the Images table
SELECT site_image_id, COUNT(*)
FROM Images i 
WHERE i.site_name_id = 3
GROUP BY site_image_id
HAVING COUNT(*) > 1;

-- Identify duplicate image_id entries in the Segment table
SELECT image_id, COUNT(*)
FROM SegmentOct20 so  
GROUP BY image_id
HAVING COUNT(*) > 1;



-- fix bad imagename paths (maybe not working anymore?)
-- get count
       SELECT COUNT(image_id)
        FROM Images
        WHERE site_name_id = 1
        AND imagename LIKE '/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/getty_33333_china/images_china_lastset/%'
; 


-- update Images
        UPDATE Images
    SET imagename = REPLACE(imagename, '/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/getty_33333_china/images_china_lastset/', '')        
        WHERE site_name_id = 1
        AND imagename LIKE '/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/getty_33333_china/images_china_lastset/%'
        LIMIT 1000000; 


-- update segment with images 
        UPDATE SegmentOct20 so
        JOIN Images i ON i.image_id = so.image_id
     SET so.imagename = i.imagename
        WHERE i.site_name_id = 1
        AND so.imagename LIKE '/Users/%'
        ;


-- set cells to NULL
SET GLOBAL innodb_buffer_pool_size=8294967296;


SELECT COUNT(image_id)
FROM SegmentBig_isface
;
       
UPDATE SegmentBig_isface
SET    mongo_tokens = NULL
WHERE  image_id > 88000000
AND  image_id < 89300000
AND mongo_tokens = 1
;





ALTER TABLE HandsGestures128 RENAME TO HandsGestures;
ALTER TABLE ImagesHandsGestures128 RENAME TO ImagesHandsGestures;

ALTER TABLE SegmentBig_isface 
ADD column    pitch DECIMAL (6,3),
ADD column    yaw DECIMAL (6,3),
ADD column    roll DECIMAL (6,3)
;


SELECT *
FROM Encodings
WHERE image_id in (1835647, 1851989, 1900768, 1943275, 1972513, 1982528, 1996886, 2021344, 2049678, 2052201)
;

-- SET IS DUPE OF

USE Stock;
UPDATE Encodings 
SET is_dupe_of = 1
WHERE image_id IN (
110661171
);


UPDATE Images22412
SET object_id = 0
WHERE image_id in (113153070)
;

UPDATE Images553
SET object_id = 1, orientation = 0
WHERE object_id = 10
;


-- >>> 66

SELECT DISTINCT is_dupe_of
FROM Encodings
ORDER BY is_dupe_of;




;

USE Stock;
CREATE TABLE BsonFileLog (
    counter_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    completed_bson_file varchar(300)
);



CREATE TABLE BsonIdLog (
    counter_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    completed_bson_file varchar(100)
);

DELETE FROM BsonFileLog;

INSERT INTO BsonFileLog (completed_bson_file) VALUES ('encodings_batch_7900001.bson');

/* 
I have 30M rows in compare_sql_mongo_results.
None of them have is_face or is_body
I need to move that data from Encodings to here.
I need to move all of the is_face and is_body data from Encodings to compare_sql_mongo_results.
*/
Use STOCK;


-- select count of each BOOL column in compare_sql_mongo_results that is TRUE
SELECT 
    SUM(CASE WHEN face_landmarks = 1 THEN 1 ELSE 0 END) AS face_landmarks_true,
    SUM(CASE WHEN body_landmarks = 1 THEN 1 ELSE 0 END) AS body_landmarks_true,
    SUM(CASE WHEN face_encodings68 = 1 THEN 1 ELSE 0 END) AS face_encodings68_true,
    SUM(CASE WHEN nlms = 1 THEN 1 ELSE 0 END) AS nlms_true,
    SUM(CASE WHEN left_hand = 1 THEN 1 ELSE 0 END) AS left_hand_true,
    SUM(CASE WHEN right_hand = 1 THEN 1 ELSE 0 END) AS right_hand_true,
    SUM(CASE WHEN body_world_landmarks = 1 THEN 1 ELSE 0 END) AS body_world_landmarks_true,
    SUM(CASE WHEN is_face = 1 THEN 1 ELSE 0 END) AS is_face_true,
    SUM(CASE WHEN is_body = 1 THEN 1 ELSE 0 END) AS is_body_true
FROM compare_sql_mongo_results_ultradone
;

USE Stock;
-- select count of each BOOL column in compare_sql_mongo_results that is FALSE  
SELECT 
    SUM(CASE WHEN face_landmarks = 0 THEN 1 ELSE 0 END) AS face_landmarks_false,
    SUM(CASE WHEN body_landmarks = 0 THEN 1 ELSE 0 END) AS body_landmarks_false,
    SUM(CASE WHEN face_encodings68 = 0 THEN 1 ELSE 0 END) AS face_encodings68_false,
    SUM(CASE WHEN nlms = 0 THEN 1 ELSE 0 END) AS nlms_false,
    SUM(CASE WHEN left_hand = 0 THEN 1 ELSE 0 END) AS left_hand_false,
    SUM(CASE WHEN right_hand = 0 THEN 1 ELSE 0 END) AS right_hand_false,
    SUM(CASE WHEN body_world_landmarks = 0 THEN 1 ELSE 0 END) AS body_world_landmarks_false,
    SUM(CASE WHEN is_face = 1 THEN 1 ELSE 0 END) AS is_face_true,
    SUM(CASE WHEN is_body = 1 THEN 1 ELSE 0 END) AS is_body_true
FROM compare_sql_mongo_results_ultradone
;



-- update segmentbig_isface to set pitch, yaw, roll from encodings
USE Stock;
UPDATE SegmentBig_isface so
JOIN Encodings e ON so.image_id = e.image_id
JOIN ImagesArmsPoses3D iap ON so.image_id = iap.image_id
SET so.pitch = e.pitch,
    so.yaw = e.yaw,
    so.roll = e.roll
WHERE so.pitch IS NULL
AND so.yaw IS NULL
AND so.roll IS NULL
AND e.pitch IS NOT NULL
AND e.yaw IS NOT NULL
AND e.roll IS NOT NULL
-- AND iap.cluster_id = 237
;

-- check how many rows were updated
SELECT COUNT(*)
FROM SegmentBig_isface
WHERE pitch IS NOT NULL
AND yaw IS NOT NULL
AND roll IS NOT NULL
;

DELETE FROM SegmentHelper_nov2025_faces_without_bbox_still;

-- insert into SegmentHelper_nov2025_faces_without_bbox_still
INSERT INTO SegmentHelper_nov2025_faces_without_bbox_still (image_id)
SELECT s.image_id
FROM SegmentHelper_nov2025_faces_without_bbox s
JOIN Encodings e on s.image_id = e.image_id
WHERE e.is_face = 1
AND e.bbox is NULL
-- LIMIT 10
;

 SELECT *
 FROM Images
 WHERE site_name_id = 4
 AND site_image_id = "1002087854"
 ;
 
 SELECT *
 FROM WanderingImages wi 
 WHERE wi.wandering_name_site_id = "1002087854.3"
;

DELETE 
FROM WanderingImages wi 
WHERE wi.wandering_image_id > 6544803
;

 USE Stock;
 SELECT *
 FROM Encodings
 WHERE image_id = 38921748
 ;

