

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
WHERE image_id = 32219
;

 SELECT *
 FROM Images
  WHERE site_name_id = 2
 AND site_image_id = "1090080953"
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
 WHERE image_id = 1405531
--   WHERE encoding_id = 125661925
 ;

 -- 129159642
 -- 128788809
 
 
 SELECT *
FROM ImagesHSV
WHERE image_id = 30402
;

-- gets me HEFT object images for YOLO/bbox testing
SELECT i.site_name_id, i.imagename
FROM Images i
JOIN ImagesKeywords ik on i.image_id = ik.image_id
JOIN ImagesKeywords ik2 on i.image_id = ik2.image_id
JOIN ImagesKeywords ik3 on i.image_id = ik3.image_id
JOIN SegmentOct20 s on i.image_id = s.image_id
-- WHERE ik.keyword_id in (22411)
-- WHERE ik.keyword_id in (22101,444,22191,16045,11549,133300,133777)
-- WHERE ik.keyword_id in (1991,220,133822)
-- WHERE ik.keyword_id in (22269, 5271)
-- WHERE ik.keyword_id in (827, 1070, 22412,23029,25287,133768,24593, 404)
WHERE ik.keyword_id in (32892)
AND ik3.keyword_id in (22859, 23883)
AND ik2.keyword_id not in (3748,8094,11092,25891,46110,8094,29700,98975,107639,46531,90881,107640,115069,24806,25886,6474,1946)
;


SELECT i.site_name_id, i.imagename
FROM Images i
-- JOIN I
magesKeywords ik on i.image_id = ik.image_id
JOIN SegmentHelper_nov2025_SQL_only_last3K_hands sh on sh.image_id = i.image_id
;
-- AND sh.seg_image_id >= 100000000

USE Stock;
SELECT i.site_name_id, i.imagename
FROM Images i
JOIN ImagesKeywords ik on i.image_id = ik.image_id
JOIN ImagesKeywords ik2 on i.image_id = ik2.image_id
JOIN SegmentOct20 s on i.image_id = s.image_id
WHERE ik.keyword_id = 22269
AND ik2.keyword_id = 21610
LIMIT 10;


USE Stock;
SELECT COUNT(*)
FROM SegmentHelperObjectYOLO
;

UPDATE Images i
SET i.no_image = NULL
WHERE i.image_id = 96323620
;

SELECT *
FROM Images
WHERE image_id = 25983
;

SELECT *
FROM SegmentHelperMissing_nov2025 
Where image_id = 62775971


CREATE TABLE SegmentHelperMissing_dec2025 (
id INT NOT NULL AUTO_INCREMENT,
encoding_id INT NOT NULL,
image_id INT NOT NULL,
body_landmarks TINYINT(1) NULL,
body_landmarks_norm TINYINT(1) NULL,
face_landmarks TINYINT(1) NULL,
face_encodings TINYINT(1) NULL,
hand_landmarks TINYINT(1) NULL,
body_world_landmarks TINYINT(1) NULL,
PRIMARY KEY (id),
INDEX idx_encoding_id (encoding_id),
INDEX idx_image_id (image_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

SELECT i.site_name_id, i.imagename
FROM Images i
JOIN SegmentHelperMissing_nov2025 s on i.image_id = s.image_id
WHERE s.body_world_landmarks is NULL
OR s.body_landmarks_norm is NULL
OR hand_landmarks is NULL
;

DELETE FROM BsonFileLog;

INSERT INTO SegmentHelperMissing_dec2025 (encoding_id, image_id)
SELECT encoding_id, image_id
FROM SegmentHelperMissing_nov2025
WHERE body_world_landmarks is NULL
OR body_landmarks_norm is NULL
OR hand_landmarks is NULL
;

'''
select from encodings, join to SegmentHelperMissing_nov2025 on image_id
select image_id
where mongo_body_landmarks =1
and normalized landmarks is null
'''
USE Stock;
-- SELECT e.image_id, e.encoding_id
INSERT INTO SegmentHelper_dec2025_missing_norm_body (image_id)
SELECT s.image_id 
FROM Encodings e
JOIN SegmentHelperMissing_nov2025 s on e.image_id = s.image_id
WHERE e.mongo_body_landmarks = 1
AND (e.mongo_body_landmarks_norm is NULL)
;

SELECT *
FROM Encodings 
WHERE image_id = 154
;

-- what is in wandering images?

SELECT wi.site_name_id, COUNT(wi.wandering_image_id)
FROM WanderingImages wi
GROUP BY (wi.site_name_id)
ORDER BY (wi.site_name_id)
;


SELECT *
FROM WanderingImages wi
WHERE wi.site_name_id = 2
LIMIT 100
;

SELECT *
FROM Images i
WHERE i.site_image_id = "10009813"
;

