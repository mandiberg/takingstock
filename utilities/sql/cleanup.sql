

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



UPDATE Encodings
SET    mongo_hand_landmarks_norm = NULL
WHERE  image_id < 20000
AND mongo_hand_landmarks = 1
;


ALTER TABLE HandsGestures128 RENAME TO HandsGestures;
ALTER TABLE ImagesHandsGestures128 RENAME TO ImagesHandsGestures;

ALTER TABLE CountAge_Location 
    missing INT DEFAULT 0
    ;

-- then create

-- SET IS DUPE OF

UPDATE Encodings e SET e.is_dupe_of = 1 JOIN Images i ON i.image_id = e.image_id WHERE i.site_name_id =

AND i.site_image_id

UPDATE Encodings 
SET is_dupe_of = 1
WHERE image_id IN (
5468774,
105453287,
109130286,
101761963,
36249935,
92518570,
101165235,
45959782,
101557700,
15832307,
97164287,
102603247,
83728936,
36303026
);




SELECT COUNT(*) 
FROM Encodings 
WHERE is_dupe_of IN (36384061, 4934107, 127134921, 83517166, 14879779, 62041012, 33089599, 37863650, 7849564, 36335990, 14981836, 42217662, 125086145, 27585264);


-- >>> 66

SELECT DISTINCT is_dupe_of
FROM Encodings
ORDER BY is_dupe_of;



-- I am trying to figure out the intersection of migrated_SQL and migrated_Mongo

SELECT COUNT(*)
FROM Encodings
WHERE migrated_SQL = 1
;
-- returns 22089785

SELECT COUNT(*)
FROM Encodings
WHERE migrated_Mongo = 1
;
-- returns 9676336

SELECT COUNT(*)
FROM Encodings
WHERE migrated_SQL = 1
AND migrated_Mongo = 1
;
-- returns 9465922

-- this implies that 9676336 - 9465922 = 210414 migrated to Mongo but not SQL
-- and 22089785 - 9465922 = 12623863 migrated to SQL but not Mongo
-- But I want to verify this, but both of these queries return 0


SELECT *
FROM Encodings
WHERE migrated_SQL = 1
AND migrated_Mongo is NULL
LIMIT 100
;


SELECT *
FROM Encodings
WHERE migrated_SQL is NULL
AND migrated_Mongo = 1
LIMIT 100
;

-- so I am confused. What is going on here?

USE Stock;
CREATE TABLE compare_sql_mongo_results_ultradone (
    encoding_id INT NOT NULL,
    image_id INT NOT NULL,
    site_name_id INT NOT NULL,
    face_landmarks BOOL,
    body_landmarks BOOL,
    face_encodings68 BOOL,
    nlms BOOL,
    left_hand BOOL,
    right_hand BOOL,
    body_world_landmarks BOOL,
    is_face BOOL,
    is_body BOOL,
    PRIMARY KEY (encoding_id, image_id)
);

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

SELECT COUNT(e.face_landmarks), COUNT(e.body_landmarks), COUNT(e.face_encodings68), COUNT(e.nlms), COUNT(e.body_world_landmarks)
FROM compare_sql_mongo_results e
WHERE e.encoding_id < 46000000
AND (e.face_landmarks = 0
OR e.body_landmarks = 0
OR e.face_encodings68 = 0
OR e.nlms = 0
OR e.body_world_landmarks = 0)
;

-- AND e.encoding_id < 57300000

SELECT i.site_name_id, COUNT(e.face_landmarks), COUNT(e.body_landmarks), COUNT(e.face_encodings68), COUNT(e.nlms), COUNT(e.right_hand), COUNT(e.left_hand), COUNT(e.body_world_landmarks)
FROM compare_sql_mongo_results e
JOIN Images i ON e.image_id = i.image_id
WHERE e.encoding_id > 0
AND (e.face_landmarks = 0
OR e.body_landmarks = 0
OR e.face_encodings68 = 0
OR e.right_hand = 0
OR e.left_hand = 0
OR e.nlms = 0
OR e.body_world_landmarks = 0)
GROUP BY i.site_name_id
ORDER BY i.site_name_id DESC;

;

SELECT * FROM encodings WHERE encoding_id = 1419291;
SELECT * FROM compare_sql_mongo_results WHERE encoding_id = 1419291;
UPDATE encodings SET mongo_body_landmarks_norm=NULL WHERE image_id = 28458;


SELECT *
FROM Images i
WHERE i.image_id = 52475
;


SELECT *
FROM compare_sql_mongo_results_ultradone
WHERE encoding_id > 14000
;

SELECT encoding_id FROM Encodings WHERE image_id = 5107843;

UPDATE compare_sql_mongo_results c JOIN Encodings e ON c.image_id = e.image_id SET c.body_landmarks=NULL WHERE e.image_id = 5107843;



WHERE (face_landmarks IS NOT NULL OR body_landmarks IS NOT NULL OR face_encodings68 IS NOT NULL OR nlms IS NOT NULL OR left_hand IS NOT NULL OR right_hand IS NOT NULL OR body_world_landmarks IS NOT NULL) 
AND encoding_id >= 50001 LIMIT 10000


INSERT INTO compare_sql_mongo_results_isbody1_isface1 (encoding_id, image_id, is_face, is_body)
SELECT encoding_id, image_id, is_face, is_body
FROM Encodings
WHERE is_face =1 AND is_body =1
AND encoding_id > 0
AND encoding_id < 1000
ON DUPLICATE KEY UPDATE
    is_face = VALUES(is_face),
    is_body = VALUES(is_body)
;

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


INSERT INTO SegmentHelper_oct2025_needs_validation (image_id)
SELECT image_id 
FROM compare_sql_mongo_results2
WHERE (face_landmarks IS NOT NULL
OR body_landmarks IS NOT NULL
OR face_encodings68 IS NOT NULL
OR nlms IS NOT NULL
OR left_hand IS NOT NULL
OR right_hand IS NOT NULL
OR body_world_landmarks IS NOT NULL)
AND encoding_id < 10000000

;


SELECT COUNT(image_id) 
FROM compare_sql_mongo_results_ultradone
WHERE (face_landmarks IS  NULL
AND body_landmarks IS  NULL
AND face_encodings68 IS  NULL
AND nlms IS  NULL
AND (left_hand IS NOT NULL
OR right_hand IS NOT NULL)
AND body_world_landmarks IS NOT NULL)
;

SELECT COUNT(image_id) 
FROM compare_sql_mongo_results_ultradone
WHERE (left_hand IS NOT NULL
OR right_hand IS NOT NULL)
AND is_body = 1
;

SELECT COUNT(encoding_id) 
FROM Encodings
WHERE (is_hand_left =1
OR is_hand_right =1)
AND is_body = 1

;

SELECT COUNT(encoding_id) 
FROM Encodings
WHERE (mongo_hand_landmarks = 1)
AND is_body = 1

;


SELECT *
FROM Encodings
WHERE image_id = 125059647
;


SELECT COUNT(image_id) 
FROM compare_sql_mongo_results_ultradone
;

SELECT i.site_name_id, COUNT(c.image_id) as ccount
FROM compare_sql_mongo_results_ultradone c
JOIN Images i on i.image_id = c.image_id
WHERE body_world_landmarks IS NOT NULL
AND is_body = 1
GROUP BY (i.site_name_id)
ORDER BY i.site_name_id
;

SELECT COUNT(*)
FROM Encodings e 
WHERE e.is_body = 1
AND e.mongo_body_landmarks_3D = 0
AND e.mongo_body_landmarks = 1
LIMIT 10
;


SELECT c2.encoding_id, c2.image_id, c2.body_world_landmarks, c3.body_world_landmarks
FROM compare_sql_mongo_results2 c2
JOIN compare_sql_mongo_results3 c3 ON c2.encoding_id = c3.encoding_id
WHERE c2.encoding_id > 58375000
AND c2.encoding_id < 58400000
AND c3.body_world_landmarks = 0
AND c2.body_world_landmarks IS NULL
LIMIT 10
;

SELECT *
FROM compare_sql_mongo_results_ultradone
WHERE encoding_id = 9599567
;

SELECT COUNT(body_landmarks)
AND c2.body_world_landmarks IS NULL
WHERE cu.body_world_landmarks IS FALSE


SELECT *
FROM compare_sql_mongo_results_ultradone cu
WHERE cu.nlms = 0
AND cu.face_landmarks IS NULL
AND cu.body_landmarks IS NULL
AND cu.is_body = 1
LIMIT 10
;

    SUM(CASE WHEN face_landmarks = 0 THEN 1 ELSE 0 END) AS face_landmarks_false,
    SUM(CASE WHEN body_landmarks = 0 THEN 1 ELSE 0 END) AS body_landmarks_false,
    SUM(CASE WHEN face_encodings68 = 0 THEN 1 ELSE 0 END) AS face_encodings68_false,
    SUM(CASE WHEN nlms = 0 THEN 1 ELSE 0 END) AS nlms_false,

-- do this after finishing SSD4Green ingest
UPDATE compare_sql_mongo_results_ultradone cu
JOIN compare_sql_mongo_results2 c2 ON c2.encoding_id = cu.encoding_id
SET cu.right_hand = NULL
WHERE cu.right_hand IS FALSE
  AND c2.right_hand IS NULL
;


SELECT COUNT(*)
FROM BsonFileLog
;


Use Stock;
SELECT * FROM
Encodings
WHERE image_id = 78947390
;

DELETE FROM
compare_sql_mongo_results_ultradone
WHERE encoding_id > 1050000
; 

DELETE FROM BsonFileLog
WHERE completed_bson_file like '%encoding%'
;



SELECT c2.encoding_id, c2.body_world_landmarks, c3.body_world_landmarks
FROM compare_sql_mongo_results2 c2
JOIN compare_sql_mongo_results4 c3 ON c2.encoding_id = c3.encoding_id
WHERE c2.body_world_landmarks != c3.body_world_landmarks
AND c2.encoding_id > 58500000
LIMIT 1
;



SELECT *
FROM Encodings e
JOIN compare_sql_mongo_results c
ON e.encoding_id = c.encoding_id
WHERE c.body_world_landmarks =0 
AND c.is_face = 0
AND c.is_body = 1
AND e.two_noses IS NULL
AND e.is_small IS NULL
LIMIT 100
;

-- I need to delete from compareresults where two noses and is small are 1



SELECT DISTINCT(s.image_id), s.site_name_id, s.contentUrl, ibg.hue, ibg.hue_bb, ibg.sat, ibg.sat_bb, ibg.val, ibg.val_bb, ibg.lum_torso, ibg.lum_torso_bb  
FROM SegmentBig_isface s  JOIN Encodings e ON s.image_id = e.image_id  JOIN ImagesArmsPoses3D ihp ON s.image_id = ihp.image_id  
JOIN ImagesKeywords it ON s.image_id = it.image_id  JOIN SegmentHelper_sept2025_heft_keywords sh ON s.image_id = sh.image_id  
JOIN ImagesBackground ibg ON s.image_id = ibg.image_id  JOIN Images553 ik_obj ON s.image_id = ik_obj.image_id    
JOIN ImagesHSV ihsv ON s.image_id = ihsv.image_id JOIN ClustersMetaHSV cmhsv ON ihsv.cluster_id = cmhsv.cluster_id  WHERE  e.is_dupe_of IS NULL  AND s.age_id NOT IN (1,2,3)  
AND k_obj.object_id = 1  AND k_obj.orientation_id = 0   
AND cmhsv.meta_cluster_id  IN (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22)    
AND ihp.cluster_id = 57 AND it.keyword_id  IN (4553, 23594, 1291, 7691, 17874, 7704, 4552, 7711, 7689, 4558, 4557, 21470, 26212, 4555, 4565, 4556, 4564, 4567, 7687, 4550, 4566, 28819, 4549, 21914, 4551, 13803, 12241, 4833, 4562, 4554, 15205, 4561, 4559, 11631, 31100, 7699, 7706, 12444, 7690, 7708, 112600, 13802, 4580, 7701, 7713, 16219, 16220, 32398, 127922, 7694, 13044, 17420, 32068, 4560, 7702, 7715, 13524, 40248, 52627, 116407, 4569, 7707, 7709, 11753, 16215, 93310, 101066, 104375, 109437, 125676, 7688, 17417, 55135, 96980, 98888, 106578, 7700, 7710, 7714, 15973, 39758, 41729, 5520, 5521, 7696, 7698, 7705, 7712, 7716, 7717, 16214, 16223, 18685, 19286, 19382, 60065, 77414, 78766, 85303, 86515, 90928, 96426, 105503, 105504, 105628, 108894, 127864, 553)  LIMIT 2000;]

