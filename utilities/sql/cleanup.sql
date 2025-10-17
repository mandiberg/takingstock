

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


-- this is very slow
UPDATE Encodings
SET is_dupe_of = CASE image_id
    WHEN 15945000 THEN 36384061
    WHEN 4949837 THEN 4934107
    WHEN 108146196 THEN 127134921
    WHEN 127134900 THEN 127134921
    WHEN 127126682 THEN 127134921
    WHEN 5871139 THEN 36335990
    WHEN 52084410 THEN 14981836
    WHEN 2852639 THEN 42217662
    WHEN 123884246 THEN 125086145
    WHEN 8810268 THEN 125086145
    WHEN 8567955 THEN 27585264
    ELSE is_dupe_of
END;


UPDATE SegmentOct20
SET is_dupe_of = CASE image_id
	WHEN 101483920 THEN 126099212
	WHEN 40569705 THEN 126099212
	WHEN 13680241 THEN 108063663
	WHEN 54229819 THEN 108063663
	WHEN 105592535 THEN 41687881
	WHEN 15434142 THEN 108035357
	WHEN 60333992 THEN 5284501
	WHEN 4612094 THEN 45825628
	WHEN 4581876 THEN 101831056
	ELSE is_dupe_of
END;

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


CREATE TABLE compare_sql_mongo_results (
    encoding_id INT NOT NULL,
    image_id INT NOT NULL,
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


SELECT i.site_name_id, COUNT(e.face_landmarks), COUNT(e.body_landmarks), COUNT(e.face_encodings68), COUNT(e.nlms), COUNT(e.right_hand), COUNT(e.left_hand), COUNT(e.body_world_landmarks)
FROM compare_sql_mongo_results e
JOIN Images i ON e.image_id = i.image_id
WHERE e.encoding_id > 0
AND e.encoding_id < 57300000
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
FROM compare_sql_mongo_results
;


-- select count of each BOOL column in compare_sql_mongo_results that is FALSE  
SELECT 
    SUM(CASE WHEN face_landmarks = 0 THEN 1 ELSE 0 END) AS face_landmarks_false,
    SUM(CASE WHEN body_landmarks = 0 THEN 1 ELSE 0 END) AS body_landmarks_false,
    SUM(CASE WHEN face_encodings68 = 0 THEN 1 ELSE 0 END) AS face_encodings68_false,
    SUM(CASE WHEN nlms = 0 THEN 1 ELSE 0 END) AS nlms_false,
    SUM(CASE WHEN left_hand = 0 THEN 1 ELSE 0 END) AS left_hand_false,
    SUM(CASE WHEN right_hand = 0 THEN 1 ELSE 0 END) AS right_hand_false,
    SUM(CASE WHEN body_world_landmarks = 0 THEN 1 ELSE 0 END) AS body_world_landmarks_false,
    SUM(CASE WHEN is_face = 0 THEN 1 ELSE 0 END) AS is_face_false,
    SUM(CASE WHEN is_body = 0 THEN 1 ELSE 0 END) AS is_body_false
FROM compare_sql_mongo_results
WHERE is_face = 1
AND is_body = 0
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


