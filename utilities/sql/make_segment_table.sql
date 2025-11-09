
USE stock;

SET GLOBAL innodb_buffer_pool_size = 8053063680;

-- cleanup
DROP TABLE SegmentHelper_dec27_getty_noface ;
DELETE FROM SegmentHelper_sept2025_heft_keywords;

-- create helper segment table
CREATE TABLE SegmentHelper_nov2025_placard (
    seg_image_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    image_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES Images(image_id)
);


INSERT INTO SegmentHelper_oct2025_every40 (image_id)
SELECT image_id 
FROM Images
WHERE image_id % 40 = 0
ORDER BY image_id;


-- create segment table
CREATE TABLE SegmentBig_ALLgetty4faces (
    seg_image_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    image_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES Images(image_id),
    site_name_id INTEGER,
    FOREIGN KEY (site_name_id) REFERENCES Site (site_name_id),
    site_image_id varchar(50),
    contentUrl varchar(300),
    imagename varchar(200),
    description varchar(150),    
    age_id INTEGER,
	FOREIGN KEY (age_id) REFERENCES Age (age_id),
    age_detail_id INTEGER,
	FOREIGN KEY (age_detail_id) REFERENCES AgeDetail (age_detail_id),
    gender_id INTEGER,
    FOREIGN KEY (gender_id) REFERENCES Gender (gender_id),
    location_id INTEGER,
    FOREIGN KEY (location_id) REFERENCES Location (location_id),
    face_x DECIMAL (6,3),
    face_y DECIMAL (6,3),
    face_z DECIMAL (6,3),
    mouth_gap DECIMAL (6,3),
    face_landmarks BLOB,
    bbox JSON,
    face_encodings68 BLOB,
    body_landmarks BLOB,  
    keyword_list BLOB,
    tokenized_keyword_list BLOB,
    ethnicity_list BLOB,
    mongo_tokens boolean,
    mongo_body_landmarks boolean,
    mongo_face_landmarks boolean,
    mongo_body_landmarks_norm boolean,
    no_image boolean,
    is_dupe_of INTEGER,
    FOREIGN KEY (is_dupe_of) REFERENCES Images(image_id),
    mongo_hand_landmarks boolean,    
    mongo_hand_landmarks_norm boolean,
    UNIQUE (image_id)


);


SELECT MAX(seg_image_id) 
FROM SegmentOct20 so 
;

-- 3938723


-- segment the segment table based on TOPICS and add to helpersegment
INSERT INTO SegmentHelperMay22_anger (image_id)
SELECT DISTINCT so.image_id
FROM SegmentOct20 so 
LEFT JOIN SegmentHelperMay22_anger sh ON so.image_id = sh.image_id
LEFT JOIN ImagesTopics it ON it.image_id = so.image_id 
WHERE 
	it.topic_id  IN (22)
    AND sh.image_id IS NULL
LIMIT 400000; -- Adjust the batch size as needed

-- count how many will be in this topics helper table
SELECT count( seg1.image_id) 
FROM SegmentOct20 seg1 
LEFT JOIN ImagesTopics it ON it.image_id = seg1.image_id 
WHERE 
	it.topic_id  IN (15,17)
;

SELECT COUNT(seg1.image_id)
FROM SegmentOct20 seg1 LEFT JOIN Encodings e ON seg1.image_id = e.image_id 
INNER JOIN SegmentHelperMay7_fingerpoint ht ON seg1.image_id = ht.image_id 
LEFT JOIN ImagesTopics it ON seg1.image_id = it.image_id 
WHERE e.body_landmarks IS NULL  
;

-- segment the segment table and add to helpersegment
INSERT INTO SegmentHelperApril12_2x2x33x27 (image_id)
SELECT DISTINCT so.image_id
FROM SegmentOct20 so 
LEFT JOIN SegmentHelperMar23_headon sh ON so.image_id = sh.image_id
WHERE 
    so.face_x > -33 AND so.face_x < -27 
    AND so.face_y > -2 AND so.face_y < 2
    AND so.face_z > -2 AND so.face_z < 2
    AND sh.image_id IS NULL
LIMIT 100000; -- Adjust the batch size as needed



-- create helper from imageskeywords
INSERT INTO SegmentHelperMay24_allfingers (image_id)
SELECT DISTINCT i.image_id
FROM Images i 
LEFT JOIN ImagesKeywords ik ON ik.image_id = i.image_id 
LEFT JOIN SegmentHelperMay24_allfingers j ON i.image_id = j.image_id

WHERE 
	ik.keyword_id = 908
	-- ik.keyword_id IN (908, 1510, 6255, 8716, 9457, 10162, 11300, 11355, 12411, 16165, 16294, 19751, 22379, 25320, 25591, 30721, 38577, 39258, 39711)
    AND j.image_id IS NULL    

	LIMIT 1000;

SELECT MAX(seg_image_id) 
FROM SegmentHelperMay24_allfingers
;

-- count how many will be in this keywords helper table
-- SELECT count(DISTINCT i.image_id) 
SELECT DISTINCT i.image_id
FROM Images i 
LEFT JOIN ImagesKeywords ik ON ik.image_id = i.image_id 
LEFT JOIN SegmentHelperMay24_allfingers j ON i.image_id = j.image_id

WHERE 
	ik.keyword_id = 908
	-- ik.keyword_id IN (908, 1510, 6255, 8716, 9457, 10162, 11300, 11355, 12411, 16165, 16294, 19751, 22379, 25320, 25591, 30721, 38577, 39258, 39711)
    AND j.image_id IS NULL    

	LIMIT 100;

SELECT *
FROM Keywords k 
WHERE k.keyword_text LIKE "%number%"
;


-- insert data into segment table with join to ensure not selecting dupes
-- I think this is the most current and up to date??
-- uses helper table
SET GLOBAL innodb_buffer_pool_size=8294967296;

INSERT INTO SegmentOct20 (image_id, site_name_id, site_image_id, contentUrl, imagename, description, face_x, face_y, face_z, mouth_gap, face_landmarks, bbox, face_encodings68,body_landmarks)
SELECT DISTINCT i.image_id, i.site_name_id, i.site_image_id, i.contentUrl, i.imagename, i.description, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings68, e.body_landmarks 
FROM Images i
LEFT JOIN Encodings e ON i.image_id = e.image_id
LEFT JOIN SegmentOct20 j ON i.image_id = j.image_id
LEFT JOIN ImagesKeywords ik ON ik.image_id = i.image_id 
WHERE e.face_encodings68 IS NOT NULL 
	AND e.face_x > -40 AND e.face_x < -24 
    AND e.face_y > -10 AND e.face_y < 10
    AND e.face_z > -10 AND e.face_z < 10
    AND e.is_face IS TRUE
    AND j.image_id IS NULL    
    AND ik.keyword_id = 908    
    AND i.age_id  > 3
LIMIT 100; -- Adjust the batch size as needed


-- modifying to only do image_id etc, and from e
INSERT INTO SegmentMar20  (image_id)
SELECT DISTINCT e.image_id
FROM Encodings e
WHERE e.face_encodings68 IS NOT NULL 
	AND e.face_x > -33 AND e.face_x < -27
    AND e.face_y > -2 AND e.face_y < 2
    AND e.face_z > -2 AND e.face_z < 2
LIMIT 100000; -- Adjust the batch size as needed


SELECT *
FROM keywords k
WHERE k.keyword_text LIKE "%quiet%"
;

-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
-- 
--    UPDATE SEGMENT, AUGUST 2024
-- 
--    1. create helper
--    2. flag helper rows where is_new
--    3. move is_new ids to segment
-- 
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

-- DO THIS FIRST
-- create a helpertable (use SQL above)
-- insert into new temp segment 
-- this skips existing so you can rerun ontop of existing data
-- seg big is -45 to -3, with yz at 10

INSERT INTO SegmentHelper_dec27_shutter_noface (image_id)
SELECT DISTINCT e.image_id
FROM Encodings e
JOIN Images i ON i.image_id = e.image_id
-- JOIN ImagesKeywords ik ON ik.image_id = i.image_id
JOIN ImagesTopics it ON it.image_id = e.image_id 
WHERE e.mongo_encodings = 1 
	AND e.mongo_body_landmarks IS NULL
	AND it.topic_id = 22
--	AND ik.keyword_id IN (9758, 76667, 83610, 100134, 102145, 22126, 99525, 115178, 116993, 125891)
--	AND i.age_id IS NULL
	AND e.face_x > -45 AND e.face_x < -3
    AND e.face_y > -10 AND e.face_y < 10
    AND e.face_z > -10 AND e.face_z < 10
    AND NOT EXISTS (
        SELECT 1
        FROM SegmentOct20 s
        WHERE s.image_id = e.image_id
    );

   
-- for making a helper from segmentbig
INSERT INTO SegmentHelper_june2025_nmlGPU300k (image_id)
SELECT DISTINCT e.image_id
FROM SegmentBig_isface e
JOIN Images i 
ON i.image_id = e.image_id
JOIN NMLImages n ON n.image_id = e.image_id
WHERE e.face_x > -45 AND e.face_x < -6
    AND e.face_y > -4 AND e.face_y < 4
    AND e.face_z > -4 AND e.face_z < 4
    AND NOT EXISTS (
        SELECT 1
        FROM SegmentHelper_june2025_nmlGPU300k s
        WHERE s.image_id = e.image_id
    )
    AND n.nml_id > 4191363
LIMIT 2000000
;

-- for making a helper from segmentbig based on keywords
INSERT INTO SegmentHelper_nov2025_placard (image_id)
SELECT DISTINCT e.image_id
FROM SegmentBig_isface e
JOIN ImagesKeywords ik 
ON ik.image_id = e.image_id
WHERE ik.keyword_id IN (23375,13130,21463,184,23726,4222,8874,8136,133749,26241,22814,133787,4587,133627)
    AND NOT EXISTS (
        SELECT 1
        FROM SegmentHelper_nov2025_placard s
        WHERE s.image_id = e.image_id
    )
LIMIT 400000
;



SELECT ik.keyword_id, COUNT(e.image_id)
FROM SegmentBig_isface e
JOIN ImagesKeywords ik 
ON ik.image_id = e.image_id
WHERE ik.keyword_id IN (23375,13130,21463,184,23726,4222,8874,8136,133749,26241,22814,133787,4587,133627)
    AND NOT EXISTS (
        SELECT 1
        FROM SegmentHelper_nov2025_placard s
        WHERE s.image_id = e.image_id
    )
GROUP BY ik.keyword_id
;




-- this just to build a helper with image_ids. from ibg
INSERT INTO SegmentHelper_oct3_bg_doover (image_id)
SELECT DISTINCT ib.image_id
FROM ImagesBackground ib
WHERE ib.hue IS NULL
;
  
   
-- DO THIS SECOND
-- add column for is_new 
ALTER TABLE SegmentHelper_jan30_ALLgetty4faces
ADD COLUMN is_new BOOL ;
-- and set is_new to True for new image_id
UPDATE SegmentHelper_jan30_ALLgetty4faces sh
LEFT JOIN SegmentBig_ALLgetty4faces s ON sh.image_id = s.image_id
SET sh.is_new = True
WHERE s.image_id IS NULL;

-- check how many added
SELECT COUNT(*)
FROM SegmentHelper_jan30_ALLgetty4faces sh
;

-- check how many added
SELECT COUNT(sh.is_new)
FROM SegmentHelper_jan30_ALLgetty4faces sh
WHERE sh.is_new = True
;

-- DO THIS THIRD
-- Add info from Images
-- INSERT INTO SegmentOct20 (image_id, site_name_id, site_image_id, contentUrl, imagename, age_id, age_detail_id, gender_id, location_id, face_x, face_y, face_z, mouth_gap, bbox, mongo_body_landmarks, mongo_face_landmarks)
-- SELECT DISTINCT i.image_id, i.site_name_id, i.site_image_id, i.contentUrl, i.imagename, i.age_id, i.age_detail_id, i.gender_id, i.location_id, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.bbox, e.mongo_body_landmarks, e.mongo_face_landmarks

INSERT INTO SegmentBig_ALLgetty4faces (image_id, site_name_id, site_image_id, contentUrl, imagename, age_id, age_detail_id, gender_id, location_id, face_x, face_y, face_z, mouth_gap, bbox)
SELECT DISTINCT i.image_id, i.site_name_id, i.site_image_id, i.contentUrl, i.imagename, i.age_id, i.age_detail_id, i.gender_id, i.location_id, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.bbox
FROM Images i
LEFT JOIN Encodings e ON i.image_id = e.image_id
LEFT JOIN SegmentHelper_jan30_ALLgetty4faces sh ON sh.image_id = i.image_id 
LEFT JOIN SegmentBig_isnotface j ON i.image_id = j.image_id
WHERE sh.is_new IS TRUE
    AND j.image_id IS NULL    
--	AND i.age_id IS NULL
LIMIT 1000; -- Adjust the batch size as needed

-- DO THIS EXTRA
-- Add Location_id from Images
UPDATE SegmentOct20 j
JOIN Images i ON j.image_id = i.image_id
JOIN SegmentHelper_Aug25_6and12 sh ON sh.image_id = i.image_id
SET j.location_id = i.location_id
WHERE  j.location_id IS NULL
    AND i.location_id IS NOT NULL
    AND i.age_id > 3
; -- Adjust the batch size as needed




-- DO THIS EXTRA PART TWO
-- Add images metadata

UPDATE SegmentBig_isface sb
JOIN Images i ON sb.image_id = i.image_id
SET
    sb.description = IF(sb.description IS NULL, i.description, sb.description),
    sb.site_name_id = IF(sb.site_name_id IS NULL, i.site_name_id, sb.site_name_id),
    sb.site_image_id = IF(sb.site_image_id IS NULL, i.site_image_id, sb.site_image_id),
    sb.contentUrl = IF(sb.contentUrl IS NULL, i.contentUrl, sb.contentUrl),
    sb.imagename = IF(sb.imagename IS NULL, i.imagename, sb.imagename),
    sb.age_id = IF(sb.age_id IS NULL, i.age_id, sb.age_id),
    sb.gender_id = IF(sb.gender_id IS NULL, i.gender_id, sb.gender_id),
    sb.location_id = IF(sb.location_id IS NULL, i.location_id, sb.location_id)
WHERE sb.image_id IN (
    SELECT image_id 
    FROM (
        SELECT sb.image_id
        FROM SegmentBig_isface sb
        JOIN Images i ON sb.image_id = i.image_id
        WHERE 
            (sb.description IS NULL 
            OR sb.site_name_id IS NULL
            OR sb.site_image_id IS NULL
            OR sb.contentUrl IS NULL
            OR sb.imagename IS NULL
            OR sb.age_id IS NULL
            OR sb.gender_id IS NULL
            OR sb.location_id IS NULL)
        LIMIT 1000  -- Adjust the batch size as needed
    ) AS tmp
);

