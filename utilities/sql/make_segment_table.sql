
USE stock;

SET GLOBAL innodb_buffer_pool_size = 8053063680;

-- cleanup
DROP TABLE SegmentHelper_dec27_getty_noface ;
DELETE FROM SegmentHelper_sept2025_heft_keywords;

-- create helper segment table
CREATE TABLE SegmentHelper_oct2025_every40 (
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

-- for making a helper from segmentbig
INSERT INTO SegmentHelper_sept2025_heft_keywords (image_id)
SELECT DISTINCT e.image_id
FROM SegmentBig_isface e
JOIN ImagesKeywords ik 
ON ik.image_id = e.image_id
WHERE ik.keyword_id IN (22411,220,22269,827,1070,22412,553,807,1644,5310)
    AND NOT EXISTS (
        SELECT 1
        FROM SegmentHelper_sept2025_heft_keywords s
        WHERE s.image_id = e.image_id
    )
LIMIT 2000000
;

SELECT COUNT(*)
FROM SegmentHelper_sept2025_heft_keywords
where image_id > 55732013
;

-- for making a complete site_name_id segment helper
INSERT INTO SegmentHelper_jan30_ALLgetty4faces (image_id)
SELECT DISTINCT i.image_id
FROM Images i 
WHERE i.site_name_id = 1
    AND NOT EXISTS (
        SELECT 1
        FROM SegmentHelper_jan30_ALLgetty4faces s
        WHERE s.image_id = i.image_id
    )
;


CREATE TABLE Encodings_Site2 AS
SELECT e.encoding_id, e.image_id, e.is_face, e.is_body, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.bbox
FROM Encodings e
JOIN Images i ON i.image_id = e.image_id
WHERE i.site_name_id = 2
;


ALTER TABLE Encodings_Site2
ADD face_x DECIMAL ,
ADD face_y DECIMAL ,
ADD face_z DECIMAL ,
ADD mouth_gap DECIMAL, 
ADD bbox JSON
;


CREATE INDEX idx_is_face ON Encodings_Site2 (is_face);


SELECT DISTINCT image_id
FROM Encodings_Site2
WHERE is_face = 0
LIMIT 10
;


SELECT *
FROM SegmentBig_isnotface
WHERE site_name_id = 2
LIMIT 100
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



-- CLEANUP TO ADD MISSING DATA FROM IMAGES TABLE
-- run this tomorrow

UPDATE SegmentBig_isface sb
JOIN Images i ON sb.image_id = i.image_id
SET 
    sb.site_name_id = i.site_name_id,
    sb.site_image_id = i.site_image_id,
    sb.contentUrl = i.contentUrl,
    sb.imagename = i.imagename,
    sb.description = i.description,
    sb.age_id = i.age_id,
    sb.age_detail_id = i.age_detail_id,
    sb.gender_id = i.gender_id,
    sb.location_id = i.location_id
WHERE sb.imagename IS NULL 
	AND sb.image_id >= 103748034;




SELECT MAX(seg_image_id)
FROM SegmentBig_isface
;

SELECT *
FROM SegmentBig_isface
WHERE  image_id = 107737723
;

SELECT MAX(seg_image_id)
FROM SegmentBig_isface
;

-- to test the last row inserted
SELECT * FROM SegmentBig_isface so ORDER BY so.seg_image_id DESC LIMIT 1



SELECT 
    COUNT(*) AS total_rows, -- Total number of rows in the table
    COUNT(image_id) AS image_id_count,
    COUNT(site_name_id) AS site_name_id_count,
    COUNT(site_image_id) AS site_image_id_count,
    COUNT(contentUrl) AS contentUrl_count,
    COUNT(imagename) AS imagename_count,
    COUNT(description) AS description_count,
    COUNT(age_id) AS age_id_count,
    COUNT(age_detail_id) AS age_detail_id_count,
    COUNT(gender_id) AS gender_id_count,
    COUNT(location_id) AS location_id_count,
    COUNT(face_x) AS face_x_count,
    COUNT(face_y) AS face_y_count,
    COUNT(face_z) AS face_z_count,
    COUNT(mouth_gap) AS mouth_gap_count,
    COUNT(face_landmarks) AS face_landmarks_count,
    COUNT(bbox) AS bbox_count,
    COUNT(face_encodings68) AS face_encodings68_count,
    COUNT(body_landmarks) AS body_landmarks_count,
    COUNT(keyword_list) AS keyword_list_count,
    COUNT(tokenized_keyword_list) AS tokenized_keyword_list_count,
    COUNT(ethnicity_list) AS ethnicity_list_count,
    COUNT(mongo_tokens) AS mongo_tokens_count
FROM SegmentBig_isface;


-- rows missing descriptions that have images.desc
SELECT 
    COUNT(*) AS total_rows_missing_desc -- Total number of rows in the table
FROM SegmentBig_isface sbi
LEFT JOIN Images i on i.image_id = sbi.image_id
WHERE i.description IS  NULL
AND sbi.description IS NULL
;


-- count the above insert, before doing it
SELECT count(DISTINCT e.encoding_id) 
FROM Encodings e
WHERE e.mongo_encodings = 1 
	AND e.face_x > -40 AND e.face_x < -3
    AND e.face_y > -4 AND e.face_y < 4
    AND e.face_z > -4 AND e.face_z < 4
; -- 

-- count the above UPDATE, before doing it
SELECT count(DISTINCT i.image_id) 
FROM Images i
LEFT JOIN Encodings e ON i.image_id = e.image_id
LEFT JOIN SegmentHelper_Aug25_6and12 sh ON sh.image_id = i.image_id 
LEFT JOIN SegmentOct20 j ON i.image_id = j.image_id
WHERE  j.location_id IS NULL
    AND i.location_id IS NOT NULL
    AND i.age_id > 3
; -- 


SELECT MAX(so.seg_image_id)
FROM SegmentOct20 so 
;

-- modifying to only do image_id etc, and from e
INSERT INTO SegmentMar21  (image_id)
SELECT DISTINCT e.image_id
FROM Encodings e
WHERE e.face_encodings68 IS NOT NULL 
	AND e.face_x > -33 AND e.face_x < -27
    AND e.face_y > -2 AND e.face_y < 2
    AND e.face_z > -2 AND e.face_z < 2
LIMIT 50000; -- Adjust the batch size as needed


-- modifying to only do image_id etc, and from e
INSERT INTO SegmentMar20  (image_id, face_x, face_y, face_z, mouth_gap, face_landmarks, bbox, face_encodings68,body_landmarks)
SELECT DISTINCT e.image_id, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings68, e.body_landmarks
FROM Encodings e
LEFT JOIN SegmentMar20 j ON e.image_id = j.image_id
WHERE e.face_encodings68 IS NOT NULL 
	AND e.face_x > -33 AND e.face_x < -27
    AND e.face_y > -5 AND e.face_y < 5
    AND e.face_z > -5 AND e.face_z < 5
    AND j.image_id IS NULL
LIMIT 2000; -- Adjust the batch size as needed







-- create temporary ImagesKeywordsMini table. this will be renamed after export/import
-- This is the junction table.
CREATE TABLE ImagesKeywordsMini (
    image_id int REFERENCES Images (image_id),
    keyword_id int REFERENCES Keywords (keyword_id),
    PRIMARY KEY (image_id, keyword_id)
);

-- NEED TO DO THIS WITH IMAGES AS WELL!
-- Insert rows into ImagesKeywordsMini, excluding existing entries
INSERT IGNORE INTO ImagesKeywordsMini (image_id, keyword_id)
SELECT ik.image_id, ik.keyword_id
FROM ImagesKeywords AS ik
WHERE ik.image_id IN (SELECT st.image_id FROM SegmentDec20 AS st);

LIMIT 10;


SELECT COUNT(ikm.image_id) FROM ImagesKeywordsMini ikm;




-- get count of data for segment table
-- SELECT DISTINCT i.image_id, i.site_name_id, i.site_image_id, i.contentUrl, i.imagename, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings68
SELECT COUNT(i.image_id)
FROM Images i
LEFT JOIN Encodings e ON i.image_id = e.image_id
-- WHERE e.face_encodings68 IS NOT NULL
WHERE e.is_face IS TRUE AND e.face_encodings68 IS NOT NULL AND e.bbox IS NOT NULL AND i.site_name_id = 8
AND i.age_id NOT IN (1,2,3,4)
AND e.face_x > -40 AND e.face_x < -25 
-- AND e.face_x > -40 AND e.face_x < -24 
AND e.face_y > -4 AND e.face_y < 4 
AND e.face_z > -3 AND e.face_z < 3 
;

USE Stock;
-- get count of data for segment table JUST Encodings
-- SELECT DISTINCT i.image_id, i.site_name_id, i.site_image_id, i.contentUrl, i.imagename, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings68
SELECT e.image_id
FROM Encodings e
-- WHERE e.face_encodings68 IS NOT NULL
WHERE e.is_face IS TRUE AND e.face_encodings68 IS NOT NULL 
AND e.face_x > -40 AND e.face_x < -24 
AND e.face_y > -4 AND e.face_y < 4 
AND e.face_z > -3 AND e.face_z < 3 
LIMIT 100
;






-- MINISTOCK STUFF

USE ministock1023;

SELECT COUNT(image_id)
FROM Images ;

DELETE FROM ImagesKeywords 
WHERE image_id NOT IN (SELECT image_id FROM SegmentDec20)
LIMIT 1000; -- Adjust the limit as needed for testing

DELETE FROM Images
WHERE image_id NOT IN (SELECT image_id FROM SegmentDec20)
AND image_id IS NOT NULL
LIMIT 1000000;

DELETE FROM Clusters ;



UPDATE Encodings 
SET is_dupe_of = NULL
WHERE is_dupe_of IS NOT NULL;


