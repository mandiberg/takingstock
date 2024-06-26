
USE stock;

-- cleanup
-- DROP TABLE SegmentAug30Straightahead ;
DELETE FROM SegmentHelperMar23_headon;

-- create helper segment table
CREATE TABLE SegmentHelperMay24_allfingers (
    seg_image_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    image_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES Images(image_id)
);


-- create segment table
CREATE TABLE SegmentBig_isface (
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


-- insert into new temp segment
INSERT INTO SegmentMar21  (image_id)
SELECT DISTINCT e.image_id
FROM Encodings e
WHERE e.face_encodings68 IS NOT NULL 
	AND e.face_x > -40 AND e.face_x < -24
    AND e.face_y > -2 AND e.face_y < 2
    AND e.face_z > -2 AND e.face_z < 2
; -- Adjust the batch size as needed


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



-- debugging makevideo query
SELECT *
FROM SegmentAug30Straightahead AS seg JOIN ImagesClusters AS ic ON seg.image_id = ic.image_id
WHERE seg.age_id NOT IN (1, 2, 3, 4) AND seg.mouth_gap > 15 AND ic.cluster_id = 77
LIMIT 1000;

SELECT * FROM SegmentAug30Straightahead AS seg JOIN ImagesClusters AS ic ON seg.image_id = ic.image_id 
WHERE seg.age_id NOT IN (1,2,3,4) and seg.mouth_gap > 15 AND ImagesClusters.cluster_id = 77 LIMIT 100;

SET GLOBAL innodb_buffer_pool_size=4294967296;
SHOW VARIABLES LIKE '%innodb_buffer_pool_size%';

SELECT COUNT(s.image_id) as ccount
FROM SegmentDec20 s
;

USE ministock;
SELECT COUNT(it.image_id) as ccount
FROM ImagesTopics it
;

USE ministock;
SELECT COUNT(i.image_id) as ccount
FROM Images i
;


SELECT COUNT(s.image_id) as ccount
FROM SegmentAug30Straightahead s
WHERE s.mouth_gap > 15
;

SELECT COUNT(e.image_id) FROM Encodings e 
WHERE e.is_face IS TRUE AND e.face_encodings68 IS NULL
;


SELECT COUNT(s.image_id)
FROM SegmentAug30Straightahead s
WHERE mouth_gap > 15
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


