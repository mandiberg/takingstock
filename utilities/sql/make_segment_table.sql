
USE ministock1023;

-- cleanup
-- DROP TABLE SegmentAug30Straightahead ;
DELETE FROM SegmentFeb15;

-- create segment table
CREATE TABLE SegmentFeb15Temp (
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
    face_encodings BLOB,
    face_encodings68 BLOB,
    body_encodings68 BLOB    

);


-- insert data into segment table with join to ensure not selecting dupes
-- I think this is the most current and up to date??
SET GLOBAL innodb_buffer_pool_size=4294967296;

INSERT INTO SegmentFeb15Temp (image_id, site_name_id, site_image_id, contentUrl, imagename, description, face_x, face_y, face_z, mouth_gap, face_landmarks, bbox, face_encodings68,body_encodings68)
SELECT DISTINCT i.image_id, i.site_name_id, i.site_image_id, i.contentUrl, i.imagename, i.description, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings68, body_encodings68
FROM Images i
LEFT JOIN Encodings e ON i.image_id = e.image_id
LEFT JOIN SegmentFeb15Temp j ON i.image_id = j.image_id
WHERE e.face_encodings68 IS NOT NULL 
	AND e.face_x > -40 AND e.face_x < -24 
    AND e.face_y > -2 AND e.face_y < 2
    AND e.face_z > -2 AND e.face_z < 2
    AND j.image_id IS NULL
    AND i.site_name_id = 4
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

-- get count of data for segment table JUST Encodings
-- SELECT DISTINCT i.image_id, i.site_name_id, i.site_image_id, i.contentUrl, i.imagename, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings68
SELECT COUNT(e.image_id)
FROM Encodings e
-- WHERE e.face_encodings68 IS NOT NULL
WHERE e.is_face IS TRUE AND e.face_encodings68 IS NOT NULL 
AND e.face_x > -40 AND e.face_x < -24 
AND e.face_y > -4 AND e.face_y < 4 
AND e.face_z > -3 AND e.face_z < 3 
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


