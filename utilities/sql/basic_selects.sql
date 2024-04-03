
USE stock;

ALTER TABLE ImagesBackground 
ADD val FLOAT,
ADD torso_lum FLOAT,
ADD val_bb FLOAT,
ADD torso_lum_bb FLOAT
;

SELECT MAX(so.seg_image_id)
FROM SegmentOct20 so ;

SELECT ie.ethnicity_id 
FROM ImagesEthnicity ie 
JOIN Images i ON i.image_id = ie.image_id 
WHERE i.site_image_id  = 1204627074
LIMIT 1000
;

SELECT *
FROM SegmentOct20 so 
WHERE so.image_id = 2082968
;

SELECT DISTINCT ImagesBackground.image_id
FROM ImagesBackground
LEFT JOIN SegmentHelperApril1_topic7 ON ImagesBackground.image_id = SegmentHelperApril1_topic7.image_id
WHERE SegmentHelperApril1_topic7.image_id IS NOT NULL
  AND ImagesBackground.lum_torso IS NULL
LIMIT 10;

SELECT COUNT(ib.image_id) 
FROM ImagesBackground ib 
WHERE ib.lum_torso IS NOT NULL 
;

SELECT *
FROM ImagesBackground ib 
WHERE ib.image_id = 2082968
;

ALTER TABLE ImagesBackground 
RENAME COLUMN torso_lum to lum_torso;


SELECT *
FROM Images i 
WHERE i.image_id > 100000000
AND i.site_name_id = 1
AND i.site_image_id = 499757170
;

DELETE 
FROM SegmentHelperApril1_topic7;

WHERE ibg.lum_bb = -2
LIMIT 1000;

SELECT *
FROM SegmentOct20 so 
WHERE so.image_id = 2819946
;

SELECT COUNT(it.image_id)
FROM ImagesTopics it  
;

SELECT *
FROM Images i 
WHERE i.site_image_id  = 15880596
AND i.site_name_id = 5;

AND i.image_id > 100172671;


SELECT COUNT(ik.keyword_id)
FROM ImagesKeywords ik 
JOIN Images i ON ik.image_id = i.image_id 
-- WHERE ik.image_id >= 122172671 
WHERE  i.site_name_id = 7 
AND ik.keyword_id = 33992
;



-- count of gender
SELECT COUNT(i.gender_id) as thiscount, i.gender_id 
FROM Images i   
WHERE i.image_id > 126887900
-- WHERE i.site_name_id = 7 
GROUP BY i.gender_id 
ORDER BY thiscount DESC;

SELECT COUNT(i.image_id) as thiscount 
FROM Images i   
WHERE i.site_name_id = 1
AND i.image_id > 131570175
;

3528798

SELECT distinct(i.image_id), i.description 
FROM Images i 
LEFT JOIN ImagesEthnicity ie ON i.image_id = ie.image_id 
WHERE i.site_name_id = 7 
AND ie.image_id IS NOT NULL
LIMIT 1000
;

SELECT ie.ethnicity_id 
FROM ImagesEthnicity ie 
WHERE ie.image_id = 126887995
;

SELECT k.keyword_text  
FROM ImagesKeywords ik
JOIN Keywords k ON ik.keyword_id = k.keyword_id 
WHERE ik.image_id = 114519467
;

SELECT COUNT(ik.keyword_id)
FROM ImagesKeywords ik 
WHERE ik.keyword_id = 12021;

USE Stock; 
SELECT COUNT(i.image_id)
FROM Images i 
-- RIGHT JOIN Encodings e ON i.image_id = e.image_id 
WHERE i.site_name_id =1
AND i.image_id > 100887900
;

-- 1165977 total getty
-- 1161196 with encoding
;

USE ministock1023;
DELETE
FROM Images i
WHERE i.image_id > 126887900;

Where i.site_name_id = 7 ;

-- 18739967 rows
-- 18752139
-- 17455620 keys


-- BG color testing
USE ministock1023;
DELETE 
FROM ImagesBG; 

SELECT SegmentOct20.image_id, SegmentOct20.imagename, SegmentOct20.site_name_id, SegmentOct20.face_x, SegmentOct20.face_y, SegmentOct20.face_z 
FROM SegmentOct20
LEFT JOIN ImagesBG ON SegmentOct20.image_id = ImagesBG.image_id
WHERE ImagesBG.image_id IS NULL
AND SegmentOct20.face_x >= -33
AND SegmentOct20.face_x <= -27
AND SegmentOct20.face_y >= -2
AND SegmentOct20.face_y <= 2
AND SegmentOct20.face_z >= -2
AND SegmentOct20.face_z <= 2
LIMIT 100;

SELECT *
FROM ImagesBG ib 
WHERE ib.image_id = 189001;

WHERE ib.sat is not NULL;


SELECT COUNT(image_id)
FROM SegmentMar21 
;

USE stock;
SELECT MAX(i.image_id) 
FROM Images i ;
114518973 - before pond5
ministock1023 - 126887888 for getty

INSERT INTO Keywords (keyword_number, keyword_text)
VALUES (11995,'arabic'); 

SELECT i.site_name_id, COUNT(bok.image_id) as ccount
FROM BagOfKeywords bok
LEFT JOIN Images i ON i.image_id = bok.image_id 
GROUP BY i.site_name_id 
ORDER BY i.site_name_id 
;

-- times out
SELECT i.site_name_id, COUNT(bok.image_id) as bok_ct, COUNT(bok.keyword_list) as keys_ct, COUNT(e.is_face) as face_ct
FROM BagOfKeywords bok
LEFT JOIN Images i ON i.image_id = bok.image_id
LEFT JOIN Encodings e ON e.image_id = bok.image_id 
GROUP BY i.site_name_id 
ORDER BY i.site_name_id 
;


SELECT site_name_id, COUNT(i.image_id) as ccount
FROM Images i 
GROUP BY i.site_name_id 
ORDER BY ccount
;


WHERE i.site_name_id IN (2,4)
;


SELECT MAX(i.image_id)
FROM Images i  ;
-- 114357130


USE ministock1023;

 
-- SELECT *
DELETE
FROM Images i
Where i.image_id > 126887888 ;

USE stock;
SELECT k.keyword_text  
FROM ImagesKeywords ik 
JOIN Keywords k ON ik.keyword_id = k.keyword_id 
WHERE ik.image_id = 3091;

SELECT e.ethnicity  
FROM ImagesEthnicity ie  
JOIN Ethnicity e  ON ie.ethnicity_id  = e.ethnicity_id  
WHERE ie.image_id = 118623872;




SELECT *
FROM Images i  
WHERE i.site_name_id = 1;

FROM BagOfKeywords bok  ;
-- 14891783

SELECT DISTINCT(s.image_id), s.age_id, s.site_name_id, s.contentUrl, s.imagename, s.face_x, s.face_y, s.face_z, s.mouth_gap, s.face_landmarks, s.bbox, s.face_encodings68, s.site_image_id, s.body_landmarks 
FROM SegmentOct20 s  JOIN ImagesTopics it ON s.image_id = it.image_id  
WHERE s.site_name_id != 1 AND face_encodings68 IS NOT NULL 
AND face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2 
-- AND s.age_id > 4  
AND it.topic_score > .3  AND it.topic_id = 10  
LIMIT 10;


SELECT COUNT(so.age_id)
FROM SegmentOct20 so 
WHERE so.age_id IS NOT NULL;

    SELECT COUNT(i.image_id)
    FROM Images i LEFT JOIN Encodings e ON i.image_id = e.image_id
    WHERE e.encoding_id IS NOT NULL AND i.site_name_id = 2 AND i.location_id = 8;
   
    SELECT COUNT(i.image_id)
    FROM Images i 
    WHERE i.site_name_id = 4;

    SELECT COUNT(i.image_id)
    
    SELECT *
    FROM Images i 
    WHERE i.site_name_id = 2 AND i.location_id = 8
   LIMIT 100;

USE stock;
SELECT MAX(e.encoding_id)
FROM Encodings e;


USE stock;
SELECT * FROM Encodings e 
WHERE e.encoding_id = 54056241;

WHERE e.body_landmarks IS NOT NULL;
= 56678099;

SELECT COUNT(*)
FROM Encodings e 
JOIN Images i ON e.image_id = i.image_id 
WHERE e.body_landmarks IS NOT NULL AND i.site_name_id =2;

SELECT COUNT(*)
FROM Encodings e 
JOIN SegmentOct20 so ON so.image_id = e.image_id
WHERE e.body_landmarks IS NOT NULL;


-- SELECT DISTINCT seg1.image_id, seg1.site_name_id, seg1.contentUrl, seg1.imagename, e.encoding_id, seg1.site_image_id, e.face_landmarks, e.bbox 
SELECT COUNT(seg1.image_id)
FROM SegmentOct20 seg1 LEFT JOIN Encodings e ON seg1.image_id = e.image_id 
WHERE e.body_landmarks IS NULL AND e.image_id 
IN (SELECT seg1.image_id FROM SegmentOct20 seg1 WHERE face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2 AND seg1.site_name_id !=1) 
;
82976
LIMIT 1000;


USE ministock1023
SELECT so.image_id
FROM SegmentOct20 so 
LIMIT 10;

SELECT image_id 
FROM SegmentOct20 
LIMIT 10;

WHERE so.site_image_id = 429400262;


INSERT INTO SegmentOct20 (image_id, site_name_id, site_image_id, contentUrl, imagename, face_x, face_y, face_z, mouth_gap, face_landmarks, bbox, face_encodings68)
SELECT DISTINCT i.image_id, i.site_name_id, i.site_image_id, i.contentUrl, i.imagename, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings68

USE stock;
SELECT COUNT(i.image_id)
FROM Images i
LEFT JOIN Encodings e ON i.image_id = e.image_id
LEFT JOIN SegmentOct20 j ON i.image_id = j.image_id
WHERE i.site_name_id = 2
	AND e.face_x > -40 AND e.face_x < -24 
    AND e.face_y > -4 AND e.face_y < 4 
    AND e.face_z > -3 AND e.face_z < 3
    AND j.image_id IS NOT NULL
    AND e.body_landmarks IS NULL
;
    
LIMIT 2000000; -- Adjust the batch size as needed


-- feb 24, delete me
USE Stock;
SELECT *
FROM BagOfKeywords bok  
WHERE bok.image_id = 16370;
