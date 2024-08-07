-- 86.3gb, 2.46tb



USE stock
;


SELECT *
FROM ImagesBackground ib 
WHERE ib.image_id = 9634

SELECT COUNT(ib.image_id)
FROM ImagesBackground ib 
WHERE ib.is_left_shoulder = 1
;


UPDATE ImagesBackground
SET    is_left_shoulder = NULL
WHERE  is_left_shoulder IS NOT NULL;

UPDATE ImagesBackground
SET    is_right_shoulder  = NULL
WHERE  is_right_shoulder IS NOT NULL;


SELECT COUNT(ib.image_id)
FROM ImagesBackground ib 
WHERE JSON_EXTRACT(ib.selfie_bbox , '$.left') = 0
OR JSON_EXTRACT(ib.selfie_bbox , '$.right') = 0
;



SELECT JSON_EXTRACT(sales_data, '$.amount') AS total_amount 
FROM sales 
WHERE JSON_EXTRACT(sales_data, '$.product') = 'Widget X';




SELECT COUNT(i.image_id) 
FROM Images i 
WHERE i.site_name_id = 11

;


SELECT COUNT(so.seg_image_id) as ccount, COUNT(so.mongo_face_landmarks) as fcount, COUNT(so.mongo_body_landmarks) as bcount
FROM SegmentOct20 so 
JOIN ImagesBackground ib on ib.image_id = so.image_id 
;


SELECT *
FROM SegmentOct20 so 
LEFT JOIN ImagesBackground ON so.image_id = ImagesBackground.image_id
WHERE ImagesBackground.image_id IS NULL
LIMIT 10;

SELECT COUNT(ib.image_id)
FROM ImagesBackground ib
WHERE ib.selfie_bbox IS NOT NULL
;


SELECT *
FROM Encodings e 
WHERE e.image_id = 9923170
;

SELECT COUNT(so.image_id) 
FROM SegmentOct20 so 
WHERE so.site_name_id = 1
;


-- ENCODINGS STUFFFFFFFFFF

SELECT COUNT(i.image_id) as ccount
FROM Images i  
JOIN Encodings e ON i.image_id = e.image_id 
WHERE e.mongo_face_landmarks = 1
AND i.site_name_id = 15
;

SELECT COUNT(i.image_id) as ccount
FROM Images i  
JOIN Encodings e ON i.image_id = e.image_id 
WHERE e.face_landmarks is not NULL 
AND i.site_name_id = 13
;


SELECT *
FROM Encodings e 
WHERE e.image_id  = 122582509
AND e.face_landmarks IS NOT NULL
;

SELECT COUNT(so.seg_image_id) as ccount
FROM SegmentOct20 so 
JOIN Encodings e ON so.image_id = e.image_id 
WHERE e.mongo_body_landmarks = 1
;


SELECT count(seg1.image_id) 
FROM SegmentOct20 seg1 
JOIN ImagesTopics it ON seg1.image_id = it.image_id 
WHERE seg1.mongo_body_landmarks IS NULL 
AND it.topic_id = 5

;



SELECT *
FROM ImagesKeywords ik 
JOIN Images i ON ik.image_id = i.image_id 
WHERE ik.keyword_id = 1762
AND i.site_name_id = 13
;

SELECT *
FROM Keywords k 
WHERE k.keyword_id = 1762
;


SELECT so.mongo_body_landmarks, so.mongo_face_landmarks 
FROM SegmentOct20 so 
WHERE so.image_id = 2894566
;


SELECT *
FROM Images i
WHERE i.site_image_id  = 10133878
AND i.site_name_id = 5
;

SELECT COUNT(i.image_id) as ccount
FROM Images	i
LEFT JOIN Encodings e on e.image_id = i.image_id 
WHERE e.encoding_id IS NULL
AND i.site_name_id = 2
;

-- 1 199143


-- 126887888 at 6pm, may 15 ministock

SELECT ie.ethnicity_id 
FROM ImagesEthnicity ie 
JOIN Images i ON i.image_id = ie.image_id 
WHERE i.site_image_id  = 1204627074
LIMIT 1000
;

SHOW VARIABLES LIKE 'tmpdir';


SELECT DISTINCT ImagesBackground.image_id
FROM ImagesBackground
LEFT JOIN SegmentHelperApril1_topic7 ON ImagesBackground.image_id = SegmentHelperApril1_topic7.image_id
WHERE SegmentHelperApril1_topic7.image_id IS NOT NULL
  AND ImagesBackground.lum_torso IS NULL
LIMIT 10;

SELECT COUNT(i.image_id)
FROM Images	i
JOIN Encodings e on e.image_id = i.image_id 
WHERE face_x > -40 AND face_x < -24 AND face_y > -5 AND face_y < 5 AND face_z > -5 AND face_z < 5 
AND e.is_face IS TRUE 
AND i.age_id NOT IN (1,2,3)   
;

SELECT COUNT(e.image_id) 
FROM Encodings e  
JOIN SegmentHelperMay7_fingerpoint so ON e.image_id = so.image_id 
WHERE e.body_landmarks is  NULL  
-- AND so.image_id NOT IN (1,2,3)
-- 268348
;


SELECT s.image_id, s.description, it.topic_score 
FROM SegmentOct20 s  JOIN ImagesTopics it ON s.image_id = it.image_id  
-- WHERE s.body_landmarks IS NOT NULL 
WHERE face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2 AND it.topic_score > .3 AND s.age_id NOT IN (1,2,3)   
AND it.topic_id IN (17)
;

SELECT *
FROM Images i
WHERE i.site_image_id  = 5778
AND i.site_name_id = 5
;

SELECT *
FROM Encodings e 
WHERE e.image_id = 423851
;

SELECT MAX(sbi.seg_image_id) 
FROM SegmentBig_isface sbi 
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


SELECT *
FROM Images i 
WHERE i.image_id in (9924753, 9924032)
;

SELECT COUNT(i.image_id) as ccount, i.site_name_id 
FROM Images i 
LEFT JOIN Encodings e on i.image_id = e.image_id 
-- WHERE i.site_name_id = 2
WHERE e.encoding_id is NULL
AND i.image_id > 100000000
GROUP BY i.site_name_id 
;


-- missing encodings
SELECT i.image_id, i.site_name_id, i.imagename 
FROM Images i 
LEFT JOIN Encodings e on i.image_id = e.image_id 
WHERE e.encoding_id is NULL
-- AND i.site_name_id not in (1)
LIMIT 10
;



SELECT DISTINCT(s.image_id), s.age_id, s.site_name_id, s.contentUrl, s.imagename, s.face_x, s.face_y, s.face_z, s.mouth_gap, s.face_landmarks, s.bbox, s.face_encodings68, s.site_image_id, s.body_landmarks 
FROM SegmentOct20 s  JOIN ImagesTopics it ON s.image_id = it.image_id  
WHERE s.site_name_id != 1 AND face_encodings68 IS NOT NULL 
AND face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2 
-- AND s.age_id > 4  
AND it.topic_score > .3  AND it.topic_id = 10  
LIMIT 10;




-- SELECT DISTINCT seg1.image_id, seg1.site_name_id, seg1.contentUrl, seg1.imagename, e.encoding_id, seg1.site_image_id, e.face_landmarks, e.bbox 
SELECT COUNT(seg1.image_id)
FROM SegmentOct20 seg1 LEFT JOIN Encodings e ON seg1.image_id = e.image_id 
WHERE e.body_landmarks IS NULL AND e.image_id 
IN (SELECT seg1.image_id FROM SegmentOct20 seg1 WHERE face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2 AND seg1.site_name_id !=1) 
;
82976
LIMIT 1000;



-- AND i.imagename LIKE '/Volumes%' 

-- fixing getty image paths

SELECT COUNT(e.image_id) 
FROM Encodings e
INNER JOIN SegmentOct20 so on so.image_id = e.image_id 
WHERE e.body_landmarks IS NOT NULL
;

SELECT COUNT(i.image_id)
FROM Images i 
WHERE i.site_name_id = 1
AND i.imagename LIKE 'images/%'
;


SELECT MAX(i.image_id)
FROM Images i 
WHERE i.site_name_id = 1
-- AND i.imagename LIKE '/Volumes%' 
-- AND i.imagename LIKE 'images%'
;



USE stocktest;

CREATE TABLE Model_Release (
    release_name_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    release_name varchar(20)
); 


SELECT COUNT(ik.keyword_id)
FROM Images i
JOIN ImagesKeywords ik on ik.image_id = i.image_id 
WHERE i.site_name_id = 10
;

SELECT COUNT(i.image_id)
FROM Images i
WHERE i.site_name_id = 10
;

DELETE FROM Images
WHERE site_name_id = 10
;

SELECT k.keyword_text 
FROM ImagesKeywords ik 
JOIN Images i ON i.image_id = ik.image_id 
JOIN Keywords k ON ik.keyword_id = k.keyword_id
WHERE i.site_image_id = 10393345
;

SELECT COUNT(ik.image_id)  
FROM ImagesKeywords ik 
JOIN Images i ON i.image_id = ik.image_id 
JOIN Keywords k ON ik.keyword_id = k.keyword_id
WHERE k.keyword_text = "gorgeous"
AND i.site_name_id = 9
;

