-- 86.3gb, 2.46tb



USE stock
;


DELETE 
FROM Poses
;
DELETE 
FROM ImagesPoses
;

SELECT COUNT(i.image_id) 
FROM Images i
JOIN Encodings e ON e.image_id = i.image_id 
-- WHERE i.site_image_id = "___638KPrmA"
WHERE i.site_name_id in (6,12)
AND e.is_face = 1
    AND e.face_x > -40 AND e.face_x < -3
    AND e.face_y > -4 AND e.face_y < 4
    AND e.face_z > -4 AND e.face_z < 4
   	AND NOT i.age_id <= 3)

;

SELECT COUNT(e.image_id)
FROM Encodings e 
LEFT JOIN 
    SegmentBig_isface sb 
ON 
    e.image_id = sb.image_id
WHERE 
    sb.image_id IS NULL
    AND e.image_id IS NOT NULL
    AND e.face_x > -45
    AND e.face_x < -20
    AND e.face_y > -10
    AND e.face_y < 10
    AND e.face_z > -10
    AND e.face_z < 10
;

SELECT MAX(sbi.seg_image_id)
FROM SegmentBig_isface sbi 
;

SELECT *
FROM SegmentBig_isface sbi 
WHERE sbi.image_id = 89000001
;



SELECT 
    e.encoding_id, 
    e.image_id, 
    e.bbox, 
    e.face_x, 
    e.face_y, 
    e.face_z, 
    e.mouth_gap, 
    e.face_landmarks, 
    e.face_encodings68, 
    e.body_landmarks
FROM 
    Encodings e
LEFT JOIN 
    SegmentBig_isface sb 
ON 
    e.image_id = sb.image_id
WHERE 
    sb.image_id IS NULL
    AND e.image_id IS NOT NULL
    AND e.face_x > -45
    AND e.face_x < -20
    AND e.face_y > -10
    AND e.face_y < 10
    AND e.face_z > -10
    AND e.face_z < 10
LIMIT 100;  -- Replace 100 with the desired LIMIT value



   
   
SELECT *
FROM PhoneBbox
WHERE image_id = 118060332
;

SELECT COUNT(image_id) AS count,
       it.topic_id,
       t.topic
FROM ImagesTopics it
JOIN Topics t ON it.topic_id = t.topic_id
GROUP BY it.topic_id, t.topic;


SELECT COUNT(image_id) AS count,
       ip.cluster_id
FROM ImagesPoses ip 
JOIN Poses p ON ip.cluster_id = p.cluster_id
GROUP BY ip.cluster_id;


SELECT k.keyword_id, k.keyword_number, k.keyword_text 
FROM ImagesKeywords ik 
JOIN Keywords k ON ik.keyword_id = k.keyword_id 
-- WHERE i.site_image_id = "___638KPrmA"
WHERE ik.image_id = 121233771
;

SELECT COUNT(*)  
FROM SegmentOct20 so
WHERE so.mongo_body_landmarks_norm = 1
;

SELECT *
FROM Encodings e 
WHERE e.image_id = 81533412
;

SELECT COUNT(s.image_id)
FROM SegmentOct20  s
WHERE s.mongo_body_landmarks = 1
AND s.face_x > -33 AND s.face_x < -27 AND s.face_y > -2 AND s.face_y < 2 AND s.face_z > -2 AND s.face_z < 2
;

CREATE TABLE SegmentHelperAug16_SegOct20_preAlamy (
    seg_image_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    image_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES Images(image_id)
);


SELECT 
    FLOOR(face_z) AS face_x_unit,
    COUNT(*) AS row_count
FROM 
    SegmentOct20
GROUP BY 
    FLOOR(face_z)
ORDER BY 
    face_x_unit;


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
WHERE e.encoding_id  = 114852545
;

SELECT COUNT(so.image_id) 
FROM SegmentOct20 so 
WHERE so.site_name_id = 1
;


-- ENCODINGS STUFFFFFFFFFF

SELECT COUNT(i.image_id) as ccount
FROM Images i  
JOIN Encodings e ON i.image_id = e.image_id 
WHERE  i.site_name_id = 13
;


DELETE Encodings
FROM Encodings 
JOIN Images ON images.image_id = encodings.image_id 
WHERE images.site_name_id = 13
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
WHERE i.site_name_id = 11
AND i.location_id IS NULL
;

SELECT *
FROM Images i 
WHERE i.site_image_id = 7696
AND i.site_name_id = 15
;

SELECT DISTINCT(i.author)
FROM Images i 
WHERE i.site_name_id = 15

;

USE Stock;


SELECT DISTINCT i.image_id
FROM Images i
LEFT JOIN SegmentOct20 so ON i.image_id = so.image_id
WHERE so.bbox IS NOT NULL
AND so.mongo_body_landmarks = 1
AND i.h IS NULL
LIMIT 1000;

SELECT COUNT(i.image_id)
FROM Images i
LEFT JOIN SegmentOct20 so ON i.image_id = so.image_id
WHERE so.bbox IS NOT NULL
AND so.mongo_body_landmarks = 1
AND i.h IS NOT NULL
;

SELECT COUNT(*)
FROM PhoneBbox pb 
WHERE pb.bbox_67_norm IS NOT NULL
;


    
SELECT COUNT(i.image_id)
FROM Images i
WHERE i.site_name_id = 6

;

DELETE FROM Images
WHERE site_name_id = 18
;


SELECT *
FROM Images i 
WHERE i.image_id = 2402477
;


SELECT COUNT(i.image_id)
FROM ImagesKeywords ik 
JOIN Images i ON i.image_id = ik.image_id 
JOIN Keywords k ON ik.keyword_id = k.keyword_id
JOIN ImagesEthnicity ie ON ie.image_id = i.image_id 
WHERE ik.keyword_id = 22310
AND i.site_name_id = 9
;


SELECT k.keyword_text
FROM ImagesKeywords ik 
JOIN Images i ON i.image_id = ik.image_id 
JOIN Keywords k ON ik.keyword_id = k.keyword_id
JOIN ImagesEthnicity ie ON ie.image_id = i.image_id 
WHERE i.image_id = 2400730
AND i.site_name_id = 9
;


SELECT i.image_id
FROM Images i 
 WHERE i.site_image_id = 1008421694
AND i.site_name_id = 10
;

SELECT ie.ethnicity_id 
FROM ImagesEthnicity ie 
WHERE ie.image_id = 1847892
;

SELECT COUNT(ik.image_id)  
FROM ImagesKeywords ik 
JOIN Images i ON i.image_id = ik.image_id 
JOIN Keywords k ON ik.keyword_id = k.keyword_id
WHERE k.keyword_text = "gorgeous"
AND i.site_name_id = 9
;

