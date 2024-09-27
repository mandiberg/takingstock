

USE stock;



       

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


SELECT COUNT(*)
FROM SegmentBig_isface
WHERE image_id > 88000000
AND mongo_tokens = 1
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



-- SET IS DUPE OF

UPDATE SegmentOct20 
SET is_dupe_of = 12682917
WHERE image_id IN (
12681455,10426433,12679698,10421256,12680957,12677985,10416342,12680034,12681718,12679245,10430244,12679949,12680701,10443790,12681306,10413156,12679203,12682432,10433634,12672305,10417976,12678000,12681565,12678600,10422845,12681452,12679748,10446180,10425676,12679191,10433451,12672468,10426701,12681159,10436956,10440146,12677834,12680811,12679649,12678525,10418322,12684060,12681988,12681134,10443370,10430293,10402922,10434239,12680401,12681820,10423095,12683281,12682722,12683687,10421165,10430202,12680812,12684227,12680947,10394893,12679469,10425296,12678832,12682097,12680865,10438415,12680227,12680046,12684369,10406288,10424060,12683028,10442218,12682997,12678006,12678076,12680054,12680132,12680053,10433461
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


SELECT COUNT(*) 
FROM Encodings 
WHERE is_dupe_of IN (36384061, 4934107, 127134921, 83517166, 14879779, 62041012, 33089599, 37863650, 7849564, 36335990, 14981836, 42217662, 125086145, 27585264);


-- >>> 66

SELECT DISTINCT is_dupe_of
FROM Encodings
ORDER BY is_dupe_of;

