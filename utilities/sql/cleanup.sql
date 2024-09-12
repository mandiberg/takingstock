

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




-- SET IS DUPE OF

UPDATE Encodings 
SET is_dupe_of = 16530032
WHERE image_id IN (


16628973,
16630539,
16646078,
15659235,
16642820,
16642664,
16583137,
16649930,
16639226,
16647319,
16635533,
16609103,
15672528,
16638684,
15614372,
16625901,
16641942,
15671237,
15562556,
16614191,
15978840,
16618924,
16501167,
15975659,
16642327,
16499328,
16572751,
16636102,
16648575,
16657410,
15579014



);

