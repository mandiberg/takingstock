

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

-- Identify duplicate image_id entries in the Encodings table
SELECT site_image_id, COUNT(*)
FROM Images i 
WHERE i.site_name_id = 1
GROUP BY site_image_id
HAVING COUNT(*) > 1;




-- fix bad imagename paths (maybe not working anymore?)
-- get count
       SELECT COUNT(image_id)
        FROM Images
        WHERE site_name_id = 1
        AND imagename LIKE '/Users/michael.mandiberg/Desktop/getty_33333/images_getty_33333_last4/%'
; 


-- update
        UPDATE Images
    SET imagename = REPLACE(imagename, '/Volumes/SSD4/images_getty_reDL/', '')        
        WHERE site_name_id = 1
        AND imagename LIKE '/Volumes/SSD4/images_getty_reDL/%'
        LIMIT 1000000; 



        UPDATE SegmentOct20 so
        JOIN Images i ON i.image_id = so.image_id
     SET so.imagename = REPLACE(so.imagename, '/Users/michael.mandiberg/Desktop/getty_33333/images_getty_33333_last4/', '')        
        WHERE i.site_name_id = 1
        AND so.imagename LIKE '/Users/michael.mandiberg/Desktop/getty_33333/images_getty_33333_last4/%'
        LIMIT 200000;






SELECT MAX(i.image_id) 
FROM Images i 
;
-- 121943655 stock may 31

-- 126887888 pnd5

SELECT *
FROM Images i
WHERE i.image_id  = 126887888
;


SELECT MAX(image_id) 
FROM Images i 
;


SELECT *
FROM Encodings e 
WHERE e.image_id = 108079574
;
-- 97102179
-- max 108079574
