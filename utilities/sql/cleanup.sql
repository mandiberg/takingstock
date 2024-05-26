

USE ministock1023;

SELECT e.image_id, i.imagename
FROM Encodings e 
JOIN Images i ON e.image_id = i.image_id
WHERE i.site_name_id = 9 and e.is_face IS NULL
;


SELECT COUNT(e.image_id) 
FROM Encodings e 
WHERE e.is_face IS NULL
;


-- Identify duplicate image_id entries in the Encodings table
SELECT site_image_id, COUNT(*)
FROM Images i 
WHERE i.site_name_id = 1
GROUP BY site_image_id
HAVING COUNT(*) > 1;


SELECT MAX(i.image_id) 
FROM Images i 
;

-- 126887888 pnd5

SELECT *
FROM Images i
WHERE i.image_id  = 126887888
;


SELECT MAX(e.encoding_id) 
FROM Encodings e 
;


SELECT *
FROM Encodings e 
WHERE e.image_id = 108079574
;
-- 97102179
-- max 108079574
