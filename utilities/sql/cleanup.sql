Use Stock;

SELECT e.image_id, i.imagename
FROM Encodings e 
JOIN Images i ON e.image_id = i.image_id
WHERE i.site_name_id = 9 and e.is_face IS NULL
;


SELECT COUNT(e.image_id) 
FROM Encodings e 
WHERE e.is_face IS NULL
;
