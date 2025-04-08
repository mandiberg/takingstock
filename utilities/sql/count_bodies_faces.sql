-- 86.3gb, 2.46tb





USE stock;
SET GLOBAL innodb_buffer_pool_size=8053063680;

-- this will need to be run after all images are reprocessed

SELECT COUNT(*)
FROM Encodings e
WHERE e.is_face IS TRUE
OR e.is_body IS TRUE
OR e.is_face_distant 
OR e.is_face_no_lms 
;


SELECT COUNT(*)
FROM Encodings e
WHERE e.is_face IS TRUE
;
