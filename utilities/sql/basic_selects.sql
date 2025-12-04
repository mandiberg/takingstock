-- 86.3gb, 2.46tb





USE stock;
SET GLOBAL innodb_buffer_pool_size=8053063680;




-- 44374493

SELECT *
FROM Encodings
WHERE migrated_SQL = 1
AND migrated_Mongo is NULL
LIMIT 10
;

Select *
FROM Images
WHERE image_id = 53748578
;

SELECT *
FROM Encodings i 
WHERE i.image_id = 123694200
;

SELECT *
FROM Images
WHERE site_image_id = 1375949586
AND site_name_id = 10
;

UPDATE encodings e
SET e.migrated_SQL = 1, e.is_face = NULL
WHERE e.is_face = 1 
AND e.bbox is NULL
AND e.migrated = 1
;



SELECT i.site_image_id, ib.lum_torso, ib.lum_torso_bb, ib.selfie_bbox
FROM ImagesBackground ib
JOIN Images i ON i.image_id = ib.image_id
WHERE i.site_image_id IN ('1374307379', '1321070633')
;

SELECT i.site_name_id, i.imagename, i.description
FROM Images i 
JOIN encodings 	e ON i.image_id = e.image_id
WHERE e.migrated_SQL = 1
AND e.is_face = NULL
AND i.site_name_id = 3
LIMIT 1

;


SELECT *
FROM Images i
WHERE i.image_id = 65761606
;

SELECT *
FROM Encodings
WHERE image_id = 2284907
;


SELECT *
FROM Images i 
WHERE i.site_name_id = 10
AND i.site_image_id = 800181355
;





DELETE FROM BodyPoses3D;
DELETE FROM ImagesBodyPoses3D;

DELETE FROM ArmsPoses3D;
DELETE FROM ImagesArmsPoses3D;


USE stock;
SELECT cluster_id, COUNT(image_id) 
FROM ImagesBodyPoses3D
GROUP BY cluster_id
ORDER BY cluster_id
;


CREATE TABLE BodyPoses3D (
    cluster_id int NOT NULL PRIMARY KEY,
    cluster_median BLOB
);

-- This is the poses junction table.
CREATE TABLE ImagesBodyPoses3D (
    image_id INTEGER REFERENCES Images (image_id),
    cluster_id INTEGER REFERENCES BodyPoses3D (cluster_id),
    cluster_dist FLOAT DEFAULT NULL,
    PRIMARY KEY (image_id)
);


CREATE TABLE ArmsPoses3D (
    cluster_id int NOT NULL PRIMARY KEY,
    cluster_median BLOB
);

-- This is the poses junction table.
CREATE TABLE ImagesArmsPoses3D (
    image_id INTEGER REFERENCES Images (image_id),
    cluster_id INTEGER REFERENCES ArmsPoses3D (cluster_id),
    cluster_dist FLOAT DEFAULT NULL,
    PRIMARY KEY (image_id)
);

SELECT COUNT(image_id)
FROM ImagesBodyPoses3D
;

CREATE TABLE MetaBodyPoses3D (
    cluster_id int NOT NULL PRIMARY KEY,
    cluster_median BLOB
);

-- This is the poses junction table.
CREATE TABLE ClustersMetaBodyPoses3D (
    cluster_id INTEGER REFERENCES BodyPoses3D (cluster_id),
    meta_cluster_id INTEGER REFERENCES MetaBodyPoses3D (cluster_id),
    cluster_dist FLOAT DEFAULT NULL,
    PRIMARY KEY (cluster_id)
);


-- This is the poses junction table.
CREATE TABLE ClustersMetaHSV (
    cluster_id INTEGER REFERENCES HSV (cluster_id),
    meta_cluster_id INTEGER,
    cluster_dist FLOAT DEFAULT NULL,
    cluster_name varchar(40),
    PRIMARY KEY (cluster_id)
);


SELECT * FROM ArmsPoses3D ;
SELECT * FROM ImagesArmsPoses3D ;

DELETE FROM HSV ;
DELETE FROM ImagesHSV ;

Use Stock;
DELETE FROM MetaBodyPoses3D ;
DELETE FROM ClustersMetaBodyPoses3D  ;

SELECT cmbp.meta_cluster_id, COUNT(cmbp.cluster_id)
FROM ClustersMetaBodyPoses3D cmbp 
GROUP BY cmbp.meta_cluster_id
ORDER BY cmbp.meta_cluster_id
;

SELECT *
FROM ClustersMetaBodyPoses3D
WHERE ClustersMetaBodyPoses3D.cluster_id =235 

SELECT COUNT(s.image_id) 
FROM SegmentBig_isnotface s  JOIN Encodings e ON s.image_id = e.image_id  
WHERE  e.is_dupe_of IS NULL  
AND e.mongo_body_landmarks_3D = 1 and e.is_feet = 1
 ;

SELECT cmb.meta_cluster_id, COUNT(cmb.cluster_id)
FROM ClustersMetaBodyPoses3D cmb
GROUP BY cmb.meta_cluster_id
ORDER BY cmb.meta_cluster_id
;


SELECT cmb.cluster_id, COUNT(cmb.image_id)
FROM ImagesBodyPoses3D cmb
JOIN Encodings e ON cmb.image_id = e.image_id
WHERE e.is_feet = 0
GROUP BY cmb.cluster_id
ORDER BY cmb.cluster_id
;

-- delete rows from ImagesBodyPoses3D where the joined Encodings row matches criteria
-- MySQL syntax: DELETE <alias> FROM <table> <alias> JOIN ... WHERE ...
DELETE cmb
FROM ImagesBodyPoses3D cmb
JOIN Encodings e ON cmb.image_id = e.image_id
WHERE e.is_feet = 1 AND e.mongo_hand_landmarks = 1;


USE stock;
SELECT cmb.meta_cluster_id, cmb.cluster_id
FROM ClustersMetaBodyPoses3D cmb
WHERE cmb.meta_cluster_id = 181
;

SELECT cmb.meta_cluster_id, cmb.cluster_id
FROM ClustersMetaBodyPoses3D cmb
WHERE cmb.cluster_id = 27
;

SELECT ib.cluster_id, COUNT(ib.image_id)
FROM ImagesBodyPoses3D ib
JOIN ImagesKeywords ik ON ib.image_id = ik.image_id 
JOIN Encodings i ON ib.image_id = i.image_id
WHERE ib.cluster_id in (97, 253, 261, 313, 379, 401, 467, 475, 501, 528, 605)
AND ik.keyword_id = 22412
AND i.is_feet = 1
GROUP BY ib.cluster_id
ORDER BY ib.cluster_id
;

-- phone arms finder
SELECT image_id, cluster_id
FROM ImagesArmsPoses3D
WHERE image_id in (111180393,113985651,108214238,113889218,93164028,10887045,85248085,119098167,12955852)

SELECT *
FROM Encodings
WHERE image_id = 93164028
;

-- select count of ArmsPoses3D for heft keywords
SELECT  ik.cluster_id, COUNT(ik.image_id)
FROM SegmentHelper_sept2025_heft_keywords h
JOIN ImagesArmsPoses3D ik ON h.image_id = ik.image_id 
GROUP BY ik.cluster_id
ORDER BY ik.cluster_id


-- select count of HSV for heft keywords
SELECT  ik.cluster_id, COUNT(ik.image_id)
FROM SegmentHelper_sept2025_heft_keywords h
JOIN ImagesHSV ik ON h.image_id = ik.image_id 
GROUP BY ik.cluster_id
ORDER BY ik.cluster_id
;

-- select count of keywords for heft keywords
SELECT  ik.keyword_id, k.keyword_text, COUNT(ik.image_id)
FROM SegmentHelper_sept2025_heft_keywords h
JOIN ImagesKeywords ik ON h.image_id = ik.image_id 
JOIN Keywords k ON k.keyword_id = ik.keyword_id 
GROUP BY ik.keyword_id
ORDER BY ik.keyword_id
;


SELECT DISTINCT a.image_id
FROM ImagesKeywords a
JOIN ImagesKeywords b USING (image_id)
WHERE a.keyword_id = 184
  AND b.keyword_id = 7969
LIMIT 100
;


SELECT h.image_id
FROM SegmentHelper_oct2025_every40 h
JOIN ImagesKeywords ik ON h.image_id = ik.image_id 
WHERE ik.keyword_id IN (22411,22101,444,22191,16045,11549,133300,133777)
LIMIT 100
;

-- 193, 149

USE Stock;
-- select count of HSV for heft keywords
SELECT  i.site_name_id, COUNT(i.image_id)
FROM Images i
JOIN SegmentHelper_nov2025_SQL_only ik ON i.image_id = ik.image_id 
GROUP BY i.site_name_id
ORDER BY i.site_name_id
;


USE Stock;
SELECT COUNT(*)
FROM Images553 i5
WHERE i5.object_id = 1
;

SELECT *
FROM Images i
WHERE i.site_image_id = "1981048862"
AND i.site_name_id =2
;


SELECT * 
FROM Encodings e
WHERE e.image_id = 99136936
;

select version();







