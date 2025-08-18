-- 
-- DE-SHARDING 
-- 

	-- PT 0: MySQL dumping
        -- Images.no_image -- below in SQL near "export noimages", create, insert, then dump
            -- mysqldump -u root stock no_image_migration > no_image_migration_COMPNAME.sql
        -- Wandering Images -- just dump the entire table and upsert them.
            -- mysqldump -u root stock WanderingImages --where="site_name_id in (3,4)" > WanderingImages_COMPNAME.sql
            -- /Applications/MAMP/Library/bin/mysql80/bin/mysqldump --host=localhost -uroot -proot stock WanderingImages --where="site_name_id in (3,4)" > WanderingImages_COMPNAME.sql
        -- SegmentBig -- I don't have to do this directly, I can pull from Encodings. Will do on upsert

    -- PT 1: Export Mongo - /facemap/utilities/deshard_mysql.py
    -- PT 2: Upsert SQL
		-- update Encodings and create Encodings_Migration
		-- Move all NML data into migration table
    -- copy over everything from Encodings_Migration to Encodings where migrated_Mongo is None
    -- update everything that is not NULL? Or just update everything? 
    --  
    -- PT 3: Upsert Mongo - python

USE stock;
SET GLOBAL innodb_buffer_pool_size=8053063680;


-- export noimages
CREATE TABLE no_image_migration (
    migration_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    image_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES Images(image_id),
    migrated_SQL boolean
    )

INSERT INTO no_image_migration (image_id)
SELECT i.image_id
FROM Images i
WHERE i.no_image = 1
AND i.site_name_id in (3,4)
LIMIT 10
;



-- Encodings - needs an "migrated" boolean. Set it to 0 if is_nml indicates it needs to be migrated. Once migrted, set it to 1.
ALTER TABLE Encodings
ADD migrated_SQL boolean,
ADD migrated_Mongo boolean; 

-- Create migration table
CREATE TABLE Encodings_Migration (
    migration_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    encoding_id INTEGER,
    FOREIGN KEY (encoding_id) REFERENCES Encodings(encoding_id),
    image_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES Images(image_id),
    is_face boolean,
    is_body boolean,
    is_face_distant boolean,
    face_x DECIMAL (6,3),
    face_y DECIMAL (6,3),
    face_z DECIMAL (6,3),
    mouth_gap DECIMAL (6,3),
    face_landmarks BLOB,
    bbox JSON,
    face_encodings68 BLOB,
    body_landmarks BLOB,
    mongo_encodings boolean,
    mongo_body_landmarks boolean,
    mongo_face_landmarks boolean,
    is_small boolean,
    mongo_body_landmarks_norm boolean,
    two_noses boolean,
    is_dupe_of INTEGER,
    FOREIGN KEY (is_dupe_of) REFERENCES Images(image_id),
    mongo_hand_landmarks boolean,
    mongo_hand_landmarks_norm boolean,
    is_face_no_lms boolean,
    is_feet boolean,
    mongo_body_landmarks_3D boolean,
    is_hand_left boolean,
    is_hand_right boolean,
    migrated_SQL boolean,
    migrated_Mongo boolean,
    UNIQUE (image_id),
    UNIQUE (encoding_id)
); 


SELECT *
FROM NMLImages
WHERE nml_id = 19827792
;

-- does a merge on NMLImages table to update Encodings Depricated for now
-- OOPS have to undo this for the settings as below. 
UPDATE Encodings AS c
INNER JOIN (
    SELECT o.image_id
    FROM NMLImages AS o
    WHERE o.nml_id > 19827792
    AND o.is_nml_db = 0
    ORDER BY o.nml_id DESC
    LIMIT 100
) AS is_nml_db ON c.image_id = is_nml_db.image_id
SET c.migrated = 0
;

-- last non NML image: 19827793 for is_nml_db = 0 before fork
-- Move all NML data into migration table
INSERT INTO Encodings_Migration (image_id, encoding_id, is_face, is_body, is_face_distant, face_x, face_y, face_z, mouth_gap, face_landmarks, bbox, face_encodings68, body_landmarks, mongo_encodings, mongo_body_landmarks, mongo_face_landmarks, is_small, mongo_body_landmarks_norm, two_noses, is_dupe_of, mongo_hand_landmarks, mongo_hand_landmarks_norm, is_face_no_lms, is_feet, mongo_body_landmarks_3D, is_hand_left, is_hand_right)
SELECT DISTINCT  e.image_id, e.encoding_id, e.is_face, e.is_body, e.is_face_distant, e.face_x, e.face_y, e.face_z, e.mouth_gap, e.face_landmarks, e.bbox, e.face_encodings68, e.body_landmarks, e.mongo_encodings, e.mongo_body_landmarks, e.mongo_face_landmarks, e.is_small, e.mongo_body_landmarks_norm, e.two_noses, e.is_dupe_of, e.mongo_hand_landmarks, e.mongo_hand_landmarks_norm, e.is_face_no_lms, e.is_feet, e.mongo_body_landmarks_3D, e.is_hand_left, e.is_hand_right
FROM Encodings e 
JOIN NMLImages n on e.image_id = n.image_id 
LEFT JOIN Encodings_Migration em on e.image_id = em.image_id 
WHERE n.is_nml_db = 0

    AND em.image_id IS NULL    
LIMIT 10; -- Adjust the batch size as needed



-- just for resetting tests // Depricated for now - doing this in python
UPDATE Encodings_Migration em
SET em.migrated_Mongo = NULL
WHERE is_face = 1 OR is_body = 1 OR mongo_hand_landmarks = 1
;

