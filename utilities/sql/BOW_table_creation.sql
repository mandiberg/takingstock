USE stock;

-- create the three tables used by fetch_bagofkeywords.py

CREATE TABLE BagOfKeywords (
    image_id INT AUTO_INCREMENT PRIMARY KEY,
    age_id INT,
    gender_id INT,
    location_id INT,
--    site_name_id INT,
    description VARCHAR(150),
    keyword_list BLOB,
    tokenized_keyword_list BLOB,
    ethnicity_list BLOB,
    FOREIGN KEY (age_id) REFERENCES age(age_id),
    FOREIGN KEY (gender_id) REFERENCES gender(gender_id),
    FOREIGN KEY (location_id) REFERENCES location(location_id)
--     FOREIGN KEY (site_name_id) REFERENCES site(site_name_id)
);

CREATE TABLE Topics (
	topic_id INT,
	topic VARCHAR(250)
);

CREATE TABLE ImagesTopics (
	image_id INT,
    topic_id INT,
    topic_score FLOAT
);


SELECT MAX(sbi.image_id)
FROM SegmentBig_isface sbi 
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
    AND e.image_id >= 89000000
    AND e.face_x > -45
    AND e.face_x < -20
    AND e.face_y > -10
    AND e.face_y < 10
    AND e.face_z > -10
    AND e.face_z < 10
;




-- this is meta stuff


-- to delete topics (in this order)
DELETE FROM ImagesTopics;
DELETE FROM CountGender_Topics_so t ;
DELETE FROM CountEthnicity_Topics_so t ;
DELETE FROM Topics t ;

SELECT * FROM BagOfKeywords bok 
LIMIT 1000;

SELECT COUNT(bok.image_id) FROM BagOfKeywords bok ;
SELECT COUNT(bok.keyword_list) FROM BagOfKeywords bok ;

SELECT COUNT(it.image_id) FROM ImagesTopics it  ;

SELECT k.keyword_text  
FROM ImagesKeywords ik
JOIN Keywords k ON ik.keyword_id = k.keyword_id 
WHERE ik.image_id = 51148251



SELECT * 
FROM BagOfKeywords bok 
-- JOIN Images i ON ik.image_id = i.image_id 
WHERE bok.image_id = 51148251

-- count of images by topic
SELECT t.topic_id,
       COUNT(it.image_id) AS total_images,
       SUM(CASE WHEN i.gender_id = 8 THEN 1 ELSE 0 END) AS gender_8_count,
       (SUM(CASE WHEN i.gender_id = 8 THEN 1 ELSE 0 END) / COUNT(it.image_id)) * 100 AS gender_8_percentage,
       t.topic       
FROM Topics t
LEFT JOIN ImagesTopics it ON t.topic_id = it.topic_id
LEFT JOIN Images i ON it.image_id = i.image_id
GROUP BY t.topic_id, t.topic
ORDER BY total_images DESC;



-- select tests, can delete later

SELECT * 
FROM BagOfKeywords 
LIMIT 100 OFFSET 100;


-- tokenized column
ALTER TABLE BagOfKeywords
ADD COLUMN tokenized_keyword_list BLOB;

SELECT * 
FROM BagOfKeywords bok 
WHERE bok.image_id = 58154422;


-- adding new autoinc pkey for random index

use ministock1023;

-- Step 1: Drop the AUTO_INCREMENT attribute from the image_id column
ALTER TABLE BagOfKeywords MODIFY COLUMN image_id INT;

-- Step 2: Add a new column for the primary key
ALTER TABLE BagOfKeywords
ADD COLUMN bag_id INT NOT NULL DEFAULT 0;

-- Step 3: Populate the new column with unique values
SET @counter:= 0;
-- UPDATE BagOfKeywords SET bag_id=bag_id + @counter + 1;
UPDATE BagOfKeywords SET bag_id = @counter := @counter + 1;

-- 3.5 -- doesn't work, but also maybe not necessary...
ALTER TABLE BagOfKeywords
DROP PRIMARY KEY,
ADD PRIMARY KEY (bag_id);

-- Step 4: Create an index on the existing bag_id column
CREATE INDEX idx_bag_id ON BagOfKeywords (bag_id);




-- count of keywords overall
USE ministock;
SELECT k.keyword_id, k.keyword_text, COUNT(*) as keyword_count
FROM (SELECT DISTINCT image_id, keyword_id FROM imageskeywords) AS ik
JOIN keywords AS k ON ik.keyword_id = k.keyword_id
GROUP BY k.keyword_id, k.keyword_text
ORDER BY keyword_count DESC;

using the base above, write an SQL query that will return:
count of keyword_id 
the corresponding keywords.keyword_text
for all rows in imageskeywords

SELECT k.keyword_id, k.keyword_text, COUNT(ik.keyword_id) AS keyword_count
FROM imageskeywords AS ik
JOIN keywords AS k ON ik.keyword_id = k.keyword_id
GROUP BY k.keyword_id, k.keyword_text
ORDER BY keyword_count DESC;



