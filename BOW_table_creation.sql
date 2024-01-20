USE ministock;

-- create the three tables used by fetch_bagofkeywords.py

CREATE TABLE BagOfKeywords (
    image_id INT AUTO_INCREMENT PRIMARY KEY,
    age_id INT,
    gender_id INT,
    location_id INT,
    description VARCHAR(150),
    keyword_list BLOB,
    ethnicity_list BLOB,
    FOREIGN KEY (age_id) REFERENCES age(age_id),
    FOREIGN KEY (gender_id) REFERENCES gender(gender_id),
    FOREIGN KEY (location_id) REFERENCES location(location_id)
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




-- this is meta stuff


DELETE FROM BagOfKeywords;
DELETE FROM Topics t ;
DELETE FROM ImagesTopics;

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



