USE stock;

SET GLOBAL innodb_buffer_pool_size=8053063680;

-- create the three tables used by fetch_bagofkeywords.py



CREATE TABLE Topics (
	topic_id INT,
	topic VARCHAR(250)
);

CREATE TABLE ImagesTopics (
	image_id INT,
    topic_id INT,
    topic_score FLOAT,
    topic_id2 INT,
    topic_score2 FLOAT,
    topic_id3 INT,
    topic_score3 FLOAT
);







-- this is meta stuff


-- to delete topics (in this order)
DELETE FROM ImagesTopics;
DELETE FROM CountGender_Topics_so t ;
DELETE FROM CountEthnicity_Topics_so t ;
DELETE FROM Topics t ;


DELETE FROM ImagesPoses ;
DELETE FROM Poses ;



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


-- count of images by topic SEGMENT ONLY
SELECT t.topic_id,
       COUNT(it.image_id) AS total_images,
       SUM(CASE WHEN i.gender_id = 8 THEN 1 ELSE 0 END) AS gender_8_count,
       (SUM(CASE WHEN i.gender_id = 8 THEN 1 ELSE 0 END) / COUNT(it.image_id)) * 100 AS gender_8_percentage,
       t.topic       
FROM Topics t
LEFT JOIN ImagesTopics it ON t.topic_id = it.topic_id
INNER JOIN SegmentOct20 i ON it.image_id = i.image_id
GROUP BY t.topic_id, t.topic
ORDER BY total_images DESC;




-- count of keywords and key text, by topic
SELECT k.keyword_id, k.keyword_text, COUNT(ik.keyword_id) AS keyword_count
FROM imageskeywords AS ik
JOIN keywords AS k ON ik.keyword_id = k.keyword_id
JOIN imagestopics it ON it.image_id = ik.image_id
WHERE it.topic_id = 37
AND k.keyword_text LIKE 'franc%'
GROUP BY k.keyword_id, k.keyword_text
ORDER BY keyword_count DESC;




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



