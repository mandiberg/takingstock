USE Stock;

-- this works, and is pretty quick
-- count by topic
SELECT COUNT(image_id) AS count,
       it.topic_id,
       t.topic
FROM ImagesTopics it
JOIN Topics t ON it.topic_id = t.topic_id
GROUP BY it.topic_id, t.topic;


-- count by keyword (basedon helper segment, doing the whole thing is too big)
SELECT COUNT(ik.image_id) AS count,
       ik.keyword_id,
       k.keyword_text
FROM ImagesKeywords ik
JOIN Keywords k ON ik.keyword_id = k.keyword_id
JOIN SegmentOct20 sh ON ik.image_id = sh.image_id
GROUP BY ik.keyword_id, k.keyword_text;




<<<<<<< Updated upstream
=======
-- take two
SELECT 
    COALESCE(topic_count, 0) AS topic_count,
    it.topic_id,
    t.topic,
    COALESCE(SUM(CASE WHEN so.gender_id = 1 THEN 1 ELSE 0 END), 0) AS men_count
FROM 
    (
        SELECT 
            COUNT(it.image_id) AS topic_count,
            it.topic_id
        FROM 
            ImagesTopics it
        JOIN 
            Topics t ON it.topic_id = t.topic_id
        GROUP BY 
            it.topic_id, t.topic
    ) AS topic_counts
LEFT JOIN 
    SegmentOct20 so ON so.image_id = topic_counts.image_id
GROUP BY 
    topic_counts.topic_id, topic_counts.topic
   ;


SELECT so.image_id, so.gender_id, it.topic_id 
FROM SegmentOct20 so
JOIN ImagesTopics it ON so.image_id = it.image_id  
LIMIT 10000
;

SELECT COUNT(so.location_id)  
FROM SegmentOct20 so 
;

SELECT 
    COALESCE(topic_count.count, 0) AS topic_count,
    topic_count.topic_id,
    topic_count.topic,
    COALESCE(SUM(CASE WHEN so.gender_id = 1 THEN 1 ELSE 0 END), 0) AS men_count,
    COALESCE(SUM(CASE WHEN so.gender_id = 2 THEN 1 ELSE 0 END), 0) AS none_count,
    COALESCE(SUM(CASE WHEN so.gender_id = 3 THEN 1 ELSE 0 END), 0) AS oldmen_count,
    COALESCE(SUM(CASE WHEN so.gender_id = 4 THEN 1 ELSE 0 END), 0) AS oldwomen_count,
    COALESCE(SUM(CASE WHEN so.gender_id = 5 THEN 1 ELSE 0 END), 0) AS nonbinary_count,
    COALESCE(SUM(CASE WHEN so.gender_id = 6 THEN 1 ELSE 0 END), 0) AS other_count,
    COALESCE(SUM(CASE WHEN so.gender_id = 7 THEN 1 ELSE 0 END), 0) AS trans_count,
    COALESCE(SUM(CASE WHEN so.gender_id = 8 THEN 1 ELSE 0 END), 0) AS women_count,
    COALESCE(SUM(CASE WHEN so.gender_id = 9 THEN 1 ELSE 0 END), 0) AS youngmen_count,
    COALESCE(SUM(CASE WHEN so.gender_id = 10 THEN 1 ELSE 0 END), 0) AS youngwomen_count,
    COALESCE(SUM(CASE WHEN so.gender_id = 11 THEN 1 ELSE 0 END), 0) AS both_count,
    COALESCE(SUM(CASE WHEN so.gender_id = 12 THEN 1 ELSE 0 END), 0) AS intersex_count,
    COALESCE(SUM(CASE WHEN so.gender_id = 13 THEN 1 ELSE 0 END), 0) AS androgynous_count
FROM 
    (
        SELECT 
            COUNT(image_id) AS count,
            it.topic_id,
            t.topic
        FROM 
            ImagesTopics it
        JOIN 
            Topics t ON it.topic_id = t.topic_id
        GROUP BY 
            it.topic_id, t.topic
    ) AS topic_count
LEFT JOIN 
    SegmentOct20 so ON so.image_id = topic_count.image_id
GROUP BY 
    topic_count.topic_id, topic_count.topic;

   
USE stock;   
   SELECT DISTINCT seg1.image_id, seg1.site_name_id, seg1.contentUrl, seg1.imagename, e.encoding_id, seg1.site_image_id, e.face_landmarks, e.bbox FROM SegmentOct20 seg1 INNER JOIN SegmentHelperApril4_topic17 ht ON seg1.image_id = ht.image_id LEFT JOIN ImagesTopics it ON seg1.image_id = it.image_id WHERE e.body_landmarks IS NULL  LIMIT 100;

>>>>>>> Stashed changes
