USE Stock;

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


-- county by topic with age and gender
SELECT 
    COALESCE(age_count.count, 0) AS age_count,
    COALESCE(gender_count.count, 0) AS gender_count,
    COALESCE(topic_count.count, 0) AS topic_count,
    age_count.age_id,
    gender_count.gender_id,
    topic_count.topic_id
FROM
    (
        SELECT 
            COUNT(image_id) AS count,
            age_id
        FROM 
            COUNT(image_id) AS count,
            gender_id
        FROM 
            SegmentOct20
        GROUP BY 
            gender_id
    ) AS gender_count ON 1=1
LEFT JOIN
    (
        SELECT 
            COUNT(image_id) AS count,
            it.topic_id
        FROM 
            ImagesTopics it
        JOIN 
            Topics t ON it.topic_id = t.topic_id
        GROUP BY 
            it.topic_id
    ) AS topic_count ON 1=1;
