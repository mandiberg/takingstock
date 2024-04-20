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




