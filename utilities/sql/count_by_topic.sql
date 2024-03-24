USE Stock;

SELECT COUNT(image_id) AS count,
       it.topic_id,
       t.topic
FROM ImagesTopics it
JOIN Topics t ON it.topic_id = t.topic_id
GROUP BY it.topic_id, t.topic;
