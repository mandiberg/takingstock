-- for Taking Stock description artist book sketch

Use Stock;
SET GLOBAL innodb_buffer_pool_size=8053063680;


SELECT i.description,
       COUNT(*) AS count
FROM ImagesTopics it
JOIN Images i ON it.image_id = i.image_id
WHERE it.topic_id = 37
GROUP BY i.description
ORDER BY count DESC;
