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



SELECT k.keyword, COUNT(*) AS count
FROM ImagesTopics it
JOIN Images i ON it.image_id = i.image_id
JOIN ImagesKeywords ik ON it.image_id = ik.image_id
JOIN Keywords k ON ik.keyword_id = k.keyword_id
WHERE it.topic_id = 57
GROUP BY k.keyword
ORDER BY count DESC;


