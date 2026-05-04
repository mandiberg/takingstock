'''
This script will generate a list of image_id, images.description, and ImagesTopics.topic_score (as image_id	description	topic_fit)
From all images in a specified topic_id (11 for example)
Where that image_id is in ImagesObjectFusion
And that ImagesObjectFusion cluster_dist is less than 0.5 regardless of the cluster_id
and ImagesTopics.model_score is between 0.5 and 0.6
'''

USE Stock;

SELECT iof.image_id, i.description, it.topic_score as topic_fit
;

SELECT COUNT(iof.image_id)
FROM ImagesObjectFusion iof
INNER JOIN Images i ON iof.image_id = i.image_id
INNER JOIN ImagesTopics it ON iof.image_id = it.image_id
WHERE iof.cluster_dist < 30
AND it.topic_id = 11
AND it.topic_score BETWEEN 0.5 AND 0.6
-- LIMIT 100
;