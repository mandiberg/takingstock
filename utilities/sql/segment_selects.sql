USE stock;

-- SELECT DISTINCT(s.image_id), s.site_name_id, s.contentUrl, s.imagename, s.face_x, s.face_y, s.face_z, s.mouth_gap, s.face_landmarks, s.bbox, s.face_encodings68, s.site_image_id, s.body_landmarks, it.topic_score, ibg.lum, ibg.lum_bb, ibg.hue, ibg.hue_bb, ibg.sat, ibg.sat_bb  
SELECT COUNT(s.image_id)
FROM SegmentOct20 s  JOIN ImagesTopics it ON s.image_id = it.image_id  JOIN ImagesBackground ibg ON s.image_id = ibg.image_id  
WHERE s.site_name_id != 1 AND face_encodings68 IS NOT NULL 
AND face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2 
AND it.topic_score > .5
AND s.age_id NOT IN (1,2,3)   
AND it.topic_id IN (7)  
LIMIT 200000