USE stock;


-- get move csvs for each topic 0, 3, 15, 45
SELECT i.site_name_id, i.imagename
FROM Images i
INNER JOIN SegmentHelper_NewBig s on i.image_id = s.image_id
JOIN ImagesTopics it on it.image_id = i.image_id
WHERE it.topic_id = 45
;



-- get move csvs for each topic 0, 3, 15, 45
SELECT i.site_name_id, i.imagename
FROM Images i
JOIN ImagesTopics it on it.image_id = i.image_id
WHERE it.topic_id = 46
AND (
	(EXISTS (SELECT 1 FROM NoDetections nd WHERE nd.image_id = i.image_id))
	AND (EXISTS (SELECT 1 FROM NoDetectionsCustom ndc WHERE ndc.image_id = i.image_id))
)
;



'''
move gym topic image_ids to T0 etc segments
move those same files to the T folders 
move all nodetections form newbig and delete
do i need to keep other newbig with detections? unclear
move all nodetections out of the LaCie remainder detections folders
delete those nodetections files
move all segmenthelpers for TheGym to a Gym unified topic
'''

-- select the intersection of the nodetections tables to move and delete 
-- SELECT DISTINCT(i.image_id)
SELECT i.site_name_id, i.imagename
FROM Images i
INNER JOIN SegmentHelper_NewBig s on i.image_id = s.image_id
-- JOIN Detections d on i.image_id = d.image_id
WHERE (
	(EXISTS (SELECT 1 FROM NoDetections nd WHERE nd.image_id = i.image_id))
	AND (EXISTS (SELECT 1 FROM NoDetectionsCustom ndc WHERE ndc.image_id = i.image_id))
)
-- LIMIT 10
;

SELECT *
FROM SegmentBig_isface d
WHERE d.image_id = 26385
;


SELECT COUNT(sb.image_id)
FROM SegmentBig_isface sb
INNER JOIN SegmentHelper_NewBig nb on nb.image_id = sb.image_id
LIMIT 10
;



SELECT COUNT(sb.image_id)
FROM SegmentHelper_NewBig sb
WHERE NOT (EXISTS (SELECT 1 FROM SegmentBig_isface nd WHERE nd.image_id = sb.image_id))
;


SELECT COUNT(sb.image_id)
FROM SegmentBig_isface sb
INNER JOIN SegmentHelper_T0_sport nb on nb.image_id = sb.image_id
WHERE sb.pitch IS NULL OR sb.yaw IS NULL or sb.roll IS NULL
;

-- 72M need pitch
-- 840k from newbig
-- 3.4M from T0 - so all the Gym needs XYZ




