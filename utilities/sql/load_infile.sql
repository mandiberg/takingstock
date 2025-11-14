

USE Stock;

ALTER TABLE SegmentHelper_nov2025_SQL_only
ADD COLUMN image_path VARCHAR(55)
;


# Populate the new column with data from the Images table
UPDATE SegmentHelper_nov2025_SQL_only sh
JOIN Images i ON sh.image_id = i.image_id
SET sh.imagename = i.imagename
;

-- split the imagename on the first "/" and save the first part to image_path
UPDATE SegmentHelper_nov2025_SQL_only sh
SET sh.image_path = SUBSTRING_INDEX(sh.imagename, '/', 1)
WHERE sh.image_path IS NULL
;


USE Stock;
-- select count of HSV for heft keywords
SELECT  i.site_name_id, COUNT(i.image_id)
FROM Images i
JOIN SegmentHelper_nov2025_SQL_only ik ON i.image_id = ik.image_id 
GROUP BY i.site_name_id
ORDER BY i.site_name_id
;


USE Stock;
-- select count of HSV for heft keywords
SELECT i.site_name_id,
       ik.image_path,
       COUNT(i.image_id) AS image_count
FROM Images i
JOIN SegmentHelper_nov2025_SQL_only ik ON i.image_id = ik.image_id
GROUP BY i.site_name_id, ik.image_path
ORDER BY i.site_name_id, ik.image_path
;
