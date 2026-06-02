-- normalize object bbox
-- use this instead of normalize_lms.py
-- only works when image sizes are already in db

Use Stock;



-- PRODUCTION

SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET autocommit = 0;

START TRANSACTION;

UPDATE Detections d
JOIN Encodings e ON e.image_id = d.image_id
JOIN SegmentHelper_TheOffice sh ON sh.image_id = d.image_id
SET d.bbox_norm = JSON_OBJECT(
'left', (CAST(COALESCE(JSON_EXTRACT(d.bbox,'$.left'),
IF(JSON_VALID(JSON_UNQUOTE(d.bbox)),
JSON_EXTRACT(CAST(JSON_UNQUOTE(d.bbox) AS JSON),'$.left'), NULL)) AS DECIMAL(12,6)) - e.nose_pixel_x) / e.face_height,
'top', (CAST(COALESCE(JSON_EXTRACT(d.bbox,'$.top'),
IF(JSON_VALID(JSON_UNQUOTE(d.bbox)),
JSON_EXTRACT(CAST(JSON_UNQUOTE(d.bbox) AS JSON),'$.top'), NULL)) AS DECIMAL(12,6)) - e.nose_pixel_y) / e.face_height,
'right', (CAST(COALESCE(JSON_EXTRACT(d.bbox,'$.right'),
IF(JSON_VALID(JSON_UNQUOTE(d.bbox)),
JSON_EXTRACT(CAST(JSON_UNQUOTE(d.bbox) AS JSON),'$.right'), NULL)) AS DECIMAL(12,6)) - e.nose_pixel_x) / e.face_height,
'bottom', (CAST(COALESCE(JSON_EXTRACT(d.bbox,'$.bottom'),
IF(JSON_VALID(JSON_UNQUOTE(d.bbox)),
JSON_EXTRACT(CAST(JSON_UNQUOTE(d.bbox) AS JSON),'$.bottom'), NULL)) AS DECIMAL(12,6)) - e.nose_pixel_y) / e.face_height
)
WHERE d.detection_id BETWEEN 5000000 AND 10000000
AND d.bbox IS NOT NULL
AND d.conf != -1
AND e.face_height > 0
AND e.nose_pixel_x IS NOT NULL
AND e.nose_pixel_y IS NOT NULL
AND (
  d.bbox_norm IS NULL
  OR NOT (
    JSON_EXTRACT(d.bbox_norm, '$.left') IS NOT NULL
    OR (
      JSON_TYPE(d.bbox_norm) = 'STRING'
      AND JSON_VALID(CAST(JSON_UNQUOTE(d.bbox_norm) AS JSON)) = 1
      AND JSON_EXTRACT(CAST(JSON_UNQUOTE(d.bbox_norm) AS JSON), '$.left') IS NOT NULL
    )
  )
)
;

-- SELECT ROW_COUNT();

COMMIT;

