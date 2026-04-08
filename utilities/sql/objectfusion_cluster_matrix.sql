-- cluster-class_id matrix
-- produces cluster-class_id matrix withcounts for each intersection
-- INCLUDES ALL OBJECT PLACEMENT IMAGES, including non ones

-- PIVOT MATRIX VERSION:
-- one row per cluster_id, columns class_0..class_119, plus class_none for no-object images.
SET SESSION group_concat_max_len = 1000000;

WITH RECURSIVE class_nums AS (
  SELECT 0 AS class_id
  UNION ALL
  SELECT class_id + 1
  FROM class_nums
  WHERE class_id < 119
)
SELECT GROUP_CONCAT(
  CONCAT('SUM(CASE WHEN ic.class_id = ', class_id, ' THEN 1 ELSE 0 END) AS class_', class_id)
  ORDER BY class_id
  SEPARATOR ',\n  '
) INTO @pivot_cols
FROM class_nums;

SET @pivot_sql = CONCAT(
'WITH selected_det AS (
  SELECT image_id, left_hand_object_id AS detection_id FROM ImagesDetections WHERE left_hand_object_id IS NOT NULL
  UNION
  SELECT image_id, right_hand_object_id AS detection_id FROM ImagesDetections WHERE right_hand_object_id IS NOT NULL
  UNION
  SELECT image_id, top_face_object_id AS detection_id FROM ImagesDetections WHERE top_face_object_id IS NOT NULL
  UNION
  SELECT image_id, left_eye_object_id AS detection_id FROM ImagesDetections WHERE left_eye_object_id IS NOT NULL
  UNION
  SELECT image_id, right_eye_object_id AS detection_id FROM ImagesDetections WHERE right_eye_object_id IS NOT NULL
  UNION
  SELECT image_id, mouth_object_id AS detection_id FROM ImagesDetections WHERE mouth_object_id IS NOT NULL
  UNION
  SELECT image_id, shoulder_object_id AS detection_id FROM ImagesDetections WHERE shoulder_object_id IS NOT NULL
  UNION
  SELECT image_id, waist_object_id AS detection_id FROM ImagesDetections WHERE waist_object_id IS NOT NULL
  UNION
  SELECT image_id, feet_object_id AS detection_id FROM ImagesDetections WHERE feet_object_id IS NOT NULL
),
image_classes AS (
  SELECT sd.image_id, d.class_id
  FROM selected_det sd
  JOIN Detections d ON sd.detection_id = d.detection_id
  GROUP BY sd.image_id, d.class_id
)
SELECT
  iof.cluster_id,
  SUM(CASE WHEN ic.class_id IS NULL THEN 1 ELSE 0 END) AS class_none,
  ', @pivot_cols, ',
  COUNT(DISTINCT iof.image_id) AS total_images
FROM ImagesObjectFusion iof
LEFT JOIN image_classes ic ON iof.image_id = ic.image_id
GROUP BY iof.cluster_id
ORDER BY iof.cluster_id'
);

PREPARE stmt FROM @pivot_sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
