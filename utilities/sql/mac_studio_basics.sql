-- Pivot table: cluster_id (rows) x class_id (columns)
-- Shows count of distinct image_ids for each cluster_id and class_id combination

USE Stock;
SET GLOBAL innodb_buffer_pool_size=8053063680;


DELETE FROM ImagesObjectFusion ;
DELETE FROM ObjectFusion ;


DELETE FROM ImagesObjectSignatures ;
DELETE FROM ObjectSignatures ;


SELECT COUNT(sh.image_id)
FROM SegmentHelper_oct2025_every40 sh
INNER JOIN SegmentHelper_T11_Oct20_COCO_Custom t on t.image_id = sh.image_id  
;
-- 963574

-- Verify the changes
DESCRIBE ImagesDetections;

SELECT *
FROM ImagesDetections 
WHERE left_pointer_x != 0;

SELECT *
FROM ImagesDetections 
WHERE both_hands_object_id  != 0
OR left_hand_object_id  != 0;


SELECT DISTINCT(s.image_id) , e.pitch, e.yaw, e.roll  ;

SELECT COUNT(s.image_id)
FROM SegmentBig_isface s  
JOIN Encodings e ON s.image_id = e.image_id 
INNER JOIN Detections h ON h.image_id = s.image_id  
WHERE  e.is_dupe_of IS NULL
AND e.two_noses IS NULL
AND e.is_face = 1
AND e.mongo_body_landmarks_norm = 1
;

SELECT COUNT(d.image_id)
FROM Detections d 
WHERE d.conf = -1
;

SELECT COUNT(d.image_id)
FROM Detections d 
WHERE d.bbox_norm IS NULL
LIMIT 10
;













-- cluster-class_id matrix
-- produces cluster-class_id matrix withcounts for each intersection
-- is not location specific. 
-- EXCLUDES ALL NON-OBJECT PLACEMENT IMAGES!!

   SELECT iof.cluster_id,
    SUM(CASE WHEN d.class_id = 0 THEN 1 ELSE 0 END) AS class_0,
    SUM(CASE WHEN d.class_id = 1 THEN 1 ELSE 0 END) AS class_1,
    SUM(CASE WHEN d.class_id = 2 THEN 1 ELSE 0 END) AS class_2,
    SUM(CASE WHEN d.class_id = 3 THEN 1 ELSE 0 END) AS class_3,
    SUM(CASE WHEN d.class_id = 4 THEN 1 ELSE 0 END) AS class_4,
    SUM(CASE WHEN d.class_id = 5 THEN 1 ELSE 0 END) AS class_5,
    SUM(CASE WHEN d.class_id = 6 THEN 1 ELSE 0 END) AS class_6,
    SUM(CASE WHEN d.class_id = 7 THEN 1 ELSE 0 END) AS class_7,
    SUM(CASE WHEN d.class_id = 8 THEN 1 ELSE 0 END) AS class_8,
    SUM(CASE WHEN d.class_id = 9 THEN 1 ELSE 0 END) AS class_9,
    SUM(CASE WHEN d.class_id = 10 THEN 1 ELSE 0 END) AS class_10,
    SUM(CASE WHEN d.class_id = 11 THEN 1 ELSE 0 END) AS class_11,
    SUM(CASE WHEN d.class_id = 12 THEN 1 ELSE 0 END) AS class_12,
    SUM(CASE WHEN d.class_id = 13 THEN 1 ELSE 0 END) AS class_13,
    SUM(CASE WHEN d.class_id = 14 THEN 1 ELSE 0 END) AS class_14,
    SUM(CASE WHEN d.class_id = 15 THEN 1 ELSE 0 END) AS class_15,
    SUM(CASE WHEN d.class_id = 16 THEN 1 ELSE 0 END) AS class_16,
    SUM(CASE WHEN d.class_id = 17 THEN 1 ELSE 0 END) AS class_17,
    SUM(CASE WHEN d.class_id = 18 THEN 1 ELSE 0 END) AS class_18,
    SUM(CASE WHEN d.class_id = 19 THEN 1 ELSE 0 END) AS class_19,
    SUM(CASE WHEN d.class_id = 20 THEN 1 ELSE 0 END) AS class_20,
    SUM(CASE WHEN d.class_id = 21 THEN 1 ELSE 0 END) AS class_21,
    SUM(CASE WHEN d.class_id = 22 THEN 1 ELSE 0 END) AS class_22,
    SUM(CASE WHEN d.class_id = 23 THEN 1 ELSE 0 END) AS class_23,
    SUM(CASE WHEN d.class_id = 24 THEN 1 ELSE 0 END) AS class_24,
    SUM(CASE WHEN d.class_id = 25 THEN 1 ELSE 0 END) AS class_25,
    SUM(CASE WHEN d.class_id = 26 THEN 1 ELSE 0 END) AS class_26,
    SUM(CASE WHEN d.class_id = 27 THEN 1 ELSE 0 END) AS class_27,
    SUM(CASE WHEN d.class_id = 28 THEN 1 ELSE 0 END) AS class_28,
    SUM(CASE WHEN d.class_id = 29 THEN 1 ELSE 0 END) AS class_29,
    SUM(CASE WHEN d.class_id = 30 THEN 1 ELSE 0 END) AS class_30,
    SUM(CASE WHEN d.class_id = 31 THEN 1 ELSE 0 END) AS class_31,
    SUM(CASE WHEN d.class_id = 32 THEN 1 ELSE 0 END) AS class_32,
    SUM(CASE WHEN d.class_id = 33 THEN 1 ELSE 0 END) AS class_33,
    SUM(CASE WHEN d.class_id = 34 THEN 1 ELSE 0 END) AS class_34,
    SUM(CASE WHEN d.class_id = 35 THEN 1 ELSE 0 END) AS class_35,
    SUM(CASE WHEN d.class_id = 36 THEN 1 ELSE 0 END) AS class_36,
    SUM(CASE WHEN d.class_id = 37 THEN 1 ELSE 0 END) AS class_37,
    SUM(CASE WHEN d.class_id = 38 THEN 1 ELSE 0 END) AS class_38,
    SUM(CASE WHEN d.class_id = 39 THEN 1 ELSE 0 END) AS class_39,
    SUM(CASE WHEN d.class_id = 40 THEN 1 ELSE 0 END) AS class_40,
    SUM(CASE WHEN d.class_id = 41 THEN 1 ELSE 0 END) AS class_41,
    SUM(CASE WHEN d.class_id = 42 THEN 1 ELSE 0 END) AS class_42,
    SUM(CASE WHEN d.class_id = 43 THEN 1 ELSE 0 END) AS class_43,
    SUM(CASE WHEN d.class_id = 44 THEN 1 ELSE 0 END) AS class_44,
    SUM(CASE WHEN d.class_id = 45 THEN 1 ELSE 0 END) AS class_45,
    SUM(CASE WHEN d.class_id = 46 THEN 1 ELSE 0 END) AS class_46,
    SUM(CASE WHEN d.class_id = 47 THEN 1 ELSE 0 END) AS class_47,
    SUM(CASE WHEN d.class_id = 48 THEN 1 ELSE 0 END) AS class_48,
    SUM(CASE WHEN d.class_id = 49 THEN 1 ELSE 0 END) AS class_49,
    SUM(CASE WHEN d.class_id = 50 THEN 1 ELSE 0 END) AS class_50,
    SUM(CASE WHEN d.class_id = 51 THEN 1 ELSE 0 END) AS class_51,
    SUM(CASE WHEN d.class_id = 52 THEN 1 ELSE 0 END) AS class_52,
    SUM(CASE WHEN d.class_id = 53 THEN 1 ELSE 0 END) AS class_53,
    SUM(CASE WHEN d.class_id = 54 THEN 1 ELSE 0 END) AS class_54,
    SUM(CASE WHEN d.class_id = 55 THEN 1 ELSE 0 END) AS class_55,
    SUM(CASE WHEN d.class_id = 56 THEN 1 ELSE 0 END) AS class_56,
    SUM(CASE WHEN d.class_id = 57 THEN 1 ELSE 0 END) AS class_57,
    SUM(CASE WHEN d.class_id = 58 THEN 1 ELSE 0 END) AS class_58,
    SUM(CASE WHEN d.class_id = 59 THEN 1 ELSE 0 END) AS class_59,
    SUM(CASE WHEN d.class_id = 60 THEN 1 ELSE 0 END) AS class_60,
    SUM(CASE WHEN d.class_id = 61 THEN 1 ELSE 0 END) AS class_61,
    SUM(CASE WHEN d.class_id = 62 THEN 1 ELSE 0 END) AS class_62,
    SUM(CASE WHEN d.class_id = 63 THEN 1 ELSE 0 END) AS class_63,
    SUM(CASE WHEN d.class_id = 64 THEN 1 ELSE 0 END) AS class_64,
    SUM(CASE WHEN d.class_id = 65 THEN 1 ELSE 0 END) AS class_65,
    SUM(CASE WHEN d.class_id = 66 THEN 1 ELSE 0 END) AS class_66,
    SUM(CASE WHEN d.class_id = 67 THEN 1 ELSE 0 END) AS class_67,
    SUM(CASE WHEN d.class_id = 68 THEN 1 ELSE 0 END) AS class_68,
    SUM(CASE WHEN d.class_id = 69 THEN 1 ELSE 0 END) AS class_69,
    SUM(CASE WHEN d.class_id = 70 THEN 1 ELSE 0 END) AS class_70,
    SUM(CASE WHEN d.class_id = 71 THEN 1 ELSE 0 END) AS class_71,
    SUM(CASE WHEN d.class_id = 72 THEN 1 ELSE 0 END) AS class_72,
    SUM(CASE WHEN d.class_id = 73 THEN 1 ELSE 0 END) AS class_73,
    SUM(CASE WHEN d.class_id = 74 THEN 1 ELSE 0 END) AS class_74,
    SUM(CASE WHEN d.class_id = 75 THEN 1 ELSE 0 END) AS class_75,
    SUM(CASE WHEN d.class_id = 76 THEN 1 ELSE 0 END) AS class_76,
    SUM(CASE WHEN d.class_id = 77 THEN 1 ELSE 0 END) AS class_77,
    SUM(CASE WHEN d.class_id = 78 THEN 1 ELSE 0 END) AS class_78,
    SUM(CASE WHEN d.class_id = 79 THEN 1 ELSE 0 END) AS class_79,
    SUM(CASE WHEN d.class_id = 80 THEN 1 ELSE 0 END) AS class_80,
    SUM(CASE WHEN d.class_id = 81 THEN 1 ELSE 0 END) AS class_81,
    SUM(CASE WHEN d.class_id = 82 THEN 1 ELSE 0 END) AS class_82,
    SUM(CASE WHEN d.class_id = 83 THEN 1 ELSE 0 END) AS class_83,
    SUM(CASE WHEN d.class_id = 84 THEN 1 ELSE 0 END) AS class_84,
    SUM(CASE WHEN d.class_id = 85 THEN 1 ELSE 0 END) AS class_85,
    SUM(CASE WHEN d.class_id = 86 THEN 1 ELSE 0 END) AS class_86,
    SUM(CASE WHEN d.class_id = 87 THEN 1 ELSE 0 END) AS class_87,
    SUM(CASE WHEN d.class_id = 88 THEN 1 ELSE 0 END) AS class_88,
    SUM(CASE WHEN d.class_id = 89 THEN 1 ELSE 0 END) AS class_89,
    SUM(CASE WHEN d.class_id = 90 THEN 1 ELSE 0 END) AS class_90,
    SUM(CASE WHEN d.class_id = 91 THEN 1 ELSE 0 END) AS class_91,
    SUM(CASE WHEN d.class_id = 92 THEN 1 ELSE 0 END) AS class_92,
    SUM(CASE WHEN d.class_id = 93 THEN 1 ELSE 0 END) AS class_93,
    SUM(CASE WHEN d.class_id = 94 THEN 1 ELSE 0 END) AS class_94,
    SUM(CASE WHEN d.class_id = 95 THEN 1 ELSE 0 END) AS class_95,
    SUM(CASE WHEN d.class_id = 96 THEN 1 ELSE 0 END) AS class_96,
    SUM(CASE WHEN d.class_id = 97 THEN 1 ELSE 0 END) AS class_97,
    SUM(CASE WHEN d.class_id = 98 THEN 1 ELSE 0 END) AS class_98,
    SUM(CASE WHEN d.class_id = 99 THEN 1 ELSE 0 END) AS class_99,
    SUM(CASE WHEN d.class_id = 100 THEN 1 ELSE 0 END) AS class_100,
    SUM(CASE WHEN d.class_id = 101 THEN 1 ELSE 0 END) AS class_101,
    SUM(CASE WHEN d.class_id = 102 THEN 1 ELSE 0 END) AS class_102,
    SUM(CASE WHEN d.class_id = 103 THEN 1 ELSE 0 END) AS class_103,
    SUM(CASE WHEN d.class_id = 104 THEN 1 ELSE 0 END) AS class_104,
    SUM(CASE WHEN d.class_id = 105 THEN 1 ELSE 0 END) AS class_105,
    SUM(CASE WHEN d.class_id = 106 THEN 1 ELSE 0 END) AS class_106,
    SUM(CASE WHEN d.class_id = 107 THEN 1 ELSE 0 END) AS class_107,
    SUM(CASE WHEN d.class_id = 108 THEN 1 ELSE 0 END) AS class_108,
    SUM(CASE WHEN d.class_id = 109 THEN 1 ELSE 0 END) AS class_109,
    SUM(CASE WHEN d.class_id = 110 THEN 1 ELSE 0 END) AS class_110,
    SUM(CASE WHEN d.class_id = 111 THEN 1 ELSE 0 END) AS class_111,
    SUM(CASE WHEN d.class_id = 112 THEN 1 ELSE 0 END) AS class_112,
    SUM(CASE WHEN d.class_id = 113 THEN 1 ELSE 0 END) AS class_113,
    SUM(CASE WHEN d.class_id = 114 THEN 1 ELSE 0 END) AS class_114,
    SUM(CASE WHEN d.class_id = 115 THEN 1 ELSE 0 END) AS class_115,
    SUM(CASE WHEN d.class_id = 116 THEN 1 ELSE 0 END) AS class_116,
    SUM(CASE WHEN d.class_id = 117 THEN 1 ELSE 0 END) AS class_117,
    SUM(CASE WHEN d.class_id = 118 THEN 1 ELSE 0 END) AS class_118,
    SUM(CASE WHEN d.class_id = 119 THEN 1 ELSE 0 END) AS class_119,
    COUNT(DISTINCT iof.image_id) AS total_images
FROM ImagesObjectFusion iof
INNER JOIN (
  SELECT image_id, left_hand_object_id AS detection_id
  FROM ImagesDetections
  WHERE left_hand_object_id IS NOT NULL
  UNION
  SELECT image_id, right_hand_object_id AS detection_id
  FROM ImagesDetections
  WHERE right_hand_object_id IS NOT NULL
  UNION
  SELECT image_id, top_face_object_id AS detection_id
  FROM ImagesDetections
  WHERE top_face_object_id IS NOT NULL
  UNION
  SELECT image_id, left_eye_object_id AS detection_id
  FROM ImagesDetections
  WHERE left_eye_object_id IS NOT NULL
  UNION
  SELECT image_id, right_eye_object_id AS detection_id
  FROM ImagesDetections
  WHERE right_eye_object_id IS NOT NULL
  UNION
  SELECT image_id, mouth_object_id AS detection_id
  FROM ImagesDetections
  WHERE mouth_object_id IS NOT NULL
  UNION
  SELECT image_id, shoulder_object_id AS detection_id
  FROM ImagesDetections
  WHERE shoulder_object_id IS NOT NULL
  UNION
  SELECT image_id, waist_object_id AS detection_id
  FROM ImagesDetections
  WHERE waist_object_id IS NOT NULL
  UNION
  SELECT image_id, feet_object_id AS detection_id
  FROM ImagesDetections
  WHERE feet_object_id IS NOT NULL
) AS selected_det ON iof.image_id = selected_det.image_id
INNER JOIN Detections d ON selected_det.detection_id = d.detection_id
GROUP BY iof.cluster_id
ORDER BY iof.cluster_id;


SELECT COUNT(*)
FROM SegmentHelper_T11_Oct20_COCO_Custom_evens_quarters id 
;
-- 22103472
  
-- class_id x object-position matrix (from ImagesDetections)
-- rows are class_id; columns are ObjectFusion positions in ImagesDetections
SELECT
  pos.class_id,
  SUM(CASE WHEN pos.object_position = 'left_hand' THEN 1 ELSE 0 END) AS left_hand_count,
  SUM(CASE WHEN pos.object_position = 'right_hand' THEN 1 ELSE 0 END) AS right_hand_count,
  SUM(CASE WHEN pos.object_position = 'top_face' THEN 1 ELSE 0 END) AS top_face_count,
  SUM(CASE WHEN pos.object_position = 'left_eye' THEN 1 ELSE 0 END) AS left_eye_count,
  SUM(CASE WHEN pos.object_position = 'right_eye' THEN 1 ELSE 0 END) AS right_eye_count,
  SUM(CASE WHEN pos.object_position = 'mouth' THEN 1 ELSE 0 END) AS mouth_count,
  SUM(CASE WHEN pos.object_position = 'shoulder' THEN 1 ELSE 0 END) AS shoulder_count,
  SUM(CASE WHEN pos.object_position = 'waist' THEN 1 ELSE 0 END) AS waist_count,
  SUM(CASE WHEN pos.object_position = 'feet' THEN 1 ELSE 0 END) AS feet_count,
  COUNT(*) AS total_position_assignments,
  COUNT(DISTINCT pos.image_id) AS total_images_with_class
FROM (
  SELECT idet.image_id, 'left_hand' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.left_hand_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'right_hand' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.right_hand_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'top_face' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.top_face_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'left_eye' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.left_eye_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'right_eye' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.right_eye_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'mouth' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.mouth_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'shoulder' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.shoulder_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'waist' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.waist_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'feet' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.feet_object_id = d.detection_id
) AS pos
GROUP BY pos.class_id
ORDER BY pos.class_id;


-- quick slot population sanity check (helps verify whether waist/feet are being written at all)
SELECT
  COUNT(*) AS rows_total,
  SUM(CASE WHEN waist_object_id IS NOT NULL THEN 1 ELSE 0 END) AS waist_rows_nonnull,
  SUM(CASE WHEN feet_object_id IS NOT NULL THEN 1 ELSE 0 END) AS feet_rows_nonnull,
  SUM(CASE WHEN shoulder_object_id IS NOT NULL THEN 1 ELSE 0 END) AS shoulder_rows_nonnull,
  SUM(CASE WHEN mouth_object_id IS NOT NULL THEN 1 ELSE 0 END) AS mouth_rows_nonnull
FROM ImagesDetections;


-- tie-only destination breakdown (class_id 27)
SELECT
  pos.object_position,
  COUNT(*) AS tie_assignments,
  COUNT(DISTINCT pos.image_id) AS tie_images
FROM (
  SELECT idet.image_id, 'left_hand' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.left_hand_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'right_hand' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.right_hand_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'top_face' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.top_face_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'left_eye' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.left_eye_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'right_eye' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.right_eye_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'mouth' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.mouth_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'shoulder' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.shoulder_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'waist' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.waist_object_id = d.detection_id

  UNION ALL

  SELECT idet.image_id, 'feet' AS object_position, d.class_id
  FROM ImagesDetections idet
  INNER JOIN Detections d ON idet.feet_object_id = d.detection_id
) AS pos
WHERE pos.class_id = 27
GROUP BY pos.object_position
ORDER BY tie_assignments DESC;







'''
Detections table has bbox and bbox_norm collumns
I want to know how many rows have null in bbox_norm but not in bbox, 
and how many have null in both, and how many have values in both.
for class_id 110
'''

USE Stock;

SELECT
  SUM(CASE WHEN bbox IS NULL AND (bbox_norm IS NULL OR JSON_EXTRACT(bbox_norm, '$.left') IS NULL) THEN 1 ELSE 0 END) AS null_bbox_and_null_bbox_norm,
  SUM(CASE WHEN bbox IS NOT NULL AND (bbox_norm IS NULL OR JSON_EXTRACT(bbox_norm, '$.left') IS NULL) THEN 1 ELSE 0 END) AS not_null_bbox_and_null_bbox_norm,
  SUM(CASE WHEN bbox IS NOT NULL AND bbox_norm IS NOT NULL AND JSON_EXTRACT(bbox_norm, '$.left') IS NOT NULL THEN 1 ELSE 0 END) AS not_null_bbox_and_not_null_bbox_norm
FROM Detections d
JOIN SegmentHelper_T11_Oct20_COCO_Custom s ON s.image_id = d.image_id 
;
-- WHERE class_id = 111;

-- SegmentHelper_T11_Oct20_COCO_Custom
-- 0	6464930	31286406


-- SegmentHelper_T11_Oct20_COCO_Custom_evens_quarters 4/7
-- 0	3198738	4906887
0	690974	7478491

-- SegmentHelper_YOLO_Selects 4/7
-- 0	11996771	16094438
-- 0	5378188	22820307

-- 4/5 total: 0	32944858	20900495
-- 4/7 total: 0	31934191	22982959



'''
Detections table has bbox and bbox_norm collumns
NoDetections and NoDetectionsCustom contain image_ids of known no detections images
I want to know the breakdown of SegmentHelper_T11_Oct20_COCO_Custom_evens_quarters by 
unprocessed_image -- only in SegmentHelper_T11_Oct20_COCO_Custom_evens_quarters
no_detections -- only in NoDetections
no_detections_custom -- only in NoDetectionsCustom
not_null_bbox_and_null_bbox_norm -- in Detections and has bbox but null bbox_norm
not_null_bbox_and_not_null_bbox_norm -- in Detections and has bbox and bbox_norm
'''

WITH detection_flags AS (
  SELECT
    d.image_id,
    MAX(CASE
      WHEN d.bbox IS NOT NULL
       AND (d.bbox_norm IS NULL OR JSON_EXTRACT(d.bbox_norm, '$.left') IS NULL)
      THEN 1 ELSE 0 END) AS has_bbox_and_null_bbox_norm,
    MAX(CASE
      WHEN d.bbox IS NOT NULL
       AND d.bbox_norm IS NOT NULL
       AND JSON_EXTRACT(d.bbox_norm, '$.left') IS NOT NULL
      THEN 1 ELSE 0 END) AS has_bbox_and_not_null_bbox_norm
  FROM Detections d
  GROUP BY d.image_id
)
SELECT
  SUM(CASE WHEN df.image_id IS NULL AND nd.image_id IS NULL AND ndc.image_id IS NULL THEN 1 ELSE 0 END) AS unprocessed_image,
  SUM(CASE WHEN nd.image_id IS NOT NULL THEN 1 ELSE 0 END) AS no_detections,
  SUM(CASE WHEN ndc.image_id IS NOT NULL THEN 1 ELSE 0 END) AS no_detections_custom,
  SUM(CASE WHEN df.has_bbox_and_null_bbox_norm = 1 THEN 1 ELSE 0 END) AS not_null_bbox_and_null_bbox_norm,
  SUM(CASE WHEN df.has_bbox_and_not_null_bbox_norm = 1 THEN 1 ELSE 0 END) AS not_null_bbox_and_not_null_bbox_norm
FROM SegmentHelper_T11_Oct20_COCO_Custom_evens_quarters s
LEFT JOIN (SELECT DISTINCT image_id FROM NoDetections) nd ON s.image_id = nd.image_id
LEFT JOIN (SELECT DISTINCT image_id FROM NoDetectionsCustom) ndc ON s.image_id = ndc.image_id
LEFT JOIN detection_flags df ON s.image_id = df.image_id
;


-- results:
-- 3476286	0	0	887547	7244879


SELECT COUNT(image_id)
FROM SegmentHelper_T11_Oct20_COCO_Custom_evens_quarters s
;
-- 8033889


-- WHERE class_id = 111;


'''
count how many image_ids have face_x but pitch yaw roll are NULL, how many have both
'''

SELECT COUNT(DISTINCT s.image_id) AS total_images_with_face_x_and_null_pitch_yaw_roll
FROM Detections s
JOIN Encodings e ON s.image_id = e.image_id
JOIN SegmentHelper_YOLO_Selects sh ON sh.image_id = s.image_id 
WHERE e.face_x IS NOT NULL
AND (e.pitch IS NULL OR e.yaw IS NULL OR e.roll IS NULL)
; 

-- 4/5 total: 2165712



'''
count how many image_ids have mongo_body_landmarks but not mongo_body_landmarks_norm, how many have both, how many have neither
'''

SELECT
  SUM(CASE WHEN mongo_body_landmarks IS NULL AND mongo_body_landmarks_norm IS NULL THEN 1 ELSE 0 END) AS null_landmarks_and_null_norm,
  SUM(CASE WHEN mongo_body_landmarks IS NOT NULL AND mongo_body_landmarks_norm IS NULL THEN 1 ELSE 0 END) AS not_null_landmarks_and_null_norm,
  SUM(CASE WHEN mongo_body_landmarks IS NOT NULL AND mongo_body_landmarks_norm IS NOT NULL THEN 1 ELSE 0 END) AS not_null_landmarks_and_not_null_norm
FROM Encodings
WHERE mongo_body_landmarks IS NOT NULL OR mongo_body_landmarks_norm IS NOT NULL
;

-- 4/7 0	518902	100126063

'''
count how many image_ids have mongo_hand_landmarks but not mongo_hand_landmarks_norm, how many have both, how many have neither
'''

SELECT
  SUM(CASE WHEN mongo_hand_landmarks IS NULL AND mongo_hand_landmarks_norm IS NULL THEN 1 ELSE 0 END) AS null_hand_landmarks_and_null_norm,
  SUM(CASE WHEN mongo_hand_landmarks IS NOT NULL AND mongo_hand_landmarks_norm IS NULL THEN 1 ELSE 0 END) AS not_null_hand_landmarks_and_null_norm,
  SUM(CASE WHEN mongo_hand_landmarks IS NOT NULL AND mongo_hand_landmarks_norm IS NOT NULL THEN 1 ELSE 0 END) AS not_null_hand_landmarks_and_not_null_norm
FROM Encodings
WHERE mongo_hand_landmarks IS NOT NULL OR mongo_hand_landmarks_norm IS NOT NULL

-- 0	71496732	20636721

;


USE Stock; 
SHOW TABLES;

'''
Total images in table: 130,184,028
Images with h AND w populated: 84,176,129
Images missing h OR w: 46,007,899
'''



-- create helper segment table

CREATE TABLE SegmentHelper_T11_Oct20_COCO_Custom_evens_quarters (
    seg_image_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    image_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES Images(image_id)
);

USE Stock;

CREATE TABLE SegmentHelper_YOLO_Selects (
    seg_image_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    image_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES Images(image_id)
);

CREATE TABLE SegmentHelper_T11_Oct20_COCO_Custom_every40 (
    seg_image_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    image_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES Images(image_id)
);


INSERT INTO SegmentHelper_T11_Oct20_COCO_Custom_evens_quarters (image_id)
    SELECT shoeq.image_id 
    FROM SegmentHelper_T11_Oct20_COCO_Custom t
    JOIN SegmentHelper_oct2025_evens_quarters shoeq ON shoeq.image_id = t.image_id 
    WHERE shoeq.image_id IS NOT NULL
;
    
INSERT INTO SegmentHelper_T11_Oct20_COCO_Custom (image_id)
    SELECT image_id FROM SegmentHelper_YOLO_Selects WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelper_T4_T11_T37_T40 WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObjectYOLO WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentOct20 WHERE image_id IS NOT NULL
;
-- 38.5M

-- insert into SegmentHelper_T11_Oct20_COCO_Custom of union of all image_id in these helper tables

USE Stock;


USE Stock;
-- for making a helper from Segment Object based on t64 Topic
INSERT INTO SegmentHelper_T37_money (image_id)
SELECT DISTINCT e.image_id
FROM SegmentBig_isface e
JOIN ImagesTopics iap ON iap.image_id = e.image_id
WHERE (iap.topic_id = 37 OR iap.topic_id2 = 37)
AND NOT EXISTS (
        SELECT 1
        FROM SegmentHelper_T37_money s
        WHERE s.image_id = e.image_id
    )
LIMIT 2000000
;


INSERT INTO SegmentHelper_T11_Oct20_COCO_Custom_every40 (image_id)
    SELECT sh.image_id FROM SegmentHelper_oct2025_every40 sh
	INNER JOIN SegmentHelper_T11_Oct20_COCO_Custom t on t.image_id = sh.image_id  

    
    ; 
-- 8M

INSERT INTO SegmentHelper_YOLO_Selects (image_id)
    SELECT image_id FROM SegmentHelperObject_100_tulip WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_101_flowers_other WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_102_orchid WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_103_peony WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_41_cup_glass WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_45_salad WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_55_cake WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_67_phone WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_73_book WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_74_clock WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_80_sign WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_81_gift WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_82_money WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_83_bag WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_84_valentine WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_86_dumbbell WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_87_flag WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_89_mask WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_90_stethoscope WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_91_gun WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_92_headphones WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_94_piggybank WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_96_bitcoin WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_97_rose WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_98_lily WHERE image_id IS NOT NULL
    UNION
    SELECT image_id FROM SegmentHelperObject_99_iris WHERE image_id IS NOT NULL
;
-- 18.4M


'''
SegmentHelper_T11_Oct20_COCO_Custom
SegmentHelper_T4_occupation
SegmentHelper_T11_business
SegmentHelper_T37_money
SegmentHelper_T40_technology
SegmentHelperObject_100_tulip
SegmentHelperObject_101_flowers_other
SegmentHelperObject_102_orchid
SegmentHelperObject_103_peony
SegmentHelperObject_41_cup_glass
SegmentHelperObject_45_salad
SegmentHelperObject_55_cake
SegmentHelperObject_67_phone
SegmentHelperObject_73_book
SegmentHelperObject_74_clock
SegmentHelperObject_80_sign
SegmentHelperObject_81_gift
SegmentHelperObject_82_money
SegmentHelperObject_83_bag
SegmentHelperObject_84_valentine
SegmentHelperObject_86_dumbbell
SegmentHelperObject_87_flag
SegmentHelperObject_89_mask
SegmentHelperObject_90_stethoscope
SegmentHelperObject_91_gun
SegmentHelperObject_92_headphones
SegmentHelperObject_94_piggybank
SegmentHelperObject_96_bitcoin
SegmentHelperObject_97_rose
SegmentHelperObject_98_lily
SegmentHelperObject_99_iris
SegmentHelperObjectYOLO
SegmentOct20
'''




