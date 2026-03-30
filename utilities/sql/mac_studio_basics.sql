-- Pivot table: cluster_id (rows) x class_id (columns)
-- Shows count of distinct image_ids for each cluster_id and class_id combination

USE Stock;
SET GLOBAL innodb_buffer_pool_size=8053063680;


DELETE FROM ImagesObjectFusion ;

DELETE FROM ImagesDetections  ;

DELETE FROM ObjectFusion ;


SELECT COUNT(image_id)
FROM ImagesDetections
;

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


LIMIT 22000000

-- cluster-class_id matrix
-- produces cluster-class_id matrix withcounts for each intersection
-- is not location specific. 
SELECT 
    iof.cluster_id,
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
    SUM(CASE WHEN d.class_id = 21 
    THEN 1 ELSE 0 END) AS class_21,
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
) AS selected_det ON iof.image_id = selected_det.image_id
INNER JOIN Detections d ON selected_det.detection_id = d.detection_id
GROUP BY iof.cluster_id
ORDER BY iof.cluster_id;

  
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
) AS pos
GROUP BY pos.class_id
ORDER BY pos.class_id;



USE Stock;

DROP TABLE IF EXISTS ImagesObjectFusion;





SELECT 
    COUNT(*) as num_clusters,
    MIN(size) as min_size,
    MAX(size) as max_size,
    AVG(size) as avg_size,
    STDDEV(size) as stddev_size,
    SUM(CASE WHEN size > 10000 THEN 1 ELSE 0 END) as clusters_over_10k
FROM (
    SELECT cluster_id, COUNT(*) as size
    FROM ImagesObjectFusion
    GROUP BY cluster_id
) AS cluster_sizes;

-- existing version: 512	37	355544	8145.4219	30235.87846232398	85
-- revised weighting: 512	19	218434	8145.4219	21739.415825170578	76
-- reduced face angle: 768	11	694946	5430.2813	37600.473401539195	46
-- denormalized class_id: 1024	8	589417	4072.7109	29453.3158708742	67
	-- face angle 0.3: 1024	2	1430938	4072.7109	50742.639562944336	53

