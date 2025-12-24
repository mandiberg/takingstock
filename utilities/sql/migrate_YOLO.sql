USE Stock;
SET GLOBAL innodb_buffer_pool_size=8053063680;


INSERT INTO YoloClasses (yolo_class, class_name, model_version) VALUES
(0, 'person', 'yolov8'),
(1, 'bicycle', 'yolov8'),
(2, 'car', 'yolov8'),
(3, 'motorcycle', 'yolov8'),
(4, 'airplane', 'yolov8'),
(5, 'bus', 'yolov8'),
(6, 'train', 'yolov8'),
(7, 'truck', 'yolov8'),
(8, 'boat', 'yolov8'),
(9, 'traffic light', 'yolov8'),
(10, 'fire hydrant', 'yolov8'),
(11, 'stop sign', 'yolov8'),
(12, 'parking meter', 'yolov8'),
(13, 'bench', 'yolov8'),
(14, 'bird', 'yolov8'),
(15, 'cat', 'yolov8'),
(16, 'dog', 'yolov8'),
(17, 'horse', 'yolov8'),
(18, 'sheep', 'yolov8'),
(19, 'cow', 'yolov8'),
(20, 'elephant', 'yolov8'),
(21, 'bear', 'yolov8'),
(22, 'zebra', 'yolov8'),
(23, 'giraffe', 'yolov8'),
(24, 'backpack', 'yolov8'),
(25, 'umbrella', 'yolov8'),
(26, 'handbag', 'yolov8'),
(27, 'tie', 'yolov8'),
(28, 'suitcase', 'yolov8'),
(29, 'frisbee', 'yolov8'),
(30, 'skis', 'yolov8'),
(31, 'snowboard', 'yolov8'),
(32, 'sports ball', 'yolov8'),
(33, 'kite', 'yolov8'),
(34, 'baseball bat', 'yolov8'),
(35, 'baseball glove', 'yolov8'),
(36, 'skateboard', 'yolov8'),
(37, 'surfboard', 'yolov8'),
(38, 'tennis racket', 'yolov8'),
(39, 'bottle', 'yolov8'),
(40, 'wine glass', 'yolov8'),
(41, 'cup', 'yolov8'),
(42, 'fork', 'yolov8'),
(43, 'knife', 'yolov8'),
(44, 'spoon', 'yolov8'),
(45, 'bowl', 'yolov8'),
(46, 'banana', 'yolov8'),
(47, 'apple', 'yolov8'),
(48, 'sandwich', 'yolov8'),
(49, 'orange', 'yolov8'),
(50, 'brocolli', 'yolov8'),
(51, 'carrot', 'yolov8'),
(52, 'hot dog', 'yolov8'),
(53, 'pizza', 'yolov8'),
(54, 'donut', 'yolov8'),
(55, 'cake', 'yolov8'),
(56, 'chair', 'yolov8'),
(57, 'couch', 'yolov8'),
(58, 'potted plant', 'yolov8'),
(59, 'bed', 'yolov8'),
(60, 'dining table', 'yolov8'),
(61, 'toilet', 'yolov8'),
(62, 'tv', 'yolov8'),
(63, 'laptop', 'yolov8'),
(64, 'mouse', 'yolov8'),
(65, 'remote', 'yolov8'),
(66, 'keyboard', 'yolov8'),
(67, 'cell phone', 'yolov8'),
(68, 'microwave', 'yolov8'),
(69, 'oven', 'yolov8'),
(70, 'toaster', 'yolov8'),
(71, 'sink', 'yolov8'),
(72, 'refrigerator', 'yolov8'),
(73, 'book', 'yolov8'),
(74, 'clock', 'yolov8'),
(75, 'vase', 'yolov8'),
(76, 'scissors', 'yolov8'),
(77, 'teddy bear', 'yolov8'),
(78, 'hair drier', 'yolov8'),
(79, 'toothbrush', 'yolov8');


-- Now migrate the data from PhoneBbox to Detections
-- Each class becomes a separate row
INSERT INTO Detections (image_id, class_id, obj_no, bbox, conf, bbox_norm)
SELECT 
    image_id,
    67 as class_id,
    1 as obj_no,
    bbox_67 as bbox,
    conf_67 as conf,
    bbox_67_norm as bbox_norm
FROM PhoneBbox
WHERE bbox_67 IS NOT NULL

UNION ALL
;

INSERT INTO Detections (image_id, class_id, obj_no, bbox, conf, bbox_norm)
SELECT 
    image_id,
    63 as class_id,
    1 as obj_no,
    bbox_63 as bbox,
    conf_63 as conf,
    bbox_63_norm as bbox_norm
FROM PhoneBbox
WHERE bbox_63 IS NOT NULL

UNION ALL
;

INSERT INTO Detections (image_id, class_id, obj_no, bbox, conf, bbox_norm)
SELECT 
    image_id,
    26 as class_id,
    1 as obj_no,
    bbox_26 as bbox,
    conf_26 as conf,
    bbox_26_norm as bbox_norm
FROM PhoneBbox
WHERE bbox_26 IS NOT NULL

UNION ALL
;

INSERT INTO Detections (image_id, class_id, obj_no, bbox, conf, bbox_norm)
SELECT 
    image_id,
    27 as class_id,
    1 as obj_no,
    bbox_27 as bbox,
    conf_27 as conf,
    bbox_27_norm as bbox_norm
FROM PhoneBbox
WHERE bbox_27 IS NOT NULL

UNION ALL
;

INSERT INTO Detections (image_id, class_id, obj_no, bbox, conf, bbox_norm)
SELECT 
    image_id,
    32 as class_id,
    1 as obj_no,
    bbox_32 as bbox,
    conf_32 as conf,
    bbox_32_norm as bbox_norm
FROM PhoneBbox
WHERE bbox_32 IS NOT NULL;




SELECT sho.class_id, yc.class_name, COUNT(*)
FROM SegmentHelperObjectYOLO sho
JOIN YoloClasses yc ON yc.class_id = sho.class_id
GROUP BY sho.class_id
ORDER BY sho.class_id
;


SELECT i.site_name_id, i.imagename
FROM SegmentHelperObjectYOLO i
JOIN SegmentOct20 s on i.image_id = s.image_id
WHERE i.class_id = 73
LIMIT 10;

SELECT *
FROM SegmentHelperObjectYOLO sy
-- I want to delete all item from SegmentHelperObjectYOLO WHERE sy.image_id is not in SegmentBig_isface 
JOIN SegmentBig_isface sb ON sy.image_id = sb.image_id
WHERE sb.image_id IS NULL
LIMIT 10;

SELECT sy.*
FROM SegmentHelperObjectYOLO sy
WHERE NOT EXISTS (
SELECT 1 FROM SegmentBig_isface sb
WHERE sb.image_id = sy.image_id
)
LIMIT 10;


USE Stock;
ALTER TABLE Slogans DROP PRIMARY KEY;

ALTER TABLE Detections
ADD hue Float,
ADD sat Float,
ADD lum Float,
ADD val Float, 

ALTER TABLE Detections
ADD 	orientation INT,
ADD 	exclude TINYINT,
;


ALTER TABLE Detections
    ADD COLUMN cluster_id INT,
    ADD COLUMN meta_cluster_id INT,
    ADD CONSTRAINT fk_detections_cluster
        FOREIGN KEY (cluster_id) REFERENCES HSV(cluster_id);

INSERT INTO BsonFileLog (completed_bson_file) VALUES ('encodings_batch_7900001.bson');


INSERT INTO YoloClasses (class_id, class_name, model_version) VALUES
(80,'Sign','YoloCustom'),
(81,'Gift','YoloCustom'),
(82,'money','YoloCustom'),
(83,'Bag','YoloCustom'),
(84,'valentine','YoloCustom'),
(85,'Salad','YoloCustom'),
(86,'Dumbbell','YoloCustom'),
(87,'rose','YoloCustom'),
(88,'Groceries','YoloCustom'),
(89,'mask','other'),
(90,'Stethoscope','other'),
(91,'Gun','other'),
(92,'Headphones','other'),
(93,'Clipboard','other')
;