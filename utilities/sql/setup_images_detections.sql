-- ============================================================================
-- Setup script for ImagesDetections table and Encodings modifications
-- ============================================================================

Use Stock;

-- 1. Add columns to Encodings table for face geometry
ALTER TABLE Encodings 
ADD COLUMN nose_x INT,
ADD COLUMN nose_y INT,
ADD COLUMN face_height INT;

SHOW VARIABLES LIKE 'local_infile';

-- export encoding_id, image_id, nose_x, nose_y, face_height from Encodings to migrate existing data to a different shard
SELECT encoding_id, image_id, nose_x, nose_y, face_height
FROM Encodings
WHERE nose_x IS NOT NULL OR nose_y IS NOT NULL OR face_height IS NOT NULL
LIMIT 10
;


-- import the data back to the Encodings table on the new shard (after copying the file to the new server)
LOAD DATA INFILE '/tmp/encodings_face_geometry.csv'
INTO TABLE Encodings
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
(encoding_id, image_id, nose_x, nose_y, face_height)
SET nose_x = NULLIF(nose_x, ''),
    nose_y = NULLIF(nose_y, ''),
    face_height = NULLIF(face_height, '')
;

-- 2. Recreate ImagesDetections table with current ObjectFusion schema
DROP TABLE IF EXISTS ImagesDetections;

CREATE TABLE ImagesDetections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_id INT NOT NULL UNIQUE,
    
    -- Left hand finger position (from body landmarks, normalized coordinates)
    left_pointer_x FLOAT,
    left_pointer_y FLOAT,
    left_source ENUM('body', 'default') DEFAULT 'default',
    
    -- Right hand finger position (from body landmarks, normalized coordinates)
    right_pointer_x FLOAT,
    right_pointer_y FLOAT,
    right_source ENUM('body', 'default') DEFAULT 'default',
    
    -- Object detection associations (foreign keys to Detections table)
    left_hand_object_id INT,
    right_hand_object_id INT,
    top_face_object_id INT,
    mouth_object_id INT,
    shoulder_object_id INT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for queries
    KEY (image_id),
    KEY (left_source),
    KEY (right_source),
    KEY (created_at),
    
    -- Foreign key constraints
    CONSTRAINT fk_img_det_image_id FOREIGN KEY (image_id) REFERENCES Images(image_id),
    CONSTRAINT fk_img_det_left_hand FOREIGN KEY (left_hand_object_id) REFERENCES Detections(detection_id),
    CONSTRAINT fk_img_det_right_hand FOREIGN KEY (right_hand_object_id) REFERENCES Detections(detection_id),
    CONSTRAINT fk_img_det_top_face FOREIGN KEY (top_face_object_id) REFERENCES Detections(detection_id),
    CONSTRAINT fk_img_det_mouth FOREIGN KEY (mouth_object_id) REFERENCES Detections(detection_id),
    CONSTRAINT fk_img_det_shoulder FOREIGN KEY (shoulder_object_id) REFERENCES Detections(detection_id)
);

-- Verify tables
SELECT 'ImagesDetections table created/verified' AS status;
DESCRIBE ImagesDetections;
