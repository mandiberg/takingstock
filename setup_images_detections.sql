-- ============================================================================
-- Setup script for ImagesDetections table and Encodings modifications
-- ============================================================================

Use Stock;

-- 1. Add columns to Encodings table for face geometry
ALTER TABLE Encodings 
ADD COLUMN IF NOT EXISTS nose_x INT,
ADD COLUMN IF NOT EXISTS nose_y INT,
ADD COLUMN IF NOT EXISTS face_height INT;

-- 2. Create ImagesDetections table
CREATE TABLE IF NOT EXISTS ImagesDetections (
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
    both_hands_object_id INT,
    left_hand_object_id INT,
    right_hand_object_id INT,
    top_face_object_id INT,
    bottom_face_object_id INT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for queries
    KEY (image_id),
    KEY (left_source),
    KEY (right_source),
    KEY (created_at),
    
    -- Foreign key constraints
    CONSTRAINT fk_img_det_image_id FOREIGN KEY (image_id) REFERENCES Images(image_id),
    CONSTRAINT fk_img_det_both_hands FOREIGN KEY (both_hands_object_id) REFERENCES Detections(detection_id),
    CONSTRAINT fk_img_det_left_hand FOREIGN KEY (left_hand_object_id) REFERENCES Detections(detection_id),
    CONSTRAINT fk_img_det_right_hand FOREIGN KEY (right_hand_object_id) REFERENCES Detections(detection_id),
    CONSTRAINT fk_img_det_top_face FOREIGN KEY (top_face_object_id) REFERENCES Detections(detection_id),
    CONSTRAINT fk_img_det_bottom_face FOREIGN KEY (bottom_face_object_id) REFERENCES Detections(detection_id)
);

-- Verify tables
SELECT 'ImagesDetections table created/verified' AS status;
DESCRIBE ImagesDetections;
