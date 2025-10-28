USE Stock;

-- These remove repeated values and reduce repetition

CREATE TABLE Ethnicity (
    ethnicity_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    ethnicity varchar(40)
); 


-- These remove repeated values and reduce repetition
CREATE TABLE Gender (
    gender_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    gender varchar(20)
); 

CREATE TABLE Age (
    age_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    age varchar(20)
); 

CREATE TABLE AgeDetail (
    age_detail_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    age_detail varchar(20)
); 

CREATE TABLE Site (
    site_name_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    site_name varchar(20)
); 

CREATE TABLE Model_Release (
    release_name_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    release_name varchar(20)
); 

CREATE TABLE Location (
    location_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    location_number_getty INTEGER,
    getty_name varchar(70),
    nation_name varchar(70),
    nation_name_alpha varchar(70),
    official_nation_name varchar(150),
    sovereignty varchar(70),
    code_alpha2 varchar(70),
    code_alpha3 varchar(70),
    code_numeric INTEGER,
    code_iso varchar(70),
    population INTEGER,
    region varchar(50),
    subregion varchar(50),
    intermediateregion varchar(50),
    WarsawPact varchar(50)
); 


-- location_number_getty,getty_name,nation_name,nation_name_alpha,official_nation_name,sovereignty,code_alpha2,code_alpha3,code_numeric,code_iso,population,region,subregion,intermediateregion,WarsawPact
CREATE TABLE Images (
    image_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    site_name_id INTEGER,
    FOREIGN KEY (site_name_id) REFERENCES Site (site_name_id),
    site_image_id varchar(50) NOT NULL,
    age_id INTEGER,
	FOREIGN KEY (age_id) REFERENCES Age (age_id),
    age_detail_id INTEGER,
	FOREIGN KEY (age_detail_id) REFERENCES AgeDetail (age_detail_id),
    gender_id INTEGER,
    FOREIGN KEY (gender_id) REFERENCES Gender (gender_id),
    location_id INTEGER,
    FOREIGN KEY (location_id) REFERENCES Location (location_id),
    author varchar(100),
    caption varchar(150),
    contentUrl varchar(300) NOT NULL,
    description varchar(150),
    imagename varchar(200),
    uploadDate DATE,
    release_name_id INTEGER,
    FOREIGN KEY (release_name_id) REFERENCES Model_Release (release_name_id),
    h INTEGER,
    w INTEGER,
    no_image boolean
);

CREATE TABLE WanderingImages (
    wandering_image_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    wandering_name_site_id varchar(50) NOT NULL,
    site_image_id varchar(50),
    site_name_id int,
    FOREIGN KEY (site_name_id) REFERENCES Site (site_name_id),
    UNIQUE (wandering_name_site_id),
    INDEX idx_wandering_name_site_id (wandering_name_site_id)
    );

CREATE TABLE NMLImages (
    nml_id INT AUTO_INCREMENT PRIMARY KEY,
    image_id INT NOT NULL UNIQUE,
    is_nml_db BOOLEAN,
    FOREIGN KEY (image_id) REFERENCES images(image_id)
);

CREATE TABLE Allmaps (
    Allmaps_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    site_image_id varchar(50) NOT NULL,
    filename varchar(150),
    uploadDate DATE,
    INDEX idx_site_image_id (site_image_id)
);


CREATE TABLE Keywords (
    keyword_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    keyword_number INTEGER,
    keyword_text varchar(50) NOT NULL, 
    keytype varchar(50), 
    weight INT,
    parent_keyword_id varchar(50), 
    parent_keyword_text varchar(50),
    INDEX idx_keyword_text (keyword_text)
    

);

-- This is the junction table.
CREATE TABLE ImagesKeywords (
    image_id int REFERENCES Images (image_id),
    keyword_id int REFERENCES Keywords (keyword_id),
    PRIMARY KEY (image_id, keyword_id)
);

-- This is the ethnicity semijunction table.
CREATE TABLE ImagesEthnicity (
    image_id int REFERENCES Images (image_id),
    ethnicity_id int REFERENCES Ethnicity (ethnicity_id),
    PRIMARY KEY (image_id, ethnicity_id)
);

-- Store new calculated data
CREATE TABLE Encodings (
    encoding_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    image_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES Images(image_id),
    is_face boolean,
    is_body boolean,
    is_face_distant boolean,
    face_x DECIMAL (6,3),
    face_y DECIMAL (6,3),
    face_z DECIMAL (6,3),
    mouth_gap DECIMAL (6,3),
    face_landmarks BLOB,
    bbox JSON,
    face_encodings68 BLOB,
    body_landmarks BLOB,
    mongo_encodings boolean,
    mongo_body_landmarks boolean,
    mongo_face_landmarks boolean,
    is_small boolean,
    mongo_body_landmarks_norm boolean,
    two_noses boolean,
    is_dupe_of INTEGER,
    FOREIGN KEY (is_dupe_of) REFERENCES Images(image_id),
    mongo_hand_landmarks boolean,
    mongo_hand_landmarks_norm boolean,
    is_face_no_lms boolean,
    is_feet boolean,
    mongo_body_landmarks_3D boolean,
    is_hand_left boolean,
    is_hand_right boolean,
    UNIQUE (image_id)
); 


-- calculations that will come later
CREATE TABLE Clusters (
    cluster_id int NOT NULL PRIMARY KEY,
    cluster_median BLOB
);

-- This is the clusters junction table.
CREATE TABLE ImagesClusters (
    image_id INTEGER REFERENCES Images (image_id),
    cluster_id INTEGER REFERENCES Clusters (cluster_id),
    cluster_dist FLOAT DEFAULT NULL,
    PRIMARY KEY (image_id)
);


CREATE TABLE Poses (
    cluster_id int NOT NULL PRIMARY KEY,
    cluster_median BLOB
);

-- This is the poses junction table.
CREATE TABLE ImagesPoses (
    image_id INTEGER REFERENCES Images (image_id),
    cluster_id INTEGER REFERENCES Poses (cluster_id),
    cluster_dist FLOAT DEFAULT NULL,
    PRIMARY KEY (image_id)
);


CREATE TABLE ImagesBackground (
    image_id INT PRIMARY KEY,
    hue FLOAT,
    lum FLOAT,
    sat FLOAT,
    val FLOAT,
    torso_lum FLOAT,
    hue_bb FLOAT,
    lum_bb FLOAT,
    sat_bb FLOAT,
    val_bb FLOAT,
    torso_lum_bb FLOAT,
    selfie_bbox JSON,
    is_left_shoulder boolean,
    is_right_shoulder boolean,
    FOREIGN KEY (image_id) REFERENCES images(image_id)
);

CREATE TABLE PhoneBbox (
    image_id INT,
    bbox_67 JSON,
    conf_67 Float,
    bbox_63 JSON,
    conf_63 Float,
    bbox_26 JSON,
    conf_26 Float,
    bbox_27 JSON,
    conf_27 Float,
    bbox_32 JSON,
    conf_32 Float,
    bbox_67_norm JSON,
    bbox_63_norm JSON,
    bbox_26_norm JSON,
    bbox_27_norm JSON,
    bbox_32_norm JSON

    PRIMARY KEY (image_id),
    FOREIGN KEY (image_id) REFERENCES images(image_id)
);

CREATE TABLE Counters (
    counter_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    counter_name varchar(50),
    counter_value int
);


CREATE TABLE LocationHandsFeet (
    image_id INT PRIMARY KEY,
    left_hand_x FLOAT,
    left_hand_y FLOAT,
    right_hand_x FLOAT,
    right_hand_y FLOAT,
    left_foot_x FLOAT,
    left_foot_y FLOAT,
    right_foot_x FLOAT,
    right_foot_y FLOAT
);


-- This is the junction table.
CREATE TABLE IsNotDupeOf (
    image_id_i int REFERENCES Images (image_id),
    image_id_j int REFERENCES Encodings (encoding_id),
    PRIMARY KEY (image_id_i, image_id_j)
);


-- calculations that will come later
CREATE TABLE HSV (
    cluster_id int NOT NULL PRIMARY KEY,
    cluster_median BLOB
);

-- This is the clusters junction table.
CREATE TABLE ImagesHSV (
    image_id INTEGER REFERENCES Images (image_id),
    cluster_id INTEGER REFERENCES HSV (cluster_id),
    cluster_dist FLOAT DEFAULT NULL,
    PRIMARY KEY (image_id)
);

