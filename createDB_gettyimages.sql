USE gettytest3;

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

CREATE TABLE Site (
    site_name_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    site_name varchar(20)
); 

CREATE TABLE Location (
    location_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    location_text varchar(50),
    location_number varchar(50),
    location_code varchar(50)
); 

CREATE TABLE ImagesTest (
    image_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    site_name_id INTEGER,
    FOREIGN KEY (site_name_id) REFERENCES Site (site_name_id),
    site_image_id varchar(50) NOT NULL,
    age_id INTEGER,
	FOREIGN KEY (age_id) REFERENCES Age (age_id),
    gender_id INTEGER,
    FOREIGN KEY (gender_id) REFERENCES Gender (gender_id),
    location_id INTEGER,
    FOREIGN KEY (location_id) REFERENCES Location (location_id),
    author varchar(100),
    caption varchar(150),
    contentUrl varchar(200) NOT NULL,
    description varchar(150),
    imagename varchar(100),
    uploadDate DATE
);

CREATE TABLE Keywords (
    keyword_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    keyword_number INTEGER,
    keyword_text varchar(50) NOT NULL, 
    keytype varchar(50), 
    weight INT,
    parent_keyword_id varchar(50), 
    parent_keyword_text varchar(50)

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
    face_x DECIMAL (5,2),
    face_y DECIMAL (5,2),
    face_z DECIMAL (5,2),
    mouth_gap DECIMAL (5,2),
    face_landmarks BLOB,
    face_encodings BLOB,
    body_landmarks BLOB
); 

-- calculations that will come later
CREATE TABLE Clusters (
    cluster_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    cluster_median JSON
);

-- This is the clusters junction table.
CREATE TABLE ImagesClusters (
    site_image_id INTEGER REFERENCES Images (site_image_id),
    cluster_id INTEGER REFERENCES Clusters (cluster_id),
    PRIMARY KEY (site_image_id)
);
