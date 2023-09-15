CREATE TABLE BagOfKeywords (
    image_id INT AUTO_INCREMENT PRIMARY KEY,
    age_id INT,
    gender_id INT,
    location_id INT,
    description VARCHAR(150),
    keyword_list BLOB,
    ethnicity_list BLOB,
    FOREIGN KEY (age_id) REFERENCES age(age_id),
    FOREIGN KEY (gender_id) REFERENCES gender(gender_id),
    FOREIGN KEY (location_id) REFERENCES location(location_id)
);
