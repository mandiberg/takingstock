USE minitest;

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

CREATE TABLE Topics (
	topic_id INT,
	topic VARCHAR(250)
);

CREATE TABLE ImagesTopics (
	image_id INT,
    topic_id INT,
    topic_score FLOAT
);


SELECT k.keyword_text  
FROM ImagesKeywords ik
JOIN Keywords k ON ik.keyword_id = k.keyword_id 
WHERE ik.image_id = 51148251


SELECT * 
FROM BagOfKeywords bok 
-- JOIN Images i ON ik.image_id = i.image_id 
WHERE bok.image_id = 51148251