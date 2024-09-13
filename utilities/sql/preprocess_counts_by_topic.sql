Use Stock;



CREATE TABLE CountGender_Location (
    location_id INT,
    men INT DEFAULT 0,
    nogender INT DEFAULT 0,
    oldmen INT DEFAULT 0,
    oldwomen INT DEFAULT 0,
    nonbinary INT DEFAULT 0,
    other INT DEFAULT 0,
    trans INT DEFAULT 0,
    women INT DEFAULT 0,
    youngmen INT DEFAULT 0,
    youngwomen INT DEFAULT 0,
    manandwoman INT DEFAULT 0,
    intersex INT DEFAULT 0,
    androgynous INT DEFAULT 0,
    FOREIGN KEY (location_id) REFERENCES Location (location_id)
);

CREATE INDEX idx_topic_id ON Topics(topic_id);

CREATE TABLE CountGender_Topics_so (
    topic_id INT,
    men INT DEFAULT 0,
    nogender INT DEFAULT 0,
    oldmen INT DEFAULT 0,
    oldwomen INT DEFAULT 0,
    nonbinary INT DEFAULT 0,
    other INT DEFAULT 0,
    trans INT DEFAULT 0,
    women INT DEFAULT 0,
    youngmen INT DEFAULT 0,
    youngwomen INT DEFAULT 0,
    manandwoman INT DEFAULT 0,
    intersex INT DEFAULT 0,
    androgynous INT DEFAULT 0,
    FOREIGN KEY (topic_id) REFERENCES Topics (topic_id)
);



CREATE TABLE CountEthnicity_Location (
    location_id INT,
	POC INT DEFAULT 0,
	Black INT DEFAULT 0,
	caucasian INT DEFAULT 0,
	eastasian INT DEFAULT 0,
	hispaniclatino INT DEFAULT 0,
	middleeastern INT DEFAULT 0,
	mixedraceperson INT DEFAULT 0,
	nativeamericanfirstnations INT DEFAULT 0,
	pacificislander INT DEFAULT 0,
	southasian INT DEFAULT 0,
	southeastasian INT DEFAULT 0,
	afrolatinx INT DEFAULT 0,
	personofcolor INT DEFAULT 0,
	FOREIGN KEY (location_id) REFERENCES Location (location_id)
);

CREATE TABLE CountEthnicity_Topics_so (
    topic_id INT,
	POC INT DEFAULT 0,
	Black INT DEFAULT 0,
	caucasian INT DEFAULT 0,
	eastasian INT DEFAULT 0,
	hispaniclatino INT DEFAULT 0,
	middleeastern INT DEFAULT 0,
	mixedraceperson INT DEFAULT 0,
	nativeamericanfirstnations INT DEFAULT 0,
	pacificislander INT DEFAULT 0,
	southasian INT DEFAULT 0,
	southeastasian INT DEFAULT 0,
	afrolatinx INT DEFAULT 0,
	personofcolor INT DEFAULT 0,
    FOREIGN KEY (topic_id) REFERENCES Topics (topic_id)
);


DELETE 
FROM CountEthnicity_Location_so;



SELECT l.nation_name, g.gender, COUNT(*) AS gender_count
FROM SegmentOct20 i
JOIN Location l ON i.location_id = l.location_id
JOIN Gender g ON i.gender_id = g.gender_id
GROUP BY l.nation_name, g.gender;



SELECT 
    Location.location_id,
    Ethnicity.ethnicity_id,
    COUNT(*) AS ethnicity_count
FROM 
    Location
JOIN 
    SegmentOct20 ON Location.location_id = SegmentOct20.location_id
JOIN 
    ImagesEthnicity ON SegmentOct20.image_id = ImagesEthnicity.image_id
JOIN 
    Ethnicity ON ImagesEthnicity.ethnicity_id = Ethnicity.ethnicity_id
WHERE Location.location_id = 139
GROUP BY 
    Location.location_id, Ethnicity.ethnicity_id;


-- selects distinct_POC_count by location
SELECT 
	Location.location_id,
    COUNT(DISTINCT SegmentOct20.image_id) as distinct_POC_count
FROM 
    SegmentOct20
JOIN 
    Location ON Location.location_id = SegmentOct20.location_id
JOIN 
    ImagesEthnicity ON SegmentOct20.image_id = ImagesEthnicity.image_id
JOIN 
    Ethnicity ON ImagesEthnicity.ethnicity_id = Ethnicity.ethnicity_id
WHERE Location.location_id = 139
-- AND Ethnicity.ethnicity_id != 2
GROUP BY 
    Location.location_id
;

-- scratch
SELECT Topics.topic_id, Ethnicity.ethnicity_id, COUNT(*) AS ethnicity_count
FROM Topics
INNER JOIN ImagesTopics ON Topics.topic_id = ImagesTopics.topic_id
INNER JOIN SegmentOct20 ON ImagesTopics.image_id = SegmentOct20.image_id
INNER JOIN 
    ImagesEthnicity ON SegmentOct20.image_id = ImagesEthnicity.image_id
INNER JOIN 
    Ethnicity ON ImagesEthnicity.ethnicity_id = Ethnicity.ethnicity_id
WHERE Topics.topic_id = 7
GROUP BY Topics.topic_id, Ethnicity.ethnicity_id;


-- selects distinct_POC_count by topic
SELECT 
	Topics.topic_id,
    COUNT(DISTINCT SegmentOct20.image_id) as distinct_POC_count
FROM 
    SegmentOct20
JOIN ImagesTopics ON ImagesTopics.image_id = SegmentOct20.image_id 
JOIN Topics ON Topics.topic_id = ImagesTopics.topic_id
JOIN 
    ImagesEthnicity ON SegmentOct20.image_id = ImagesEthnicity.image_id
JOIN 
    Ethnicity ON ImagesEthnicity.ethnicity_id = Ethnicity.ethnicity_id
WHERE Topics.topic_id = 7
-- AND Ethnicity.ethnicity_id != 2
GROUP BY 
    Topics.topic_id
;


-- count of all items by location_id
SELECT 
	Location.location_id,
    COUNT(DISTINCT SegmentOct20.image_id) as total_count_inc_noeth
FROM 
    SegmentOct20
JOIN 
    Location ON Location.location_id = SegmentOct20.location_id
WHERE Location.location_id = 139
-- AND Ethnicity.ethnicity_id != 2
GROUP BY 
    Location.location_id
;




-- select eth count by topic - this works
SELECT Topics.topic_id, Ethnicity.ethnicity_id, COUNT(*) AS ethnicity_count
FROM Topics
INNER JOIN ImagesTopics ON Topics.topic_id = ImagesTopics.topic_id
INNER JOIN SegmentOct20 ON ImagesTopics.image_id = SegmentOct20.image_id
INNER JOIN 
    ImagesEthnicity ON SegmentOct20.image_id = ImagesEthnicity.image_id
INNER JOIN 
    Ethnicity ON ImagesEthnicity.ethnicity_id = Ethnicity.ethnicity_id
WHERE Topics.topic_id = 7
GROUP BY Topics.topic_id, Ethnicity.ethnicity_id;


-- select ethnicity count by topic
SELECT Topics.topic_id, Gender.gender, COUNT(*) AS gender_count
FROM Topics
INNER JOIN ImagesTopics ON Topics.topic_id = ImagesTopics.topic_id
INNER JOIN SegmentOct20 ON ImagesTopics.image_id = SegmentOct20.image_id
INNER JOIN Gender ON SegmentOct20.gender_id = Gender.gender_id
WHERE Topics.topic_id = 7
GROUP BY Topics.topic_id, Gender.gender;
