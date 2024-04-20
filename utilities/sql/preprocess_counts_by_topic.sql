Use Stock;



CREATE TABLE CountGender_Location_so (
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


DELETE 
FROM CountGender_Location_so;



SELECT l.nation_name, g.gender, COUNT(*) AS gender_count
FROM SegmentOct20 i
JOIN Location l ON i.location_id = l.location_id
JOIN Gender g ON i.gender_id = g.gender_id
GROUP BY l.nation_name, g.gender;



SELECT 
    l.nation_name AS country,
    SUM(CASE WHEN g.gender = 'men' THEN gender_count ELSE 0 END) AS men,
    SUM(CASE WHEN g.gender = 'none' THEN gender_count ELSE 0 END) AS none,
    SUM(CASE WHEN g.gender = 'oldmen' THEN gender_count ELSE 0 END) AS oldmen,
    SUM(CASE WHEN g.gender = 'oldwomen' THEN gender_count ELSE 0 END) AS oldwomen,
    SUM(CASE WHEN g.gender = 'nonbinary' THEN gender_count ELSE 0 END) AS nonbinary,
    SUM(CASE WHEN g.gender = 'other' THEN gender_count ELSE 0 END) AS other,
    SUM(CASE WHEN g.gender = 'trans' THEN gender_count ELSE 0 END) AS trans,
    SUM(CASE WHEN g.gender = 'women' THEN gender_count ELSE 0 END) AS women,
    SUM(CASE WHEN g.gender = 'youngmen' THEN gender_count ELSE 0 END) AS youngmen,
    SUM(CASE WHEN g.gender = 'youngwomen' THEN gender_count ELSE 0 END) AS youngwomen,
    SUM(CASE WHEN g.gender = 'both' THEN gender_count ELSE 0 END) AS bothg,
    SUM(CASE WHEN g.gender = 'intersex' THEN gender_count ELSE 0 END) AS intersex,
    SUM(CASE WHEN g.gender = 'androgynous' THEN gender_count ELSE 0 END) AS androgynous
FROM (
    SELECT l.nation_name, g.gender, COUNT(*) AS gender_count
    FROM SegmentOct20 i
    JOIN Location l ON i.location_id = l.location_id
    JOIN Gender g ON i.gender_id = g.gender_id
    GROUP BY l.nation_name, g.gender
) AS counts
GROUP BY l.nation_name;
