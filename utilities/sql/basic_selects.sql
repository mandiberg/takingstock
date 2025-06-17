-- 86.3gb, 2.46tb





USE stock;
SET GLOBAL innodb_buffer_pool_size=8053063680;



UPDATE Images
SET w = NULL
WHERE image_id < 5000;

DELETE 
FROM CountGender_Topics
;

SELECT COUNT(i.image_id) 
FROM Images i
JOIN Encodings e ON e.image_id = i.image_id 
-- WHERE i.site_image_id = "___638KPrmA"
WHERE i.site_name_id in (6,12)
AND e.is_face = 1
    AND e.face_x > -40 AND e.face_x < -3
    AND e.face_y > -4 AND e.face_y < 4
    AND e.face_z > -4 AND e.face_z < 4
   	AND NOT i.age_id <= 3)

;

SELECT COUNT(e.image_id)
FROM Encodings e 
LEFT JOIN 
    SegmentBig_isface sb 
ON 
    e.image_id = sb.image_id
WHERE 
    sb.image_id IS NULL
    AND e.image_id IS NOT NULL
    AND e.face_x > -45
    AND e.face_x < -20
    AND e.face_y > -10
    AND e.face_y < 10
    AND e.face_z > -10
    AND e.face_z < 10
;

SELECT MAX(sbi.image_id)
FROM SegmentBig_isface sbi 
WHERE sbi.mongo_tokens IS NOT NULL
;

-- 44374493

-- all topics
SELECT i.site_name_id, i.imagename
FROM Images i 
JOIN encodings e ON i.image_id = e.image_id
WHERE i.image_id in (303051, 399765)
AND (e.is_face = 1 OR e.is_body = 1)
;


-- topic only
SELECT i.site_name_id, i.imagename
FROM Images i 
JOIN encodings 	e ON i.image_id = e.image_id
JOIN ImagesTopics it ON it.image_id = e.image_id
WHERE i.image_id in (4280, 7525, 8886)

AND (e.is_face = 1 OR e.is_body = 1)
AND it.topic_id = 32
;

-- topic only, bodies as t distance
SELECT i.site_name_id, i.imagename
FROM Images i 
JOIN encodings 	e ON i.image_id = e.image_id
JOIN ImagesTopics it ON it.image_id = e.image_id
WHERE i.site_name_id = 1
AND e.is_face = 1 AND e.is_body = 1
AND it.topic_id = 32
LIMIT 5000


SELECT *
FROM Encodings i 
WHERE i.image_id = 31678563
;

SELECT COUNT(*)
FROM SegmentOct20 so
JOIN ImagesTopics it ON it.image_id = so.image_id
JOIN ImagesHandsPositions ih ON ih.image_id = so.image_id
JOIN ImagesHandsGestures ig ON ig.image_id = so.image_id
WHERE ih.cluster_id = 13
AND ig.cluster_id  = 2
AND so.face_x > -33 AND so.face_x < -27 AND so.face_y > -2 AND so.face_y < 2 AND so.face_z > -2 AND so.face_z < 2
;


SELECT *
FROM WanderingImages wi 
WHERE wi.wandering_image_id > 6000000
LIMIT 5000
;

WHERE wi.site_image_id = 1032126784
AND wi.site_name_id = 3
;

SELECT *
FROM Encodings
WHERE image_id = 893
;


SELECT *
FROM Images i 
WHERE i.site_name_id = 3
AND i.site_image_id = 1973520
;


SELECT DISTINCT(s.image_id), s.site_name_id, s.contentUrl, s.imagename, s.description, s.face_x, s.face_y, s.face_z, s.mouth_gap, s.bbox, s.site_image_id, it.topic_score, ibg.lum, ibg.lum_bb, ibg.hue, ibg.hue_bb, ibg.sat, ibg.sat_bb, ibg.val, ibg.val_bb, ibg.lum_torso, ibg.lum_torso_bb  
FROM SegmentBig_isface s  JOIN Encodings e ON s.image_id = e.image_id  JOIN ImagesHandsPositions ihp ON s.image_id = ihp.image_id  
JOIN ImagesHandsGestures ih ON s.image_id = ih.image_id  JOIN ImagesTopics it ON s.image_id = it.image_id  
JOIN SegmentHelper_may2025_4x4faces sh ON s.image_id = sh.image_id  JOIN ImagesBackground ibg ON s.image_id = ibg.image_id  
WHERE  e.is_dupe_of IS NULL  AND s.face_x > -50  AND it.topic_score > .1   AND ihp.cluster_id = 24  AND ih.cluster_id = 112 AND it.topic_id IN (35)  

LIMIT 2500;

UPDATE SegmentBig_isface AS seg
JOIN Images AS img ON seg.image_id = img.image_id
SET seg.contentUrl = img.contentUrl, seg.imagename = img.imagename, seg.description = img.description, seg.site_image_id = img.site_image_id
WHERE seg.contentUrl IS NULL
AND img.contentUrl IS NOT NULL;


CREATE TABLE SegmentHelper_may2025_4x4faces (
    seg_image_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    image_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES Images(image_id)
);

SELECT COUNT(*)
FROM Encodings e
WHERE e.is_face IS TRUE
OR e.is_body IS TRUE
OR e.is_face_distant 
OR e.is_face_no_lms 
;


SELECT MAX(sbi.image_id)
FROM SegmentBig_isface sbi 
WHERE sbi.mongo_tokens_affect = 1
;

CREATE TABLE WanderingImages (
    wandering_image_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    site_image_id INTEGER,
    FOREIGN KEY (site_image_id) REFERENCES Site (site_image_id)
    site_name_id INTEGER,
    FOREIGN KEY (site_name_id) REFERENCES Site (site_name_id)
);



SELECT * 
FROM SegmentBig_isface e 
Where e.image_id = 92362382
;

SELECT * 
FROM Images e 
Where e.image_id = 92362382
;




SELECT *
FROM Images i
WHERE i.site_image_id = 1058475875
AND i.site_name_id = 2
;


SELECT COUNT(image_id)
FROM imagestopics_ALLgetty4faces_isfacemodel

;

ALTER TABLE encodings
RENAME COLUMN is_not_face_xyz TO is_face_no_lms
;

ALTER TABLE Encodings 
ADD is_bodyhand_left boolean, 
ADD is_bodyhand_right boolean

;


SELECT COUNT(*)
FROM Encodings
WHERE mongo_body_landmarks_3D  = 1
;

UPDATE Images
SET mongo_body_landmarks_3D  = 1, mongo_hand_landmarks = 0
WHERE image_id IN ()
;


SELECT i.is_hand_left, i.is_hand_right
FROM SegmentOct20 i
WHERE i.image_id = 54116622
;

SELECT *
FROM Images i
JOIN Encodings e ON i.image_id = e.image_id
WHERE i.site_name_id = 1
AND e.is_face = 0
AND e.two_noses is NULL AND i.no_image IS NULL
AND e.mongo_encodings =1
LIMIT 100;



AND i.image_id > 370000
AND i.image_id < 400000


SELECT DISTINCT i.image_id, e.encoding_id, e.is_face, e.face_landmarks,
e.bbox 
FROM Images i 
LEFT JOIN Encodings e ON i.image_id = e.image_id 
WHERE e.encoding_id IS NOT NULL AND e.is_face = 0 AND e.mongo_encodings is NULL AND i.site_name_id = 1 AND i.no_image IS NULL 
LIMIT 10;


-- testing imagestopics select
SELECT DISTINCT i.image_id, i.site_name_id, i.contentUrl, i.imagename, e.encoding_id, i.site_image_id, e.face_landmarks, e.bbox 
FROM Images i LEFT JOIN Encodings e ON i.image_id = e.image_id 
LEFT JOIN ImagesTopics_isnotface it ON i.image_id = it.image_id 
WHERE e.encoding_id IS NOT NULL AND e.is_face = 0 AND e.mongo_encodings is NULL AND e.two_noses is NULL AND i.no_image IS NULL AND i.site_name_id = 1 
AND it.topic_id IN (16, 17, 18, 23, 24, 45, 53)  AND i.no_image IS NULL 
LIMIT 10;

SELECT DISTINCT i.image_id, i.site_name_id, i.contentUrl, i.imagename, e.encoding_id, i.site_image_id, e.face_landmarks, e.bbox 
FROM Images i LEFT JOIN Encodings e ON i.image_id = e.image_id 
LEFT JOIN ImagesTopics_isnotface it ON i.image_id = it.image_id 
WHERE e.encoding_id IS NOT NULL AND e.is_face = 0 AND e.mongo_encodings is NULL AND e.two_noses is NULL AND i.no_image IS NULL 
AND i.site_name_id = 1
AND it.topic_id IN (16, 17, 18, 23, 24, 45, 53)  AND i.no_image IS NULL 
LIMIT 10;


SELECT *
FROM Images i LEFT JOIN Encodings e ON i.image_id = e.image_id 
LEFT JOIN ImagesTopics_isnotface it ON i.image_id = it.image_id 
-- WHERE e.encoding_id IS NOT NULL AND e.is_face = 0 AND e.mongo_encodings is NULL AND e.two_noses is NULL AND i.no_image IS NULL 
 WHERE e.is_face =0 AND e.mongo_encodings =1
 AND e.two_noses is NULL AND i.no_image IS NULL

AND i.site_name_id = 1 
LIMIT 10;


SELECT DISTINCT i.image_id, i.site_name_id, i.contentUrl, i.imagename, e.encoding_id, i.site_image_id, e.face_landmarks, e.bbox 
FROM Images i LEFT JOIN Encodings e ON i.image_id = e.image_id LEFT JOIN ImagesTopics_isnotface it ON i.image_id = it.image_id 
WHERE e.encoding_id IS NOT NULL AND e.is_face = 0 AND e.mongo_encodings is NULL AND e.two_noses is NULL AND i.no_image IS NULL AND i.site_name_id = 2 
AND it.topic_id IN (16, 17, 18, 23, 24, 45, 53)  AND i.no_image IS NULL 
LIMIT 10;



SELECT COUNT(i.image_id)
FROM Images i LEFT JOIN Encodings e ON i.image_id = e.image_id LEFT JOIN ImagesTopics_isnotface it ON i.image_id = it.image_id 
WHERE e.is_face = 0 AND e.mongo_encodings IS NULL AND i.site_name_id = 2
AND it.topic_id IN (16, 17, 18, 23, 24, 45, 53)  AND i.no_image IS NULL 
;

SELECT COUNT(image_id)
FROM SegmentBig_isnotface
WHERE site_name_id = 1
;

SELECT COUNT(i.image_id)
FROM Images i LEFT JOIN Encodings e ON i.image_id = e.image_id
-- JOIN ImagesTopics_isnotface it ON i.image_id = it.image_id
 WHERE e.mongo_face_landmarks = 0 and e.mongo_body_landmarks =1
 AND e.two_noses is NULL AND i.no_image IS NULL
-- WHERE e.encoding_id IS NULL AND i.no_image IS NOT NULL AND i.site_name_id = 1
-- WHERE e.encoding_id IS NOT NULL AND e.is_face = 0 AND e.mongo_encodings = 0  AND e.mongo_face_landmarks =0 AND e.is_face_no_lms =0  AND i.site_name_id = 2
-- WHERE e.encoding_id IS NOT NULL AND e.is_face = 0 AND e.mongo_encodings IS NULL  AND e.mongo_face_landmarks IS NULL AND e.is_face_no_lms IS NULL  AND i.site_name_id = 2
-- WHERE e.encoding_id IS NOT NULL AND e.bbox IS NULL AND e.mongo_encodings =1 AND e.is_body IS NULL AND  e.two_noses is NULL AND i.no_image IS NULL AND i.site_name_id = 2
AND i.site_name_id = 1
;




SELECT *
FROM Images i LEFT JOIN Encodings e ON i.image_id = e.image_id
-- WHERE e.is_face IS NULL AND e.mongo_encodings IS NULL 
-- WHERE e.encoding_id IS NULL AND i.no_image IS NOT NULL AND i.site_name_id = 1
-- WHERE e.encoding_id IS NOT NULL AND e.is_face = 0 AND e.mongo_encodings IS NULL AND e.mongo_face_landmarks IS NULL AND e.is_face_no_lms IS NULL AND i.site_name_id = 1
WHERE e.encoding_id IS NOT NULL AND e.is_face = 0 AND e.mongo_encodings = 0  AND e.mongo_face_landmarks =0 AND e.is_face_no_lms =0  AND i.site_name_id = 1
AND e.is_body IS NULL
LIMIT 100
;


SELECT DISTINCT i.image_id, i.site_name_id, i.contentUrl, i.imagename, e.encoding_id, i.site_image_id, e.face_landmarks, e.bbox 
FROM Images i LEFT JOIN Encodings e ON i.image_id = e.image_id 
WHERE e.encoding_id IS NOT NULL AND e.is_face = 0 AND e.mongo_encodings is NULL 
AND e.two_noses is NULL AND i.no_image IS NULL AND i.site_name_id = 1  
LIMIT 10;



SELECT DISTINCT i.image_id, i.site_name_id, i.contentUrl, i.imagename, e.encoding_id, i.site_image_id, e.face_landmarks, e.bbox 
FROM Images i LEFT JOIN Encodings e ON i.image_id = e.image_id 
WHERE e.encoding_id IS NOT NULL AND e.is_face = 0 AND e.mongo_encodings is NULL AND e.two_noses is NULL AND i.no_image IS NULL 
AND i.site_name_id = 1 AND i.no_image IS NULL LIMIT 10;


UPDATE Encodings e
LEFT JOIN Images i ON i.image_id = e.image_id
SET mongo_body_landmarks = NULL, mongo_body_landmarks_norm = NULL
WHERE e.image_id IN (138523797, 138535951)
;

UPDATE Encodings e
SET e.is_face = NULL, e.mongo_encodings = NULL,  e.mongo_face_landmarks = NULL 
WHERE e.image_id IN (92371199, 85691026)
;

UPDATE Images i
LEFT JOIN Encodings e ON i.image_id = e.image_id
SET i.no_image = NULL
WHERE i.no_image IS NOT NULL 
AND i.imagename LIKE '%F/FE/%'
AND (e.is_face != 1 OR e.is_body != 1)
;


UPDATE SegmentBig_isnotface i
SET i.no_image = NULL
WHERE i.no_image IS NOT NULL 
AND i.image_id IN ()
;


SELECT COUNT(*)
FROM Images i 
JOIN encodings e ON i.image_id = e.image_id 
WHERE i.no_image IS NOT NULL 
AND i.imagename LIKE '%F/FE/%'
AND (e.is_face != 1 OR e.is_body != 1)
;


SELECT COUNT(*)
FROM Images i 
JOIN encodings e ON i.image_id = e.image_id 
WHERE i.no_image IS NOT NULL 
AND i.imagename LIKE '%F/FE/%'
AND (e.is_face != 1 OR e.is_body != 1)
;


SELECT *
FROM encodings e 
WHERE e.image_id = 110670144
;

SELECT *
FROM images sbi 
WHERE sbi.image_id = 103671377
;

SELECT COUNT(e.encoding_id)
FROM encodings
WHERE e.is_face_no_lms IS NOT NULL
;


SELECT MAX(image_id)
FROM encodings
WHERE is_face_no_lms IS NOT NULL
;



118204-117598
2647

606/56*60

SELECT COUNT(image_id) 
FROM SegmentBig_isnotface WHERE  mongo_tokens IS NOT NULL AND image_id NOT IN (SELECT image_id FROM imagestopics_isnotface) AND image_id > 94292000 
;


SELECT *
FROM ImagesTopics_isnotface it
WHERE it.image_id = 84856634
;


SELECT COUNT(sbi.image_id) 
FROM SegmentBig_isnotface sbi 
WHERE sbi.site_name_id = 1
;

SELECT COUNT(i.image_id) 
FROM Images i 
LEFT OUTER JOIN Encodings e ON i.image_id = e.image_id
JOIN SegmentBig_isnotface sbin ON i.image_id = sbin.image_id
JOIN SegmentBig_isface sb ON i.image_id = sb.image_id
WHERE i.site_name_id = 1
AND e.image_id IS NULL
-- AND sbin.image_id IS NULL
;



SELECT *
FROM SegmentBig_isnotface sbi
WHERE sbi.image_id = 118776377
;

SELECT COUNT(i.image_id)
FROM Segment i 
;

UPDATE Images i 
SET i.no_image = NULL
WHERE i.image_id < 400000
AND i.no_image = 1
AND i.site_name_id = 1
;


SELECT COUNT(i.image_id)
FROM Images i
WHERE i.no_image = 1
;


UPDATE SegmentBig_isnotface
SET    mongo_tokens = NULL
WHERE  mongo_tokens = 1;

SELECT *
FROM Encodings e
WHERE e.image_id = 120218167
;

SELECT *
FROM Images i
WHERE i.image_id = 120040598
;


SELECT MAX(e.encoding_id)
FROM Encodings e
;

-- max 125129808

-- 125129851

-- 125116988

SELECT *
FROM ImagesKeywords ik 
WHERE ik.image_id = 1881
LIMIT 10
;


SELECT COUNT(ib.image_id)
FROM ImagesBackground ib 
JOIN ImagesTopics it ON ib.image_id = it.image_id 
WHERE it.topic_id = 23
AND ib.hue IS NULL

;

SELECT COUNT(ib.image_id)
FROM ImagesBackground ib 
WHERE ib.hue IS NULL

;




SELECT COUNT(*)
FROM SegmentBig_isface
WHERE image_id > 103748034
AND imagename IS NULL
AND location_id IS NOT NULL
;

-- 2658545

SELECT COUNT(*)
FROM SegmentOct20 so 
WHERE so.contentUrl LIKE '%wasja%'
AND description LIKE 'Young woman with%'
;

SELECT COUNT(*)
FROM SegmentOct20 so 
WHERE so.mongo_body_landmarks_norm IS NULL
AND so.mongo_body_landmarks  = 1
;

SELECT *
FROM ImagesPoses128 ip 
WHERE ip.image_id = 894
LIMIT 10
;

SELECT ip.image_id FROM ImagesPoses128 ip WHERE ip.cluster_id = 6
LIMIT 10
;

SELECT DISTINCT seg1.image_id, seg1.site_name_id, seg1.contentUrl, seg1.imagename, seg1.site_image_id, seg1.mongo_body_landmarks, seg1.mongo_face_landmarks, seg1.bbox 
FROM SegmentOct20 seg1 
WHERE  seg1.mongo_hand_landmarks IS NULL and seg1.no_image IS NULL AND seg1.image_id > 592091  
AND seg1.image_id 
IN (SELECT seg1.image_id FROM ImagesPoses128 ip WHERE ip.cluster_id = 6) LIMIT 10000;

-- 105617677 

-- calculations that will come later
CREATE TABLE HandsGestures (
    cluster_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    cluster_median BLOB
);

-- This is the clusters junction table.
CREATE TABLE ImagesHandsGestures (
    image_id INTEGER REFERENCES Images (image_id),
    cluster_id INTEGER REFERENCES HandsGestures (cluster_id),
    PRIMARY KEY (image_id)
);


DELETE 
FROM ImagesHandsPositions 
;

DELETE 
FROM HandsPositions 
;

INSERT INTO Clusters (cluster_id, cluster_median) VALUES (0, 'TK');

SELECT COUNT(image_id)
FROM ImagesHandsGestures
;


DELETE 
FROM BodyPoses ibp
WHERE ibp.cluster_id = 32
;

CREATE TABLE FingertipsPositions (
    cluster_id int NOT NULL PRIMARY KEY,
    cluster_median BLOB
);

-- This is the poses junction table.
-- need to change the cluster_id reference to the correct cluster table
CREATE TABLE ImagesFingertipsPositions (
    image_id INTEGER REFERENCES Images (image_id),
    cluster_id INTEGER REFERENCES FingertipsPositions (cluster_id),
    PRIMARY KEY (image_id)
);


SELECT COUNT(s.image_id)
FROM SegmentOct20 s  JOIN ImagesHandsPositions ihp ON s.image_id = ihp.image_id  
JOIN ImagesHandsGestures ih ON s.image_id = ih.image_id  JOIN ImagesTopics it ON s.image_id = it.image_id  
JOIN ImagesBackground ibg ON s.image_id = ibg.image_id  WHERE  s.is_dupe_of IS NULL  AND s.face_x > -50  AND it.topic_score > .3 
AND s.age_id NOT IN (1,2,3)  AND s.mongo_body_landmarks = 1     
AND ihp.cluster_id = 1 AND ih.cluster_id = 1  AND it.topic_id IN (32)  LIMIT 25000;


SELECT ihp.image_id, it.topic_id , ihp.cluster_id , ihg.cluster_id 
FROM ImagesHandsPositions ihp
LEFT JOIN ImagesTopics it ON it.image_id = ihp.image_id 
LEFT JOIN ImagesHandsGestures ihg ON it.image_id = ihg.image_id 
WHERE ihp.image_id in (87529894, 93061378, 82835605, 28032927, 52675622, 12667760, 124133068, 10110887, 8928417, 6287086, 95856693, 100275516, 13028142, 100140349, 84532882, 36437265, 11071088, 11428941, 119870634, 48494496, 78354342, 53975177, 35790451, 107845006, 100149157, 21847543, 99105685, 104238922, 126870708, 93373843, 93011321, 94127100, 16522939, 5940462, 83295836, 98375165, 128974552, 122213062, 127679497, 122547629, 81126515, 36754837, 86529937, 124369013, 11707409, 81126614, 99020756, 6348771, 10669616, 90565588, 85032388, 14789521, 127240794, 10677522, 36327670, 118193223, 126925623, 9178170, 84628149, 65120089, 78066606, 84532432, 102511159, 100703818, 83692874)

;



SELECT *
FROM Encodings so
WHERE so.image_id = 37859699
;

SELECT DISTINCT(ic.image_id)
FROM ImagesHandsGestures ic 
JOIN SegmentOct20 so ON so.image_id = ic.image_id
WHERE ic.cluster_dist IS NULL
AND so.is_dupe_of IS NULL
LIMIT 100
;

SELECT *
FROM ImagesHandsGestures
WHERE image_id = 48540227
;


SELECT DISTINCT(s.image_id)
FROM SegmentOct20 s LEFT JOIN ImagesHandsGestures ic ON s.image_id = ic.image_id 
INNER JOIN HandsGestures c ON c.cluster_id = ic.cluster_id 
WHERE  s.mongo_hand_landmarks = 1  AND s.is_dupe_of IS NULL  
AND ic.cluster_id IS NOT NULL  
LIMIT 100;

-- pink phone
-- WHERE it.image_id in (108857144, 108830407, 112545449, 126358530, 113558254, 92480518, 93677974, 126321229, 88707614, 94210529, 107334555, 5894645, 95554150, 108461820, 108231912, 113879167, 108462112, 108438278, 126816627, 14684516, 108768428, 108414305, 85238953, 15879156, 102126993, 113667355, 108550308, 5547807, 91785683, 107975455, 6104844, 128285728, 102023157, 108580711, 107937260, 95085931, 108608080, 94274491, 15341047, 88704677, 105542090, 88648757, 125296392, 125236278, 108066547, 88986500, 113077848, 88039265, 101352824, 111730105, 2917794, 129819016, 94939452, 102036402, 113442765, 94285887, 108433608, 85239073, 125043303, 94911083, 108905580, 85240834, 108658101, 112780562, 111809220, 15887453, 114164649, 126820728, 127771732)

-- black phone
-- 36385216, 99351971, 99590487, 6620894, 33421861, 125934571, 91799295, 88091734, 127525434, 6108709, 127517966, 126156286, 14471970, 6604497, 3572053, 105732084, 10280172, 126934802, 12971706, 88024637, 109855326, 88794089, 124673922, 127205916, 93723102, 107680813, 108902249, 108681716, 88745785, 108081589, 16501227, 93363578, 5912816, 108834843, 93419717, 128930119, 108384364, 124462054, 107883780, 101321205, 108427434, 105611429, 124294409, 82521526, 88092677, 127382768, 87526294, 124883472, 105580830, 99475890, 3361427, 103102658, 113852419, 14765179, 86657999, 3069468, 129634624, 127664457, 126678866, 127669082, 89137977, 93656572, 105842999, 125194442, 84318559, 97839001, 88151044, 95179136, 10826663, 109052876, 15948916, 

-- silver phone
-- 127834582, 6042616, 13403185, 105687209, 86620460, 126099058, 103727791, 4912655, 124638455, 126935345, 128202648, 125309846, 108809166, 124982984, 128605382, 11267329, 111872629, 88017822, 6420695, 124624114, 10157591, 94186715, 125483138, 111978050, 5916876, 11514002, 6179862, 96918051, 108434992, 93402824, 6159522, 83750235, 92188983, 102090810, 88990754, 14836391, 109079909, 87034115, 92267437, 10500284, 16335244, 125430109, 16525269, 105557297, 92401228, 88709354, 128049664, 107336700, 105337072, 15980078, 124790664, 85738966, 93019538, 94987089, 5832389, 92582619, 123982400, 127819014, 88979823, 108054204, 125795443, 97695620, 3312964, 107767403, 

-- silence
-- 120818645, 50329705, 11127833, 34078483, 57489753, 123786553, 11043607, 16072962, 124205719, 92698052, 124202610, 42092375, 123886204, 6948641, 8448649, 68932117, 125909693, 19322835, 121296516, 101984848, 80327558, 49689403, 99593509, 49709579, 8685121, 14705116, 88815080, 16713006, 122787575, 92443706, 37949484, 22858772, 41893168, 118124701, 126179348, 128990077, 84813984, 6253531, 15486743, 104356466, 8476864, 11846928, 41742040, 11046884, 126241008, 46240891, 9343948, 111692230, 113800951, 6043855, 39726585, 105666767, 35624554, 109692998, 67877491, 78803496, 71835527, 92398454, 46241044, 126862280, 80441672, 33453842, 88537052, 114046999, 15641411, 41683771, 14848825, 15246967, 93763944, 94318197, 126145329, 46241314, 7316902, 119744855, 8045073, 119287498, 59110678, 128677864, 107854807, 112029573, 8694207, 126272999, 

-- face cradle
-- 42708432, 83128028, 108941346, 66486376, 8692809, 69995778, 68527044, 16023483, 105939616, 15448813, 106115645, 114511681, 104924112, 121309788, 104170069, 108575183, 15011330, 50308092, 126585208, 103316104, 82882661, 39272926, 5907462, 16052180, 105837666, 7168563, 107915087, 83114032, 123940238, 37136243, 36957510, 122526896, 104880875, 113659071, 118821773, 118821538, 118960625, 22963858, 67769499, 32684806, 33162298, 69260695, 113444216, 24007462, 103423399, 59096207, 88651929, 77318674, 121307528, 79175591, 97641478, 121280000, 62358340, 72177434, 92197380, 98148845, 14682907, 105542740, 30950664, 7315881, 35057932, 118975471, 111439721, 92369226, 15822409, 86339400, 99980857, 23273802, 9185945, 121856764, 8688328, 60499469, 84772261, 95633774, 118828909, 60375128, 6545895, 68431821, 39536143, 215453, 92362045

-- buttoning coat
-- 87529894, 93061378, 82835605, 28032927, 52675622, 12667760, 124133068, 10110887, 8928417, 6287086, 95856693, 100275516, 13028142, 100140349, 84532882, 36437265, 11071088, 11428941, 119870634, 48494496, 78354342, 53975177, 35790451, 107845006, 100149157, 21847543, 99105685, 104238922, 126870708, 93373843, 93011321, 94127100, 16522939, 5940462, 83295836, 98375165, 128974552, 122213062, 127679497, 122547629, 81126515, 36754837, 86529937, 124369013, 11707409, 81126614, 99020756, 6348771, 10669616, 90565588, 85032388, 14789521, 127240794, 10677522, 36327670, 118193223, 126925623, 9178170, 84628149, 65120089, 78066606, 84532432, 102511159, 100703818, 83692874

-- headache
-- WHERE ihp.image_id in (124023744, 127621342, 126075894, 45810065, 124023261, 36937627, 93721272, 15741714, 37862543, 33750104, 87566400, 94914804, 92559830, 124121047, 85817606, 127111255, 89281349, 107776668, 97622392, 9959452, 35505606, 94996077, 102181280, 87524087, 16325000, 127916201, 78855957, 107500588, 23850518, 93385475, 91609344, 128040142, 78855929, 125929901, 108726557, 89261781, 125450211, 99185994, 11087866, 39707194, 109873529, 103911378, 109857101, 14695480, 89076233, 42849380, 16611957, 34738209, 93387515, 84485259, 71578960, 84843988, 21392641, 94482563, 81071891, 127299644, 9959146, 11151258, 107064281, 109824607, 14824356, 85824315, 40628891, 13608976, 89060231, 109904624, 37075686, 95435041, 84314662, 85998135, 88732267, 85553299, 16325620, 48228333, 9817284, 105521503, 55020962, 107624985, 86087651, 38582970, 42893244, 6020244, 129300513, 94926133, 6256387, 59861351, 129148178, 88693344, 43176620, 103897755, 89565668, 40172384, 35358425, 108950421, 58470730, 95345218)


SELECT * 
FROM Images
WHERE image_id = 30295
;

WHERE description LIKE 'White guy holding a flag of%'
;

-- kurdistan and holds his hand on his heart isolated on a white background with love to kurdistan

SELECT COUNT(s.image_id)
FROM SegmentOct20 s
WHERE s.mongo_body_landmarks = 1
AND s.face_x > -35 AND s.face_x < -24 AND s.face_y > -3 AND s.face_y < 3 AND s.face_z > -3 AND s.face_z < 3
;






   
   
SELECT *
FROM PhoneBbox
WHERE image_id = 118060332
;

SELECT COUNT(image_id) AS count,
       it.topic_id,
       t.topic
FROM ImagesTopics it
JOIN Topics t ON it.topic_id = t.topic_id
GROUP BY it.topic_id, t.topic;


SELECT COUNT(image_id) AS count,
       ip.cluster_id
FROM ImagesPoses ip 
JOIN Poses p ON ip.cluster_id = p.cluster_id
GROUP BY ip.cluster_id;


SELECT k.keyword_id, k.keyword_number, k.keyword_text 
FROM ImagesKeywords ik 
JOIN Keywords k ON ik.keyword_id = k.keyword_id 
-- WHERE i.site_image_id = "___638KPrmA"
WHERE ik.image_id = 15522520
;

SELECT COUNT(*)  
FROM SegmentOct20 so
WHERE so.mongo_body_landmarks_norm = 1
;

SELECT *
FROM SegmentOct20 e 
WHERE e.image_id = 110110042
;

SELECT COUNT(s.image_id)
FROM SegmentOct20  s
WHERE s.mongo_body_landmarks = 1
AND s.face_x > -33 AND s.face_x < -27 AND s.face_y > -2 AND s.face_y < 2 AND s.face_z > -2 AND s.face_z < 2
;

CREATE TABLE SegmentHelperAug16_SegOct20_preAlamy (
    seg_image_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    image_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES Images(image_id)
);



SELECT DISTINCT seg1.image_id, seg1.site_name_id, seg1.contentUrl, seg1.imagename, seg1.site_image_id, seg1.mongo_body_landmarks, seg1.mongo_face_landmarks, seg1.bbox 
FROM SegmentBig_isface seg1 
INNER JOIN SegmentHelper_sept18_silence ht ON seg1.image_id = ht.image_id  
WHERE  seg1.mongo_body_landmarks IS NULL and seg1.no_image IS NULL   
LIMIT 10000;




SELECT 
    FLOOR(face_z) AS face_x_unit,
    COUNT(*) AS row_count
FROM 
    SegmentOct20
GROUP BY 
    FLOOR(face_z)
ORDER BY 
    face_x_unit;


SELECT COUNT(i.image_id) 
FROM Images i 
WHERE i.site_name_id = 11

;


SELECT COUNT(so.seg_image_id) as ccount, COUNT(so.mongo_face_landmarks) as fcount, COUNT(so.mongo_body_landmarks) as bcount
FROM SegmentOct20 so 
JOIN ImagesBackground ib on ib.image_id = so.image_id 
;


SELECT *
FROM SegmentOct20 so 
LEFT JOIN ImagesBackground ON so.image_id = ImagesBackground.image_id
WHERE ImagesBackground.image_id IS NULL
LIMIT 10;

SELECT COUNT(ib.image_id)
FROM ImagesBackground ib
WHERE ib.selfie_bbox IS NOT NULL
;


SELECT *
FROM Encodings e 
WHERE e.encoding_id  = 114852545
;

SELECT COUNT(so.image_id) 
FROM SegmentOct20 so 
WHERE so.site_name_id = 1
;


-- ENCODINGS STUFFFFFFFFFF

SELECT COUNT(i.image_id) as ccount
FROM Images i  
JOIN Encodings e ON i.image_id = e.image_id 
WHERE  i.site_name_id = 13
;


DELETE Encodings
FROM Encodings 
JOIN Images ON images.image_id = encodings.image_id 
WHERE images.site_name_id = 13
;


SELECT COUNT(i.image_id) as ccount
FROM Images i  
JOIN Encodings e ON i.image_id = e.image_id 
WHERE e.face_landmarks is not NULL 
AND i.site_name_id = 13
;


SELECT *
FROM Encodings e 
WHERE e.image_id  = 122582509
AND e.face_landmarks IS NOT NULL
;


SELECT COUNT(so.seg_image_id) as ccount
FROM SegmentOct20 so 
JOIN Encodings e ON so.image_id = e.image_id 
WHERE e.mongo_body_landmarks = 1
;


SELECT count(seg1.image_id) 
FROM SegmentOct20 seg1 
JOIN ImagesTopics it ON seg1.image_id = it.image_id 
WHERE seg1.mongo_body_landmarks IS NULL 
AND it.topic_id = 5

;



SELECT *
FROM ImagesKeywords ik 
JOIN Images i ON ik.image_id = i.image_id 
WHERE ik.keyword_id = 1762
AND i.site_name_id = 13
;

SELECT *
FROM Keywords k 
WHERE k.keyword_id = 1762
;


SELECT so.mongo_body_landmarks, so.mongo_face_landmarks 
FROM SegmentOct20 so 
WHERE so.image_id = 2894566
;


SELECT *
FROM Images i
WHERE i.site_image_id  = 10133878
AND i.site_name_id = 5
;

SELECT COUNT(i.image_id) as ccount
FROM Images	i
LEFT JOIN Encodings e on e.image_id = i.image_id 
WHERE e.encoding_id IS NULL
AND i.site_name_id = 2
;

-- 1 199143


-- 126887888 at 6pm, may 15 ministock

SELECT ie.ethnicity_id 
FROM ImagesEthnicity ie 
JOIN Images i ON i.image_id = ie.image_id 
WHERE i.site_image_id  = 1204627074
LIMIT 1000
;

SHOW VARIABLES LIKE 'tmpdir';


SELECT DISTINCT ImagesBackground.image_id
FROM ImagesBackground
LEFT JOIN SegmentHelperApril1_topic7 ON ImagesBackground.image_id = SegmentHelperApril1_topic7.image_id
WHERE SegmentHelperApril1_topic7.image_id IS NOT NULL
  AND ImagesBackground.lum_torso IS NULL
LIMIT 10;

SELECT COUNT(i.image_id)
FROM Images	i
JOIN Encodings e on e.image_id = i.image_id 
WHERE face_x > -40 AND face_x < -24 AND face_y > -5 AND face_y < 5 AND face_z > -5 AND face_z < 5 
AND e.is_face IS TRUE 
AND i.age_id NOT IN (1,2,3)   
;

SELECT COUNT(e.image_id) 
FROM Encodings e  
JOIN SegmentHelperMay7_fingerpoint so ON e.image_id = so.image_id 
WHERE e.body_landmarks is  NULL  
-- AND so.image_id NOT IN (1,2,3)
-- 268348
;


SELECT s.image_id, s.description, it.topic_score 
FROM SegmentOct20 s  JOIN ImagesTopics it ON s.image_id = it.image_id  
-- WHERE s.body_landmarks IS NOT NULL 
WHERE face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2 AND it.topic_score > .3 AND s.age_id NOT IN (1,2,3)   
AND it.topic_id IN (17)
;

SELECT *
FROM Images i
WHERE i.site_image_id  = 5778
AND i.site_name_id = 5
;

SELECT *
FROM Encodings e 
WHERE e.image_id = 423851
;

SELECT MAX(sbi.seg_image_id) 
FROM SegmentBig_isface sbi 
;



SELECT *
FROM Images i 
WHERE i.site_image_id  = 15880596
AND i.site_name_id = 5;

AND i.image_id > 100172671;


SELECT COUNT(ik.keyword_id)
FROM ImagesKeywords ik 
JOIN Images i ON ik.image_id = i.image_id 
-- WHERE ik.image_id >= 122172671 
WHERE  i.site_name_id = 7 
AND ik.keyword_id = 33992
;



-- count of gender
SELECT COUNT(i.gender_id) as thiscount, i.gender_id 
FROM Images i   
WHERE i.image_id > 126887900
-- WHERE i.site_name_id = 7 
GROUP BY i.gender_id 
ORDER BY thiscount DESC;

SELECT COUNT(i.image_id) as thiscount 
FROM Images i   
WHERE i.site_name_id = 1
AND i.image_id > 131570175
;

3528798

SELECT distinct(i.image_id), i.description 
FROM Images i 
LEFT JOIN ImagesEthnicity ie ON i.image_id = ie.image_id 
WHERE i.site_name_id = 7 
AND ie.image_id IS NOT NULL
LIMIT 1000
;

SELECT ie.ethnicity_id 
FROM ImagesEthnicity ie 
WHERE ie.image_id = 126887995
;

SELECT k.keyword_text  
FROM ImagesKeywords ik
JOIN Keywords k ON ik.keyword_id = k.keyword_id 
WHERE ik.image_id = 114519467
;

SELECT COUNT(ik.keyword_id)
FROM ImagesKeywords ik 
WHERE ik.keyword_id = 12021;

USE Stock; 

SELECT COUNT(i.image_id)
FROM Images i 
-- RIGHT JOIN Encodings e ON i.image_id = e.image_id 
WHERE i.site_name_id =1
AND i.image_id > 100887900
;


SELECT *
FROM Images i 
WHERE i.image_id in (9924753, 9924032)
;

SELECT COUNT(i.image_id) as ccount, i.site_name_id 
FROM Images i 
LEFT JOIN Encodings e on i.image_id = e.image_id 
-- WHERE i.site_name_id = 2
WHERE e.encoding_id is NULL
AND i.image_id > 100000000
GROUP BY i.site_name_id 
;


-- missing encodings
SELECT i.image_id, i.site_name_id, i.imagename 
FROM Images i 
LEFT JOIN Encodings e on i.image_id = e.image_id 
WHERE e.encoding_id is NULL
-- AND i.site_name_id not in (1)
LIMIT 10
;



SELECT DISTINCT(s.image_id), s.age_id, s.site_name_id, s.contentUrl, s.imagename, s.face_x, s.face_y, s.face_z, s.mouth_gap, s.face_landmarks, s.bbox, s.face_encodings68, s.site_image_id, s.body_landmarks 
FROM SegmentOct20 s  JOIN ImagesTopics it ON s.image_id = it.image_id  
WHERE s.site_name_id != 1 AND face_encodings68 IS NOT NULL 
AND face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2 
-- AND s.age_id > 4  
AND it.topic_score > .3  AND it.topic_id = 10  
LIMIT 10;




-- SELECT DISTINCT seg1.image_id, seg1.site_name_id, seg1.contentUrl, seg1.imagename, e.encoding_id, seg1.site_image_id, e.face_landmarks, e.bbox 
SELECT COUNT(seg1.image_id)
FROM SegmentOct20 seg1 LEFT JOIN Encodings e ON seg1.image_id = e.image_id 
WHERE e.body_landmarks IS NULL AND e.image_id 
IN (SELECT seg1.image_id FROM SegmentOct20 seg1 WHERE face_x > -33 AND face_x < -27 AND face_y > -2 AND face_y < 2 AND face_z > -2 AND face_z < 2 AND seg1.site_name_id !=1) 
;
82976
LIMIT 1000;



-- AND i.imagename LIKE '/Volumes%' 

-- fixing getty image paths

SELECT COUNT(e.image_id) 
FROM Encodings e
INNER JOIN SegmentOct20 so on so.image_id = e.image_id 
WHERE e.body_landmarks IS NOT NULL
;

SELECT COUNT(i.image_id)
FROM Images i 
WHERE i.site_name_id = 1
AND i.imagename LIKE 'images/%'
;


SELECT MAX(i.image_id)
FROM Images i 
WHERE i.site_name_id = 1
-- AND i.imagename LIKE '/Volumes%' 
-- AND i.imagename LIKE 'images%'
;



USE stocktest;

CREATE TABLE Model_Release (
    release_name_id int NOT NULL AUTO_INCREMENT PRIMARY KEY,
    release_name varchar(20)
); 


SELECT COUNT(ik.keyword_id)
FROM Images i
JOIN ImagesKeywords ik on ik.image_id = i.image_id 
WHERE i.site_name_id = 10
;


SELECT COUNT(i.image_id)
FROM Images i
WHERE i.site_name_id = 11
AND i.location_id IS NULL
;

SELECT *
FROM Images i 
WHERE i.site_image_id = 7696
AND i.site_name_id = 15
;

SELECT DISTINCT(i.author)
FROM Images i 
WHERE i.site_name_id = 15

;

USE Stock;


SELECT DISTINCT i.image_id
FROM Images i
LEFT JOIN SegmentOct20 so ON i.image_id = so.image_id
WHERE so.bbox IS NOT NULL
AND so.mongo_body_landmarks = 1
AND i.h IS NULL
LIMIT 1000;

SELECT COUNT(i.image_id)
FROM Images i
LEFT JOIN SegmentOct20 so ON i.image_id = so.image_id
WHERE so.bbox IS NOT NULL
AND so.mongo_body_landmarks = 1
AND i.h IS NOT NULL
;

SELECT COUNT(*)
FROM PhoneBbox pb 
WHERE pb.bbox_67_norm IS NOT NULL
;


    
SELECT COUNT(i.image_id)
FROM Images i
WHERE i.site_name_id = 6

;

DELETE FROM Images
WHERE site_name_id = 18
;


SELECT *
FROM Images i 
WHERE i.image_id = 2402477
;


SELECT COUNT(i.image_id)
FROM ImagesKeywords ik 
JOIN Images i ON i.image_id = ik.image_id 
JOIN Keywords k ON ik.keyword_id = k.keyword_id
JOIN ImagesEthnicity ie ON ie.image_id = i.image_id 
WHERE ik.keyword_id = 22310
AND i.site_name_id = 9
;


SELECT k.keyword_text
FROM ImagesKeywords ik 
JOIN Images i ON i.image_id = ik.image_id 
JOIN Keywords k ON ik.keyword_id = k.keyword_id
JOIN ImagesEthnicity ie ON ie.image_id = i.image_id 
WHERE i.image_id = 2400730
AND i.site_name_id = 9
;


SELECT i.image_id
FROM Images i 
 WHERE i.site_image_id = 1008421694
AND i.site_name_id = 10
;

SELECT ie.ethnicity_id 
FROM ImagesEthnicity ie 
WHERE ie.image_id = 1847892
;

SELECT COUNT(ik.image_id)  
FROM ImagesKeywords ik 
JOIN Images i ON i.image_id = ik.image_id 
JOIN Keywords k ON ik.keyword_id = k.keyword_id
WHERE k.keyword_text = "gorgeous"
AND i.site_name_id = 9
;

