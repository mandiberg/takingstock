USE Stock;

-- ObjectFusion class funnel analysis
-- Purpose:
-- 1) Measure raw detection availability in the current ObjectFusion cohort.
-- 2) Measure what survives slot assignment into ImagesDetections.
-- 3) Measure how assigned classes spread across ImagesObjectFusion clusters.
--
-- This file is scoped to the current cohort pattern used in Clustering_SQL.py:
--   SegmentBig_isface
--   Encodings filters: is_dupe_of IS NULL, two_noses IS NULL
--   SegmentHelper_oct2025_evens_quarters helper subset
--   ImagesObjectFusion membership
--
-- Adjust these class lists and helper table name as needed.

USE Stock;

-- =========================================================
-- 0) Target and comparison classes
-- =========================================================
DROP TEMPORARY TABLE IF EXISTS target_classes;

CREATE TEMPORARY TABLE target_classes (
    class_id INT PRIMARY KEY,
    bucket VARCHAR(16) NOT NULL
);

INSERT INTO target_classes (class_id, bucket) VALUES
    (80, 'target'),
    (82, 'target'),
    (95, 'target'),
    (81, 'compare'),
    (83, 'compare'),
    (92, 'compare');


-- =========================================================
-- 1) Current ObjectFusion cohort
-- =========================================================
DROP TEMPORARY TABLE IF EXISTS cohort_image_ids;

CREATE TEMPORARY TABLE cohort_image_ids AS
SELECT DISTINCT
    s.image_id
FROM SegmentBig_isface s
JOIN Encodings e
    ON e.image_id = s.image_id
JOIN SegmentHelper_oct2025_evens_quarters h
    ON h.image_id = s.image_id
JOIN ImagesObjectFusion iof
    ON iof.image_id = s.image_id
WHERE e.is_dupe_of IS NULL
  AND e.two_noses IS NULL;

ALTER TABLE cohort_image_ids
    ADD PRIMARY KEY (image_id);

SELECT
    COUNT(*) AS cohort_images
FROM cohort_image_ids;


-- =========================================================
-- 2) Raw detections in cohort after ObjectFusion detection filters
--    Mirrors pipeline assumptions: bbox_norm present and conf >= 0.4
-- =========================================================
DROP TEMPORARY TABLE IF EXISTS raw_detections_filtered;

CREATE TEMPORARY TABLE raw_detections_filtered AS
SELECT
    d.image_id,
    d.detection_id,
    d.class_id,
    d.conf
FROM Detections d
JOIN cohort_image_ids c
    ON c.image_id = d.image_id
WHERE d.bbox_norm IS NOT NULL
  AND d.conf >= 0.4;

ALTER TABLE raw_detections_filtered
    ADD PRIMARY KEY (detection_id),
    ADD INDEX idx_raw_image_class (image_id, class_id),
    ADD INDEX idx_raw_class (class_id);

SELECT
    tc.bucket,
    rdf.class_id,
    COUNT(*) AS raw_det_rows,
    COUNT(DISTINCT rdf.detection_id) AS raw_det_unique,
    COUNT(DISTINCT rdf.image_id) AS raw_images
FROM raw_detections_filtered rdf
JOIN target_classes tc
    ON tc.class_id = rdf.class_id
GROUP BY tc.bucket, rdf.class_id
ORDER BY tc.bucket, rdf.class_id;

'''
compare	81	49706	49706	34370
compare	83	10514	10514	8085
compare	92	28562	28562	28344
target	80	4636	4636	4579
target	82	12465	12465	10628
target	95	12209	12209	12058
'''

-- =========================================================
-- 3) Slot assignments persisted in ImagesDetections
--    One row per image-slot-detection assignment
-- =========================================================
DROP TEMPORARY TABLE IF EXISTS assigned_slots_all;

-- updated 248938
CREATE TEMPORARY TABLE assigned_slots_all AS
SELECT idt.image_id, 'left_hand' AS slot_name, d.detection_id, d.class_id
FROM ImagesDetections idt
JOIN cohort_image_ids c ON c.image_id = idt.image_id
JOIN Detections d ON d.detection_id = idt.left_hand_object_id

UNION ALL

SELECT idt.image_id, 'right_hand' AS slot_name, d.detection_id, d.class_id
FROM ImagesDetections idt
JOIN cohort_image_ids c ON c.image_id = idt.image_id
JOIN Detections d ON d.detection_id = idt.right_hand_object_id

UNION ALL

SELECT idt.image_id, 'top_face' AS slot_name, d.detection_id, d.class_id
FROM ImagesDetections idt
JOIN cohort_image_ids c ON c.image_id = idt.image_id
JOIN Detections d ON d.detection_id = idt.top_face_object_id

UNION ALL

SELECT idt.image_id, 'left_eye' AS slot_name, d.detection_id, d.class_id
FROM ImagesDetections idt
JOIN cohort_image_ids c ON c.image_id = idt.image_id
JOIN Detections d ON d.detection_id = idt.left_eye_object_id

UNION ALL

SELECT idt.image_id, 'right_eye' AS slot_name, d.detection_id, d.class_id
FROM ImagesDetections idt
JOIN cohort_image_ids c ON c.image_id = idt.image_id
JOIN Detections d ON d.detection_id = idt.right_eye_object_id

UNION ALL

SELECT idt.image_id, 'mouth' AS slot_name, d.detection_id, d.class_id
FROM ImagesDetections idt
JOIN cohort_image_ids c ON c.image_id = idt.image_id
JOIN Detections d ON d.detection_id = idt.mouth_object_id

UNION ALL

SELECT idt.image_id, 'shoulder' AS slot_name, d.detection_id, d.class_id
FROM ImagesDetections idt
JOIN cohort_image_ids c ON c.image_id = idt.image_id
JOIN Detections d ON d.detection_id = idt.shoulder_object_id

UNION ALL

SELECT idt.image_id, 'waist' AS slot_name, d.detection_id, d.class_id
FROM ImagesDetections idt
JOIN cohort_image_ids c ON c.image_id = idt.image_id
JOIN Detections d ON d.detection_id = idt.waist_object_id

UNION ALL

SELECT idt.image_id, 'feet' AS slot_name, d.detection_id, d.class_id
FROM ImagesDetections idt
JOIN cohort_image_ids c ON c.image_id = idt.image_id
JOIN Detections d ON d.detection_id = idt.feet_object_id;

ALTER TABLE assigned_slots_all
    ADD INDEX idx_assigned_class (class_id),
    ADD INDEX idx_assigned_image_class (image_id, class_id),
    ADD INDEX idx_assigned_slot_class (slot_name, class_id);

SELECT
    tc.bucket,
    asa.class_id,
    asa.slot_name,
    COUNT(*) AS slot_hits,
    COUNT(DISTINCT asa.detection_id) AS unique_detections,
    COUNT(DISTINCT asa.image_id) AS assigned_images
FROM assigned_slots_all asa
JOIN target_classes tc
    ON tc.class_id = asa.class_id
GROUP BY tc.bucket, asa.class_id, asa.slot_name
ORDER BY tc.bucket, asa.class_id, asa.slot_name;

'''
compare	81	left_hand	22487	22487	22487
compare	83	left_hand	3121	3121	3121
compare	92	left_hand	12547	12547	12547
target	80	left_hand	2087	2087	2087
target	82	left_hand	6295	6295	6295
target	95	left_hand	5147	5147	5147
'''

-- =========================================================
-- 4) Funnel summary: raw detections -> assigned slots
-- =========================================================

-- This throws error: "SQL Error [1137] [HY000]: Can't reopen table: 'tc'"
WITH raw_agg AS (
    SELECT
        class_id,
        COUNT(*) AS raw_det_rows,
        COUNT(DISTINCT detection_id) AS raw_det_unique,
        COUNT(DISTINCT image_id) AS raw_images
    FROM raw_detections_filtered
    WHERE class_id IN (80, 81, 82, 83, 92, 95)
    GROUP BY class_id
),
assigned_agg AS (
    SELECT
        class_id,
        COUNT(*) AS assigned_slot_hits,
        COUNT(DISTINCT detection_id) AS assigned_det_unique,
        COUNT(DISTINCT image_id) AS assigned_images,
        SUM(CASE WHEN slot_name IN ('left_hand', 'right_hand') THEN 1 ELSE 0 END) AS hand_slot_hits,
        SUM(CASE WHEN slot_name NOT IN ('left_hand', 'right_hand') THEN 1 ELSE 0 END) AS nonhand_slot_hits
    FROM assigned_slots_all
    WHERE class_id IN (80, 81, 82, 83, 92, 95)
    GROUP BY class_id
)
SELECT
    tc.bucket,
    tc.class_id,
    COALESCE(r.raw_det_rows, 0) AS raw_det_rows,
    COALESCE(r.raw_det_unique, 0) AS raw_det_unique,
    COALESCE(r.raw_images, 0) AS raw_images,
    COALESCE(a.assigned_slot_hits, 0) AS assigned_slot_hits,
    COALESCE(a.assigned_det_unique, 0) AS assigned_det_unique,
    COALESCE(a.assigned_images, 0) AS assigned_images,
    COALESCE(a.hand_slot_hits, 0) AS hand_slot_hits,
    COALESCE(a.nonhand_slot_hits, 0) AS nonhand_slot_hits,
    ROUND(COALESCE(a.assigned_images, 0) / NULLIF(r.raw_images, 0), 4) AS image_capture_rate,
    ROUND(COALESCE(a.assigned_det_unique, 0) / NULLIF(r.raw_det_unique, 0), 4) AS detection_capture_rate,
    ROUND(COALESCE(a.hand_slot_hits, 0) / NULLIF(a.assigned_slot_hits, 0), 4) AS hand_slot_share
FROM target_classes tc
LEFT JOIN raw_agg r
    ON r.class_id = tc.class_id
LEFT JOIN assigned_agg a
    ON a.class_id = tc.class_id
ORDER BY tc.bucket, tc.class_id;


'''
compare	81	49706	49706	34370	22487	22487	22487	22487	0	0.6543	0.4524	1.0000
compare	83	10514	10514	8085	3121	3121	3121	3121	0	0.3860	0.2968	1.0000
compare	92	28562	28562	28344	12547	12547	12547	12547	0	0.4427	0.4393	1.0000
target	80	4636	4636	4579	2087	2087	2087	2087	0	0.4558	0.4502	1.0000
target	82	12465	12465	10628	6295	6295	6295	6295	0	0.5923	0.5050	1.0000
target	95	12209	12209	12058	5147	5147	5147	5147	0	0.4269	0.4216	1.0000
'''


-- =========================================================
-- 5) Image-level assigned classes for cluster spread analysis
--    One row per image-class, deduped across slots
-- =========================================================
DROP TEMPORARY TABLE IF EXISTS image_class_assignments_all;

CREATE TEMPORARY TABLE image_class_assignments_all AS
SELECT DISTINCT
    image_id,
    class_id
FROM assigned_slots_all;

ALTER TABLE image_class_assignments_all
    ADD INDEX idx_image_class (image_id, class_id),
    ADD INDEX idx_class_image (class_id, image_id);


-- =========================================================
-- 6) Cluster sizes and cluster-class counts
-- =========================================================
DROP TEMPORARY TABLE IF EXISTS cluster_sizes;

CREATE TEMPORARY TABLE cluster_sizes AS
SELECT
    cluster_id,
    COUNT(DISTINCT image_id) AS total_images
FROM ImagesObjectFusion
GROUP BY cluster_id;

ALTER TABLE cluster_sizes
    ADD PRIMARY KEY (cluster_id);


DROP TEMPORARY TABLE IF EXISTS cluster_class_counts_all;

CREATE TEMPORARY TABLE cluster_class_counts_all AS
SELECT
    iof.cluster_id,
    ica.class_id,
    COUNT(DISTINCT ica.image_id) AS images_n
FROM ImagesObjectFusion iof
JOIN image_class_assignments_all ica
    ON ica.image_id = iof.image_id
GROUP BY iof.cluster_id, ica.class_id;

ALTER TABLE cluster_class_counts_all
    ADD INDEX idx_cluster_class (cluster_id, class_id),
    ADD INDEX idx_class_cluster (class_id, cluster_id);


-- =========================================================
-- 7) Cluster spread summary for target and comparison classes
-- =========================================================
SELECT
    tc.bucket,
    ccc.class_id,
    SUM(ccc.images_n) AS assigned_images_in_clusters,
    COUNT(*) AS occupied_clusters,
    ROUND(AVG(ccc.images_n), 2) AS avg_images_per_occupied_cluster,
    MAX(ccc.images_n) AS max_images_in_one_cluster,
    SUM(CASE WHEN ccc.images_n >= 25 THEN 1 ELSE 0 END) AS clusters_ge_25_images,
    SUM(CASE WHEN ccc.images_n >= 100 THEN 1 ELSE 0 END) AS clusters_ge_100_images
FROM cluster_class_counts_all ccc
JOIN target_classes tc
    ON tc.class_id = ccc.class_id
GROUP BY tc.bucket, ccc.class_id
ORDER BY tc.bucket, ccc.class_id;


'''
compare	81	22487	151	148.92	6155	49	20
compare	83	3121	98	31.85	779	14	6
compare	92	12547	107	117.26	6083	31	13
target	80	2087	61	34.21	800	6	5
target	82	6295	114	55.22	2361	14	8
target	95	5147	80	64.34	1488	18	4
'''


-- =========================================================
-- 8) Rank / dominance summary inside clusters
--    Useful for distinguishing slot-loss from pose-splitting.
-- =========================================================
WITH ranked AS (
    SELECT
        ccc.cluster_id,
        ccc.class_id,
        ccc.images_n,
        cs.total_images,
        ROW_NUMBER() OVER (PARTITION BY ccc.cluster_id ORDER BY ccc.images_n DESC, ccc.class_id) AS class_rank_in_cluster
    FROM cluster_class_counts_all ccc
    JOIN cluster_sizes cs
        ON cs.cluster_id = ccc.cluster_id
)
SELECT
    tc.bucket,
    tc.class_id,
    COALESCE(SUM(CASE WHEN r.class_rank_in_cluster = 1 THEN 1 ELSE 0 END), 0) AS dominant_clusters,
    COALESCE(SUM(CASE WHEN r.class_rank_in_cluster <= 3 THEN 1 ELSE 0 END), 0) AS top3_clusters,
    COALESCE(SUM(CASE WHEN r.class_rank_in_cluster = 1 THEN r.total_images ELSE 0 END), 0) AS dominant_cluster_total_images,
    ROUND(COALESCE(AVG(CASE WHEN r.class_rank_in_cluster = 1 THEN r.images_n / NULLIF(r.total_images, 0) END), 0), 4) AS avg_share_when_dominant,
    ROUND(COALESCE(AVG(CASE WHEN r.class_rank_in_cluster <= 3 THEN r.images_n / NULLIF(r.total_images, 0) END), 0), 4) AS avg_share_when_top3
FROM target_classes tc
LEFT JOIN ranked r
    ON r.class_id = tc.class_id
GROUP BY tc.bucket, tc.class_id
ORDER BY tc.bucket, tc.class_id;

'''
compare	81	30	65	38898	0.3803	0.2631
compare	83	0	9	0	0.0000	0.0685
compare	92	23	44	16493	0.5359	0.3693
target	80	0	4	0	0.0000	0.0878
target	82	1	10	373	0.2788	0.1537
target	95	2	16	4558	0.4705	0.1545
'''

-- =========================================================
-- 9) Largest cluster presences for the classes of interest
-- =========================================================
SELECT
    tc.bucket,
    ccc.class_id,
    ccc.cluster_id,
    ccc.images_n AS class_images_in_cluster,
    cs.total_images,
    ROUND(ccc.images_n / NULLIF(cs.total_images, 0), 4) AS pct_of_cluster
FROM cluster_class_counts_all ccc
JOIN cluster_sizes cs
    ON cs.cluster_id = ccc.cluster_id
JOIN target_classes tc
    ON tc.class_id = ccc.class_id
ORDER BY ccc.class_id, ccc.images_n DESC, ccc.cluster_id
LIMIT 500;

'''
target	80	14	800	9994	0.0800
target	80	451	451	10601	0.0425
target	80	231	272	3984	0.0683
target	80	74	168	5500	0.0305
target	80	291	147	1382	0.1064
target	80	385	35	1316	0.0266
target	80	77	13	1709	0.0076
target	80	97	13	506	0.0257
target	80	258	12	558	0.0215
target	80	288	12	7375	0.0016
target	80	368	12	373	0.0322
target	80	108	11	223	0.0493
target	80	393	11	652	0.0169
target	80	161	10	1016	0.0098
target	80	92	9	359	0.0251
target	80	185	8	741	0.0108
target	80	149	6	604	0.0099
target	80	164	6	11599	0.0005
target	80	54	5	7231	0.0007
target	80	93	5	337	0.0148
target	80	257	5	497	0.0101
target	80	446	5	211	0.0237
target	80	15	4	792	0.0051
target	80	24	4	1099	0.0036
target	80	263	4	1007	0.0040
target	80	306	4	853	0.0047
target	80	68	3	384	0.0078
target	80	71	3	790	0.0038
target	80	287	3	333	0.0090
target	80	398	3	20	0.1500
target	80	416	3	164	0.0183
target	80	80	2	414	0.0048
target	80	130	2	288	0.0069
target	80	131	2	100	0.0200
target	80	160	2	1584	0.0013
target	80	199	2	798	0.0025
target	80	281	2	109	0.0183
target	80	463	2	2974	0.0007
target	80	494	2	1132	0.0018
target	80	500	2	126	0.0159
target	80	506	2	700	0.0029
target	80	43	1	444	0.0023
target	80	52	1	360	0.0028
target	80	85	1	360	0.0028
target	80	96	1	340	0.0029
target	80	163	1	84	0.0119
target	80	195	1	159	0.0063
target	80	211	1	325	0.0031
target	80	266	1	194	0.0052
target	80	298	1	322	0.0031
target	80	312	1	290	0.0034
target	80	325	1	217	0.0046
target	80	363	1	58	0.0172
target	80	378	1	182	0.0055
target	80	384	1	261	0.0038
target	80	403	1	226	0.0044
target	80	412	1	436	0.0023
target	80	470	1	79	0.0127
target	80	474	1	131	0.0076
target	80	490	1	401	0.0025
target	80	511	1	100	0.0100
compare	81	14	6155	9994	0.6159
compare	81	451	5267	10601	0.4968
compare	81	231	2611	3984	0.6554
compare	81	74	1736	5500	0.3156
compare	81	77	929	1709	0.5436
compare	81	15	530	792	0.6692
compare	81	306	490	853	0.5744
compare	81	291	455	1382	0.3292
compare	81	97	312	506	0.6166
compare	81	258	275	558	0.4928
compare	81	263	200	1007	0.1986
compare	81	185	198	741	0.2672
compare	81	24	197	1099	0.1793
compare	81	149	183	604	0.3030
compare	81	506	177	700	0.2529
compare	81	161	142	1016	0.1398
compare	81	68	138	384	0.3594
compare	81	160	137	1584	0.0865
compare	81	43	136	444	0.3063
compare	81	164	130	11599	0.0112
compare	81	288	89	7375	0.0121
compare	81	368	88	373	0.2359
compare	81	65	87	218	0.3991
compare	81	393	73	652	0.1120
compare	81	403	71	226	0.3142
compare	81	195	65	159	0.4088
compare	81	130	62	288	0.2153
compare	81	21	56	490	0.1143
compare	81	375	52	137	0.3796
compare	81	117	51	159	0.3208
compare	81	500	47	126	0.3730
compare	81	369	46	589	0.0781
compare	81	361	45	105	0.4286
compare	81	93	43	337	0.1276
compare	81	287	42	333	0.1261
compare	81	342	41	114	0.3596
compare	81	80	39	414	0.0942
compare	81	257	38	497	0.0765
compare	81	268	38	130	0.2923
compare	81	92	37	359	0.1031
compare	81	474	37	131	0.2824
compare	81	192	35	178	0.1966
compare	81	317	35	1011	0.0346
compare	81	440	33	209	0.1579
compare	81	445	33	120	0.2750
compare	81	140	30	202	0.1485
compare	81	416	29	164	0.1768
compare	81	321	28	125	0.2240
compare	81	511	28	100	0.2800
compare	81	434	24	95	0.2526
compare	81	146	23	250	0.0920
compare	81	212	22	120	0.1833
compare	81	267	22	236	0.0932
compare	81	131	21	100	0.2100
compare	81	281	21	109	0.1927
compare	81	450	21	58	0.3621
compare	81	112	20	486	0.0412
compare	81	298	20	322	0.0621
compare	81	54	18	7231	0.0025
compare	81	472	18	116	0.1552
compare	81	163	16	84	0.1905
compare	81	11	15	849	0.0177
compare	81	336	15	217	0.0691
compare	81	126	14	499	0.0281
compare	81	245	14	1055	0.0133
compare	81	392	14	81	0.1728
compare	81	156	13	1833	0.0071
compare	81	174	13	179	0.0726
compare	81	385	12	1316	0.0091
compare	81	272	11	64	0.1719
compare	81	352	11	69	0.1594
compare	81	364	11	43	0.2558
compare	81	438	11	136	0.0809
compare	81	313	10	131	0.0763
compare	81	353	10	70	0.1429
compare	81	371	10	366	0.0273
compare	81	490	10	401	0.0249
compare	81	279	9	101	0.0891
compare	81	312	9	290	0.0310
compare	81	52	8	360	0.0222
compare	81	71	8	790	0.0101
compare	81	85	8	360	0.0222
compare	81	115	8	239	0.0335
compare	81	437	8	31	0.2581
compare	81	463	8	2974	0.0027
compare	81	199	6	798	0.0075
compare	81	266	6	194	0.0309
compare	81	284	6	735	0.0082
compare	81	292	6	53	0.1132
compare	81	309	6	88	0.0682
compare	81	330	6	31	0.1935
compare	81	406	6	65	0.0923
compare	81	494	6	1132	0.0053
compare	81	96	5	340	0.0147
compare	81	119	5	35	0.1429
compare	81	215	5	102	0.0490
compare	81	246	5	220	0.0227
compare	81	430	5	345	0.0145
compare	81	446	5	211	0.0237
compare	81	10	4	391	0.0102
compare	81	218	4	126	0.0317
compare	81	277	4	388	0.0103
compare	81	485	4	64	0.0625
compare	81	39	3	248	0.0121
compare	81	50	3	284	0.0106
compare	81	63	3	316	0.0095
compare	81	109	3	60	0.0500
compare	81	144	3	165	0.0182
compare	81	191	3	64	0.0469
compare	81	205	3	394	0.0076
compare	81	237	3	269	0.0112
compare	81	376	3	24	0.1250
compare	81	423	3	54	0.0556
compare	81	462	3	32	0.0938
compare	81	480	3	249	0.0120
compare	81	502	3	22	0.1364
compare	81	88	2	568	0.0035
compare	81	99	2	39	0.0513
compare	81	101	2	111	0.0180
compare	81	108	2	223	0.0090
compare	81	184	2	194	0.0103
compare	81	295	2	54	0.0370
compare	81	316	2	102	0.0196
compare	81	320	2	161	0.0124
compare	81	325	2	217	0.0092
compare	81	378	2	182	0.0110
compare	81	384	2	261	0.0077
compare	81	412	2	436	0.0046
compare	81	420	2	31	0.0645
compare	81	425	2	268	0.0075
compare	81	442	2	323	0.0062
compare	81	470	2	79	0.0253
compare	81	479	2	210	0.0095
compare	81	78	1	77	0.0130
compare	81	122	1	27	0.0370
compare	81	125	1	25	0.0400
compare	81	141	1	99	0.0101
compare	81	189	1	169	0.0059
compare	81	194	1	763	0.0013
compare	81	209	1	182	0.0055
compare	81	211	1	325	0.0031
compare	81	276	1	40	0.0250
compare	81	293	1	69	0.0145
compare	81	326	1	35	0.0286
compare	81	328	1	73	0.0137
compare	81	381	1	521	0.0019
compare	81	382	1	117	0.0085
compare	81	398	1	20	0.0500
compare	81	407	1	831	0.0012
compare	81	422	1	31	0.0323
compare	81	504	1	118	0.0085
target	82	451	2361	10601	0.2227
target	82	74	1072	5500	0.1949
target	82	14	828	9994	0.0828
target	82	231	444	3984	0.1114
target	82	291	355	1382	0.2569
target	82	288	247	7375	0.0335
target	82	160	160	1584	0.1010
target	82	368	104	373	0.2788
target	82	77	63	1709	0.0369
target	82	506	47	700	0.0671
target	82	263	38	1007	0.0377
target	82	15	35	792	0.0442
target	82	161	31	1016	0.0305
target	82	267	26	236	0.1102
target	82	43	23	444	0.0518
target	82	306	21	853	0.0246
target	82	68	20	384	0.0521
target	82	185	18	741	0.0243
target	82	393	18	652	0.0276
target	82	71	17	790	0.0215
target	82	92	17	359	0.0474
target	82	97	17	506	0.0336
target	82	490	17	401	0.0424
target	82	287	16	333	0.0480
target	82	479	13	210	0.0619
target	82	149	12	604	0.0199
target	82	24	11	1099	0.0100
target	82	403	11	226	0.0487
target	82	371	10	366	0.0273
target	82	336	9	217	0.0415
target	82	382	9	117	0.0769
target	82	54	8	7231	0.0011
target	82	93	8	337	0.0237
target	82	96	7	340	0.0206
target	82	146	7	250	0.0280
target	82	258	7	558	0.0125
target	82	385	7	1316	0.0053
target	82	425	6	268	0.0224
target	82	440	6	209	0.0287
target	82	80	5	414	0.0121
target	82	174	5	179	0.0279
target	82	298	5	322	0.0155
target	82	313	5	131	0.0382
target	82	375	5	137	0.0365
target	82	438	5	136	0.0368
target	82	21	4	490	0.0082
target	82	52	4	360	0.0111
target	82	65	4	218	0.0183
target	82	130	4	288	0.0139
target	82	245	4	1055	0.0038
target	82	279	4	101	0.0396
target	82	317	4	1011	0.0040
target	82	416	4	164	0.0244
target	82	434	4	95	0.0421
target	82	85	3	360	0.0083
target	82	88	3	568	0.0053
target	82	117	3	159	0.0189
target	82	122	3	27	0.1111
target	82	126	3	499	0.0060
target	82	140	3	202	0.0149
target	82	257	3	497	0.0060
target	82	266	3	194	0.0155
target	82	381	3	521	0.0058
target	82	406	3	65	0.0462
target	82	445	3	120	0.0250
target	82	463	3	2974	0.0010
target	82	472	3	116	0.0259
target	82	480	3	249	0.0120
target	82	112	2	486	0.0041
target	82	131	2	100	0.0200
target	82	195	2	159	0.0126
target	82	211	2	325	0.0062
target	82	237	2	269	0.0074
target	82	268	2	130	0.0154
target	82	276	2	40	0.0500
target	82	281	2	109	0.0183
target	82	312	2	290	0.0069
target	82	342	2	114	0.0175
target	82	369	2	589	0.0034
target	82	376	2	24	0.0833
target	82	474	2	131	0.0153
target	82	494	2	1132	0.0018
target	82	504	2	118	0.0169
target	82	11	1	849	0.0012
target	82	99	1	39	0.0256
target	82	109	1	60	0.0167
target	82	115	1	239	0.0042
target	82	119	1	35	0.0286
target	82	144	1	165	0.0061
target	82	156	1	1833	0.0005
target	82	163	1	84	0.0119
target	82	184	1	194	0.0052
target	82	191	1	64	0.0156
target	82	194	1	763	0.0013
target	82	198	1	40	0.0250
target	82	199	1	798	0.0013
target	82	205	1	394	0.0025
target	82	212	1	120	0.0083
target	82	253	1	544	0.0018
target	82	272	1	64	0.0156
target	82	277	1	388	0.0026
target	82	292	1	53	0.0189
target	82	309	1	88	0.0114
target	82	316	1	102	0.0098
target	82	321	1	125	0.0080
target	82	325	1	217	0.0046
target	82	353	1	70	0.0143
target	82	361	1	105	0.0095
target	82	378	1	182	0.0055
target	82	384	1	261	0.0038
target	82	420	1	31	0.0323
target	82	446	1	211	0.0047
target	82	470	1	79	0.0127
target	82	500	1	126	0.0079
compare	83	14	779	9994	0.0779
compare	83	288	503	7375	0.0682
compare	83	451	448	10601	0.0423
compare	83	74	378	5500	0.0687
compare	83	231	210	3984	0.0527
compare	83	77	108	1709	0.0632
compare	83	15	59	792	0.0745
compare	83	306	56	853	0.0657
compare	83	185	52	741	0.0702
compare	83	393	38	652	0.0583
compare	83	160	33	1584	0.0208
compare	83	43	29	444	0.0653
compare	83	149	29	604	0.0480
compare	83	24	27	1099	0.0246
compare	83	97	23	506	0.0455
compare	83	291	23	1382	0.0166
compare	83	490	19	401	0.0474
compare	83	161	17	1016	0.0167
compare	83	403	15	226	0.0664
compare	83	54	13	7231	0.0018
compare	83	68	13	384	0.0339
compare	83	96	13	340	0.0382
compare	83	117	13	159	0.0818
compare	83	257	11	497	0.0221
compare	83	434	11	95	0.1158
compare	83	80	10	414	0.0242
compare	83	112	9	486	0.0185
compare	83	258	9	558	0.0161
compare	83	263	8	1007	0.0079
compare	83	368	8	373	0.0214
compare	83	195	7	159	0.0440
compare	83	312	7	290	0.0241
compare	83	130	6	288	0.0208
compare	83	385	6	1316	0.0046
compare	83	144	5	165	0.0303
compare	83	93	4	337	0.0119
compare	83	108	4	223	0.0179
compare	83	140	4	202	0.0198
compare	83	146	4	250	0.0160
compare	83	245	4	1055	0.0038
compare	83	268	4	130	0.0308
compare	83	279	4	101	0.0396
compare	83	416	4	164	0.0244
compare	83	440	4	209	0.0191
compare	83	506	4	700	0.0057
compare	83	65	3	218	0.0138
compare	83	163	3	84	0.0357
compare	83	192	3	178	0.0169
compare	83	281	3	109	0.0275
compare	83	287	3	333	0.0090
compare	83	317	3	1011	0.0030
compare	83	369	3	589	0.0051
compare	83	381	3	521	0.0058
compare	83	511	3	100	0.0300
compare	83	21	2	490	0.0041
compare	83	52	2	360	0.0056
compare	83	115	2	239	0.0084
compare	83	174	2	179	0.0112
compare	83	199	2	798	0.0025
compare	83	272	2	64	0.0313
compare	83	276	2	40	0.0500
compare	83	298	2	322	0.0062
compare	83	342	2	114	0.0175
compare	83	382	2	117	0.0171
compare	83	392	2	81	0.0247
compare	83	430	2	345	0.0058
compare	83	438	2	136	0.0147
compare	83	485	2	64	0.0313
compare	83	502	2	22	0.0909
compare	83	63	1	316	0.0032
compare	83	71	1	790	0.0013
compare	83	78	1	77	0.0130
compare	83	85	1	360	0.0028
compare	83	125	1	25	0.0400
compare	83	126	1	499	0.0020
compare	83	156	1	1833	0.0005
compare	83	198	1	40	0.0250
compare	83	211	1	325	0.0031
compare	83	212	1	120	0.0083
compare	83	213	1	28	0.0357
compare	83	253	1	544	0.0018
compare	83	266	1	194	0.0052
compare	83	267	1	236	0.0042
compare	83	277	1	388	0.0026
compare	83	316	1	102	0.0098
compare	83	321	1	125	0.0080
compare	83	336	1	217	0.0046
compare	83	361	1	105	0.0095
compare	83	378	1	182	0.0055
compare	83	412	1	436	0.0023
compare	83	420	1	31	0.0323
compare	83	437	1	31	0.0323
compare	83	445	1	120	0.0083
compare	83	446	1	211	0.0047
compare	83	463	1	2974	0.0003
compare	83	474	1	131	0.0076
compare	83	479	1	210	0.0048
compare	83	480	1	249	0.0040
compare	92	54	6083	7231	0.8412
compare	92	262	2383	4880	0.4883
compare	92	253	492	544	0.9044
compare	92	381	412	521	0.7908
compare	92	160	283	1584	0.1787
compare	92	52	282	360	0.7833
compare	92	63	266	316	0.8418
compare	92	298	241	322	0.7484
compare	92	50	175	284	0.6162
compare	92	93	158	337	0.4688
compare	92	378	135	182	0.7418
compare	92	130	129	288	0.4479
compare	92	263	108	1007	0.1072
compare	92	80	93	414	0.2246
compare	92	267	86	236	0.3644
compare	92	144	82	165	0.4970
compare	92	313	76	131	0.5802
compare	92	43	67	444	0.1509
compare	92	438	54	136	0.3971
compare	92	266	53	194	0.2732
compare	92	293	51	69	0.7391
compare	92	361	42	105	0.4000
compare	92	115	37	239	0.1548
compare	92	215	37	102	0.3627
compare	92	474	37	131	0.2824
compare	92	445	31	120	0.2583
compare	92	131	30	100	0.3000
compare	92	185	29	741	0.0391
compare	92	199	29	798	0.0363
compare	92	258	28	558	0.0502
compare	92	511	28	100	0.2800
compare	92	414	24	51	0.4706
compare	92	506	24	700	0.0343
compare	92	291	23	1382	0.0166
compare	92	195	22	159	0.1384
compare	92	218	21	126	0.1667
compare	92	342	21	114	0.1842
compare	92	279	19	101	0.1881
compare	92	375	19	137	0.1387
compare	92	472	18	116	0.1552
compare	92	392	17	81	0.2099
compare	92	406	17	65	0.2615
compare	92	112	16	486	0.0329
compare	92	174	16	179	0.0894
compare	92	485	16	64	0.2500
compare	92	463	13	2974	0.0044
compare	92	232	10	106	0.0943
compare	92	368	10	373	0.0268
compare	92	77	9	1709	0.0053
compare	92	122	9	27	0.3333
compare	92	192	9	178	0.0506
compare	92	312	9	290	0.0310
compare	92	450	9	58	0.1552
compare	92	458	8	43	0.1860
compare	92	92	7	359	0.0195
compare	92	140	7	202	0.0347
compare	92	68	6	384	0.0156
compare	92	161	6	1016	0.0059
compare	92	257	6	497	0.0121
compare	92	364	6	43	0.1395
compare	92	440	6	209	0.0287
compare	92	479	6	210	0.0286
compare	92	21	5	490	0.0102
compare	92	126	5	499	0.0100
compare	92	146	5	250	0.0200
compare	92	149	5	604	0.0083
compare	92	434	5	95	0.0526
compare	92	490	5	401	0.0125
compare	92	353	4	70	0.0571
compare	92	125	3	25	0.1200
compare	92	212	3	120	0.0250
compare	92	276	3	40	0.0750
compare	92	292	3	53	0.0566
compare	92	325	3	217	0.0138
compare	92	352	3	69	0.0435
compare	92	382	3	117	0.0256

'''

-- =========================================================
-- 10) Diagnostic: raw class detections present but never assigned to any slot
--     Start with class 80; change class_id as needed.
-- =========================================================
SELECT
    rdf.image_id,
    rdf.class_id,
    COUNT(*) AS raw_det_rows,
    COUNT(DISTINCT rdf.detection_id) AS raw_det_unique,
    MAX(rdf.conf) AS max_conf
FROM raw_detections_filtered rdf
LEFT JOIN image_class_assignments_all ica
    ON ica.image_id = rdf.image_id
   AND ica.class_id = rdf.class_id
WHERE rdf.class_id = 80
  AND ica.image_id IS NULL
GROUP BY rdf.image_id, rdf.class_id
ORDER BY max_conf DESC, raw_det_unique DESC, rdf.image_id
LIMIT 200;


-- =========================================================
-- 11) Diagnostic: hand-heavy classes by slot mix
--     Useful for classes 82 and 95.
-- =========================================================
SELECT
    asa.class_id,
    asa.slot_name,
    COUNT(*) AS slot_hits,
    COUNT(DISTINCT asa.image_id) AS images_n
FROM assigned_slots_all asa
WHERE asa.class_id IN (82, 95)
GROUP BY asa.class_id, asa.slot_name
ORDER BY asa.class_id, slot_hits DESC, asa.slot_name;


-- =========================================================
-- 12) Right-hand anomaly diagnostics
-- =========================================================

-- 12.1 Cohort-level slot occupancy by side (raw ImagesDetections columns)
SELECT
    COUNT(*) AS cohort_rows_in_imagesdetections,
    SUM(CASE WHEN idt.left_hand_object_id  IS NOT NULL THEN 1 ELSE 0 END) AS rows_left_hand_nonnull,
    SUM(CASE WHEN idt.right_hand_object_id IS NOT NULL THEN 1 ELSE 0 END) AS rows_right_hand_nonnull,
    SUM(CASE WHEN idt.left_hand_object_id  IS NOT NULL AND idt.right_hand_object_id IS NOT NULL THEN 1 ELSE 0 END) AS rows_both_hands_nonnull,
    SUM(CASE WHEN idt.left_hand_object_id  IS NOT NULL AND idt.right_hand_object_id IS NULL THEN 1 ELSE 0 END) AS rows_left_only,
    SUM(CASE WHEN idt.left_hand_object_id  IS NULL AND idt.right_hand_object_id IS NOT NULL THEN 1 ELSE 0 END) AS rows_right_only
FROM ImagesDetections idt
JOIN cohort_image_ids c
    ON c.image_id = idt.image_id;

'''
3999459	248938	264671	190230	58708	74441
'''

-- 12.2 Per-class hand-side occupancy in cohort
SELECT
    tc.bucket,
    tc.class_id,
    SUM(CASE WHEN asa.slot_name = 'left_hand' AND asa.class_id = tc.class_id THEN 1 ELSE 0 END) AS left_hits,
    SUM(CASE WHEN asa.slot_name = 'right_hand' AND asa.class_id = tc.class_id THEN 1 ELSE 0 END) AS right_hits,
    ROUND(
        SUM(CASE WHEN asa.slot_name = 'right_hand' AND asa.class_id = tc.class_id THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN asa.slot_name = 'left_hand' AND asa.class_id = tc.class_id THEN 1 ELSE 0 END), 0),
        4
    ) AS right_to_left_ratio
FROM target_classes tc
JOIN assigned_slots_all asa
    ON asa.class_id = tc.class_id
   AND asa.slot_name IN ('left_hand', 'right_hand')
GROUP BY tc.bucket, tc.class_id
ORDER BY tc.bucket, tc.class_id;

'''compare	81	22487	22647	1.0071
compare	83	3121	3115	0.9981
compare	92	12547	13150	1.0481
target	80	2087	2051	0.9828
target	82	6295	7133	1.1331
target	95	5147	5863	1.1391
'''


-- 12.3 Any class with extreme side imbalance (right_to_left_ratio < 0.05)
WITH per_class AS (
    SELECT
        asa.class_id,
        SUM(CASE WHEN asa.slot_name = 'left_hand' THEN 1 ELSE 0 END) AS left_hits,
        SUM(CASE WHEN asa.slot_name = 'right_hand' THEN 1 ELSE 0 END) AS right_hits
    FROM assigned_slots_all asa
    WHERE asa.slot_name IN ('left_hand', 'right_hand')
    GROUP BY asa.class_id
)
SELECT
    class_id,
    left_hits,
    right_hits,
    ROUND(right_hits / NULLIF(left_hits, 0), 4) AS right_to_left_ratio
FROM per_class
WHERE left_hits >= 100
  AND (right_hits / NULLIF(left_hits, 0)) < 0.05
ORDER BY left_hits DESC, class_id;

67	43855	0	0.0000
63	43201	0	0.0000
81	22487	0	0.0000
73	18152	0	0.0000
92	12547	0	0.0000
86	10240	0	0.0000
26	7428	0	0.0000
32	7312	0	0.0000
82	6295	0	0.0000
95	5147	0	0.0000
57	4873	0	0.0000
59	4312	0	0.0000
90	4093	0	0.0000
84	4021	0	0.0000
41	3648	0	0.0000
56	3121	0	0.0000
83	3121	0	0.0000
60	2506	0	0.0000
45	2191	0	0.0000
80	2087	0	0.0000
58	1933	0	0.0000
110	1862	0	0.0000
27	1624	0	0.0000
55	1596	0	0.0000
74	1567	0	0.0000
2	1351	0	0.0000
97	1305	0	0.0000
39	1204	0	0.0000
100	1198	0	0.0000
47	1118	0	0.0000
37	1068	0	0.0000
79	1005	0	0.0000
24	1004	0	0.0000
25	983	0	0.0000
77	942	0	0.0000
29	929	0	0.0000
16	805	0	0.0000
103	761	0	0.0000
40	720	0	0.0000
108	712	0	0.0000
109	709	0	0.0000
34	694	0	0.0000
54	661	0	0.0000
1	627	0	0.0000
96	582	0	0.0000
94	580	0	0.0000
49	557	0	0.0000
44	509	0	0.0000
76	480	0	0.0000
33	459	0	0.0000
111	456	0	0.0000
28	453	0	0.0000
48	388	0	0.0000
46	376	0	0.0000
43	371	0	0.0000
65	360	0	0.0000
42	357	0	0.0000
62	280	0	0.0000
8	276	0	0.0000
38	273	0	0.0000
15	258	0	0.0000
72	250	0	0.0000
50	246	0	0.0000
66	245	0	0.0000
36	240	0	0.0000
53	239	0	0.0000
7	234	0	0.0000
3	231	0	0.0000
88	212	0	0.0000
89	199	0	0.0000
6	187	0	0.0000
101	184	0	0.0000
75	183	0	0.0000
78	177	0	0.0000
102	163	0	0.0000
52	158	0	0.0000
112	149	0	0.0000
98	147	0	0.0000
17	136	0	0.0000
51	112	0	0.0000


-- 12.4 Check for duplicate assignment of the same detection to both hands
SELECT
    COUNT(*) AS rows_same_detection_both_hands
FROM ImagesDetections idt
JOIN cohort_image_ids c
    ON c.image_id = idt.image_id
WHERE idt.left_hand_object_id IS NOT NULL
  AND idt.right_hand_object_id IS NOT NULL
  AND idt.left_hand_object_id = idt.right_hand_object_id;


156369

-- total hand assignments images
SELECT
    COUNT(*) AS rows_any_detection_hands
FROM ImagesDetections idt
JOIN cohort_image_ids c
    ON c.image_id = idt.image_id
WHERE (idt.left_hand_object_id IS NOT NULL
  OR idt.right_hand_object_id IS NOT NULL)
  ;
323379

-- 12.5 Hand pointer quality from ImagesDetections (are right pointers missing/zero?)
SELECT
    COUNT(*) AS cohort_rows_in_imagesdetections,
    SUM(CASE WHEN idt.left_pointer_x IS NOT NULL AND idt.left_pointer_y IS NOT NULL THEN 1 ELSE 0 END) AS rows_left_pointer_nonnull,
    SUM(CASE WHEN idt.right_pointer_x IS NOT NULL AND idt.right_pointer_y IS NOT NULL THEN 1 ELSE 0 END) AS rows_right_pointer_nonnull,
    SUM(CASE WHEN idt.left_pointer_x = 0 AND idt.left_pointer_y = 0 THEN 1 ELSE 0 END) AS rows_left_pointer_zero,
    SUM(CASE WHEN idt.right_pointer_x = 0 AND idt.right_pointer_y = 0 THEN 1 ELSE 0 END) AS rows_right_pointer_zero
FROM ImagesDetections idt
JOIN cohort_image_ids c
    ON c.image_id = idt.image_id;


'''
3999459	3999459	3999459	0	0
'''





-- 12.6 Left vs right detection-id resolvability audit
SELECT
  COUNT(*) AS cohort_rows,
  SUM(CASE WHEN idt.left_hand_object_id IS NOT NULL THEN 1 ELSE 0 END) AS left_id_nonnull,
  SUM(CASE WHEN idt.right_hand_object_id IS NOT NULL THEN 1 ELSE 0 END) AS right_id_nonnull,
  SUM(CASE WHEN idt.left_hand_object_id IS NOT NULL AND dl.detection_id IS NOT NULL THEN 1 ELSE 0 END) AS left_resolved,
  SUM(CASE WHEN idt.right_hand_object_id IS NOT NULL AND dr.detection_id IS NOT NULL THEN 1 ELSE 0 END) AS right_resolved,
  SUM(CASE WHEN idt.left_hand_object_id IS NOT NULL AND dl.detection_id IS NULL THEN 1 ELSE 0 END) AS left_unresolved,
  SUM(CASE WHEN idt.right_hand_object_id IS NOT NULL AND dr.detection_id IS NULL THEN 1 ELSE 0 END) AS right_unresolved
FROM ImagesDetections idt
JOIN cohort_image_ids c ON c.image_id = idt.image_id
LEFT JOIN Detections dl ON dl.detection_id = idt.left_hand_object_id
LEFT JOIN Detections dr ON dr.detection_id = idt.right_hand_object_id;

3999459	248938	264671	248938	264671	0	0


-- 12.7 Sample unresolved right-hand ids
SELECT
  idt.image_id,
  idt.right_hand_object_id
FROM ImagesDetections idt
JOIN cohort_image_ids c ON c.image_id = idt.image_id
LEFT JOIN Detections dr ON dr.detection_id = idt.right_hand_object_id
WHERE idt.right_hand_object_id IS NOT NULL
  AND dr.detection_id IS NULL
LIMIT 200;

0 results

-- 12.8 Range sanity: do right-hand ids look like detection_id values?
SELECT
  MIN(idt.left_hand_object_id) AS left_min_id,
  MAX(idt.left_hand_object_id) AS left_max_id,
  MIN(idt.right_hand_object_id) AS right_min_id,
  MAX(idt.right_hand_object_id) AS right_max_id
FROM ImagesDetections idt
JOIN cohort_image_ids c ON c.image_id = idt.image_id
WHERE idt.left_hand_object_id IS NOT NULL
   OR idt.right_hand_object_id IS NOT NULL;

286	62754245	16	62754245]



SELECT slot_name, COUNT(*) 
FROM assigned_slots_all
GROUP BY slot_name
ORDER BY slot_name;

left_hand	248938





USE Stock;

WITH cohort AS (
  SELECT DISTINCT s.image_id
  FROM SegmentBig_isface s
  JOIN Encodings e ON e.image_id = s.image_id
  JOIN SegmentHelper_oct2025_evens_quarters h ON h.image_id = s.image_id
  JOIN ImagesObjectFusion iof ON iof.image_id = s.image_id
  WHERE e.is_dupe_of IS NULL
    AND e.two_noses IS NULL
)
SELECT
  COUNT(*) AS rows_in_imagesdetections,
  SUM(idt.left_hand_object_id IS NOT NULL) AS left_nonnull,
  SUM(idt.right_hand_object_id IS NOT NULL) AS right_nonnull,
  SUM(idt.left_hand_object_id IS NOT NULL AND idt.right_hand_object_id IS NOT NULL) AS both_nonnull,
  SUM(idt.left_hand_object_id = idt.right_hand_object_id
      AND idt.left_hand_object_id IS NOT NULL) AS same_det_both_hands
FROM ImagesDetections idt
JOIN cohort c ON c.image_id = idt.image_id;

3999459	248938	264671	190230	156369



USE Stock;

WITH cohort AS (
  SELECT DISTINCT s.image_id
  FROM SegmentBig_isface s
  JOIN Encodings e ON e.image_id = s.image_id
  JOIN SegmentHelper_oct2025_evens_quarters h ON h.image_id = s.image_id
  JOIN ImagesObjectFusion iof ON iof.image_id = s.image_id
  WHERE e.is_dupe_of IS NULL
    AND e.two_noses IS NULL
),
hand_class_hits AS (
  SELECT d.class_id, 'left_hand' AS side, COUNT(*) AS hits
  FROM ImagesDetections idt
  JOIN cohort c ON c.image_id = idt.image_id
  JOIN Detections d ON d.detection_id = idt.left_hand_object_id
  GROUP BY d.class_id
  UNION ALL
  SELECT d.class_id, 'right_hand' AS side, COUNT(*) AS hits
  FROM ImagesDetections idt
  JOIN cohort c ON c.image_id = idt.image_id
  JOIN Detections d ON d.detection_id = idt.right_hand_object_id
  GROUP BY d.class_id
)
SELECT
  class_id,
  SUM(CASE WHEN side = 'left_hand' THEN hits ELSE 0 END) AS left_hits,
  SUM(CASE WHEN side = 'right_hand' THEN hits ELSE 0 END) AS right_hits,
  ROUND(
    SUM(CASE WHEN side = 'right_hand' THEN hits ELSE 0 END) /
    NULLIF(SUM(CASE WHEN side = 'left_hand' THEN hits ELSE 0 END), 0),
    4
  ) AS right_to_left_ratio
FROM hand_class_hits
GROUP BY class_id
ORDER BY left_hits DESC;



67	43855	52138	1.1889
63	43201	42952	0.9942
81	22487	22647	1.0071
73	18152	17930	0.9878
92	12547	13150	1.0481
86	10240	10889	1.0634
26	7428	7648	1.0296
32	7312	7930	1.0845
82	6295	7133	1.1331
95	5147	5863	1.1391
57	4873	4978	1.0215
59	4312	4358	1.0107
90	4093	4764	1.1639
84	4021	4244	1.0555
41	3648	4157	1.1395
56	3121	3043	0.9750
83	3121	3115	0.9981
60	2506	2237	0.8927
45	2191	1498	0.6837
80	2087	2051	0.9828
58	1933	2047	1.0590
110	1862	2088	1.1214
27	1624	1988	1.2241
55	1596	1525	0.9555
74	1567	1697	1.0830
2	1351	1386	1.0259
97	1305	1361	1.0429
39	1204	1366	1.1346
100	1198	1229	1.0259
47	1118	1186	1.0608
37	1068	1065	0.9972
79	1005	1359	1.3522
24	1004	1067	1.0627
25	983	1070	1.0885
77	942	947	1.0053
29	929	971	1.0452
16	805	788	0.9789
103	761	802	1.0539
40	720	924	1.2833
108	712	851	1.1952
109	709	751	1.0592
34	694	788	1.1354
54	661	696	1.0530
1	627	612	0.9761
96	582	808	1.3883
94	580	486	0.8379
49	557	595	1.0682
44	509	942	1.8507
76	480	610	1.2708
33	459	481	1.0479
111	456	489	1.0724
28	453	505	1.1148
48	388	368	0.9485
46	376	415	1.1037
43	371	495	1.3342
65	360	419	1.1639
42	357	749	2.0980
62	280	238	0.8500
8	276	279	1.0109
38	273	307	1.1245
15	258	249	0.9651
72	250	273	1.0920
50	246	232	0.9431
66	245	203	0.8286
36	240	248	1.0333
53	239	249	1.0418
7	234	239	1.0214
3	231	234	1.0130
88	212	189	0.8915
89	199	213	1.0704
6	187	178	0.9519
101	184	174	0.9457
75	183	164	0.8962
78	177	186	1.0508
102	163	173	1.0613
52	158	174	1.1013
112	149	145	0.9732
98	147	143	0.9728
17	136	146	1.0735
51	112	117	1.0446
14	99	107	1.0808
69	97	95	0.9794
114	89	84	0.9438
35	87	77	0.8851
71	83	75	0.9036
4	76	85	1.1184
118	67	71	1.0597
5	64	67	1.0469
30	61	66	1.0820
105	61	70	1.1475
68	55	44	0.8000
64	54	96	1.7778
104	49	48	0.9796
19	39	36	0.9231
31	34	37	1.0882
18	31	33	1.0645
61	29	21	0.7241
116	28	28	1.0000
99	26	33	1.2692
115	14	12	0.8571
113	14	15	1.0714
20	11	11	1.0000
117	11	16	1.4545
21	7	6	0.8571
107	7	11	1.5714
23	6	4	0.6667
106	5	5	1.0000
70	5	2	0.4000
119	4	7	1.7500
22	3	5	1.6667

USE Stock;

WITH cohort AS (
  SELECT DISTINCT s.image_id
  FROM SegmentBig_isface s
  JOIN Encodings e ON e.image_id = s.image_id
  JOIN SegmentHelper_oct2025_evens_quarters h ON h.image_id = s.image_id
  JOIN ImagesObjectFusion iof ON iof.image_id = s.image_id
  WHERE e.is_dupe_of IS NULL
    AND e.two_noses IS NULL
),
raw_det AS (
  SELECT d.image_id, d.detection_id, d.class_id
  FROM Detections d
  JOIN cohort c ON c.image_id = d.image_id
  WHERE d.bbox_norm IS NOT NULL
    AND d.conf >= 0.4
    AND d.class_id IN (80,81,82,83,92,95)
),
assigned_hand AS (
  SELECT idt.image_id, d.detection_id, d.class_id, 'left_hand' AS side
  FROM ImagesDetections idt
  JOIN cohort c ON c.image_id = idt.image_id
  JOIN Detections d ON d.detection_id = idt.left_hand_object_id
  WHERE d.class_id IN (80,81,82,83,92,95)
  UNION ALL
  SELECT idt.image_id, d.detection_id, d.class_id, 'right_hand' AS side
  FROM ImagesDetections idt
  JOIN cohort c ON c.image_id = idt.image_id
  JOIN Detections d ON d.detection_id = idt.right_hand_object_id
  WHERE d.class_id IN (80,81,82,83,92,95)
)
SELECT
  r.class_id,
  COUNT(DISTINCT r.image_id) AS raw_images,
  COUNT(DISTINCT a.image_id) AS assigned_images,
  COUNT(DISTINCT a.detection_id) AS assigned_det_unique,
  ROUND(COUNT(DISTINCT a.image_id) / NULLIF(COUNT(DISTINCT r.image_id),0), 4) AS image_capture_rate
FROM raw_det r
LEFT JOIN assigned_hand a
  ON a.image_id = r.image_id
 AND a.class_id = r.class_id
GROUP BY r.class_id
ORDER BY r.class_id;


80	4579	2367	2378	0.5169
81	34370	26251	27846	0.7638
82	10628	9124	9748	0.8585
83	8085	4418	5108	0.5464
92	28344	16905	16919	0.5964
95	12058	9324	9375	0.7733