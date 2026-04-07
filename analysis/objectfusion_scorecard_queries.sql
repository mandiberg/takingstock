-- ObjectFusion scorecard query pack
-- Tables expected:
--   ImagesObjectFusion(image_id, cluster_id, cluster_dist)
--   ImagesObjectFusion_odd(image_id, cluster_id, cluster_dist)
--   ImagesDetections(image_id, left_hand_object_id, right_hand_object_id, top_face_object_id,
--                    left_eye_object_id, right_eye_object_id, mouth_object_id,
--                    shoulder_object_id, waist_object_id, feet_object_id)
--   Detections(detection_id, class_id)

USE Stock;

-- utility: rename 

ALTER TABLE ImagesObjectFusion RENAME TO ImagesObjectFusion_baseline;
ALTER TABLE ObjectFusion RENAME TO ObjectFusion_baseline;

ALTER TABLE ImagesObjectFusion RENAME TO ImagesObjectFusion_c1_class_separation;
ALTER TABLE ObjectFusion RENAME TO ObjectFusion_c1_class_separation;

ALTER TABLE ImagesObjectFusion RENAME TO ImagesObjectFusion_c2_spatial_tighter;
ALTER TABLE ObjectFusion RENAME TO ObjectFusion_c2_spatial_tighter;

ALTER TABLE ImagesObjectFusion RENAME TO ImagesObjectFusion_c3_spatial_looser;
ALTER TABLE ObjectFusion RENAME TO ObjectFusion_c3_spatial_looser;

ALTER TABLE ImagesObjectFusion RENAME TO ImagesObjectFusion_c4_face_angle_neutralized;
ALTER TABLE ObjectFusion RENAME TO ObjectFusion_c4_face_angle_neutralized;
-- candidate tables to compare in section 10:

_c3_spatial_looser
_c4_face_angle_neutralized

-- =========================================================
-- 0) Base checks
-- =========================================================
SELECT 'even' AS split_name, COUNT(*) AS rows_n, COUNT(DISTINCT image_id) AS images_n, COUNT(DISTINCT cluster_id) AS clusters_n
FROM ImagesObjectFusion
UNION ALL
SELECT 'odd' AS split_name, COUNT(*) AS rows_n, COUNT(DISTINCT image_id) AS images_n, COUNT(DISTINCT cluster_id) AS clusters_n
FROM ImagesObjectFusion_odd;


-- =========================================================
-- 1) Cohesion summary (run-level)
-- lower is better
-- =========================================================
SELECT 'even' AS split_name,
       AVG(cluster_dist) AS mean_cluster_dist,
       STDDEV_POP(cluster_dist) AS sd_cluster_dist
FROM ImagesObjectFusion
UNION ALL
SELECT 'odd' AS split_name,
       AVG(cluster_dist) AS mean_cluster_dist,
       STDDEV_POP(cluster_dist) AS sd_cluster_dist
FROM ImagesObjectFusion_odd;


-- =========================================================
-- 2) Cohesion by cluster (includes p90 via window ranking)
-- =========================================================
WITH even_ranked AS (
  SELECT
      cluster_id,
      cluster_dist,
      ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY cluster_dist) AS rn,
      COUNT(*) OVER (PARTITION BY cluster_id) AS n
  FROM ImagesObjectFusion
), even_p90 AS (
  SELECT cluster_id, MIN(cluster_dist) AS p90_dist
  FROM even_ranked
  WHERE rn >= CEIL(0.9 * n)
  GROUP BY cluster_id
), even_stats AS (
  SELECT cluster_id,
         COUNT(*) AS n_images,
         AVG(cluster_dist) AS mean_dist,
         STDDEV_POP(cluster_dist) AS sd_dist
  FROM ImagesObjectFusion
  GROUP BY cluster_id
), odd_ranked AS (
  SELECT
      cluster_id,
      cluster_dist,
      ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY cluster_dist) AS rn,
      COUNT(*) OVER (PARTITION BY cluster_id) AS n
  FROM ImagesObjectFusion_odd
), odd_p90 AS (
  SELECT cluster_id, MIN(cluster_dist) AS p90_dist
  FROM odd_ranked
  WHERE rn >= CEIL(0.9 * n)
  GROUP BY cluster_id
), odd_stats AS (
  SELECT cluster_id,
         COUNT(*) AS n_images,
         AVG(cluster_dist) AS mean_dist,
         STDDEV_POP(cluster_dist) AS sd_dist
  FROM ImagesObjectFusion_odd
  GROUP BY cluster_id
)
SELECT 'even' AS split_name, s.cluster_id, s.n_images, s.mean_dist, s.sd_dist, p.p90_dist
FROM even_stats s
JOIN even_p90 p USING (cluster_id)
UNION ALL
SELECT 'odd' AS split_name, s.cluster_id, s.n_images, s.mean_dist, s.sd_dist, p.p90_dist
FROM odd_stats s
JOIN odd_p90 p USING (cluster_id)
ORDER BY split_name, cluster_id;


-- =========================================================
-- 3) Mixedness proxy rate
-- Rule: cluster p90_dist > run-level p90_dist
-- lower is better
-- =========================================================
WITH even_ranked AS (
  SELECT cluster_id, cluster_dist,
         ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY cluster_dist) AS rn,
         COUNT(*) OVER (PARTITION BY cluster_id) AS n
  FROM ImagesObjectFusion
), even_cluster_p90 AS (
  SELECT cluster_id, MIN(cluster_dist) AS p90_dist
  FROM even_ranked
  WHERE rn >= CEIL(0.9 * n)
  GROUP BY cluster_id
), even_run_ranked AS (
  SELECT cluster_dist,
         ROW_NUMBER() OVER (ORDER BY cluster_dist) AS rn,
         COUNT(*) OVER () AS n
  FROM ImagesObjectFusion
), even_run_p90 AS (
  SELECT MIN(cluster_dist) AS run_p90
  FROM even_run_ranked
  WHERE rn >= CEIL(0.9 * n)
), odd_ranked AS (
  SELECT cluster_id, cluster_dist,
         ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY cluster_dist) AS rn,
         COUNT(*) OVER (PARTITION BY cluster_id) AS n
  FROM ImagesObjectFusion_odd
), odd_cluster_p90 AS (
  SELECT cluster_id, MIN(cluster_dist) AS p90_dist
  FROM odd_ranked
  WHERE rn >= CEIL(0.9 * n)
  GROUP BY cluster_id
), odd_run_ranked AS (
  SELECT cluster_dist,
         ROW_NUMBER() OVER (ORDER BY cluster_dist) AS rn,
         COUNT(*) OVER () AS n
  FROM ImagesObjectFusion_odd
), odd_run_p90 AS (
  SELECT MIN(cluster_dist) AS run_p90
  FROM odd_run_ranked
  WHERE rn >= CEIL(0.9 * n)
)
SELECT
  'even' AS split_name,
  SUM(CASE WHEN c.p90_dist > r.run_p90 THEN 1 ELSE 0 END) AS mixed_cluster_count,
  COUNT(*) AS total_clusters,
  SUM(CASE WHEN c.p90_dist > r.run_p90 THEN 1 ELSE 0 END) / COUNT(*) AS mixed_rate
FROM even_cluster_p90 c
CROSS JOIN even_run_p90 r
UNION ALL
SELECT
  'odd' AS split_name,
  SUM(CASE WHEN c.p90_dist > r.run_p90 THEN 1 ELSE 0 END) AS mixed_cluster_count,
  COUNT(*) AS total_clusters,
  SUM(CASE WHEN c.p90_dist > r.run_p90 THEN 1 ELSE 0 END) / COUNT(*) AS mixed_rate
FROM odd_cluster_p90 c
CROSS JOIN odd_run_p90 r;


-- =========================================================
-- 4) Cluster-size concentration
-- =========================================================
WITH even_sizes AS (
  SELECT cluster_id, COUNT(*) AS n
  FROM ImagesObjectFusion
  GROUP BY cluster_id
), even_tot AS (
  SELECT SUM(n) AS total_n FROM even_sizes
), odd_sizes AS (
  SELECT cluster_id, COUNT(*) AS n
  FROM ImagesObjectFusion_odd
  GROUP BY cluster_id
), odd_tot AS (
  SELECT SUM(n) AS total_n FROM odd_sizes
)
SELECT
  'even' AS split_name,
  MAX(s.n) AS largest_cluster,
  AVG(s.n) AS avg_cluster_size,
  STDDEV_POP(s.n) AS sd_cluster_size,
  SUM(POW(s.n / t.total_n, 2)) AS hhi_size_concentration
FROM even_sizes s
CROSS JOIN even_tot t
UNION ALL
SELECT
  'odd' AS split_name,
  MAX(s.n) AS largest_cluster,
  AVG(s.n) AS avg_cluster_size,
  STDDEV_POP(s.n) AS sd_cluster_size,
  SUM(POW(s.n / t.total_n, 2)) AS hhi_size_concentration
FROM odd_sizes s
CROSS JOIN odd_tot t;


-- =========================================================
-- 5) Object-slot fill consistency by cluster
-- higher is generally better
-- =========================================================
WITH even_joined AS (
  SELECT i.cluster_id,
         idt.left_hand_object_id,
         idt.right_hand_object_id,
         idt.top_face_object_id,
         idt.left_eye_object_id,
         idt.right_eye_object_id,
         idt.mouth_object_id,
         idt.shoulder_object_id,
         idt.waist_object_id,
         idt.feet_object_id
  FROM ImagesObjectFusion i
  LEFT JOIN ImagesDetections idt ON idt.image_id = i.image_id
), odd_joined AS (
  SELECT i.cluster_id,
         idt.left_hand_object_id,
         idt.right_hand_object_id,
         idt.top_face_object_id,
         idt.left_eye_object_id,
         idt.right_eye_object_id,
         idt.mouth_object_id,
         idt.shoulder_object_id,
         idt.waist_object_id,
         idt.feet_object_id
  FROM ImagesObjectFusion_odd i
  LEFT JOIN ImagesDetections idt ON idt.image_id = i.image_id
)
SELECT 'even' AS split_name,
       cluster_id,
       AVG(CASE WHEN left_hand_object_id IS NOT NULL THEN 1 ELSE 0 END) AS left_hand_fill,
       AVG(CASE WHEN right_hand_object_id IS NOT NULL THEN 1 ELSE 0 END) AS right_hand_fill,
       AVG(CASE WHEN top_face_object_id IS NOT NULL THEN 1 ELSE 0 END) AS top_face_fill,
       AVG(CASE WHEN mouth_object_id IS NOT NULL THEN 1 ELSE 0 END) AS mouth_fill,
       AVG(CASE WHEN shoulder_object_id IS NOT NULL THEN 1 ELSE 0 END) AS shoulder_fill,
       AVG(CASE WHEN waist_object_id IS NOT NULL THEN 1 ELSE 0 END) AS waist_fill,
       AVG(CASE WHEN feet_object_id IS NOT NULL THEN 1 ELSE 0 END) AS feet_fill
FROM even_joined
GROUP BY cluster_id
UNION ALL
SELECT 'odd' AS split_name,
       cluster_id,
       AVG(CASE WHEN left_hand_object_id IS NOT NULL THEN 1 ELSE 0 END) AS left_hand_fill,
       AVG(CASE WHEN right_hand_object_id IS NOT NULL THEN 1 ELSE 0 END) AS right_hand_fill,
       AVG(CASE WHEN top_face_object_id IS NOT NULL THEN 1 ELSE 0 END) AS top_face_fill,
       AVG(CASE WHEN mouth_object_id IS NOT NULL THEN 1 ELSE 0 END) AS mouth_fill,
       AVG(CASE WHEN shoulder_object_id IS NOT NULL THEN 1 ELSE 0 END) AS shoulder_fill,
       AVG(CASE WHEN waist_object_id IS NOT NULL THEN 1 ELSE 0 END) AS waist_fill,
       AVG(CASE WHEN feet_object_id IS NOT NULL THEN 1 ELSE 0 END) AS feet_fill
FROM odd_joined
GROUP BY cluster_id
ORDER BY split_name, cluster_id;


-- =========================================================
-- 6) Dominant class share per slot (example: left_hand)
-- higher is better
-- Repeat by swapping slot field as needed.
-- =========================================================
WITH even_slot AS (
  SELECT i.cluster_id, d.class_id
  FROM ImagesObjectFusion i
  JOIN ImagesDetections idt ON idt.image_id = i.image_id
  JOIN Detections d ON d.detection_id = idt.left_hand_object_id
), even_counts AS (
  SELECT cluster_id, class_id, COUNT(*) AS c
  FROM even_slot
  GROUP BY cluster_id, class_id
), even_tot AS (
  SELECT cluster_id, SUM(c) AS n
  FROM even_counts
  GROUP BY cluster_id
), even_ranked AS (
  SELECT c.cluster_id, c.class_id, c.c, t.n,
         ROW_NUMBER() OVER (PARTITION BY c.cluster_id ORDER BY c.c DESC) AS rn
  FROM even_counts c
  JOIN even_tot t ON t.cluster_id = c.cluster_id
), odd_slot AS (
  SELECT i.cluster_id, d.class_id
  FROM ImagesObjectFusion_odd i
  JOIN ImagesDetections idt ON idt.image_id = i.image_id
  JOIN Detections d ON d.detection_id = idt.left_hand_object_id
), odd_counts AS (
  SELECT cluster_id, class_id, COUNT(*) AS c
  FROM odd_slot
  GROUP BY cluster_id, class_id
), odd_tot AS (
  SELECT cluster_id, SUM(c) AS n
  FROM odd_counts
  GROUP BY cluster_id
), odd_ranked AS (
  SELECT c.cluster_id, c.class_id, c.c, t.n,
         ROW_NUMBER() OVER (PARTITION BY c.cluster_id ORDER BY c.c DESC) AS rn
  FROM odd_counts c
  JOIN odd_tot t ON t.cluster_id = c.cluster_id
)
SELECT 'even' AS split_name,
       cluster_id,
       class_id AS dominant_class_id,
       c / n AS dominant_class_share
FROM even_ranked
WHERE rn = 1
UNION ALL
SELECT 'odd' AS split_name,
       cluster_id,
       class_id AS dominant_class_id,
       c / n AS dominant_class_share
FROM odd_ranked
WHERE rn = 1
ORDER BY split_name, cluster_id;


-- =========================================================
-- 7) Reviewer sampling helper: top/middle/bottom/random clusters
-- Returns all best (10) + worst (15) + middle (~11) + random (10) clusters.
-- Random uses a subquery with its own LIMIT so it cannot crowd out deterministic rows.
-- Clusters with fewer than 20 images are excluded (too small for visual evaluation).
-- =========================================================
WITH per_cluster AS (
  SELECT cluster_id,
         COUNT(*) AS n_images,
         AVG(cluster_dist) AS mean_dist
  FROM ImagesObjectFusion
  GROUP BY cluster_id
  HAVING COUNT(*) >= 20
), ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (ORDER BY mean_dist ASC) AS rank_best,
         ROW_NUMBER() OVER (ORDER BY mean_dist DESC) AS rank_worst,
         COUNT(*) OVER () AS total_clusters
  FROM per_cluster
)
SELECT 'best' AS bucket, cluster_id, n_images, mean_dist
FROM ranked
WHERE rank_best <= 10
UNION ALL
SELECT 'worst' AS bucket, cluster_id, n_images, mean_dist
FROM ranked
WHERE rank_worst <= 15
UNION ALL
SELECT 'middle' AS bucket, cluster_id, n_images, mean_dist
FROM ranked
WHERE ABS(rank_best - FLOOR(total_clusters / 2)) <= 5
UNION ALL
SELECT 'random' AS bucket, cluster_id, n_images, mean_dist
FROM (
  SELECT cluster_id, n_images, mean_dist
  FROM per_cluster
  ORDER BY RAND()
  LIMIT 10
) AS rnd
ORDER BY bucket, mean_dist;


-- =========================================================
-- 8) Reviewer image panel helper for one cluster
-- Replace :cluster_id and :limit_n manually if your SQL client does not support params.
-- =========================================================
SELECT image_id, cluster_dist
FROM ImagesObjectFusion
WHERE cluster_id = :cluster_id
ORDER BY cluster_dist ASC
LIMIT :limit_n;

SELECT image_id, cluster_dist
FROM ImagesObjectFusion
WHERE cluster_id = :cluster_id
ORDER BY cluster_dist DESC
LIMIT :limit_n;


-- =========================================================
-- 9) Run-level summary view (single table for export)
-- =========================================================
WITH even AS (
  SELECT
    AVG(cluster_dist) AS cohesion_mean,
    STDDEV_POP(cluster_dist) AS cohesion_sd
  FROM ImagesObjectFusion
), odd AS (
  SELECT
    AVG(cluster_dist) AS cohesion_mean,
    STDDEV_POP(cluster_dist) AS cohesion_sd
  FROM ImagesObjectFusion_odd
)
SELECT
  even.cohesion_mean AS even_cohesion_mean,
  odd.cohesion_mean AS odd_cohesion_mean,
  odd.cohesion_mean - even.cohesion_mean AS cohesion_delta_abs,
  (odd.cohesion_mean - even.cohesion_mean) / NULLIF(ABS(even.cohesion_mean), 0) AS cohesion_delta_rel,
  even.cohesion_sd AS even_cohesion_sd,
  odd.cohesion_sd AS odd_cohesion_sd
FROM even, odd;


-- =========================================================
-- 10) Candidate comparison on EVEN only (baseline vs candidates)
--
-- How to use:
-- 1) Keep baseline row as ImagesObjectFusion.
-- 2) Replace candidate table names below with your run tables.
-- 3) Export this result to CSV and feed it to analysis/objectfusion_analyze_candidates.py
--
-- Expected per table schema:
--   (image_id, cluster_id, cluster_dist)
-- =========================================================
WITH candidate_tables AS (
  SELECT 'baseline' AS candidate_name, image_id, cluster_id, cluster_dist
  FROM ImagesObjectFusion

  UNION ALL
  SELECT 'candidate_A' AS candidate_name, image_id, cluster_id, cluster_dist
  FROM ImagesObjectFusion_candidate_A

  UNION ALL
  SELECT 'candidate_B' AS candidate_name, image_id, cluster_id, cluster_dist
  FROM ImagesObjectFusion_candidate_B
),
cluster_sizes AS (
  SELECT candidate_name, cluster_id, COUNT(*) AS n
  FROM candidate_tables
  GROUP BY candidate_name, cluster_id
),
cluster_size_rollup AS (
  SELECT
    candidate_name,
    MAX(n) AS largest_cluster,
    AVG(n) AS avg_cluster_size,
    STDDEV_POP(n) AS sd_cluster_size,
    SUM(POW(n / SUM(n) OVER (PARTITION BY candidate_name), 2)) AS hhi_size_concentration
  FROM cluster_sizes
  GROUP BY candidate_name
),
ranked_cluster_dist AS (
  SELECT
    candidate_name,
    cluster_id,
    cluster_dist,
    ROW_NUMBER() OVER (PARTITION BY candidate_name, cluster_id ORDER BY cluster_dist) AS rn,
    COUNT(*) OVER (PARTITION BY candidate_name, cluster_id) AS n
  FROM candidate_tables
),
cluster_p90 AS (
  SELECT candidate_name, cluster_id, MIN(cluster_dist) AS cluster_p90_dist
  FROM ranked_cluster_dist
  WHERE rn >= CEIL(0.9 * n)
  GROUP BY candidate_name, cluster_id
),
ranked_run_dist AS (
  SELECT
    candidate_name,
    cluster_dist,
    ROW_NUMBER() OVER (PARTITION BY candidate_name ORDER BY cluster_dist) AS rn,
    COUNT(*) OVER (PARTITION BY candidate_name) AS n
  FROM candidate_tables
),
run_p90 AS (
  SELECT candidate_name, MIN(cluster_dist) AS run_p90_dist
  FROM ranked_run_dist
  WHERE rn >= CEIL(0.9 * n)
  GROUP BY candidate_name
),
mixedness AS (
  SELECT
    c.candidate_name,
    SUM(CASE WHEN c.cluster_p90_dist > r.run_p90_dist THEN 1 ELSE 0 END) AS mixed_cluster_count,
    COUNT(*) AS total_clusters,
    SUM(CASE WHEN c.cluster_p90_dist > r.run_p90_dist THEN 1 ELSE 0 END) / COUNT(*) AS mixed_rate
  FROM cluster_p90 c
  JOIN run_p90 r ON r.candidate_name = c.candidate_name
  GROUP BY c.candidate_name
),
cohesion AS (
  SELECT
    candidate_name,
    COUNT(*) AS rows_n,
    COUNT(DISTINCT image_id) AS images_n,
    COUNT(DISTINCT cluster_id) AS clusters_n,
    AVG(cluster_dist) AS mean_cluster_dist,
    STDDEV_POP(cluster_dist) AS sd_cluster_dist
  FROM candidate_tables
  GROUP BY candidate_name
)
SELECT
  c.candidate_name,
  c.rows_n,
  c.images_n,
  c.clusters_n,
  c.mean_cluster_dist,
  c.sd_cluster_dist,
  m.mixed_cluster_count,
  m.total_clusters,
  m.mixed_rate,
  s.largest_cluster,
  s.avg_cluster_size,
  s.sd_cluster_size,
  s.hhi_size_concentration
FROM cohesion c
JOIN mixedness m ON m.candidate_name = c.candidate_name
JOIN cluster_size_rollup s ON s.candidate_name = c.candidate_name
ORDER BY c.candidate_name;
