USE stock;
SET GLOBAL innodb_buffer_pool_size=8053063680;



SELECT 
    ihp.cluster_id AS ihp_cluster,
	COUNT(so.image_id)
FROM 
    SegmentBig_isface so
JOIN 
    SegmentHelper_sept2025_heft_keywords sh ON sh.image_id = so.image_id
JOIN 
    ImagesBodyPoses3D ihp ON ihp.image_id = so.image_id
JOIN 
    ImagesKeywords it ON it.image_id = so.image_id
WHERE it.keyword_id = 5310
GROUP BY
    ihp.cluster_id
ORDER BY 
    ihp_cluster;



SELECT 
    ibp.cluster_id AS ibp_cluster,
    SUM(CASE WHEN ik.keyword_id = 184 THEN 1 ELSE 0 END) AS ik_184,
    SUM(CASE WHEN ik.keyword_id = 22411 THEN 1 ELSE 0 END) AS ik_22411,
    SUM(CASE WHEN ik.keyword_id = 1991 THEN 1 ELSE 0 END) AS ik_1991,
    SUM(CASE WHEN ik.keyword_id = 220 THEN 1 ELSE 0 END) AS ik_220,
    SUM(CASE WHEN ik.keyword_id = 22269 THEN 1 ELSE 0 END) AS ik_22269,
    SUM(CASE WHEN ik.keyword_id = 5271 THEN 1 ELSE 0 END) AS ik_5271,
    SUM(CASE WHEN ik.keyword_id = 827 THEN 1 ELSE 0 END) AS ik_827,
    SUM(CASE WHEN ik.keyword_id = 1070 THEN 1 ELSE 0 END) AS ik_1070,
    SUM(CASE WHEN ik.keyword_id = 22412 THEN 1 ELSE 0 END) AS ik_22412,
    SUM(CASE WHEN ik.keyword_id = 404 THEN 1 ELSE 0 END) AS ik_404,
    SUM(CASE WHEN ik.keyword_id = 553 THEN 1 ELSE 0 END) AS ik_553,
    SUM(CASE WHEN ik.keyword_id = 22961 THEN 1 ELSE 0 END) AS ik_22961,
    SUM(CASE WHEN ik.keyword_id = 3856 THEN 1 ELSE 0 END) AS ik_3856,
    SUM(CASE WHEN ik.keyword_id = 6286 THEN 1 ELSE 0 END) AS ik_6286,
    SUM(CASE WHEN ik.keyword_id = 807 THEN 1 ELSE 0 END) AS ik_807,
    SUM(CASE WHEN ik.keyword_id = 1644 THEN 1 ELSE 0 END) AS ik_1644,
    SUM(CASE WHEN ik.keyword_id = 5310 THEN 1 ELSE 0 END) AS ik_5310,
    SUM(CASE WHEN ik.keyword_id = 22251 THEN 1 ELSE 0 END) AS ik_22251,
    SUM(CASE WHEN ik.keyword_id = 8911 THEN 1 ELSE 0 END) AS ik_8911
FROM SegmentBig_isface so
JOIN SegmentHelper_sept2025_heft_keywords sh ON sh.image_id = so.image_id
JOIN ImagesBodyPoses3D ibp ON ibp.image_id = so.image_id
JOIN ImagesKeywords ik ON ik.image_id = so.image_id
GROUP BY
    ibp.cluster_id
ORDER BY 
    ibp.cluster_id;






SELECT 
    ibp.cluster_id AS ihp_cluster,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 0 THEN 1 ELSE 0 END) AS hsv_0,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 1 THEN 1 ELSE 0 END) AS hsv_1,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 2 THEN 1 ELSE 0 END) AS hsv_2,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 3 THEN 1 ELSE 0 END) AS hsv_3,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 4 THEN 1 ELSE 0 END) AS hsv_4,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 5 THEN 1 ELSE 0 END) AS hsv_5,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 6 THEN 1 ELSE 0 END) AS hsv_6,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 7 THEN 1 ELSE 0 END) AS hsv_7,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 8 THEN 1 ELSE 0 END) AS hsv_8,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 9 THEN 1 ELSE 0 END) AS hsv_9,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 10 THEN 1 ELSE 0 END) AS hsv_10,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 11 THEN 1 ELSE 0 END) AS hsv_11,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 12 THEN 1 ELSE 0 END) AS hsv_12,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 13 THEN 1 ELSE 0 END) AS hsv_13,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 14 THEN 1 ELSE 0 END) AS hsv_14,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 15 THEN 1 ELSE 0 END) AS hsv_15,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 16 THEN 1 ELSE 0 END) AS hsv_16,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 17 THEN 1 ELSE 0 END) AS hsv_17,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 18 THEN 1 ELSE 0 END) AS hsv_18,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 19 THEN 1 ELSE 0 END) AS hsv_19,    
    SUM(CASE WHEN cmhsv.meta_cluster_id = 20 THEN 1 ELSE 0 END) AS hsv_20,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 21 THEN 1 ELSE 0 END) AS hsv_21,
    SUM(CASE WHEN cmhsv.meta_cluster_id = 22 THEN 1 ELSE 0 END) AS hsv_22
FROM SegmentBig_isface so
JOIN SegmentHelper_sept2025_heft_keywords sh ON sh.image_id = so.image_id
JOIN ImagesBodyPoses3D ibp ON ibp.image_id = so.image_id
JOIN ImagesHSV ihsv ON ihsv.image_id = so.image_id
JOIN ClustersMetaHSV cmhsv ON cmhsv.cluster_id = ihsv.cluster_id
JOIN ImagesKeywords it ON it.image_id = so.image_id
WHERE it.keyword_id = 22411
GROUP BY
    ibp.cluster_id
ORDER BY 
    ibp.cluster_id;

SELECT COUNT(ibp.image_id)
FROM SegmentBig_isface so
JOIN SegmentHelper_sept2025_heft_keywords sh ON sh.image_id = so.image_id
JOIN ImagesBodyPoses3D ibp ON ibp.image_id = so.image_id
JOIN ImagesHSV ihsv ON ihsv.image_id = so.image_id
JOIN ClustersMetaHSV cmhsv ON cmhsv.cluster_id = ihsv.cluster_id
JOIN ImagesKeywords it ON it.image_id = so.image_id
WHERE it.keyword_id = 22411
AND ibp.cluster_id = 24
AND cmhsv.meta_cluster_id = 1


SELECT DISTINCT(s.image_id), s.site_name_id, s.contentUrl, s.imagename, s.description, s.face_x, s.face_y, s.face_z, s.mouth_gap, s.bbox, s.site_image_id,
ibg.lum, ibg.lum_bb, ibg.hue, ibg.hue_bb, ibg.sat, ibg.sat_bb, ibg.val, ibg.val_bb, ibg.lum_torso, ibg.lum_torso_bb , pb.bbox_67, pb.conf_67 
FROM SegmentBig_isface s  JOIN Encodings e ON s.image_id = e.image_id  
JOIN ImagesBodyPoses3D ihp ON s.image_id = ihp.image_id  
JOIN ImagesKeywords it ON s.image_id = it.image_id  
JOIN SegmentHelper_sept2025_heft_keywords sh ON s.image_id = sh.image_id  
JOIN ImagesBackground ibg ON s.image_id = ibg.image_id  
JOIN PhoneBbox pb ON s.image_id = pb.image_id   
JOIN ImagesHSV ihsv ON s.image_id = ihsv.image_id  
JOIN ClustersMetaHSV cmhsv ON cmhsv.cluster_id = ihsv.cluster_id
WHERE  e.is_dupe_of IS NULL  AND s.age_id NOT IN (1,2,3)  AND pb.bbox_67 IS NOT NULL   AND cmhsv.meta_cluster_id  = 1    
AND ihp.cluster_id = 24 AND it.keyword_id  IN (22101, 444, 22191, 16045, 11549, 133300, 133777, 22411)  
LIMIT 2000;


SELECT COUNT(s.image_id)
FROM SegmentBig_isface s  JOIN Encodings e ON s.image_id = e.image_id  
JOIN ImagesBodyPoses3D ihp ON s.image_id = ihp.image_id  
JOIN ImagesKeywords it ON s.image_id = it.image_id  
JOIN SegmentHelper_sept2025_heft_keywords sh ON s.image_id = sh.image_id  
JOIN ImagesBackground ibg ON s.image_id = ibg.image_id  
JOIN PhoneBbox pb ON s.image_id = pb.image_id   
JOIN ImagesHSV ihsv ON s.image_id = ihsv.image_id 
JOIN ClustersMetaHSV cmhsv ON ihsv.cluster_id = cmhsv.cluster_id  
WHERE  e.is_dupe_of IS NULL  AND s.age_id NOT IN (1,2,3)     
AND cmhsv.meta_cluster_id  = 1    AND ihp.cluster_id = 24 AND it.keyword_id  IN (22101, 444, 22191, 16045, 11549, 133300, 133777, 22411);





-- JOIN ImagesHSV ihsv ON ihsv.image_id = so.image_id
'''
JOIN Images{MODE} it ON it.image_id = so.image_id
WHERE it.{MODE_ID} = {THIS_MODE_ID}


    
FROM 
    SegmentOct20 so
JOIN 
    ImagesHandsPositions ihp ON ihp.image_id = so.image_id
JOIN 
    ImagesHandsGestures ihg ON ihg.image_id = so.image_id
JOIN 
    ImagesKeywords it ON it.image_id = so.image_id
WHERE it.keyword_id = 5310
GROUP BY
    ihp.cluster_id
ORDER BY 
    ihp_cluster;
...


