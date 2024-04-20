-- attempted to run Mar23, and quit after 300s

-- ran successfully on april 14. 

Use Stock;

SELECT
    i.site_name_id,
    COUNT(i.image_id) AS image_count,
    COUNT(e.encoding_id) AS encoding_count,
    SUM(e.is_face) AS is_face_count,
    SUM(CASE WHEN e.face_encodings68 IS NOT NULL THEN 1 ELSE 0 END) AS face_encodings68_not_null_count
FROM
    Images i
LEFT JOIN
    Encodings e ON i.image_id = e.image_id
GROUP BY
    i.site_name_id
ORDER BY
    i.site_name_id;
