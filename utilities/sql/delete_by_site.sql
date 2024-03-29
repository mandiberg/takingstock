use stock;

DELETE FROM ImagesKeywords
WHERE image_id IN (
  SELECT image_id
  FROM Images
  WHERE site_name_id = 8
);

DELETE FROM ImagesEthnicity
WHERE image_id IN (
  SELECT image_id
  FROM Images
  WHERE site_name_id = 8
);

DELETE FROM Images
WHERE site_name_id = 8;
