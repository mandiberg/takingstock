
USE stock;

-- AND i.imagename LIKE '/Volumes%' 

-- /Volumes/SSD4/images_getty_reDL/2/2c/1365472478.jpg
-- images/

-- /Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/getty_22222_us/images_usa_lastset/

-- Images and SegmentOct20

SELECT 
    i.image_id, 
    i.imagename
FROM Images i
WHERE i.site_name_id = 1
;



SELECT 
    i.image_id, 
    REPLACE(i.imagename, '/Volumes/SSD4/images_getty_reDL/', '') AS new_imagename
FROM Images i
WHERE i.site_name_id = 1
AND i.imagename LIKE '/Volumes/SSD4/images_getty_reDL/%'
LIMIT 10;


SELECT COUNT(i.image_id)
FROM Images i 
WHERE i.site_name_id = 1
AND i.imagename LIKE '/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/getty_22222_us/images_usa_lastset/%'
;



-- small updates

UPDATE Images
SET imagename = REPLACE(imagename, '/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/getty_22222_us/images_usa_lastset/', '')
WHERE site_name_id = 1
AND imagename LIKE '/Users/michaelmandiberg/Documents/projects-active/facemap_production/getty_scrape/getty_22222_us/images_usa_lastset/%'
LIMIT 750000
;


-- big updates

-- create procedure

