-- for Taking Stock description artist book sketch

Use Stock;
SET GLOBAL innodb_buffer_pool_size=8053063680;


UPDATE Images
SET no_image  = NULL
WHERE image_id IN ()
;

