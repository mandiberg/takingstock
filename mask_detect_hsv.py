import cv2
import numpy as np  
import json
import os
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool    
from tools_yolo import YOLOTools
from mp_db_io import DataIO

from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, Boolean, DECIMAL, BLOB, JSON, String, Date, ForeignKey, update, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
# my ORM
from my_declarative_base import Base, Images, Encodings, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON


io = DataIO()
db = io.db

yolo = YOLOTools(DEBUGGING=True)

engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
))

# metadata = MetaData(engine)
metadata = MetaData() # apparently don't pass engine
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

image_folder = "/Volumes/LaCie/segment_images_mask/images_shutterstock/7/7A"
test_image = "/Volumes/LaCie/segment_images_mask/images_shutterstock/7/7A/1981048862.jpg"
all_images = io.get_img_list(image_folder)
site_name_id = 2  # Shutterstock
# for each image in folder, get the bbox
for img_name in all_images:
    site_image_id = img_name.split(".")[0]
    imagename, bbox = session.query(Images.imagename, Encodings.bbox) \
    .join(Images, Encodings.image_id == Images.image_id) \
    .filter(Images.site_image_id == site_image_id, Images.site_name_id == site_name_id).first()
    print("Image: ", img_name, " BBox: ", bbox)

# test_bbox =  "{\n    \"left\": 505,\n    \"right\": 890,\n    \"top\": 254,\n    \"bottom\": 638\n}"
    text_bbox_dict = io.unstring_json(bbox)
    print("text_bbox_dict: ", text_bbox_dict)
    print("type of text_bbox_dict: ", type(text_bbox_dict))

    image_path = os.path.join(image_folder, os.path.basename(img_name))
    # open the image using cv2
    image = cv2.imread(image_path)
    image_shape = image.shape
    print("image shape: ", image_shape)

    top_hsl, bot_hsl, hsl_distance = yolo.compute_mask_hsv(image, text_bbox_dict)
    print("Top Half HSL: ", top_hsl)
    print("Bottom Half HSL: ", bot_hsl)
    # now compute the distance between the two    
    print("HSL Distance: ", hsl_distance)

    # show the image with bbox
    left = text_bbox_dict['left']
    right = text_bbox_dict['right']
    top = text_bbox_dict['top']
    bottom = text_bbox_dict['bottom']
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.imshow("Image with BBox", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
session.close()
engine.dispose()



