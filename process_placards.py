import cv2
import numpy as np
import os
from paddleocr import PaddleOCR
import re
from openai import OpenAI
from openaiAPI import api_key

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# importing project-specific models
import sys
from my_declarative_base import Slogans, ImagesSlogans
from tools_ocr import OCRTools
from tools_clustering import ToolsClustering
from tools_yolo import YOLOTools

# MySQL setup (preserving credentials framework)
from mp_db_io import DataIO
io = DataIO()
db = io.db
engine = create_engine(
    f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
    poolclass=NullPool
)
Session = sessionmaker(bind=engine)
session = Session()

openai_client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)

ocr_engine = PaddleOCR(
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False) # text detection + text recognition

ocr = OCRTools(DEBUGGING=True)
yolo = YOLOTools(DEBUGGING=True)

root_folder = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/labeled_images_nov19/"
images_folder = root_folder + "images_testing/"
labels_folder = root_folder + "labels/"
blank = False
DEBUGGING = True

CLUSTER_TYPE = "HSV" # only works with cluster save, not with assignment
VERBOSE = True
META = False # to return the meta clusters (out of 23, not 96)
cl = ToolsClustering(CLUSTER_TYPE, VERBOSE=VERBOSE)
table_cluster_type = cl.set_table_cluster_type(META)

Clusters, ImagesClusters, MetaClusters, ClustersMetaClusters = cl.construct_table_classes(table_cluster_type)
this_cluster, this_crosswalk = cl.set_cluster_metacluster(Clusters, ImagesClusters, MetaClusters, ClustersMetaClusters)
meta_cluster_dict = cl.get_meta_cluster_dict(session, ClustersMetaClusters)
print("this_cluster: ", this_cluster)

slogan_dict = ocr.get_all_slogans(session, Slogans)
print("Loaded slogans:", slogan_dict)

median_dict = cl.get_cluster_medians(session, this_cluster)
print("Loaded cluster medians:", median_dict)

# load all images in images_folder
image_filenames = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
print("Found image files:", image_filenames)

# loop through each image file
for image_filename in image_filenames:

    # this is all stuff for the temporary testing with local non DB files
    image_path = os.path.join(images_folder, image_filename)
    label_filename = image_filename.replace('.jpg', '.txt')
    label_path = os.path.join(labels_folder, label_filename)

    # Read the image
    image = cv2.imread(image_path)

    if not os.path.exists(label_path):
        print(f"Label file not found for image {image_filename}, skipping.")
        continue
    else:
        # Read the label file
        with open(label_path, 'r') as file:
            label_data = file.readlines()

    hue, saturation, value, luminance, cropped_image_bbox = yolo.get_image_bbox_hsv(image, label_data)
    print(f"Image {image_filename} - Hue: {hue}, Saturation: {saturation}, Value: {value}, Luminance: {luminance}")
    normalized_hue = hue/360
    normalized_saturation = saturation/255
    normalized_value = value/255
    normalized_luminance = luminance/255
    average_hsl = [normalized_hue, normalized_saturation, normalized_luminance]
    cluster_id, cluster_dist = cl.prep_pose_clusters_enc(average_hsl)
    meta_cluster_id = meta_cluster_dict.get(cluster_id, None)
    print(f"Image {image_filename} assigned to meta_cluster_id {meta_cluster_id} based on cluster_id {cluster_id} with distance {cluster_dist}.")

    # continue # skip OCR for now
    found_texts = ocr.ocr_on_cropped_image(cropped_image_bbox, ocr_engine, image_filename)
    print("Found Texts:", found_texts)  
    slogan_id = None
    if bool(found_texts) is False:
        print(f"No text found in image {image_filename}, will assign blank = True.")
        slogan_id = 1 # blank sign - no slogan
    else:
        slogan_id = ocr.check_existing_slogans(found_texts, slogan_dict)
        if slogan_id is None:
            refined_text = ocr.clean_ocr_text(openai_client, found_texts)
            print("No match, so refined Text:", refined_text)
            slogan_id = ocr.check_existing_slogans(refined_text, slogan_dict)

        if slogan_id is not None:
            print(f"Slogan already exists in DB with slogan_id: {slogan_id}.")
        else:
            # Save new slogan to DB
            slogan_id = ocr.save_slogan_text(session, Slogans, refined_text)
            slogan_dict[slogan_id] = refined_text
    # Here, save image_id, slogan_id info to Placards table
    if slogan_id is not None:
        print(f"Saving image {image_filename} with slogan_id {slogan_id} to ImagesSlogans table.")
        # this needs to wait for ImagesSlogans table to be created
        # ocr.save_images_slogans(session, ImagesSlogans, image_filename, slogan_id)
    else:
        print(f"Error: slogan_id is None for image {image_filename}, skipping save to Placards table.")

session.close()
engine.dispose()
