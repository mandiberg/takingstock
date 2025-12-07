import cv2
import numpy as np
import os
from paddleocr import PaddleOCR
import re
from openai import OpenAI
from openaiAPI import api_key
import time
from sqlalchemy.exc import OperationalError

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# importing project-specific models
import sys
from my_declarative_base import Images, Slogans, ImagesSlogans
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


blank = False
DEBUGGING = True

FILE_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/segment_images_mask"
BATCH_SIZE = 10
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

folder_indexes = range(1, len(io.folder_list))  # adjust as needed
print("Processing folders with indexes:", folder_indexes)

def format_site_name_ids(folder_index, batch_img_list):
    if folder_index == 8:
    # 123rf
        batch_site_image_ids = [img.split("-")[0] for img in batch_img_list]
    elif folder_index == 5:
        batch_site_image_ids = [img.split("-")[-1].replace(".jpg","") for img in batch_img_list]
        print("pexels batch_site_image_ids", batch_site_image_ids[:5])
    elif folder_index == 1:
    # gettyimages
        batch_site_image_ids = [img.split("-id")[-1].replace(".jpg", "") for img in batch_img_list]
    else:
    # # Adobe and pexels and shutterstock and istock
        batch_site_image_ids = [img.split(".")[0] for img in batch_img_list]
    return batch_site_image_ids

def do_detections(result, folder_index):
    image_id = result.image_id
    imagename = result.imagename
    
    image_path = os.path.join(FILE_FOLDER, os.path.basename(io.folder_list[folder_index]), imagename)
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}. Skipping.")
        return

    print(f"Processing image_id: {image_id}, imagename: {imagename}")

    return
    # ACTION gets cluster assignment info
    # will need to streamline the bbox format
    normalized_hue, normalized_saturation, normalized_value, normalized_luminance, cropped_image_bbox = yolo.get_image_bbox_hsv(image, label_data)
    print(f"Image {image_id} - Hue: {normalized_hue}, Saturation: {normalized_saturation}, Value: {normalized_value}, Luminance: {normalized_luminance}")
    average_hsl = [normalized_hue, normalized_saturation, normalized_luminance]
    cluster_id, cluster_dist = cl.prep_pose_clusters_enc(average_hsl)
    meta_cluster_id = meta_cluster_dict.get(cluster_id, None)
    print(f"Image {image_id} assigned to meta_cluster_id {meta_cluster_id} based on cluster_id {cluster_id} with distance {cluster_dist}.")

    # ACTION OCR on cropped image
    # return # skip OCR for now
    found_texts = ocr.ocr_on_cropped_image(cropped_image_bbox, ocr_engine, image_id)
    print("Found Texts:", found_texts)
    # assign slogan_id
    slogan_id = ocr.assign_slogan_id(session, openai_client, Slogans, slogan_dict, image_id, found_texts)
    # Here, save image_id, slogan_id info to Placards table
    if slogan_id is not None:
        print(f"Saving image {image_id} with slogan_id {slogan_id} to ImagesSlogans table.")
        # this needs to wait for ImagesSlogans table to be created
        # ocr.save_images_slogans(session, ImagesSlogans, image_id, slogan_id)
    else:
        print(f"Error: slogan_id is None for image {image_id}, skipping save to Placards table.")

def main():
    folder_count = 0
    for folder_index in folder_indexes:
        site_folder = os.path.join(FILE_FOLDER, os.path.basename(io.folder_list[folder_index]))
        folder_paths = io.make_hash_folders(site_folder, as_list=True)
        csv_foldercount_path = os.path.join(site_folder, "foldercount.csv")
        if not os.path.exists(csv_foldercount_path): completed_folders = []
        else: completed_folders = io.get_csv_aslist(csv_foldercount_path)
        if not os.path.exists(site_folder):
            print("no folder here:",site_folder)
            continue
        print(site_folder, len(completed_folders))
        for folder_path in folder_paths:
            print("checking folder_path: ", folder_path)
            if folder_path not in completed_folders:
                folder = os.path.join(site_folder,folder_path)
                folder_count += 1
                if not os.path.exists(folder):
                    print(str(folder_count), "no folder here:",folder)
                    continue
                else:
                    print(str(folder_count), folder)
                    
                img_list = io.get_img_list(folder, force_ls=True)
                print("len(img_list)", len(img_list))

                # Initialize an empty list to store all the results
                all_results = []

                # Split the img_list into smaller batches and process them one by one
                for i in range(0, len(img_list), BATCH_SIZE):
                    tasks_in_this_round = 0

                    batch_img_list = img_list[i : i + BATCH_SIZE]
                    batch_site_image_ids = format_site_name_ids(folder_index, batch_img_list)

                    print(f"total img_list: {len(img_list)} no. processed: {i} no. left: {len(img_list)-i}")
                    if len(img_list)-i<BATCH_SIZE: print("last_round for img_list")

                    # query the database for the current batch and return image_id and encoding_id
                    for _ in range(io.max_retries):

                        try:
                            if VERBOSE: print(f"Processing batch {i//BATCH_SIZE + 1}...")
                            # init_session()

                            # get Images and Encodings values for each site_image_id in the batch
                            # adding in mongo stuff. should return NULL if not there
                            batch_query = session.query(Images.image_id, Images.site_image_id, Images.imagename) \
                                                .filter(Images.site_image_id.in_(batch_site_image_ids), Images.site_name_id == folder_index)
                                                                                                        
                            batch_results = batch_query.all()

                            all_results.extend(batch_results)
                            if VERBOSE: print("about to close_session()")
                            # # Close the session and dispose of the engine before the worker process exits
                            # close_session()

                        except OperationalError as e:
                            print("error getting batch results")
                            print(e)
                            time.sleep(io.retry_delay)

                    if VERBOSE: print(f"no. all_results: {len(all_results)}")

                    # print("results:", all_results)
                    results_dict = {result.site_image_id: result for result in batch_results}

                    # going back through the img_list, to use as key for the results_dict

                    images_left_to_process = len(batch_site_image_ids)
                    for site_image_id in batch_site_image_ids:
                        print(f"image_id to process: {site_image_id}, images left to process in batch: {images_left_to_process}")
                        images_left_to_process -= 1
                        if site_image_id in results_dict:
                            do_detections(results_dict[site_image_id], folder_index)
                            print(f"Found image_id: {results_dict[site_image_id].image_id} for site_image_id: {site_image_id}, imagename: {results_dict[site_image_id].imagename}")
                        else:
                            print(f"site_image_id: {site_image_id} not found in DB, skipping.")
                            continue
                # save success to csv_foldercount_path
                io.write_csv(csv_foldercount_path, [folder_path])

    session.close()
    engine.dispose()

if __name__ == "__main__":
    main()
