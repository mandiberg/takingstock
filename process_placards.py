import cv2
import numpy as np
import pandas as pd
import os
from paddleocr import PaddleOCR
import re
from openai import OpenAI
import torch
from openaiAPI import api_key
import time
from sqlalchemy.exc import OperationalError
from ultralytics import YOLO
import json


import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# importing project-specific models
import sys
from my_declarative_base import Encodings, Images, Slogans, ImagesSlogans, Detections, NoDetections, NoDetectionsCustom
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

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
yolo_model = YOLO("yolov8x.pt").to(device)  # load a pretrained YOLOv8x model
yolo_custom_model = YOLO("models/takingstock_yolov8x/weights/best.pt").to(device)

ocr = OCRTools(DEBUGGING=True)
yolo = YOLOTools(DEBUGGING=True)


blank = False
DEBUGGING = True
DO_COCO = True
DO_CUSTOM = False
DO_MASK = False
DO_VALENTINE = False
DO_OCR = False

FILE_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/segment_images_book_clock_bowl"
# FILE_FOLDER = "/Volumes/OWC52/segment_images_money_cards"
MAKE_VIDEO_CSVS_PATH = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/heft_keyword_fusion_clusters/object_detection_csvs"
OUTPUT_FOLDER = os.path.join(FILE_FOLDER, "test_output")
BATCH_SIZE = 100
MASK_THRESHOLD = .15  # HSV distance threshold for mask detection
CONF_THRESHOLD = 0.25
IS_DRAW_BOX = True
IS_SAVE_UNDETECTED = True
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

folder_indexes = range(0, len(io.folder_list))  # adjust as needed
print("Processing folders with indexes:", folder_indexes)

def format_site_name_ids(folder_index, batch_img_list):
    if folder_index == 8:
    # 123rf
        batch_site_image_ids = []
        for img in batch_img_list:
            print("123rf img:", img)
            
            if "-id" in img:
                this_site_image_id = img.split("-id")[-1].replace(".jpg", "")
                batch_site_image_ids.append(this_site_image_id)
            elif "-" in img:
                this_site_image_id = img.split("-")[-1].replace(".jpg", "")
                batch_site_image_ids.append(this_site_image_id)
            else:
                this_site_image_id = img.replace(".jpg", "")
                batch_site_image_ids.append(this_site_image_id)
        print("123rf batch_site_image_ids", batch_site_image_ids[:5])
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

def bbox_to_cluster_id(image, bbox=None):
    if bbox is None: 
        image_bbox_slice = image
    else:
        # slice out the bbox area from the image
        image_bbox_slice = image[bbox['top']:bbox['bottom'], bbox['left']:bbox['right']]
    normalized_hue, normalized_saturation, normalized_value, normalized_luminance,image_bbox_slice = yolo.get_image_bbox_hsv(image_bbox_slice)
    # print(f"Image {image_id} - Hue: {normalized_hue}, Saturation: {normalized_saturation}, Value: {normalized_value}, Luminance: {normalized_luminance}")
    average_hsl = [normalized_hue, normalized_saturation, normalized_luminance]
    cluster_id, cluster_dist = cl.prep_pose_clusters_enc(average_hsl)
    meta_cluster_id = meta_cluster_dict.get(cluster_id, None)

    return meta_cluster_id, cluster_id, cluster_dist

def mask_to_cluster_id(image, face_bbox):
    width = face_bbox['right'] - face_bbox['left']
    height = face_bbox['bottom'] - face_bbox['top']
    mask_bbox = {
            'left': face_bbox['left'] + width//4,
            'right': face_bbox['right'] - width//4,
            'top': face_bbox['top'] + height//2 + height//8,
            'bottom': face_bbox['bottom'] - height//4
        }
    meta_cluster_id, cluster_id, cluster_dist = bbox_to_cluster_id(image, mask_bbox)
    return meta_cluster_id, cluster_id, cluster_dist

def save_debug_image(output_image_path, image, imagename):
    # save image to OUTPUT_FOLDER for review
    if not os.path.exists(os.path.dirname(output_image_path)):
        os.makedirs(os.path.dirname(output_image_path))
    cv2.imwrite(output_image_path, image)
    print(f"Image {imagename} no detections, saved to {output_image_path}. ")
def draw_bbox_on_image(image, bbox):
    left = bbox['left']
    right = bbox['right']
    top = bbox['top']
    bottom = bbox['bottom']
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    return image

def get_existing_detections(image_id, class_id=None):
    if class_id is not None:
        existing_detections_query = session.query(Detections).filter(Detections.image_id == image_id, Detections.class_id == class_id)
    else:
        existing_detections_query = session.query(Detections).filter(Detections.image_id == image_id)
    existing_detections = existing_detections_query.all()
    return existing_detections

def get_existing_no_detections(image_id, NoDetectionTable=NoDetections):
    existing_no_detections_query = session.query(NoDetectionTable).filter(NoDetectionTable.image_id == image_id)
    existing_no_detections = existing_no_detections_query.all()
    return existing_no_detections

def save_no_dectections(session,image_id, NoDetectionTable=NoDetections):
    # save to NoDetections table
    new_no_detection = NoDetectionTable(
        image_id=image_id
    )
    session.add(new_no_detection)
    session.commit()
    print(f"Image {image_id} - No detections found, saved to {NoDetectionTable} table.")

def assign_hsv_detect_results(detect_results, image):
    for result_dict in detect_results:
        bbox = io.unstring_json(result_dict['bbox'])
        meta_cluster_id, cluster_id, cluster_dist = bbox_to_cluster_id(image, bbox)
        result_dict['meta_cluster_id'] = meta_cluster_id
        result_dict['cluster_id'] = cluster_id
        # result_dict['cluster_dist'] = cluster_dist
    return detect_results

def do_yolo_detections(result, image, existing_detections, custom=False):
    if custom: 
        ThisNoDetectionTable=NoDetectionsCustom
        existing_detection_ids = [det.detection_id for det in existing_detections if det.class_id >= 80]
        this_yolo_model = yolo_custom_model
    else:
        ThisNoDetectionTable=NoDetections
        existing_detection_ids = [det.detection_id for det in existing_detections if det.class_id < 80]
        this_yolo_model = yolo_model
    existing_no_detections = get_existing_no_detections(result.image_id, NoDetectionTable=ThisNoDetectionTable)

    image_id = result.image_id
    imagename = result.imagename
    if len(existing_no_detections) > 0:
        print(f"Skipping image_id {result.image_id} due to existing no detections record.")
        return
    elif len(existing_detection_ids) > 0:
        print(f"Skipping image_id {result.image_id} due to existing detections record.")
        return
        # TK uncomment this when no longer testing and handle custom = False
        # for det_id in existing_detection_ids:
            # if det_id > 12455146:
            #     print(f"Skipping image_id {result.image_id} due to first round detection_id {det_id} > 12455146")
            #     return
    else:
        # YOLO object detection
        unrefined_detect_results = yolo.detect_objects_return_bbox(this_yolo_model,image, device, conf_thresh=CONF_THRESHOLD)
        detect_results = yolo.merge_yolo_detections(unrefined_detect_results, iou_threshold=0.3, adjacency_threshold_px=50)
        detect_results = assign_hsv_detect_results(detect_results, image)
        print(f"Image {image_id} - YOLO detections: {detect_results}")
        # save_debug_image_yolo_bbox(image_id, imagename, image, detect_results)
        save_debug_image_yolo_bbox(image_id, imagename, image, detect_results, draw_box=IS_DRAW_BOX, save_undetected=IS_SAVE_UNDETECTED)
        if len(detect_results) == 0:
            save_no_dectections(session,image_id, NoDetectionTable=ThisNoDetectionTable)
            return
        else:
            yolo.save_obj_bbox(session, image_id, detect_results, Detections)
    return detect_results

def do_detections(result, folder_index):

    existing_detections = get_existing_detections(result.image_id)
    
    image_id = result.image_id
    imagename = result.imagename
    try:
        face_bbox = io.unstring_json(result.bbox)
    except Exception as e:
        print(f"Error parsing label_data JSON for image_id {image_id}: {e}")
        print("Skipping this face_bbox.", result.bbox)
        return
    image_path = os.path.join(FILE_FOLDER, os.path.basename(io.folder_list[folder_index]), imagename)
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}. Skipping.")
        return

    print(f"Processing image_id: {image_id}, imagename: {imagename}")
    if DO_COCO: 
        coco_detections = do_yolo_detections(result, image, existing_detections, custom=False)

    if DO_CUSTOM: 
        custom_detections = do_yolo_detections(result, image, existing_detections, custom=True)

    if DO_VALENTINE:
        pass

    if DO_MASK:
        # detect facemask via hsv distance and clustering
        top_hsl, bot_hsl, hsl_distance = yolo.compute_mask_hsv(image, face_bbox)
        meta_cluster_id, cluster_id, cluster_dist = mask_to_cluster_id(image, face_bbox)
    
        # for debugging mask, saving to folders
        # debug_file_name = f"{hsl_distance:.2f}_{image_id}_mask_debug.jpg"
        # if hsl_distance > MASK_THRESHOLD: output_image_path = os.path.join(OUTPUT_FOLDER, str(meta_cluster_id),debug_file_name)
        # else: output_image_path = os.path.join(OUTPUT_FOLDER, "no_mask",debug_file_name)
        # # save image to OUTPUT_FOLDER for review
        # save_debug_image(output_image_path, image, imagename)

    if DO_OCR and DO_CUSTOM and custom_detections is not None:
        sign_detections = [det for det in custom_detections if det['class_id'] == 80]
        for detection in sign_detections:
            bbox = io.unstring_json(detection['bbox'])
            cropped_image_bbox = image[bbox['top']:bbox['bottom'], bbox['left']:bbox['right']]
            print(f"Image {image_id} - Performing OCR on cropped image with bbox: {bbox}")

            # ACTION gets cluster assignment info
            # will need to streamline the bbox format
            # this is now handled in the yolo funciton
            # normalized_hue, normalized_saturation, normalized_value, normalized_luminance, cropped_image_bbox = yolo.get_image_bbox_hsv(image, label_data)
            # print(f"Image {image_id} - Hue: {normalized_hue}, Saturation: {normalized_saturation}, Value: {normalized_value}, Luminance: {normalized_luminance}")
            # average_hsl = [normalized_hue, normalized_saturation, normalized_luminance]
            # cluster_id, cluster_dist = cl.prep_pose_clusters_enc(average_hsl)
            # meta_cluster_id = meta_cluster_dict.get(cluster_id, None)
            # print(f"Image {image_id} assigned to meta_cluster_id {meta_cluster_id} based on cluster_id {cluster_id} with distance {cluster_dist}.")

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

def save_debug_image_yolo_bbox(image_id, imagename, image, detect_results, draw_box=True, save_undetected=True):
    if not detect_results and save_undetected:
        output_image_path = os.path.join(OUTPUT_FOLDER, "no_detections",imagename)
        save_debug_image(output_image_path, image, imagename)
    else:
        # Group detections by class_id
        detections_by_class = {}
        for result_dict in detect_results:
            class_id = result_dict['class_id']
            if class_id == 0:  # skip person class
                continue
            if class_id not in detections_by_class:
                detections_by_class[class_id] = []
            detections_by_class[class_id].append(result_dict)
        
        # Process each class
        for class_id, class_detections in detections_by_class.items():
            drawable_image = image.copy()
            print(f"Processing class_id {class_id} with {len(class_detections)} detection(s)")
            all_class_conf = []
            for result_dict in class_detections:

                print(f"  Detected class: {result_dict['class_id']} with bbox: {result_dict['bbox']} and confidence: {result_dict['conf']}")
                # for debugging, saving to folders
                if draw_box:
                    drawable_image = draw_bbox_on_image(drawable_image, io.unstring_json(result_dict['bbox']))
                all_class_conf.append(result_dict['conf'])
            avg_conf = sum(all_class_conf) / len(all_class_conf)
            debug_file_name = f"{avg_conf:.2f}_{image_id}_YOLO_debug.jpg"
            output_image_path = os.path.join(OUTPUT_FOLDER, str(result_dict['class_id']), debug_file_name)
            save_debug_image(output_image_path, drawable_image, imagename)

def load_csvs(folder_path):
    # get a list of all files in folder_path
    folder_files = os.listdir(folder_path)
    csv_files = [f for f in folder_files if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in {folder_path}.")
    # load each csv into a dataframe and concatenate them
    df_list = []
    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(csv_path)
        df_list.append(df)
    df_csvs = pd.concat(df_list, ignore_index=True)
    print(f"Combined CSVs into a single dataframe with {len(df_csvs)} rows.")
    return df_csvs

def main():
    if MAKE_VIDEO_CSVS_PATH is not None:
        df_csvs = load_csvs(MAKE_VIDEO_CSVS_PATH)
    else:
        df_csvs = pd.DataFrame(columns=['site_name_id','site_image_id'])  
    print("lenth df_csvs:", len(df_csvs))     
    folder_count = 0
    for folder_index in folder_indexes:
        df_csvs_folder = df_csvs[df_csvs['site_name_id'] == folder_index]
        if MAKE_VIDEO_CSVS_PATH is not None and len(df_csvs_folder) == 0:
            print("Skipping folder_index:", folder_index, "no entries in MAKE_VIDEO_CSVS_PATH")
            continue
        else:
            print("meets or ignores MAKE_VIDEO_CSVS_PATH, folder_index:", folder_index, "len df_csvs_folder:", len(df_csvs_folder))
        site_folder = os.path.join(FILE_FOLDER, os.path.basename(io.folder_list[folder_index]))
        print(f"folder_index: {folder_index}, site_folder: {site_folder}")
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
                    
                detect_from_folder(folder_index, csv_foldercount_path, folder_path, folder, df_csvs_folder)

    session.close()
    engine.dispose()

def detect_from_folder(folder_index, csv_foldercount_path, folder_path, folder, df_csvs_folder):
    img_list = io.get_img_list(folder, force_ls=True)
    print("len(img_list)", len(img_list))
    if len(df_csvs_folder) > 0:
        img_list = [img for img in img_list if img.split(".")[0] in df_csvs_folder['site_image_id'].values]
        print("after filtering with MAKE_VIDEO_CSVS_PATH, len(img_list)", len(img_list))

                # Initialize an empty list to store all the results
    all_results = []

                # Split the img_list into smaller batches and process them one by one
    for i in range(0, len(img_list), BATCH_SIZE):
        tasks_in_this_round = 0

        batch_img_list = img_list[i : i + BATCH_SIZE]
        batch_site_image_ids = format_site_name_ids(folder_index, batch_img_list)

        print(f"total img_list: {len(img_list)} no. processed: {i} no. left: {len(img_list)-i}")
        print(f"folder_index: {folder_index}, {batch_site_image_ids} images.")
        if len(img_list)-i<BATCH_SIZE: print("last_round for img_list")

                    # query the database for the current batch and return image_id and encoding_id
        for _ in range(io.max_retries):
            try:
                if VERBOSE: print(f"Processing batch {i//BATCH_SIZE + 1}...")
                            # init_session()

                            # get Images and Encodings values for each site_image_id in the batch
                            # adding in mongo stuff. should return NULL if not there
                batch_query = session.query(Images.image_id, Images.site_image_id, Images.imagename, Encodings.bbox) \
                                .join(Encodings, Encodings.image_id == Images.image_id) \
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
                print(f"site_name_id: {folder_index} site_image_id: {site_image_id} not found in DB, skipping.")
                continue
                # save  success to csv_foldercount_path
    io.write_csv(csv_foldercount_path, [folder_path])

if __name__ == "__main__":
    main()
