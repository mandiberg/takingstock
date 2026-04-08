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
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.exc import OperationalError
from ultralytics import YOLO
import json


import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# importing project-specific models
import sys
from my_declarative_base import Encodings, Images, Slogans, ImagesSlogans, RefinedText, Detections, NoDetections, NoDetectionsCustom
from tools_ocr import OCRTools
from tools_clustering import ToolsClustering
from tools_yolo import YOLOTools

# MySQL setup (preserving credentials framework)
from mp_db_io import DataIO
io = DataIO()
db = io.db
engine = create_engine(
    f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
    pool_pre_ping=True,
    pool_recycle=600,
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
yolo_custom_model = YOLO("models/takingstock_c36_v1_yolo26x/weights/best.pt").to(device)
# yolo_custom_model = YOLO("models/takingstock_c11v3_yolov8m/weights/best.pt").to(device)

VERBOSE = False
ocr = OCRTools(DEBUGGING=True)
yolo = YOLOTools(DEBUGGING=True, VERBOSE=VERBOSE)


blank = False
DEBUGGING = False # saves debug images (option for bboxes drawn)
SAVE_NEW_LABELS = False # saves new yolo labels to feed back into training data
SAVE_NODETECTIONS_JPG_FILES = False
TESTING_NO_DB_WRITE = False # if True, will not write to database
DO_COCO = True
DO_CUSTOM = True
DO_OCR = False
CLASSES_TO_COMBINE = [89,90]

# False means reruns skip images that already have detections/no-detections state.
OVERWRITE_EXISTING_DETECTIONS_COCO = False
OVERWRITE_EXISTING_DETECTIONS_CUSTOM = False
# False means custom no-detections are treated as complete and skipped on reruns.
IGNORE_EXISTING_NO_DETECTIONS = False
DET_ID_THRESHOLD_CUSTOM = 59955150
DET_ID_THRESHOLD_COCO = 12455146
# this is for merging books and stuff, but it messes up cucumbers. 
IOU_THRESHOLD = 0.7
ADJACENCY_THRESHOLD_PX = 10

FILE_FOLDER = "/Volumes/OWC52/segment_images_OWC4"
# FILE_FOLDER = "/Volumes/LaCie/segment_images_COCO" # must be a folder holding the site folder(s)
# FILE_FOLDER ="/Volumes/OWC54/segment_images_40xDetections"
# FILE_FOLDER = "/Volumes/RAID18" # must be a folder holding the site folder(s)
# MAKE_VIDEO_CSVS_PATH = "/Users/michael.mandiberg/Documents/projects-active/facemap_production/make_video_CSVs/book_csvs"
MAKE_VIDEO_CSVS_PATH = None  # to process all images in folder
OUTPUT_FOLDER = os.path.join(FILE_FOLDER, "test_output")
BATCH_SIZE = 100
YOLO_BATCH_SIZE = 8  # number of images per YOLO batch inference call (M3 Ultra: try 32-64)
IMAGE_LOAD_WORKERS = 8  # concurrent cv2.imread workers before each YOLO batch
CONF_THRESHOLD = 0.3
IS_DRAW_BOX = True
IS_SAVE_UNDETECTED = False
MOVE_OR_COPY = "copy"  # "move" or "copy"
CLUSTER_TYPE = "HSV" # only works with cluster save, not with assignment
META = False # to return the meta clusters (out of 23, not 96)
cl = ToolsClustering(CLUSTER_TYPE, VERBOSE=VERBOSE)
table_cluster_type = cl.set_table_cluster_type(META)

# custom_ids_to_global_dict = {
#   0: 89,
#   1: 90,
#   2: 92,
#   3: 84,
# }

# full class
# custom_ids_to_global_dict = {
#   0: 100,
#   1: 88,
#   2: 97,
#   3: 83,
#   4: 81,
#   5: 82,
#   6: 98,
#   7: 94,
#   8: 95,
#   9: 86,
#   10: 80,
#   11: 102,
#   12: 96,
#   13: 101,
#   14: 99,
#   15: 103

# }

# flowers 11 class
# custom_ids_to_global_dict = {
#   0: 100,
#   1: 107,
#   2: 97,
#   3: 104,
#   4: 98,
#   5: 106,
#   6: 102,
#   7: 101,
#   8: 99,
#   9: 105,
#   10: 103
# }

# # masks class
# custom_ids_to_global_dict = {
#   0: 117,
#   1: 113,
#   2: 116,
#   3: 114,
#   4: 112,
#   5: 118,
#   6: 119,
#   7: 110,
#   8: 115,
#   9: 111,
# }

# flags 230 class
# custom_ids_to_global_dict = {i:i for i in range(228)} # for testing, map custom ids to same global ids

# complete 36 class model
custom_ids_to_global_dict = {

  0: 109,
  1: 89,
  2: 117,
  3: 113,
  4: 108,
  5: 100,
  6: 116,
  7: 114,
  8: 88,
  9: 112,
  10: 118,
  11: 90,
  12: 92,
  13: 107,
  14: 84,
  15: 97,
  16: 83,
  17: 81,
  18: 104,
  19: 82,
  20: 98,
  21: 94,
  22: 95,
  23: 86,
  24: 119,
  25: 106,
  26: 80,
  27: 102,
  28: 110,
  29: 96,
  30: 101,
  31: 99,
  32: 105,
  33: 103,
  34: 115,
  35: 111,

}

# custom_ids_to_global_dict = {
#     0: 92,
#     1: 84,
# }


# custom_ids_to_global_dict = {
#     0: 85,
# }

# custom_ids_to_global_dict = {
#   0: 100,
#   1: 97,
#   2: 83,
#   3: 81,
#   4: 86,
#   5: 103,
# }
Clusters, ImagesClusters, MetaClusters, ClustersMetaClusters = cl.construct_table_classes(table_cluster_type)
this_cluster, this_crosswalk = cl.set_cluster_metacluster(Clusters, ImagesClusters, MetaClusters, ClustersMetaClusters)
meta_cluster_dict = cl.get_meta_cluster_dict(session, ClustersMetaClusters)
print("this_cluster: ", this_cluster)

slogan_dict = ocr.get_all_slogans(session, Slogans)
print("Loaded slogans:", slogan_dict)
refined_dict = ocr.get_all_refined(session, RefinedText)
print("Loaded refined:", refined_dict)

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
                last_position = img.split("-")[-1].replace(".jpg", "")
                first_position = img.split("-")[0].replace(".jpg", "")
                if last_position.isdigit():
                    this_site_image_id = last_position
                elif first_position.isdigit():
                    this_site_image_id = first_position
                else:
                    print({" ❌ Warning: could not parse site_image_id from filename:", img})
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

def save_no_dectections(session,image_id, NoDetectionTable=NoDetections, do_commit=True):
    # Save to no-detections table only if missing (idempotent for reruns).
    existing_row = session.query(NoDetectionTable).filter(NoDetectionTable.image_id == image_id).first()
    if existing_row is not None:
        print(f"Image {image_id} - Existing no-detections row already present in {NoDetectionTable.__tablename__}; leaving untouched.")
        return

    new_no_detection = NoDetectionTable(
        image_id=image_id
    )
    session.add(new_no_detection)
    if do_commit and not TESTING_NO_DB_WRITE:
        session.commit()
    print(f"Image {image_id} - No detections found, saved to {NoDetectionTable} table.")


def delete_no_detections(session, image_id, NoDetectionTable=NoDetections, do_commit=True):
    deleted_count = session.query(NoDetectionTable).filter(NoDetectionTable.image_id == image_id).delete(synchronize_session=False)
    if deleted_count > 0:
        if do_commit and not TESTING_NO_DB_WRITE:
            session.commit()
        print(f"Image {image_id} - Removed stale no-detections row from {NoDetectionTable.__tablename__} after successful detections.")

def backfill_image_dimensions_if_missing(result, image):
    image_id = result.image_id
    current_h = getattr(result, "h", None)
    current_w = getattr(result, "w", None)
    if current_h is None or current_w is None:
        image_h, image_w = image.shape[:2]
        update_values = {}
        if current_h is None:
            update_values["h"] = int(image_h)
        if current_w is None:
            update_values["w"] = int(image_w)
        if update_values:
            print(f"Image {image_id} - Updating missing dimensions in Images table: {update_values}")
            session.query(Images).filter(Images.image_id == image_id).update(update_values, synchronize_session=False)
            if not TESTING_NO_DB_WRITE:
                session.commit()
            else:
                print("TESTING_NO_DB_WRITE is True, skipping Images table update commit.")

def load_image_for_inference(load_item):
    result, image_path, state = load_item
    image = cv2.imread(image_path)
    return result, image_path, state, image

def assign_hsv_detect_results(detect_results, image):
    for result_dict in detect_results:
        bbox = io.unstring_json(result_dict['bbox'])
        meta_cluster_id, cluster_id, cluster_dist = bbox_to_cluster_id(image, bbox)
        result_dict['meta_cluster_id'] = meta_cluster_id
        result_dict['cluster_id'] = cluster_id
        # result_dict['cluster_dist'] = cluster_dist
    return detect_results

def do_yolo_detections(result, image, image_path, existing_detections, custom=False, precomputed_raw_detections=None, do_commit=True):
    image_id = result.image_id
    imagename = result.imagename

    if custom: 
        ThisNoDetectionTable=NoDetectionsCustom
        this_yolo_model = yolo_custom_model
    else:
        ThisNoDetectionTable=NoDetections
        this_yolo_model = yolo_model
    # YOLO object detection
    if precomputed_raw_detections is not None:
        unrefined_detect_results = precomputed_raw_detections
    else:
        unrefined_detect_results = yolo.detect_objects_return_bbox(this_yolo_model, image, device, conf_thresh=CONF_THRESHOLD)
    if VERBOSE: print(f"Image {image_id} - Unrefined YOLO detections: {unrefined_detect_results}")
    if custom:
        unrefined_detect_results = yolo.map_custom_ids_to_global(unrefined_detect_results, custom_ids_to_global_dict)
    detect_results = yolo.merge_yolo_detections(unrefined_detect_results, iou_threshold=IOU_THRESHOLD, adjacency_threshold_px=ADJACENCY_THRESHOLD_PX)
    detect_results = assign_hsv_detect_results(detect_results, image)
    if VERBOSE: print(f"Image {image_id} - YOLO detections: {detect_results}")
    class_ids = [det.get('class_id') for det in detect_results if det.get('class_id') is not None]
    if bool(class_ids):
        print(f" ☑️ Image {image_id} class_ids: {class_ids}")
    # save_debug_image_yolo_bbox(image_id, imagename, image, detect_results)
    if DEBUGGING:
        yolo.save_debug_image_yolo_bbox(image_id, imagename, image, detect_results, image_path, 
                                        OUTPUT_FOLDER, io, draw_box=IS_DRAW_BOX, 
                                        save_undetected=IS_SAVE_UNDETECTED, move_or_copy=MOVE_OR_COPY,
                                        combined_class_pairs=CLASSES_TO_COMBINE)
    if SAVE_NEW_LABELS:
        yolo.save_new_yolo_labels(image_id, image, image_path, detect_results, OUTPUT_FOLDER, io)

    if TESTING_NO_DB_WRITE:
        print("TESTING_NO_DB_WRITE is True, skipping DB write.")
        return detect_results
    if len(detect_results) == 0:
        # Persist no-detections for both COCO and custom so reruns can skip completed images.
        save_no_dectections(session,image_id, NoDetectionTable=ThisNoDetectionTable, do_commit=do_commit)
        return detect_results
    else:
        delete_no_detections(session, image_id, NoDetectionTable=ThisNoDetectionTable, do_commit=do_commit)
        yolo.save_obj_bbox(session, image_id, detect_results, Detections, do_commit=do_commit)
    return detect_results

def check_for_existing_detections(image_id, existing_detections, custom=False):
    if custom:
        all_existing_detection_ids = [det.detection_id for det in existing_detections if det.class_id >= 80]
        existing_detection_ids = [det_id for det_id in all_existing_detection_ids if det_id > DET_ID_THRESHOLD_CUSTOM]
        existing_no_detections = get_existing_no_detections(image_id, NoDetectionTable=NoDetectionsCustom)
    else:
        all_existing_detection_ids = [det.detection_id for det in existing_detections if det.class_id < 80]
        # only consider detection_ids > DET_ID_THRESHOLD_COCO which are the new ones, not the original ones
        existing_detection_ids = [det_id for det_id in all_existing_detection_ids if det_id > DET_ID_THRESHOLD_COCO]
        existing_no_detections = get_existing_no_detections(image_id, NoDetectionTable=NoDetections)
    return existing_detection_ids, existing_no_detections


def _get_skip_state(result):
    """Check DB skip conditions for one image without loading it. Returns state dict."""
    image_id = result.image_id
    existing_detections = get_existing_detections(image_id)
    existing_detection_ids, existing_no_detections = check_for_existing_detections(image_id, existing_detections, custom=False)
    existing_custom_detection_ids, existing_no_detections_custom = check_for_existing_detections(image_id, existing_detections, custom=True)

    skip_coco = False if OVERWRITE_EXISTING_DETECTIONS_COCO else bool(existing_no_detections)
    skip_custom = False if OVERWRITE_EXISTING_DETECTIONS_CUSTOM else (bool(existing_no_detections_custom) and not IGNORE_EXISTING_NO_DETECTIONS)
    existing_coco = False if OVERWRITE_EXISTING_DETECTIONS_COCO else (bool(existing_detection_ids) or skip_coco)
    existing_custom = False if OVERWRITE_EXISTING_DETECTIONS_CUSTOM else (bool(existing_custom_detection_ids) or skip_custom)
    skip_all = (
        (existing_coco and existing_custom) or
        (existing_coco and not DO_CUSTOM) or
        (existing_custom and not DO_COCO)
    )
    return {
        'skip_all': skip_all,
        'skip_coco': skip_coco,
        'skip_custom': skip_custom,
        'existing_detections': existing_detections,
        'existing_no_detections_custom': existing_no_detections_custom,
    }


def do_detections(result, folder_index, _state=None, _preloaded_image=None, _precomputed_coco=None, _precomputed_custom=None, _do_commit=True):
    coco_detections = custom_detections = None
    # start a timer for this function to track how long it takes
    start_time = time.time()
    image_id = result.image_id
    imagename = result.imagename
    if _state is not None:
        existing_detections = _state['existing_detections']
        skip_coco_due_to_no_det = _state['skip_coco']
        skip_custom_due_to_no_det = _state['skip_custom']
        if _state['existing_no_detections_custom'] and IGNORE_EXISTING_NO_DETECTIONS:
            print(f"Image {image_id} - Ignoring existing no-detections row in NoDetectionsCustom due to IGNORE_EXISTING_NO_DETECTIONS=True.")
    else:
        existing_coco = existing_custom = False
        existing_detections = get_existing_detections(result.image_id)
        if VERBOSE: print(f"Image {image_id} - Retrieved {len(existing_detections)} existing detections from database.")
        existing_detection_ids, existing_no_detections = check_for_existing_detections(image_id, existing_detections, custom=False)
        existing_custom_detection_ids, existing_no_detections_custom = check_for_existing_detections(image_id, existing_detections, custom=True)

        skip_coco_due_to_no_det = False if OVERWRITE_EXISTING_DETECTIONS_COCO else bool(existing_no_detections)
        skip_custom_due_to_no_det = False if OVERWRITE_EXISTING_DETECTIONS_CUSTOM else (bool(existing_no_detections_custom) and not IGNORE_EXISTING_NO_DETECTIONS)

        # only skip custom no_detections. we never redo COCO no_detections for COCO
        if bool(existing_no_detections_custom) and IGNORE_EXISTING_NO_DETECTIONS:
            print(f"Image {image_id} - Ignoring existing no-detections row in NoDetectionsCustom due to IGNORE_EXISTING_NO_DETECTIONS=True.")

        if not OVERWRITE_EXISTING_DETECTIONS_COCO and (existing_detection_ids or skip_coco_due_to_no_det):
            existing_coco = True
        if not OVERWRITE_EXISTING_DETECTIONS_CUSTOM and (existing_custom_detection_ids or skip_custom_due_to_no_det):
            existing_custom = True

        if (existing_coco and existing_custom) or (existing_coco and not DO_CUSTOM) or (existing_custom and not DO_COCO):
            print(f"Skipping image_id {image_id} due to existing detections record for this configuration, time taken: {time.time() - start_time:.10f} seconds.")
            print
            return
    # elif len(existing_detection_ids) > 0:
    #     print(f"Skipping image_id {result.image_id} due to existing detections record.")
    #     return
        # TK uncomment this when no longer testing and handle custom = False
        # for det_id in existing_detection_ids:
            # if det_id > 12455146:
            #     print(f"Skipping image_id {result.image_id} due to first round detection_id {det_id} > 12455146")
            #     return

    try:
        face_bbox = io.unstring_json(result.bbox)
    except Exception as e:
        print(f"Error parsing label_data JSON for image_id {image_id}: {e}")
        print("Skipping this face_bbox.", result.bbox)
        return
    image_path = os.path.join(FILE_FOLDER, os.path.basename(io.folder_list[folder_index]), imagename)
    if _preloaded_image is not None:
        image = _preloaded_image
        # backfill already done in batch pre-check phase
    else:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image at {image_path}. Skipping.")
            return
        backfill_image_dimensions_if_missing(result, image)

    print(f"Processing image_id: {image_id}, imagename: {imagename}")
    if DO_COCO and not skip_coco_due_to_no_det:
        coco_detections = do_yolo_detections(result, image, image_path, existing_detections, custom=False,
                                                                                         precomputed_raw_detections=_precomputed_coco,
                                                                                         do_commit=_do_commit)

    if DO_CUSTOM and not skip_custom_due_to_no_det:
        custom_detections = do_yolo_detections(result, image, image_path, existing_detections, custom=True,
                                                                                             precomputed_raw_detections=_precomputed_custom,
                                                                                             do_commit=_do_commit)

    if DO_OCR and DO_CUSTOM:
        # make a list of sign_detections
        if custom_detections is not None:
            sign_detections = [det for det in custom_detections if det['class_id'] == 80]
        elif existing_detections:
            sign_detections = [det for det in existing_detections if det['class_id'] == 80]
        else:
            print("no sign_detections, returning")
            return
        
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
            # save found_texts

            # assign slogan_id
            slogan_id, slogan_dict, refined_dict = ocr.assign_slogan_id(session, openai_client, Slogans, slogan_dict, refined_dict, image_id, found_texts)
            # Here, save image_id, slogan_id info to Placards table
            if slogan_id is not None:
                print(f"Saving image {image_id} with slogan_id {slogan_id} to ImagesSlogans table.")
                # this needs to wait for ImagesSlogans table to be created
                # ocr.save_images_slogans(session, ImagesSlogans, image_id, slogan_id)
            else:
                print(f"Error: slogan_id is None for image {image_id}, skipping save to Placards table.")

def save_debug_image_yolo_bbox(image_id, imagename, image, detect_results, image_path, draw_box=True, save_undetected=True):
    if not detect_results and save_undetected:
        output_image_path = os.path.join(OUTPUT_FOLDER, "no_detections",imagename)
        # only undetected get MOVE_OR_COPY option
        save_debug_image(output_image_path, image, imagename, image_path=image_path, move_or_copy=MOVE_OR_COPY)
    elif MOVE_OR_COPY != "move":
        # only save detected if not moving undetected
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
    print("lenth df_csvs if MAKE_VIDEO_CSVS_PATH:", len(df_csvs))     
    folder_count = 0
    for folder_index in folder_indexes:
        # goes through all. if no folder found, skips and moves on, until it finds the correct site folder
        df_csvs_folder = df_csvs[df_csvs['site_name_id'] == folder_index]
        if MAKE_VIDEO_CSVS_PATH is not None and len(df_csvs_folder) == 0:
            print("Skipping folder_index:", folder_index, "no entries in MAKE_VIDEO_CSVS_PATH")
            continue
        else:
            print("meets or ignores MAKE_VIDEO_CSVS_PATH, folder_index:", folder_index, "len df_csvs_folder:", len(df_csvs_folder))
        print(io.folder_list)
        print("basename:", os.path.basename(io.folder_list[folder_index]))
        print("FILE_FOLDER:", FILE_FOLDER)
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

TIMER_INTERVAL = 500  # print throughput every N images processed

def detect_from_folder(folder_index, csv_foldercount_path, folder_path, folder, df_csvs_folder):
    img_list = io.get_img_list(folder, force_ls=True)
    print("len(img_list)", len(img_list))
    if len(df_csvs_folder) > 0:
        img_list = [img for img in img_list if img.split(".")[0] in df_csvs_folder['site_image_id'].values]
        print("after filtering with MAKE_VIDEO_CSVS_PATH, len(img_list)", len(img_list))

    all_results = []
    processed_count = 0
    timer_start = time.time()
    last_timer_count = 0
    interval_phase_times = {
        'skip_state': 0.0,
        'image_load': 0.0,
        'backfill': 0.0,
        'coco_infer': 0.0,
        'custom_infer': 0.0,
        'postprocess_save': 0.0,
    }

    for i in range(0, len(img_list), BATCH_SIZE):
        batch_img_list = img_list[i : i + BATCH_SIZE]
        batch_site_image_ids = format_site_name_ids(folder_index, batch_img_list)

        print(f"total img_list: {len(img_list)} no. processed: {i} no. left: {len(img_list)-i}")
        print(f"folder_index: {folder_index}, {batch_site_image_ids} images.")
        if len(img_list) - i < BATCH_SIZE:
            print("last_round for img_list")

        batch_results = []
        for _ in range(io.max_retries):
            try:
                if VERBOSE:
                    print(f"Processing batch {i // BATCH_SIZE + 1}...")
                batch_query = session.query(Images.image_id, Images.site_image_id, Images.imagename, Images.h, Images.w, Encodings.bbox) \
                    .join(Encodings, Encodings.image_id == Images.image_id) \
                    .filter(Images.site_image_id.in_(batch_site_image_ids), Images.site_name_id == folder_index)
                batch_results = batch_query.all()
                all_results.extend(batch_results)
                break
            except OperationalError as e:
                print("error getting batch results")
                print(e)
                time.sleep(io.retry_delay)

        if VERBOSE:
            print(f"no. all_results: {len(all_results)}")

        results_dict = {result.site_image_id: result for result in batch_results}
        images_left_to_process = len(batch_site_image_ids)

        for j in range(0, len(batch_site_image_ids), YOLO_BATCH_SIZE):
            yolo_sub_ids = batch_site_image_ids[j:j + YOLO_BATCH_SIZE]

            load_candidates = []
            for site_image_id in yolo_sub_ids:
                images_left_to_process -= 1
                if site_image_id not in results_dict:
                    if VERBOSE:
                        print(f"site_name_id: {folder_index} site_image_id: {site_image_id} not found in DB, skipping.")
                    continue

                result = results_dict[site_image_id]
                image_id = result.image_id
                skip_state_start = time.time()
                state = _get_skip_state(result)
                interval_phase_times['skip_state'] += time.time() - skip_state_start

                if state['skip_all']:
                    print(f"Skipping image_id {image_id} due to existing detections.")
                    continue

                try:
                    io.unstring_json(result.bbox)
                except Exception as e:
                    print(f"Error parsing bbox for image_id {image_id}: {e}. Skipping.")
                    continue

                image_path = os.path.join(FILE_FOLDER, os.path.basename(io.folder_list[folder_index]), result.imagename)
                load_candidates.append((result, image_path, state))

            if not load_candidates:
                continue

            image_load_start = time.time()
            max_workers = min(IMAGE_LOAD_WORKERS, len(load_candidates))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                loaded_items = list(executor.map(load_image_for_inference, load_candidates))
            interval_phase_times['image_load'] += time.time() - image_load_start

            to_process = []
            for result, image_path, state, image in loaded_items:
                image_id = result.image_id
                if image is None:
                    print(f"Error: Unable to read image at {image_path}. Skipping.")
                    continue

                backfill_start = time.time()
                backfill_image_dimensions_if_missing(result, image)
                interval_phase_times['backfill'] += time.time() - backfill_start
                to_process.append((result, image, image_path, state))

                if VERBOSE:
                    print(f"Found image_id: {image_id} for site_image_id: {result.site_image_id}, imagename: {result.imagename}")

            if not to_process:
                continue

            coco_precomputed = [None] * len(to_process)
            custom_precomputed = [None] * len(to_process)

            coco_indices = [idx for idx, (_, _, _, state) in enumerate(to_process) if DO_COCO and not state['skip_coco']]
            if coco_indices:
                coco_imgs = [to_process[idx][1] for idx in coco_indices]
                coco_infer_start = time.time()
                coco_batch = yolo.detect_objects_return_bbox_batch(yolo_model, coco_imgs, device, CONF_THRESHOLD)
                interval_phase_times['coco_infer'] += time.time() - coco_infer_start
                for idx, det in zip(coco_indices, coco_batch):
                    coco_precomputed[idx] = det

            custom_indices = [idx for idx, (_, _, _, state) in enumerate(to_process) if DO_CUSTOM and not state['skip_custom']]
            if custom_indices:
                custom_imgs = [to_process[idx][1] for idx in custom_indices]
                custom_infer_start = time.time()
                custom_batch = yolo.detect_objects_return_bbox_batch(yolo_custom_model, custom_imgs, device, CONF_THRESHOLD)
                interval_phase_times['custom_infer'] += time.time() - custom_infer_start
                for idx, det in zip(custom_indices, custom_batch):
                    custom_precomputed[idx] = det

            for idx, (result, image, image_path, state) in enumerate(to_process):
                postprocess_start = time.time()
                do_detections(
                    result,
                    folder_index,
                    _state=state,
                    _preloaded_image=image,
                    _precomputed_coco=coco_precomputed[idx],
                    _precomputed_custom=custom_precomputed[idx],
                    _do_commit=False
                )
                interval_phase_times['postprocess_save'] += time.time() - postprocess_start
                processed_count += 1

            if to_process and not TESTING_NO_DB_WRITE:
                commit_start = time.time()
                try:
                    session.commit()
                except Exception:
                    session.rollback()
                    raise
                interval_phase_times['postprocess_save'] += time.time() - commit_start

            while processed_count - last_timer_count >= TIMER_INTERVAL:
                elapsed = time.time() - timer_start
                interval_count = processed_count - last_timer_count
                rate = interval_count / elapsed if elapsed > 0 else 0
                print(
                    f"⏱  {processed_count} images processed | last {interval_count} in {elapsed:.1f}s | {rate:.1f} img/s"
                    f" | skip/db {interval_phase_times['skip_state']:.1f}s"
                    f" | load {interval_phase_times['image_load']:.1f}s"
                    f" | backfill {interval_phase_times['backfill']:.1f}s"
                    f" | coco {interval_phase_times['coco_infer']:.1f}s"
                    f" | custom {interval_phase_times['custom_infer']:.1f}s"
                    f" | save {interval_phase_times['postprocess_save']:.1f}s"
                )
                last_timer_count = processed_count
                timer_start = time.time()
                interval_phase_times = {
                    'skip_state': 0.0,
                    'image_load': 0.0,
                    'backfill': 0.0,
                    'coco_infer': 0.0,
                    'custom_infer': 0.0,
                    'postprocess_save': 0.0,
                }
    io.write_csv(csv_foldercount_path, [folder_path])

if __name__ == "__main__":
    main()
