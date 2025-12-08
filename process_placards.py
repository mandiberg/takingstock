import cv2
import numpy as np
import os
from paddleocr import PaddleOCR
import re
from openai import OpenAI
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
from my_declarative_base import Encodings, Images, Slogans, ImagesSlogans
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

yolo_model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model

ocr = OCRTools(DEBUGGING=True)
yolo = YOLOTools(DEBUGGING=True)


blank = False
DEBUGGING = True

FILE_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/segment_images_book_clock_bowl"
OUTPUT_FOLDER = os.path.join(FILE_FOLDER, "test_output")
BATCH_SIZE = 100
MASK_THRESHOLD = .15  # HSV distance threshold for mask detection
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

def do_detections(result, folder_index):
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

    def merge_yolo_detections(detect_results, iou_threshold=0.5, adjacency_threshold_px=20):
        '''
        Merge multiple detections of the same class if they overlap (IoU > iou_threshold)
        or if they are adjacent/touching (gap < adjacency_threshold_px).
        Use case: stack of books detected as 6 separate objects; merge into 1.
        
        Args:
            detect_results: list of dicts with class_id, obj_no, bbox (JSON str), conf
            iou_threshold: IoU threshold for overlap merging (default 0.5)
            adjacency_threshold_px: max gap (pixels) to consider bboxes adjacent (default 20)
        Returns:
            refined_results: merged list in same format
        '''
        def bboxes_are_adjacent(bbox1, bbox2, adjacency_threshold_px):
            """Check if two bboxes are adjacent (touching or close)."""
            left1, right1, top1, bottom1 = bbox1['left'], bbox1['right'], bbox1['top'], bbox1['bottom']
            left2, right2, top2, bottom2 = bbox2['left'], bbox2['right'], bbox2['top'], bbox2['bottom']
            
            # Horizontal adjacency: gap between right1 and left2, or right2 and left1
            h_gap = min(abs(right1 - left2), abs(right2 - left1))
            # Vertical adjacency: gap between bottom1 and top2, or bottom2 and top1
            v_gap = min(abs(bottom1 - top2), abs(bottom2 - top1))
            
            # Check if either horizontal or vertical gap is within threshold and they overlap in the other dimension
            h_overlap = not (right1 < left2 or right2 < left1)  # horizontal overlap exists
            v_overlap = not (bottom1 < top2 or bottom2 < top1)  # vertical overlap exists
            
            # Adjacent if: (h_overlap and small v_gap) or (v_overlap and small h_gap)
            if h_overlap and v_gap <= adjacency_threshold_px:
                return True
            if v_overlap and h_gap <= adjacency_threshold_px:
                return True
            return False
        
        refined_results = []
        used_indices = set()
        
        for i in range(len(detect_results)):
            if i in used_indices:
                continue
            
            current = detect_results[i]
            current_bbox = io.unstring_json(current['bbox'])
            current_left = current_bbox['left']
            current_right = current_bbox['right']
            current_top = current_bbox['top']
            current_bottom = current_bbox['bottom']
            
            for j in range(i+1, len(detect_results)):
                if j in used_indices:
                    continue
                
                compare = detect_results[j]
                if compare['class_id'] != current['class_id']:
                    continue
                
                compare_bbox = io.unstring_json(compare['bbox'])
                
                # Check for adjacency first (simpler, faster)
                if bboxes_are_adjacent(current_bbox, compare_bbox, adjacency_threshold_px):
                    # Merge
                    current_left = min(current_left, compare_bbox['left'])
                    current_right = max(current_right, compare_bbox['right'])
                    current_top = min(current_top, compare_bbox['top'])
                    current_bottom = max(current_bottom, compare_bbox['bottom'])
                    used_indices.add(j)
                    # Update current_bbox for next comparison
                    current_bbox = {'left': current_left, 'right': current_right, 'top': current_top, 'bottom': current_bottom}
                else:
                    # Check for overlap via IoU
                    inter_left = max(current_left, compare_bbox['left'])
                    inter_right = min(current_right, compare_bbox['right'])
                    inter_top = max(current_top, compare_bbox['top'])
                    inter_bottom = min(current_bottom, compare_bbox['bottom'])
                    
                    if inter_left < inter_right and inter_top < inter_bottom:
                        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
                        current_area = (current_right - current_left) * (current_bottom - current_top)
                        compare_area = (compare_bbox['right'] - compare_bbox['left']) * (compare_bbox['bottom'] - compare_bbox['top'])
                        union_area = current_area + compare_area - inter_area
                        iou = inter_area / union_area
                        
                        if iou > iou_threshold:
                            # Merge
                            current_left = min(current_left, compare_bbox['left'])
                            current_right = max(current_right, compare_bbox['right'])
                            current_top = min(current_top, compare_bbox['top'])
                            current_bottom = max(current_bottom, compare_bbox['bottom'])
                            used_indices.add(j)
                            # Update current_bbox for next comparison
                            current_bbox = {'left': current_left, 'right': current_right, 'top': current_top, 'bottom': current_bottom}
            
            # After checking all others, add the merged bbox to refined results
            merged_bbox = {
                'class_id': current['class_id'],
                'obj_no': current['obj_no'],
                'bbox': json.dumps({'left': current_left, 'top': current_top, 'right': current_right, 'bottom': current_bottom}),
                'conf': current['conf']
            }
            refined_results.append(merged_bbox)
        
        return refined_results
    

    image_id = result.image_id
    imagename = result.imagename
    face_bbox = io.unstring_json(result.bbox)
    image_path = os.path.join(FILE_FOLDER, os.path.basename(io.folder_list[folder_index]), imagename)
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}. Skipping.")
        return

    print(f"Processing image_id: {image_id}, imagename: {imagename}")

    unrefined_detect_results = yolo.detect_objects_return_bbox(yolo_model,image)
    detect_results = merge_yolo_detections(unrefined_detect_results, iou_threshold=0.3, adjacency_threshold_px=50)
    # print out a list of classes detected in the yolo detection results
    # print(f"Detected objects in image {imagename}: {list_of_detected_objects}")
    # {'class_id': 0, 'obj_no': 2, 'bbox': '{"left": 1121, "top": 393, "right": 1244, "bottom": 642}', 'conf': 0.47} 
    if not detect_results:
        output_image_path = os.path.join(OUTPUT_FOLDER, "no_detections",debug_file_name)
        save_debug_image(output_image_path, image, imagename)
    else:
        for result_dict in detect_results:
            if result_dict['class_id'] == 0: continue # skip person class
            print(f"Detected class: {result_dict['class_id']} with bbox: {result_dict['bbox']} and confidence: {result_dict['conf']}")
            # for debugging mask, saving to folders
            debug_file_name = f"{result_dict['conf']:.2f}_{image_id}_YOLO_debug.jpg"
            output_image_path = os.path.join(OUTPUT_FOLDER, str(result_dict['class_id']),debug_file_name)
            image_with_bbox = draw_bbox_on_image(image.copy(), io.unstring_json(result_dict['bbox']))
            save_debug_image(output_image_path, image_with_bbox, imagename)

    return

    # detect valentine heart


    # detect facemask via hsv distance and clustering
    top_hsl, bot_hsl, hsl_distance = yolo.compute_mask_hsv(image, face_bbox)
    meta_cluster_id, cluster_id, cluster_dist = mask_to_cluster_id(image, face_bbox)
   
    # for debugging mask, saving to folders
    # debug_file_name = f"{hsl_distance:.2f}_{image_id}_mask_debug.jpg"
    # if hsl_distance > MASK_THRESHOLD: output_image_path = os.path.join(OUTPUT_FOLDER, str(meta_cluster_id),debug_file_name)
    # else: output_image_path = os.path.join(OUTPUT_FOLDER, "no_mask",debug_file_name)
    # # save image to OUTPUT_FOLDER for review
    # save_debug_image(output_image_path, image, imagename)

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
                            print(f"site_image_id: {site_image_id} not found in DB, skipping.")
                            continue
                # save  success to csv_foldercount_path
                io.write_csv(csv_foldercount_path, [folder_path])

    session.close()
    engine.dispose()

if __name__ == "__main__":
    main()
