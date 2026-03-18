import cv2
import os
import torch
from ultralytics import YOLO
from tools_yolo import YOLOTools
from mp_db_io import DataIO

io = DataIO()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
yolo_custom_model = YOLO("models/takingstock_gun_v8_yolov8m/weights/best.pt").to(device)
yolo = YOLOTools(DEBUGGING=True)


# Configuration
DEBUGGING = True
SAVE_NEW_LABELS = True
FILE_FOLDER = "/Volumes/OWC5/segment_images_91_gun/guns_unprocessed_mar5"
# FILE_FOLDER = "/Volumes/LaCie/segment_images_101_flowers_all/flower_image_repository"
OUTPUT_FOLDER = os.path.join(FILE_FOLDER, "test_output")
CONF_THRESHOLD = 0.3
CREATE_YOLO_CLASS_ID = [90]
IS_DRAW_BOX = True
IS_SAVE_UNDETECTED = True
MOVE_OR_COPY = "copy"
CLASSES_TO_COMBINE = [89, 90]  # merge these classes onto one label, and draw them to one debug image

# custom_ids_to_global_dict = {
#   0: 89,
#   1: 90,
#   2: 92,
#   3: 84,
# }

# guns 2 class
custom_ids_to_global_dict = {
    0: 109,
    1: 108,
}


# # flowers 11 class
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


def do_yolo_detections(result, image, image_path, existing_detections=None, custom=False):
    image_id = result.image_id
    imagename = result.imagename

    if custom: 
        this_yolo_model = yolo_custom_model
    
    print(f"Image {image_id} - Performing YOLO detections. custom={custom}")
    unrefined_detect_results = yolo.detect_objects_return_bbox(this_yolo_model, image, device, conf_thresh=CONF_THRESHOLD)
    print(f"Image {image_id} - Unrefined YOLO detections: {unrefined_detect_results}")
    
    if custom:
        unrefined_detect_results = yolo.map_custom_ids_to_global(unrefined_detect_results, custom_ids_to_global_dict)
    
    detect_results = yolo.merge_yolo_detections(unrefined_detect_results, iou_threshold=0.3, adjacency_threshold_px=50)
    print(f"Image {image_id} - YOLO detections: {detect_results}")
    
    if DEBUGGING:
        yolo.save_debug_image_yolo_bbox(image_id, imagename, image, detect_results, image_path, 
                                        OUTPUT_FOLDER, io, draw_box=IS_DRAW_BOX, 
                                        save_undetected=IS_SAVE_UNDETECTED, move_or_copy=MOVE_OR_COPY,
                                        combined_class_pairs=CLASSES_TO_COMBINE)
    
    if SAVE_NEW_LABELS:
        yolo.save_new_yolo_labels(image_id, image, image_path, detect_results, OUTPUT_FOLDER, io)
    
    return detect_results

def main():
    # FILE_FOLDER contains images and labels subfolders. 
    this_folder = os.path.join(FILE_FOLDER, "images")
    img_list = io.get_img_list(this_folder, force_ls=True)
    print("len(img_list)", len(img_list))
    for img_name in img_list:
        print("processing img_name:", img_name)
        # load image for processing
        image_path = os.path.join(this_folder, img_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image at {image_path}. Skipping.")
            continue
        img_parts = img_name.split(".")
        if len(img_parts) == 2:
            image_id = img_parts[0]
        elif len(img_parts) == 3:
            image_id = img_parts[1].replace("_YOLO_debug", "")
        else:
            print(f"Error: Unexpected image name format: {img_name}. Skipping.")
            continue
        result = type('obj', (object,), {'image_id': image_id, 'imagename': img_name, 'bbox': '{}'})
        custom_detections = do_yolo_detections(result, image, image_path, existing_detections=None, custom=True)
        print("custom_detections:", custom_detections)

if __name__ == "__main__":
    main()
