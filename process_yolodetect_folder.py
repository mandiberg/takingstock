import cv2
import os
import torch
from ultralytics import YOLO
from tools_yolo import YOLOTools
from mp_db_io import DataIO

io = DataIO()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
yolo_custom_model = YOLO("models/takingstock_steth_head_heart_yolov8m/weights/best.pt").to(device)
yolo = YOLOTools(DEBUGGING=True)


# Configuration
DEBUGGING = True
SAVE_NEW_LABELS = True
FILE_FOLDER = "/Users/michael.mandiberg/Documents/YOLO_Training_Data/sorted_images/90_stethoscope"
OUTPUT_FOLDER = os.path.join(FILE_FOLDER, "test_output")
CONF_THRESHOLD = 0.4
CREATE_YOLO_CLASS_ID = 90
IS_DRAW_BOX = True
IS_SAVE_UNDETECTED = True
MOVE_OR_COPY = "copy"

custom_ids_to_global_dict = {
    0: 90,
    1: 92,
    2: 84,
}


def save_debug_image(output_image_path, image, imagename, image_path=None, move_or_copy=False):
    # save image to OUTPUT_FOLDER for review
    if not os.path.exists(os.path.dirname(output_image_path)):
        os.makedirs(os.path.dirname(output_image_path))
    cv2.imwrite(output_image_path, image)
    if move_or_copy == "move" and image_path is not None:
        os.remove(image_path)
        print(f"Image {imagename} no detections, MOVED to {output_image_path}. ")
    elif move_or_copy == "copy" and image_path is not None:
        print(f"Image {imagename} no detections, saved to {output_image_path}. ")

def draw_bbox_on_image(image, bbox):
    left = bbox['left']
    right = bbox['right']
    top = bbox['top']
    bottom = bbox['bottom']
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    return image

def map_custom_ids_to_global(detect_results):
    for result_dict in detect_results:
        result_dict['class_id'] = custom_ids_to_global_dict.get(result_dict['class_id'], None)
        if result_dict['class_id'] is None:
            print(f" ⚠️ Warning: custom class_id {result_dict['class_id']} not found in mapping dictionary, setting to None.")
            result_dict = None
        else:
            print(f" ✅ Mapped custom class_id to global class_id: {result_dict['class_id']}")
    return detect_results

def save_new_yolo_labels(image_id, image, image_path, results):
    if len(results) == 0:
        return
    
    new_results = [res for res in results if res['class_id'] == CREATE_YOLO_CLASS_ID]
    if len(new_results) == 0:
        return
    
    this_bbox_dict = io.unstring_json(new_results[0]['bbox'])
    print(f"Image {image_id} - Saving YOLO label for bbox: {this_bbox_dict} from results: {new_results}")
    # ensure all values are int
    this_bbox_dict = {k: int(v) for k, v in this_bbox_dict.items()}
    print(f"Image {image_id} - Converted bbox to int: {this_bbox_dict}")
    # use this_bbox_dict to creat a YOLO-style bbox for saving as detection label for training
    # class_id, x_center, y_center, width, height (all normalized)
    x_center = (this_bbox_dict['left'] + this_bbox_dict['right']) / 2 / image.shape[1]
    y_center = (this_bbox_dict['top'] + this_bbox_dict['bottom']) / 2 / image.shape[0]
    width = (this_bbox_dict['right'] - this_bbox_dict['left']) / image.shape[1]
    height = (this_bbox_dict['bottom'] - this_bbox_dict['top']) / image.shape[0]
    yolo_label = f"{CREATE_YOLO_CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

    all_yolo_labels_folder = os.path.join(OUTPUT_FOLDER, "sort", "all_yolo_labels")
    files_to_move_folder = os.path.join(OUTPUT_FOLDER, "sort", "move_these")
    if not os.path.exists(all_yolo_labels_folder):
        os.makedirs(all_yolo_labels_folder)
    if not os.path.exists(files_to_move_folder):
        os.makedirs(files_to_move_folder)
    file_basename = f"{new_results[0]['conf']:.2f}_{image_id}_YOLO_debug"


    label_file_path = os.path.join(all_yolo_labels_folder, file_basename + ".txt")
    image_file_path = os.path.join(all_yolo_labels_folder, file_basename + ".jpg")
    print(f"Saving new YOLO label to {label_file_path} and image to {image_file_path}")
    with open(label_file_path, "w") as label_file:
        label_file.write(yolo_label)

    # save a copy of the original image to the yolo folder
    cv2.imwrite(image_file_path, image)

def do_yolo_detections(result, image, image_path, existing_detections=None, custom=False):
    image_id = result.image_id
    imagename = result.imagename

    if custom: 
        this_yolo_model = yolo_custom_model
    
    print(f"Image {image_id} - Performing YOLO detections. custom={custom}")
    unrefined_detect_results = yolo.detect_objects_return_bbox(this_yolo_model, image, device, conf_thresh=CONF_THRESHOLD)
    print(f"Image {image_id} - Unrefined YOLO detections: {unrefined_detect_results}")
    
    if custom:
        unrefined_detect_results = map_custom_ids_to_global(unrefined_detect_results)
    
    detect_results = yolo.merge_yolo_detections(unrefined_detect_results, iou_threshold=0.3, adjacency_threshold_px=50)
    print(f"Image {image_id} - YOLO detections: {detect_results}")
    
    if DEBUGGING:
        save_debug_image_yolo_bbox(image_id, imagename, image, detect_results, image_path, draw_box=IS_DRAW_BOX, save_undetected=IS_SAVE_UNDETECTED)
    
    if SAVE_NEW_LABELS:
        save_new_yolo_labels(image_id, image, image_path, detect_results)
    
    return detect_results

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
        image_id = img_name.split(".")[0]
        result = type('obj', (object,), {'image_id': image_id, 'imagename': img_name, 'bbox': '{}'})
        custom_detections = do_yolo_detections(result, image, image_path, existing_detections=None, custom=True)
        print("custom_detections:", custom_detections)

if __name__ == "__main__":
    main()
