import cv2
import os
import json
import math
import shutil
import torch
from ultralytics import YOLO
from tools_yolo import YOLOTools
from mp_db_io import DataIO

io = DataIO()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
yolo_custom_model = YOLO("models/takingstock_flatstuff_c5_v3_yolo26x/weights/best.pt").to(device)
yolo_model = YOLO('yolov8x.pt')  # Options: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26x-objv1-150.pt

print(yolo_custom_model.names)

yolo = YOLOTools(DEBUGGING=True)


# exit()

# Configuration
DEBUGGING = True
# FILE_FOLDER = "/Volumes/OWC5/segment_images_91_gun/guns_unprocessed_mar5"
FILE_FOLDER = "/Users/michaelmandiberg/Documents/YOLO_Training_Data/sorted_images_reprocess/flowers_only"
OUTPUT_FOLDER = os.path.join(FILE_FOLDER, "test_output")
INPUT_IMAGES_FOLDER = os.path.join(FILE_FOLDER, "images")
INPUT_LABELS_FOLDER = os.path.join(FILE_FOLDER, "labels")

# Primary output for review/import workflows.
OUTPUT_MERGED_FOLDER = os.path.join(OUTPUT_FOLDER, "merged_dataset")
OUTPUT_MERGED_IMAGES = os.path.join(OUTPUT_MERGED_FOLDER, "images")
OUTPUT_MERGED_LABELS = os.path.join(OUTPUT_MERGED_FOLDER, "labels")

# Review branch export control.
EXPORT_ONLY_WITH_NEW_DETECTIONS = True
OUTPUT_REVIEW_FOLDER = os.path.join(OUTPUT_FOLDER, "review_new_detections")
OUTPUT_REVIEW_IMAGES = os.path.join(OUTPUT_REVIEW_FOLDER, "images")
OUTPUT_REVIEW_LABELS = os.path.join(OUTPUT_REVIEW_FOLDER, "labels")

# Secondary debug output (optional).
ENABLE_DEBUG_EXPORT = True
OUTPUT_DEBUG_FOLDER = os.path.join(OUTPUT_FOLDER, "debug")

# Keep predictions only for these mapped global class_ids.
# If set to None, all mapped class_ids are allowed.
NEW_CLASSES_TO_ADD = None

# Deterministic suppression for predicted boxes against existing labels
# of the same class.
IOU_SAME_CLASS_THRESHOLD = 0.65
USE_CENTER_DISTANCE_CHECK = True
CENTER_DIST_THRESHOLD_NORM = 0.05

# Existing labels in these classes always win over overlapping new detections.
PROTECTED_EXISTING_CLASS_IDS = {80,81,82,83, 84, 95, 93, 127}
PROTECTED_CLASS_IOU_THRESHOLD = 0.10

CONF_THRESHOLD = 0.6
CREATE_YOLO_CLASS_ID = [90]
IS_DRAW_BOX = True
IS_SAVE_UNDETECTED = True
MOVE_OR_COPY = "copy"
CLASSES_TO_COMBINE = [89, 90]  # merge these classes onto one label, and draw them to one debug image

# custom_ids_to_global_dict = {
#   0: 92,
#   1: 84,
# }

#  placardplus class
# custom_ids_to_global_dict = {
#     0: 0,
#     1: 83,
#     2: 81,
#     3: 88,
#     4: 80,
# }

# # moneymix  class
# custom_ids_to_global_dict = {
#     0: 82,
#     1: 94,
#     2: 95,
#     3: 96,
# }

# # guns 2 class
# custom_ids_to_global_dict = {
#     0: 109,
#     1: 108,
# }


# # steth 2 class
# custom_ids_to_global_dict = {
#     0: 89,
#     1: 90,
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


# flatstuff
custom_ids_to_global_dict = {
  0: 124,
  1: 127,
  2: 93,
#   3: 133,
#   4: 137,
}


{0: '124_Tablet', 1: '127_calculator', 2: '93_clipboard', 3: '136_Laptop', 4: '137_Phone_handheld'}

# # flatstuff
# custom_ids_to_global_dict = {
#   0: 124,
#   1: 135,
#   2: 134,
#   3: 133,
#   4: 136,
#   5: 137,
# }

# {0: '124_Tablet', 1: '135_Computer_monitor', 2: '134_Book', 3: '133_Remote_control', 4: '136_Laptop', 5: '137_Phone_handheld'}


# COCO for 63 67
COCO_ids_to_global_dict = {
  62: 132,
  63: 133,
  65: 135,
  66: 136,
  67: 137,
  73: 123,
}



if NEW_CLASSES_TO_ADD is None:
    NEW_CLASSES_TO_ADD = set(custom_ids_to_global_dict.values()) | set(COCO_ids_to_global_dict.values())


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images_recursive(root_folder):
    image_paths = []
    for root, _, files in os.walk(root_folder):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, filename))
    image_paths.sort()
    return image_paths


def image_id_from_filename(filename):
    stem = os.path.splitext(filename)[0]
    return stem.replace("_YOLO_debug", "")


def yolo_to_xyxy(box):
    x = box["x_center"]
    y = box["y_center"]
    w = box["width"]
    h = box["height"]
    return {
        "left": x - (w / 2.0),
        "top": y - (h / 2.0),
        "right": x + (w / 2.0),
        "bottom": y + (h / 2.0),
    }


def xyxy_iou(a, b):
    inter_left = max(a["left"], b["left"])
    inter_top = max(a["top"], b["top"])
    inter_right = min(a["right"], b["right"])
    inter_bottom = min(a["bottom"], b["bottom"])

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    a_area = max(0.0, (a["right"] - a["left"])) * max(0.0, (a["bottom"] - a["top"]))
    b_area = max(0.0, (b["right"] - b["left"])) * max(0.0, (b["bottom"] - b["top"]))
    union = a_area + b_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def normalized_center_distance(a, b):
    ax = (a["left"] + a["right"]) / 2.0
    ay = (a["top"] + a["bottom"]) / 2.0
    bx = (b["left"] + b["right"]) / 2.0
    by = (b["top"] + b["bottom"]) / 2.0
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def parse_yolo_label_file(label_path):
    parsed = []
    if not os.path.exists(label_path):
        return parsed

    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                print(f"Warning: malformed label line {line_no} in {label_path}: {stripped}")
                continue
            try:
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                print(f"Warning: non-numeric label line {line_no} in {label_path}: {stripped}")
                continue

            parsed.append(
                {
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "source": "existing",
                }
            )
    return parsed


def prediction_to_yolo(prediction, image_shape):
    bbox = io.unstring_json(prediction["bbox"])
    img_h, img_w = image_shape[:2]
    x_center = (bbox["left"] + bbox["right"]) / 2.0 / img_w
    y_center = (bbox["top"] + bbox["bottom"]) / 2.0 / img_h
    width = (bbox["right"] - bbox["left"]) / img_w
    height = (bbox["bottom"] - bbox["top"]) / img_h
    return {
        "class_id": prediction["class_id"],
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
        "source": "predicted",
        "conf": prediction.get("conf"),
    }


def should_suppress_prediction(pred_box, existing_box):
    pred_xyxy = yolo_to_xyxy(pred_box)
    existing_xyxy = yolo_to_xyxy(existing_box)
    iou = xyxy_iou(pred_xyxy, existing_xyxy)
    if iou < IOU_SAME_CLASS_THRESHOLD:
        return False
    if not USE_CENTER_DISTANCE_CHECK:
        return True
    center_dist = normalized_center_distance(pred_xyxy, existing_xyxy)
    return center_dist <= CENTER_DIST_THRESHOLD_NORM


def is_protected_class_conflict(pred_box, existing_box):
    existing_class_id = existing_box.get("class_id")
    if existing_class_id not in PROTECTED_EXISTING_CLASS_IDS:
        return False

    pred_xyxy = yolo_to_xyxy(pred_box)
    existing_xyxy = yolo_to_xyxy(existing_box)
    iou = xyxy_iou(pred_xyxy, existing_xyxy)
    return iou >= PROTECTED_CLASS_IOU_THRESHOLD


def merge_existing_and_predictions(existing_labels, predicted_labels):
    merged = list(existing_labels)
    suppressed = 0
    kept_predictions = 0
    suppressed_by_protected_class = 0
    kept_prediction_indices = []

    for pred_idx, pred in enumerate(predicted_labels):
        pred_class = pred["class_id"]
        if pred_class not in NEW_CLASSES_TO_ADD:
            continue

        # Protected existing classes (money/credit-card) always overrule
        # overlapping new detections, regardless of predicted class.
        protected_conflict = False
        for existing in existing_labels:
            if is_protected_class_conflict(pred, existing):
                protected_conflict = True
                suppressed += 1
                suppressed_by_protected_class += 1
                break
        if protected_conflict:
            continue

        suppress = False
        for existing in existing_labels:
            if existing["class_id"] != pred_class:
                continue
            if should_suppress_prediction(pred, existing):
                suppress = True
                suppressed += 1
                break

        if not suppress:
            merged.append(pred)
            kept_predictions += 1
            kept_prediction_indices.append(pred_idx)

    return merged, kept_predictions, suppressed, suppressed_by_protected_class, kept_prediction_indices


def write_yolo_labels(label_path, labels):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for lbl in labels:
            f.write(
                f"{lbl['class_id']} {lbl['x_center']:.6f} {lbl['y_center']:.6f} {lbl['width']:.6f} {lbl['height']:.6f}\n"
            )


def do_yolo_detections(result, image, image_path, model, class_map, model_name="model"):
    image_id = result.image_id
    imagename = result.imagename

    print(f"Image {image_id} - Performing YOLO detections. model={model_name}")
    unrefined_detect_results = yolo.detect_objects_return_bbox(model, image, device, conf_thresh=CONF_THRESHOLD)
    print(f"Image {image_id} - Unrefined {model_name} detections: {unrefined_detect_results}")

    mapped_results = yolo.map_custom_ids_to_global(unrefined_detect_results, class_map)
    mapped_results = [d for d in mapped_results if d.get("class_id") is not None]

    detect_results = yolo.merge_yolo_detections(mapped_results, iou_threshold=0.3, adjacency_threshold_px=50)
    print(f"Image {image_id} - {model_name} detections: {detect_results}")

    return detect_results


def main():
    # FILE_FOLDER contains images and labels subfolders.
    image_paths = list_images_recursive(INPUT_IMAGES_FOLDER)
    print("len(image_paths)", len(image_paths))

    total_existing = 0
    total_predicted = 0
    total_kept_predictions = 0
    total_suppressed_predictions = 0
    total_suppressed_by_protected_class = 0
    total_exported_images = 0
    total_skipped_without_new = 0

    for image_path in image_paths:
        rel_image_path = os.path.relpath(image_path, INPUT_IMAGES_FOLDER)
        img_name = os.path.basename(image_path)
        print("processing image:", rel_image_path)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image at {image_path}. Skipping.")
            continue

        image_id = image_id_from_filename(img_name)
        result = type('obj', (object,), {'image_id': image_id, 'imagename': img_name, 'bbox': '{}'})
        coco_detections = do_yolo_detections(
            result,
            image,
            image_path,
            yolo_model,
            COCO_ids_to_global_dict,
            model_name="COCO",
        )
        custom_detections = do_yolo_detections(
            result,
            image,
            image_path,
            yolo_custom_model,
            custom_ids_to_global_dict,
            model_name="custom",
        )
        combined_detections = coco_detections + custom_detections

        rel_no_ext = os.path.splitext(rel_image_path)[0]
        existing_label_path = os.path.join(INPUT_LABELS_FOLDER, rel_no_ext + ".txt")

        existing_labels = parse_yolo_label_file(existing_label_path)
        predicted_labels = [prediction_to_yolo(d, image.shape) for d in combined_detections]

        merged_labels, kept_predictions, suppressed_predictions, suppressed_by_protected_class, kept_prediction_indices = merge_existing_and_predictions(
            existing_labels,
            predicted_labels,
        )

        total_existing += len(existing_labels)
        total_predicted += len(predicted_labels)
        total_kept_predictions += kept_predictions
        total_suppressed_predictions += suppressed_predictions
        total_suppressed_by_protected_class += suppressed_by_protected_class

        has_new_detections = kept_predictions > 0
        if EXPORT_ONLY_WITH_NEW_DETECTIONS and not has_new_detections:
            total_skipped_without_new += 1
            print(f"Image {image_id} - no kept new detections, skipping review export")
            continue

        if ENABLE_DEBUG_EXPORT and has_new_detections:
            kept_debug_detections = [combined_detections[i] for i in kept_prediction_indices if i < len(combined_detections)]
            yolo.save_debug_image_yolo_bbox(
                image_id,
                img_name,
                image,
                kept_debug_detections,
                image_path,
                OUTPUT_DEBUG_FOLDER,
                io,
                draw_box=IS_DRAW_BOX,
                save_undetected=False,
                move_or_copy=MOVE_OR_COPY,
                combined_class_pairs=CLASSES_TO_COMBINE,
            )

        out_image_path = os.path.join(OUTPUT_REVIEW_IMAGES, rel_image_path)
        out_label_path = os.path.join(OUTPUT_REVIEW_LABELS, rel_no_ext + ".txt")
        os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
        shutil.copy2(image_path, out_image_path)
        write_yolo_labels(out_label_path, merged_labels)
        total_exported_images += 1

        print(
            f"Image {image_id} - existing={len(existing_labels)} predicted={len(predicted_labels)} "
            f"kept_new={kept_predictions} suppressed_same_class={suppressed_predictions} "
            f"merged_total={len(merged_labels)}"
        )

    print("\\n=== Merge Summary ===")
    print(f"Images processed: {len(image_paths)}")
    print(f"Existing labels read: {total_existing}")
    print(f"Predicted labels produced: {total_predicted}")
    print(f"Predicted labels kept: {total_kept_predictions}")
    print(f"Predicted labels suppressed (same class + same location): {total_suppressed_predictions}")
    print(f"Predicted labels suppressed by protected classes {sorted(PROTECTED_EXISTING_CLASS_IDS)}: {total_suppressed_by_protected_class}")
    print(f"Images exported for review: {total_exported_images}")
    if EXPORT_ONLY_WITH_NEW_DETECTIONS:
        print(f"Images skipped (no new detections): {total_skipped_without_new}")
    print(f"Review dataset output: {OUTPUT_REVIEW_FOLDER}")
    if ENABLE_DEBUG_EXPORT:
        print(f"Debug output: {OUTPUT_DEBUG_FOLDER}")

if __name__ == "__main__":
    main()
