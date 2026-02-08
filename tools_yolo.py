import os
import re
import cv2
import json
import numpy as np
from mp_db_io import DataIO
from sqlalchemy.dialects.mysql import insert

class YOLOTools:
    
    def __init__(self, DEBUGGING=False):
        self.DEBUGGING = DEBUGGING 
        self.VERBOSE = True     
        self.io = DataIO(False)


    def process_image_normalize_object_bbox(self, detections_list, nose_pixel_pos, face_height, image_shape, OBJ_CLS_LIST=None):
        """
        Normalize bboxes for each detection in the list.
        Args:
            detections_list: list of dicts with keys: class_id, obj_no, bbox (JSON string), conf
            nose_pixel_pos: dict with 'x' and 'y' keys
            face_height: float
            image_shape: tuple (height, width, channels)
        Returns:
            detections_list with added 'bbox_norm' key (JSON string)
        """
        if OBJ_CLS_LIST is not None:
            print(" Normalizing object bbox THE OLD WAY for bbox_dict:", bbox_dict)
            bbox_dict = detections_list
            for OBJ_CLS_ID in OBJ_CLS_LIST:
                # if self.VERBOSE: print("OBJ_CLS_ID to norm", OBJ_CLS_ID)
                bbox_key = "bbox"
                # conf_key = "conf_{0}".format(OBJ_CLS_ID)
                bbox_n_key = "bbox_{0}_norm".format(OBJ_CLS_ID)
                # if self.VERBOSE: print("OBJ_CLS_ID", OBJ_CLS_ID)
                try: 
                    # if self.VERBOSE: print("trying to get bbox", OBJ_CLS_ID)
                    bbox_dict_value = bbox_dict[OBJ_CLS_ID]["bbox"]
                    bbox_dict_value = self.io.unstring_json(bbox_dict_value)
                except: 
                    # if self.VERBOSE: print("no bbox", OBJ_CLS_ID)
                    bbox_dict_value = None
                if bbox_dict_value and not nose_pixel_pos:
                    print("normalized bbox but no nose_pixel_pos for ")
                elif bbox_dict_value:
                    # if self.VERBOSE: print("setting normed bbox for OBJ_CLS_ID", OBJ_CLS_ID)
                    if self.VERBOSE: print("bbox_dict_value", OBJ_CLS_ID, bbox_dict_value)
                    # if self.VERBOSE: print("bbox_n_key", bbox_n_key)

                    n_phone_bbox=self.normalize_phone_bbox(bbox_dict_value,nose_pixel_pos,face_height,image_shape)
                    bbox_dict[bbox_n_key]=n_phone_bbox
                    # if self.VERBOSE: print("normed bbox", bbox_dict[bbox_n_key])
                else:
                    pass
                    # if self.VERBOSE: print(f"NO {bbox_key} for",)
            return bbox_dict
        else:
            # the new way - allowing multiple objects per class
            for detection in detections_list:
                bbox_str = detection.get("bbox")
                if not bbox_str:
                    detection["bbox_norm"] = None
                    continue
                
                try:
                    bbox_dict = json.loads(bbox_str) if isinstance(bbox_str, str) else bbox_str
                except:
                    if self.VERBOSE: print(f"Could not parse bbox for class_id {detection['class_id']}, obj_no {detection['obj_no']}")
                    detection["bbox_norm"] = None
                    continue
                
                if bbox_dict and not nose_pixel_pos:
                    print(f"Warning: bbox exists but no nose_pixel_pos for class_id {detection['class_id']}")
                    detection["bbox_norm"] = None
                elif bbox_dict and nose_pixel_pos:
                    if self.VERBOSE: print(f"Normalizing bbox for class_id {detection['class_id']}, obj_no {detection['obj_no']}")
                    n_bbox = self.normalize_phone_bbox(bbox_dict, nose_pixel_pos, face_height, image_shape)
                    detection["bbox_norm"] = json.dumps(n_bbox)
                else:
                    detection["bbox_norm"] = None
            
            return detections_list


    def save_obj_bbox(self, session, image_id, detections_list, Detections, OBJ_CLS_LIST=None):
        """
        Save or update detections for an image using MySQL upsert.
        Args:
            session: SQLAlchemy session
            image_id: int
            detections_list: list of dicts with keys: class_id, obj_no, bbox, conf, bbox_norm
            Detections: SQLAlchemy model class
        """
        if OBJ_CLS_LIST is not None:
            print(" SAVING object bbox THE OLD WAY for bbox_dict:", detections_list)
            bbox_dict = detections_list
            for OBJ_CLS_ID in OBJ_CLS_LIST:
                bbox_n_key = f"bbox_{OBJ_CLS_ID}_norm"
                bbox_norm = bbox_dict.get(bbox_n_key, None)
                        # print(bbox_dict)
                if bbox_dict[OBJ_CLS_ID]["bbox"]:
                    # if we have a bbox for this ID Create a new Detections entry
                    new_entry_Detections = Detections(
                        image_id=image_id,
                        class_id=OBJ_CLS_ID,
                        obj_no=1, # for the moment, with the old way, only kept one object per class
                        bbox = bbox_dict[OBJ_CLS_ID]["bbox"],
                        conf = bbox_dict[OBJ_CLS_ID]["conf"],
                        bbox_norm = bbox_norm
                        )
                    # Upsert using MySQL ON DUPLICATE KEY UPDATE
                    stmt = (
                        insert(Detections)
                        .values(
                            image_id=image_id,
                            class_id=OBJ_CLS_ID,
                            obj_no=1,
                            bbox=bbox_dict[OBJ_CLS_ID]["bbox"],
                            conf=bbox_dict[OBJ_CLS_ID]["conf"],
                            bbox_norm=bbox_norm,
                        )
                        .on_duplicate_key_update(
                            conf=Detections.conf,
                            bbox=Detections.bbox,
                            bbox_norm=Detections.bbox_norm,
                        )
                    )
                    session.execute(stmt)
                            
                    if self.VERBOSE: print(f"New Bbox {OBJ_CLS_ID} session entry for image_id {image_id} created successfully.")
                else:
                    pass
                    # if self.VERBOSE: print(f"No bbox for {OBJ_CLS_ID} in image_id {image_id}")
        else:
            # the new way - allowing multiple objects per class
            for detection in detections_list:
                bbox_str = detection.get("bbox")
                bbox_norm_str = detection.get("bbox_norm")
                print("detection is", detection)
                if not bbox_str:
                    if self.VERBOSE: print(f"Skipping empty bbox for class_id {detection['class_id']}, obj_no {detection['obj_no']}")
                    continue
                
                # Upsert using MySQL ON DUPLICATE KEY UPDATE
                stmt = (
                    insert(Detections)
                    .values(
                        image_id=image_id,
                        class_id=detection.get("class_id", None),
                        obj_no=detection.get("obj_no", None),
                        bbox=bbox_str,
                        conf=detection.get("conf", None),
                        bbox_norm=bbox_norm_str,
                        meta_cluster_id = detection.get("meta_cluster_id", None),
                        cluster_id = detection.get("cluster_id", None)
                    )
                    .on_duplicate_key_update(
                        bbox=bbox_str,
                        conf=detection.get("conf", None),
                        bbox_norm=bbox_norm_str,
                    )
                )
                try:
                    # print the statement first
                    print(f"Executing upsert for image_id {image_id}, class_id {detection['class_id']}, obj_no {detection['obj_no']}")
                    result = session.execute(stmt)
                    rowcount = getattr(result, "rowcount", None)
                except Exception as e:
                    if self.VERBOSE: print(f"Upsert failed for image_id {image_id}, class_id {detection['class_id']}, obj_no {detection['obj_no']}: {e}")
                    raise
                
                if self.VERBOSE:
                    print(f"Upserted detection for image_id {image_id}, class_id {detection['class_id']}, obj_no {detection['obj_no']}")
            
        # Commit the transaction so the upserts are persisted.
        try:
            session.commit()
            if self.VERBOSE:
                print(f"Committed session for image_id {image_id}")
        except Exception as e:
            session.rollback()
            if self.VERBOSE:
                print(f"Commit failed for image_id {image_id}: {e}")
            raise

        return session


### YOLO
    def detect_objects_return_bbox(self, model, image, device=None, conf_thresh=0.45, OBJ_CLS_LIST=None):
        """
        Detect objects and return list of detections, allowing multiple instances per class.
        Returns: list of dicts, each with keys: class_id, obj_no, bbox, conf
        """
        bbox_dict={}
        if OBJ_CLS_LIST is not None:
            # the old way
            result = model(image,classes=[OBJ_CLS_LIST])[0]
            bbox_count=np.zeros(len(OBJ_CLS_LIST))
            for i,obj_cls_id in enumerate(OBJ_CLS_LIST):
                for box in result.boxes:
                    if int(box.cls[0].item())==obj_cls_id:
                        bbox = box.xyxy[0].tolist()    #the coordinates of the box as an array [x1,y1,x2,y2]
                        bbox = {"left":round(bbox[0]),"top":round(bbox[1]),"right":round(bbox[2]),"bottom":round(bbox[3])}
                        bbox=json.dumps(bbox)
                        # bbox=json.dumps(bbox, indent = 4) 
                        conf = round(box.conf[0].item(), 2)                
                        bbox_count[i]+=1 
                        bbox_dict[obj_cls_id]={"bbox": bbox, "conf": conf}

            for i,obj_cls_id in enumerate(OBJ_CLS_LIST):
                if bbox_count[i]>1: # checking to see it there are more than one objects of a class and removing 
                    bbox_dict.pop(obj_cls_id)
                    bbox_dict[obj_cls_id]={"bbox": None, "conf": -1} ##setting to default
                if bbox_count[i]==0:
                    bbox_dict[obj_cls_id]={"bbox": None, "conf": -1} ##setting to default
            return bbox_dict

        else:
            # results = model(image)

            # # Show results
            # results[0].show()

            # # Or get detailed info
            # for r in results:
            #     boxes = r.boxes
            #     for box in boxes:
            #         cls = int(box.cls[0])
            #         conf = float(box.conf[0])
            #         print(f"Class: {r.names[cls]}, Confidence: {conf:.2f}")

            # the new way - allowing multiple objects per class
            result = model.predict(
                        image,
                        conf=conf_thresh,
                        device=device,
                        verbose=False
                    )[0]        
            # Group detections by class_id and assign obj_no
            detections_by_class = {}
            # print("YOLO detection results:", result)
            for box in result.boxes:
                # print("box info:", box)
                obj_cls_id = int(box.cls[0].item())
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                bbox_dict = {
                    "left": round(bbox[0]),
                    "top": round(bbox[1]),
                    "right": round(bbox[2]),
                    "bottom": round(bbox[3])
                }
                conf = round(box.conf[0].item(), 2)
                # print(f"Detected class_id {obj_cls_id} with conf {conf} and bbox {bbox_dict}")
                if obj_cls_id not in detections_by_class:
                    detections_by_class[obj_cls_id] = []
                
                detections_by_class[obj_cls_id].append({
                    "bbox": bbox_dict,
                    "conf": conf
                })
            # print("Detections by class (before obj_no assignment):", detections_by_class)
            # Build final list with obj_no assigned per class
            detections_list = []
            for class_id, class_detections in detections_by_class.items():
                for obj_no, detection in enumerate(class_detections, start=1):
                    detections_list.append({
                        "class_id": class_id,
                        "obj_no": obj_no,
                        "bbox": json.dumps(detection["bbox"]),
                        "conf": detection["conf"]
                    })
            
            return detections_list




    def normalize_phone_bbox(self,phone_bbox,nose_pos,face_height,shape):
        height,width = shape[:2]
        if self.VERBOSE: print("phone_bbox",phone_bbox)
        if self.VERBOSE: print("type phone_bbox",type(phone_bbox))

        n_phone_bbox=phone_bbox
        n_phone_bbox["right"] =(nose_pos["x"]-n_phone_bbox["right"] )/face_height
        n_phone_bbox["left"]  =(nose_pos["x"]-n_phone_bbox["left"]  )/face_height
        n_phone_bbox["top"]   =(nose_pos["y"]-n_phone_bbox["top"]   )/face_height
        n_phone_bbox["bottom"]=(nose_pos["y"]-n_phone_bbox["bottom"])/face_height
        if self.VERBOSE: print("type phone_bbox",type(n_phone_bbox["right"]))

        return n_phone_bbox

    # not currently in use (deprecated)
    def parse_bbox_dict(self, session, target_image_id, PhoneBbox, OBJ_CLS_LIST, bbox_dict):
        # I don't think it likes me sending PhoneBbox as a class
        # for calc face pose i'm moving this back to function
        for OBJ_CLS_ID in OBJ_CLS_LIST:
            bbox_n_key = "bbox_{0}_norm".format(OBJ_CLS_ID)
            print(bbox_dict)
            if bbox_dict[OBJ_CLS_ID]["bbox"]:
                PhoneBbox_entry = (
                    session.query(PhoneBbox)
                    .filter(PhoneBbox.image_id == target_image_id)
                    .first()
                )

                if PhoneBbox_entry:
                    setattr(PhoneBbox_entry, "bbox_{0}".format(OBJ_CLS_ID), bbox_dict[OBJ_CLS_ID]["bbox"])
                    setattr(PhoneBbox_entry, "conf_{0}".format(OBJ_CLS_ID), bbox_dict[OBJ_CLS_ID]["conf"])
                    setattr(PhoneBbox_entry, bbox_n_key, bbox_dict[bbox_n_key])
                    print("image_id:", PhoneBbox_entry.target_image_id)
                    #session.commit()
                    print(f"Bbox {OBJ_CLS_ID} for image_id {target_image_id} updated successfully.")
                else:
                    print(f"Bbox {OBJ_CLS_ID} for image_id {target_image_id} not found.")
            else:
                print(f"No bbox for {OBJ_CLS_ID} in image_id {target_image_id}")
        
        return session

    def hsv_on_cropped_image(self, cropped_image_bbox_hsvslice):
        # Calculate the average HSV of the cropped_image_bbox_hsvslice
        hsv_image = cv2.cvtColor(cropped_image_bbox_hsvslice, cv2.COLOR_BGR2HSV)
        lab_image = cv2.cvtColor(cropped_image_bbox_hsvslice, cv2.COLOR_BGR2LAB)

        average_hsv = np.median(hsv_image, axis=(0, 1))
        average_lab = np.median(lab_image, axis=(0, 1))

        hue = average_hsv[0]
        saturation = average_hsv[1]
        value = average_hsv[2]
        luminance = average_lab[0]
        # print("Average HSV:", average_hsv)

        # display the average HSV values as a color swatch 100x100 pixels
        swatch = cv2.cvtColor(np.uint8([[average_hsv]]), cv2.COLOR_HSV2BGR)
        swatch = cv2.resize(swatch, (100, 100), interpolation=cv2.INTER_NEAREST)

        return hue, saturation, value, luminance


    def get_image_bbox_hsv(self,image_bbox_slice):
        # # label data looks like: ['1 0.44188750000000004 0.7556818181818181 0.4017159090909091 0.3340909090909092\n']
        # # Parse the label data
        # label_info = label_data[0].strip().split()
        # class_id = int(label_info[0])
        # x_center = float(label_info[1])
        # y_center = float(label_info[2])
        # width = float(label_info[3])
        # height = float(label_info[4])

        # # Calculate bounding box coordinates
        # img_height, img_width = image.shape[:2]
        # x_min = int((x_center - width / 2) * img_width)
        # x_max = int((x_center + width / 2) * img_width)
        # y_min = int((y_center - height / 2) * img_height)
        # y_max = int((y_center + height / 2) * img_height)

        # # Slice the image to the bounding box
        # cropped_image_bbox = image[y_min:y_max, x_min:x_max]

        # # Take cropped_image_bbox and use cv2 to create a cropped_image_bbox_hsvslice that has the middle 50% of the pixels vertically and horizontally
        # crop_height, crop_width = cropped_image_bbox.shape[:2]
        # x_start = int(crop_width * 0.25)
        # x_end = int(crop_width * 0.75)
        # y_start = int(crop_height * 0.25)   
        # y_end = int(crop_height * 0.75)

        # cropped_image_bbox_hsvslice = cropped_image_bbox[y_start:y_end, x_start:x_end]


        hue, saturation, value, luminance = self.hsv_on_cropped_image(image_bbox_slice)

        normalized_hue = hue/360
        normalized_saturation = saturation/255
        normalized_value = value/255
        normalized_luminance = luminance/255

        return normalized_hue, normalized_saturation, normalized_value, normalized_luminance, image_bbox_slice

    def compute_mask_hsv(self,image, text_bbox_dict):
        def prep_hsl(hue, saturation, luminance):
            normalized_hue = hue / 360
            normalized_saturation = saturation / 255
            normalized_luminance = luminance / 255
            return [normalized_hue, normalized_saturation, normalized_luminance]
        # crop the image using the bbox
        cropped_image_bbox = image[text_bbox_dict['top']:text_bbox_dict['bottom'], text_bbox_dict['left']:text_bbox_dict['right']]
        print("cropped_image_bbox shape: ", cropped_image_bbox.shape)

        top_half_of_cropped = cropped_image_bbox[0:cropped_image_bbox.shape[0]//2, 0:cropped_image_bbox.shape[1]]
        bottom_half_of_cropped = cropped_image_bbox[cropped_image_bbox.shape[0]//2:cropped_image_bbox.shape[0], 0:cropped_image_bbox.shape[1]]

        top_hue, top_saturation, top_value, top_luminance = self.hsv_on_cropped_image(top_half_of_cropped)
        bot_hue, bot_saturation, bot_value, bot_luminance = self.hsv_on_cropped_image(bottom_half_of_cropped)

        # print("Top Half - Hue: {}, Saturation: {}, Value: {}, Luminance: {}".format(top_hue, top_saturation, top_value, top_luminance))
        # print("Bottom Half - Hue: {}, Saturation: {}, Value: {}, Luminance: {}".format(bot_hue, bot_saturation, bot_value, bot_luminance))

        top_hsl = prep_hsl(top_hue, top_saturation, top_luminance)
        bot_hsl = prep_hsl(bot_hue, bot_saturation, bot_luminance)
        hsl_distance = np.sqrt( (top_hsl[0]-bot_hsl[0])**2 + (top_hsl[1]-bot_hsl[1])**2 + (top_hsl[2]-bot_hsl[2])**2 )

        return top_hsl, bot_hsl, hsl_distance

    def merge_yolo_detections(self,detect_results, iou_threshold=0.5, adjacency_threshold_px=20, skip_class_zero=True):
        '''
        Merge multiple detections of the same class if they overlap (IoU > iou_threshold)
        or if they are adjacent/touching (gap < adjacency_threshold_px).
        Use case: stack of books detected as 6 separate objects; merge into 1.
        
        Args:
            detect_results: list of dicts with class_id, obj_no, bbox (JSON str), conf
            iou_threshold: IoU threshold for overlap merging (default 0.5)
            adjacency_threshold_px: max gap (pixels) to consider bboxes adjacent (default 20)
            skip_class_zero: if True, ignore class_id 0 while merging (useful for COCO persons)
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
            if skip_class_zero and detect_results[i]['class_id'] == 0:
                continue
            
            current = detect_results[i]
            current_bbox = self.io.unstring_json(current['bbox'])
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
                
                compare_bbox = self.io.unstring_json(compare['bbox'])
                
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

    def find_valentine_bbox(self, bbox, img, RED_THRESH=200, RED_DOM=150, use_average_per_row=True):
        """
        Finds a bounding box tightly enclosing the contiguous red band, using the x values of the bbox to determine the column to check.
        Searches the full image height (not restricted to bbox's y range), matching is_valentine's behavior.

        Args:
            bbox: Dictionary with keys 'left', 'top', 'right', 'bottom' of bounding box coordinates
            img: Image array
            use_average_per_row: If True, uses average channel values per row. If False, uses pixel-by-pixel detection.

        Returns a tuple (x1, y1_new, x2, y2_new) where y1_new and y2_new correspond to the red region.
        Returns None if no suitable red area is found.
        """
        print("Finding valentine bbox within:", bbox, type(bbox))
        x1 = bbox['left']
        y1 = bbox['top']
        x2 = bbox['right']
        y2 = bbox['bottom']

        height, width, channels = img.shape
        x1c = max(int(round(x1)), 0)
        x2c = min(int(round(x2)), width)
        # Calculate bbox width for padding
        bbox_width = x2c - x1c
        # Pad by one bbox width on either side
        x1c_padded = max(x1c - bbox_width, 0)
        x2c_padded = min(x2c + bbox_width, width)
        # Extract the vertical band using x values from bbox, but search full image height
        patch = img[:, x1c_padded:x2c_padded, :]

        # Thresholds for valentine-red
        # Note: OpenCV uses BGR format, so channel 2 is red, channel 1 is green, channel 0 is blue


        # Convert to int to prevent uint8 overflow when subtracting
        red_channel = patch[..., 2].astype(np.int16)
        green_channel = patch[..., 1].astype(np.int16)
        blue_channel = patch[..., 0].astype(np.int16)
        
        if use_average_per_row:
            # Average method: compute mean channel values per row
            mean_red_per_row = red_channel.mean(axis=1)  # shape (patch_height,)
            mean_green_per_row = green_channel.mean(axis=1)  # shape (patch_height,)
            mean_blue_per_row = blue_channel.mean(axis=1)  # shape (patch_height,)
            # Check if row averages meet the red dominance criteria
            red_per_row = (
                (mean_red_per_row >= RED_THRESH) &
                (mean_red_per_row - mean_green_per_row > RED_DOM) &
                (mean_red_per_row - mean_blue_per_row > RED_DOM)
            )
        else:
            # Pixel-by-pixel method: check each pixel individually
            red_mask = (
                (red_channel >= RED_THRESH) &
                (red_channel - green_channel > RED_DOM) &
                (red_channel - blue_channel > RED_DOM)
            )
            # Combine horizontally -- any pixel in row counts as "red row"
            red_per_row = red_mask.any(axis=1)  # shape (patch_height,)

        # Find start and end of largest contiguous span of True
        max_len = 0
        max_start = None
        max_end = None
        curr_start = None
        curr_len = 0
        for idx, is_red in enumerate(red_per_row):
            if is_red:
                if curr_start is None:
                    curr_start = idx
                curr_len += 1
            else:
                if curr_len > max_len:
                    max_len = curr_len
                    max_start = curr_start
                    max_end = curr_start + curr_len
                curr_start = None
                curr_len = 0
        # Handle the case where the max run is at the end
        if curr_len > max_len:
            max_len = curr_len
            max_start = curr_start
            max_end = curr_start + curr_len

        if max_len == 0 or max_start is None:  # No red detected
            return None

        # max_start and max_end are already in image coordinates (0 to height-1)
        y1_red = max_start
        y2_red = max_end

        # Tighten horizontally: find columns with red pixels in the largest red span (rows max_start:max_end)
        # Get the relevant subpatch
        relevant_patch = patch[max_start:max_end, :, :]

        # Recompute red detection for the relevant_patch (may be redundant, but ensures correctness)
        # Note: OpenCV uses BGR format, so channel 2 is red, channel 1 is green, channel 0 is blue
        # Convert to int to prevent uint8 overflow when subtracting
        relevant_red_channel = relevant_patch[..., 2].astype(np.int16)
        relevant_green_channel = relevant_patch[..., 1].astype(np.int16)
        relevant_blue_channel = relevant_patch[..., 0].astype(np.int16)
        
        if use_average_per_row:
            # Average method: compute mean channel values per column
            mean_red_per_col = relevant_red_channel.mean(axis=0)  # shape (patch_width,)
            mean_green_per_col = relevant_green_channel.mean(axis=0)  # shape (patch_width,)
            mean_blue_per_col = relevant_blue_channel.mean(axis=0)  # shape (patch_width,)
            # Check if column averages meet the red dominance criteria
            red_per_col = (
                (mean_red_per_col >= RED_THRESH) &
                (mean_red_per_col - mean_green_per_col > RED_DOM) &
                (mean_red_per_col - mean_blue_per_col > RED_DOM)
            )
        else:
            # Pixel-by-pixel method: check each pixel individually
            relevant_red_mask = (
                (relevant_red_channel >= RED_THRESH) &
                (relevant_red_channel - relevant_green_channel > RED_DOM) &
                (relevant_red_channel - relevant_blue_channel > RED_DOM)
            )
            red_per_col = relevant_red_mask.any(axis=0)  # shape (patch_width,)

        # Find the leftmost and rightmost columns containing at least one red pixel
        red_cols = red_per_col.nonzero()[0]
        if len(red_cols) == 0:
            # Should not happen if vertical band is red, but just in case
            return (x1, y1_red, x2, y2_red)
        x1_rel = red_cols[0]
        x2_rel = red_cols[-1] + 1  # upper bound exclusive

        x1_red = x1c_padded + x1_rel
        x2_red = x1c_padded + x2_rel

        # Convert back to float coordinates for output, but you may want to round or keep as ints depending on convention
        return (x1_red, y1_red, x2_red, y2_red)

