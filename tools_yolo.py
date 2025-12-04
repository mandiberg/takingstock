import os
import re
import cv2
import json
import numpy as np
from mp_db_io import DataIO
from sqlalchemy.dialects.mysql import insert

class YOLOTools:
    """
    Two-Phase Iterative TSP Solver with Point Skipping
    
    Phase 1: Solve TSP for current point set
    Phase 2: Greedily remove high-overhead points
    Iterate: Re-optimize TSP on remaining points
    """
    
    def __init__(self, DEBUGGING=False):
        self.DEBUGGING = DEBUGGING 
        self.VERBOSE = True     
        self.io = DataIO(False)


    def process_image_normalize_object_bbox(self,bbox_dict, nose_pixel_pos, face_height, image_shape, OBJ_CLS_LIST):
        ### normed object bbox
        if self.VERBOSE: print("Normalizing object bbox for bbox_dict:", bbox_dict)
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


    def save_obj_bbox(self, session, image_id, bbox_dict, Detections, OBJ_CLS_LIST):
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
        return session


### YOLO
    def return_bbox(self, model, image, OBJ_CLS_LIST):
        result = model(image,classes=[OBJ_CLS_LIST])[0]
        bbox_dict={}
        bbox_count=np.zeros(len(OBJ_CLS_LIST))
        for i,OBJ_CLS_ID in enumerate(OBJ_CLS_LIST):
            for box in result.boxes:
                if int(box.cls[0].item())==OBJ_CLS_ID:
                    bbox = box.xyxy[0].tolist()    #the coordinates of the box as an array [x1,y1,x2,y2]
                    bbox = {"left":round(bbox[0]),"top":round(bbox[1]),"right":round(bbox[2]),"bottom":round(bbox[3])}
                    bbox=json.dumps(bbox)
                    # bbox=json.dumps(bbox, indent = 4) 
                    conf = round(box.conf[0].item(), 2)                
                    bbox_count[i]+=1 
                    bbox_dict[OBJ_CLS_ID]={"bbox": bbox, "conf": conf}

        for i,OBJ_CLS_ID in enumerate(OBJ_CLS_LIST):
            if bbox_count[i]>1: # checking to see it there are more than one objects of a class and removing 
                bbox_dict.pop(OBJ_CLS_ID)
                bbox_dict[OBJ_CLS_ID]={"bbox": None, "conf": -1} ##setting to default
            if bbox_count[i]==0:
                bbox_dict[OBJ_CLS_ID]={"bbox": None, "conf": -1} ##setting to default
        return bbox_dict


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


    def get_image_bbox_hsv(self,image, label_data):
        # label data looks like: ['1 0.44188750000000004 0.7556818181818181 0.4017159090909091 0.3340909090909092\n']
        # Parse the label data
        label_info = label_data[0].strip().split()
        class_id = int(label_info[0])
        x_center = float(label_info[1])
        y_center = float(label_info[2])
        width = float(label_info[3])
        height = float(label_info[4])

        # Calculate bounding box coordinates
        img_height, img_width = image.shape[:2]
        x_min = int((x_center - width / 2) * img_width)
        x_max = int((x_center + width / 2) * img_width)
        y_min = int((y_center - height / 2) * img_height)
        y_max = int((y_center + height / 2) * img_height)

        # Slice the image to the bounding box
        cropped_image_bbox = image[y_min:y_max, x_min:x_max]

        # Take cropped_image_bbox and use cv2 to create a cropped_image_bbox_hsvslice that has the middle 50% of the pixels vertically and horizontally
        crop_height, crop_width = cropped_image_bbox.shape[:2]
        x_start = int(crop_width * 0.25)
        x_end = int(crop_width * 0.75)
        y_start = int(crop_height * 0.25)   
        y_end = int(crop_height * 0.75)

        cropped_image_bbox_hsvslice = cropped_image_bbox[y_start:y_end, x_start:x_end]


        hue, saturation, value, luminance = self.hsv_on_cropped_image(cropped_image_bbox_hsvslice)

        return hue, saturation, value, luminance, cropped_image_bbox

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
