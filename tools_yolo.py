import os
import re
import json
import numpy as np
from mp_db_io import DataIO

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
        for OBJ_CLS_ID in OBJ_CLS_LIST:
            if self.VERBOSE: print("OBJ_CLS_ID to norm", OBJ_CLS_ID)
            bbox_key = "bbox"
            # conf_key = "conf_{0}".format(OBJ_CLS_ID)
            bbox_n_key = "bbox_{0}_norm".format(OBJ_CLS_ID)
            if self.VERBOSE: print("OBJ_CLS_ID", OBJ_CLS_ID)
            try: 
                if self.VERBOSE: print("trying to get bbox", OBJ_CLS_ID)
                bbox_dict_value = bbox_dict[OBJ_CLS_ID]["bbox"]
                bbox_dict_value = self.io.unstring_json(bbox_dict_value)
            except: 
                if self.VERBOSE: print("no bbox", OBJ_CLS_ID)
                bbox_dict_value = None
            if bbox_dict_value and not nose_pixel_pos:
                print("normalized bbox but no nose_pixel_pos for ")
            elif bbox_dict_value:
                if self.VERBOSE: print("setting normed bbox for OBJ_CLS_ID", OBJ_CLS_ID)
                if self.VERBOSE: print("bbox_dict_value", bbox_dict_value)
                if self.VERBOSE: print("bbox_n_key", bbox_n_key)

                n_phone_bbox=self.normalize_phone_bbox(bbox_dict_value,nose_pixel_pos,face_height,image_shape)
                bbox_dict[bbox_n_key]=n_phone_bbox
                if self.VERBOSE: print("normed bbox", bbox_dict[bbox_n_key])
            else:
                pass
                if self.VERBOSE: print(f"NO {bbox_key} for",)
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

                # if self.VERBOSE: print(f"bbox_dict[OBJ_CLS_ID][bbox]: {bbox_dict[OBJ_CLS_ID]['bbox']}")
                # if self.VERBOSE: print(f"bbox_dict[OBJ_CLS_ID][conf]: {bbox_dict[OBJ_CLS_ID]['conf']}")
                # if self.VERBOSE: print("bbox_n_key:", bbox_n_key)
                # if self.VERBOSE: print(f"bbox_dict[bbox_n_key]: {bbox_dict[bbox_n_key]}")

                        # Set attributes
                # setattr(new_entry_Detections, f"bbox_{OBJ_CLS_ID}", bbox_dict[OBJ_CLS_ID]["bbox"])
                # setattr(new_entry_Detections, f"conf_{OBJ_CLS_ID}", bbox_dict[OBJ_CLS_ID]["conf"])
                # try:
                #     setattr(new_entry_Detections, bbox_n_key, bbox_dict[bbox_n_key])
                # except:
                #     print(f"Error setting {bbox_n_key} for {image_id}")
                #         # Add the new entry to the session
                session.merge(new_entry_Detections)
                        
                if self.VERBOSE: print(f"New Bbox {OBJ_CLS_ID} session entry for image_id {image_id} created successfully.")
            else:
                pass
                if self.VERBOSE: print(f"No bbox for {OBJ_CLS_ID} in image_id {image_id}")
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
