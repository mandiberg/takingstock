'''
use testocr environment
this code opens images, segments to bbox, gets hsv
does ocr, cleans text with openai
needs to: 
1. get image list from sql (segmenthelper)
1.5 load all slogans from slogans table (slogan_id and slogan_text) into a dict
2. match hsv to hsvclusters
3. save hsvcluster
3.5 check if unrefined slogan text matches any dict.values
    - if so, skip the refinement step, and save_placard_text(image_id, slogan_id) to placards table
    - if not, refine with openai
4. save text info: 
    1st check to see if refined slogan text matches any dict.values
    - if so, save corresponding slogan_id to slogans table
    - if not, save text to slogans table with slogan_id and slogan_text. 
        - add slogan_id and slogan_text to dict
    2nd save_placard_text(image_id, slogan_id) to placards table
'''

import cv2
import numpy as np
import os
from paddleocr import PaddleOCR
import re
from spellchecker import SpellChecker
from openai import OpenAI
from openaiAPI import api_key
from ocr_tools import OCRTools

spell = SpellChecker()

openai_client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)

ocr_engine = PaddleOCR(
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False) # text detection + text recognition

ocr = OCRTools(DEBUGGING=True)

root_folder = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/labeled_images_nov19/"
iamges_folder = root_folder + "images_testing/"
labels_folder = root_folder + "labels/"

DEBUGGING = True
# # Load processor + model
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


# # Paths
# image_path = iamges_folder + "0a1ae152-360973921.jpg"
# label_path = labels_folder + "0a1ae152-360973921.txt"

def get_image_bbox_hsv(image, label_data):
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

    # Calculate the average HSV of the cropped_image_bbox_hsvslice
    hsv_image = cv2.cvtColor(cropped_image_bbox_hsvslice, cv2.COLOR_BGR2HSV)
    average_hsv = np.median(hsv_image, axis=(0, 1))
    # print("Average HSV:", average_hsv)

    # display the average HSV values as a color swatch 100x100 pixels
    swatch = cv2.cvtColor(np.uint8([[average_hsv]]), cv2.COLOR_HSV2BGR)
    swatch = cv2.resize(swatch, (100, 100), interpolation=cv2.INTER_NEAREST)

    # put the cropped_image_bbox_hsvslice next to the swatch for comparison
    comparison_image = np.hstack((cv2.resize(cropped_image_bbox_hsvslice, (100, 100), interpolation=cv2.INTER_NEAREST), swatch))

    # cv2.imshow("Average HSV Color Swatch", comparison_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return average_hsv, cropped_image_bbox


# load all images in images_folder
image_filenames = [f for f in os.listdir(iamges_folder) if f.endswith('.jpg')]
print("Found image files:", image_filenames)

# loop through each image file
for image_filename in image_filenames:
    image_path = os.path.join(iamges_folder, image_filename)
    label_filename = image_filename.replace('.jpg', '.txt')
    label_path = os.path.join(labels_folder, label_filename)

    # Read the image
    image = cv2.imread(image_path)

    if not os.path.exists(label_path):
        print(f"Label file not found for image {image_filename}, skipping.")
        continue
    else:
        # Read the label file
        with open(label_path, 'r') as file:
            label_data = file.readlines()

    average_hsv, cropped_image_bbox = get_image_bbox_hsv(image, label_data)
    found_texts = ocr.ocr_on_cropped_image(cropped_image_bbox, ocr_engine, image_filename)
    # print("Final Average HSV:", average_hsv)
    print("Found Texts:", found_texts)
    refined_text = ocr.clean_ocr_text(openai_client, found_texts)
    print("Refined Text:", refined_text)
