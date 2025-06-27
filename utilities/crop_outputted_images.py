import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2

IS_DICT_CROP = True  # Set to True if you want to crop images, False if you want to resize
crop_dict = {
    8288: 3448,
    16576: 6898,
    24866: 10344,
    33154: 13792
}

# Define the cropping parameters
CROP_TOP = 0
CROP_RIGHT = 75
CROP_BOTTOM = 150
CROP_LEFT = 75

def crop_whitespace(img):
    cropped_img = None
    height, width, _ = img.shape
    if width not in crop_dict:
        print(f"Width {width} not found in crop_dict, skipping")
        return cropped_img
    crop_width = crop_dict.get(width, 0)
    # crop the image to the crop_width keeping the image centered
    CROP_LEFT = (width - crop_width) // 2
    CROP_RIGHT = width - CROP_LEFT - crop_width
    CROP_TOP = (height - crop_width) // 2
    CROP_BOTTOM = height - CROP_TOP - crop_width
    # Define the cropped area
    cropped_img = img[CROP_TOP:height - CROP_BOTTOM, CROP_LEFT:width - CROP_RIGHT]
    return cropped_img

# Base folder containing subfolders with jpgs
base_folder = "/Volumes/OWC4/segment_images/ninecrop"

def crop_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Check if the image was successfully opened
    if img is None:
        print(f"Failed to open {image_path}")
        return
    
    # Get the original dimensions
    height, width, _ = img.shape

    if IS_DICT_CROP:
        # Get the cropping dimensions from the dictionary based on the image width
        cropped_img = crop_whitespace(img)
        if cropped_img is None:
            print(f"Failed to crop {image_path}, width {width} not found in crop_dict")
            return

    else:
        # Define the cropped area
        cropped_img = img[CROP_TOP:height - CROP_BOTTOM, CROP_LEFT:width - CROP_RIGHT]

    # Save the cropped image back to the original path
    cv2.imwrite(image_path, cropped_img)

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Only process jpg files
            if file.lower().endswith(".jpg"):
                image_path = os.path.join(root, file)
                print(f"Processing {image_path}")
                crop_image(image_path)

# Run the script
process_folder(base_folder)
