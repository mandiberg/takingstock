import os
import cv2

# Define the cropping parameters
CROP_TOP = 0
CROP_RIGHT = 75
CROP_BOTTOM = 150
CROP_LEFT = 75

# Base folder containing subfolders with jpgs
base_folder = "/Volumes/OWC4/segment_images/topic32_128d_crop"

def crop_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Check if the image was successfully opened
    if img is None:
        print(f"Failed to open {image_path}")
        return
    
    # Get the original dimensions
    height, width, _ = img.shape

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
