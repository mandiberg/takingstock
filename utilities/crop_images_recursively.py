import os
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT_FOLDER_PATH = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/'
FOLDER_NAME = "body3D_testcrop"
FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH, FOLDER_NAME)

OUTPUT_DIMS = 4000

OUTPUT_TRIM = 2072
EXISTING_DIM = 8288

def trim_top_crop(image):
    height, width = image.shape[:2]
    if height == (EXISTING_DIM-OUTPUT_TRIM) and width == EXISTING_DIM:
        print(f"Already cropped:")
        return None
    elif height == EXISTING_DIM and width == EXISTING_DIM:
        # remove OUTPUT_TRIM pixels from the top, no changes to the width
        cropped_img = image[OUTPUT_TRIM:, :width]
        return cropped_img
    else:
        print(f"Image dimensions do not match expected: {height}x{width}")
        return None
    
def center_crop(image, output_size):
    height, width = image.shape[:2]
    start_x = (width - output_size) // 2
    start_y = (height - output_size) // 2
    return image[start_y:start_y+output_size, start_x:start_x+output_size]

def process_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return f"Could not read: {image_path}"

        height, width = img.shape[:2]
        # return f"Processing {image_path}: {width}x{height}"
        if height == EXISTING_DIM or width == EXISTING_DIM:
            # If the image is already at the expected dimension, we can trim the top
            img = trim_top_crop(img)
            if img is None:
                return f"already cropped or bad dims: {image_path}"
            else:
                cv2.imwrite(image_path, img)
            return f"Cropped and saved: {image_path}"
        # if height == (EXISTING_DIM-OUTPUT_TRIM) and width == EXISTING_DIM:
        #     return f"Already cropped: {image_path}"
        # elif height == EXISTING_DIM and width == EXISTING_DIM:
        #     # remove OUTPUT_TRIM pixels from the top, no changes to the width
        #     cropped_img = img[OUTPUT_TRIM:, :width]
        #     cv2.imwrite(image_path, cropped_img)
        #     return f"Cropped and saved: {image_path}"

        elif width > OUTPUT_DIMS and height > OUTPUT_DIMS:
            cropped_img = center_crop(img, OUTPUT_DIMS)
            cv2.imwrite(image_path, cropped_img)
            return f"Cropped and saved: {image_path}"
        else:
            return f"Skipped (too small): {image_path}"
    except Exception as e:
        return f"Error processing {image_path}: {e}"

def gather_image_paths(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

if __name__ == "__main__":
    image_paths = gather_image_paths(FOLDER_PATH)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, path) for path in image_paths]
        for future in as_completed(futures):
            print(future.result())
