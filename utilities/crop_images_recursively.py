import os
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT_FOLDER_PATH = '/Volumes/OWC4/segment_images'
FOLDER_NAME = "cluster24_126_1745979928.328164"
FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH, FOLDER_NAME)

OUTPUT_DIMS = 4000

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
        if width > OUTPUT_DIMS and height > OUTPUT_DIMS:
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
