import cv2
import numpy as np
import os
import math
import sys

# mine
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from mp_db_io import DataIO

# I/O utils
io = DataIO()
SORT_ORDER = "Chronological"
ROOT_FOLDER_PATH = '/Volumes/OWC4/segment_images'
FOLDER_NAME = "topic32_128d_FINAL"
FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH,FOLDER_NAME)


def get_image_paths(root_folder):
    """Get all jpg image paths from the root folder and its subfolders."""
    all_image_paths = []
    folder_image_paths = []
    subfolders = io.get_folders(root_folder, SORT_ORDER)
    print(f"subfolders: {subfolders}")
    # for root, _, files in os.walk(root_folder):
    for folder in subfolders:
        print(f"Scanning folder: {folder}")
        files = os.listdir(folder)
        # print(f"Files: {files}")
        # print(f"type of files: {type(files)}")
        for file in files:
            if file.lower().endswith('.jpg'):
                folder_image_paths.append(os.path.join(root_folder, folder, file))
        all_image_paths.extend(sorted(folder_image_paths))
        folder_image_paths = []
    # return sorted(image_paths)
    return all_image_paths

def process_image(image_path, target_size):
    """Open and resize an image using cv2."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load {image_path}")
            return None
        return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def create_composite(images, rows, cols, img_size):
    """Create a composite image from a list of images."""
    # Create a white background (OpenCV uses BGR)
    composite = np.full((rows * img_size, cols * img_size, 3), 255, dtype=np.uint8)
    
    for idx, img in enumerate(images):
        if img is None:
            continue
        row = idx // cols
        col = idx % cols
        y_start = row * img_size
        y_end = y_start + img_size
        x_start = col * img_size
        x_end = x_start + img_size
        composite[y_start:y_end, x_start:x_end] = img
    
    return composite

def main():
    # Constants
    # FOLDER_PATH = '/Volumes/OWC4/segment_images/topic32_128d_FINAL'
    IMG_SIZE = 400
    ROWS = 15
    COLS = 20
    IMAGES_PER_COMPOSITE = ROWS * COLS
    OUTPUT_DIR = os.path.join(FOLDER_PATH,'composite_output')
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all image paths
    print("Scanning for images...")
    image_paths = get_image_paths(FOLDER_PATH)
    total_images = len(image_paths)
    print(f"Found {total_images} images")
    
    # Calculate number of composite images needed
    num_composites = math.ceil(total_images / IMAGES_PER_COMPOSITE)
    
    for composite_idx in range(num_composites):
        print(f"\nProcessing composite {composite_idx + 1} of {num_composites}")
        
        # Get the slice of images for this composite
        start_idx = composite_idx * IMAGES_PER_COMPOSITE
        end_idx = min(start_idx + IMAGES_PER_COMPOSITE, total_images)
        
        # Process images for this composite
        processed_images = []
        for i, path in enumerate(image_paths[start_idx:end_idx]):
            if i % 100 == 0:  # Progress update every 100 images
                print(f"Processing image {i + 1} of {end_idx - start_idx}")
            processed_img = process_image(path, IMG_SIZE)
            print(f"Processed {path}")
            if processed_img is not None:
                processed_images.append(processed_img)
        
        # Pad with None if we don't have enough images
        while len(processed_images) < IMAGES_PER_COMPOSITE:
            processed_images.append(None)
        
        # Create and save composite
        composite = create_composite(processed_images, ROWS, COLS, IMG_SIZE)
        output_path = os.path.join(OUTPUT_DIR, f'composite_{composite_idx+1}.jpg')
        cv2.imwrite(output_path, composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"Saved composite {composite_idx+1} to {output_path}")
        
        # No need to explicitly clean up with OpenCV as it handles memory automatically

if __name__ == "__main__":
    main()