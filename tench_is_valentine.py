import pandas as pd
import os
import sys
import pickle
import numpy as np
import cv2
from itertools import groupby

INPUT_PATH = "/Users/tenchc/Desktop/valentines"
OUTPUT_PATH = "/Users/tenchc/Desktop/outputs"
CSV_FILENAME = "valentines.csv"

RED_THRESH = 200
RED_DOM = 150

def process_valentine_images(
    input_folder=INPUT_PATH,
    csv_filename=CSV_FILENAME,
    output_folder=OUTPUT_PATH,
    use_average_per_row=False
):
    """
    Processes all the images and bounding boxes in the CSV file in the input_folder, applies is_valentine and find_valentine_bbox,
    draws a pink outline box for found valentine regions, and saves the images into output_folder.
    """

    csv_path = os.path.join(input_folder, csv_filename)
    # Read CSV - expecting single column with comma-separated values: image_name,x1,x2,y1,y2
    try:
        df = pd.read_csv(csv_path, header=None, sep="\t")
    except Exception:
        df = pd.read_csv(csv_path, header=None)

    # Skip the first row (header)
    df = df.iloc[1:].reset_index(drop=True)
    
    for idx, row in df.iterrows():
        # Split the comma-separated values in the first (and only) column
        values = str(row.iloc[0]).split(',')
        if len(values) < 5:
            print(f"Warning: Row {idx} has fewer than 5 comma-separated values: {values}")
            continue
        
        image_stem = str(values[0]).strip()
        x1, x2, y1, y2 = [float(v.strip()) for v in values[1:5]]
        bbox = (x1, y1, x2, y2)

        # Try both .jpg and .jpeg extensions for images
        for ext in ['jpg', 'jpeg', 'png']:
            candidate_path = os.path.join(input_folder, f"{image_stem}.{ext}")
            if os.path.exists(candidate_path):
                image_path = candidate_path
                print(f"Found image at {image_path}")
                break
        else:
            print(f"Image file for {image_stem} not found in {input_folder}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue

        # The analysis functions require the image in global scope for find_valentine_bbox
        global_img_backup = globals().get("img", None)
        globals()["img"] = img  # For find_valentine_bbox compatibility

        valentine_bbox = find_valentine_bbox(bbox, img, use_average_per_row=use_average_per_row)
        # Draw pink outline if a valentine bbox is found and is at least 20px x 20px
        img_draw = img.copy()
        if valentine_bbox is not None:
            x1b, y1b, x2b, y2b = map(int, map(round, valentine_bbox))
            # Check if bbox is at least 20px x 20px
            width = abs(x2b - x1b)
            height = abs(y2b - y1b)
            if width >= 20 and height >= 20:
                # cv2.rectangle(img, pt1, pt2, color, thickness)
                cv2.rectangle(img_draw, (x1b, y1b), (x2b, y2b), (203, 73, 150), 3)  # pink in BGR

       

        # Clean up global img to avoid side effects
        if global_img_backup is not None:
            globals()["img"] = global_img_backup
        else:
            del globals()["img"]

        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, img_draw)
        print(f"Wrote outlined image to {output_path}")



def is_valentine(bbox, img, use_average_per_row=True):
    x1, y1, x2, y2 = bbox


    height, width, channels = img.shape
    x1c = max(int(round(x1)), 0)
    x2c = min(int(round(x2)), width)
    # Take the vertical band inside x1, x2
    column_band = img[:, x1c:x2c, :]  # shape (H, x2c-x1c, 3)

    # Threshold for "red" -- red much higher than green and blue
    # Note: OpenCV uses BGR format, so channel 2 is red, channel 1 is green, channel 0 is blue
    # Make a mask where red is dominant and over threshold
    # Convert to int to prevent uint8 overflow when subtracting
    red_channel = column_band[..., 2].astype(np.int16)
    green_channel = column_band[..., 1].astype(np.int16)
    blue_channel = column_band[..., 0].astype(np.int16)
    
    if use_average_per_row:
        # Average method: compute mean channel values per row
        mean_red_per_row = red_channel.mean(axis=1)  # shape (H,)
        mean_green_per_row = green_channel.mean(axis=1)  # shape (H,)
        mean_blue_per_row = blue_channel.mean(axis=1)  # shape (H,)
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
        # Combine horizontally (any red pixel in column counts)
        red_per_row = red_mask.any(axis=1)  # shape (H,)


    max_span = 0
    current_span = 0
    for is_red, group in groupby(red_per_row):
        cnt = sum(1 for _ in group)
        if is_red:
            if cnt > max_span:
                max_span = cnt

    # Define "large" as at least 40% of bbox's height
    bbox_height = abs(y2 - y1)
    min_large = int(0.4 * bbox_height)
    return max_span >= min_large


def find_valentine_bbox(bbox, img, use_average_per_row=True):
    """
    Finds a bounding box tightly enclosing the contiguous red band, using the x values of the bbox to determine the column to check.
    Searches the full image height (not restricted to bbox's y range), matching is_valentine's behavior.

    Args:
        bbox: Tuple (x1, y1, x2, y2) of bounding box coordinates
        img: Image array
        use_average_per_row: If True, uses average channel values per row. If False, uses pixel-by-pixel detection.

    Returns a tuple (x1, y1_new, x2, y2_new) where y1_new and y2_new correspond to the red region.
    Returns None if no suitable red area is found.
    """
    x1, y1, x2, y2 = bbox

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

# Optionally, you can update is_valentine to use this for consistency

if __name__ == "__main__":
    process_valentine_images()