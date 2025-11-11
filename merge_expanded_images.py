import math
import cv2
import os
import numpy as np
import pandas as pd

# mine
from mp_db_io import DataIO
import re
from pymediainfo import MediaInfo



# conda activate minimal_ds -- requires mps_torch310 or minimal_ds but not base. minimal_ds for video

# I/O utils
io = DataIO()
db = io.db

MODES = ["merge_images_paris_photo", "merge_images_body_autocrop", "make_video", "make_video_heft_keyword_fusion"]
MODE_CHOICE = 3
CURRENT_MODE = MODES[MODE_CHOICE]


# iterate through folders? 
IS_CLUSTER = True

LOOPING = False # defaults
last_image_written = None
run_counter = 0

MERGE_COUNT = START_MERGE = PERIOD = FRAMERATE = ALREADY_CROPPED = None
IS_VIDEO = IS_VIDEO_MERGE = SMOOTH_MERGE = False
if "merge_images" in CURRENT_MODE:
# are we making videos or making merged stills?
    IS_VIDEO = False

    # handle the autocropped bodies
    if "autocrop" in CURRENT_MODE:
        ALREADY_CROPPED = True # images are already cropped to ratio for each subfolder, and need to be scaled
    else:
        ALREADY_CROPPED = False # images are not cropped to ratio, need to be cropped and scaled

elif "make_video" in CURRENT_MODE:
    IS_VIDEO = True

    # control default merging behavior
    FRAMERATE = 12
    PERIOD = 30 # how many images in each merge cycle
    MERGE_COUNT = 12 # largest number of merged images 
    START_MERGE = 1 # number of images merged into the first image. Can be 1 (no merges) or >1 (two or more images merged)

    if "keyword_fusion" in CURRENT_MODE:
        IS_VIDEO_MERGE = True
        OSCILATING_MERGE = True # if true, will do an oscillating merge from START_MERGE up to MERGE_COUNT and back down to START_MERGE 
        SMOOTH_MERGE = True # if true, will do a smooth merge from MERGE_COUNT down to START_MERGE
        if SMOOTH_MERGE: FRAMERATE = 30
        LOOPING = True
        PERIOD = 30 # how many images in each merge cycle
        MERGE_COUNT = 10 # largest number of merged images 
        START_MERGE = 1 # number of images merged into the first image. Can be 1 (no merges) or >1 (two or more images merged)

# import moviepy only if making videos
if IS_VIDEO:
    # need conda activate minimal_ds
    from moviepy import *
    from moviepy import VideoFileClip, AudioFileClip

SAVE_METAS_AUDIO = False
BUILD_WITH_AUDIO = False
ALL_ONE_VIDEO = False
LOWEST_DIMS = True # make this False if assembling big images eg full body # False if doing Paris Photo faces
FULLBODY = False # this is for full body images, will change GIGA_DIMS to FULLBODY_DIMS
SORT_ORDER = "Chronological"
DO_RATIOS = False
GIGA_DIMS = [20688,20648]
FULLBODY_DIMS = [32000,32000]
TEST_DIMS = [4000,4000] 
REG_DIMS = [3448,3448]
VID_DIMS_HEFTTEST = [1080,1080]
SKIP_PREFIX = "_x"
FORCE_LS = True
if LOWEST_DIMS: 
    if "heft" in CURRENT_MODE:
        GIGA_DIMS = VID_DIMS_HEFTTEST
    else:
        GIGA_DIMS = REG_DIMS
    SCALE_IMGS = True
elif FULLBODY:
    GIGA_DIMS = FULLBODY_DIMS
    SCALE_IMGS = False
elif ALREADY_CROPPED:
    SCALE_IMGS = True
else:
    SCALE_IMGS = False
VERBOSE = True
None_counter = 0
# Provide the path to the folder containing the images
ROOT_FOLDER_PATH = '/Volumes/OWC4/images_to_assemble'
# ROOT_FOLDER_PATH = '/Users/michaelmandiberg/Documents/projects-active/facemap_production'
# if IS_CLUSTER this should be the folder holding all the cluster folders
# if not, this should be the individual folder holding the images
# will not accept clusterNone -- change to cluster00
FOLDER_NAME = "body3D_512_"
FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH,FOLDER_NAME)
# FOLDER_PATH = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/keyword_body3D_tests/body3D_512" # temp override for testing
FOLDER_PATH = '/Volumes/OWC4/segment_images/renderfolder'
DIRS = ["1x1", "4x3", "16x10"]
OUTPUT = os.path.join(io.ROOTSSD, "audioproduction")
# Extract the topic number from the folder name
match = re.search(r'topic(\d+)', FOLDER_NAME)
if match:
    TOPIC = int(match.group(1))
    print("TOPIC", TOPIC)
else:
    TOPIC = None
    print("TOPIC", TOPIC)
CSV_FILE = f"metas_{TOPIC}.csv"

def merge_images_numpy(image_list, make_first_image=False):
    global last_image_written
    """
    Merge multiple cv2 images with equal weighting using pure NumPy operations.
    
    Args:
        image_list: List of cv2 images (already loaded with cv2.imread)
    
    Returns:
        Merged image as a cv2/numpy array
    """
    # print("merge_images_numpy merging", len(image_list), "images with numpy")
    if not image_list:
        print("No images to merge")
        return None
    elif len(image_list) == 1 and last_image_written is None:
        print("only one image, returning it directly AS LIST, shape is", image_list[0].shape)
        return [image_list[0]]
    
    if len(image_list[0].shape) == 2:
        print("removed item 0 from the list - i think for non-images?")
        image_list.pop(0)
    # Get dimensions of the first image
    h, w = image_list[0].shape[:2]
    
    # Ensure all images are the same size and format
    processed_images = []
    images_to_return = []
    # print(f"last_image_written before processing: {last_image_written.shape if last_image_written is not None else 'None'}")
    for img in image_list:
        # Handle grayscale images
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Resize if needed
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        processed_images.append(img.astype(np.float32))
    
    # old route, two images to one:
    # Stack all images along a new axis
    stacked = np.stack(processed_images, axis=0)
    
    # Take the mean along the stacking axis
    merged_img = np.mean(stacked, axis=0)
        
    # Convert back to uint8
    merged_img = np.clip(merged_img, 0, 255).astype(np.uint8)
    merged_img_float = merged_img.astype(np.float32)

    if make_first_image == True: 
        print("make_first_image is True, returning merged_img_float as first image, shape", merged_img_float.shape)
        last_image_written = merged_img_float
        return last_image_written
    
    this_last_image_written = last_image_written.astype(np.float32) if last_image_written is not None else None

    # print("this_last_image_written shape", this_last_image_written.shape if this_last_image_written is not None else 'None')
    # print("merged_img_float shape", merged_img_float.shape)
    if SMOOTH_MERGE:
        # print("doing smooth merge with this many images", len(processed_images))
        transition_image1 = cv2.addWeighted(this_last_image_written, 0.67, merged_img_float, 0.33, 0)
        transition_image2 = cv2.addWeighted(this_last_image_written, 0.33, merged_img_float, 0.67, 0)
        # Convert to uint8 for output
        transition_image1 = np.clip(transition_image1, 0, 255).astype(np.uint8)
        transition_image2 = np.clip(transition_image2, 0, 255).astype(np.uint8)
        merged_img_uint8 = np.clip(merged_img, 0, 255).astype(np.uint8)
        images_to_return = [transition_image1, transition_image2, merged_img_uint8]
    else:
        merged_img_uint8 = np.clip(merged_img, 0, 255).astype(np.uint8)
        images_to_return.append(merged_img_uint8)
    return images_to_return

def get_median_image_dimensions(all_img_path_list, subfolder_path=None):
    # all_img_path_list is a list of paths to images
    # use pymediainfo to get median dimensions withouth opening all the files
    # just check the metadata
    heights = []
    widths = []
    counter = 0
    for image_path in all_img_path_list:
        # print("checking image for dimensions", image_path)
        if subfolder_path:
            image_path = os.path.join(subfolder_path, image_path)
        metadata = MediaInfo.parse(image_path)
        if metadata is None or metadata.tracks is None: continue
        for track in metadata.tracks:   
            if track.track_type == 'Image':
                # if VERBOSE: print("track.height, track.width", track.height, track.width)
                if track.height is not None and track.width is not None:
                    heights.append(track.height)
                    widths.append(track.width)
        counter += 1
        if counter > 20: break
    if len(heights) == 0 or len(widths) == 0:
        print("no heights or widths found, need to skip this one")
        return None
    median_height = np.median(heights)
    median_width = np.median(widths)
    return int(median_height), int(median_width)

def crop_scale_giga(img1, DIMS=GIGA_DIMS):
    if SCALE_IMGS:
        # this is potentially messy, because it was originally designed to crop images when there were small size differences
        # but now I'm using it to resize gigas during test runs. 
        # Resize the image to GIGA_DIMS
        if abs(img1.shape[0] - img1.shape[1]) > img1.shape[0] // 10:
            # if the image is not square, establish a DIMS ratio based on the larger dimension
            if img1.shape[0] > img1.shape[1]:
                ratio = DIMS[0] / img1.shape[0]
                resize_width = int(img1.shape[1] * ratio)
                resize_height = DIMS[0]
            else:
                ratio = DIMS[1] / img1.shape[1]
                resize_height = int(img1.shape[0] * ratio)
                resize_width = DIMS[1]
        else:
            resize_width = DIMS[1]
            resize_height = DIMS[0]
        if VERBOSE: print("resizing image to", resize_width, resize_height)
        img1 = cv2.resize(img1, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        if VERBOSE: print("img1.shape after resize", img1.shape)
    else:
        if VERBOSE: print("cropping image to", DIMS)
        # height, width = img1.shape[:3]
        if img1.shape[0] > DIMS[0] or img1.shape[1] > DIMS[1]:
            height, width, _ = img1.shape
            start_row = (height - DIMS[0]) // 2
            start_col = (width - DIMS[1]) // 2
            print("start_row", start_row, "start_col", start_col)
            img1 = img1[start_row:start_row + DIMS[0], start_col:start_col + DIMS[1]]
    return img1

def iterate_image_list(FOLDER_PATH,image_files, successes, output_dims=None):

    def handle_giga_dims(img1, output_dims=None):
        # Always resize img1 to GIGA_DIMS
        if output_dims is not None:
            print("resizing to output_dims", output_dims)
            output_height, output_width = output_dims
            # need to flip these for cv2
            img1 = cv2.resize(img1, (output_width, output_height), interpolation=cv2.INTER_AREA)
            print("img1.shape after resize to output_dims", img1.shape)
        elif img1.shape[0] > TEST_DIMS[0] or img1.shape[1] > TEST_DIMS[1]:
            # if VERBOSE: print("image shape >", img1.shape, image_files[i])
            img1 = crop_scale_giga(img1)
        elif img1.shape[0] == REG_DIMS[0] and img1.shape[1] == REG_DIMS[1]:
            # not change needed
            pass
        elif img1.shape[0] > REG_DIMS[0] or img1.shape[1] > REG_DIMS[1]:
            if VERBOSE: print(f"image shape {img1.shape} > {REG_DIMS}, {image_files[i]}")
            img1 = crop_scale_giga(img1, REG_DIMS)
        return img1
    
    def load_image(img_file_or_path):
        if type(img_file_or_path) is np.ndarray:
            img1 = img_file_or_path
        else:
            print("loading", img_file_or_path)
            img1 = cv2.imread(os.path.join(FOLDER_PATH, img_file_or_path))
        return img1
    
    # Initialize the merged pairs list with the images in pairs
    merged_pairs = []
    if type(image_files[0]) is np.ndarray:
        loaded = True
    else:
        loaded = False
    # print("loaded", loaded)
    # Iterate through the image files and merge them in pairs
    for i in range(0, len(image_files), 2):
        img1 = load_image(image_files[i])
        img1 = handle_giga_dims(img1, output_dims)
        print("starting round", i, "img1.shape", img1.shape)
        # Check if there is a second image available
        if i + 1 < len(image_files):
            img2 = load_image(image_files[i+1])
            img2 = handle_giga_dims(img2, output_dims)

            if len(img1.shape) == 2:  # img1 is grayscale
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            if len(img2.shape) == 2:  # img2 is grayscale
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            print(i, img1.shape, img2.shape)

            # Merge the pair of images 50/50
            try:
                blend = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)
                merged_pairs.append(blend)
                successes += 1
            except:
                print("failed to merge", image_files[i], image_files[i+1])
        else:
            if IS_VIDEO and IS_VIDEO_MERGE:
                merged_pairs.append(img1)
                successes += 1
            else:
                print("skipping image key number", str(i))
            # Only one image left, add it to the merged pairs list directly
            # merged_pairs.append(img1)
    print("len(merged_pairs) after iterate_image_list", len(merged_pairs))
    # quit()
    return merged_pairs, successes

def merge_images(images_to_build, FOLDER_PATH, output_dims=None):
    global None_counter
    print("merging images, this many images_to_build", len(images_to_build))
    if len(images_to_build) % 2 != 0:
        print("odd number of images, skipping last image")
    # Get a list of image files in the folder
    image_files = io.get_img_list(FOLDER_PATH, FORCE_LS)
    # if VERBOSE: print(image_files)
    cluster_no = handpose_no = hsv_no = topic_no = None
    successes = 0
    if len(image_files) > 1:
        # this is legacy stuff to get the cluster number and handpose number from the folder name
        image_folder = FOLDER_PATH.split("/")[-1]
        if "cluster" in image_folder:
            filters = image_folder.split("_")
            print(f"filters: {filters}")
            for f in filters:
                if f.startswith("cluster"):
                    try:
                        cluster_no = int(f.replace("cluster",""))
                    except:
                        try:
                            cluster_no = int(f.replace("clustercc",""))
                        except:
                            print("could not parse cluster number from", f)
                            cluster_no = None_counter
                            None_counter += 1
                if f.startswith("p"):
                    try:
                        handpose_no = int(f.replace("p",""))
                    except:
                        print("could not parse handpose number from", f)
                        handpose_no = None
                if f.startswith("h"):
                        try:
                            hsv_no = (f.replace("h",""))
                        except:
                            print("could not parse hue number from", f)
                            hsv_no = None
                if f.startswith("t"):
                    try:
                        topic_no = int(f.replace("t",""))
                    except:
                        print("could not parse topic number from", f)
                        topic_no = None
            print("FOUND cluster_no", cluster_no, "handpose_no", handpose_no, "hsv_no", hsv_no, "topic_no", topic_no)
                
            # if "None" in image_folder:
            #     cluster_no = None_counter
            #     None_counter += 1
            #     print("cluster_no is None, incrementing None_counter to", None_counter)
            # else:
            #     print("cluster found", image_folder)
            #     cluster_no = int(image_folder.split("_")[0].replace("cluster",""))
            #     try: handpose_no = int(image_folder.split("_")[1])
            #     except: print("handpose_no = None")
        count = len(image_files)
        print("about to iterate_image_list with ", len(images_to_build), "images")
        if len(images_to_build) == 0: 
            print("no images to build")
            return None, 0, cluster_no, handpose_no, hsv_no
        merged_pairs, successes = iterate_image_list(FOLDER_PATH,images_to_build,successes, output_dims)
        if merged_pairs is not None and len(merged_pairs) > 0: 
            print("len merged pairs are", len(merged_pairs))
            # Continue merging until there is only one merged image left
            while len(merged_pairs) >= 2:
                merged_pairs, successes = iterate_image_list(FOLDER_PATH,merged_pairs,successes, output_dims)

            final_merged = merged_pairs[0]
        else:
            print("no merged pairs found")
            final_merged = None
        return final_merged, successes, cluster_no, handpose_no, hsv_no, topic_no
    else:
        return None, 0, cluster_no, handpose_no, hsv_no, topic_no

def save_merge(merged_image, count, cluster_no, handpose_no, hsv_no, topic_no, FOLDER_PATH):
    if cluster_no is not None:
        savename = 'merged_cluster_' + str(cluster_no)+ "_"+ str(handpose_no)+ "_"+ str(hsv_no)+ "_"+ str(topic_no)+ "_"+ str(count*2)+'.jpg'
    else:
        savename = 'merged_image'+str(count)+'.jpg'
    output_path = os.path.join(FOLDER_PATH, savename)
    cv2.imwrite(output_path, merged_image)

    print('Merged image/video saved successfully here', output_path)


def get_img_list_subfolders(subfolders):
    all_img_path_list = []
    for subfolder_path in subfolders:
        print("subfolder_path", subfolder_path)
        img_list = io.get_img_list(subfolder_path, FORCE_LS)
        # if any file ends with .json, remove it
        img_list = [img for img in img_list if not img.endswith(".json")]
        img_list.sort()
        img_path_list = []
        for img in img_list:
            img_path = os.path.join (subfolder_path, img)
            img_path_list.append(img_path)
        # print(img_path_list)
        all_img_path_list += img_path_list
    # print(all_img_path_list)
    return all_img_path_list

def get_path(subfolder_path, img_array):
    if subfolder_path:
        cluster_no = subfolder_path.split("/")[-1]
        image_path = os.path.join(subfolder_path, img_array[0])
    else:
        cluster_no = img_array[0].replace(FOLDER_PATH,"").split("/")[0]
        image_path = img_array[0]
    return cluster_no, image_path

def crop_images(img_array, subfolder_path=None):
    cluster_no, image_path = get_path(subfolder_path, img_array)
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    cropfset = 70
    if height >= width:
        crop_dict = {
            "1x1": [0,width, cropfset,width+cropfset],
            "4x3": [0,width, int(cropfset/2),int(width*4/3+cropfset/2)],
            "4x2": [0,width, 0,height],
            "16x10": [0,int(height*10/16) , 0,height]
        }
    elif width > height:
        # this offset isn't configured right but without hsomething the code doesn't work 
        crop_dict = {
            "1x1": [0, height, cropfset, height + cropfset],
            "4x3": [0, height, int(cropfset / 2), int(height * 4 / 3 + cropfset / 2)],
            "4x2": [0, height, 0, width],
            "16x10": [0, int(width * 10 / 16), 0, width]
        }
    # elif height == width:
    #     crop_dict = []


    for dir in DIRS:
        if subfolder_path:
            dir_path = os.path.join(subfolder_path, dir)
        else:
            dir_path = os.path.join(FOLDER_PATH, dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for filename in img_array:
            if subfolder_path:
                image_path = os.path.join(subfolder_path, filename)
            else:
                image_path = filename
            print(image_path)
            img = cv2.imread(image_path)
            # check if img is not None
            if img is not None:
                crop_img = img[crop_dict[dir][2]:crop_dict[dir][3], crop_dict[dir][0]:crop_dict[dir][1]]
                # crop_img = img[0:crop_dict[dir][1], 0:crop_dict[dir][0]]
                cv2.imwrite(os.path.join(dir_path, filename), crop_img)
            else:
                print("no image found at", image_path)
        write_video(io.get_img_list(dir_path), dir_path)

def shift_video_frames(video_path):
    # I want to start the video at the MERGE_COUNT frame, so that it starts with the fully merged image
    # and add the removed frames to the end of the video
    print("reordering video frames to start at MERGE_COUNT frame")
    # get dimensions of video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap.release()
    temp_video_path = video_path.replace(".mp4", "_temp.mp4")
    temp_video_writer = cv2.VideoWriter(temp_video_path, fourcc, FRAMERATE, (width, height))
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Write frames starting from MERGE_COUNT to end
    for i in range(MERGE_COUNT - 1, len(frames)):
        temp_video_writer.write(frames[i])
    # Then write frames from start to MERGE_COUNT - 1
    for i in range(0, MERGE_COUNT - 1):
        temp_video_writer.write(frames[i])
    
    temp_video_writer.release()
    
    # Replace original video with reordered video
    os.remove(video_path)
    os.rename(temp_video_path, video_path)
    print("reordered video saved")

def construct_incrementor(merge_count, current_pos, this_period, total_images):
    '''
    the leading image speeds through the remaining merge_count images in half the time
    the trailing image super-speeds through the images to get to the last this_second_merge_count images
    once it gets to there, it will do normal increments to the end
    '''
    print(f"constructing incrementor from merge_count, current_pos, this_period, total_images")
    print(f"{merge_count}, {current_pos}, {this_period}, {total_images}")
    this_first_merge_count = math.floor(merge_count / 2)
    this_second_merge_count = merge_count - this_first_merge_count
    print("this_first_merge_count", this_first_merge_count, "this_second_merge_count", this_second_merge_count)
    leading_incrementor = []
    trailing_incrementor = []
    end_image = min(current_pos + this_period + 1, total_images)
    current_image_no = current_pos+(merge_count*2)
    trailing_start_image = current_pos + merge_count + 1
    print("trailing_start_image", trailing_start_image, "current_image_no", current_image_no, "end_image", end_image)

    # construct leading incrementor to cover the full distance in 1/2 of the merge_count (this_first_merge_count)
    leading_increment = (end_image - current_image_no) // this_first_merge_count # might need to be merge_count -1 TBD
    leading_remainder = (end_image - current_image_no) - (leading_increment * this_first_merge_count)
    print("leading_increment", leading_increment, "leading_remainder", leading_remainder)
    # first append remainder
    leading_incrementor.append(leading_increment+leading_remainder)
    for i in range(this_first_merge_count-1):
        leading_incrementor.append(leading_increment)
    print(f"len {len(leading_incrementor)} leading_incrementor:", this_first_merge_count)

    # construct trailing incrementor to cover 
    images_to_speed_through = (end_image-trailing_start_image) - this_first_merge_count
    print("images_to_speed_through", images_to_speed_through)
    trailing_increment_speedy = images_to_speed_through // this_second_merge_count
    trailing_remainder = images_to_speed_through - (trailing_increment_speedy * this_second_merge_count)
    print("trailing_increment_speedy", trailing_increment_speedy, "trailing_remainder", trailing_remainder)
    # first append remainder
    trailing_incrementor.append(trailing_increment_speedy + trailing_remainder)
    for i in range(this_second_merge_count - 1):
        trailing_incrementor.append(trailing_increment_speedy)
    print(f"len {len(trailing_incrementor)} trailing_incrementor:", this_second_merge_count)
    for i in range(end_image - (trailing_start_image + sum(trailing_incrementor))):
        trailing_incrementor.append(1)
    print(f"final len {len(trailing_incrementor)} trailing_incrementor:", len(trailing_incrementor))

    return leading_incrementor, trailing_incrementor, trailing_start_image, current_image_no

     

def save_images_to_video(images_to_return, video_writer):
    global last_image_written
    global run_counter
    # print("save_images_to_video called with", len(images_to_return) if images_to_return is not None else 0, "images")
    if images_to_return is not None: 
        for img in images_to_return:
            run_counter += 1
            # save this image for testing
            # cv2.imwrite(os.path.join(FOLDER_PATH, f"test_{run_counter}.png"), img)
            # print(f"save_images_to_video test_{run_counter}.png", img.shape)
            # print(img.shape, img.dtype, img.flags['C_CONTIGUOUS'])
            # print("video_writer.isOpened:", video_writer.isOpened())
            video_writer.write(img)
        last_image_written = images_to_return[-1]

def process_images(images_to_build, video_writer, total_images, current_pos=0):
    global last_image_written

    last_ten_images = images_to_build[-10:]
    first_ten_images = images_to_build[:10]

    padded_images_to_build = last_ten_images + images_to_build + first_ten_images

    # no periods for this one. Just merge merge_count images at a time, sliding through the array
    for i in range(current_pos, total_images + MERGE_COUNT + 1):

        make_first_image = True
        start_idx = i
        end_idx = i + MERGE_COUNT
        if last_image_written is None:
            print(" ~~~~ making first last_image_written for i", i, "start_idx", start_idx, "end_idx", end_idx)
            last_image_written = merge_images_numpy(padded_images_to_build[start_idx:end_idx], make_first_image=make_first_image)
        # print("type images_to_return, last_image_written", type(last_image_written))
        # Load and merge images from current_pos to current_pos + MERGE_COUNT
        images_to_return = merge_images_numpy(padded_images_to_build[start_idx:end_idx])
        print(f"have images for i {i}, start_idx {start_idx}, end_idx {end_idx}, len images_to_build {len(images_to_build)}")
        if i <= MERGE_COUNT: continue
        if i + MERGE_COUNT*2 >= len(padded_images_to_build): continue
        save_images_to_video(images_to_return, video_writer)
        # Print the current number of frames written so far
        # print(f"{video_writer.get(cv2.CAP_PROP_POS_FRAMES)} frames written so far")
        print(f"saved merged img current_pos+i", start_idx, "len images_to_build", len(images_to_build), MERGE_COUNT)

def process_images_osc(images_to_build, video_writer, total_images, period, current_pos=0, merge_count=MERGE_COUNT):
    global last_image_written
    # take the total remaining, and make one full cycle with it
    # it should start with START_MERGE, go up to merge_count, then slide through, then go back down to START_MERGE
    # if it can't reach merge_count, it should just go up to the max it can
    # and it should not add the last image, since it's a duplicate of the first image

    if total_images - current_pos < period:
        this_period = total_images - current_pos
        last_cycle = True
        print(f"adjusting this_period to {this_period} because {total_images} - {current_pos} < {period}")
    else:
        this_period = period
        last_cycle = False
    # Phase 1: Increase merge count from START_MERGE to merge_count
    # video_writer.write(images_to_build[current_pos])
    for i in range(START_MERGE, merge_count + 1):
        # Load and merge images from current_pos to current_pos + i
    # if i % 2 == 0:
        images_to_return = merge_images_numpy(images_to_build[current_pos:current_pos+i])
        # print("type merged_img", type(merged_img))
        save_images_to_video(images_to_return, video_writer)
        # Print the current number of frames written so far
        # print(f"{video_writer.get(cv2.CAP_PROP_POS_FRAMES)} frames written so far")
        print(f"merged starting img current_pos+i", current_pos+i, "len images_to_build", len(images_to_build), merge_count)

    print("finished phase 1, does i carry over?", i)
    # if the remaining images are greater than merge_count, then write the middle even section
    if total_images - (current_pos) > merge_count:
        print(f"going to write the middle even section because {total_images} - ({current_pos} + {i}) > {merge_count}")
        # Phase 2: Slide through the array maintaining merge_count images merged
        # this_period - merge_count allows for a flexible number of middle images
        # for heft, it this_period - merge_count == merge_count*2
        build_even_merge(images_to_build, video_writer, current_pos, merge_count, this_period)

    else:
        print(f"skipping the middle even section because {total_images} - ({current_pos} + {i}) <= {merge_count}")
    
    # Phase 3: Decrease merge count back to START_MERGE
    # not subtracting 1 from start_merge to not include the final image
    # adding 1 to PERIOD to include the first image of the next cycle

    leading_incrementor, trailing_incrementor, trailing_start_image, current_image_no = construct_incrementor(merge_count, current_pos, this_period, total_images)
    print("leading_incrementor", leading_incrementor)
    print("trailing_incrementor", trailing_incrementor)

    trailing_inc = 0
    leading_inc = 0
    for i, t_inc, in enumerate(trailing_incrementor):
        trailing_inc += t_inc
        if i <= len(leading_incrementor)-1:
            leading_inc += leading_incrementor[i]
        print(f"trailing i {i}, trailing_inc {trailing_inc}, leading_inc {leading_inc}")
        start_idx = trailing_start_image + trailing_inc
        end_idx = current_image_no + leading_inc
        print("start_idx", start_idx, "end_idx", end_idx, "i", i, "len images_to_build", len(images_to_build), merge_count)
        if end_idx == total_images and start_idx >= end_idx:
            print("skipping this iteration because start_idx >= end_idx")
            # continue
        elif start_idx >= end_idx-1:
            print("IDK why but it seems that non-final ones have an extra frame, so skipping")
            continue
        images_to_return = merge_images_numpy(images_to_build[start_idx:end_idx])
        print(i, "{current_pos+i}", current_pos+i, "start_idx", start_idx, "end_idx", end_idx, "len images_to_build", len(images_to_build), merge_count)
        print(f"last_cycle is {last_cycle}, i is {i}")
        # if last_cycle is not True :
        if last_cycle is True and i >= 3:
            print("skipping last image to avoid duplicate of first image")
        elif last_cycle is False and i == 1:
            print("skipping last image to avoid duplicate of first image")
        else:
            print("writing descending image")
            # if it isn't the last cycle, do the regular write
            save_images_to_video(images_to_return, video_writer)

def build_even_merge(images_to_build, video_writer, current_pos, merge_count, this_period):
    for i in range(MERGE_COUNT + 1, this_period - MERGE_COUNT + 1):
            # Load and merge images from current_pos + i to current_pos + i + MERGE_COUNT
            # start counting at the second image in the set, because I want to only add on one new image at a time
            # as I enter the middle section
            # (START_MERGE-1)+(i - MERGE_COUNT):(START_MERGE-1)+(i)
            # current_pos + i:current_pos + i + MERGE_COUNT
        images_to_return = merge_images_numpy(images_to_build[(current_pos)+(i - MERGE_COUNT):(current_pos)+(i)])
        save_images_to_video(images_to_return, video_writer)
        print("merged middle img {current_pos+i}", current_pos+i, "len images_to_build", len(images_to_build), MERGE_COUNT)
    return
            # if merged_img is not None: video_writer.write(merged_img)

    

def calculate_period(images_to_build):
    image_count = len(images_to_build)-1 # subtract one for the duplicate first image at the end
    clean_reps = image_count // PERIOD
    leftover = image_count - clean_reps * PERIOD
    diff = leftover/clean_reps
    print("image_count", image_count, "clean_reps", clean_reps, "leftover", leftover, "diff", diff)
    if PERIOD - leftover > PERIOD / 2:
        # eg. 25 leftover, shrink period to come out even
        calculated_period = math.floor(PERIOD - diff)
    else:
        # eg. 15 leftover, expand period to come out even
        calculated_period = math.ceil(PERIOD + diff)
    return calculated_period

def write_video(img_array, subfolder_path=None):
    global last_image_written
    last_image_written = None
    img_array = [img for img in img_array if img.endswith(".jpg")]
    print("len img_array before cropping", len(img_array))
    if len(img_array) == 0:
        print("no jpg images found, skipping this folder")
        return
    if IS_VIDEO_MERGE: img_array.append(img_array[0]) # add the first image to the end to make a loop
    images_to_build = load_images(img_array, subfolder_path)
    # print("img_array", img_array)
    print("len images_to_build", len(images_to_build))
    if LOOPING:
        period = calculate_period(images_to_build)
    else:
        period = PERIOD

    if LOWEST_DIMS:
        # run crop_scale_giga(img1) on all images to ensure they are the same dimensions
        for i in range(len(images_to_build)):
            images_to_build[i] = crop_scale_giga(images_to_build[i])

    print("using period of", period)
    # Get the dimensions of the first image in the array
    cluster_no, image_path = get_path(subfolder_path, img_array)
    # img = cv2.imread(image_path)
    height, width, _ = images_to_build[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if IS_VIDEO_MERGE: merge_info = f"_p{period}_st{START_MERGE}_ct{MERGE_COUNT}"
    else: merge_info = ""
    video_path = os.path.join(FOLDER_PATH, FOLDER_NAME.replace("/","_")+cluster_no+merge_info+".mp4")
    print("video_path", video_path)
    video_writer = cv2.VideoWriter(video_path, fourcc, FRAMERATE, (width, height))


    if IS_VIDEO_MERGE:
        # Validate and adjust parameters for edge cases
        total_images = len(img_array)
        current_pos = 0

        if OSCILATING_MERGE:
            # Calculate images_per_cycle and check if MERGE_COUNT needs adjustment
            # this is when the period is shorter than the merge up and down cycle
            # images_per_cycle = period + (MERGE_COUNT - START_MERGE * 2)
            if period < MERGE_COUNT * 2:
                merge_count = period // 2
                print(f"Warning: Adjusting MERGE_COUNT from {MERGE_COUNT} to {merge_count} based on period and START_MERGE")
            else:
                merge_count = MERGE_COUNT
                
            # Process only if there are enough images to complete at least one cycle
            if total_images >= period:
                
                while current_pos + period <= total_images:
                    process_images_osc(images_to_build, video_writer, total_images, period, current_pos, merge_count)
                    # Move to the next cycle
                    current_pos += period
                    print("current_pos", current_pos)

                # Handle remaining images because there are not enough for a full cycle
                # make sure to merge up to the last image, and then discard it, since it's a duplicate of the first image
                remaining = total_images - current_pos
                if "heft_keyword_fusion" in CURRENT_MODE:
                    if remaining > START_MERGE:
                        print("remaining", remaining)
                        if merge_count *2 > remaining:
                            merge_count = math.floor(remaining / 2)
                            print("adjusted merge_count to", merge_count)
                        # remaining_merge_count = math.floor(remaining // 2)
                        # merge_count = min(merge_count, remaining_merge_count)
                        process_images_osc(images_to_build, video_writer, total_images, period, current_pos, merge_count)

                    # coda: make one final merge to ease the final image into to the first image in the loop
                    first_img = images_to_build[0]
                    last_img = images_to_build[-1]
                    print("creating final transition merge between last and first image", first_img.shape, last_img.shape)
                    images_to_return = merge_images_numpy([last_img, first_img])
                    print("images_to_return", len(images_to_return), "images_to_return[0].shape", images_to_return[0].shape)
                    if images_to_return is not None: 
                        # save all but the last image to avoid duplicate of first image
                        save_images_to_video(images_to_return[:-1], video_writer)
                        

                else:
                    print("handling remaining images the old way")
                    if remaining > 0:
                        for i in range(START_MERGE, min(merge_count + 1, remaining + 1)):
                            merged_img = merge_images_numpy(images_to_build[current_pos:current_pos + i])
                            if merged_img is not None: video_writer.write(merged_img)
                        
                        if remaining > merge_count:
                            for i in range(1, remaining - merge_count + 1):
                                merged_img = merge_images_numpy(images_to_build[current_pos + i:current_pos + i + merge_count])
                                if merged_img is not None: video_writer.write(merged_img)
                            
                            for i in range(merge_count - 1, START_MERGE - 1, -1):
                                end_idx = total_images
                                start_idx = end_idx - i
                                if start_idx < end_idx:
                                    merged_img = merge_images_numpy(images_to_build[start_idx:end_idx])
                                    if merged_img is not None: video_writer.write(merged_img)
        else:
            process_images(images_to_build, video_writer, total_images, current_pos)

    else:
        # Original behavior - write each frame directly
        for img in images_to_build:
            video_writer.write(img)


    # Release the video writer and close the video file
    video_writer.release()
    print("outfile size", os.path.getsize(video_path))

    print(f"Video saved at: {video_path}")




    if LOOPING: shift_video_frames(video_path)

    # Add audio to the video if needed
    if BUILD_WITH_AUDIO:
        audio_file = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/audioproduction/multitrack_mixdown_offset_32.71.wav"
        print("Adding audio to video...")
        print("video_path", video_path)
        print("audio_file", audio_file)
        
        # Load video and audio using moviepy
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_file)
        
        # Set the audio clip to both channels (stereo)
        final_clip = video_clip.set_audio(audio_clip)
        
        # Save the final video with audio
        final_video_path = video_path.replace(".mp4", "_with_audio.mp4")
        final_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac")
        
        print(f"Video with audio saved at: {final_video_path}")
        return final_video_path
    
    return video_path


def load_images(image_list, subfolder_path=None):
    """
    Load and merge a list of images using cv2.addWeighted
    
    Args:
        image_list: List of image paths to merge
        subfolder_path: Optional subfolder path containing the images
    
    Returns:
        Merged image
    """
    if len(image_list) == 0:
        raise ValueError("Empty image list provided")

    def construct_image_path(subfolder_path, image_list, index):
        # Load the image at the specified index
        if subfolder_path:
            image_path = os.path.join(subfolder_path, image_list[index])
        else:
            image_path = image_list[index]
        return image_path

    result = cv2.imread(construct_image_path(subfolder_path, image_list, 0))

    # If only one image, return it
    if len(image_list) == 1:
        return [result]
    
    # Merge multiple images with equal weighting
    weight_per_image = 1.0 / len(image_list)
    
    # Start with the first image at full weight
    result = result.copy()
    
    images_to_build = [result]
    # Gradually blend in the other images
    for i in range(1, len(image_list)):

        image_path = construct_image_path(subfolder_path, image_list, i)        
        img = cv2.imread(image_path)        

        # Handle grayscale images
        if len(result.shape) == 2:  # result is grayscale
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        if len(img.shape) == 2:  # img is grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Ensure dimensions match
        if img.shape != result.shape:
            img = cv2.resize(img, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_AREA)
        
        if i % 10 == 0:
            print("i", i, "image_path", image_path)
        # # Calculate weights to maintain proper blending
        # alpha = (len(image_list) - i) / len(image_list)
        # beta = 1.0 / (len(image_list) - i)
        
        # alpha = 1.0 / (i + 1)
        # beta = 1.0 - alpha

        # # Blend current result with new image
        # result = cv2.addWeighted(result, alpha, img, beta, 0.0)

        images_to_build.append(img)

    
    return images_to_build

def save_concatenated_metas(subfolders, output_path, csv_file):
    cat_metas = pd.DataFrame(columns=["image_id", "description", "topic_fit"])
    print(cat_metas)
    for subfolder_path in subfolders:
        metas_path = os.path.join(subfolder_path, "metas.csv")
        if os.path.exists(metas_path):
            # Load the metas into a DataFrame
            metas_df = pd.read_csv(metas_path)
            # Assign columns=["image_id", "description", "topic_fit"] to metas_df
            metas_df.columns = ["image_id", "description", "topic_fit"]

            print(len(metas_df))
            # Append metas_df to cat_metas
            cat_metas = pd.concat([cat_metas, metas_df], ignore_index=True)
    # Save cat_metas to CSV
    output_csv_path = os.path.join(output_path, csv_file)
    print("Output path for CSV:", output_csv_path)
    cat_metas.to_csv(output_csv_path, index=False)


def main():
    print("starting merge_expanded_images.py")
    if IS_CLUSTER is True:
        subfolders = io.get_folders(FOLDER_PATH, SORT_ORDER)
        # skip any folder with SKIP_PREFIX in it
        subfolders = [subfolder for subfolder in subfolders if SKIP_PREFIX not in subfolder]
        print("subfolders", subfolders)
        if IS_VIDEO is True and ALL_ONE_VIDEO is True:
            print("making regular combined video")
            all_img_path_list = get_img_list_subfolders(subfolders)
            write_video(all_img_path_list)
            if SAVE_METAS_AUDIO is True:
                print("saving metas")
                save_concatenated_metas(subfolders, OUTPUT, CSV_FILE)
            # # const_videowriter(subfolder_path)
            # for subfolder_path in subfolders:
            #     write_video(subfolder_path)
        elif IS_VIDEO is True and ALL_ONE_VIDEO is False:
            for subfolder_path in subfolders:
                all_img_path_list = io.get_img_list(subfolder_path, FORCE_LS)
                # only inlcude jpgs in the list
                write_video(all_img_path_list, subfolder_path)

        elif SAVE_METAS_AUDIO is True:
            save_concatenated_metas(subfolders, OUTPUT, CSV_FILE)
        else:
            # for merging images into stills
            for subfolder_path in subfolders:
                # print("subfolder_path", subfolder_path)
                all_img_path_list = io.get_img_list(subfolder_path, FORCE_LS)
                output_dims = get_median_image_dimensions(all_img_path_list, subfolder_path)
                if output_dims is not None:
                    print(" ", output_dims)
                else:
                    print("no output_dims found, skipping this folder", subfolder_path)
                    continue
                images_to_build = load_images(all_img_path_list, subfolder_path)
                merged_image, count, cluster_no, handpose_no, hsv_no, topic_no = merge_images(images_to_build, subfolder_path, output_dims)
                if count == 0:
                    print("no images here")
                    continue
                else:
                    save_merge(merged_image, count, cluster_no, handpose_no, hsv_no, topic_no, FOLDER_PATH)
    else:
        print("going to get folder ls", FOLDER_PATH)
        all_img_path_list = io.get_img_list(FOLDER_PATH)
        if IS_VIDEO is True:
            if DO_RATIOS: crop_images(all_img_path_list, FOLDER_PATH)
            write_video(all_img_path_list, FOLDER_PATH)
        else:
            print("going to merge images to make still")
            images_to_build = load_images(all_img_path_list, FOLDER_PATH)
            print("len images_to_build", len(images_to_build))
            merged_image, count, cluster_no, handpose_no, hsv_no, topic_no = merge_images(images_to_build, FOLDER_PATH)
            save_merge(merged_image, count, cluster_no, handpose_no, hsv_no, topic_no, FOLDER_PATH)





if __name__ == '__main__':
    main()










