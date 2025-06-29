import cv2
import os
import numpy as np
import pandas as pd

# mine
from mp_db_io import DataIO
import re



# Conda minimal_ds or env37 but not base

# I/O utils
io = DataIO()
db = io.db

# iterate through folders? 
IS_CLUSTER = True

# are we making videos or making merged stills?
IS_VIDEO = True
IS_VIDEO_MERGE = True
FRAMERATE = 12
PERIOD = 30 # how many images in each merge cycle
MERGE_COUNT = 12 # largest number of merged images 
START_MERGE = 1 # number of images merged into the first image. Can be 1 (no merges) or >1 (two or more images merged)

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
GIGA_DIMS = 20688
FULLBODY_DIMS = 32000
TEST_DIMS = 4000
REG_DIMS = 3448
if LOWEST_DIMS: 
    GIGA_DIMS = REG_DIMS
    SCALE_IMGS = True
elif FULLBODY:
    GIGA_DIMS = FULLBODY_DIMS
    SCALE_IMGS = False
else:
    SCALE_IMGS = False
VERBOSE = True
# Provide the path to the folder containing the images
ROOT_FOLDER_PATH = '/Volumes/OWC4/images_to_assemble'
# if IS_CLUSTER this should be the folder holding all the cluster folders
# if not, this should be the individual folder holding the images
# will not accept clusterNone -- change to cluster00
FOLDER_NAME = "Topic0_short_vidtest"
FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH,FOLDER_NAME)
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

def merge_images_numpy(image_list):
    """
    Merge multiple cv2 images with equal weighting using pure NumPy operations.
    
    Args:
        image_list: List of cv2 images (already loaded with cv2.imread)
    
    Returns:
        Merged image as a cv2/numpy array
    """
    if not image_list:
        print("No images to merge")
        return None
    elif len(image_list) == 1:
        return image_list[0]
    
    # Get dimensions of the first image
    h, w = image_list[0].shape[:2]
    
    # Ensure all images are the same size and format
    processed_images = []
    for img in image_list:
        # Handle grayscale images
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Resize if needed
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        
        processed_images.append(img.astype(np.float32))
    
    # Stack all images along a new axis
    stacked = np.stack(processed_images, axis=0)
    
    # Take the mean along the stacking axis
    merged_img = np.mean(stacked, axis=0)
    
    # Convert back to uint8
    merged_img = np.clip(merged_img, 0, 255).astype(np.uint8)
    
    return merged_img

def iterate_image_list(FOLDER_PATH,image_files, successes):
    def crop_scale_giga(img1, DIMS=GIGA_DIMS):
        if SCALE_IMGS:
            # this is potentially messy, because it was originally designed to crop images when there were small size differences
            # but now I'm using it to resize gigas during test runs. 
            # Resize the image to GIGA_DIMS
            if VERBOSE: print("resizing image to", DIMS)
            img1 = cv2.resize(img1, (DIMS, DIMS), interpolation=cv2.INTER_AREA)
        else:
            if VERBOSE: print("cropping image to", DIMS)
            # height, width = img1.shape[:3]
            if img1.shape[0] > DIMS or img1.shape[1] > DIMS:
                height, width, _ = img1.shape
                start_row = (height - DIMS) // 2
                start_col = (width - DIMS) // 2
                print("start_row", start_row, "start_col", start_col)
                img1 = img1[start_row:start_row + DIMS, start_col:start_col + DIMS]
        return img1

    def handle_giga_dims(img1):
        # Always resize img1 to GIGA_DIMS
        if img1.shape[0] > TEST_DIMS or img1.shape[1] > TEST_DIMS:
            # if VERBOSE: print("image shape >", img1.shape, image_files[i])
            img1 = crop_scale_giga(img1)
        elif img1.shape[0] == REG_DIMS and img1.shape[1] == REG_DIMS:
            # not change needed
            pass
        elif img1.shape[0] > REG_DIMS or img1.shape[1] > REG_DIMS:
            # if VERBOSE: print("image shape < ", img1.shape, image_files[i])
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
        img1 = handle_giga_dims(img1)
        print("starting round", i, "img1.shape", img1.shape)
        # Check if there is a second image available
        if i + 1 < len(image_files):
            img2 = load_image(image_files[i+1])
            img2 = handle_giga_dims(img2)

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

def merge_images(images_to_build, FOLDER_PATH):
    print("merging images, this many images_to_build", len(images_to_build))
    if len(images_to_build) % 2 != 0:
        print("odd number of images, skipping last image")
    # Get a list of image files in the folder
    image_files = io.get_img_list(FOLDER_PATH)
    # if VERBOSE: print(image_files)
    cluster_no = handpose_no = None
    successes = 0
    if len(image_files) > 1:
        # this is legacy stuff to get the cluster number and handpose number from the folder name
        image_folder = FOLDER_PATH.split("/")[-1]
        if "cluster" in image_folder:
            print("cluster found", image_folder)
            cluster_no = int(image_folder.split("_")[0].replace("cluster",""))
            try: handpose_no = int(image_folder.split("_")[1])
            except: print("handpose_no = None")
        count = len(image_files)
        print("about to iterate_image_list with ", len(images_to_build), "images")
        if len(images_to_build) == 0: 
            print("no images to build")
            return None, 0, cluster_no, handpose_no
        merged_pairs, successes = iterate_image_list(FOLDER_PATH,images_to_build,successes)
        if merged_pairs is not None: 
            print("len merged pairs are", len(merged_pairs))
            # Continue merging until there is only one merged image left
            while len(merged_pairs) >= 2:
                merged_pairs, successes = iterate_image_list(FOLDER_PATH,merged_pairs,successes)

            final_merged = merged_pairs[0]
        else:
            print("no merged pairs found")
            final_merged = None
        return final_merged, successes, cluster_no, handpose_no
    else:
        return None, 0, cluster_no, handpose_no

def save_merge(merged_image, count, cluster_no, handpose_no, FOLDER_PATH):
    if cluster_no is not None:
        savename = 'merged_cluster_' + str(cluster_no)+ "_"+ str(handpose_no)+ "_"+ str(count*2)+'.jpg'
    else:
        savename = 'merged_image'+str(count)+'.jpg'
    output_path = os.path.join(FOLDER_PATH, savename)
    cv2.imwrite(output_path, merged_image)

    print('Merged image/video saved successfully here', output_path)


def get_img_list_subfolders(subfolders):
    all_img_path_list = []
    for subfolder_path in subfolders:
        print("subfolder_path", subfolder_path)
        img_list = io.get_img_list(subfolder_path)
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
        write_video(io.get_img_list(dir_path), FRAMERATE, dir_path)

        


def write_video(img_array, FRAMERATE=15, subfolder_path=None):
    img_array = [img for img in img_array if img.endswith(".jpg")]
    images_to_build = load_images(img_array, subfolder_path)
    # print("img_array", img_array)
    print("len images_to_build", len(images_to_build))
    # Get the dimensions of the first image in the array
    cluster_no, image_path = get_path(subfolder_path, img_array)
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if IS_VIDEO_MERGE: merge_info = f"_p{PERIOD}_st{START_MERGE}_ct{MERGE_COUNT}"
    else: merge_info = ""
    video_path = os.path.join(FOLDER_PATH, FOLDER_NAME.replace("/","_")+cluster_no+merge_info+".mp4")
    print("video_path", video_path)
    video_writer = cv2.VideoWriter(video_path, fourcc, FRAMERATE, (width, height))

    if IS_VIDEO_MERGE:
        # Validate and adjust parameters for edge cases
        total_images = len(img_array)
        
        # Calculate images_per_cycle and check if MERGE_COUNT needs adjustment
        images_per_cycle = PERIOD + (MERGE_COUNT - START_MERGE * 2)
        if images_per_cycle < MERGE_COUNT * 2:
            adjusted_merge_count = images_per_cycle // 2
            print(f"Warning: Adjusting MERGE_COUNT from {MERGE_COUNT} to {adjusted_merge_count} based on PERIOD and START_MERGE")
            merge_count = adjusted_merge_count
        else:
            merge_count = MERGE_COUNT
            
        # Process only if there are enough images to complete at least one cycle
        if total_images >= PERIOD:
            current_pos = 0
            
            while current_pos + PERIOD <= total_images:
                # Phase 1: Increase merge count from START_MERGE to merge_count
                # video_writer.write(images_to_build[current_pos])
                for i in range(START_MERGE, merge_count + 1):
                    # Load and merge images from current_pos to current_pos + i
                # if i % 2 == 0:
                    merged_img = merge_images_numpy(images_to_build[current_pos:current_pos + i])
                    # print("merged_img", merged_img.shape)
                    # print("type merged_img", type(merged_img))
                    if merged_img is not None: video_writer.write(merged_img)
                
                # Phase 2: Slide through the array maintaining merge_count images merged
                for i in range(1, PERIOD - merge_count + 1):
                    # Load and merge images from current_pos + i to current_pos + i + merge_count
                    merged_img = merge_images_numpy(images_to_build[current_pos + i:current_pos + i + merge_count])
                    if merged_img is not None: video_writer.write(merged_img)
                
                # Phase 3: Decrease merge count back to START_MERGE
                # not subtracting 1 from start_merge to not include the final image
                # adding 1 to PERIOD to include the first image of the next cycle
                for i in range(merge_count - 1, START_MERGE , -1):
                    # Load and merge images from PERIOD - i to PERIOD
                    end_idx = min(current_pos + PERIOD + 1, total_images)
                    start_idx = end_idx - i
                    # if i % 2 == 0:
                    merged_img = merge_images_numpy(images_to_build[start_idx:end_idx])
                    if merged_img is not None: video_writer.write(merged_img)
                
                # Move to the next cycle
                current_pos += PERIOD
                print("current_pos", current_pos)
            
            # Handle remaining images if there are not enough for a full cycle
            remaining = total_images - current_pos
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
        # Original behavior - write each frame directly
        for img in images_to_build:
            video_writer.write(img)

    # Release the video writer and close the video file
    video_writer.release()
    print(f"Video saved at: {video_path}")

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
    
    # Load the first image
    if subfolder_path:
        image_path = os.path.join(subfolder_path, image_list[0])
    else:
        image_path = image_list[0]
    
    result = cv2.imread(image_path)
    
    # If only one image, return it
    if len(image_list) == 1:
        return result
    
    # Merge multiple images with equal weighting
    weight_per_image = 1.0 / len(image_list)
    
    # Start with the first image at full weight
    result = result.copy()
    
    images_to_build = []
    # Gradually blend in the other images
    for i in range(1, len(image_list)):
        if subfolder_path:
            image_path = os.path.join(subfolder_path, image_list[i])
        else:
            image_path = image_list[i]
        
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
        print("subfolders", subfolders)
        if IS_VIDEO is True and ALL_ONE_VIDEO is True:
            print("making regular combined video")
            all_img_path_list = get_img_list_subfolders(subfolders)
            write_video(all_img_path_list, FRAMERATE)
            if SAVE_METAS_AUDIO is True:
                print("saving metas")
                save_concatenated_metas(subfolders, OUTPUT, CSV_FILE)
            # # const_videowriter(subfolder_path, FRAMERATE)
            # for subfolder_path in subfolders:
            #     write_video(subfolder_path, FRAMERATE)
        elif IS_VIDEO is True and ALL_ONE_VIDEO is False:
            for subfolder_path in subfolders:
                all_img_path_list = io.get_img_list(subfolder_path)
                # only inlcude jpgs in the list
                write_video(all_img_path_list, FRAMERATE, subfolder_path)

        elif SAVE_METAS_AUDIO is True:
            save_concatenated_metas(subfolders, OUTPUT, CSV_FILE)
        else:
            # for merging images into stills
            for subfolder_path in subfolders:
                # print(subfolder_path)
                all_img_path_list = io.get_img_list(subfolder_path)
                images_to_build = load_images(all_img_path_list, subfolder_path)
                merged_image, count, cluster_no, handpose_no = merge_images(images_to_build, subfolder_path)
                if count == 0:
                    print("no images here")
                    continue
                else:
                    save_merge(merged_image, count, cluster_no, handpose_no, FOLDER_PATH)
    else:
        print("going to get folder ls", FOLDER_PATH)
        all_img_path_list = io.get_img_list(FOLDER_PATH)
        if IS_VIDEO is True:
            if DO_RATIOS: crop_images(all_img_path_list, FOLDER_PATH)
            write_video(all_img_path_list, FRAMERATE, FOLDER_PATH)
        else:
            print("going to merge images to make still")
            images_to_build = load_images(all_img_path_list, FOLDER_PATH)
            print("len images_to_build", len(images_to_build))
            merged_image, count, cluster_no, handpose_no = merge_images(images_to_build, FOLDER_PATH)
            save_merge(merged_image, count, cluster_no, handpose_no, FOLDER_PATH)
     








if __name__ == '__main__':
    main()










