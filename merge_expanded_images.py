import cv2
import os
import numpy as np

# mine
from mp_db_io import DataIO



# Conda minimal_ds or env37 but not base

# I/O utils
io = DataIO()
db = io.db

# iterate through folders? 
IS_CLUSTER = False

# are we making videos or making merged stills?
IS_VIDEO = True
ALL_ONE_VIDEO = False

# MERGE
# Provide the path to the folder containing the images
ROOT_FOLDER_PATH = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/segment_images/'
# if IS_CLUSTER this should be the folder holding all the cluster folders
# if not, this should be the individual folder holding the images
# will not accept clusterNone -- change to cluster00
FOLDER_NAME ="June21_topic7"
FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH,FOLDER_NAME)
DIRS = ["1x1", "4x3", "16x10"]

#temporary
# FOLDER_PATH = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/excite_small_portrait_jumpshot15after_3D_wgt1_max1.4_delta.5"

# WRITE VIDEO
# img_array = ['image1.jpg', 'image2.jpg', 'image3.jpg']
# HOLDER = '/Users/michaelmandiberg/Dropbox/facemap_dropbox/June_tests/'
FRAMERATE = 12
# FOLDER = "June4_smilescream_itter_25Ksegment"
# ROOT = os.path.join(HOLDER,FOLDER)
# list_of_files= io.get_img_list(ROOT)
# print(list_of_files)
# list_of_files.sort()


def iterate_image_list(FOLDER_PATH,image_files, successes):
    # Initialize the merged pairs list with the images in pairs
    merged_pairs = []
    if type(image_files[0]) is np.ndarray:
        loaded = True
    else:
        loaded = False
    print(loaded)
    # Iterate through the image files and merge them in pairs
    for i in range(0, len(image_files), 2):
        # print(FOLDER_PATH, image_files[i])

        if loaded:
            img1 = image_files[i]
        else:
            img1 = cv2.imread(os.path.join(FOLDER_PATH, image_files[i]))

        # Check if there is a second image available
        if i + 1 < len(image_files):
            if loaded:
                img2 = image_files[i+1]
            else:
                img2 = cv2.imread(os.path.join(FOLDER_PATH, image_files[i + 1]))

            # Merge the pair of images 50/50
            blend = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)
            merged_pairs.append(blend)
            successes += 1
        else:
            print("skipping image key number", str(i))
            # Only one image left, add it to the merged pairs list directly
            # merged_pairs.append(img1)
    # print(len(merged_pairs))
    # quit()
    return merged_pairs, successes

def merge_images(FOLDER_PATH):
    # Get a list of image files in the folder
    image_files = io.get_img_list(FOLDER_PATH)
    # print(image_files)
    cluster_no = None
    successes = 0
    if len(image_files) > 1:
        image_folder = FOLDER_PATH.split("/")[-1]
        if "cluster" in image_folder:
            print("cluster found", image_folder)
            cluster_no = int(image_folder.split("_")[0].replace("cluster",""))
        count = len(image_files)
        print(count)
        merged_pairs, successes = iterate_image_list(FOLDER_PATH,image_files,successes)
        print(type(merged_pairs[0]))

        # Continue merging until there is only one merged image left
        while len(merged_pairs) >= 2:
            merged_pairs, successes = iterate_image_list(FOLDER_PATH,merged_pairs,successes)

        final_merged = merged_pairs[0]

        return final_merged, successes, cluster_no
    else:
        return None, 0, cluster_no

def save_merge(merged_image, count, cluster_no, FOLDER_PATH):
    if cluster_no is not None:
        savename = 'merged_cluster_' + str(cluster_no)+ "_"+str(count*2)+'.jpg'
    else:
        savename = 'merged_image'+str(count)+'.jpg'
    output_path = os.path.join(FOLDER_PATH, savename)
    cv2.imwrite(output_path, merged_image)

    print('Merged image/video saved successfully here', output_path)


# def const_videowriter(subfolder_path, FRAMERATE=15):
#     img_array = io.get_img_list(subfolder_path)

#     # Get the dimensions of the first image in the array
#     image_path = os.path.join(subfolder_path, img_array[0])
#     img = cv2.imread(image_path)
#     height, width, _ = img.shape

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     video_path = os.path.join(FOLDER_PATH, FOLDER_NAME+cluster_no+".mp4")
#     video_writer = cv2.VideoWriter(video_path, fourcc, FRAMERATE, (width, height))
#     return video_writer

def get_img_list_subfolders(subfolders):
    all_img_path_list = []
    for subfolder_path in subfolders:
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
    if height > width:
        crop_dict = {
            "1x1": [0,width, cropfset,width+cropfset],
            "4x3": [0,width, int(cropfset/2),int(width*4/3+cropfset/2)],
            "4x2": [0,width, 0,height],
            "16x10": [0,int(height*10/16) , 0,height]
        }
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
    # Check if the ROOT folder exists, create it if not
    # print("subfolder_path")
    # print(subfolder_path)
    # img_array = io.get_img_list(subfolder_path)
    # print(img_array)

    # Get the dimensions of the first image in the array
    cluster_no, image_path = get_path(subfolder_path, img_array)
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(FOLDER_PATH, FOLDER_NAME+cluster_no+".mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, FRAMERATE, (width, height))

    # Iterate over the image array and write frames to the video


    for filename in img_array:
        if subfolder_path:
            image_path = os.path.join(subfolder_path, filename)
        else:
            image_path = filename
        print(image_path)
        img = cv2.imread(image_path)
        video_writer.write(img)

    # Release the video writer and close the video file
    video_writer.release()

    print(f"Video saved at: {video_path}")



def main():
    print("starting merge_expanded_images.py")
    if IS_CLUSTER is True:
        subfolders = io.get_folders(FOLDER_PATH)
        # print(subfolders)
        if IS_VIDEO is True and ALL_ONE_VIDEO is True:
            all_img_path_list = get_img_list_subfolders(subfolders)
            write_video(all_img_path_list, FRAMERATE)

            # # const_videowriter(subfolder_path, FRAMERATE)
            # for subfolder_path in subfolders:
            #     write_video(subfolder_path, FRAMERATE)
        elif IS_VIDEO is True and ALL_ONE_VIDEO is False:
            for subfolder_path in subfolders:
                all_img_path_list = io.get_img_list(subfolder_path)
                write_video(all_img_path_list, FRAMERATE, subfolder_path)

        else:
            for subfolder_path in subfolders:
                # print(subfolder_path)
                merged_image, count, cluster_no = merge_images(subfolder_path)
                if count == 0:
                    print("no images here")
                    continue
                else:
                    save_merge(merged_image, count, cluster_no, FOLDER_PATH)
    else:
        if IS_VIDEO is True:
            print("going to get folder ls")
            all_img_path_list = io.get_img_list(FOLDER_PATH)
            # print(all_img_path_list)
            crop_images(all_img_path_list, FOLDER_PATH)
            write_video(all_img_path_list, FRAMERATE, FOLDER_PATH)

            # # const_videowriter(subfolder_path, FRAMERATE)
            # for subfolder_path in subfolders:
            #     write_video(subfolder_path, FRAMERATE)
        else:

            merged_image, count, cluster_no = merge_images(FOLDER_PATH)
            save_merge(merged_image, count, cluster_no, FOLDER_PATH)
     








if __name__ == '__main__':
    main()










