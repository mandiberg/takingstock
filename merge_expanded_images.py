import cv2
import os
import numpy as np

def iterate_image_list(folder_path,image_files, successes):
    # Initialize the merged pairs list with the images in pairs
    merged_pairs = []
    if type(image_files[0]) is np.ndarray:
        loaded = True
    else:
        loaded = False
    print(loaded)
    # Iterate through the image files and merge them in pairs
    for i in range(0, len(image_files), 2):
        # print(folder_path, image_files[i])

        if loaded:
            img1 = image_files[i]
        else:
            img1 = cv2.imread(os.path.join(folder_path, image_files[i]))

        # Check if there is a second image available
        if i + 1 < len(image_files):
            if loaded:
                img2 = image_files[i+1]
            else:
                img2 = cv2.imread(os.path.join(folder_path, image_files[i + 1]))

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

def merge_images(folder_path):
    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('.')]
    # print(image_files)
    cluster_no = None
    successes = 0
    if len(image_files) > 1:
        image_folder = folder_path.split("/")[-1]
        if "cluster" in image_folder:
            print("cluster found", image_folder)
            cluster_no = int(image_folder.split("_")[0].replace("cluster",""))
        count = len(image_files)
        print(count)
        merged_pairs, successes = iterate_image_list(folder_path,image_files,successes)
        print(type(merged_pairs[0]))

        # Continue merging until there is only one merged image left
        while len(merged_pairs) >= 2:
            merged_pairs, successes = iterate_image_list(folder_path,merged_pairs,successes)

        final_merged = merged_pairs[0]

        return final_merged, successes, cluster_no
    else:
        return None, 0, cluster_no

def get_folders(folder):
    subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
    return subfolders

def save_merge(merged_image, count, cluster_no, folder_path):
    if cluster_no is not None:
        savename = 'merged_cluster_' + str(cluster_no)+ "_"+str(count*2)+'.jpg'
    else:
        savename = 'merged_image'+str(count)+'.jpg'
    output_path = os.path.join(folder_path, savename)
    cv2.imwrite(output_path, merged_image)

# iterate through folders? 
IS_CLUSTER = True

# Provide the path to the folder containing the images
root_folder_path = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/'

# if IS_CLUSTER this should be the folder holding all the cluster folders
# if not, this should be the individual folder holding the images
folder_name ="June2_clusters"
folder_path = os.path.join(root_folder_path,folder_name)

if IS_CLUSTER is True:
    subfolders = get_folders(folder_path)
    # print(subfolders)
    for subfolder_path in subfolders:
        # print(subfolder_path)
        merged_image, count, cluster_no = merge_images(subfolder_path)
        if count == 0:
            print("no images here")
            continue
        else:
            save_merge(merged_image, count, cluster_no, folder_path)
else:
    merged_image, count, cluster_no = merge_images(folder_path)
    save_merge(merged_image, count, cluster_no, folder_path)
 


print('Merged image saved successfully.')
















