import cv2
import os
import numpy as np

def iterate_image_list(image_files):
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
        else:
            print("skipping image key number", str(i))
            # Only one image left, add it to the merged pairs list directly
            # merged_pairs.append(img1)

    return merged_pairs
def merge_images(folder_path):
    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('.')]

    merged_pairs = iterate_image_list(image_files)
    print(type(merged_pairs[0]))

    # Continue merging until there is only one merged image left
    while len(merged_pairs) >= 2:
        merged_pairs = iterate_image_list(merged_pairs)

    final_merged = merged_pairs[0]

    return final_merged, len(image_files)

# Provide the path to the folder containing the images
root_folder_path = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/'
folder_name ="images1685438827.369812"
folder_path = os.path.join(root_folder_path,folder_name)
merged_image, count = merge_images(folder_path)


output_path = os.path.join(folder_path, 'merged_image'+str(count)+'.jpg')
cv2.imwrite(output_path, merged_image)

print('Merged image saved successfully.')