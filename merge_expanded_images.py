import cv2
import os

def merge_images(folder_path):
    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('.')]    

    # Calculate the opacity value based on the total number of images
    opacity = 100 / len(image_files)

    # Initialize the merged image with the first image
    merged = cv2.imread(os.path.join(folder_path, image_files[0]))

    # Iterate through the remaining images and merge them
    for i in range(1, len(image_files)):
        img = cv2.imread(os.path.join(folder_path, image_files[i]))

        # Resize the image to match the size of the merged image if necessary
        if img.shape != merged.shape:
            img = cv2.resize(img, (merged.shape[1], merged.shape[0]))

        # Calculate the weight based on the opacity value
        weight = opacity / 100

        # Merge the current image with the merged image
        merged = cv2.addWeighted(merged, 1 - weight, img, weight, 0)

    return merged, len(image_files)

# Provide the path to the folder containing the images
folder_path = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/Screams_expanded_for_merging/'
folder_name ="13women"
file_path = os.path.join(folder_path,folder_name)
merged_image, count = merge_images(file_path)


output_path = os.path.join(file_path, 'merged_image'+str(count)+'.jpg')
cv2.imwrite(output_path, merged_image)

print('Merged image saved successfully.')
# # Display the merged image
# cv2.imshow('Merged Image', merged_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
