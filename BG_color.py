import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil

FOLDER_PATH = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/bg_color/0900"
results = []
SORTTYPE = "luminosity"  # "hue" or "luminosity"
output_folder = os.path.join(FOLDER_PATH, SORTTYPE)
os.makedirs(output_folder, exist_ok=True)

get_background_mp = mp.solutions.selfie_segmentation
get_bg_segment = get_background_mp.SelfieSegmentation()

'''
now itterates through folder for testing.
luminosity works well.
hue isn't as useful, as it is only the chroma, so it treats 95% white with a blue cast the same as a vivid blue

the next step is to pull from db and write back to db. 
let's stick to the segment table.
I revised fetch_bagofkeywords.py, and I would suggest using that as a template for this.
it uses myslqalchemy now. and the threading is stable

it needs to pull the bbox, and crop the image to bbox and then run that through the function
because that is what will actually be in the final image
most images have even continuous backgrounds, but some have uneven backgrounds that throw off the average.
'''

def get_bg_hue_lum(file):
    sample_img = cv2.imread(file)
    result = get_bg_segment.process(sample_img[:,:,::-1])
    mask=np.repeat((1-result.segmentation_mask)[:, :, np.newaxis], 3, axis=2)
    masked_img=mask*sample_img[:,:,::-1]/255 ##RGB format
    # Identify black pixels where R=0, G=0, B=0
    black_pixels_mask = np.all(masked_img == [0, 0, 0], axis=-1)
    # Filter out black pixels and compute the mean color of the remaining pixels
    mean_color = np.mean(masked_img[~black_pixels_mask], axis=0)[np.newaxis,np.newaxis,:] # ~ is negate
    hue=cv2.cvtColor(mean_color, cv2.COLOR_RGB2HSV)[0,0,0]
    lum=cv2.cvtColor(mean_color, cv2.COLOR_RGB2LAB)[0,0,0]
    return hue,lum

def get_bg_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            hue, lum = get_bg_hue_lum(file_path)
            results.append({"file": filename, "hue": hue, "luminosity": lum})

    # Create DataFrame from results and sort by SORTYPE
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by=SORTTYPE)

    print(df_sorted)

    # Iterate over sorted DataFrame and save copies of each file to output folder
    counter = 0
    total = len(df_sorted)

    for index, row in df_sorted.iterrows():
        old_file_path = os.path.join(folder_path, row["file"])
        filename = f"{str(counter)}_{int(row[SORTTYPE])}_{row['file']}"
        print(filename)
        new_file_path = os.path.join(output_folder, filename)
        shutil.copyfile(old_file_path, new_file_path)
        print(f"File '{row['file']}' copied to '{filename}'")
        counter += 1

    print("Files saved to", output_folder)


get_bg_folder(FOLDER_PATH)

