import os

'''
This script counts files in subfolders based on fusion cluster. 
This is for when you are making a looping video and need to know how many images are in each cluster folder.
Useful to figuring out the min values
And for comparing one run to an other to see if any need further pruning
'''

# FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/_looping_june22_BER"
FOLDER = "/Volumes/LaCie/output_folder/_looping_june22_bk_arms110ct"

def extract_fustion_cluster(name):
    name = name.replace("_ct", "_")
    folder_arms_pose = folder_signature = folder_hsv = None
    # print(f"extracting fusion cluster from name {name}")
    if "wav" in name:
        # handle audio file format: multitrack_mixdown_offset_cc183_p1_t0_1781177644.763904.wav
        folder_arms_pose = name.split("cc")[1].split("_")[0]
        folder_signature = name.split("p")[1].split("_")[0]
    elif "cc" in name:
        folder_arms_pose = name.split("cc")[1].split("_")[0]
    elif "_cluster" in name:
        folder_arms_pose = name.split("_cluster")[1].split("_")[0]
    elif "_c" in name:    
        folder_arms_pose = name.split("_c")[1].split("_")[0]
    if "_p" in name:
        folder_signature = name.split("_p")[1].split("_")[0]
    if "_om" in name:
        folder_hsv = name.split("_om")[1].split("_")[0]
    if folder_arms_pose is not None and folder_signature is not None:
        # print(f"extracted fusion cluster {folder_arms_pose} and {folder_signature} from name {name}")
        folder_arms_pose = int(folder_arms_pose)
        folder_signature = int(folder_signature)
    return folder_arms_pose, folder_signature, folder_hsv

def get_list(folderpath):
    image_list = []
    folder_list = []
    filelist = os.listdir(folderpath)
    for file in filelist:
        # print("file to sort: ", file)
        if "jpg" in file:
            image_list.append(file)
            # print("is jpg", file)
            # print("current image_list", image_list)
            # print("current folder_list", folder_list)
        else:
            # print("is folder,", file)
            folder_list.append(file)
    #         print("current image_list", image_list)
            # print("current folder_list", folder_list)
    # print(f"image_list, {image_list}")
    # print(f"folder_list {folder_list}")
    return image_list, folder_list

def main():
    image_list, folder_list = get_list(FOLDER)
    for folder in folder_list:
        if "DS_Store" in folder: continue
        folderpath = os.path.join(FOLDER,folder)
        # print(f"Processing file: {folderpath}")
        this_arms_pose, this_signature, this_hsv = extract_fustion_cluster(folder)
        image_list, folder_list = get_list(folderpath)
        print(this_arms_pose, this_signature, len(image_list))



if __name__ == "__main__":
    main()