import os

'''
This script is for moving composite files into folders
It looks in a folder, and moves files into folders with matching cluster. 
'''

FOLDER = "/Volumes/LaCie/output_folder/_p15_hsv_refine200"


def extract_fustion_cluster(name):
    name = name.replace("_ct", "_")
    folder_arms_pose = folder_signature = folder_hsv = None
    print(f"extracting fusion cluster from name {name}")
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
        print(f"extracted fusion cluster {folder_arms_pose} and {folder_signature} from name {name}")
        folder_arms_pose = int(folder_arms_pose)
        folder_signature = int(folder_signature)
    return folder_arms_pose, folder_signature, folder_hsv

def move_folder(folderpath, new_folderpath):
    # foldername = os.path.basename(folderpath)
    # new_folderpath = os.path.join(new_root, foldername)
    os.makedirs(new_folderpath, exist_ok=True)
    for filename in os.listdir(folderpath):
        old_file = os.path.join(folderpath, filename)
        new_file = os.path.join(new_folderpath, filename)
        os.rename(old_file, new_file)
    # delete old folder
    os.rmdir(folderpath)
    print(f"Moved folder {folderpath} to {new_folderpath}")

def move_file(filename, new):
    if filename.startswith(".") or filename.startswith("image_ids.txt") or filename.endswith(".sql"):
        print(f"Skipping file {filename} because it is a hidden file or an output file")
        return
    filepath = os.path.join(FOLDER, filename)
    new_filepath = os.path.join(NEW_FOLDER, filename)
    os.rename(filepath, new_filepath)
    print(f"Moved file {filepath} to {new_filepath}")

def test_match(name, this_arms_pose, this_signature, this_hsv):
    this_match = False
    if str(this_arms_pose)+"_p"+str(this_signature) in name:
        if "_om" not in name:
            this_match = True
        elif this_hsv and "_om"+this_hsv+"_" in name:
            this_match = True
    # print(this_match, name, this_arms_pose, this_signature, this_hsv)
    return this_match

def get_list(folderpath):
    image_list = []
    folder_list = []
    filelist = os.listdir(folderpath)
    for file in filelist:
        print("file to sort: ", file)
        if "jpg" in file:
            image_list.append(file)
            # print("is jpg", file)
            print("current image_list", image_list)
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
    for name in image_list:
        print(f"Processing file: {name}")
        this_arms_pose, this_signature, this_hsv = extract_fustion_cluster(name)
        if this_arms_pose is not None and this_signature is not None:
            for folder in folder_list:
                # print("folder", folder)
                if test_match(folder, this_arms_pose, this_signature, this_hsv):
                    filepath = os.path.join(FOLDER, name)
                    newfilepath = os.path.join(FOLDER,folder, name)
                    print(f"goingt to move {filepath} to {newfilepath}")
                    os.rename(filepath, newfilepath)

if __name__ == "__main__":
    main()