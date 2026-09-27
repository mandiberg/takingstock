import os

'''
This script is for moving folders (and files I think) based on the fusion list. 
It moves files with the fusion cluster pairs into a a subfolder in that directory
I believe it is designed to work with make_video MODES 0 and 1:
It parses clustercc and clusterc_ filenames
'''
FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/make_video_CSVs/SegmentHelper_TheGym/_ARMS_T15_p1_legposes_sep9/"
# FOLDER = "//Volumes/LaCie/output_folder/_3800_plus/videos"
FOLDER = "/Volumes/LaCie/segment_images_theoffice/output_folder/_Body_thoma_sept20/"
MOVED_FOLDERS = "moved_folders"
NEW_FOLDER = os.path.join(FOLDER, MOVED_FOLDERS)
os.makedirs(NEW_FOLDER, exist_ok=True)

FUSION_PAIR_DICT_DETECTIONS_TOMOVE = {
    0: [

# exclude
# [105, 1],[14, 1],[172, 1],[191, 1],[249, 1],[252, 1],[263, 1],[270, 1],[283, 1],[295, 1],[32, 1],[4, 1],[41, 1],[42, 76],[420, 1],[44, 1],[467, 1],[54, 1],[547, 1],[624, 1],[647, 1],[702, 1],[715, 1],[725, 1],[94, 1],[99, 1]

# # closer  
# [103, 1],[181, 1],[183, 1],[188, 1],[190, 1],[193, 1],[197, 1],[214, 1],[271, 1],[296, 1],[328, 1],[341, 1],[370, 1],[378, 1],[433, 1],[436, 1],[436, 15],[447, 1],[567, 1],[606, 1],[666, 1],[673, 1],[674, 1],[682, 1],[694, 1],[695, 1],[713, 1],[723, 1],[91, 1]

# # recanon
[109, 1],[132, 1],[133, 1],[135, 1],[152, 1],[156, 1],[196, 1],[200, 1],[260, 1],[265, 1],[275, 1],[284, 1],[312, 1],[343, 1],[345, 1],[364, 1],[387, 1],[404, 1],[406, 1],[412, 1],[514, 1],[581, 1],[698, 1],[707, 1],[731, 1],[86, 1],[97, 1]

    ]
}

def extract_fustion_cluster(name):
    name = name.replace("_ct", "_")
    folder_arms_pose = folder_signature = None
    print(f"extracting fusion cluster from name {name}")
    if "wav" in name:
        # handle audio file format: multitrack_mixdown_offset_cc183_p1_t0_1781177644.763904.wav
        folder_arms_pose = name.split("cc")[1].split("_")[0]
        folder_signature = name.split("p")[1].split("_")[0]
    elif "cluster-" in name:
        folder_arms_pose = name.split("cluster-")[1].split("_")[0]
    elif "clustercc" in name:
        folder_arms_pose = name.split("clustercc")[1].split("_")[0]
    elif "merged_cluster" in name:
        folder_arms_pose = name.split("merged_cluster")[1].split("_")[0]
    elif "_c" in name:    
        folder_arms_pose = name.split("_c")[1].split("_")[0]
    if "_p" in name:
        folder_signature = name.split("_p")[1].split("_")[0]
    if folder_arms_pose is not None and folder_signature is not None:
        print(f"extracted fusion cluster {folder_arms_pose} and {folder_signature} from name {name}")
        folder_arms_pose = int(folder_arms_pose)
        folder_signature = int(folder_signature)
    else:
        print(f"Failed to extract fusion cluster from name {name}")
    return folder_arms_pose, folder_signature

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

def move_file(filename):
    if filename.startswith(".") or filename.startswith("image_ids.txt") or filename.endswith(".sql"):
        print(f"Skipping file {filename} because it is a hidden file or an output file")
        return
    filepath = os.path.join(FOLDER, filename)
    new_filepath = os.path.join(NEW_FOLDER, filename)
    # check to see if filename already exists at new_filepath
    if os.path.exists(new_filepath):
        print("the new_filepath already exists")
        return
    os.rename(filepath, new_filepath)
    print(f"Moved file {filepath} to {new_filepath}")


def get_list(folderpath):
    filelist = os.listdir(folderpath)
    print(f"checking for {filelist[0]}")
    if "mp4" in filelist[0] or "wav" in filelist[0]:
        files_only = True
    elif "df_sorted" in filelist[0]:
        files_only = True
    elif "clustercc" in filelist[0]:
        files_only = False
    else:
        print (f"fileleist {filelist}")
        raise Exception(f"Unexpected folder contents in {folderpath}, expected either df_sorted or clustercc")
    return filelist, files_only

def main():
    filelist, files_only = get_list(FOLDER)
    for name in filelist:
        print(f"Processing file: {name}")
        if MOVED_FOLDERS in name:
            print(f"Skipping folder {name} because it is in the moved folders")
            continue
        folderpath = os.path.join(FOLDER, name)
        this_arms_pose, this_signature = extract_fustion_cluster(name)
        if this_arms_pose is not None and this_signature is not None:
            for dict_arms, dict_sig in FUSION_PAIR_DICT_DETECTIONS_TOMOVE[0]:
                if dict_arms == this_arms_pose and dict_sig == this_signature:
                    print(f"FOUND {name} is in cluster {this_arms_pose} and p {this_signature}, files_only is {files_only}")
                    if os.path.isdir(folderpath):
                        new_root = os.path.join(NEW_FOLDER, os.path.basename(folderpath))
                        move_folder(folderpath, new_root)
                        break
                    elif files_only:
                        print(f"doing move on df_sorted files in {name}")
                        # new_root = os.path.join(NEW_FOLDER, os.path.basename(folderpath))
                        move_file(name)


if __name__ == "__main__":
    main()