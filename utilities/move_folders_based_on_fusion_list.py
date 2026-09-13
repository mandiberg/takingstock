import os

'''
This script is for moving folders (and files I think) based on the fusion list. 
It moves files with the fusion cluster pairs into a a subfolder in that directory
I believe it is designed to work with make_video MODES 0 and 1:
It parses clustercc and clusterc_ filenames
'''
FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/make_video_CSVs/SegmentHelper_TheGym/_ARMS_T15_p1_legposes_sep9/"
# FOLDER = "//Volumes/LaCie/output_folder/_3800_plus/videos"
FOLDER = "/Volumes/LaCie/output_folder/_BODY_T45_p1_sept10_pt1/"
MOVED_FOLDERS = "moved_folders"
NEW_FOLDER = os.path.join(FOLDER, MOVED_FOLDERS)
os.makedirs(NEW_FOLDER, exist_ok=True)

FUSION_PAIR_DICT_DETECTIONS_THEOFFICE = {
    0: [

 # fullbody look closer - datto opening day
# [134, 1], [147, 1], [167, 1], [170, 1], [197, 1], [216, 1], [284, 1], [297, 1], [316, 1], [344, 1], [346, 1], [351, 1], [353, 1], [355, 1], [360, 1], [366, 1], [369, 1], [370, 1], [371, 1], [387, 1], [389, 1], [393, 1], [394, 1], [396, 1], [399, 1], [4, 1], [404, 1], [410, 1], [413, 1], [414, 1], [416, 1], [426, 1], [427, 1], [442, 1], [444, 1], [445, 1], [447, 1], [448, 1], [470, 1], [474, 1], [475, 1], [497, 1], [514, 1], [519, 1], [52, 1], [524, 1], [57, 1], [58, 1], [589, 1], [605, 1], [610, 1], [637, 1], [641, 1], [649, 1], [651, 1], [652, 1], [654, 1], [669, 1], [671, 1], [68, 1], [680, 1], [686, 1], [698, 1], [701, 1], [703, 1], [704, 1], [705, 1], [707, 1], [710, 1], [712, 1], [721, 1], [724, 1], [729, 1], [732, 1], [736, 1], [746, 1], [749, 1], [755, 1], [756, 1], [757, 1], [760, 1], [761, 1], [763, 1], [765, 1], [83, 1], [85, 1], [88, 1], [9, 1], [91, 1], [92, 1], [94, 1],

# fiullbody recrop - gatto opening day
# [123, 1], [126, 1], [135, 1], [152, 1], [172, 1], [183, 1], [196, 1], [200, 1], [206, 1], [214, 1], [217, 1], [222, 1], [229, 1], [234, 1], [245, 1], [249, 1], [260, 1], [263, 1], [264, 1], [265, 1], [275, 1], [285, 1], [295, 1], [296, 1], [312, 1], [318, 1], [332, 1], [336, 1], [343, 1], [345, 1], [364, 1], [412, 1], [424, 1], [450, 1], [593, 1], [635, 1], [643, 1], [648, 1], [694, 1], [695, 1], [709, 1], [731, 1], [86, 1], [97, 1], 

# good nature
[104, 1], [107, 1], [160, 1], [199, 1], [219, 1], [340, 1], [350, 1], [391, 1], [395, 1],


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
            for dict_arms, dict_sig in FUSION_PAIR_DICT_DETECTIONS_THEOFFICE[0]:
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