import os

'''
This script is for moving folders (and files I think) based on the fusion list. 
It moves files with the fusion cluster pairs into a a subfolder in that directory
I believe it is designed to work with make_video MODES 0 and 1:
It parses clustercc and clusterc_ filenames
'''
# FOLDER = "/Volumes/LaCie/output_folder/_small_clusters_test"
FOLDER = "//Volumes/LaCie/output_folder/_3800_plus/videos"
# FOLDER = "/Volumes/OWC5/basel_audio_sort/audio.BSC_FINAL"
MOVED_FOLDERS = "moved_folders"
NEW_FOLDER = os.path.join(FOLDER, MOVED_FOLDERS)
os.makedirs(NEW_FOLDER, exist_ok=True)

FUSION_PAIR_DICT_DETECTIONS_THEOFFICE = {
    0: [

[240, 3984], [705, 3984], [150, 3984], [270, 3984], [45, 3984], [734, 3984], [528, 3984], [45, 3987], [181, 3984], [340, 3986], [705, 3986], [250, 3984], [51, 3984], [146, 3984], [150, 3987], [240, 3987], [156, 3984], [662, 3984], [396, 3984], [21, 3987], [459, 3984], [432, 3984], [280, 3984], [517, 3984], [523, 3984], [471, 3987], [705, 3987], [274, 3987], [196, 3987], [477, 3987], [726, 3984], [369, 3984], [240, 3986], [270, 3987], [503, 3984], [359, 3984], [722, 3987], [181, 3987], [463, 3984], [541, 3984], [196, 3988], [240, 4018], [528, 4018], [45, 4018], [705, 4018], [459, 3942], [752, 4631], [270, 4018], [156, 4018], [726, 3940], [235, 3942], [21, 3942], [490, 4631], [117, 3988], [726, 3939], [708, 3988], [477, 3988], [459, 3939], [150, 4018], [463, 4018]
    ]
}
# FUSION_PAIR_DICT_DETECTIONS_THEOFFICE = {
#     0: [
#         # looping videos
# #### first block is 15 Tie ####
# # arms_crossed
# [155,15],  [399,15],   [541,15], [605,15], 

# #blan
# [234,15],   [443,15],  [471,15],

# #_arms_ou
# [134,15], [344,15], [364,15],   [597,15],   [707,15], [738,15], [78,15],

# #_arms_stress:], 
# [254,15], [104,15], [128,15], [182,15], [390,15], [390,15],   [7,15], 

# #_weird:], 
#  [519,15], [593,15], [649,15], [710,15], [647,15],  

# # pointing
# [101,15], [176,15], [204,15], [25,15], [494,15], [670,15], [756,15],

# #_standin
# [126,15], [183,15], [332,15], [347,15], [626,15], [632,15], [702,15],

# #_arms_backhea
# [286,15], [343,15], 

# #pensiv
# [119,15], [197,15], [265,15],

# #_arms_hip
# [514,15], [698,15], 

# # video_good/selects,], 
# [117,7], [181,2230], [18,586], [18,733], [197,25], [253,7], [307,25], [308,7], [350,25], [369,11], [378,25], [426,11], [479,2230], [482,7], 
# [487,2263], [537,282], [552,11], [572,734], [69,282], [729,7], [745,11], [77,11], [82,282], [83,1685], [18,3052], [359,11], [385,11], [432,11], 
# [45,11], [463,11], [472,7], [486,7], [628,11], [673,11], [698,11], [726,58],
#     ]
# }


# FUSION_PAIR_DICT_DETECTIONS_THEOFFICE = {
#     0: [
# #         # LUMEN tests
# #         #[arms_pose_cluster, object_signature_cluster]

# #         # phone:
#         [350,25],
       
# #         # laptop:
#         [224,34],[291,34],[503,34],[711,34],

# #         # # arms_head:
#         [126,15],[332,15],[662,15],[702,15],[734,15],[654,15],

# #         # # money both hands
#         [319,1685],

# #         # # credit card both hands
#         [272,258],[701,258],
        
#         [47,734],
        
# #         # arms_pointing:
#         [101,15],[204,15],[226,15],[494,15],[602,15],[670,15],[756,15],

#         # money:
#         [106,2341],[129,2263],[129,2341],[140,2341],[173,2341],[174,2341],[176,2341],[181,1685],[18,1685],
#         [218,1685],[221,2341],[232,1685],[240,1685],[252,2263],[276,2341],[299,2341],[410,2341],[479,1685],[47,2341],[722,2341],[752,1685],[752,2341],[84,2341],
#         # [319,1685],
#     ]
# }


# FUSION_PAIR_DICT_DETECTIONS_THEOFFICE = {
#     0: [
#         # SELECTS, for short gifs
#         #[arms_pose_cluster, object_signature_cluster]
#         # arms_head:
#         [105,15],[128,15],[182,15],[286,15],[33,15],[343,15],[581,15],[601,15],[731,15],

#         # arms_misc:
#         [119,15],[197,15],[299,15],[328,15],[514,15],[736,15],

#         # arms_out:
#         [134,15],[190,15],[282,15],[364,15],[593,15],[597,15],[619,15],[707,15],[738,15],[78,15],

#         # arms_pointing:
#         [101,15],[204,15],[226,15],[494,15],[602,15],[670,15],[756,15],

#         # card:
#         [173,734],[18,727],[207,734],[221,258],[252,727],[272,258],[276,258],[276,734],[294,727],[460,727],[47,734],[487,727],
#         [490,733],[587,734],[701,258],[752,733],[89,258],

#         # laptop:
#         [124,34], [17,11],[188,34],[224,34],[291,34],[425,34],[426,11],[45,34],[503,34],[51,34],[550,11],[580,34],[598,34],[608,34],
#         [673,11],[676,11],[698,11],[711,34],[721,34],[726,34],[734,11],[745,11],[745,34],[747,34],[74,34],

#         # money:
#         [106,2341],[129,2263],[129,2341],[140,2341],[173,2341],[174,2341],[176,2341],[181,1685],[18,1685],
#         [218,1685],[221,2341],[232,1685],[240,1685],[252,2263],[276,2341],[299,2341],[319,1685],[410,2341],[479,1685],[47,2341],[722,2341],[752,1685],[752,2341],[84,2341],

#         # phone:
#         [131,25],[181,79],[197,25],[218,58],[235,29],[308,7],[340,7],[350,25],[474,7],[479,35],[479,79],[611,25],[69,282],[701,58],[729,7],[82,282],[82,437],

#         # standing:
#         [121,15],[126,15],[332,15],[626,15],[632,15],[698,15],[702,15],

#         # misc
#         [168,15],[181,2230],[490,960]
#         ]
#     }

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
    elif "cc" in name:
        folder_arms_pose = name.split("cc")[1].split("_")[0]
    elif "_c" in name:    
        folder_arms_pose = name.split("_c")[1].split("_")[0]
    if "_p" in name:
        folder_signature = name.split("_p")[1].split("_")[0]
    if folder_arms_pose is not None and folder_signature is not None:
        print(f"extracted fusion cluster {folder_arms_pose} and {folder_signature} from name {name}")
        folder_arms_pose = int(folder_arms_pose)
        folder_signature = int(folder_signature)
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
    os.rename(filepath, new_filepath)
    print(f"Moved file {filepath} to {new_filepath}")


def get_list(folderpath):
    filelist = os.listdir(folderpath)
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