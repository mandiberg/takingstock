import os

'''
This script is for moving folders (and files I think) based on the fusion list. 
It moves files with the fusion cluster pairs into a a subfolder in that directory
I believe it is designed to work with make_video MODES 0 and 1:
It parses clustercc and clusterc_ filenames
'''
FOLDER = "/Volumes/LaCie/output_folder/_small_clusters_test"
# FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/make_video_CSVs/SegmentHelper_TheOffice/new_sig_test_oldones/obj_not_p15"
# FOLDER = "/Volumes/OWC5/basel_audio_sort/audio.BSC_FINAL"
MOVED_FOLDERS = "moved_folders"
NEW_FOLDER = os.path.join(FOLDER, MOVED_FOLDERS)
os.makedirs(NEW_FOLDER, exist_ok=True)

FUSION_PAIR_DICT_DETECTIONS_THEOFFICE = {
    0: [

[-1, 1002], [-1, 1014], [-1, 104], [-1, 110], [-1, 1182], [-1, 1255], [-1, 1315], [-1, 1380], 
[-1, 1465], [-1, 149], [-1, 150], [-1, 1733], [-1, 1734], [-1, 1781], [-1, 1828], [-1, 2058], 
[-1, 206], [-1, 207], [-1, 21], [-1, 2161], [-1, 256], [-1, 261], [-1, 2818], [-1, 2883], [-1, 2982],
 [-1, 2996], [-1, 3058], [-1, 3082], [-1, 3096], [-1, 3130], [-1, 3131], [-1, 3150], [-1, 321], [-1, 3234],
   [-1, 3252], [-1, 3256], [-1, 3268], [-1, 3276], [-1, 3407], [-1, 3408], [-1, 3430], [-1, 3546], 
   [-1, 3561], [-1, 359], [-1, 3634], [-1, 3640], [-1, 3682], [-1, 3685], [-1, 37], [-1, 3768], [-1, 3776],
     [-1, 438], [-1, 494], [-1, 50], [-1, 595], [-1, 612],
 [-1, 621], [-1, 712], [-1, 754], [-1, 802], [-1, 849], [-1, 952], [-1, 974], [-1, 999], [-1, 1254],
[-1, 392], [-1, 948], [-1, 1478], [-1, 3512],
        # # credit card
        # [100,733],[100,734],[129,586],[129,727],[140,734],[153,727],[164,586],[173,734],[174,734],[18,3052],[18,586],[18,727],[18,733],[18,734],[207,734],[221,258],[221,727],[252,586],[252,727],[252,734],[272,258],[276,258],[276,727],[276,734],[285,258],[294,727],[299,586],[299,734],[322,727],[329,727],[329,733],[32,586],[376,734],[410,734],[423,258],[460,727],[47,734],[56,258],[89,258],[89,727],[472,734],[474,734],[487,727],[490,727],[490,733],[518,586],[547,734],[572,734],[587,734],[638,734],[701,258],[718,734],[720,727],[722,734],[729,734],[749,727],[749,733],[752,586],[752,727],[752,733],

        # # money
        # [100,2263],[106,2341],[129,2263],[129,2341],[138,1685],[140,2341],[164,2263],[173,1685],[173,2341],[174,2341],[176,2341],[181,1685],[18,1685],[18,2263],[207,2341],[218,1685],[221,1685],[221,2341],[232,1685],[240,1685],[252,2263],[252,2341],[272,1685],[276,1685],[276,2263],[290,1685],[299,2341],[311,1685],[319,1685],[322,1685],[329,1685],[336,2263],[363,1685],[396,1685],[3,1685],[410,2263],[410,2341],[422,1685],[423,1685],[431,1685],[455,1685],[460,2263],[47,1685],[47,2341],[56,1685],[65,2341],[83,1685],[84,2341],[472,2341],[479,1685],[485,2263],[487,2263],[487,2341],[490,2263],[490,2341],[515,1685],[523,1685],[528,1685],[542,2341],[547,2341],[566,2341],[579,2341],[589,2341],[638,2341],[701,1685],[718,2341],[720,2341],[722,2341],[749,2341],[752,1685],[752,2263],[752,2341],[757,2341],[764,2263],

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