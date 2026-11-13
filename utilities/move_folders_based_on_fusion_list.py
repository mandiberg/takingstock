import os

'''
This script is for moving folders (and files I think) based on the fusion list. 
It moves files with the fusion cluster pairs into a a subfolder in that directory
I believe it is designed to work with make_video MODES 0 and 1:
It parses clustercc and clusterc_ filenames
'''
# FOLDER = "/Volumes/LaCie/segment_images_theoffice/output_folder/_ARMS_Thoma_nolimit/_4check"
# FOLDER = "/Volumes/LaCie/output_folder/_ARMS_T15_prints_oct17_round1"
FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/make_video_CSVs/SegmentHelper_TheOffice/_ARMS_handgesture_test_oct24"
MOVED_FOLDERS = "moved_folders"
NEW_FOLDER = os.path.join(FOLDER, MOVED_FOLDERS)
os.makedirs(NEW_FOLDER, exist_ok=True)

FUSION_PAIR_DICT_DETECTIONS_TOMOVE = {
    0: [

# exclude


# arms_stub_set
# [119, 15], [126, 15], [156, 15], [168, 1], [169, 15], [176, 2341], [183, 1], [18, 1685], [18, 1], [203, 1], [204, 15], [234, 15], [25, 15], [265, 15], [280, 11], [328, 1], [32, 15], [344, 15], [364, 15], [399, 15], [399, 1], [410, 1], [423, 1685], [423, 1], [443, 15], [460, 2263], [46, 1], [471, 15], [472, 734], [474, 15], [479, 58], [487, 727], [490, 15], [490, 960], [507, 15], [514, 15], [519, 15], [528, 58], [537, 282], [541, 15], [550, 1], [552, 1], [56, 1685], [571, 1], [578, 1], [579, 960], [57, 1], [583, 15], [593, 15], [597, 1], [609, 15], [609, 1], [619, 15], [638, 734], [647, 15], [649, 15], [707, 15], [710, 15], [726, 58], [734, 11], [738, 15], [741, 15], [745, 11], [749, 15], [752, 960], [756, 15], [7, 15], 

#/Volumes/LaCie/body_stub_set
# [466, 15], [472, 1], [473, 1], [476, 1], [489, 1], [520, 15], [59, 15], [601, 76], [607, 1], [620, 15], [62, 1], [65, 1], [692, 359], [694, 15], [70, 1], [739, 1], [765, 1], [76, 15], [85, 1], 



# 2recrop 
# [17, 11], [186, 1], [188, 1], [380, 15], [385, 11], [550, 1], [550, 1], [578, 15], 

# 4check  
# [100, 2263], [101, 15], [101, 1], [105, 15], [110, 1], [118, 15], [121, 15], [123, 1], [124, 1], [125, 15], [125, 1], [127, 1], [127, 58], [129, 1], [129, 7], [130, 15], [131, 1], [133, 15], [133, 1], [136, 1], [137, 11], [138, 1], [140, 1], [140, 1], [140, 1], [149, 15], [149, 1], [149, 1], [150, 1], [150, 1], [150, 28], [153, 727], [153, 960], [154, 15], [156, 58], [158, 15], [164, 1], [164, 1], [164, 1], [164, 2263], [164, 586], [167, 15], [168, 1], [168, 1], [168, 1], [170, 15], [170, 1], [173, 15], [173, 1], [173, 2341], [173, 734], [174, 1], [174, 1], [174, 2341], [174, 734], [176, 15], [176, 1], [179, 1], [181, 11], [181, 1], [181, 1], [181, 1], [181, 1], [181, 1], [181, 28], [184, 11], [184, 1], [185, 15], [186, 1], [186, 1], [188, 1], [194, 1], [197, 15], [197, 1], [197, 1], [197, 1], [197, 1], [197, 1], [197, 1], [197, 1], [197, 25], [203, 1], [204, 1], [216, 1], [216, 1], [216, 1], [221, 15], [221, 1685], [221, 1], [221, 1], [221, 2341], [221, 727], [222, 1], [226, 15], [226, 1], [226, 1], [227, 1], [227, 1], [227, 1], [227, 1], [227, 1], [227, 1], [232, 1], [232, 1], [232, 1], [232, 58], [240, 1], [240, 1], [240, 1], [240, 1], [240, 1], [240, 28], [241, 1], [245, 1], [252, 727], [252, 734], [253, 1], [253, 1], [253, 7], [258, 1], [25, 1], [263, 15], [263, 1], [265, 15], [265, 1], [265, 1], [265, 1], [265, 1], [265, 1], [265, 1], [265, 1], [265, 1], [265, 1], [270, 15], [270, 1], [270, 1], [272, 1], [272, 1], [276, 727], [277, 15], [277, 1], [277, 1], [285, 1], [285, 1], [285, 1], [285, 1], [28, 15], [290, 11], [290, 15], [290, 1], [290, 1], [291, 1], [294, 727], [298, 1], [298, 1], [299, 586], [299, 7], [302, 7], [303, 1], [303, 1], [307, 15], [307, 1], [307, 1], [307, 1], [311, 1685], [319, 1], [322, 1685], [322, 727], [329, 1685], [329, 727], [329, 733], [330, 1], [330, 58], [335, 15], [335, 1], [335, 1], [336, 1], [336, 1], [35, 15], [35, 1], [35, 1], [35, 1], [35, 1], [363, 1685], [369, 15], [369, 1], [370, 1], [370, 1], [373, 1], [375, 1], [379, 11], [37, 11], [37, 15], [380, 1], [381, 1], [384, 15], [385, 1], [390, 1], [402, 15], [402, 1], [413, 1], [416, 15], [418, 58], [422, 1685], [423, 1], [423, 1], [423, 1], [423, 1], [42, 15], [42, 1], [42, 1], [42, 1], [42, 1], [431, 1685], [436, 1], [436, 1], [443, 1], [455, 1685], [457, 1], [458, 15], [45, 15], [45, 1], [45, 1], [462, 1], [468, 1], [468, 1], [47, 1], [47, 7], [486, 1], [486, 1], [486, 1], [486, 1], [486, 1], [486, 1], [486, 7], [490, 1], [490, 1], [497, 15], [497, 1], [498, 15], [498, 1], [498, 1], [498, 1], [498, 1], [50, 1], [510, 15], [510, 1], [515, 1685], [516, 11], [517, 1], [517, 1], [519, 1], [523, 15], [523, 1], [528, 1], [528, 1], [537, 15], [542, 1], [544, 15], [544, 1], [546, 1], [550, 11], [550, 1], [552, 1], [560, 11], [562, 1], [566, 1], [567, 1], [567, 1], [571, 1], [573, 1], [574, 11], [574, 1], [575, 15], [579, 15], [580, 1], [581, 1], [582, 1], [583, 15], [583, 1], [587, 1], [590, 1], [592, 1], [593, 1], [598, 1], [599, 1], [602, 15], [602, 1], [602, 1], [602, 1], [627, 1], [638, 2341], [638, 734], [638, 7], [65, 2341], [660, 1], [670, 15], [676, 11], [678, 15], [680, 11], [699, 1], [699, 1], [69, 15], [704, 1], [704, 1], [704, 1], [704, 1], [705, 15], [705, 28], [718, 2341], [718, 734], [71, 1], [720, 2341], [722, 2341], [722, 734], [726, 58], [741, 15], [749, 15], [749, 727], [749, 733], [752, 2263], [756, 15], [764, 2263], [79, 1], [79, 1], [7, 1], [80, 58], [83, 1], [83, 1], [83, 1], [84, 2341], 

# 5exclude
# [360, 1], [360, 1], [463, 1], [463, 1], [552, 1], [560, 1], [560, 1], [560, 1], [560, 1], [560, 1], [576, 1], [576, 7], [578, 1], [590, 1], [590, 1], [598, 1], [609, 1], [609, 1], [609, 1], [609, 1], [609, 1], [609, 1], [609, 1], [618, 1], [618, 1], 


# from 4check

# _4check/x1  -- handsort
# [100, 2263], [105, 15], [121, 15], [129, 1], [133, 1], [133, 1], [133, 1], [133, 1], [133, 1], [133, 1], [133, 1], [133, 15], [133, 15], [133, 15], [136, 1], [140, 1], [140, 1], [140, 1], [140, 1], [140, 1], [164, 1], [164, 1], [164, 1], [167, 15], [173, 1], [173, 15], [173, 734], [174, 1], [174, 1], [174, 2341], [174, 734], [221, 1], [221, 1], [221, 1], [221, 15], [221, 2341], [227, 1], [227, 1], [227, 1], [227, 1], [227, 1], [227, 1], [232, 1], [232, 1], [232, 1], [232, 1], [232, 1], [263, 1], [263, 15], [272, 1], [272, 1], [285, 1], [285, 1], [285, 1], [285, 1], [290, 1], [290, 1], [290, 1], [290, 11], [290, 15], [298, 1], [298, 1], [299, 586], [299, 7], [303, 1], [303, 1], [307, 1], [307, 1], [307, 1], [307, 15], [319, 1], [319, 1], [330, 1], [335, 1], [335, 1], [335, 15], [336, 1], [336, 1], [35, 1], [35, 1], [35, 1], [35, 1], [35, 1], [35, 15], [384, 15], [390, 1], [390, 1], [402, 1], [402, 15], [416, 15], [42, 1], [42, 1], [42, 1], [42, 1], [42, 1], [42, 15], [423, 1], [423, 1], [423, 1], [423, 1], [436, 1], [436, 1], [468, 1], [468, 1], [47, 1], [47, 7], [490, 1], [490, 1], [498, 1], [498, 1], [498, 1], [498, 1], [498, 1], [498, 15], [50, 1], [510, 1], [510, 15], [523, 1], [523, 1], [523, 15], [537, 15], [542, 1], [566, 1], [573, 1], [581, 1], [592, 1], [593, 1], [65, 2341], [660, 1], [678, 15], [69, 15], [7, 1], [720, 2341], [749, 15], [752, 2263], [764, 2263], [83, 1], [83, 1], [83, 1], [84, 2341],

# oct 25 - arms handsort
[100, 2263],
[105, 15],
[121, 15],
[129, 1],
[133, 1],
[133, 15],
[136, 1],
[140, 1],
[164, 1],
[167, 15],
[173, 1],
[173, 15],
[173, 734],
[174, 1],
[174, 2341],
[174, 734],
[221, 1],
[221, 15],
[221, 2341],
[227, 1],
[232, 1],
[263, 1],
[263, 15],
[272, 1],
[285, 1],
[290, 1],
[290, 11],
[290, 15],
[298, 1],
[299, 586],
[299, 7],
[303, 1],
[307, 1],
[307, 15],
[319, 1],
[330, 1],
[335, 1],
[335, 15],
[336, 1],
[35, 1],
[35, 15],
[384, 15],
[390, 1],
[402, 1],
[402, 15],
[416, 15],
[42, 1],
[42, 15],
[423, 1],
[436, 1],
[468, 1],
[47, 1],
[47, 7],
[490, 1],
[498, 1],
[498, 15],
[50, 1],
[510, 1],
[510, 15],
[523, 1],
[523, 15],
[537, 15],
[542, 1],
[566, 1],
[573, 1],
[581, 1],
[592, 1],
[593, 1],
[65, 2341],
[660, 1],
[678, 15],
[69, 15],
[7, 1],
[720, 2341],
[749, 15],
[752, 2263],
[764, 2263],
[83, 1],
[84, 2341],


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
    if filename.endswith("manifest.json"):
        # allow it to proceed
        pass
    elif filename.startswith(".") or filename.startswith("_") or filename.startswith("image_ids.txt") or filename.endswith(".sql") or filename.endswith(".json"):
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
        # folders and files may be mixed (e.g. clustercc folders and merged_cluster jpgs),
    # so decide per item in main() rather than from the first entry
    return os.listdir(folderpath)

def main():
    filelist = get_list(FOLDER)
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
                    print(f"FOUND {name} is in cluster {this_arms_pose} and p {this_signature}")
                    if os.path.isdir(folderpath):
                        new_root = os.path.join(NEW_FOLDER, os.path.basename(folderpath))
                        move_folder(folderpath, new_root)
                    else:
                        move_file(name)
                    break


if __name__ == "__main__":
    main()