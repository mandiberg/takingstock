import os

'''
Script uses FOCUS_CLUSTERS to parse SOURCE_FOLDER for the clusters of interest, and copies those clusters to OUTPUT_FOLDER.
OUTPUT_FOLDER may contain mp4 or folders
mp4 follow this naming convention: {prefix}clustercc{arms_pose_cluster}_p{object_signature_cluster}_{suffix}.mp4
folders follow this naming convention: clustercc{arms_pose_cluster}_p{object_signature_cluster}_{suffix}
Script walks SOURCE_FOLDER and moves files/folders with arms_pose_cluster, object_signature_cluster in FOCUS_CLUSTERS to OUTPUT_FOLDER, using dict key as subfolder name.


'''
SOURCE_FOLDER = "/Volumes/LaCie/output_folder/_selectGIFtestMay19/videos_selectGIFtestMay19"
OUTPUT_FOLDER = os.path.join(SOURCE_FOLDER, "sorted_fusion_clusters")

FOCUS_CLUSTERS = [
    #[arms_pose_cluster, object_signature_cluster]
    {"arms_head": [[105,15],[128,15],[182,15],[286,15],[33,15],[343,15],[581,15],[601,15],[731,15]]},

    # arms_misc:
    {"arms_misc": [[119,15],[197,15],[299,15],[328,15],[514,15],[736,15]]},

    # arms_out:
    {"arms_out": [[134,15],[190,15],[282,15],[364,15],[593,15],[597,15],[619,15],[707,15],[738,15],[78,15]]},

    # arms_pointing:
    {"arms_pointing": [[101,15],[204,15],[226,15],[494,15],[602,15],[670,15],[756,15]]},

    # card:
    {"card": [[173,734],[18,727],[207,734],[221,258],[252,727],[272,258],[276,258],[276,734],[294,727],[460,727],[47,734],[487,727],[490,733],[587,734],[701,258],[752,733],[89,258]]},

    # laptop:
    {"laptop": [[124,34],[17,11],[188,34],[224,34],[291,34],[425,34],[426,11],[45,34],[503,34],[51,34],[550,11],[580,34],[598,34],[608,34],
                [673,11],[676,11],[698,11],[711,34],[721,34],[726,34],[734,11],[745,11],[745,34],[747,34],[74,34]]},

    # money:
    {"money": [[106,2341],[129,2263],[129,2341],[140,2341],[173,2341],[174,2341],[176,2341],[181,1685],[18,1685],
     [218,1685],[221,2341],[232,1685],[240,1685],[252,2263],[276,2341],[299,2341],[319,1685],[410,2341],[479,1685],[47,2341],[722,2341],[752,1685],[752,2341],[84,2341]]},

    # phone:
    {"phone": [[131,25],[181,79],[197,25],[218,58],[235,29],[308,7],[340,7],[350,25],[474,7],[479,35],[479,79],[611,25],[69,282],[701,58],[729,7],[82,282],[82,437]]},

    # standing:
    {"standing": [[121,15],[126,15],[332,15],[626,15],[632,15],[698,15],[702,15]]},

    # misc
    {"misc": [[168,15],[181,2230],[490,960]]}
]


def move_files_folders():
    for cluster_dict in FOCUS_CLUSTERS:
        for cluster_name, cluster_values in cluster_dict.items():
            for arms_pose_cluster, object_signature_cluster in cluster_values:
                # construct expected filename
                expected_filename = f"clustercc{arms_pose_cluster}_p{object_signature_cluster}"
                # search SOURCE_FOLDER for files/folders containing expected_filename
                for item in os.listdir(SOURCE_FOLDER):
                    if expected_filename in item:
                        # move item to OUTPUT_FOLDER/cluster_name/
                        destination_folder = os.path.join(OUTPUT_FOLDER, cluster_name)
                        os.makedirs(destination_folder, exist_ok=True)
                        source_path = os.path.join(SOURCE_FOLDER, item)
                        destination_path = os.path.join(destination_folder, item)
                        os.rename(source_path, destination_path)
                        print(f"Moved {source_path} to {destination_path}")

                    
if __name__ == "__main__":
    move_files_folders()
