import os
import shutil
import exiftool


'''
This script is for the cluster cleaning process -- for the looping videos
Save your good images in a folder, and then move them into a subfolder 
so you only are cleaning *new* images.
This will walk the folder and move for all subfolders
'''

FOLDER = "/Volumes/LaCie/output_folder/_hsv_bg_2tier_trim2/"
GOOD_IMAGES = "/Volumes/LaCie/output_folder/_excludes/_current_good_images"

# Default is True. If False, it scores good images with exif metadata
MOVE_GOOD_IMAGES = False
GOOD_RATING = 4
if MOVE_GOOD_IMAGES: SUBFOLDER = "good_ids"
else: SUBFOLDER = "not_good_ids"
VERBOSE = False
GOOD_IDS = set()
# open folder and load files

def score_image(image_path, star_rating=GOOD_RATING):
    # with exiftool.ExifTool() as et:
    #     metadata = et.get_metadata(image_path)
    #     rating = metadata.get('XMP:Rating', 0)
    #     return rating

    # Force in-place metadata writes so ExifTool does not create backup files.
    with exiftool.ExifToolHelper(common_args=["-overwrite_original_in_place"]) as et:
        # Set the standard XMP:Rating tag that Adobe Bridge monitors
        et.set_tags(
            files=[image_path],
            tags={"XMP:Rating": star_rating}
        )

    print(f"Successfully applied {star_rating} stars to {image_path}")

def load_files_from_folder(folder):
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            files.append(os.path.join(folder, filename))
    return files

good_filenames = load_files_from_folder(GOOD_IMAGES)
# filenames look like this: X-180-180_Y-180-180_Z-180-180_cc-1_p321_t0_00001_114665785.jpg
# extract the last number before the file extension and store that in GOOD_IDS
for filename in good_filenames:
    base = os.path.basename(filename)
    image_id = base.split("_")[-1].split(".")[0]
    GOOD_IDS.add(image_id)
print(f"Loaded {len(GOOD_IDS)} good image IDs from {GOOD_IMAGES}")
# for each UID in GOOD_IDS, find the file with that UID in the filename and move it to a new folder
# NEW_FOLDER = os.path.join(FOLDER, "good_ids")
# if folder contains folders, walk through them and do the same thing
for root, dirs, files in os.walk(FOLDER):
    for dir in dirs:
        this_new_folder = os.path.join(FOLDER,dir, SUBFOLDER)
        os.makedirs(this_new_folder, exist_ok=True)

        dir_path = os.path.join(root, dir)
        files = load_files_from_folder(dir_path)
        for uid in GOOD_IDS:
            is_found = False
            for filename in files:
                new_path = os.path.join(this_new_folder, os.path.basename(filename))
                if str(uid) in filename:
                    is_found = True
                    try:
                        if MOVE_GOOD_IMAGES:
                            print(f" ✅ Moving {dir}-{uid} to {this_new_folder}")
                            os.rename(filename, new_path)
                        else:
                            print(f"updating metadata for good image {dir}-{uid}")
                            score_image(filename, star_rating=GOOD_RATING)
                    except OSError as e:
                        print(f" ❌ Error occurred while moving {dir}-{uid}: {e}")
                # this isn't working. It recursively goes through every folder, and eventually moves every file.
                # elif str(uid) not in filename and not MOVE_GOOD_IMAGES:
                #     try:
                #         if MOVE_GOOD_IMAGES:
                #             print(f"not moving {dir}-{uid} because it does not match any good image ID")
                #         else:
                #             # check if filename actually exists before trying to move it
                #             if os.path.exists(filename):
                #                 print(f" ☑️ Moving {dir}-{uid} to {this_new_folder}")
                #                 os.rename(filename, new_path)
                #                 is_found = True
                #             elif VERBOSE:
                #                 print(f" ✖️ File {dir}-{uid} does not exist, skipping move") 
                #     except OSError as e:
                #         print(f" ❌❌ Error occurred while moving {dir}-{uid}: {e}")
                if is_found:
                    break