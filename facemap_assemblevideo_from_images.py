import cv2
import os

def get_img_list(folder):
    img_list=[]
    for file in os.listdir(folder):
        if not file.startswith('.') and not file.endswith('.mp4') and os.path.isfile(os.path.join(folder, file)):
            filepath = os.path.join(folder, file)
            filepath=filepath.replace('\\' , '/')
            img_list.append(file)
    return img_list        
    print("got image list")


def write_video(img_array, ROOT, FRAMERATE=15):
    # Check if the ROOT folder exists, create it if not
    if not os.path.exists(ROOT):
        os.makedirs(ROOT)

    # Get the dimensions of the first image in the array
    image_path = os.path.join(ROOT, img_array[0])
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(ROOT, FOLDER+".mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, FRAMERATE, (width, height))

    # Iterate over the image array and write frames to the video
    for filename in img_array:
        image_path = os.path.join(ROOT, filename)
        img = cv2.imread(image_path)
        video_writer.write(img)

    # Release the video writer and close the video file
    video_writer.release()

    print(f"Video saved at: {video_path}")



# img_array = ['image1.jpg', 'image2.jpg', 'image3.jpg']
HOLDER = '/Users/michaelmandiberg/Dropbox/facemap_dropbox/June_tests/'
FRAMERATE = 15
FOLDER = "June4_smilescream_itter_25Ksegment"
ROOT = os.path.join(HOLDER,FOLDER)
list_of_files= get_img_list(ROOT)
print(list_of_files)
list_of_files.sort()

write_video(list_of_files, ROOT, FRAMERATE)
