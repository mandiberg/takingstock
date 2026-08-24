'''
This script will open a folder of labels 
It will look through each line of each label and check if it is 4 coordinates or 8 coordinates
If it is 8 coordinates, it will convert it to 4 coordinates and save the label
'''

import os


LABELS_FOLDER = "/Users/michaelmandiberg/Documents/yolo/sorted_images_balls_yoga/misc_balls_detect_round2/labels"

def convert_8_to_4_coordinates(x1, y1, x2, y2, x3, y3, x4, y4):
    # convert from x,y coord to YOLO format (x_center, y_center, width, height)
    x_min = min(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    x_max = max(x1, x2, x3, x4)
    y_max = max(y1, y2, y3, y4)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return x_center, y_center, width, height

# open folder and load files
files = []
for filename in os.listdir(LABELS_FOLDER):
    if filename.endswith(".txt"):
        files.append(os.path.join(LABELS_FOLDER, filename))

for filename in files:
    with open(filename, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 9:
            # 8 coordinates, convert to 4 coordinates
            class_id = parts[0]
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[1:])
            x_center, y_center, width, height = convert_8_to_4_coordinates(x1, y1, x2, y2, x3, y3, x4, y4)
            new_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
            new_lines.append(new_line)
            print(f"Converted 8 coordinates to 4 coordinates in {filename}: {line.strip()} -> {new_line.strip()}")
        else:
            # already 4 coordinates
            new_lines.append(line)
    
    with open(filename, "w") as f:
        f.writelines(new_lines)