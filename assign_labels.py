import csv
import os

import cv2

output_folder = '/home/sashank/data/blade-load-segmentation/0915/labeled1/rgb'
output_mask_folder = '/home/sashank/data/blade-load-segmentation/0915/labeled1/masks'
input_folder = '/home/sashank/data/blade-load-segmentation/0915/labeled1/class_label'

# read pre-defined mask
mask_path = '/home/sashank/data/blade-load-segmentation/000-001/masks/mask_1693959721881_left.jpg'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# read csv label file
csv_path = os.path.join(input_folder, 'gt.csv')
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

# remove the first line
lines.pop(0)
for each in lines:
    print(each[1])
    if each[1] == '0':
        # move images to output folder
        os.rename(os.path.join(input_folder, each[0]), os.path.join(output_folder, each[0]))
        # copy mask to output folder
        cv2.imwrite(os.path.join(output_mask_folder, each[0]), mask)