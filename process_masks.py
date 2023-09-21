import os

import cv2

# Get a list of all files in the current directory
files = os.listdir('/home/sashank/data/blade-load-segmentation/d5-luyuan/masks')

# Loop through the files
for f in files:
    file = os.path.join('/home/sashank/data/blade-load-segmentation/d5-luyuan/masks', f)
    img = cv2.imread(file)
    img = cv2.resize(img, (1280, 720))
    cv2.imwrite(os.path.join('/home/sashank/data/blade-load-segmentation/d5-luyuan/mask_proc', f), img)