import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_img_and_maskv2(img, mask, returns_img=False):
    """
    Overlay mask on image with light and dark colours using cv2
    """
    mask = np.dstack((mask, mask, mask)) * np.array([[0, 255, 0]])
    mask = mask.astype(np.uint8)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    weighted_sum = cv2.addWeighted(mask, 0.25, img, 0.75, 0.)
    img = np.concatenate((img, weighted_sum), axis=1)
    if returns_img:
        return img
    plt.imshow(img)
    plt.show()

def plot_img_and_maskv3(img, mask, returns_img=False, max_mask_count = 96324, min_mask_count = 42428):
    """
    Overlay mask on image with light and dark colours using cv2
    """
    mask = mask.astype(np.uint8)
    img = np.array(img)
    mask_count = np.sum(mask == 0)
    mask_percentage = (mask_count - min_mask_count )*100 / (max_mask_count - min_mask_count)
    #Write mask percent top left into image
    cv2.putText(img, f"Blade Load Percent: {100 - int(mask_percentage)}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if returns_img:
        return img
    else:
        plt.imshow(img)
        plt.show()
