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
    weighted_sum = cv2.addWeighted(mask, 0.25, img, 0.75, 0.)
    img = np.concatenate((img, weighted_sum), axis=1)
    if returns_img:
        return img
    plt.imshow(img)
    plt.show()
