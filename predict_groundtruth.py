import argparse
import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from PIL import Image

from predict import predict_img
from unet import UNet
from utils.data_loading import BasicDataset
from utils.utils import plot_img_and_maskv2 as plot_img_and_mask


def image_sort_key(image_name: str) -> int:
    """Sort images by names
    In this example, we sort images by timestamp.

    Args:
        image_name (str): The file name.

    Returns:
        int: timestamp
    """
    return int(os.path.splitext(image_name)[0][:-5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "-gt",
        "--ground-truth",
        help="Path to groundtruth folder",
        default="/home/sashank/data/blade-load-segmentation/000-001/checkpoints/checkpoint_epoch100.trt",
        type=Path,
    )
    parser.add_argument(
        "-if",
        "--input-folder",
        help="folder containing images",
        type=Path,
        default="/home/sashank/data/blade-load-segmentation/d5-2",
    )
    parser.add_argument(
        "-of",
        "--output-folder",
        help="folder to save images",
        type=Path,
        default="/home/sashank/data/blade-load-segmentation/d5-predictions",
    )
    args = parser.parse_args()
    image_files = [x.name for x in Path(args.input_folder).glob("*left.jpg")]
    image_files.sort(key=image_sort_key)
    # create cv2 mp4 writer
    output_dir = Path(args.output_folder)
    output_dir.mkdir(exist_ok=True)
    output_pth = output_dir / (Path(args.ground_truth).name + ".mp4")
    writer = cv2.VideoWriter(
        str(output_pth), cv2.VideoWriter_fourcc(*"mp4v"), 10, (2560, 720)
    )
    i = 0
    for image_file in tqdm.tqdm(image_files):
        img = Image.open(Path(args.input_folder) / image_file)
        mask_file = Path(args.ground_truth) / (
            "mask_" + image_file.replace(".jpg", ".png")
        )
        mask = Image.open(mask_file).resize(img.size)
        mask = np.asarray(mask)
        img = plot_img_and_mask(img, mask[:, :, 0] * 255, returns_img=True)
        i += 1
        writer.write(img)
    writer.release()
