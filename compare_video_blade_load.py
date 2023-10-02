import argparse
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image

from tensorrt_model import UnetTensorrt


def image_sort_key(image_name: str) -> int:
    """Sort images by names
    In this example, we sort images by timestamp.

    Args:
        image_name (str): The file name.

    Returns:
        int: timestamp
    """
    return int(os.path.splitext(image_name)[0][:-5])


def plot_img_masks(model1, model2, img, model1_str="", model2_str="", returns_img=False):
    """
    Overlay mask on image with light and dark colours using cv2
    """
    mask1 = model1(img)[0, :, :]
    mask1 = 1 - mask1
    mask2 = model2(img)[0, :, :]
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((23, 23), np.uint8))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, np.ones((23, 23), np.uint8))
    mask1 = np.dstack((mask1, mask1, mask1)) * np.array([[0, 255, 0]])
    mask1 = mask1.astype(np.uint8)
    mask2 = np.dstack((mask2, mask2, mask2)) * np.array([[0, 255, 0]])
    mask2 = mask2.astype(np.uint8)
    img = np.array(img)
    img = model1.preprocess(img)
    img = img.transpose((1, 2, 0))
    img = img * 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    weighted_sum1 = cv2.addWeighted(mask1, 0.25, img, 0.75, 0.0)
    weighted_sum2 = cv2.addWeighted(mask2, 0.25, img, 0.75, 0.0)
    cv2.putText(weighted_sum1, model1_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(weighted_sum2, model2_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    img = np.concatenate((weighted_sum1, weighted_sum2), axis=1)
    if returns_img:
        return img
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "-tep1",
        "--tensorrt-engine-path1",
        help="Path to tensorrt engine file",
        default="/home/sashank/data/blade-load-segmentation/000-001/checkpoints/checkpoint_epoch100.trt",
        type=Path,
    )
    parser.add_argument(
        "-tep2",
        "--tensorrt-engine-path2",
        help="Path to tensorrt engine file",
        default="/home/sashank/data/blade-load-segmentation/000-001/checkpoints/checkpoint_epoch100.trt",
        type=Path,
    )
    parser.add_argument(
        "-tep1_str",
        "--tensorrt-engine-path1-str",
        help="String to display on image",
        default="Model 1",
    )
    parser.add_argument(
        "-tep2_str",
        "--tensorrt-engine-path2-str",
        help="String to display on image",
        default="Model 2",
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
    model_left = UnetTensorrt(args.tensorrt_engine_path1)
    model_right = UnetTensorrt(args.tensorrt_engine_path2)
    image_files = [x.name for x in Path(args.input_folder).glob("*.jpg")]
    image_files.sort(key=image_sort_key)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(exist_ok=True)
    img = Image.open(Path(args.input_folder) / image_files[0])
    img = np.asarray(img)
    img_out = plot_img_masks(
        model_left, model_right, img, args.tensorrt_engine_path1_str, args.tensorrt_engine_path2_str, returns_img=True
    )
    width, height, _ = img_out.shape
    # MP4 video writer
    output_pth = output_dir / (
        Path(args.input_folder).name
        + "_compare_"
        + args.tensorrt_engine_path1.name
        + "_"
        + args.tensorrt_engine_path2.name
        + ".mp4"
    )
    video_writer = cv2.VideoWriter(str(output_pth), cv2.VideoWriter_fourcc(*"mp4v"), 10, (height, width))
    for image_file in tqdm.tqdm(image_files):
        img = Image.open(Path(args.input_folder) / image_file)
        img = np.asarray(img)
        img_out = plot_img_masks(
            model_left,
            model_right,
            img,
            args.tensorrt_engine_path1_str,
            args.tensorrt_engine_path2_str,
            returns_img=True,
        )
        video_writer.write(img_out)
    video_writer.release()
    model_left.trt_model.cuda_ctx.pop()
    model_right.trt_model.cuda_ctx.pop()
