import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import tqdm
from PIL import Image

from tensorrt_model import UnetTensorrt
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
        "-tep",
        "--tensorrt-engine-path",
        help="Path to tensorrt engine file",
        default="/home/sashank/data/blade-load-segmentation/000-001/checkpoints/checkpoint_epoch100.trt",
        type=Path,
    )
    parser.add_argument(
        "-if", "--input-folder", help="folder containing images", type=Path, default="/home/sashank/data/blade-load-segmentation/d5-2"
    )
    parser.add_argument(
        "-of", "--output-folder", help="folder to save images", type=Path, default="/home/sashank/data/blade-load-segmentation/d5-predictions"
    )
    args = parser.parse_args()
    model = UnetTensorrt(args.tensorrt_engine_path)
    image_files = [x.name for x in Path(args.input_folder).glob("*.jpg")]
    image_files.sort(key=image_sort_key)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(exist_ok=True)
    img = Image.open(Path(args.input_folder) / image_files[0])
    img = np.asarray(img)
    mask = model(img)
    img = model.preprocess(img)
    img = img.transpose((1, 2, 0))
    img = img * 255
    img = img.astype(np.uint8)
    img = plot_img_and_mask(img, mask[0, :, :], returns_img=True)
    width, height, _ = img.shape
    #MP4 video writer
    output_pth = output_dir / (Path(args.input_folder).name + ".mp4")
    video_writer = cv2.VideoWriter(str(output_pth), cv2.VideoWriter_fourcc(*"mp4v"), 10, (height, width))
    for image_file in tqdm.tqdm(image_files):
        try:
            img = Image.open(Path(args.input_folder) / image_file)
            img = np.asarray(img)
            mask = model(img)
            img = model.preprocess(img)
            img = img.transpose((1, 2, 0))
            img = img * 255
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = plot_img_and_mask(img, mask[0, :, :], returns_img=True)
            video_writer.write(img)
        except:
            print("Error processing image: {}".format(image_file))
    video_writer.release()
    model.trt_model.cuda_ctx.pop()
