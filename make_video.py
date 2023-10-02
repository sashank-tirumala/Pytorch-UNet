import argparse
import os
from pathlib import Path

import cv2
import tqdm
from PIL import Image


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
        "-if",
        "--input-folder",
        help="folder containing images",
        type=Path,
        default="/home/sashank/data/blade-load-segmentation/d5-2",
    )
    parser.add_argument(
        "-of",
        "--output-file",
        help="Output file name of video ",
        type=Path,
        default="/home/sashank/data/blade-load-segmentation/d5-predictions",
    )
    args = parser.parse_args()
    image_files = [x.name for x in Path(args.input_folder).glob("*left.jpg")]
    image_files.sort(key=image_sort_key)
    # create cv2 mp4 writer
    output_dir = Path(args.output_file.parent)
    output_dir.mkdir(exist_ok=True)
    output_pth = args.output_file
    first_image = cv2.imread(str(Path(args.input_folder) / image_files[0]))
    height, width, _ = first_image.shape
    writer = cv2.VideoWriter(str(output_pth), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    i = 0
    for image_file in tqdm.tqdm(image_files):
        img = cv2.imread(str(Path(args.input_folder) / image_file))
        # Convert from BGR to RGB if necessary
        if img is not None and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.write(img)
    writer.release()
