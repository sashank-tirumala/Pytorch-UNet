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

from predict_aug import predict_img
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


def calc_blade_load(mask):
    empty = 136096
    full = 52200
    cur = np.count_nonzero(mask)
    blade_load = (cur - empty) * 100 / (full - empty)
    return blade_load


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "-cp",
        "--checkpoint-path",
        help="Path to checkpoint file",
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
    model_path = args.checkpoint_path
    n_channels = 3
    n_classes = 2
    bilinear = False
    net = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {model_path}")
    logging.info(f"Using device {device}")
    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)
    logging.info("Model loaded!")
    image_files = [x.name for x in Path(args.input_folder).glob("*left.jpg")]
    image_files.sort(key=image_sort_key)
    # create cv2 mp4 writer
    output_dir = Path(args.output_folder)
    output_dir.mkdir(exist_ok=True)
    output_pth = output_dir / (Path(args.input_folder).name + ".mp4")
    writer = cv2.VideoWriter(
        str(output_pth), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1200, 800)
    )
    i = 0
    mask_e = Image.open("mask_empty.png").resize((512, 512))
    mask_e = np.asarray(mask_e) / 255  # 136096
    mask_f = Image.open("mask_full.png").resize((512, 512))
    mask_f = np.asarray(mask_f) / 255  # 52200
    augmentations = [
        "snow",
        "rain",
        "fog",
        "sun_flare",
        "high_brightness",
        "low_brightness",
    ]
    blade_load_preds = {}
    blade_load_preds["none"] = []
    for aug in augmentations:
        blade_load_preds[aug] = []
    count = 0
    for image_file in tqdm.tqdm(image_files):
        img = Image.open(Path(args.input_folder) / image_file)
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        i = 0
        mask, inp_img = predict_img(
            net=net,
            full_img=img,
            scale_factor=0.5,
            out_threshold=0.5,
            device=device,
            mask_e=mask_e,
            aug="none",
        )
        blade_load = calc_blade_load(mask)
        if count > 10:
            blade_load_preds["none"].append(blade_load)
        fig.suptitle(f"original blade_load: {int(blade_load)}")
        for row in range(2):
            for col in range(3):
                aug = augmentations[i]

                mask, inp_img = predict_img(
                    net=net,
                    full_img=img,
                    scale_factor=0.5,
                    out_threshold=0.5,
                    device=device,
                    mask_e=mask_e,
                    aug=aug,
                )

                blade_load = calc_blade_load(mask)
                axs[row - 1, col].set_title(f"{aug}:{int(blade_load)}")
                out_img = plot_img_and_mask(inp_img, mask, returns_img=True)
                out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                axs[row - 1, col].imshow(
                    out_img,
                )
                if count > 10:
                    blade_load_preds[aug].append(blade_load)

                i += 1

        plt.tight_layout()
        plt.savefig("/tmp/fig.png")
        plt.close()
        img = cv2.imread("/tmp/fig.png")
        writer.write(img)
        count += 1
        if count == 10:
            # break
            pass
    #     for aug in augmentations:
    #         mask, inp_img = predict_img(
    #         net=net,
    #         full_img=img,
    #         scale_factor=0.5,
    #         out_threshold=0.5,
    #         device=device,
    #         mask_e=mask_e,
    #         aug="snow",
    #         )
    #         img = plot_img_and_mask(inp_img, mask, returns_img=True)

    #     i += 1
    #     writer.write(img)
    writer.release()
    blade_load_err = {}
    max_load_error = {}
    for aug in augmentations:
        blade_load_preds[aug] = np.asarray(blade_load_preds[aug])
        avg_error = np.mean(np.abs(blade_load_preds[aug] - blade_load_preds["none"]))
        max_error = np.std(np.abs(blade_load_preds[aug] - blade_load_preds["none"]))
        blade_load_err[aug] = avg_error
        max_load_error[aug] = max_error

    # Sort keys to have 'none' at the left-most position
    sorted_keys = np.array([k for k in blade_load_err.keys() if k != "none"])

    # Get values in the same sorted order
    sorted_values = np.array([blade_load_err[k] for k in sorted_keys])
    sorted_max_error = np.array([max_load_error[k] for k in sorted_keys])

    import seaborn as sns

    # Seaborn style
    sns.set(style="whitegrid")

    # Custom colors
    colors = ["#FFCCCB", "#FFDAB9", "#E6E6FA", "#D8BFD8", "#B0E0E6", "#98FB98"]

    # Initialize the plot

    # Create the bar chart
    # barplot = sns.barplot(x=sorted_keys, y=sorted_values, palette=colors)
    # breakpoint()
    print(f"sorted_values: {sorted_values}, shape: {len(sorted_values)}")
    print(f"sorted_max_error: {sorted_max_error}, shape: {len(sorted_max_error)}")
    print(f"sorted_keys: {sorted_keys}, shape: {len(sorted_keys)}")
    # barplot = sns.barplot(
    #     x=sorted_keys,
    #     y=sorted_values,
    #     palette=colors,
    #     yerr=sorted_max_error,
    #     capsize=0.2,
    # )
    # # ax = sns.barplot(x="Augmentation", y="Average Error", data=df, estimator=np.mean, ci=85, capsize=.2, color='lightblue')

    # # Add labels and title
    # # plt.xlabel("Augmentation", fontsize=14)
    # # plt.ylabel("Average Error", fontsize=14)
    # plt.title("Average Error by Augmentation Type", fontsize=16)

    # # Annotate bars with the actual values
    # for p in barplot.patches:
    #     barplot.annotate(
    #         f"{p.get_height()}",
    # #         (p.get_x() + p.get_width() / 2.0, p.get_height()),
    #         ha="center",
    #         va="baseline",
    #         fontsize=12,
    #         color="black",
    #         xytext=(0, 5),
    #         textcoords="offset points",
    #     )

    # Remove the top and right spines for better aesthetics
    # sns.despine()
    # Create the bar chart
    fig, ax = plt.subplots()
    bars = ax.bar(
        sorted_keys,
        sorted_values,
        color=["#FFCCCB", "#FFDAB9", "#E6E6FA", "#D8BFD8", "#B0E0E6", "#98FB98"],
        yerr=sorted_max_error,
        capsize=5,
    )

    # Add labels and title
    ax.set_xlabel("Augmentation")
    ax.set_ylabel("Average Error")
    ax.set_title("Average Error by Augmentation Type")

    # Show plot
    plt.savefig("avg_error.png")
    breakpoint()
