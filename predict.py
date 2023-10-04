import argparse
import logging
import os

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_coeff, dice_loss, multiclass_dice_coeff
from utils.utils import plot_img_and_maskv2 as plot_img_and_mask


def predict_img(
    net, full_img, device, scale_factor=1, out_threshold=0.5, fixed_scale=512
):
    net.eval()
    img = BasicDataset.preprocess(
        None, full_img, scale_factor, is_mask=False, fixed_scale=fixed_scale
    )

    # transform = A.Compose([A.functional.add_snow()])
    # img = A.functional.add_snow(img, snow_point=0.1, brightness_coeff=2.5)
    # img = A.functional.add_fog(img, fog_coef=0.4, alpha_coef=0.08, haze_list=())
    # height, width = img.shape[:2]
    # area = height * width
    # num_drops = area // 770
    # slant = 6
    # rain_drops = []
    # drop_length = 30

    # for _i in range(num_drops):  # If You want heavy rain, try increasing this
    #     if slant < 0:
    #         x = np.random.randint(slant, width)
    #     else:
    #         x = np.random.randint(0, width - slant)

    #     y = np.random.randint(0, height - drop_length)

    #     rain_drops.append((x, y))
    # img = A.functional.add_rain(
    #     img,
    #     slant=slant,
    #     drop_length=drop_length,
    #     drop_width=1,
    #     drop_color=(200, 200, 200),
    #     blur_value=5,
    #     brightness_coefficient=0.6,
    #     rain_drops=rain_drops,
    # )
    # RANDOMSUNFLARE
    # height, width = img.shape[:2]
    # flare_center_x = int(0.7 * width)
    # flare_center_y = int(0.2 * height)
    # angle = 0.7
    # num_circles = 8
    # circles = []

    # x = []
    # y = []

    # def line(t):
    #     return (
    #         flare_center_x + t * np.math.cos(angle),
    #         flare_center_y + t * np.math.sin(angle),
    #     )

    # for t_val in range(-flare_center_x, width - flare_center_x, 10):
    #     rand_x, rand_y = line(t_val)
    #     x.append(rand_x)
    #     y.append(rand_y)

    # for _i in range(num_circles):
    #     alpha = np.random.uniform(0.05, 0.2)
    #     r = np.random.randint(0, len(x) - 1)
    #     rad = np.random.randint(1, max(height // 100 - 2, 2))

    #     r_color = np.random.randint(max(255 - 50, 0), 255)
    #     g_color = np.random.randint(max(255 - 50, 0), 255)
    #     b_color = np.random.randint(max(255 - 50, 0), 255)

    #     circles += [
    #         (
    #             alpha,
    #             (int(x[r]), int(y[r])),
    #             pow(rad, 3),
    #             (r_color, g_color, b_color),
    #         )
    #     ]
    # img = A.functional.add_sun_flare(
    #     img,
    #     flare_center_x,
    #     flare_center_y,
    #     400,
    #     (255, 255, 255),
    #     circles,
    # )
    img = A.functional.adjust_brightness_torchvision(img, 1.5)
    # img = transform(image=img)["image"]
    inp_img = img.copy()
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2, 0, 1))
    if img.max() > 1:
        img = img / 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        # output = F.interpolate(
        #     output, (full_img.size[1], full_img.size[0]), mode="bilinear"
        # )
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
    mask = mask[0].long().squeeze().numpy()
    mask = mask[:, :].astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((71, 71), np.uint8))

    return mask, inp_img


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--model",
        "-m",
        default="MODEL.pth",
        metavar="FILE",
        help="Specify the file in which the model is stored",
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="INPUT",
        nargs="+",
        help="Filenames of input images",
        required=True,
    )
    parser.add_argument(
        "--output", "-o", metavar="OUTPUT", nargs="+", help="Filenames of output images"
    )
    parser.add_argument(
        "--viz",
        "-v",
        action="store_true",
        help="Visualize the images as they are processed",
    )
    parser.add_argument(
        "--no-save", "-n", action="store_true", help="Do not save the output masks"
    )
    parser.add_argument(
        "--mask-threshold",
        "-t",
        type=float,
        default=0.5,
        help="Minimum probability value to consider a mask pixel white",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=0.5,
        help="Scale factor for the input images",
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    parser.add_argument(
        "--classes", "-c", type=int, default=2, help="Number of classes"
    )

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f"{os.path.splitext(fn)[0]}_OUT.png"

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros(
            (mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8
        )
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {args.model}")
    logging.info(f"Using device {device}")

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)

    logging.info("Model loaded!")

    for i, filename in enumerate(in_files):
        logging.info(f"Predicting image {filename} ...")
        img = Image.open(filename)

        mask = predict_img(
            net=net,
            full_img=img,
            scale_factor=args.scale,
            out_threshold=args.mask_threshold,
            device=device,
        )

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f"Mask saved to {out_filename}")

        if args.viz:
            logging.info(
                f"Visualizing results for image {filename}, close to continue..."
            )
            plot_img_and_mask(img, mask)
