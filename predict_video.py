import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from easydict import EasyDict
from PIL import Image
from yaml_config_override import add_arguments

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


if __name__=="__main__":
    conf = EasyDict(add_arguments())
    net = UNet(n_channels=conf.model.n_channels, n_classes=conf.model.n_classes, bilinear=conf.model.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {conf.model.path}')
    logging.info(f'Using device {device}')
    net.to(device=device)
    state_dict = torch.load(conf.model.path, map_location=device)
    state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')
    image_files = [x.name for x in Path(conf.images_dir).glob('*.jpg')]
    image_files.sort(key=image_sort_key)
    # create cv2 mp4 writer
    output_dir = Path(conf.outputs_dir)
    output_dir.mkdir(exist_ok=True)
    output_pth = output_dir / (Path(conf.images_dir).name + ".mp4")
    writer = cv2.VideoWriter(str(output_pth), cv2.VideoWriter_fourcc(*'mp4v'), 10, (2560, 720))
    i = 0
    for image_file in tqdm.tqdm(image_files):
        img = Image.open(Path(conf.images_dir) / image_file)
        mask = predict_img(net=net,
                            full_img=img,
                            scale_factor=0.5,
                            out_threshold=0.5,
                            device=device)
        img = plot_img_and_mask(img, mask, returns_img=True)
        i+=1
        writer.write(img)
    writer.release()