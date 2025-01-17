import logging
from functools import lru_cache, partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == ".npy":
        return Image.fromarray(np.load(filename))
    elif ext in [".pt", ".pth"]:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    try:
        mask_file = list(mask_dir.glob("mask_" + idx + ".*"))[0]
    except IndexError:
        raise FileNotFoundError(
            f"Mask file not found for ID {idx} in {mask_dir} with suffix {mask_suffix}"
        )
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(
            f"Loaded masks should have 2 or 3 dimensions, found {mask.ndim}"
        )


class BasicDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        mask_dir: str,
        scale: float = 1.0,
        mask_suffix: str = "",
        fixed_scale: int = None,
        augmentation: str = None,
    ):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        if not (self.images_dir.is_dir() and self.mask_dir.is_dir()):
            raise RuntimeError(
                f"Invalid dataset directories: {images_dir} or {mask_dir} are not valid directories"
            )
        assert 0 < scale <= 1, "Scale must be between 0 and 1"
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith(".")
        ]
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )

        logging.info(f"Creating dataset with {len(self.ids)} examples")
        logging.info("Scanning mask files to determine unique values")
        with Pool() as p:
            unique = list(
                tqdm(
                    p.imap(
                        partial(
                            unique_mask_values,
                            mask_dir=self.mask_dir,
                            mask_suffix=self.mask_suffix,
                        ),
                        self.ids,
                    ),
                    total=len(self.ids),
                )
            )
        self.mask_values = list(
            sorted(np.unique(np.concatenate(unique), axis=0).tolist())
        )
        logging.info(f"Unique mask values: {self.mask_values}")
        self.fixed_scale = fixed_scale
        self.augmentation = augmentation
        self.v1_transform = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.25, contrast_limit=0.4, p=0.5
                ),
                A.RandomSnow(
                    brightness_coeff=2.5,
                    snow_point_lower=0.3,
                    snow_point_upper=0.5,
                    p=0.5,
                ),
            ]
        )
        self.v2_transform = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.25, contrast_limit=0.4, p=0.5
                ),
                A.RandomSnow(
                    brightness_coeff=2.5,
                    snow_point_lower=0.3,
                    snow_point_upper=0.5,
                    p=0.5,
                ),
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, p=0.5),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.5),
                A.RandomRain(
                    brightness_coefficient=0.9, drop_width=1, blur_value=5, p=0.5
                ),
            ]
        )
        self.v3_transform = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.25, contrast_limit=0.4, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.RandomSnow(
                            brightness_coeff=2.5,
                            snow_point_lower=0.3,
                            snow_point_upper=0.5,
                            p=0.5,
                        ),
                        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, p=0.5),
                        A.RandomSunFlare(
                            flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.5
                        ),
                        A.RandomRain(
                            brightness_coefficient=0.9,
                            drop_width=1,
                            blur_value=5,
                            p=0.5,
                        ),
                    ],
                    p=0.5,
                ),
            ],
            p=0.5,
        )
        self.v4_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.25, contrast_limit=0.4, p=0.5
                        ),
                        A.RandomSnow(
                            brightness_coeff=2.5,
                            snow_point_lower=0.3,
                            snow_point_upper=0.5,
                            p=0.5,
                        ),
                        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, p=0.5),
                        A.RandomSunFlare(
                            flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.5
                        ),
                        A.RandomRain(
                            brightness_coefficient=0.9,
                            drop_width=1,
                            blur_value=5,
                            p=0.5,
                        ),
                    ],
                    p=0.95,
                ),
            ],
            p=0.5,
        )

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask, fixed_scale):
        w, h = pil_img.size
        if fixed_scale is None:
            newW, newH = int(scale * w), int(scale * h)
        else:
            newW, newH = int(fixed_scale), int(fixed_scale)
        assert (
            newW > 0 and newH > 0
        ), "Scale is too small, resized images would have no pixel"
        pil_img = pil_img.resize(
            (newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC
        )
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob("mask_" + name + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))

        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {name}: {img_file}"
        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert (
            img.size == mask.size
        ), f"Image and mask {name} should be the same size, but are {img.size} and {mask.size}"

        img = self.preprocess(
            self.mask_values,
            img,
            self.scale,
            is_mask=False,
            fixed_scale=self.fixed_scale,
        )
        mask = self.preprocess(
            self.mask_values,
            mask,
            self.scale,
            is_mask=True,
            fixed_scale=self.fixed_scale,
        )

        if self.augmentation == "v1":
            transformed = self.v1_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        elif self.augmentation == "v2":
            transformed = self.v2_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        elif self.augmentation == "v3":
            transformed = self.v3_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        elif self.augmentation == "v4":
            transformed = self.v4_transform(image=img, mask=mask)
            img = transformed["image"]

        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img / 255.0

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous(),
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix="_mask")
