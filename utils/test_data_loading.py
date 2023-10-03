import os
import unittest
from pathlib import Path

import numpy as np
import torch
from data_loading import BasicDataset  # Import your BasicDataset class
from PIL import Image


class TestBasicDataset(unittest.TestCase):
    def setUp(self):
        # Create some dummy image and mask directories
        self.img_dir = "tmp_images"
        self.mask_dir = "tmp_masks"
        Path(self.img_dir).mkdir()
        Path(self.mask_dir).mkdir()
        Image.fromarray(np.uint8(np.random.randint(0, 255, (720, 1280, 3)))).save(
            f"{self.img_dir}/img1.png"
        )
        Image.fromarray(np.uint8(np.random.randint(0, 1, (720, 1280)))).save(
            f"{self.mask_dir}/mask_img1.png"
        )

    def test_initialization(self):
        dataset = BasicDataset(self.img_dir, self.mask_dir, scale=1.0)
        self.assertIsInstance(dataset, BasicDataset)

    def test_invalid_scale(self):
        with self.assertRaises(AssertionError):
            BasicDataset(self.img_dir, self.mask_dir, scale=0)

    def test_empty_directory(self):
        with self.assertRaises(RuntimeError):
            BasicDataset("nonexistent", "nonexistent", scale=1.0)

    def test_length(self):
        dataset = BasicDataset(self.img_dir, self.mask_dir, scale=1.0)
        self.assertEqual(len(dataset), 1)

    def test_get_item(self):
        dataset = BasicDataset(self.img_dir, self.mask_dir, scale=0.5)
        item = dataset[0]
        self.assertTrue(torch.is_tensor(item["image"]))
        self.assertTrue(torch.is_tensor(item["mask"]))

    def test_fixed_scale(self):
        dataset = BasicDataset(self.img_dir, self.mask_dir, fixed_scale=512)
        item = dataset[0]
        self.assertEqual(item["image"].shape, (3, 512, 512))
        self.assertEqual(item["mask"].shape, (512, 512))

    def test_scale(self):
        dataset = BasicDataset(self.img_dir, self.mask_dir, scale=0.5)
        item = dataset[0]
        self.assertEqual(item["image"].shape, (3, 360, 640))
        self.assertEqual(item["mask"].shape, (360, 640))

    def test_augmentation_v1(self):
        dataset = BasicDataset(
            self.img_dir, self.mask_dir, fixed_scale=512, scale=0.5, augmentation="v1"
        )
        item = dataset[0]
        self.assertEqual(item["image"].shape, (3, 512, 512))
        self.assertEqual(item["mask"].shape, (512, 512))

    def tearDown(self):
        # Clean up the dummy directories
        os.remove(f"{self.img_dir}/img1.png")
        os.remove(f"{self.mask_dir}/mask_img1.png")
        os.rmdir(self.img_dir)
        os.rmdir(self.mask_dir)


if __name__ == "__main__":
    unittest.main()
