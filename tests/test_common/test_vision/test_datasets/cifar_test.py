import unittest
from pathlib import Path

import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from code_soup.common.vision.datasets import CIFARDataset


class TestCIFARDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cifar_dataset = CIFARDataset(
            transform=transforms.Compose([transforms.ToTensor()])
        )
        cifar_dataloader = DataLoader(cifar_dataset, batch_size=64, shuffle=False)
        cls.samples = next(iter(cifar_dataloader))

    def test_image_tensor_dimensions(self):
        image_tensor_shape = TestCIFARDataset.samples[0].shape
        self.assertEqual(image_tensor_shape[0], 64)
        self.assertEqual(image_tensor_shape[1], 3)
        self.assertEqual(image_tensor_shape[2], 32)
        self.assertEqual(image_tensor_shape[3], 32)

    def test_image_label_correctness(self):
        image_label = TestCIFARDataset.samples[1][0]
        self.assertEqual(int(image_label), 6)
