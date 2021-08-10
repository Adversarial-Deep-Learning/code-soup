import unittest
from pathlib import Path

import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from code_soup.common.vision.datasets import MnistDataset


class TestMnistDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mnist_dataset = MnistDataset(
            transform=transforms.Compose([transforms.ToTensor()])
        )
        mnist_dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=False)
        cls.samples = next(iter(mnist_dataloader))

    def test_image_tensor_dimensions(self):
        image_tensor_shape = TestMnistDataset.samples[0].shape
        self.assertEqual(image_tensor_shape[0], 64)
        self.assertEqual(image_tensor_shape[1], 1)
        self.assertEqual(image_tensor_shape[2], 28)
        self.assertEqual(image_tensor_shape[3], 28)

    def test_image_label_correctness(self):
        image_label = TestMnistDataset.samples[1][0]
        self.assertEqual(image_label, 5)
