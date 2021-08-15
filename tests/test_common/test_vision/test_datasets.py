import unittest

import torchvision
from parameterized import parameterized_class
from torch.utils.data import DataLoader
from torchvision import transforms

from code_soup.common.vision.datasets import ImageClassificationDataset


@parameterized_class(
    ("dataset_class", "expected_size", "expected_label"),
    [
        (torchvision.datasets.MNIST, (16, 1, 28, 28), 5),
        (torchvision.datasets.CIFAR10, (12, 3, 32, 32), 6),
    ],
)
class TestVisionDataset(unittest.TestCase):
    """Vision Dataset Parameterized TestCase

    Args: ("dataset_class", "expected_size", "expected_label")
    """

    def setUp(self):
        self.TestDataset = ImageClassificationDataset(
            self.dataset_class, transform=transforms.Compose([transforms.ToTensor()])
        )
        self.TestDatasetLoader = DataLoader(
            self.TestDataset, batch_size=self.expected_size[0], shuffle=False
        )
        self.samples = next(iter(self.TestDatasetLoader))

    def test_image_tensor_dimensions(self):
        self.assertTupleEqual(self.samples[0].size(), self.expected_size)

    def test_image_label_correctness(self):
        self.assertEqual(self.samples[1][0], self.expected_label)
