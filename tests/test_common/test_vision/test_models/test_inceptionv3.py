import unittest

import torch
import torch.nn as nn

from code_soup.common.vision.models.inceptionv3 import Inception3


class TestAllConvNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = AllConvNet(
            image_size=299, n_classes=10, device=torch.device("cpu"), lr=0.01
        )

    def test_step(self):
        self.model.step([torch.randn(299, 299, 3, 3), torch.ones(4)])
