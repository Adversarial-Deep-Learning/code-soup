import unittest

import torch
import torch.nn as nn

from code_soup.common.vision.models import NIN


class TestNIN(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = NIN(input_size=3, n_classes=1, device=torch.device("cpu"), lr=0.01)

    def test_step(self):
        self.model.step([torch.randn(3, 3, 3, 5), torch.ones(4)])
