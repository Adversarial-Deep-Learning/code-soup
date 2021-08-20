import unittest

import torch
import torch.nn as nn

from code_soup.common.vision.models.inceptionv3 import Inception3


class TestInception3(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Inception3(num_classes=10, aux_logits=True, transform_input=True)

    def test_step(self):
        self.model.step([torch.randn(32, 32, 3, 3), torch.ones(4)])
