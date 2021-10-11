import unittest

import torch
import torch.nn as nn

from code_soup.common.vision.models import SimpleCnnClassifier


class TestSimpleCnnClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = SimpleCnnClassifier(input_shape=(3, 32, 32), num_labels=10)

    def test_step(self):
        model_input = torch.randn(5, 3, 32, 32)
        model_output = self.model(model_input)
        self.assertEqual(model_output.shape, torch.Size([5, 10]))
