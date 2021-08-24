import random
import unittest

import numpy as np
import torch
from torchvision.datasets.fakedata import FakeData
from torchvision.transforms import ToTensor

from code_soup.common import VisualPerturbation


class TestVisualPerturbation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        df = FakeData(size=2, image_size=(3, 64, 64))
        a, b = tuple(df)
        a, b = ToTensor()(a[0]).unsqueeze_(0), ToTensor()(b[0]).unsqueeze_(0)
        cls.obj_tensor = VisualPerturbation(original=a, perturbed=b)
        cls.obj_numpy = VisualPerturbation(original=a.numpy(), perturbed=b.numpy())

    def test_LPNorm(self):
        self.assertAlmostEqual(
            TestVisualPerturbation.obj_tensor.calculate_LPNorm(p=1), 4143.0249, places=3
        )
        self.assertAlmostEqual(
            TestVisualPerturbation.obj_numpy.calculate_LPNorm(p="fro"),
            45.6525,
            places=3,
        )

    def test_PSNR(self):
        self.assertAlmostEqual(
            TestVisualPerturbation.obj_tensor.calculate_PSNR(),
            33.773994480876496,
            places=3,
        )

    def test_RMSE(self):
        self.assertAlmostEqual(
            TestVisualPerturbation.obj_tensor.calculate_RMSE(),
            0.018409499898552895,
            places=3,
        )

    def test_SAM(self):
        self.assertAlmostEqual(
            TestVisualPerturbation.obj_tensor.calculate_SAM(),
            89.34839413786915,
            places=3,
        )

    def test_SRE(self):
        self.assertAlmostEqual(
            TestVisualPerturbation.obj_tensor.calculate_SRE(),
            41.36633261587073,
            places=3,
        )
