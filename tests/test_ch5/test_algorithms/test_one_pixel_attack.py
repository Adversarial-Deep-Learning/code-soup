import unittest

import torch
import torch.nn as nn

from code_soup.ch5 import OnePixelAttack


class TestOnePixelAttack(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model_to_attack = nn.Sequential(nn.Flatten(), nn.Linear(2 * 2 * 3, 10)).cpu()
        cls.model = OnePixelAttack(model=model_to_attack)

    def test_step(self):
        self.model.step(
            [torch.randn(4, 3, 2, 2).cpu(), torch.zeros(4).cpu()],
            ["car"],
            pixels_perturbed=1,
            targeted=False,
            maxiter=1,
            popsize=1,
            verbose=False,
        )
