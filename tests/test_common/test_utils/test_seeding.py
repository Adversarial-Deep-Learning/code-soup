import random
import unittest

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToTensor

from code_soup.common.utils.seeding import Seeding


class TestSeeding(unittest.TestCase):
    """Test the seed function."""

    def test_seed(self):
        """Test that the seed is set."""
        Seeding.seed(42)
        self.assertEqual(np.random.get_state()[1][0], 42)
        self.assertEqual(torch.get_rng_state().tolist()[0], 42)
