import random
import unittest

import numpy as np
import torch

from code_soup.common.utils import Seeding


class TestSeeding(unittest.TestCase):
    """Test the seed function."""

    def test_seed(self):
        """Test that the seed is set."""
        random.seed(42)
        initial_state = random.getstate()
        Seeding.seed(42)
        final_state = random.getstate()
        self.assertEqual(initial_state, final_state)
        self.assertEqual(np.random.get_state()[1][0], 42)
        self.assertEqual(torch.get_rng_state().tolist()[0], 42)
