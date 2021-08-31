import unittest

import numpy as np

from code_soup.common.utils.seeding import Seeding


class TestTorchSeed(unittest.TestCase):
    """Test the torch_seed function."""

    def test_torch_seed(self):
        """Test that the torch seed is set."""
        Seeding.set_seeding(42)
        self.assertEqual(np.random.get_state()[1][0], 42)
