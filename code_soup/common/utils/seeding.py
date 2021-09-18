import random

import numpy as np
import torch


class Seeding:
    """
    A class used for seeding

    Class Variables
    ---------------
    value
        - to store value of seed

    Class Methods
    -------------
    seed(self, value)
        -Set random seed for everything
    """

    value = 42

    @classmethod
    def seed(self, value):
        self.value = value
        np.random.seed(self.value)
        torch.manual_seed(self.value)
        random.seed(self.value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
