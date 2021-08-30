import random

import numpy as np
import torch


class Seeding:
    seed = 42

    @classmethod
    def set_seeding(self, seed):
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
