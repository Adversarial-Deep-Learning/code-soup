import os
import unittest

import numpy as np
import torch
import torch.optim as optim
import torchvision.models as models
from torch.nn.modules.module import T

from code_soup.common.utils.checkpoints import Checkpoints


class TestCheckpoints(unittest.TestCase):
    def test_save(self):
        """
        Test that the model is saved
        """
        model_save = models.resnet18(pretrained=True)
        optimizer = optim.SGD(model_save.parameters(), lr=0.01, momentum=0.9)
        loss = 0.5
        epoch = 10
        Checkpoints.save(
            "tests/test_common/test_utils/test_model.pth",
            model_save,
            optimizer,
            epoch,
            loss,
        )
        self.assert_(os.path.isfile("tests/test_common/test_utils/test_model.pth"))

    def test_load(self):
        """
        Test that the model is loaded
        """
        model = models.resnet18()
        model = Checkpoints.load("tests/test_common/test_utils/test_model.pth")
        model_load = models.resnet18(pretrained=True)
        self.assertEqual(list(model.state_dict()), list(model_load.state_dict()))
