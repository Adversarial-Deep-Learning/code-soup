import os
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from code_soup.common.utils import Checkpoints


class TheModelClass(nn.Module):
    """
    Model class for tests
    """

    def __init__(self):
        super(TheModelClass, self).__init__()
        self.dense = nn.Linear(2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))


class TestCheckpoints(unittest.TestCase):
    def test_save(self):
        """
        Test that the model is saved
        """
        model_save = TheModelClass()
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
        self.assertTrue(os.path.isfile("tests/test_common/test_utils/test_model.pth"))
        os.remove("tests/test_common/test_utils/test_model.pth")

    def test_load(self):
        """
        Test that the model is loaded
        """
        model = TheModelClass()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss = 0.5
        epoch = 10
        Checkpoints.save(
            "tests/test_common/test_utils/test_model.pth",
            model,
            optimizer,
            epoch,
            loss,
        )
        model_load = Checkpoints.load("tests/test_common/test_utils/test_model.pth")
        self.assertEqual(list(model.state_dict()), list(model_load.state_dict()))
        os.remove("tests/test_common/test_utils/test_model.pth")
