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
    Model class for test_save function
    """

    def __init__(self, input_length: int):
        super(TheModelClass, self).__init__()
        self.dense = nn.Linear(int(input_length), 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))


class TestCheckpoints(unittest.TestCase):
    def test_save(self):
        """
        Test that the model is saved
        """
        model_save = TheModelClass(7)
        optimizer = optim.SGD(model_save.parameters(), lr=0.01, momentum=0.9)
        loss = 0.5
        epoch = 10
        Checkpoints.save(
            "./input/test_model.pth",
            model_save,
            optimizer,
            epoch,
            loss,
        )
        self.assert_(os.path.isfile("./input/test_model.pth"))

    def test_load(self):
        """
        Test that the model is loaded
        """
        model_load = models.resnet18(pretrained=True)
        optimizer = optim.SGD(model_load.parameters(), lr=0.01, momentum=0.9)
        loss = 0.5
        epoch = 10
        Checkpoints.save(
            "./input/test_model.pth",
            model_load,
            optimizer,
            epoch,
            loss,
        )
        model = models.resnet18()
        model = Checkpoints.load("./input/test_model.pth")
        self.assertEqual(list(model.state_dict()), list(model_load.state_dict()))
