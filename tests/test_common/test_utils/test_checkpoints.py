import unittest

import numpy as np
import torch
import torch.optim as optim
import torchvision.models as models

from code_soup.common.utils.checkpoints import Checkpoints


class TestCheckpoints(unittest.TestCase):
    def test_save_load(self):
        # model_save = models.resnet18()
        # epoch = 10
        # optimizer = optim.SGD(model_save.parameters(), lr=0.01, momentum=0.9)
        # loss = 0.5
        # Checkpoints.save('./test_model.tar',model_save,epoch,optimizer,loss)
        # checkpoint = Checkpoints.load('./test_model.tar')
        # self.assertEqual(checkpoint["model_state_dict"],model_save.state_dict())
        # self.assertEqual(checkpoint["optimizer_state_dict"],optimizer.state_dict())
        # self.assertEqual(checkpoint["epoch"],epoch)
        # self.assertEqual(checkpoint["loss"],loss)
        pass
