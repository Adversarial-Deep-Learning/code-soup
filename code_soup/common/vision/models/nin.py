from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NIN(nn.Module):
    """
     Following the architecture given in the paper: `Network in Network <https://arxiv.org/pdf/1312.4400.pdf>`_
     Methods
     --------
     forward(x)
     - return prediction tensor

    _initialize_weights(self)
     - Initializes the model with weights ans bias

     step(self, data)
     - Iterates the model for a single batch of data
    """

    def __init__(
        self, input_size: int, n_classes: int, device: torch.device, lr: float
    ):
        """
        Parameters
        ----------
        input_size : int
            Number of input dimensions aka side length of image
        n_classes: int
            Number of classes
        device : torch.device
             Device to run the model on
        lr : float
             Learning rate
        """
        super(NIN, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.device = device
        self.label = 1.0
        self.lr = lr
        self.criterion = torch.nn.BCELoss()
        self.conv1 = nn.Conv2d(input_size, 192, 5, padding=2)
        self.conv2 = nn.Conv2d(192, 160, 1)
        self.conv3 = nn.Conv2d(160, 96, 1)
        self.conv4 = nn.Conv2d(96, 192, 5, padding=6)
        self.conv5 = nn.Conv2d(192, 192, 1)
        self.conv6 = nn.Conv2d(192, 192, 1)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=6)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, n_classes, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self._initialize_weights()

    # forward pass of the model
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        Returns
        -------
        output : torch.Tensor
            Generated sample
        """
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_pool_out = F.max_pool2d(
            conv3_out, kernel_size=3, stride=2, ceil_mode=True
        )
        conv3_drop_out = F.dropout(conv3_pool_out, 0.5)

        conv4_out = F.relu(self.conv4(conv3_drop_out))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_pool_out = F.avg_pool2d(conv6_out, 1)
        conv6_drop_out = F.dropout(conv6_pool_out, 0.5)

        conv7_out = F.relu(self.conv7(conv6_drop_out))
        conv8_out = F.relu(self.conv8(conv7_out))
        conv9_out = F.relu(self.conv9(conv8_out))
        pool_out = F.adaptive_avg_pool2d(conv9_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

    def _initialize_weights(self):
        """
        Initializes the model with weights and bias
        Conv layers get random weights from a normal distribution and bias is set to 0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()

    def step(self, data: torch.Tensor) -> Tuple:
        """
        Iterates the model for a single batch of data, calculates the loss and updates the model parameters.
        Parameters
        ----------
        data : torch.Tensor
            Batch of data
        Returns
        -------
            avg_out:
            The average output (across the batch) of the model
        """
        image, _ = data
        image = image.to(self.device)
        batch_size = image.shape[0]
        label = torch.full(
            (batch_size,), self.label, dtype=torch.float, device=self.device
        )
        self.zero_grad()
        # Forward pass
        output = self(image).view(-1)
        # Calculate loss on a batch
        err = self.criterion(output, label)
        err.backward()
        avg_out = output.mean().item()
        self.optimizer.step()
        return avg_out
