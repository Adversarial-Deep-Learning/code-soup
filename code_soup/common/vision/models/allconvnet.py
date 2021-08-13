from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AllConvNet(nn.Module):
    """
    Following the architecture given in the paper: `All Convolutional Network <https://arxiv.org/abs/1412.6806>`_
    Methods
    --------
    forward(x)
    - return prediction tensor

    step(self, data)
    - Iterates the model for a single batch of data
    """

    def __init__(
        self, image_size: int, n_classes: int, device: torch.device, lr: float
    ):
        """
        Parameters
        ----------
        image_size : int
            Number of input dimensions aka side length of image
        n_classes: int
            Number of classes
        device : torch.device
             Device to run the model on
        lr : float
             Learning rate
        """
        super(AllConvNet, self).__init__()
        self.image_size = image_size
        self.n_classes = n_classes
        self.criterion = torch.nn.BCELoss()
        self.label = 1.0
        self.device = device
        self.lr = lr
        # Constructing the model as per the paper
        self.conv1 = nn.Conv2d(image_size, 96, 3, padding=2)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=2)
        self.conv3 = nn.MaxPool2d(3, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=2)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=2)
        self.conv6 = nn.MaxPool2d(3, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=2)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(10, 10, 1)
        self.class_conv = nn.Conv2d(192, n_classes, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    # Forward pass of the model
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
        conv4_out = F.relu(self.conv4(conv3_out))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv7_out = F.relu(self.conv7(conv6_out))
        conv8_out = F.relu(self.conv8(conv7_out))
        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out = F.softmax(pool_out.squeeze_(-1))
        return pool_out

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
