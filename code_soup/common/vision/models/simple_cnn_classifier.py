"""Implements a simple CNN Classifier model"""

import torch.nn as nn


class SimpleCnnClassifier(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_labels=10):
        super().__init__()
        self.num_channels = input_shape[0]
        self.image_size = input_shape[1:]
        self.num_labels = num_labels
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_channels, out_channels=32, kernel_size=(3, 3)
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(
                in_features=(((self.image_size[0] - 4) // 2 - 4) // 2)
                * (((self.image_size[1] - 4) // 2 - 4) // 2)
                * 64,
                out_features=200,
            ),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=num_labels),
        )

    def forward(self, x):
        return self.model(x)
