import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dims):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dims, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)
