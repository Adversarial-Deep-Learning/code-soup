import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dims, output_dims=784):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dims),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1, 28, 28)
