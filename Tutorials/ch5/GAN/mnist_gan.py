import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils

from code_soup.ch5.models import GAN
from code_soup.common.vision.datasets import MNISTDataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="mnist_gan.py", description="Train an MNIST GAN model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        action="store",
        help="Specifies batch size of the GAN Trainer",
        default=64,
    )
    parser.add_argument(
        "--latent_dims",
        type=int,
        action="store",
        help="Specifies size of latent vectors for generating noise",
        default=128,
    )
    parser.add_argument(
        "--learning_rate",
        type=int,
        action="store",
        help="Specifies learning rate for training",
        default=0.0002,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        action="store",
        help="Specifies learning rate for training",
        default=200,
    )
    args = parser.parse_args()
    dataloader_batch_size = args.batch_size
    latent_dims = args.latent_dims
    lr = args.learning_rate
    epochs = args.epochs

    # Loading the dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = MnistDataset(transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=dataloader_batch_size, shuffle=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gan = GAN(
        image_size=28,
        channels=1,
        latent_dims=128,
        device=torch.device("cpu"),
        lr=0.02,
    )

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            D_x, D_G_z1, errD, D_G_z2 = gan.step(data)
            # Implement Logging
            # Implement Saving
