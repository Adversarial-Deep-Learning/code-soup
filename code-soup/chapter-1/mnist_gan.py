from datasets import Mnist
from models import Generator, Discriminator
from utils import visualize_progression, visualize_loss
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optim
import torch
import argparse

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


def train_mnist_gan():
    # Loading the dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = Mnist(transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=dataloader_batch_size, shuffle=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initializing the models
    generator = Generator(latent_dims).to(device)
    discriminator = Discriminator(784).to(device)

    # Initializing the optimizers
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr)
    optimizerG = optim.Adam(generator.parameters(), lr=lr)

    # Defining Loss function
    criterion = torch.nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    fixed_noise = torch.randn(64, latent_dims, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    # Lists that will help us visualize GAN outputs & losses
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            discriminator.zero_grad()
            real_image, _ = data
            real_image = real_image.to(device)
            batch_size = real_image.shape[0]
            label = torch.full(
                (batch_size,), real_label, dtype=torch.float, device=device
            )
            # Forward pass real batch through D
            output = discriminator(real_image).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            errD_real.backward()

            D_x = output.mean().item()
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, latent_dims, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        epoch + 1,
                        epochs,
                        i,
                        len(dataloader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or (
                (epoch == epoch - 1) and (i == len(dataloader) - 1)
            ):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
    visualize_progression(img_list)
    visualize_loss(G_losses, D_losses)


if __name__ == "__main__":
    train_mnist_gan()
