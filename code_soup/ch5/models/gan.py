import torch
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
    """
    Simple generator network.
    Methods
    -------
    forward(x)
        - returns a generated sample
    """

    def __init__(self, input_dims, output_dims, lr):
        """
        Parameters
        ----------
        input_dims : int
            Number of input dimensions equal to noise vector dimensions
        output_dims : int
            Number of output dimensions
            Learning rate
        """
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
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

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
        output = self.main(x)
        return output.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """
    Simple discriminator network.
    Methods
    -------
    forward(x)
        - returns a probability that the input is real
    """

    def __init__(self, input_dims, lr):
        """
        Parameters
        ----------
        input_dims : int
            Number of input dimensions
        lr : float
            Learning rate
        """
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
        self.input_dims = input_dims
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        Returns
        -------
        output : torch.Tensor
            Probability that the input is real
        """
        x = x.view(-1, self.input_dims)
        return self.main(x)


class GAN:  # pragma: no cover
    """
    Generative Adversarial Network Model Class
    Methods
    -------
    step(self, i, data)
        Iterates the model for a single batch of data

    """

    def __init__(self, latent_dims, io_size, device, lr):
        """
        Parameters
        ----------
        latent_dims : int
            Number of dimensions in the latent space
        io_size : int
            Number of output dimensions
        device : torch.device
            Device to run the model on
        lr : float
            Learning rate
        """
        self.latent_dims = latent_dims
        self.io_size = io_size
        self.device = device
        self.generator = Generator(latent_dims, io_size, lr).to(device)
        self.discriminator = Discriminator(io_size, lr).to(device)
        self.criterion = torch.nn.BCELoss()
        self.real_label, self.fake_label = 1.0, 0.0

    def step(self, i, data):
        """
        Iterates the model for a single batch of data
        Parameters
        ----------
        i : int
            Current iteration
        data : torch.Tensor
            Batch of data
        Returns
        -------
         D_x:
            The average output (across the batch) of the discriminator for the all real batch
         D_G_z1:
           Average discriminator outputs for the all fake batch before updating discriminator
         errD:
            Discriminator loss
         D_G_z2:
            Average discriminator outputs for the all fake batch after updating discriminator

        """
        real_image, _ = data
        real_image = real_image.to(self.device)
        batch_size = real_image.shape[0]
        label = torch.full(
            (batch_size,), self.real_label, dtype=torch.float, device=self.device
        )
        # Forward pass real batch through D
        output = self.discriminator(real_image).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.criterion(output, label)
        errD_real.backward()

        D_x = output.mean().item()
        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, self.latent_dims, device=self.device)
        # Generate fake image batch with G
        fake = self.generator(noise)
        label.fill_(self.fake_label)
        # Classify all fake batch with D
        output = self.discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        self.discriminator.optimizer.step()

        self.generator.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.generator.optimizer.step()
        return D_x, D_G_z1, errD, D_G_z2
