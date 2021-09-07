import unittest

import torch
import torch.nn as nn

from code_soup.ch5 import GAN, Discriminator, Generator


class TestDiscriminator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Discriminator(image_size=28, channels=1, lr=0.002)

    def test_discriminator_output_shape(self):
        input_data = torch.randn(64, 1, 28, 28)
        output = self.model(input_data)
        self.assertEqual(output.shape, torch.Size([64, 1]))

    def test_discriminator_variable_layer_weights(self):
        self.assertEqual(
            self.model.main[0].weight.data.shape, torch.Size([1024, 28 * 28])
        )


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Generator(image_size=28, channels=1, latent_dims=128, lr=0.02)

    def test_generator_output_shape(self):
        input_data = torch.randn(64, 128)
        output = self.model(input_data)
        self.assertEqual(output.shape, torch.Size([64, 1, 28, 28]))

    def test_generator_variable_layer_weights(self):
        self.assertEqual(self.model.main[0].weight.data.shape, torch.Size([256, 128]))
        self.assertEqual(self.model.main[-2].weight.data.shape, torch.Size([784, 1024]))


class TestGAN(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = GAN(
            image_size=28,
            channels=1,
            latent_dims=128,
            device=torch.device("cpu"),
            lr=0.02,
        )

    def test_step(self):
        self.model.step([torch.randn(4, 28, 28), torch.ones(4)])
