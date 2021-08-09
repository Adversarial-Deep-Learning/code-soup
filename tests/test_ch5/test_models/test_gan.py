import unittest

import torch
import torch.nn as nn

from code_soup.ch5.models.gan import Discriminator, Generator


class TestDiscriminatorModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Discriminator(28 * 28, lr=0.002)

    def test_discriminator_output_shape(self):
        input_data = torch.randn(64, 1, 28, 28)
        output = self.model(input_data)
        self.assertEqual(output.shape, torch.Size([64, 1]))

    def test_discriminator_variable_layer_weights(self):
        self.assertEqual(
            self.model.main[0].weight.data.shape, torch.Size([1024, 28 * 28])
        )


class TestGeneratorModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Generator(128, 784, lr=0.02)

    def test_generator_output_shape(self):
        input_data = torch.randn(64, 128)
        output = self.model(input_data)
        self.assertEqual(output.shape, torch.Size([64, 1, 28, 28]))

    def test_generator_variable_layer_weights(self):
        self.assertEqual(self.model.main[0].weight.data.shape, torch.Size([256, 128]))
        self.assertEqual(self.model.main[-2].weight.data.shape, torch.Size([784, 1024]))
