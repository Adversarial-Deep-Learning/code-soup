"""Implements Adversarial Transformation Networks


Assumptions:
    - The classifier model outptus softmax logits.
"""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ATNBase(nn.Module):
    def __init__(
        self,
        classifier_model: torch.nn.Module,
        target_idx: int,
        alpha: float = 1.5,
        beta: float = 0.010,
        device: torch.device = torch.device("cpu"),
        lr: float = 0.001,
    ):
        """
        Initializes ATN Base class.

        Args:
            classifier_model (torch.nn.Module):
                A pre-trained classification model that outputs logits (without softmax).
            target_idx (int): The index of the target class.
            alpha (float): The value by which the max logit is multiplied and set at target index.
            beta (float): The weight for the perturbation loss.
            device (torch.device): The device to use.
            lr (float): The learning rate.
        """

        super(ATNBase, self).__init__()
        if alpha <= 1:
            raise ValueError("Alpha must be greater than 1")
        self.classifier_model = classifier_model
        self.alpha = alpha
        self.beta = beta
        self.target_idx = target_idx
        self.device = device
        self.lr = lr
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    # TODO: Check if this seems okay
    @torch.no_grad()
    def rerank(self, softmax_logits: torch.Tensor) -> torch.Tensor:
        """
        Re-ranks the softmax logits.

        Args:
            softmax_logits (torch.Tensor): The softmax logits to be re-ranked.
        Returns:
            torch.Tensor: The re-ranked softmax logits.
        """

        # Get the max logit
        max_logits = torch.max(softmax_logits, dim=1).values

        # Set the max logit at the target index and multiply by self.alpha
        softmax_logits[:, self.target_idx] = max_logits * self.alpha
        softmax_logits = softmax_logits / torch.linalg.norm(
            softmax_logits, dim=-1
        ).view(-1, 1)
        return softmax_logits

    def forward(self, x):
        """
        Forward pass of the model. Not implemented for this class.

        Args:
            x (torch.Tensor): The input to the model.

        Raises:
            NotImplementedError: This method is not implemented.
        """

        raise NotImplementedError(
            "Forward for ATNBase has not been implemented. Please use child classes for a model."
        )

    def compute_loss(
        self, x: torch.Tensor, x_hat: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the loss for input and output.

        Args:
            x (torch.Tensor): The original input to the classification/ATN model.
            x_hat (torch.Tensor): The adversarial output from the ATN model.
            y (torch.Tensor): The re-ranked logits from the classification model.
            y_hat (torch.Tensor): The output logits from the classifier on the adversarial input.

        Returns:
            torch.Tensor: A tensor containing loss.

        """
        return self.beta * self.loss_fn(x, x_hat) + self.loss_fn(y, y_hat)

    def step(self, data: torch.Tensor) -> Tuple[Union[torch.Tensor, float]]:
        """
        Performs a single optimization step.

        Args:
            data (torch.Tensor): The data to be used for the optimization step.

        Returns:
            Tuple[Union[torch.Tensor, float]]:
                A tuple containing the adversarial image,
                the softmax logits from the classifier model on the adversarial image,
                and the loss.
        """
        image, label = data
        image = image.to(self.device)

        adv_out, adv_logits = self(image)

        self.zero_grad()
        cls_model_out = self.classifier_model(image)
        softmax_logits = F.softmax(cls_model_out, dim=1)

        # Rerank the softmax logits
        reranked_logits = self.rerank(softmax_logits)

        # Calculate loss on a batch
        loss = self.compute_loss(image, adv_out, reranked_logits, adv_logits)
        loss.backward()

        self.optimizer.step()
        return adv_out, adv_logits, loss.item()


class SimpleAAE(ATNBase):
    def __init__(
        self,
        classifier_model: torch.nn.Module,
        target_idx: int,
        alpha: float = 1.5,
        beta: float = 0.010,
        device: torch.device = torch.device("cpu"),
        lr: float = 0.001,
        input_shape: tuple = (1, 28, 28),
        num_channels: list = [64, 64],
        deconv_num_channels: list = [64, 64],
        typ="a",
    ):
        """
        Initializes the SimpleAAE class. In these type of ATNs adversarial images are produced.

        Args:
            classifier_model (torch.nn.Module):
                A pre-trained classification model that outputs logits (without softmax).
            target_idx (int): The index of the target class.
            alpha (float): The value by which the max logit is multiplied and set at target index.
            beta (float): The weight for the perturbation loss.
            device (torch.device): The device to use.
            lr (float): The learning rate.
            input_shape (tuple): The shape of the input.
            num_channels (list): The number of channels in each convolutional layer.
            deconv_num_channels (list): The number of channels in each deconvolutional layer.
            typ (str): The type of the model. One of 'a', 'b', 'c'. Based on the paper.
        """

        assert typ in ["a", "b", "c"]
        super(SimpleAAE, self).__init__(
            classifier_model, target_idx, alpha, beta, device, lr
        )

        self.input_shape = input_shape

        if typ == "a":
            layers = []
            sizes = (
                [input_shape[0] * input_shape[1] * input_shape[2]]
                + num_channels
                + [input_shape[0] * input_shape[1] * input_shape[2]]
            )
            layers.append(nn.Flatten())
            for i in range(len(sizes) - 1):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                if i != len(sizes) - 2:
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Tanh())

        elif typ == "b":
            layers = []
            sizes = [input_shape[0]] + num_channels
            for i in range(len(sizes) - 1):
                layers.append(
                    nn.Conv2d(sizes[i], sizes[i + 1], kernel_size=3, padding=1)
                )
                layers.append(nn.ReLU())
                # TODO: Check if Max Pooling is needed here (most probably is).

            layers.append(nn.Flatten())
            layers.append(
                nn.Linear(
                    sizes[-1] * input_shape[1] * input_shape[2],
                    input_shape[0] * input_shape[1] * input_shape[2],
                )
            )
            layers.append(nn.Tanh())

        elif typ == "c":
            layers = []
            sizes = [input_shape[0]] + num_channels
            for i in range(len(sizes) - 1):
                layers.append(
                    nn.Conv2d(sizes[i], sizes[i + 1], kernel_size=3, padding=1)
                )
                layers.append(nn.ReLU())

            deconv_sizes = [num_channels[-1]] + deconv_num_channels
            for j in range(len(deconv_sizes) - 1):
                layers.append(
                    nn.ConvTranspose2d(
                        deconv_sizes[j],
                        deconv_sizes[j + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )

                layers.append(nn.ReLU())

            layers.append(nn.Flatten())

            layers.append(
                nn.Linear(
                    deconv_sizes[-1] * input_shape[1] * input_shape[2],
                    input_shape[0] * input_shape[1] * input_shape[2],
                )
            )
            layers.append(nn.Tanh())

        self.atn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Performs a forward pass on the model.

        Args:
            x (torch.Tensor): The input to the model.

        Returns:
            Tuple[torch.Tensor]:
                The adversarial output of the ATN model,
                the softmax logits from the classifier model on the adversarial image.
        """
        adv_out = self.atn(x)
        adv_out = adv_out.view(-1, *self.input_shape)
        logits = self.classifier_model(adv_out)
        softmax_logits = F.softmax(logits, dim=1)
        return adv_out, softmax_logits


class SimplePATN(ATNBase):
    def __init__(
        self,
        classifier_model: torch.nn.Module,
        target_idx: int,
        alpha: float = 1.5,
        beta: float = 0.010,
        device: torch.device = torch.device("cpu"),
        lr: float = 0.001,
        input_shape: tuple = (1, 28, 28),
        num_channels: list = [64, 64],
    ):

        """
        Initializes the SimplePATN class. In these type of ATNs, perturbations are produced.

        Args:
            classifier_model (torch.nn.Module):
                A pre-trained classification model that outputs logits (without softmax).
            target_idx (int): The index of the target class.
            alpha (float): The value by which the max logit is multiplied and set at target index.
            beta (float): The weight for the perturbation loss.
            device (torch.device): The device to use.
            lr (float): The learning rate.
            input_shape (tuple): The shape of the input.
            num_channels (list): The number of channels in each convolutional layer.
        """
        super(SimplePATN, self).__init__(
            classifier_model, target_idx, alpha, beta, device, lr
        )

        self.input_shape = input_shape

        layers = []
        sizes = [input_shape[0]] + num_channels
        for i in range(len(sizes) - 1):
            layers.append(nn.Conv2d(sizes[i], sizes[i + 1], kernel_size=3, padding=1))
            layers.append(nn.ReLU())

        layers.append(nn.Flatten())
        layers.append(
            nn.Linear(
                sizes[-1] * input_shape[1] * input_shape[2],
                input_shape[0] * input_shape[1] * input_shape[2],
            )
        )
        layers.append(
            nn.Tanh()
        )  # TODO: Check if this is the right activation function for PATN

        self.atn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs a forward pass on the model.

        Args:
            x (torch.Tensor): The input to the model.

        Returns:
            Tuple[torch.Tensor]:
                The adversarial image for the model,
                the softmax logits from the classifier model on the adversarial image.
        """
        adv_out = self.atn(x)
        adv_out = adv_out.view(-1, *self.input_shape)
        logits = self.classifier_model(adv_out + x)
        softmax_logits = F.softmax(logits, dim=1)
        return adv_out + x, softmax_logits
