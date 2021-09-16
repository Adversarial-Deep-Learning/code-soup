"""Implements Adversarial Transformation Networks


Assumptions:
    - The classifier model outptus softmax logits.
"""

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
    ):
        super(ATNBase, self).__init__()
        if alpha <= 1:
            raise ValueError("Alpha must be greater than 1")
        self.classifier_model = classifier_model
        self.alpha = alpha
        self.beta = beta
        self.target_idx = target_idx

    # TODO: Check if this seems okay
    @torch.nograd
    def rerank(self, softmax_logits):
        max_logits = torch.max(softmax_logits, dim=1).values
        softmax_logits[:, self.target_idx] = max_logits * self.alpha
        softmax_logits = softmax_logits / torch.linalg.norm(
            softmax_logits, dim=-1
        ).view(-1, 1)
        return softmax_logits

    def forward(self, x):
        raise NotImplementedError(
            "Forward for ATNBase has not been implemented. Please use child classes for a model."
        )

    def compute_loss(self, x, x_hat, y, y_hat):
        loss_fn = nn.MSELoss()
        return self.beta * loss_fn(x, x_hat) + loss_fn(y, y_hat)


class AAE(ATNBase):
    def __init__(
        self,
        classifier_model: torch.nn.Module,
        target_idx: int,
        alpha: float = 1.5,
        beta: float = 0.010,
        input_shape: tuple = (1, 28, 28),
        num_channels: list = [64, 64],
        deconv_num_channels: list = [64, 64],
        num_classes: int = 784,
        typ="a",
    ):
        assert typ in ["a", "b", "c"]
        super(AAE, self).__init__(classifier_model, target_idx, alpha, beta)

        self.input_shape = input_shape

        layers = []
        if typ == "a":
            sizes = (
                [input_shape[0] * input_shape[1] * input_shape[2]]
                + num_channels
                + [num_classes]
            )
            layers.append(nn.Flatten())
            for i in range(len(sizes) - 1):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                if i != len(sizes) - 2:
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Tanh())
        elif typ == "b":
            sizes = [input_shape[0]] + num_channels
            for i in range(len(sizes) - 1):
                layers.append(
                    nn.Conv2d(sizes[i], sizes[i + 1], kernel_size=3, padding=1)
                )
                layers.append(nn.ReLU())
                # TODO: Check if Max Pooling is needed here (most probably is).

            layers.append(nn.Flatten())
            layers.append(
                nn.Linear(sizes[-1] * input_shape[2] * input_shape[3], num_classes)
            )
            layers.append(nn.Tanh())

        elif typ == "c":
            sizes = [input_shape[0]] + num_channels
            for i in range(len(sizes) - 1):
                layers.append(
                    nn.Conv2d(sizes[i], sizes[i + 1], kernel_size=3, padding=1)
                )
                layers.append(nn.ReLU())
                # TODO: Check if Max Pooling is needed here (most probably is).

            deconv_sizes = [num_channels[-1]] + deconv_num_channels
            for j in range(len(deconv_sizes) - 1):
                layers.append(
                    nn.ConvTranspose2d(
                        deconv_num_channels[j],
                        deconv_num_channels[j + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                # TODO: Check if higher stride is needed here (most probably is).
                layers.append(nn.ReLU())

            layers.append(nn.Flatten())

            layers.append(
                nn.Linear(
                    deconv_sizes[-1] * input_shape[2] * input_shape[3], num_classes
                )
            )
            layers.append(nn.Tanh())

        self.atn = nn.ModuleList(layers)

    def forward(self, x):
        adv_out = self.atn(x)
        logits = self.classifier_model(adv_out)
        return logits


class PATN(ATNBase):
    def __init__(
        self,
        classifier_model: torch.nn.Module,
        target_idx: int,
        alpha: float = 1.5,
        beta: float = 0.010,
        input_shape: tuple = (1, 28, 28),
        num_channels: list = [64, 64],
        num_classes: int = 784,
    ):
        super(PATN, self).__init__(classifier_model, target_idx, alpha, beta)

        layers = []
        sizes = [input_shape[0]] + num_channels
        for i in range(len(sizes) - 1):
            layers.append(nn.Conv2d(sizes[i], sizes[i + 1], kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            # TODO: Check if Max Pooling is needed here (most probably is).

        layers.append(nn.Flatten())
        layers.append(
            nn.Linear(sizes[-1] * input_shape[2] * input_shape[3], num_classes)
        )
        layers.append(nn.Tanh())

        self.atn = nn.ModuleList(layers)

    def forward(self, x):
        adv_out = self.atn(x)
        logits = self.classifier_model(adv_out + x)
        return logits
