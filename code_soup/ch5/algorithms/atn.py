"""Implements Adversarial Transformation Networks


Assumptions:
    - The classifier model outptus softmax logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.activation import ReLU
from torch.nn.modules.module import T


class BilinearUpsample(nn.Module):
    def __init__(self, scale_factor):
        super(BilinearUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=True
        )


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


class AAEBase(ATNBase):
    def forward(self, x):
        adv_out = self.atn(x)
        logits = self.classifier_model(adv_out)
        return logits


class PATNBase(ATNBase):
    def forward(self, x):
        adv_out = self.atn(x)
        logits = self.classifier_model(adv_out + x)
        return logits


class SimpleAAE(AAEBase):
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
        super(SimpleAAE, self).__init__(classifier_model, target_idx, alpha, beta)

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


class SimplePATN(PATNBase):
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
        super(SimplePATN, self).__init__(classifier_model, target_idx, alpha, beta)

        layers = []
        sizes = [input_shape[0]] + num_channels
        for i in range(len(sizes) - 1):
            layers.append(nn.Conv2d(sizes[i], sizes[i + 1], kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            # TODO: Check if Max Pooling is needed here.

        layers.append(nn.Flatten())
        layers.append(
            nn.Linear(sizes[-1] * input_shape[2] * input_shape[3], num_classes)
        )
        layers.append(
            nn.Tanh()
        )  # TODO: Check if this is the right activation function for PATN

        self.atn = nn.ModuleList(layers)


class BaseDeconvAAE(AAEBase):
    def __init__(
        self,
        classifier_model: torch.nn.Module,
        pretrained_backbone: torch.nn.Module,
        target_idx: int,
        alpha: float = 1.5,
        beta: float = 0.010,
        backbone_output_shape: list = [192, 35, 35],
    ):

        if backbone_output_shape != [192, 35, 35]:
            raise ValueError("Backbone output shape must be [192, 35, 35].")

        super(BaseDeconvAAE, self).__init__(classifier_model, target_idx, alpha, beta)

        layers = [
            pretrained_backbone,
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(192, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ZeroPad2d((3, 2, 3, 2)),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        ]

        self.atn = nn.ModuleList(layers)


class ResizeConvAAE(AAEBase):
    def __init__(
        self,
        classifier_model: torch.nn.Module,
        target_idx: int,
        alpha: float = 1.5,
        beta: float = 0.010,
    ):

        super(ResizeConvAAE, self).__init__(classifier_model, target_idx, alpha, beta)

        layers = [
            nn.Conv2d(3, 128, 5, padding=2),
            nn.ReLU(),
            BilinearUpsample(scale_factor=0.5),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            BilinearUpsample(scale_factor=0.5),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            BilinearUpsample(scale_factor=0.5),
            nn.Conv2d(512, 512, 1, padding=0),
            nn.ReLU(),
            BilinearUpsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            BilinearUpsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            BilinearUpsample(scale_factor=2),
            nn.ZeroPad2d((3, 2, 3, 2)),
            nn.Conv2d(128, 3, 3, padding=1),
            nn.Tanh(),
        ]

        self.atn = nn.ModuleList(layers)


class ConvDeconvAAE(AAEBase):
    def __init__(
        self,
        classifier_model: torch.nn.Module,
        target_idx: int,
        alpha: float = 1.5,
        beta: float = 0.010,
    ):

        super(ConvDeconvAAE, self).__init__(classifier_model, target_idx, alpha, beta)

        layers = [
            nn.Conv2d(3, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 768, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
        ]

        self.atn = nn.ModuleList(layers)


class BaseDeconvPATN(PATNBase):
    def __init__(
        self,
        classifier_model: torch.nn.Module,
        pretrained_backbone: torch.nn.Module,
        target_idx: int,
        alpha: float = 1.5,
        beta: float = 0.010,
        backbone_output_shape: list = [192, 35, 35],
    ):

        if backbone_output_shape != [192, 35, 35]:
            raise ValueError("Backbone output shape must be [192, 35, 35].")

        super(BaseDeconvPATN, self).__init__(classifier_model, target_idx, alpha, beta)

        layers = [
            pretrained_backbone,
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(192, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ZeroPad2d((3, 2, 3, 2)),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # TODO: CHeck if right activation
        ]

        self.atn = nn.ModuleList(layers)


class ConvFCPATN(PATNBase):
    def __init__(
        self,
        classifier_model: torch.nn.Module,
        target_idx: int,
        alpha: float = 1.5,
        beta: float = 0.010,
    ):

        super(BaseDeconvAAE, self).__init__(classifier_model, target_idx, alpha, beta)

        layers = [
            nn.Conv2d(3, 512, 3, stride=2, padding=1),
            nn.Conv2d(512, 256, 3, stride=2, padding=1),
            nn.Conv2d(256, 128, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(184832, 512),
            nn.Linear(512, 268203),
            nn.Tanh(),
        ]

        self.atn = nn.ModuleList(layers)
