from abc import abstractmethod
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from math import log10

from code_soup.common.perturbation import Perturbation


class VisualPerturbation(Perturbation):
    """
    An abstract method for various Visual Perturbation Metrics
        Methods
        __init__(self, original : Union[np.ndarray, torch.Tensor], perturbed: Union[np.ndarray, torch.Tensor])
            - init method
    """

    def __init__(
        self,
        original: Union[np.ndarray, torch.Tensor],
        perturbed: Union[np.ndarray, torch.Tensor],
    ):
        """
        Docstring
        #Automatically cast to Tensor using the torch.from_numpy() in the __init__ using if
        """

        if type(original) == torch.Tensor:
            self.original = original
        else:
            self.original = torch.from_numpy(original)
        print(self.original.shape)

        if type(perturbed) == torch.Tensor:
            self.perturbed = perturbed
        else:
            self.perturbed = torch.from_numpy(perturbed)

    def flatten(self, array : torch.tensor) -> torch.Tensor:
        return array.flatten()

    def totensor(self, array : np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array)

    def subtract(self,original : torch.Tensor, perturbed : torch.Tensor) -> torch.Tensor:
        return torch.sub(original, perturbed)

    def calculate_LPNorm(self, p: Union[int, str]) -> float:
        if p == 'inf':
            return torch.linalg.vector_norm(self.flatten(self.subtract(self.original,self.perturbed)), ord = float('inf')).item()
        elif p == 'fro':
            return self.calculate_LPNorm(2)
        else:
            return torch.linalg.norm(self.flatten(self.subtract(self.original,self.perturbed)), ord = p).item()

    def calculate_PSNR(self) -> float:
        return 20 * log10(255.0/self.calculate_RMSE())

    def calculate_RMSE(self) -> float:
        # return torch.sqrt(torch.mean(self.subtract(self.flatten(self.original), self.flatten(self.perturbed))**2)).item()
        # loss = nn.MSELoss()
        # return (loss(self.original, self.perturbed)**0.5).item()
        raise NotImplementedError

    def calculate_SAM(self):
        raise NotImplementedError

    def calculate_SRE(self):
        raise NotImplementedError
