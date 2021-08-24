from abc import abstractmethod
from typing import Union

import numpy as np
import torch

from code_soup.common.perturbations import Perturbation


class VisualPerturbation(Perturbation):
    """
    Docstring for VisualPerturbations
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
        raise NotImplementedError

    def calculate_LPNorm(self, p: Union[int, str]):
        raise NotImplementedError

    def calculate_PSNR(self):
        raise NotImplementedError

    def calculate_RMSE(self):
        raise NotImplementedError

    def calculate_SAM(self):
        raise NotImplementedError

    def calculate_SRE(self):
        raise NotImplementedError
