from abc import abstractmethod
from typing import Union

import numpy as np
import torch

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
        self.original = self.__totensor(original)
        self.perturbed = self.__totensor(perturbed)
        self.original = self.original.type(dtype=torch.float64)
        self.perturbed = self.perturbed.type(dtype=torch.float64)

    def __flatten(self, atensor: torch.Tensor) -> torch.Tensor:
        """
        A method which will flatten out the inputted tensor

        """
        return torch.flatten(atensor)

    def __totensor(self, anarray: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        A method which will convert anything inputted into a Torch tensor.
        If it is already a torch tensor it will return itself

        """
        if type(anarray) == torch.Tensor:
            return anarray
        else:
            return torch.from_numpy(anarray)

    def sam(self, convert_to_degree=True):
        """
        Spectral Angle Mapper defines the spectral similarity by the angle between the image pixel spectrum
        Expects the Images to be in (C,H,W) format

        Parameters
        ----------
        convert_to_degree
            - will return the spectral angle in degrees(default true)

        """
        original_img = self.original
        new_img = self.perturbed

        assert (
            original_img.size() == new_img.size()
        ), "Size of the inputs not same please give correct values to SAM metric"

        # assuming the image is in (C,H,W) method
        numerator = torch.sum(torch.mul(new_img, original_img), axis=0)
        denominator = torch.linalg.norm(original_img, axis=0) * torch.linalg.norm(
            new_img, axis=0
        )
        val = torch.clip(numerator / denominator, -1, 1)
        sam_angles = torch.arccos(val)
        if convert_to_degree:
            sam_angles = sam_angles * 180.0 / np.pi

        # The original paper states that SAM values are expressed as radians, while e.g. Lanares
        # et al. (2018) use degrees. We therefore made this configurable, with degree the default
        return torch.mean(torch.nan_to_num(sam_angles)).item()

    def sre(self):
        """
        signal to reconstruction error ratio
        Expects the Images to be in (C,H,W) format
        """
        original_img = self.original
        new_img = self.perturbed

        assert (
            original_img.size() == new_img.size()
        ), "Size of the inputs not same please give correct values to SRE"

        sre_final = []
        for i in range(original_img.shape[0]):
            numerator = torch.square(
                torch.mean(
                    original_img[
                        :,
                        :,
                    ][i]
                )
            )
            denominator = (
                torch.linalg.norm(
                    original_img[
                        :,
                        :,
                    ][i]
                    - new_img[
                        :,
                        :,
                    ][i]
                )
            ) / (original_img.shape[2] * original_img.shape[1])
            sre_final.append(numerator / denominator)
        sre_final = torch.as_tensor(sre_final)
        return (10 * torch.log10(torch.mean(sre_final))).item()
