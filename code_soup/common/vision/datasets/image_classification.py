from typing import Any, Tuple

import torch
import torchvision

from code_soup.common.vision.datasets.vision_dataset import VisionDataset


class ImageClassificationDataset(torch.utils.data.Dataset, VisionDataset):
    """
    Image Classification Dataset Class, Inherits from VisionDataset Abstract class and Torch Dataset
    Parameters
    ----------
    dataset : torchvision.datasets
        - A dataset from torchvision.datasets
    transform : torchvision.transforms
        - A transform to be applied on the dataset
    root : str
        - The path where downloads are stored
    train: bool
        - If the split is training or testing
    """

    def __init__(self, dataset, transform, root="./input/data", train=True):
        self.data = dataset(root=root, train=train, download=True, transform=transform)

    def __len__(self) -> int:
        """
        Returns
        -------
        length : int
            - Length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Returns
        -------
        element : torch.Tensor
            - A element from the dataset
        """
        return self.data.__getitem__(idx)
