from abc import ABC, abstractmethod
from typing import Any, Tuple


class VisionDataset(ABC):
    @classmethod
    @abstractmethod
    def __init__(self, dataset: callable, transform: callable, root: str, train: bool):
        """
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
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns
        -------
        length : int
            - Length of the dataset
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Returns
        -------
        element : torch.Tensor
            - A element from the dataset
        """
        pass
