from typing import Any, Tuple
import torchvision.datasets as datasets
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    """
    `MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset. Built using TorchVision Dataset class.
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, transform: callable = None):
        """
        Parameters
        ----------
        transform : torchvision.transforms
            - A transform to be applied on the dataset
        """

        self.train_data = datasets.MNIST(
            root="./input/data", train=True, download=True, transform=transform
        )

    def __len__(self) -> int:
        """
        Returns
        -------
        length : int
            - Length of the dataset
        """
        return len(self.train_data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Returns
        -------
        element : torch.Tensor
            - A element from the dataset
        """
        return self.train_data.__getitem__(idx)
