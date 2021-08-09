import torchvision.datasets as datasets
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    """
    A custom MNIST Dataset Class.
    """

    def __init__(self, transform=None):
        """
        Parameters
        ----------
        transform : torchvision.transforms
            - A transform to be applied on the dataset
        """

        self.train_data = datasets.MNIST(
            root="./input/data", train=True, download=True, transform=transform
        )

    def __len__(self):
        """
        Returns
        -------
        length : int
            - Length of the dataset
        """
        return len(self.train_data)

    def __getitem__(self, idx):
        """
        Returns
        -------
        element : torch.Tensor
            - A element from the dataset
        """
        return self.train_data.__getitem__(idx)
