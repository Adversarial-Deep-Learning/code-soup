import torchvision.datasets as datasets
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    def __init__(self, transform=None):

        self.train_data = datasets.MNIST(
            root="./input/data", train=True, download=True, transform=transform
        )

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data.__getitem__(idx)
