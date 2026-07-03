import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from src.utils.paths import data_dir

DATA_DIR = data_dir("mnist")

class MNISTRowAutoregressive(Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.MNIST(
            root=DATA_DIR,
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        # image: (1, 28, 28)

        image = image.squeeze(0)  # (28, 28)

        # Build autoregressive input and target pairs.
        X = image[:-1, :]   # Rows 0-26 -> (27, 28)
        y = image[1:, :]    # Rows 1-27 -> (27, 28)

        return X, y


def load_mnist(batch_size):
    train_set = MNISTRowAutoregressive(train=True)
    test_set = MNISTRowAutoregressive(train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
