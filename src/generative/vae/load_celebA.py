import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.utils.paths import data_dir

DEFAULT_DATA_DIR = data_dir("celeba")


class CelebADataset(Dataset):
    def __init__(self, root, img_shape=(64, 64)):
        self.root = root
        self.filenames = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.filenames = self.filenames[:1000] 
        self.transform = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path).convert("RGB")
        return self.transform(img)


def get_dataloader(root=DEFAULT_DATA_DIR, batch_size=64, num_workers=4):
    dataset = CelebADataset(root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == '__main__':
    dataloader = get_dataloader(
        root=DEFAULT_DATA_DIR,
        batch_size=4,
        num_workers=0
    )
    x = next(iter(dataloader))
    print(x.shape)
