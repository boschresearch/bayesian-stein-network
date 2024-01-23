import torch
from torch.utils.data import Dataset, DataLoader


class TorchDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_data_loader(
    x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True
) -> DataLoader:
    data_loader = DataLoader(
        TorchDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True,
    )
    return data_loader
