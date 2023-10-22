import numpy as np

# from torch.utils.data import Dataset
from typing import Mapping


Batch = Mapping[str, np.ndarray]


# class AbstractDataset(Dataset):
class AbstractDataset:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

        if self.x is not None:
            self.n_samples = len(self.x)
        else:
            self.n_samples = 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_i = self.x[idx]
        y_i = self.y[idx]
        return x_i, y_i
