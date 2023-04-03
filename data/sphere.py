import torch
import numpy as np

from torch.utils.data import Dataset


class SphereDataset(Dataset):
    """
    Toy dataset generating some sinusoid data.
    Some epsilon is randomly added to the true data, s.t. for one x we can have multiple y.
    To generate data which *doesn't* belong to the distribution set sample_normal to false.
    """
    def __init__(self, num: int, rad:int):
        self.num = num
        self.rad = rad

        xs = self.sample_in_range(num, -rad, rad)
        y_max = torch.sqrt(-(torch.square(xs) - rad ** 2))
        ys = np.random.uniform(-y_max, y_max)
        ys = torch.Tensor(ys)

        self.data = torch.concat([xs, ys], dim=1)

    @staticmethod
    def sample_in_range(num, a, b):
        return torch.FloatTensor(num, 1).uniform_(a, b)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num
