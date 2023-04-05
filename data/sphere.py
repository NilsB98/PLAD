from abc import abstractmethod

import torch
import numpy as np

from torch.utils.data import Dataset


class GeometricDataset(Dataset):
    """
    Toy dataset generating some sinusoid data.
    Some epsilon is randomly added to the true data, s.t. for one x we can have multiple y.
    To generate data which *doesn't* belong to the distribution set sample_normal to false.
    """

    def __init__(self, num: int, params: dict, shift_x=0., shift_y=0.):
        self.num = num
        self.params = params
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.data = None

    def fill_data(self):
        width = self.get_width()

        xs = self.sample_in_range(self.num, -width, width)
        y_max = self.calc_y_max(xs)
        ys = np.random.uniform(-y_max, y_max)
        ys = torch.Tensor(ys)

        self.data = torch.concat([xs + self.shift_x, ys + self.shift_y], dim=1)

    @staticmethod
    def sample_in_range(num, a, b):
        return torch.FloatTensor(num, 1).uniform_(a, b)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num

    @staticmethod
    def calc_y_max(xs) -> torch.Tensor:
        return NotImplementedError()

    @staticmethod
    def get_width() -> float:
        return NotImplementedError()


class SphereDataset(GeometricDataset):
    "requires 'rad' argument in the params dict of the constructor"

    def __init__(self, num: int, params: dict, shift_x=0., shift_y=0.):
        super().__init__(num, params, shift_x, shift_y)
        self.rad = params['rad']
        self.fill_data()

    def calc_y_max(self, xs):
        return torch.sqrt(-(torch.square(xs) - self.rad ** 2))

    def get_width(self) -> float:
        return self.rad


class RectangleDataset(GeometricDataset):
    def __init__(self, num: int, params: dict, shift_x=0., shift_y=0.):
        super().__init__(num, params, shift_x, shift_y)
        self.width = params['width']
        self.height = params['height']
        self.fill_data()

    def calc_y_max(self, xs) -> torch.Tensor:
        return torch.ones_like(xs) * self.height / 2

    def get_width(self) -> float:
        return self.width
