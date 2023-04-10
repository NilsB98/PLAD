from abc import abstractmethod

import torch
import numpy as np
import os

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


class SphereDataset(Dataset):
    "requires 'rad' argument in the params dict of the constructor"

    def __init__(self, num: int, dims: int, params: dict, shift_x=0., shift_y=0.):

        assert 'rad' in params
        self.rad = params['rad']
        self.num = num
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.dims = dims
        self.data = None

        print("generating data")
        self.fill_data()
        print("data generation done")

    def fill_data(self):
        def is_inside_sphere(point: torch.Tensor):
            """
            Check if a point is inside the n-dimensional sphere of radius `radius`
            centered at the origin.
            """
            dist = torch.sum(point ** 2, dim=0)
            return dist <= self.rad ** 2

        # if os.path.exists('spheredata.pt'):
        #     self.data = torch.load('spheredata.pt')
        #     return

        points = []
        while len(points) < self.num:
            # Generate a random point in n-dimensional space
            point = [np.random.uniform(-self.rad, self.rad) for _ in range(self.dims)]
            point = torch.Tensor(point)
            if is_inside_sphere(point):
                points.append(point.reshape(1, -1))

        self.data = torch.concat(points)
        self.data += torch.tensor([self.shift_x, self.shift_y])
        # torch.save(self.data, f'spheredata.pt')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num


class RectangleDataset(Dataset):
    def __init__(self, num: int, dims: int, params: dict, shift_x=0., shift_y=0., rotation=0):
        super().__init__(num, params, shift_x, shift_y)
        assert 'width' in params
        assert 'height' in params

        self.shift_x = shift_x
        self.shift_y = shift_y
        self.num = num
        self.width = params['width']
        self.height = params['height']
        self.fill_data()
        self._rotate(rotation)
        self.data = None

    def calc_y_max(self, xs) -> torch.Tensor:
        return torch.ones_like(xs) * self.height / 2

    def get_width(self) -> float:
        return self.width

    def fill_data(self):
        xs = self.sample_in_range(self.num, -self.width, self.width)
        y_max = self.calc_y_max(xs)
        ys = np.random.uniform(-y_max, y_max)
        ys = torch.Tensor(ys)

        self.data = torch.concat([xs + self.shift_x, ys + self.shift_y], dim=1)

    @staticmethod
    def sample_in_range(num, a, b):
        return torch.FloatTensor(num, 1).uniform_(a, b)

    def _rotate(self, angle) -> None:
        """
        Rotate the dataset.

        :param angle: Angle in radians.
        :return: Inplace rotation
        """
        amount = torch.tensor([angle])  # Rotate 45 degrees

        rotation = torch.tensor([[torch.cos(amount), -torch.sin(amount)],
                                 [torch.sin(amount), torch.cos(amount)]])

        self.data = torch.matmul(self.data, rotation)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num
