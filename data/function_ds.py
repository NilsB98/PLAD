import numpy as np
import torch

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Toy dataset generating some sinusoid data.
    Some epsilon is randomly added to the true data, s.t. for one x we can have multiple y.
    To generate data which *doesn't* belong to the distribution set sample_normal to false.
    """
    def __init__(self, num: int, f=np.sin, epsilon=.5, epsilon_pert=.5, interval=(0, 10), sample_normal=True):
        self.num = num
        self.f = f
        self.epsilon = epsilon
        self.epsilon_fake = epsilon_pert
        self.interval = interval
        self.sample_normal = sample_normal

        self.xs = self.sample_in_range(num, interval[0], interval[1])
        self.ys = f(self.xs)

        if sample_normal:
            data_inherent_deviation = np.random.uniform(-epsilon, epsilon, self.xs.shape)
            self.ys += data_inherent_deviation
        else:
            direction = np.random.choice([1, -1], size=self.xs.shape)
            amount = np.random.uniform(0, 1, size=self.xs.shape)
            deviation = direction * (epsilon + amount * epsilon_pert)
            self.ys += deviation

    @staticmethod
    def sample_in_range(num, a, b):
        return torch.FloatTensor(num, 1).uniform_(a, b)

    def __getitem__(self, index):
        return torch.FloatTensor([self.xs[index], self.ys[index]])

    def __len__(self):
        return self.num


class NormalDataset(BaseDataset):
    def __init__(self, num: int, f=np.sin, epsilon=.5, epsilon_pert=.5, interval=(0, 10)):
        super().__init__(num, f, epsilon, epsilon_pert, interval, True)


class PerturbedDataset(BaseDataset):
    def __init__(self, num: int, f=np.sin, epsilon=.5, epsilon_pert=.5, interval=(0, 10)):
        super().__init__(num, f, epsilon, epsilon_pert, interval, False)


class LabeledDataset(Dataset):
    def __init__(self, num: int, f=np.sin, epsilon=.5, epsilon_fake=.5, interval=(0, 10)):
        assert num % 2 == 0

        self.num = num

        self.normal = NormalDataset(num // 2, f, epsilon, epsilon_fake, interval)
        self.perturbed = PerturbedDataset(num // 2, f, epsilon, epsilon_fake, interval)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        if index < self.num // 2:
            return self.normal[index], 0.
        else:
            return self.perturbed[index - self.num // 2], 1.


def get_normal_dataset(num: int, f=np.sin, epsilon=.5, interval=(0, 10)):
    return BaseDataset(num, f, epsilon, interval, sample_normal=True)


def get_perturbed_dataset(num: int, f=np.sin, epsilon_normal=.5, epsilon_perturbed=.5, interval=(0, 10)):
    return BaseDataset(num, f, epsilon_normal, epsilon_perturbed, interval, sample_normal=False)
