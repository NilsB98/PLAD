from typing import Optional, Callable

from torchvision.datasets import MNIST
import torch


class RestrictedMNIST(MNIST):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False, exclude_targets: list = None):
        super().__init__(root, train, transform, target_transform, download)

        if exclude_targets is not None and len(exclude_targets) > 0:
            assert max(exclude_targets) < 10 and min(exclude_targets) >= 0, "exclude_targets out of range [0-9]"

            mask = torch.logical_not(torch.isin(self.targets, torch.tensor(exclude_targets)))
            self.data = self.data[mask]
            self.targets = self.targets[mask]
