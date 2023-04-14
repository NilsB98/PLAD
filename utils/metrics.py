import torch


def bin_accuracy(target: torch.Tensor, actual: torch.Tensor, target_class=-1):
    """
    Binary accuracy
    :return: accuracy
    """

    target = target.cpu()
    actual = actual.cpu()
    actual = actual.reshape(target.shape)

    actual = torch.round(actual)

    if target_class >= 0:
        mask = target == target_class
        target = target[mask]
        actual = actual[mask]

    diffs = target == actual
    return torch.sum(diffs) / len(actual)
