import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"
