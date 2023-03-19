import numpy as np


def sample_in_range(num: int, start: int, end: int):
    """
    Uniformly sample values in a specified range between :start and :end.

    :param num: number of samples to produce
    :param start: min value of the generated samples
    :param end: max value of the generated samples
    :return: numpy array of generated samples
    """

    assert end > start

    samples = np.random.random(num)
    # center around zero
    samples -= 0.5

    # scale according to input
    samples *= end - start

    # shift according to input
    samples += (start + end) / 2

    return samples
