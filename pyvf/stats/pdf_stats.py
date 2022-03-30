import numpy as np


def variance(x, p, normalize=True):
    if normalize:
        assert np.ndim(p) == 1
        p = p / p.sum()
    e_x = x @ p
    e_x2 = (x ** 2) @ p
    return e_x2 - e_x ** 2
