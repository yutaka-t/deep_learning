import numpy as np
import pandas as pd


def cross_entropy_error(y, t, one_hot=True):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    if one_hot:
        return -np.sum(t * np.log(y)) / batch_size
    else:
        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


