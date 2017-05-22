#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import math


# 2乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


if __name__ == '__main__':
    t1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])

    print("ret1 = ", mean_squared_error(y1, t1))
    print("ret2 = ", mean_squared_error(y2, t1))

    print("ret3 = ", cross_entropy_error(y1, t1))
    print("ret4 = ", cross_entropy_error(y2, t1))
