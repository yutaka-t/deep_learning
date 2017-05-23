import numpy as np
import matplotlib.pylab as plt


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def numerical_diff(f, x):
    # 0.0001
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


if __name__ == '__main__':
    ret1 = numerical_diff(function_1, 5)
    ret2 = numerical_diff(function_1, 10)

    print("ret1 : {}".format(str(ret1)))
    print("ret2 : {}".format(str(ret2)))
