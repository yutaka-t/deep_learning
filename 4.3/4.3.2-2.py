import numpy as np
import matplotlib.pylab as plt


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


# 0～20までの0.1刻みのx列
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
