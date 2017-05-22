#!/usr/bin/env python
# -*- coding:utf-8 -*-

from deep_learning.dataset.mnist import load_mnist
import sys
import os

sys.path.append(os.pardir)

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# それぞれのデータの形状を出力
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
