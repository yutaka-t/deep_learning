#!/usr/bin/env python
# -*- coding:utf-8 -*-

from deep_learning.ch4_5.two_layer_net import TwoLayerNet
import numpy as np

if __name__ == '__main__':
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params['W1'].shape)
    print(net.params['b1'].shape)
    print(net.params['W2'].shape)
    print(net.params['b2'].shape)

    # ダミーの入力データ 100毎分
    x = np.random.rand(100, 784)

    # ダミーの正解ラベル
    t = np.random.rand(100, 10)

    # 勾配を計算
    print("---- 勾配を計算 ------")
    grads = net.numerical_gradient(x, t)

    print(grads['W1'].shape)
    print(grads['b1'].shape)
    print(grads['W2'].shape)
    print(grads['b2'].shape)
