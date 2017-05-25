# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def numerical_gradient(f, x):
    # 0.01
    h = 1e-4

    # xと同じ形状の配列を生成
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x + h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x - h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 値を元に戻す
        x[idx] = tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    勾配降下法
    :param f: 最適化したい関数 
    :param init_x: 初期値
    :param lr: 学習率(learning rate)
    :param step_num: 勾配法による繰り返しの数
    :return: 
    """
    x = init_x

    for i in range(step_num):
        # 関数の勾配を求める
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def function_2(x):
    return x[0]**2 + x[1]**2

if __name__ == '__main__':
    init_x_array = np.array([-3.0, 4.0])
    ret = gradient_descent(function_2, init_x=init_x_array, lr=0.1, step_num=100)
    print(ret)