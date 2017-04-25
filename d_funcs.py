#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np


def sigmoid(x):
    """
    シグモイド関数
    :param x: input
    :return: 変換された信号
    """
    return 1 / (1 + np.exp(-x))


def softmax(a):
    """
    ソフトマックス関数
    :param a: インプット
    :return: 変換された信号
    """
    c = np.max(a)
    # オーバーフロー対策
    exp_a = np.exp(a - c)

    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
