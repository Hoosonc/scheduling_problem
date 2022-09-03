# -*- coding : utf-8 -*-
# @Time :  2022/7/28 22:15
# @Author : hxc
# @File : test.py
# @Software : PyCharm
import numpy as np
import torch.nn.functional as f
import math
import pandas as pd
import datetime


def check_id(p_list, card_id):
    return (p_list.index(card_id)) + 1


if __name__ == '__main__':
    # a = math.factorial(5)
    # lam = 5
    # x = np.random.poisson(lam=lam, size=20)
    mean = 10
    std = 3
    sigma = math.log(std)
    # x = np.random.normal(loc=mean, scale=std, size=10)
    x = np.random.randint(1, 3, 10)
    # print(x)
    a = []
    k = []
    for x_ in x:
        x_ = round(x_)
        k.append(x_)
        # y = math.log((math.pow(lam, x_) / math.factorial(x_)) * math.exp(-lam))
        y = - sigma/2 - ((math.pow(x_ - mean, 2)) / (2*math.exp(sigma)))
        # d = x_*math.log(lam)-math.log(math.factorial(x_))-lam
        # k.append(d)
        a.append(y)
    print(a)
    print(k)


