# -*- coding : utf-8 -*-
# @Time :  2022/7/28 22:15
# @Author : hxc
# @File : test.py
# @Software : PyCharm
import numpy as np
import torch
import torch.nn.functional as f
import math
import pandas as pd
import datetime


def check_id(p_list, card_id):
    return (p_list.index(card_id)) + 1


if __name__ == '__main__':
    a = [1, 2, 3, 4]
    a = a[1:]
    print(a)
