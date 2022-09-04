# -*- coding : utf-8 -*-
# @Time :  2022/9/4 9:22
# @Author : hxc
# @File : patient.py
# @Software : PyCharm
import numpy as np


class Patient:
    def __init__(self, args):
        self.args = args
        self.pid = None
        self.reg = []
