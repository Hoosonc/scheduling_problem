# -*- coding : utf-8 -*-
# @Time :  2022/7/27 14:05
# @Author : hxc
# @File : params.py
# @Software : PyCharm
import argparse


class Params:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=None)
        self.get_parser()
        self.args = self.parser.parse_args()

    def get_parser(self):
        self.parser.add_argument('--lr', type=float, default=0.0001,
                                 help='learning rate (default: 0.0001)')
        self.parser.add_argument('--gamma', type=float, default=0.9,
                                 help='discount factor for rewards (default: 0.99)')
        self.parser.add_argument('--gae-lambda', type=float, default=1,
                                 help='lambda parameter for GAE (default: 1.00)')
        self.parser.add_argument('--entropy-coefficient', type=float, default=0.01,
                                 help='entropy term coefficient (default: 0.01)')
        self.parser.add_argument('--value-loss-coefficient', type=float, default=0.5,
                                 help='value loss coefficient (default: 0.5)')
        self.parser.add_argument('--max-grad-norm', type=float, default=50,
                                 help='max grad norm (default: 50)')
        self.parser.add_argument('--seed', type=int, default=2022,
                                 help='random seed (default: 1)')
        self.parser.add_argument('--num-steps', type=int, default=50,
                                 help='number of forward steps in A2C (default: 300)')
        self.parser.add_argument('--update-episode-length', type=int, default=1,
                                 help='maximum length of an episode (default: 1)')
        self.parser.add_argument('--episode', type=int, default=1000,
                                 help='How many episode to train the RL algorithm')
        self.parser.add_argument('--reg-path', type=str, default='./data/reg.csv',
                                 help='The path of Reg file')
        self.parser.add_argument('--doc-path', type=str, default='./data/doc.csv',
                                 help='The path of doc file')
        self.parser.add_argument('--max_reg_num', type=int, default='4',
                                 help='')
        self.parser.add_argument('--batch', type=int, default='1000',
                                 help='')
