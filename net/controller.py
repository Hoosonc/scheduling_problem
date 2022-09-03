# -*- coding : utf-8 -*-
# @Time :  2022/8/2 14:37
# @Author : hxc
# @File : controller.py
# @Software : PyCharm
import torch.nn as nn
import torch
import torch.nn.functional as f


class Center(nn.Module):
    def __init__(self, args):
        super(Center, self).__init__()
        self.args = args
        self.hyper_hidden_dim = args.num_steps * args.update_episode_length
        self.hidden_dim = args.num_steps * args.update_episode_length
        self.hyper_w1 = nn.Sequential(nn.Linear(1, self.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hyper_hidden_dim, 2 * self.hidden_dim))
        self.hyper_w2 = nn.Sequential(nn.Linear(1, self.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hyper_hidden_dim, self.hidden_dim))

        self.hyper_b1 = nn.Linear(1, self.hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(1, self.hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_dim, 1))

        self.prob = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

    def forward(self, q_input, r_input):
        log_it = self.prob(q_input.view(-1, 2 * self.hidden_dim))
        prob = f.softmax(log_it, dim=-1)
        log_prob = f.log_softmax(log_it, dim=-1)
        entropy = -(log_prob * prob).sum(0, keepdim=True)

        q_input = q_input.view(-1, 1, 2)
        r_input = r_input.reshape(-1, 1)

        w1 = torch.abs(self.hyper_w1(r_input))
        b1 = self.hyper_b1(r_input)

        w1 = w1.view(-1, 2, self.hidden_dim)
        b1 = b1.view(-1, 1, self.hidden_dim)

        hidden = f.elu(torch.bmm(q_input, w1) + b1)

        w2 = torch.abs(self.hyper_w2(r_input))
        b2 = self.hyper_b2(r_input)

        w2 = w2.view(-1, self.hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(self.args.num_steps * self.args.update_episode_length, -1, 1)

        return q_total, log_prob, entropy
