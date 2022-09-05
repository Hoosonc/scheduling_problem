# -*- coding : utf-8 -*-
# @Time :  2022/7/29 10:53
# @Author : hxc
# @File : agent_model.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
# from scipy import stats
# import math
# from net.utils import normalized_columns_initializer, weights_init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(torch.nn.Module):
    def __init__(self, in_channels, action_space, agent_id):
        super(ActorCritic, self).__init__()
        self.agent_id = agent_id
        out_channels = 10
        self.f1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), (2, 2), padding=1),
                                nn.ReLU(),
                                nn.Conv2d(out_channels, out_channels, (3, 3), (2, 2), padding=1),
                                nn.ReLU(),
                                nn.Conv2d(out_channels, out_channels, (3, 3), (2, 2), padding=1),
                                nn.ReLU())

        self.f2 = nn.Sequential(nn.Linear(2, 5),
                                nn.ReLU())
        self.num_outputs = action_space
        self.lstm = nn.LSTMCell(self.num_outputs*10, 128)

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(128, self.num_outputs)
        self.action_dense = nn.Linear(self.num_outputs, 128)
        # self.apply(weights_init)

        # self.mean_linear.weight.data = normalized_columns_initializer(
        #     self.mean_linear.weight.data, 10)
        # self.mean_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x1 = self.f1(inputs[0])
        x2 = self.f2(inputs[1])
        x1 = x1.view(-1, x1.shape[1] * x1.shape[3])
        x2 = x2.view(-1, x2.shape[0] * x2.shape[1])
        x = torch.cat([x1, x2], dim=1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        actor = self.actor_linear(x)

        # 考虑改成argmax
        action = f.softmax(actor, dim=-1).multinomial(num_samples=1).detach()[0].item()

        log_pi = f.log_softmax(actor, dim=-1).view(self.num_outputs,)[action]
        one_hot_a = np.array([0 for _ in range(self.num_outputs)], dtype="float32")
        one_hot_a[action] = 1
        one_hot_a = self.action_dense(torch.from_numpy(one_hot_a).view(1, 4).to(device))
        critic_input = torch.cat([x, one_hot_a], dim=1)
        critic_v = self.critic_linear(critic_input)
        return action, log_pi, (hx, cx), critic_v

    def choose_action(self, inputs):
        s, (hx, cx) = inputs
        action, log_pi, lstm_cells, critic_v = self.forward((s, (hx, cx)))

        return action, log_pi, lstm_cells, critic_v
