# -*- coding : utf-8 -*-
# @Time :  2022/7/29 10:53
# @Author : hxc
# @File : agent_model.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from scipy import stats
import math
from net.utils import normalized_columns_initializer, weights_init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(torch.nn.Module):
    def __init__(self, in_channels, action_space, agent_id):
        super(ActorCritic, self).__init__()
        self.agent_id = agent_id
        out_channels = 32

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (2, 2), padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (2, 2), padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, (3, 3), (2, 2), padding=1)
        # self.conv4 = nn.Conv2d(out_channels, out_channels, (3, 3), (2, 2), padding=1)

        self.lstm = nn.LSTMCell(384, 64)

        num_outputs = action_space
        # self.std_linear = nn.Linear(64, num_outputs)
        # self.mean_linear = nn.Linear(64, num_outputs)
        self.df1_linear = nn.Linear(64, 90)
        self.df2_linear = nn.Linear(64, 10)
        self.critic_linear = nn.Linear(64, 1)
        # self.actor_linear = nn.Linear(64, 1)
        # self.apply(weights_init)

        # self.mean_linear.weight.data = normalized_columns_initializer(
        #     self.mean_linear.weight.data, 10)
        # self.mean_linear.bias.data.fill_(0)
        # self.std_linear.weight.data = normalized_columns_initializer(
        #     self.std_linear.weight.data, 10)
        # self.std_linear.bias.data.fill_(0)

        # self.df1_linear.weight.data = normalized_columns_initializer(
        #     self.df1_linear.weight.data, 0.01)
        # self.df1_linear.bias.data.fill_(0)

        # self.df2_linear.weight.data = normalized_columns_initializer(
        #     self.df2_linear.weight.data, 0.01)
        # self.df2_linear.bias.data.fill_(0)

        # self.actor_linear.weight.data = normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)

        # self.critic_linear.weight.data = normalized_columns_initializer(
        #     self.critic_linear.weight.data, 5000)
        # self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = f.relu(self.conv1(inputs))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        # x = f.relu(self.conv4(x))
        x = x.view(-1, x.shape[1] * x.shape[3])
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        # return self.mean_linear(x), self.std_linear(x), (hx, cx), self.critic_linear(x)
        return self.df1_linear(x), self.df2_linear(x), (hx, cx), self.critic_linear(x)

    def choose_action(self, inputs):
        s, (hx, cx) = inputs
        df1, df2, lstm_cells, critic_v = self.forward((s, (hx, cx)))
        # mean, sigma, lstm_cells, critic_v = self.forward((s, (hx, cx)))
        df1 = f.softmax(df1, dim=-1)
        df2 = f.softmax(df2, dim=-1)
        # sigma = sigma.view(2, )
        # std = torch.exp(sigma).view(2, )
        # mean = mean.view(2, )

        p_index_list = []
        d_index_list = []
        # value = 0
        policy_v = None
        for i in range(10):

            # p_index = np.random.normal(loc=mean[1].item(), scale=std[1].item(), size=1)[0]
            # d_index = np.random.normal(loc=mean[0].item(), scale=std[0].item(), size=1)[0]
            p_index = df1.multinomial(num_samples=1).detach()[0].item()
            d_index = df2.multinomial(num_samples=1).detach()[0].item()
            p_index_list.append(p_index)
            d_index_list.append(d_index)
            policy_v = -(torch.log(df1[0][p_index]) * torch.log(df2[0][d_index]))
            # value += (-(sigma[1].item() / 2) - (((p_index-mean[1].item())**2) / (2*std[0].item())))
            # value += (-(sigma[0].item() / 2) - (((d_index - mean[0].item()) ** 2) / (2 * std[1].item())))
        # value = value / 10
        actions = []
        for j in range(len(p_index_list)):
            actions.append((p_index_list[j], d_index_list[j]))
        actions = list(set(actions))
        return actions, policy_v, lstm_cells, critic_v
