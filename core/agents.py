# -*- coding : utf-8 -*-
# @Time :  2022/7/27 15:13
# @Author : hxc
# @File : agents.py
# @Software : PyCharm
import numpy as np

from net.agent_model import ActorCritic
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(object):
    def __init__(self, args, agent_id):
        self.args = args
        self.state = None
        self.action = None
        self.model = None
        self.hx = torch.zeros(1, 64).to(device)
        self.cx = torch.zeros(1, 64).to(device)
        self.action_list = []
        self.state_list = []
        self.agent_id = agent_id
        self.values = []
        self.critic_values = []

    def get_state(self, state):
        state = state.astype(np.float32)
        state = torch.from_numpy(state).view(1, state.shape[0], 1, state.shape[1]).to(device)
        self.state = state
        self.state_list.append(state)

    def get_action(self):
        action, value, (self.hx, self.cx), critic_v = self.model.choose_action((self.state, (self.hx, self.cx)))
        # self.action = action.to(device)  # 如果模型里action变成tensor  要加载到gpu
        self.action = action
        self.action_list.append(action)
        self.values.append(value)
        self.critic_values.append(critic_v)

    def get_model(self, in_channels, action_space, agent_id):
        torch.manual_seed(self.args.seed)
        model = ActorCritic(in_channels, action_space, agent_id)
        self.model = model.to(device)

    def reset(self):
        self.state = None
        self.action = None
        self.hx = torch.zeros(1, 64).to(device)
        self.cx = torch.zeros(1, 64).to(device)
        self.action_list = []
        self.state_list = []
        self.values = []
        self.critic_values = []
