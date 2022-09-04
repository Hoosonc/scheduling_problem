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
        self.hx = torch.zeros(1, 128).to(device)
        self.cx = torch.zeros(1, 128).to(device)
        self.action_list = []
        self.state_list = []
        self.agent_id = agent_id
        self.values = []
        self.rewards = []
        self.critic_values = []
        self.doctor_list = None
        """
            doc_info:
                [[剩余号情况， 时间占用情况]
                         ...         ]]
        """
        self.doc_info = None

    def get_doc_info(self, pid, state):
        p_reg = state[pid-1]
        is_reg = np.where(p_reg == 1)[0]
        total_rows = []
        if is_reg.tolist():
            for index in is_reg:
                doctor = self.doctor_list[index+1]
                reg_v = doctor.free_pos / doctor.reg_num
                if doctor.free_pos == 0:
                    time_v = 0.
                else:
                    time_v = doctor.schedule_list[3][doctor.free_pos-1]
                row = [reg_v, time_v]
                total_rows.append(row)
        if len(total_rows) < self.args.max_reg_num:
            for i in range(self.args.max_reg_num - len(total_rows)):
                total_rows.append([1., 1.])
        total_rows = torch.from_numpy(np.array(total_rows, dtype="float32")).view(4, 2).to(device)
        return total_rows

    def get_state(self, state, pid):
        doc_info = self.get_doc_info(pid, state)
        state = state.astype(np.float32)
        state = torch.from_numpy(state).view(1, state.shape[0], 1, state.shape[1]).to(device)
        self.state = (state, doc_info)
        self.state_list.append(self.state)

    def get_action(self, state, pid, doc_list):
        self.doctor_list = doc_list
        self.get_state(state, pid)

        action, log_pi, (self.hx, self.cx), critic_v = self.model.choose_action((self.state, (self.hx, self.cx)))
        # self.action = action.to(device)  # 如果模型里action变成tensor  要加载到gpu
        self.action = action
        self.action_list.append(action)
        self.values.append(log_pi)
        self.critic_values.append(critic_v)

    def get_model(self, in_channels, action_space, agent_id):
        torch.manual_seed(self.args.seed)
        model = ActorCritic(in_channels, action_space, agent_id)
        self.model = model.to(device)

    def reset(self):
        self.state = None
        self.action = None
        self.hx = torch.zeros(1, 128).to(device)
        self.cx = torch.zeros(1, 128).to(device)
        # self.action_list = []
        # self.state_list = []
        # self.values = []
        # self.critic_values = []
