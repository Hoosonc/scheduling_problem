# -*- coding : utf-8 -*-
# @Time :  2022/7/27 15:13
# @Author : hxc
# @File : env.py
# @Software : PyCharm
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Environment:
    def __init__(self, args, init_env):
        self.args = args
        self.init_env = init_env
        self.doc_list = self.init_env.doctors
        self.agent = self.init_env.agent
        # self.agent2 = self.init_env.agent2
        self.agent.get_model(self.init_env.d_num, 1, self.agent.agent_id)
        # self.agent2.get_model(self.init_env.d_num, self.init_env.d_num, self.agent2.agent_id)
        self.state = self.init_env.state
        self.task_num = self.init_env.task_num
        self.agent.model.train()
        self.last_time_a = np.zeros((self.init_env.p_num,))   # 上一个号的结束时间
        self.last_time_p = np.zeros((self.init_env.p_num,))   # 上一个号的结束时间

        self.last_schedule_a = np.zeros((2, self.init_env.p_num))  # 上一个号的结束时间 + 随机移动时间
        """
             [已处理号数]
             [该病人上一个号结束的时间+随机移动时间]]
        """
        self.last_schedule_p = np.zeros((2, self.init_env.p_num))  # 上一个号的结束时间
        self.d_free_time = 1  # 医生工作间隔时间
        self.done = False
        self.pos = []

    def step(self):
        
        # get action by state
        self.agent.get_state(self.state)
        self.agent.get_action()

        reward = 0
        p_index_list, d_index_list = self.get_index()
        # p_index, d_index = self.get_index()

        for i in range(len(p_index_list)):

            p_index = p_index_list[i]
            d_index = d_index_list[i]

            if p_index == d_index == -1:
                reward -= 10
            elif self.state[d_index][p_index] == 0:
                pass
            else:
                doctor = self.doc_list[d_index + 1]
                time_block = doctor.time_block
                last_schedule_list = (self.last_schedule_a if time_block == 0 else self.last_schedule_p)
                if last_schedule_list[0][p_index] == 0:

                    if doctor.free_pos == 0:
                        last_time = 0
                        start_time = 0
                    else:
                        last_time = doctor.schedule_list[3][doctor.free_pos - 1] + self.d_free_time
                        start_time = last_time
                else:
                    last_time = doctor.schedule_list[3][doctor.free_pos - 1] + self.d_free_time
                    if last_schedule_list[1][p_index] <= last_time:
                        start_time = last_time
                    else:
                        start_time = last_schedule_list[1][p_index]

                pro_time = doctor.avg_pro_time
                finish_time = start_time + pro_time
                insert_data = [(p_index + 1), start_time, pro_time, finish_time]
                doctor.insert_patient(insert_data, doctor.free_pos)
                doctor.total_time = finish_time
                doctor.free_pos += 1
                # 记录当前结束的时间
                last_time_list = (self.last_time_a if time_block == 0 else self.last_time_p)
                last_time_list[p_index] = finish_time

                last_schedule_list[0][p_index] += 1
                # 随机加一个2到5单位的转换时间
                last_schedule_list[1][p_index] = (finish_time + random.randint(2, 5))

                self.task_num -= 1
                reward += 1

                time_span = abs((start_time - last_time)) + abs((start_time - last_schedule_list[1][p_index]))
                # if time_span == 0:
                #     reward += 1
                # else:
                #     reward += 1 + (1 - (time_span / 20))
                reward += 1 + (1 - (time_span / 20))

                self.update_states(p_index, d_index)
            self.run_render(p_index, d_index)
        if self.task_num == 0:
            self.done = True

        return reward, self.done

    def update_states(self, p_index, d_index):
        self.state[d_index][p_index] = 0

    def reset_h_c(self):
        self.agent.hx = torch.zeros(1, 64).to(device)
        self.agent.cx = torch.zeros(1, 64).to(device)

    def get_total_time(self):
        total_time = 0
        for did in self.doc_list:
            doctor = self.doc_list[did]
            total_time += doctor.total_time
        return total_time

    def reset(self):
        self.agent.reset()
        self.init_env.reset()
        self.doc_list = self.init_env.doctors
        self.state = self.init_env.state
        self.task_num = self.init_env.task_num
        self.last_time_a = np.zeros((self.init_env.p_num,))  # 上一个号的结束时间
        self.last_time_p = np.zeros((self.init_env.p_num,))  # 上一个号的结束时间
        self.last_schedule_a = np.zeros((2, self.init_env.p_num))  # 上一个号的结束时间 + 随机移动时间
        self.last_schedule_p = np.zeros((2, self.init_env.p_num))  # 上一个号的结束时间
        self.done = False

    def get_index(self):
        # actions = self.agent.action.view(-1, ).cpu().numpy()
        # actions = np.unique(actions)
        p_index_list = []
        d_index_list = []
        actions = self.agent.action
        for action in actions:

            if action[0] < 0 or action[1] < 0:
                d_index = -1
                p_index = -1
            elif action[0] > (self.init_env.p_num-1) or action[1] > (self.init_env.d_num-1):
                d_index = -1
                p_index = -1
            else:
                p_index = action[0]
                d_index = action[1]
            p_index_list.append(p_index)
            d_index_list.append(d_index)

        return p_index_list, d_index_list
        # return p_index, d_index

    def render(self):
        obs = np.ones((10 * 15, 90 * 15, 3))
        for i in range(10):
            for j in range(90):
                # if self.state[:, 2:][i, j] == 0:
                if self.state[i, j] == 0:
                    cv2.rectangle(obs, (j*15, i*15), (j*15+15, i*15+15), (0, 0, 0), -1)
        if self.pos[0] == self.pos[1] == -1:
            p = 0
            d = 0
        else:
            p = self.pos[0]
            d = self.pos[1]
        cv2.rectangle(obs, (d * 15, p * 15),
                      (d * 15 + 15, p * 15 + 15), (0, 0, 255), -1)
        cv2.imshow('image', obs)
        cv2.waitKey(10)

    def run_render(self, p_index, d_index):
        self.pos = [p_index, d_index]
        plt.imshow(self.agent.state.cpu().reshape((10, 90)))
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        self.render()
