# -*- coding : utf-8 -*-
# @Time :  2022/7/27 15:13
# @Author : hxc
# @File : env.py
# @Software : PyCharm
import random
import numpy as np
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Environment:
    def __init__(self, args, init_env):
        self.args = args
        self.init_env = init_env
        self.doc_list = self.init_env.doctors
        self.agent = self.init_env.agent
        self.agent.get_model(self.init_env.p_num, self.args.max_reg_num, self.agent.agent_id)
        self.state = self.init_env.state
        self.task_num = self.init_env.task_num
        self.agent.model.train()

        self.p_reg_num = self.init_env.p_reg_num

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

        # 还需要处理的病人数量
        self.pro_p_num = self.init_env.reg_file.groupby("pid").count().shape[0]
        self.pid_list = [i for i in range(1, self.init_env.p_num+1)]
        self.ordered_pid_list = np.random.choice(a=self.pid_list, replace=False, p=None, size=self.pro_p_num)

    def step(self):
        for pid in self.ordered_pid_list:
            reward = 0
            # get action by state
            self.agent.get_action(self.state, pid, self.doc_list)
            action = self.agent.action
            is_reg = np.where(self.state[pid-1] == 1)[0]
            if action >= len(is_reg):
                reward -= 1
                p_pos = 0
                d_pos = 0
            else:
                doctor = self.doc_list[is_reg[action]+1]
                time_block = doctor.time_block
                last_schedule_list = (self.last_schedule_a if time_block == 0 else self.last_schedule_p)
                if last_schedule_list[0][pid-1] == 0:

                    if doctor.free_pos == 0:
                        last_time = 0
                        start_time = 0
                    else:
                        last_time = doctor.schedule_list[3][doctor.free_pos - 1] + self.d_free_time
                        start_time = last_time
                else:
                    last_time = doctor.schedule_list[3][doctor.free_pos - 1] + self.d_free_time
                    if last_schedule_list[1][pid-1] <= last_time:
                        start_time = last_time
                    else:
                        if last_schedule_list[1][pid-1] - last_time > 8:
                            if self.pro_p_num > 5:
                                reward += 1 - ((last_schedule_list[1][pid-1] - last_time) / 10)
                                self.agent.rewards.append(reward)
                                continue
                        start_time = last_schedule_list[1][pid - 1]
                pro_time = doctor.avg_pro_time
                finish_time = start_time + pro_time
                insert_data = [pid, start_time, pro_time, finish_time]
                doctor.insert_patient(insert_data, doctor.free_pos)
                doctor.total_time = finish_time
                doctor.free_pos += 1

                # 记录当前结束的时间
                last_time_list = (self.last_time_a if time_block == 0 else self.last_time_p)
                last_time_list[pid-1] = finish_time

                last_schedule_list[0][pid-1] += 1
                # 随机加一个2到5单位的转换时间
                last_schedule_list[1][pid-1] = (finish_time + random.randint(2, 5))

                time_span = abs((start_time - last_time)) + abs((start_time - last_schedule_list[1][pid-1]))

                reward += 10 + (1 - (time_span / 10))

                self.update_states(pid-1, doctor.doc_id-1)
                self.p_reg_num[0][pid - 1] -= 1

                if np.where(self.state[pid-1] == 1)[0].tolist():
                    pass
                else:
                    self.pid_list.remove(pid)
                    self.pro_p_num -= 1
                    assert self.pro_p_num == len(self.pid_list)

                p_pos = pid-1
                d_pos = doctor.doc_id-1

            self.agent.rewards.append(reward)

            # self.run_render(p_pos, d_pos)

        if len(self.pid_list) == 0:
            self.done = True
        else:
            self.update_sequence()

        return self.done

    def update_states(self, p_index, d_index):
        self.state[p_index][d_index] = 0

    def reset_h_c(self):
        self.agent.hx = torch.zeros(1, 128).to(device)
        self.agent.cx = torch.zeros(1, 128).to(device)

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
        self.p_reg_num = self.init_env.p_reg_num
        self.pro_p_num = self.init_env.reg_file.groupby("pid").count().shape[0]
        self.pid_list = [i for i in range(1, self.init_env.p_num + 1)]
        self.ordered_pid_list = np.random.choice(a=self.pid_list, replace=False, p=None, size=self.pro_p_num)

    def update_sequence(self):
        dis_info = self.p_reg_num[0] * 0.5 + self.p_reg_num[1] * 0.5
        dis_info_index = np.where(dis_info != 0)[0]
        dis_info = np.power(dis_info[dis_info_index], 4)
        prob = f.softmax(torch.from_numpy(dis_info).to(device), dim=-1).cpu().numpy().reshape(-1, ).tolist()
        self.ordered_pid_list = np.random.choice(a=self.pid_list, replace=False, p=prob, size=self.pro_p_num)
        # self.ordered_pid_list = np.random.choice(a=self.pid_list, replace=False, p=None, size=self.pro_p_num)

    def render(self):
        obs = np.ones((90 * 5, 10 * 20, 3))
        for i in range(90):
            for j in range(10):
                # if self.state[:, 2:][i, j] == 0:
                if self.state[i, j] == 0:
                    cv2.rectangle(obs, (j*20, i*5), (j*20+20, i*5+5), (0, 0, 0), -1)
        if self.pos[0] == self.pos[1] == -1:
            p = 0
            d = 0
        else:
            p = self.pos[0]
            d = self.pos[1]
        cv2.rectangle(obs, (d * 20, p * 5),
                      (d * 20 + 20, p * 5 + 5), (0, 0, 255), -1)
        cv2.imshow('image', obs)
        cv2.moveWindow("image", 10, 10)
        cv2.waitKey(10)

    def run_render(self, p_index, d_index):
        self.pos = [p_index, d_index]
        plt.imshow(self.agent.state[0].cpu().reshape((90, 10)))
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        self.render()
