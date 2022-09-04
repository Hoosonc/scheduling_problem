# -*- coding : utf-8 -*-
# @Time :  2022/7/27 14:13
# @Author : hxc
# @File : trainer.py
# @Software : PyCharm
# import numpy as np
import math

from core.env import Environment
import torch
import torch.optim as opt
import csv
from core.init_env import InitEnv
from net.utils import get_now_date as hxc
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, args):
        self.args = args
        self.init_env = InitEnv(args)
        self.env = Environment(args, self.init_env)
        self.agent = self.env.agent
        self.model = self.agent.model
        # self.load_params()
        self.optimizer = opt.Adam(self.model.parameters(), lr=args.lr)
        self.rewards = []
        self.total_time = 0
        self.eval_rewards = None

    def train(self):
        env = self.env

        episode_length = 0
        done = True
        plt.figure(figsize=(3, 3))

        for episode in range(self.args.episode):

            if done:
                # 重置记忆细胞
                env.reset_h_c()
            else:
                self.agent.cx = self.agent.cx.detach()
                self.agent.hx = self.agent.hx.detach()

            for step in range(self.args.num_steps):
                episode_length += 1

                done = env.step()

                done = done or episode_length >= (self.args.num_steps * self.args.update_episode_length)

                # if episode != 0 and (episode + 1) % 1 == 0 and step+1 == self.args.num_steps:
                #     print("剩余任务:", self.env.task_num, "步数:", step + 1)
                #     print("总时间：", self.env.get_total_time())

                if done:
                    print("剩余任务:", self.env.task_num, "步数:", step + 1)
                    print("总时间：", self.env.get_total_time())
                    # if episode != 0 and (episode + 1) % 10 == 0:
                    #     print("剩余任务:", self.env.task_num, "步数:", step+1)
                    #     print("总时间：", self.env.get_total_time())
                    # if episode != 0 and (episode + 1) % 100 == 0:
                    #     self.save_reward_action("第五次训练_" + str(episode) + "_" + str(step))

                    # if episode != 0 and (episode+1) % 100 == 0:
                    #     self.save_data("第五次训练_" + str(episode))
                    #     print("episode:", episode+1)
                    if episode != 0 and (episode + 1) % 100 == 0:
                        self.save_model()
                    episode_length = 0
                    self.total_time = self.env.get_total_time()
                    self.learn(episode)
                    self.env.reset()
                    self.rewards = []
                    # print("env reset")
                    break

    def learn(self, episode):
        critic_v = self.agent.critic_values
        f_s_a_value = self.agent.values
        u = torch.zeros(1, 1).to(device)
        if self.env.task_num != 0:
            _, _, _, value = self.model((self.agent.state, (self.agent.hx, self.agent.cx)))  # 模型的估计值
            u = value.detach()
        critic_v.append(u)
        policy_loss = torch.zeros(1, 1).to(device)
        value_loss = 0
        gae = (torch.zeros(1, 1)).to(device)
        # mean_c = torch.cat(critic_v).mean()
        for i in reversed(range(len(self.rewards))):
            u = self.args.gamma * u + self.rewards[i]
            advantage = u - critic_v[i]  # 每一步的td target
            value_loss = value_loss + 0.5 * advantage.pow(2)  # td Loss

            # Generalized Advantage Estimation
            delta_t = self.rewards[i] + self.args.gamma * critic_v[i + 1] - critic_v[i]
            # delta_t = self.rewards[i] + self.args.gamma * critic_v[i + 1] - mean_c
            gae = gae * self.args.gamma * self.args.gae_lambda + delta_t

            policy_loss = policy_loss - f_s_a_value[i] * gae
            # policy_loss = policy_loss - f_s_a_value[i] * critic_v[i]
        self.optimizer.zero_grad()
        loss = policy_loss + self.args.value_loss_coefficient * value_loss
        # loss = policy_loss + value_loss
        loss.backward(torch.ones_like(policy_loss))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        # print(loss, episode + 1)
        if episode != 0 and (episode+1) % 1 == 0:
            print("policy:", policy_loss, episode+1)
            print("value:", value_loss, episode+1)
            # print(loss, episode+1)

        # print('para updated')

    def save_model(self):
        torch.save(self.model.state_dict(), f'./net/params/agent1/{hxc()}.pth')

    def save_data(self, file_name):
        with open(f'./data/save_data/{file_name}.csv', mode='a+', encoding='utf-8-sig', newline='') as f:
            csv_writer = csv.writer(f)
            headers = ['did', 'pid', 'start_time', 'pro_time', 'finish_time']
            csv_writer.writerow(headers)
            data_set = []
            for did in self.env.doc_list:
                doctor = self.env.doc_list[did]
                data = doctor.schedule_list
                for i in range(doctor.free_pos):
                    item = [did, data[0][i], data[1][i], data[2][i], data[3][i]]
                    data_set.append(item)
                # total_time += doctor.total_time
            csv_writer.writerows(data_set)
            print(f'保存结果文件')

    def load_params(self):
        self.model.load_state_dict(torch.load('./net/params/agent1/2022-8-26-16-48-43.pth'))

    def save_reward_action(self, file_name):
        with open(f'./data/save_data/reward_action/{file_name}.csv', mode='a+',
                  encoding='utf-8-sig', newline='') as f1:
            csv_writer = csv.writer(f1)
            headers = ['reward', 'p_index', 'd_index', "step"]
            csv_writer.writerow(headers)
            data_set = []
            for i in range(len(self.rewards)):
                for j in range(len(self.agent.action_list[i])):
                    action = self.agent.action_list[i][j]
                # action = self.agent.action_list[i].numpy().reshape(2, )
                    item = [self.rewards[i], action[0], action[1], i]
                    data_set.append(item)
            csv_writer.writerows(data_set)
            print(f'保存reward_action文件')
