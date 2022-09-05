# -*- coding : utf-8 -*-
# @Time :  2022/7/27 14:13
# @Author : hxc
# @File : trainer.py
# @Software : PyCharm
import numpy as np
# import math

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
        self.total_time = 0
        self.batch = self.args.batch
        # self.eval_rewards = None

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

                assert len(self.agent.values) == len(self.agent.state_list) == \
                       len(self.agent.action_list) == len(self.agent.critic_values)

                if done:
                    print("总时间：", self.env.get_total_time())
                    print("步数：", step+1)
                    if episode != 0 and (episode + 1) % 100 == 0:
                        self.save_model()
                    episode_length = 0
                    self.total_time = self.env.get_total_time()
                    self.env.reset()
                    # print("env reset")
                    break

            if (episode+1) % 2 == 0:
                self.learn(episode)

    def learn(self, episode):
        rewards, log_pi, critic_v, batch, critic_v_next = self.get_batch()
        td_error = rewards + self.args.gamma * critic_v_next - critic_v
        value_loss = td_error.pow(2).mean()

        delta = rewards + self.args.gamma * critic_v_next - critic_v

        gae = self.args.gamma * self.args.gae_lambda * delta
        policy_loss = (-log_pi * gae).mean()

        self.optimizer.zero_grad()

        loss = policy_loss + self.args.value_loss_coefficient * value_loss

        loss.backward(torch.ones_like(policy_loss), retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        # print(loss, episode + 1)

        print("policy:", policy_loss, episode+1)
        print("value:", value_loss, episode+1)

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

    def get_batch(self):
        if len(self.agent.state_list) < self.batch:
            batch = len(self.agent.state_list)
        else:
            batch = self.batch
        random_index = np.random.choice(a=list(range(0, len(self.agent.state_list))), size=batch, replace=False, p=None)
        rewards = torch.from_numpy(np.array(self.agent.rewards)[random_index]).view(batch, 1).to(device)
        log_pi = torch.cat([self.agent.values[i].view(1, 1) for i in random_index]).view(batch, 1)
        critic_v = torch.cat([self.agent.critic_values[i].view(1, 1) for i in random_index]).view(batch, 1)

        # log_pi_next = self.agent.values.copy()
        # log_pi_next.append(torch.tensor([0.]).to(device))
        critic_v_next = self.agent.critic_values.copy()
        critic_v_next.append(torch.tensor([0.]).to(device))
        # log_pi_next = torch.cat([log_pi_next[i+1].view(1, 1) for i in random_index]).view(batch, 1)
        critic_v_next = torch.cat([critic_v_next[i+1].view(1, 1) for i in random_index]).view(batch, 1)

        # return rewards, log_pi, critic_v, batch, log_pi_next, critic_v_next
        return rewards, log_pi, critic_v, batch, critic_v_next
