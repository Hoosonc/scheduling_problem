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
                    print("总时间：", self.env.get_total_time(),
                          "步数：", step+1,
                          "episode:", episode,
                          "剩余病人：", self.env.pro_p_num)

                    # if episode != 0 and (episode + 1) % 50 == 0:
                    #     self.save_model()
                    episode_length = 0
                    if self.env.pro_p_num == 0:
                        self.agent.temp_critic.append(torch.tensor([0]).to(device))

                    else:
                        pid = np.random.choice(a=self.env.pid_list, p=None, size=1)[0]
                        self.agent.get_state(self.env.state, pid)

                        _, _, _, critic_v = self.model.choose_action((self.agent.state, (self.agent.hx, self.agent.cx)))
                        self.agent.temp_critic.append(critic_v)
                    self.agent.critic_next_values.extend(self.agent.temp_critic[1:])

                    if len(self.agent.rewards) > 10000:
                        self.learn(episode)
                        self.agent.state_list = []
                        self.agent.action_list = []
                        self.agent.values = []
                        self.agent.critic_values = []
                        self.agent.rewards = []
                        self.agent.critic_next_values = []
                    self.total_time = self.env.get_total_time()
                    # if (episode+1) % 10 == 0:
                    #     self.save_data(f"{hxc()}")
                    # if self.total_time < 1610:
                    #     self.save_data(f"{hxc()}")
                    self.env.reset()
                    # print("env reset")
                    break

    def learn(self, episode):
        rewards, log_pi, critic_v, critic_v_next = self.get_batch()
        td_error = rewards - critic_v
        value_loss = (0.5*td_error.pow(2)).mean()

        delta = rewards + self.args.gamma * critic_v_next - critic_v

        gae = self.args.gamma * self.args.gae_lambda * delta
        policy_loss = ((-log_pi) * gae).mean()

        self.optimizer.zero_grad()

        loss = policy_loss + self.args.value_loss_coefficient * value_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        # print(loss, episode + 1)
        print("value:", value_loss, episode + 1)
        print("policy:", policy_loss, episode+1)

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
        self.model.load_state_dict(torch.load('./net/params/agent1/2022-9-6-15-33-35.pth'))

    def get_batch(self):
        row_len = len(self.agent.rewards)
        rewards = torch.from_numpy(np.array(self.agent.rewards)).view(row_len, 1).to(device)
        log_pi = torch.cat([i.view(1, 1) for i in self.agent.values]).view(row_len, 1)
        critic_v = torch.cat([i.view(1, 1) for i in self.agent.critic_values]).view(row_len, 1)

        critic_v_next = self.agent.critic_values.copy()
        critic_v_next.append(torch.tensor([0.]).to(device))
        critic_v_next = torch.cat([i.view(1, 1) for i in critic_v_next[1:]]).view(row_len, 1)

        return rewards, log_pi, critic_v, critic_v_next
