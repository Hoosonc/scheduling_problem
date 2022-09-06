# -*- coding : utf-8 -*-
# @Time :  2022/7/28 22:15
# @Author : hxc
# @File : test.py
# @Software : PyCharm
import numpy as np
import torch
import torch.nn.functional as f
import math
import csv
import pandas as pd
import datetime


def save_data(file_name, data_list):
    with open(f'./data/save_data/{file_name}.csv', mode='a+', encoding='utf-8-sig', newline='') as f:
        csv_writer = csv.writer(f)
        headers = ['pid', 'am', 'pm', "am_num", "pm_num"]
        csv_writer.writerow(headers)

        csv_writer.writerows(data_list)
        print(f'保存结果文件')

if __name__ == '__main__':
    a = pd.read_csv("./data/save_data/2022-9-5-17-30-28.csv")
    b = pd.read_csv("./data/save_data/2022-9-5-17-31-25.csv")
    c = pd.read_csv("./data/save_data/2022-9-5-21-47-48.csv")
    g1 = c.groupby("pid")
    am = [1, 3, 6, 8, 10]
    pm = [2, 4, 5, 7, 9]
    data_list = []
    for i in g1:
        am_s_list = []
        am_f_list = []
        pm_s_list = []
        pm_f_list = []
        pid = i[0]
        a_num = 0
        p_num = 0
        for row in i[1].values:
            if row[0] in am:
                am_s_list.append(row[2])
                am_f_list.append(row[4])
                a_num += 1
            else:
                pm_s_list.append(row[2])
                pm_f_list.append(row[4])
                p_num += 1
        if a_num == 0:
            a_all_time = 0
        else:
            a_all_time = max(am_f_list)-min(am_s_list)
        if p_num == 0:
            p_all_time = 0
        else:
            p_all_time = max(pm_f_list)-min(pm_s_list)
        data_list.append([pid, a_all_time, p_all_time, a_num, p_num])
    save_data("3", data_list)


