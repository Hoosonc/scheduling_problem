# -*- coding : utf-8 -*-
# @Time :  2022/8/18 15:55
# @Author : hxc
# @File : gen_data.py
# @Software : PyCharm
import numpy as np
import csv
import pandas as pd
# from net.utils import get_now_date as hxc


def save_data(header, data, file_name):
    with open(f'../data/{file_name}.csv', mode='a+', encoding='utf-8-sig', newline='') as f:
        csv_writer = csv.writer(f)
        header = header
        csv_writer.writerow(header)
        csv_writer.writerows(data)
        # print(f'保存文件')


def gen_doctors():
    doc_list = []
    for i in range(1, 11):
        am = 0
        pm = 0
        did = i
        reg_num_list = [20, 30]
        op_time1 = [7, 8, 9]
        op_time2 = [4, 5, 3]
        reg_num = np.random.choice(a=reg_num_list, size=1, replace=False, p=None)[0]
        if reg_num == 20:
            avg_pro_time = np.random.choice(a=op_time1, size=1, replace=False, p=None)[0]
        else:
            avg_pro_time = np.random.choice(a=op_time2, size=1, replace=False, p=None)[0]
        if am > 6:
            start_time = 300
        elif pm > 6:
            start_time = 0
        else:
            start_time = np.random.choice(a=[0, 300], size=1, replace=False, p=None)[0]
            if start_time == 0:
                am += 1
            else:
                pm += 1

        doc = [did, reg_num, start_time, 0, 0, 0, avg_pro_time]
        doc_list.append(doc)
    return doc_list


def gen_patient():
    patient_list = []
    pid_list = [pid for pid in range(1, 91)]
    p_reg_num = np.zeros((90,))
    d_reg_num = np.zeros((10,))
    d_p = [[] for did in range(10)]
    doc_file = pd.read_csv("../data/doc_new.csv", encoding="utf-8-sig")
    max_num = 3
    while True:
        if len(patient_list) == 300:
            break
        if len(np.where(p_reg_num == 4)[0]) == 40:
            max_num = 2
            up_id = np.where(p_reg_num > 2)[0]
            for j in up_id:
                if j in pid_list:
                    patient_list.remove(j)
        if len(np.where(p_reg_num > 2)[0]) > 70:
            max_num = 1
            up_id = np.where(p_reg_num > 2)[0]
            for j in up_id:
                if j in pid_list:
                    patient_list.remove(j)
        for doc in doc_file.values:
            if d_reg_num[doc[0]-1] == doc[1]:
                continue
            temp_list = pid_list.copy()
            for i in d_p[doc[0]-1]:
                if i in temp_list:
                    temp_list.remove(i)
            if not temp_list:
                continue
            pid = np.random.choice(a=temp_list, size=1, replace=False, p=None)[0]
            p = [pid, doc[0], 0, 0, 0 if doc[2] == 0 else 1]
            d_p[doc[0]-1].append(pid)
            p_reg_num[pid - 1] += 1
            d_reg_num[doc[0]-1] += 1
            patient_list.append(p)
            if p_reg_num[pid - 1] > max_num:
                pid_list.remove(pid)

    return patient_list


if __name__ == '__main__':
    # doctors = gen_doctors()
    # doc_header = ['did', 'reg_num', 'start_time', 'Department', 'min_pro_time', 'max_pro_time', 'avg_pro_time']
    # save_data(doc_header, doctors, "doc_new")
    # data_list = gen_patient()
    # print(data_list)
    # p_header = ['pid', 'did', 'at', 'Department', 'time_block']
    # save_data(p_header, data_list, "reg_new")
    df = pd.read_csv("../data/reg_new.csv")
    a = df.groupby("pid").count()
    print(a)
