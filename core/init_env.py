# -*- coding : utf-8 -*-
# @Time :  2022/7/27 14:01
# @Author : hxc
# @File : init_env.py
# @Software : PyCharm
import numpy as np
import pandas as pd
from core.agents import Agent
from core.doctor import Doctor


class InitEnv:
    def __init__(self, args):
        self.args = args
        self.doc_file = pd.read_csv(args.doc_path, encoding='utf-8-sig').fillna('')
        self.reg_file = pd.read_csv(args.reg_path, encoding='utf-8-sig').fillna('')
        self.d_num = self.doc_file.groupby("did").count().shape[0]
        self.p_num = self.reg_file.groupby("pid").count().shape[0]

        # self.state = np.zeros((self.d_num, self.p_num + 2), dtype="float32")
        self.state = np.zeros((self.d_num, self.p_num), dtype="float32")
        self.p_reg_num = np.zeros((2, self.p_num))
        self.agent = Agent(args, 1)
        self.task_num = self.reg_file.shape[0]
        self.doctors = self.init_doctor_list()
        self.init_state()
        self.get_p_reg_num()

    def init_state(self):
        reg_file = self.reg_file
        for patient in reg_file.values:
            pid = patient[0]
            did = patient[1]
            self.state[did-1][pid-1] = 1

    def init_doctor_list(self):
        doc_file = self.doc_file
        all_doc = doc_file.groupby('did')
        doctors = {}
        for doc in all_doc:
            doctor = Doctor(self.args)
            doctor.doc_id = doc[0]
            doc_info = doc[1]
            doctor.reg_num = doc_info['reg_num'].values[0]
            doctor.start_time = doc_info['start_time'].values[0]
            doctor.time_block = 0 if doctor.start_time == 0 else 1
            doctor.min_pro_time = doc_info['min_pro_time'].values[0]
            doctor.max_pro_time = doc_info['max_pro_time'].values[0]
            doctor.avg_pro_time = doc_info['avg_pro_time'].values[0]
            doctor.department_id = doc_info['Department'].values[0]
            doctor.schedule_list = np.zeros((4, doctor.reg_num))
            doctors[doctor.doc_id] = doctor
        return doctors

    def get_p_reg_num(self):
        reg_file = self.reg_file
        for patient in reg_file.values:
            pid = patient[0]
            self.p_reg_num[1][pid-1] += 1

    def reset(self):
        self.state = np.zeros((self.d_num, self.p_num), dtype="float32")
        self.p_reg_num = np.zeros((2, self.p_num))
        self.task_num = self.reg_file.shape[0]
        self.doctors = self.init_doctor_list()
        self.init_state()
        self.get_p_reg_num()
