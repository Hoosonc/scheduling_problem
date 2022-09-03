# -*- coding : utf-8 -*-
# @Time :  2022/7/27 15:30
# @Author : hxc
# @File : doctor.py
# @Software : PyCharm
import numpy as np


class Doctor:
    def __init__(self, args):
        self.args = args
        self.doc_id = 0
        self.start_time = 0
        self.time_block = 0
        self.reg_num = 0
        self.department_id = 0
        self.max_pro_time = 0
        self.min_pro_time = 0
        self.avg_pro_time = 0
        self.schedule_list = None
        self.free_pos = 0
        self.total_time = 0

    def insert_patient(self, insert_data, insert_index, delete_index=-1):
        self.schedule_list = np.delete(self.schedule_list, [delete_index], axis=1)  # axis = 0 删除选中行；axis = 1 删除选中列；
        self.schedule_list = np.insert(self.schedule_list, insert_index, insert_data, axis=1)
