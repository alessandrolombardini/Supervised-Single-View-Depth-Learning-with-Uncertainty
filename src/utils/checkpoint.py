import os
import sys
import numpy as np
from datetime import datetime
from functools import reduce

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.nn.modules.module import _addindent

class Checkpoint:
    def __init__(self, config):
        self.global_step = 0
        self.test_step = 0
        self.last_epoch = 0

        self.config = config
        self.exp_dir = config.exp_dir
        self.data_name = config.data_name
        self.exp_load = config.exp_load
        exp_type = config.uncertainty
        #now = datetime.now().strftime('%m%d_%H%M')

        if config.exp_load is None:
            #dir_fmt = '{}/{}_{}'.format(config.data_name, exp_type, now)
            dir_fmt = 'checkpoints/{}/{}'.format(self.data_name, exp_type)
        else:
            dir_fmt = 'checkpoints/{}/{}_{}'.format(self.data_name, exp_type, self.exp_load)

        self.model_dir = os.path.join(self.exp_dir, dir_fmt, 'model')
        self.log_dir = os.path.join(self.exp_dir, dir_fmt, 'log')
        self.save_dir = os.path.join(self.exp_dir, dir_fmt, 'save')
        self.ckpt_dir = os.path.join(self.log_dir, 'ckpt.pt')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        # save config
        self.config_file = os.path.join(self.log_dir, 'config.txt')
        with open(self.config_file, 'w') as f:
            for k, v in vars(config).items():
                f.writelines('{}: {} \n'.format(k, v))

    def step(self):
        self.global_step += 1
        return self.global_step
    
    def do_step_test(self):
        self.test_step += 1
        return self.test_step

    def save(self, epoch):
        self.last_epoch = epoch
        save_ckpt = {'global_step': self.global_step,
                     'last_epoch': self.last_epoch}
        torch.save(save_ckpt, self.ckpt_dir)

    def load(self):
        load_ckpt = torch.load(self.ckpt_dir)
        self.global_step = load_ckpt['global_step']
        self.last_epoch = load_ckpt['last_epoch']
