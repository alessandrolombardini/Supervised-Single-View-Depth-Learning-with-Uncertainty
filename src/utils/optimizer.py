import os
import sys
import numpy as np
from datetime import datetime
from functools import reduce

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.nn.modules.module import _addindent


def make_optimizer(config, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    kwargs_optimizer = {'lr': config.lr, 'weight_decay': config.weight_decay}

    if config.optimizer == 'sgd':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = config.momentum
    elif config.optimizer == 'adam':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = config.betas
        kwargs_optimizer['eps'] = config.epsilon
    elif config.optimizer == 'rmsprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = config.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), config.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': config.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, ckpt):
            save_dir = os.path.join(ckpt.model_dir, 'optimizer.pt')
            torch.save(self.state_dict(), save_dir)

        def load(self, ckpt):
            load_dir = os.path.join(ckpt.model_dir, 'optimizer.pt')
            epoch = ckpt.last_epoch
            self.load_state_dict(torch.load(load_dir))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer
