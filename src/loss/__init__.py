import torch
import torch.nn as nn
from importlib import import_module


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        print('Preparing loss function...')

        self.num_gpu = config.num_gpu
        self.losses = []
        self.loss_module = nn.ModuleList()

        if config.uncertainty == 'epistemic' or config.uncertainty == 'normal':
            module = import_module('loss.mse')
            loss_function = getattr(module, 'MSE')()
        elif config.uncertainty == 'combined':
            module = import_module('loss.gaussian')
            loss_function = getattr(module, 'GAUSSIAN')()
        elif config.uncertainty == 'aleatoric_gaussian':
            module = import_module('loss.gaussian')
            loss_function = getattr(module, 'GAUSSIAN')()
        elif config.uncertainty == 'aleatoric_laplacian':
            module = import_module('loss.laplacian')
            loss_function = getattr(module, 'LAPLACIAN')()
        elif config.uncertainty == 'aleatoric_tstudent':
            module = import_module('loss.t_student')
            loss_function = getattr(module, 'T_STUDENT')()

        self.losses.append({'function': loss_function})

        self.loss_module.to(config.device)
        if not config.cpu and config.num_gpu > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(self.num_gpu))

    def forward(self, results, label):
        losses = []
        for i, l in enumerate(self.losses):
            if l['function'] is not None:
                loss = l['function'](results, label)
                effective_loss = loss
                losses.append(effective_loss)

        loss_sum = sum(losses)
        if len(self.losses) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum
