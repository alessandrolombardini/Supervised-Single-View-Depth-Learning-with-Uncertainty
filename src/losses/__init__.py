import torch.nn as nn
from importlib import import_module


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        print('Preparing loss function...')

        self.num_gpu = config.num_gpu
        self.losses = []
        self.loss_module = nn.ModuleList()

        if config.uncertainty == 'normal':
            module = import_module('losses.mse')
            loss_function = getattr(module, 'MSE')()
        elif config.uncertainty == 'aleatoric.gaussian':
            module = import_module('losses.gaussian')
            loss_function = getattr(module, 'GAUSSIAN')()
        elif config.uncertainty == 'aleatoric.laplacian':
            module = import_module('losses.laplacian')
            loss_function = getattr(module, 'LAPLACIAN')()
        elif config.uncertainty == 'aleatoric.tstudent':
            module = import_module('losses.t_student')
            loss_function = getattr(module, 'T_STUDENT')()
        else:
            raise Exception('Not implemented')
        
        self.losses.append({'function': loss_function})

        self.loss_module.to(config.device)
        if config.device != 'cpu' and config.num_gpu > 1:
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
