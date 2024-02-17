import os
import torch
import torch.nn as nn
import torch.nn.parallel as P
from importlib import import_module



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        print('Making model...')

        self.is_train = config.is_train
        self.num_gpu = config.num_gpu
        self.uncertainty = config.uncertainty
        self.n_samples = config.n_samples
        module = import_module('models.' + config.uncertainty)
        self.model = module.make_model(config).to(config.device)

    def forward(self, input):
        if self.model.training:
            if self.num_gpu > 1:
                return P.data_parallel(self.model, input,
                                       list(range(self.num_gpu)))
            else:
                return self.model.forward(input)
        else:
            forward_func = self.model.forward
            if self.uncertainty == 'normal':
                return forward_func(input)
            elif self.uncertainty == 'aleatoric.gaussian':
                return self.test_aleatoric_gaussian(input, forward_func)
            elif self.uncertainty == 'aleatoric.laplacian':
                return self.test_aleatoric_laplacian(input, forward_func)
            elif self.uncertainty == 'aleatoric.tstudent':
                return self.test_aleatoric_tstudent(input, forward_func)
            else:
                raise Exception('Not implemented')
            
    def test_aleatoric_gaussian(self, input, forward_func):
        results = forward_func(input)
        mean = results['mean']
        var = results['var']

        # Compute variance
        var = torch.exp(var)
        #var_norm = var / var.max() #?

        new_results = {'mean': mean, 'var': var}
        return new_results
    
    def test_aleatoric_laplacian(self, input, forward_func):
        results = forward_func(input)
        mean = results['mean']
        scale = results['scale']

        # Compute variance
        scale = torch.exp(scale)
        #scale_norm = scale / scale.max() #?
        var = 2 * scale ** 2

        new_results = {'mean': mean, 'var': var}
        return new_results
    
    def test_aleatoric_tstudent(self, input, forward_func):
        results = forward_func(input)
        mean = results['mean']
        t = results['t']
        v = results['v']
        
        # Compute variance
        t = torch.exp(t)
        v = torch.exp(v)
        #var_norm = var / var.max()
        #v_norm = v / v.max()
        var = (v * t ** 2) / (v - 2)
        
        new_results = {'mean': mean, 'var': var}
        return new_results

    def save(self, ckpt, epoch):
        save_dirs = [os.path.join(ckpt.model_dir, 'model_latest.pt')]
        save_dirs.append(
            os.path.join(ckpt.model_dir, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, ckpt, cpu=False):
        epoch = ckpt.last_epoch
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        if epoch == -1:
            load_from = torch.load(
                os.path.join(ckpt.model_dir, 'model_latest.pt'), **kwargs)
        else:
            load_from = torch.load(
                os.path.join(ckpt.model_dir, 'model_{}.pt'.format(epoch)), **kwargs)
        if load_from:
            self.model.load_state_dict(load_from, strict=False)
