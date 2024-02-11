import torch
import torch.nn as nn
import torch.nn.functional as F


class LAPLACIAN(nn.Module):
    def __init__(self, var_weight):
        super(LAPLACIAN, self).__init__()
        self.var_weight = var_weight

    def forward(self, results, label):
        mean, scale = results['mean'], results['var']
        scale = self.var_weight * scale
        
        scale = torch.exp(scale)
        
        loss = torch.log(2*scale) + torch.abs(mean - label)/scale
        return loss.mean()

  