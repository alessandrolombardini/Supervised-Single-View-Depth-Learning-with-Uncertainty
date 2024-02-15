import torch
import torch.nn as nn
import torch.nn.functional as F


class LAPLACIAN(nn.Module):
    def __init__(self):
        super(LAPLACIAN, self).__init__()

    def forward(self, results, label):
        mean, scale = results['mean'], results['scale']
        
        scale = torch.exp(scale)
        
        loss = torch.log(2*scale) + torch.abs(mean - label)/scale
        return loss.mean()

  