import torch
import torch.nn as nn
import torch.nn.functional as F


class T_STUDENT(nn.Module):
    def __init__(self):
        super(T_STUDENT, self).__init__()

    def forward(self, results, label):
        mean, t, v = results['mean'], results['t'], results['v']
        t = torch.exp(t)
        v = torch.exp(v)

        #loss = - torch.lgamma((v + 1) * 0.5) \
        #        + torch.lgamma(v * 0.5) \
        #        + 0.5 * torch.log(v * torch.tensor(torch.pi)) \
        #        + 0.5 * torch.log(torch.tensor(torch.pi)) \
        #        + 0.5 * torch.log(t) \
        #        + 0.5 * (v + 1) * torch.log(1 + ((label - mean) ** 2) / (v * t**2))
        
        loss = - torch.lgamma((v + 1) * 0.5) \
               + torch.lgamma(v * 0.5) \
               + 0.5 * torch.log(torch.tensor(torch.pi)) * v \
               + torch.log(t) \
               + 0.5 * (v + 1) * torch.log(1 + ((label - mean) ** 2) / (v * t**2))
        
        return loss.sum()

