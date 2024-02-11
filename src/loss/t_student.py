import torch
import torch.nn as nn
import torch.nn.functional as F


class T_STUDENT(nn.Module):
    def __init__(self, var_weight):
        super(T_STUDENT, self).__init__()
        self.var_weight = var_weight

    def forward(self, results, label):
        mean, var, v = results['mean'], results['var'], results['v']
        
        n = results['mean'].numel() # Number of pixels (/numel gives the number of elements)
        var = torch.exp(var)
        v = torch.exp(v)

        var = self.var_weight * var

        loss = - n * 0.5 * torch.log(torch.tensor(torch.pi)) \
               + n * torch.lgamma((v + 1) * 0.5) \
               - n * torch.lgamma(v * 0.5) \
               - n * 0.5 * torch.log(v) \
               - n * torch.log(var) \
               - (v + 1) * 0.5 * torch.log(1 + ((label - mean) ** 2) / (v * var**2))
        
        return - loss.mean()

