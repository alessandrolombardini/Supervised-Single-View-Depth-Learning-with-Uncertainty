import torch
import torch.nn as nn
import torch.nn.functional as F


class GAUSSIAN(nn.Module):
    def __init__(self):
        super(GAUSSIAN, self).__init__()

    def forward(self, results, label):
        mean, var = results['mean'], results['var']

        loss1 = torch.mul(torch.exp(-var), (mean - label) ** 2)
        loss2 = var
        loss = .5 * (loss1 + loss2)
        return loss.sum()

        #return loss.mean()

        #loss = (mean - label)**2 / (2 * var) + .5 * torch.log(var)
        #return loss.sum()

