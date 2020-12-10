import torch.nn as nn
import torch
import torch.nn.functional as F

class customNLLLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index
        if self.ignore_index:
            self.loss = nn.NLLLoss(ignore_index=self.ignore_index)
        else:
            self.loss = nn.NLLLoss()
    def forward(self, inp, target):
        loss = 0
        for i, char in enumerate(inp):
            loss+=self.loss(char,target[:,i])
        return loss        

    