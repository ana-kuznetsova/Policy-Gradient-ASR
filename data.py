import os
import numpy as np
import torch
import torchaudio

def make_feats(path):
    waveform, sample_rate = torchaudio.load(path)
    #Calculate MFCC
    mfcc = torchaudio.transforms.MFCC()(waveform)
    #Calculate delta and double-delta
    deltas = torchaudio.transforms.ComputeDeltas()(mfcc)
    ddeltas = torchaudio.transforms.ComputeDeltas()(deltas)
    res = torch.cat((mfcc, deltas, ddeltas), dim=1)
    #Normalize rows
    s = torch.sum(res, dim=2, keepdim=True)
    norm = torch.div(res, s)
    return norm