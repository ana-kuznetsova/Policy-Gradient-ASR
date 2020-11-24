import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

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


def find_maxlen(path):
    fnames = os.listdir(path)
    maxlen = 0
    for n in tqdm(fnames):
        waveform, sample_rate = torchaudio.load(os.path.join(path, n))
        mfcc = torchaudio.transforms.MFCC()(waveform)
        size = mfcc.shape[2]
        if size > maxlen:
            maxlen = size
    print("Maxlen:", maxlen)