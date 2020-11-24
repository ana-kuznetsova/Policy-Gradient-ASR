import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

def make_feats(path, maxlen=4799):
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
    mask = torch.ones(norm.shape[0], norm.shape[1])
    padding = torch.nn.ZeroPad2d((0, maxlen-norm.shape[1], 0, 0))
    norm = padding(norm)
    mask = padding(mask)
    return norm, mask


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