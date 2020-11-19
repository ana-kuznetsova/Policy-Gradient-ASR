import os
import numpy as np
from tqdm import tqdm


import torch
import torchaudio

def make_feats(x_path, out_path):
    files = os.listdir(x_path)
    for f in files:
        waveform, sample_rate = torchaudio.load(os.path.join(x_path, f))
        #Calculate MFCC
        mfcc = torchaudio.transforms.MFCC()(waveform)
        print('mffc shape:', mfcc.shape)
        #Calculate delta and double-delta
        deltas = torchaudio.transforms.ComputeDeltas()(mfcc)
        ddeltas = torchaudio.transforms.ComputeDeltas()(deltas)
        res = torch.cat((mfcc, deltas, ddeltas), dim=1)
        print("res:", res.shape)
        torch.save(res, os.path.join(out_path, f.split('.')[0]+'.pt'))