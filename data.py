import os
import numpy as np
from tqdm import tqdm


import torch
import torchaudio

def make_feats(x_path, out_path):
    files = os.listdir()
    for f in files:
        waveform, sample_rate = torchaudio.load(os.path.join(x_path, f))
        #Calculate MFCC
        mfcc = torchaudio.transforms.MFCC()(waveform)
        print('mffc shape:', mfcc.shape)
        #Calculate delta and double-delta
        deltas = torchaudio.transforms.ComputeDeltas()(waveform)
        print('deltas:', deltas.shaoe)
        res = torch.column_stack((mfcc, deltas))
        torch.save(res, os.path.join(out_path, f.split('.')[0]+'.pt'))

make_feats('/N/slate/anakuzne/Data/asr/policy_gradient/eu/clips', 
           '/N/slate/anakuzne/Data/asr/policy_gradient/eu/eu_feats')