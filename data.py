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
        #Calculate delta and double-delta
        deltas = torchaudio.transforms.ComputeDeltas()(waveform)
        
