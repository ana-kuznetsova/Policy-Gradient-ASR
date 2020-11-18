import os
import numpy as np
import torch
import torchaudio

def make_feats(x_path):
    files = os.listdir()
    for f in files:
        waveform, sample_rate = torchaudio.load(f)
        #Calculate MFCC
        mfcc = torchaudio.transforms.MFCC()(waveform)
        #Calculate delta and double-delta
        deltas = torchaudio.transforms.ComputeDeltas()(waveform)
