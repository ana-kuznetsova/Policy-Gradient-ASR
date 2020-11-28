import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm


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


def extract_feats(path, maxlen=1083):
    '''
    Reads and processes one file at a time.
    Args:
        path: path to the file
        maxlen: maximum length of the spectrogram for padding
    '''
    waveform, sample_rate = torchaudio.load(path)
    #Calculate MFCC
    mfcc = torchaudio.transforms.MFCC()(waveform)
    #Calculate delta and double-delta
    deltas = torchaudio.transforms.ComputeDeltas()(mfcc)
    ddeltas = torchaudio.transforms.ComputeDeltas()(deltas)
    res = torch.cat((mfcc, deltas, ddeltas), dim=1).squeeze(0)
    #Normalize rows
    s = torch.sum(res, dim=1, keepdim=True)
    norm = torch.div(res, s)
    mask = torch.ones(1, norm.shape[1])
    padded_norm = nn.functional.pad(norm, pad=(0, maxlen-norm.shape[1], 0, 0), 
                                          mode="constant",value=0)
    padded_mask = nn.functional.pad(mask, pad=(0, maxlen-mask.shape[1], 0, 0), 
                                          mode="constant",value=0)
    return padded_norm, padded_mask


def encode_trans(trans, char2ind, maxlen_t=7):
    '''
    Encodes true transcription
    trans: 
    '''
    res = np.array([char2ind[char] for char in trans])
    res = np.pad(res, (0, maxlen_t-len(res)), 'constant', constant_values=(-1))
    mask = [1 if i>=0 else 0 for i in res]
    return torch.tensor(res), torch.tensor(mask) 