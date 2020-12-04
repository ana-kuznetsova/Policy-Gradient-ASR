import os
import numpy as np
import torch
import torchaudio
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from string import punctuation


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

def preproc(corpus_path):
    train = pd.read_csv(os.path.join(corpus_path, 'train.tsv'), sep='\t')
    dev = pd.read_csv(os.path.join(corpus_path,'dev.tsv'), sep='\t')
    test = pd.read_csv(os.path.join(corpus_path,'test.tsv'), sep='\t')
    
    ## Remove punct
    train_sents = [''.join([char for char in sent.lower() if char not in punctuation])\
                   for sent in train['sentence']]
    dev_sents =  [''.join([char for char in sent.lower() if char not in punctuation])\
                   for sent in dev['sentence']]
    
    test_sents = [''.join([char for char in sent.lower() if char not in punctuation])\
                   for sent in test['sentence']]
    
    ##Write modified df
    train['sentence'] = train_sents
    train.to_csv(os.path.join(corpus_path, 'train.tsv'), sep='\t')
    
    dev['sentence'] = dev_sents
    dev.to_csv(os.path.join(corpus_path, 'dev.tsv'), sep='\t')
    
    test['sentence'] = test_sents
    test.to_csv(os.path.join(corpus_path, 'test.tsv'), sep='\t')
    
    #Make alphabet
    chars = []
    sents = train_sents + dev_sents + test_sents
    for sent in sents:
        for char in sent:
            if char not in chars:
                chars.append(char)
    with open(os.path.join(corpus_path, "alphabet.txt"), 'w') as fo:
        for i in range(len(chars)+1):
            if i==0:
                fo.write(' '+'\n')
            else:
                fo.write(chars[i]+'\n')

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