import os
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import pandas as pd

from cvutils import Validator, Alphabet

def preproc_text(corpus_path, lang):
    train = pd.read_csv(os.path.join(corpus_path, 'train.tsv'), sep='\t')
    dev = pd.read_csv(os.path.join(corpus_path,'dev.tsv'), sep='\t')
    test = pd.read_csv(os.path.join(corpus_path,'test.tsv'), sep='\t')

    val = Validator(lang)

    train_sents = [val.validate(s) for s in train['sentence']]
    dev_sents = [val.validate(s) for s in dev['sentence']]
    test_sents = [val.validate(s) for s in test['sentence']]

    
    ##Write modified df
    train['sentence'] = train_sents
    train.to_csv(os.path.join(corpus_path, 'train.tsv'), sep='\t')
    
    dev['sentence'] = dev_sents
    dev.to_csv(os.path.join(corpus_path, 'dev.tsv'), sep='\t')
    
    test['sentence'] = test_sents
    test.to_csv(os.path.join(corpus_path, 'test.tsv'), sep='\t')
    
    #Make alphabet
    
    alph = Alphabet(lang)
    chars = alph.get_alphabet()

    with open(os.path.join(corpus_path, "alphabet.txt"), 'w') as fo:
        for char in chars:
            fo.write(char+'\n')


def extract_feats(data):
    '''
    Reads and processes one file at a time.
    Args:
        path: path to the file
        maxlen: maximum length of the spectrogram for padding
    '''
    maxlen_feats = 0
    unpadded = []

    for inst in data:
        waveform, sample_rate = torchaudio.load(inst["aud"])
        #Calculate MFCC
        mfcc = torchaudio.transforms.MFCC()(waveform)
        #Calculate delta and double-delta
        deltas = torchaudio.transforms.ComputeDeltas()(mfcc)
        ddeltas = torchaudio.transforms.ComputeDeltas()(deltas)
        res = torch.cat((mfcc, deltas, ddeltas), dim=1).squeeze(0)
        unpadded.append(res)

        if res.shape[1] > maxlen_feats:
            maxlen_feats = res.shape[1]

    padded = []
    masks = []
    for tens in unpadded:
        mask = torch.ones(1, tens.shape[1])
        tens = nn.functional.pad(tens, pad=(0, maxlen_feats-tens.shape[1], 0, 0), 
                                          mode="constant",value=0)

        mask = nn.functional.pad(mask, pad=(0, maxlen_feats-mask.shape[1], 0, 0), 
                                          mode="constant",value=0)
        padded.append(tens)
        masks.append(mask)

    return torch.stack(padded), torch.stack(masks)


def encode_trans(data):
    '''
    Encodes true transcription
    '''
    maxlen_t = 0
    unpadded = []
    masks = []
    encoded = []
    char2ind = data[0]["charmap"]

    for inst in data:
        res = torch.tensor([char2ind[char] for char in inst["trans"]])
        unpadded.append(res)
        if res.shape[0] > maxlen_t:
            maxlen_t = res.shape[0]

    for t in unpadded:
        res = nn.functional.pad(t, pad=(0, maxlen_t-t.shape[0]), mode="constant",value=0)
        encoded.append(res)
        mask = torch.tensor([1 if i>0 else 0 for i in res])
        masks.append(mask)
    
    return torch.stack(encoded), torch.stack(masks)


def collate_custom(data):
    '''
    For batch
    '''
   
    #{"aud": self.fnames[idx], "trans":self.transcrpts[idx], "charmap":self.char2ind)}

    feats, fmasks = extract_feats(data)
    transcrpts, tmasks = encode_trans(data)
    return {"feat": feats, "fmask":fmasks, "trans":transcrpts, "tmask":tmasks}

class Data(data.Dataset):
    def __init__(self, csv_path, aud_path, char2ind):
        self.df = pd.read_csv(csv_path, sep='\t')
        self.char2ind = char2ind
        self.fnames = [os.path.join(aud_path, f) for f in self.df['path']]
        self.transcrpts = self.df['sentence']


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"aud": self.fnames[idx], "trans":self.transcrpts[idx], "charmap":self.char2ind}
        return sample