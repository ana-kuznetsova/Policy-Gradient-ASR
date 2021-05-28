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
    
    #Make char2ind
    
    alph = Alphabet(lang)
    chars = alph.get_alphabet()

    char2ind = {char:i for i, char in enumerate(chars)}
    #char2ind["<sos/>"] = len(char2ind) + 1
    #char2ind["<eos/>"] = len(char2ind) + 2
    return char2ind


def extract_feats(batch):
    lens = []
    unpadded = []
    maxlen_feats = 0

    for inst in batch:
        waveform, sample_rate = torchaudio.load(inst["aud"])
        #Calculate MFCC
        mfcc = torchaudio.transforms.MFCC()(waveform)
        #Calculate delta and double-delta
        deltas = torchaudio.transforms.ComputeDeltas()(mfcc)
        ddeltas = torchaudio.transforms.ComputeDeltas()(deltas)
        res = torch.cat((mfcc, deltas, ddeltas), dim=1).squeeze(0)
        unpadded.append(res)
        lens.append(res.shape[1])
        maxlen_feats = max(maxlen_feats, res.shape[1])

    padded = []

    for i, x in enumerate(unpadded):

        x = nn.functional.pad(x, pad=(0, maxlen_feats-lens[i], 0, 0), 
                                          mode="constant",value=0)

        padded.append(x)
    return torch.stack(padded), torch.tensor(lens)


def encode_trans(batch):
    maxlen_t = 0
    unpadded = []
    padded = []
    lens = []

    for t in batch:
        char2ind = t['char2ind']
        res = torch.tensor([char2ind[char] for char in t["trans"]])
        unpadded.append(res)
        maxlen_t = max(maxlen_t, res.shape[0])
        lens.append(res.shape[0])

    for t in unpadded:
        res = nn.functional.pad(t, pad=(0, maxlen_t-t.shape[0]), mode="constant",value=0)
        padded.append(res)
    return torch.stack(padded), torch.tensor(lens)


def collate_custom(batch):
    feats, alens = extract_feats(batch)
    transcrpts, tlens = encode_trans(batch)
    return {"feat": feats, "alens":alens, "trans":transcrpts, "tlens":tlens}

class Data(data.Dataset):
    def __init__(self, csv_path, aud_path, char2ind):
        self.df = pd.read_csv(csv_path, sep='\t')
        self.fnames = [os.path.join(aud_path, f) for f in self.df['path']]
        self.transcrpts = self.df['sentence']
        self.char2ind = char2ind


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"aud": self.fnames[idx], "trans":self.transcrpts[idx], "char2ind":self.char2ind}
        return sample

corpus_path = '/nobackup/anakuzne/data/cv/cv-corpus-5.1-2020-06-22/eu'
char2ind = preproc_text(corpus_path, 'eu')

dataset_train = Data(os.path.join(corpus_path, 'train.tsv'), os.path.join(corpus_path, 'clips'), char2ind)
loader_train = data.DataLoader(dataset_train, batch_size=5, 
                               shuffle=True, collate_fn=collate_custom)

print(len(loader_train))
for batch in loader_train:
    print(batch.keys())
    print(batch['aud'].shape)