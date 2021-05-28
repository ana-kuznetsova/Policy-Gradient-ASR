import os
import pandas as pd
import copy
import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchaudio
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from data import preproc_text, Data, collate_custom


class Encoder(nn.Module):
    def __init__(self, feat_dim, inp_size, hid_size):
        super().__init__()
        self.input_layer = nn.Linear(feat_dim, inp_size)
        self.layer_norm = nn.LayerNorm(feat_dim, inp_size)
        self.blstm1 = nn.LSTM(input_size=inp_size, 
                             hidden_size=hid_size, 
                             num_layers=1,
                             bidirectional=True,
                             batch_first=True)
        ## Out from blstm1 is 512
        self.blstm2 = nn.LSTM(input_size=inp_size, 
                             hidden_size=hid_size//4, 
                             num_layers=1,
                             bidirectional=True,
                             batch_first=True)
        ## Out from blstm2 is 512//4 = 128
        self.blstm3 = nn.LSTM(input_size=inp_size//4, 
                             hidden_size=hid_size//8, 
                             num_layers=1,
                             bidirectional=True,
                             batch_first=True)
        
    def forward(self, x, lens):
        x = torch.transpose(x, 1, -1)
        x = self.layer_norm(x)
        print("Lnorm:", x.shape)
        x = F.leaky_relu(self.input_layer(x))
        print("Inp L", x.shape)
        x = pack_padded_sequence(x, lens, enforce_sorted=False, batch_first=True)
        x, _ = self.blstm1(x)
        x, _ = self.blstm2(x)
        x, _ = self.blstm3(x)
        output, _ = pad_packed_sequence(x, batch_first=True)
        print("Final blstm:", output.shape)
        return output


corpus_path = '/nobackup/anakuzne/data/cv/cv-corpus-6.1-2020-12-11/eu'
char2ind = preproc_text(corpus_path, 'eu')

dataset_train = Data(os.path.join(corpus_path, 'train.tsv'), os.path.join(corpus_path, 'clips'), char2ind)
loader_train = data.DataLoader(dataset_train, batch_size=5, 
                               shuffle=True, collate_fn=collate_custom)

device ="cuda:1"

encoder = Encoder(120, 512, 256)
encoder.to(device)

for batch in loader_train:
    x = batch["feat"].to(device)
    xlens = batch['alens']
    print("In shape:", x.shape)
    out = encoder(x, xlens)
    print("Out_shape:", out.shape)