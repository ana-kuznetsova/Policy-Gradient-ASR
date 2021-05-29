import os
import pandas as pd
import copy
import torch

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch import utils
import torchaudio
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from data import preproc_text, Data, collate_custom


class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=1, bidirectional=True)
    def forward(self,x):
        return self.blstm(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128):
        super(Encoder, self).__init__()
        self.input_layer = nn.Linear(input_dim, 2*hidden_dim)
        self.relu = nn.LeakyReLU()
        self.pBLSTM1= pBLSTM(2*hidden_dim, hidden_dim)
        self.pBLSTM2= pBLSTM(2*hidden_dim, hidden_dim)
        self.pBLSTM3= pBLSTM(2*hidden_dim, hidden_dim)
        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)

    def forward(self, x, lens):
        x = torch.transpose(x, 1, -1)
        linear_input = self.input_layer(x)
        print("inp layer:", linear_input.shape)
        linear_input = self.relu(linear_input)
        print("lin input:", linear_input.shape)
        for i in range(3):
            if linear_input.shape[0]%2!=0:
                linear_input = linear_input[:-1,:,:]
            outputs = torch.transpose(linear_input, 0, 1)
            outputs = outputs.contiguous().view(outputs.shape[0], outputs.shape[1]//2, 2, outputs.shape[2])
            outputs = torch.mean(outputs, 2)
            outputs = torch.transpose(outputs,0,1)
            lens=lens//2
            rnn_inp = pack_padded_sequence(outputs, lengths=lens, enforce_sorted=False)
            if i==0:
                outputs, _ = self.pBLSTM1(rnn_inp)
            elif i==1:
                outputs, _ = self.pBLSTM2(rnn_inp)
            else:
                outputs, _ = self.pBLSTM3(rnn_inp)
            linear_input, _ = pad_packed_sequence(outputs)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value, lens

corpus_path = '/nobackup/anakuzne/data/cv/cv-corpus-6.1-2020-12-11/eu'
char2ind = preproc_text(corpus_path, 'eu')

dataset_train = Data(os.path.join(corpus_path, 'train.tsv'), os.path.join(corpus_path, 'clips'), char2ind)
loader_train = data.DataLoader(dataset_train, batch_size=5, 
                               shuffle=True, collate_fn=collate_custom)

device ="cuda:1"

encoder = Encoder(120, 256)
encoder.to(device)

for batch in loader_train:
    x = batch["feat"].to(device)
    xlens = batch['alens']
    out = encoder(x, xlens)