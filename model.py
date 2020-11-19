import os
import pandas as pd 
from data import make_feats
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn

class TrainData(data.Dataset):
    def __init__(self, csv_path, aud_path, transform):
        self.df = pd.read_csv(csv_path, sep='\t')
        self.aud_path = aud_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        fname = os.path.join(self.aud_path, self.df['path'][idx])
        transcript = self.df['sentence'][idx]

        feat = self.transform(fname)

        sample = {'aud':feat, 'trans': transcript}
        return sample

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(120, 512)

    def forward(self, x):
        x = torch.nn.LeakyReLU(self.input_layer(x))
        