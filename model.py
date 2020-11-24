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

        feat, mask = self.transform(fname)

        sample = {'aud':feat, 'trans': transcript, 'mask':mask}
        return sample

def weights(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.1)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(120, 512)
        self.blstm = nn.LSTM(512, hidden_size=256, num_layers=3, bidirectional=True)

    def init_lstm(self, x):
        pass

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.nn.LeakyReLU(x)
        x = self.blstm(x)
        print(x)
        return(x)
        

def train(num_epochs=50):

    device = torch.device("cuda")

    encoder = Encoder()
    encoder.apply(weights)
    encoder.cuda()
    encoder = encoder.to(device)

    cv_dataset = TrainData('/nfs/nfs5/home/nobackup/anakuzne/data/cv/cv-corpus-5.1-2020-06-22/eu/train.tsv',
                            '/nfs/nfs5/home/nobackup/anakuzne/data/cv/cv-corpus-5.1-2020-06-22/eu/clips', make_feats)

    loader = data.DataLoader(cv_dataset, batch_size=32, shuffle=True)
    print("Loader", len(loader))
        