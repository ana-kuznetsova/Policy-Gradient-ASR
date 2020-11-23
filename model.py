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


##Custom collate function to feed variable size batches
def my_collate(batch):
    data = [torch.reshape(item["aud"], (item["aud"].shape[1], item["aud"].shape[2])) for item in batch]
    data = [torch.transpose(item, 0, 1) for item in data]
    target = [item["trans"] for item in batch]
    data = torch.cat(data, dim=0)
    return [data, target]


def weights(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.1)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(120, 512)
        self.blstm = nn.LSTM(512, hidden_size=256, num_layers=3, bidirectional=True)

    def forward(self, x):
        x = torch.nn.LeakyReLU(self.input_layer(x))
        x = self.blstm(x)
        return(x)
        

def train(num_epochs=50):

    device = torch.device("cuda")

    encoder = Encoder()
    encoder.apply(weights)
    encoder.cuda()
    encoder = encoder.to(device)

    cv_dataset = TrainData('/N/slate/anakuzne/Data/asr/policy_gradient/eu/train.tsv',
                            '/N/slate/anakuzne/Data/asr/policy_gradient/eu/clips', make_feats)

    for ep in range(1, num_epochs+1):
        loader = data.DataLoader(cv_dataset, batch_size=32, shuffle=True, collate_fn=my_collate)
        loader = iter(loader)
        for batch in range(len(loader)):
            x, t = loader.next()
            x = x.to(device)
            out = encoder(x)
            print(out)
        