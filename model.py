import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchaudio
from torchsummary import summary
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data import extract_feats, encode_trans

class TrainData(data.Dataset):
    def __init__(self, csv_path, aud_path, char2ind, transforms):
        self.df = pd.read_csv(csv_path, sep='\t')
        self.aud_path = aud_path
        self.char2ind = char2ind
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        fname = os.path.join(self.aud_path, self.df['path'][idx])
        transcript = self.df['sentence'][idx].lower()

        feat, fmask = self.transforms[0](fname)
        trans, tmask = self.transforms[1](transcript, self.char2ind)
        sample = {'aud': nan_to_num(feat), 'trans': trans, 'fmask':fmask, 'tmask':tmask}
        return sample
    
def weights(m):
    '''
    Intialize weights randomly
    '''
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.1)

def nan_to_num(t,mynan=0.):
    if torch.all(torch.isfinite(t)):
        return t
    if len(t.size()) == 0:
        return torch.tensor(mynan)
    return torch.cat([nan_to_num(l).unsqueeze(0) for l in t],0)

class Encoder(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.input_layer = nn.Linear(120, 512)
        self.blstm = nn.LSTM(input_size=512, 
                             hidden_size=256, 
                             num_layers=3, 
                             bidirectional=True)
        self.batch_size = batch_size
        self.register_buffer("h0", torch.randn(3*2, batch_size, 256))
        self.register_buffer("c0", torch.randn(3*2, batch_size, 256))
        
    def forward(self, x, mask):
        outputs=[]
        for i in range(x.shape[2]):
            feature = x[:,:,i]
            out = self.input_layer(feature)
            out = torch.nn.LeakyReLU()(out)
            outputs.append(out)
        #print("X", torch.isnan(x))
        outputs = torch.stack(outputs)
        lengths = torch.sum(mask, dim=1).detach().cpu()
        outputs = pack_padded_sequence(outputs, lengths, enforce_sorted=False)
        #Pass through LSTM layers
        output, (hn, cn) = self.blstm(outputs, (self.h0, self.c0))
        output, _ = pad_packed_sequence(output, total_length=mask.shape[1])
        return output, (hn, cn)
    
class Attention(nn.Module):
    def __init__(self, batch_size, enc_hidden_size):
        super().__init__()
        self.register_buffer("c_t", torch.zeros(batch_size, 2*enc_hidden_size))
        
    def forward(self, h_e, h_d):
        score = torch.matmul(h_e.T, h_d)
        a_t = nn.functional.softmax(score, dim=0)
        c_t = torch.sum(a_t, dim=0)*h_e 
        return c_t
        
    
class Decoder(nn.Module):
    def __init__(self, batch_size, enc_hidden_size):
        super().__init__()
        self.embed_layer = nn.Linear(33, 128)
        self.lstm_cell = nn.LSTMCell(128, 512)
        self.output = nn.Linear(1024, 33)
        self.attention = Attention(batch_size, enc_hidden_size)
        self.dec_h = None 
        self.dec_c = None

    def forward(self, enc_h, y):
        preds = []
        for i, hidden in enumerate(enc_h):
            if i==0:
                self.dec_h, self.dec_c = self.lstm_cell(y)
            else:
                self.dec_h, self.dec_c = self.lstm_cell(y, (self.dec_h, self.dec_c))
            c_t = self.attention(hidden, self.dec_h)
            combined_input = torch.cat([self.dec_h, c_t], 1)
            y_hat = self.output(combined_input)
            
            output = nn.functional.log_softmax(y_hat, dim=1)
            y = self.embed_layer(y_hat)
            preds.append(output)
        preds = torch.stack(preds)
        return preds
    
class Seq2Seq(nn.Module):
    def __init__(self, batch_size, enc_hidden_size):
        super().__init__()
        self.encoder = Encoder(batch_size)
        self.decoder = Decoder(batch_size, enc_hidden_size)

    def forward(self, x, mask, dec_input):
        enc_out, (he, ce) = self.encoder(x, mask)
        preds = self.decoder(enc_out, dec_input)
        return preds
    
        
def train(csv_path, aud_path, alphabet_path, num_epochs=10,  batch_size=32, enc_hidden_size=256):

    with open(alphabet_path, 'r') as fo:
        alphabet = fo.readlines() + ['f', 'i', 'r', 'e', 'o', 'x']
    char2ind = {alphabet[i].strip():i for i in range(len(alphabet))}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(batch_size, enc_hidden_size)
    model.apply(weights)
    model = model.to(device)

    criterion = nn.CTCLoss(zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    cv_dataset = TrainData(csv_path, aud_path, char2ind, [extract_feats, encode_trans])

    for epoch in range(1, num_epochs+1):
        epoch_loss = 0
        loader = data.DataLoader(cv_dataset, batch_size=32, shuffle=True)

        for batch in loader:
            x = batch['aud'].to(device)
            t = batch['trans'].to(device)
            fmask = batch['fmask'].squeeze(1).to(device)
            tmask = batch['tmask'].squeeze(1).to(device)
            dec_input = torch.randn(batch_size, 128, requires_grad=True).to(device)

            preds = model(x, fmask, dec_input)
            input_length = torch.sum(fmask, dim =1).long().to(device)
            target_length = torch.sum(tmask, dim=1).long().to(device)
            optimizer.zero_grad()
            loss = criterion(preds, t, input_length, target_length)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.detach().cpu().numpy()
        print('Epoch:{:3}/{:3} Training loss:{:>4f}'.format(epoch, num_epochs, epoch_loss/len(loader)))