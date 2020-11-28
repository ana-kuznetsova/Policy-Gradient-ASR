import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchaudio
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
        sample = {'aud':feat, 'trans': trans, 'fmask':fmask, 'tmask':tmask}
        return sample
    
def weights(m):
    '''
    Intialize weights randomly
    '''
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.1)


class Encoder(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.input_layer = nn.Linear(120, 512)
        self.blstm = nn.LSTM(input_size=512, 
                             hidden_size=256, 
                             num_layers=3, 
                             bidirectional=True)
        self.h0 = torch.zeros(3*2, batch_size, 256)
        self.c0 = torch.zeros(3*2, batch_size, 256)
        
    def forward(self, x):
        #Pass through the first linear layer
        outputs=[]
        for i in range(x.shape[2]):
            feature = x[:,:,i]
            out = self.input_layer(feature)
            out = torch.nn.LeakyReLU()(out)
            outputs.append(out)
        outputs = torch.stack(outputs)
        #Pass through LSTM layers
        output, (hn, cn) = self.blstm(outputs, (self.h0, self.c0))
        return output, (hn, cn)
    
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, h_e, h_d):
        score = torch.matmul(h_e.T, h_d)
        temp1 = torch.exp(score)
        temp2 = torch.sum(score, dim=0)
        a_t = temp1/temp2
        c_t = torch.zeros(h_e.shape)
        for a in a_t:
            c_t+=a*h_e  
        return c_t
        
    
class Decoder(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.embed_layer = nn.Linear(33, 128)
        self.lstm_cell = nn.LSTMCell(128, 512)
        self.output = nn.Linear(1024, 33)
        self.dec_h = torch.zeros(batch_size, 512)
        self.c = torch.zeros(batch_size, 512)
        self.y = torch.zeros(batch_size,  33)
        self.attention = Attention()
    
    def forward(self, enc_h):
        preds = []
        for hidden in enc_h:
            y = self.embed_layer(self.y)
            self.dec_h, self.c = self.lstm_cell(y, (self.dec_h, self.c))
            c_t = self.attention(hidden, self.dec_h)
            combined_output = torch.cat([self.dec_h, c_t],1)
            self.y = self.output(combined_output)
            y_hat = nn.functional.log_softmax(self.y, dim=1)
            preds.append(y_hat)
        preds = torch.stack(preds)
        return preds
    
class Seq2Seq(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.encoder = Encoder(batch_size)
        self.decoder = Decoder(batch_size)
    def forward(self, batch):
        enc_out, (he, ce) = self.encoder(batch)
        preds = self.decoder(enc_out)
        return preds
    
    
def collapse_fn(preds, masks):
    preds = preds.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    collapsed = []
    maxlen_t = 0
    for pred, mask in zip(preds, masks):
        temp = [pred[0]]
        for i, char in enumerate(pred[1:]):
            if mask[i]:
                if pred[i-1]==char:
                    continue
                else:
                    temp.append(char)
        collapsed.append(temp)
        maxlen_t = max(maxlen_t, len(temp))
    
    res = []
    for sent in collapsed:
        sent = np.pad(sent, (0, maxlen_t - len(sent)), 'constant', constant_values=(-1))
        res.append(sent)
        
    return torch.tensor(res)
        
def train(csv_path, aud_path, alphabet_path):

    with open(alphabet_path, 'r') as fo:
        alphabet = fo.readlines() + ['f', 'i', 'r', 'e', 'o', 'x']
    char2ind = {alphabet[i].strip():i for i in range(len(alphabet))}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(32)
    model.apply(weights)
    model = model.to(device)

    criterion = nn.CTCLoss(zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    cv_dataset = TrainData(csv_path, aud_path, char2ind, [extract_feats, encode_trans])
    loader = data.DataLoader(cv_dataset, batch_size=32, shuffle=True)

    for batch in loader:
        x = batch['aud'].to(device)
        t = batch['trans'].to(device)
        fmask = batch['fmask'].squeeze(1).to(device)
        tmask = batch['tmask'].squeeze(1).to(device)
        preds = model(x)
        input_length = torch.sum(fmask, dim =1).long().to(device)
        target_length = torch.sum(tmask, dim=1).long().to(device)
        optimizer.zero_grad()
        loss = criterion(preds, t, input_length, target_length)
        print(loss.detach().cpu().numpy())
        loss.backward(retain_graph=True)
        optimizer.step()
        print("----------------------------------------------------")