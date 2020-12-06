import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchaudio
from torchsummary import summary
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data import extract_feats, encode_trans
from CTCdecoder import CTCDecoder, collapse_fn
from metrics import evaluate, save_predictions

class Data(data.Dataset):
    def __init__(self, csv_path, aud_path, char2ind, transforms, maxlen, maxlent):
        self.df = pd.read_csv(csv_path, sep='\t')
        self.aud_path = aud_path
        self.char2ind = char2ind
        self.transforms = transforms
        self.maxlen = maxlen
        self.maxlent = maxlent

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = os.path.join(self.aud_path, self.df['path'][idx])
        transcript = self.df['sentence'][idx].lower()

        feat, fmask = self.transforms[0](fname, self.maxlen)
        trans, tmask = self.transforms[1](transcript, self.char2ind, self.maxlent)
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
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(120, 512)
        self.blstm = nn.LSTM(input_size=512, 
                             hidden_size=256, 
                             num_layers=3, 
                             bidirectional=True)
        
    def forward(self, x, mask):
        outputs=[]
        for i in range(x.shape[2]):
            feature = x[:,:,i]
            out = self.input_layer(feature)
            out = torch.nn.LeakyReLU()(out)
            outputs.append(out)
        outputs = torch.stack(outputs)
        lengths = torch.sum(mask, dim=1).detach().cpu()
        outputs = pack_padded_sequence(outputs, lengths, enforce_sorted=False)
        output, (hn, cn) = self.blstm(outputs)
        output, _ = pad_packed_sequence(output, total_length=mask.shape[1])
        return output, (hn, cn)
    
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, h_e, h_d):
        score = torch.matmul(h_e.T, h_d)
        a_t = nn.functional.softmax(score, dim=0)
        c_t = torch.sum(a_t, dim=0)*h_e 
        return c_t
        
    
class Decoder(nn.Module):
    def __init__(self, alphabet_size):
        super().__init__()
        self.embed_layer = nn.Linear(alphabet_size, 128)
        self.lstm_cell = nn.LSTMCell(128, 512)
        self.output = nn.Linear(1024, alphabet_size)
        self.attention = Attention()
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
    def __init__(self, alphabet_size):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(alphabet_size)

    def forward(self, x, mask, dec_input):
        enc_out, (he, ce) = self.encoder(x, mask)
        preds = self.decoder(enc_out, dec_input)
        return preds
    
        
def train(train_path, dev_path, aud_path, alphabet_path, model_path, maxlen, maxlent,
          num_epochs=10,  batch_size=32):

    print("Num epochs:", num_epochs, "Batch size:", batch_size)

    with open(alphabet_path, 'r') as fo:
        alphabet = fo.readlines() 
    char2ind = {alphabet[i].replace('\n', ''):i for i in range(len(alphabet))}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(alphabet_size=len(alphabet))
    model.apply(weights)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs...")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CTCLoss(zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    best_model = copy.deepcopy(model.state_dict())

    init_val_loss = 9999999

    losses = []
    val_losses = []

    train_dataset = Data(train_path, aud_path, char2ind, [extract_feats, encode_trans], maxlen, maxlent)
    print("Start training...")
    for epoch in range(1, num_epochs+1):
        epoch_loss = 0
        loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        num_steps = len(loader)
        step = 0
        for batch in loader:
            step+=1
            print("Step {}/{}".format(step, num_steps))
            x = batch['aud'].to(device)
            t = batch['trans'].to(device)
            fmask = batch['fmask'].squeeze(1).to(device)
            tmask = batch['tmask'].squeeze(1).to(device)
            dec_input = torch.randn(x.shape[0], 128, requires_grad=True).to(device)

            preds = model(x, fmask, dec_input)
            input_length = torch.sum(fmask, dim =1).long().to(device)
            target_length = torch.sum(tmask, dim=1).long().to(device)
            optimizer.zero_grad()
            loss = criterion(preds, t, input_length, target_length)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.detach().cpu().numpy()

        torch.cuda.empty_cache()
        
        losses.append(epoch_loss/len(loader))
        np.save(os.path.join(model_path, 'train_loss.npy'), np.array(losses))
        print('Epoch:{}/{} Training loss:{:>4f}'.format(epoch, num_epochs, epoch_loss/len(loader)))

        #Validation
        dev_dataset = Data(dev_path, aud_path, char2ind, [extract_feats, encode_trans], maxlen, maxlent)
        val_loss = 0
        loader = data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

        for batch in loader:
            x = batch['aud'].to(device)
            t = batch['trans'].to(device)
            fmask = batch['fmask'].squeeze(1).to(device)
            tmask = batch['tmask'].squeeze(1).to(device)
            dec_input = torch.randn(x.shape[0], 128, requires_grad=True).to(device)

            preds = model(x, fmask, dec_input)
            input_length = torch.sum(fmask, dim =1).long().to(device)
            target_length = torch.sum(tmask, dim=1).long().to(device)
            loss = criterion(preds, t, input_length, target_length)

            val_loss+=loss.detach().cpu().numpy()

        curr_val_loss = val_loss/len(loader)
        val_losses.append(curr_val_loss)
        np.save(os.path.join(model_path, "val_losses.npy"), np.array(val_losses))
        torch.cuda.empty_cache() 

        print('Epoch:{}/{} Validation loss:{:>4f}'.format(epoch, num_epochs, curr_val_loss))

        ## Model Selection
        if curr_val_loss < init_val_loss:
            torch.save(best_model, os.path.join(model_path, "model_best.pth"))
            init_val_loss = curr_val_loss
        torch.save(best_model, os.path.join(model_path, "model_last.pth"))


def predict(test_path, aud_path, alphabet_path, model_path, batch_size, maxlen, maxlent):
    with open(alphabet_path, 'r') as fo:
        alphabet = fo.readlines()
    alphabet = [char.strip() for char in alphabet] 

    char2ind = {alphabet[i].strip():i for i in range(len(alphabet))}
    ind2char = {char2ind[key]:key for key in char2ind}

    ctc_decoder = CTCDecoder(alphabet)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(alphabet_size=len(alphabet))
    model.load_state_dict(torch.load(os.path.join(model_path, "model_best.pth")))
    model = model.to(device)

    test_dataset = Data(test_path, aud_path, char2ind, [extract_feats, encode_trans], maxlen, maxlent)
    loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    total_WER = 0
    total_CER = 0
    step = 0

    targets = []
    predicted = []

    for batch in loader:
        step+=1
        print("Decoding step ", step)
        batch_WER = 0
        batch_CER = 0

        x = batch['aud'].to(device)
        t = batch['trans'].numpy()
        tmask = batch['tmask'].squeeze(1).numpy()
        fmask = batch['fmask'].squeeze(1).to(device)
        dec_input = torch.randn(x.shape[0], 128, requires_grad=True).to(device)
        preds = model(x, fmask, dec_input)
        preds = torch.transpose(preds, 0, 1).detach().cpu().numpy()
        fmask = fmask.detach().cpu().numpy()
        
        for i, probs in enumerate(preds):
            pad_ind = int(np.sum(fmask[i]))
            probs = np.exp(probs[:pad_ind,])
            seq , score = ctc_decoder.decode(probs, beam_size=5)
            seq = ''.join([ind2char[ind] for ind in seq])
            seq = collapse_fn(seq)
            pad_ind = int(np.sum(tmask[i]))
            target = t[i][:pad_ind]
            target = ''.join([ind2char[ind] for ind in target])
            targets.append(target)
            predicted.append(seq)
            cer, wer = evaluate(target, seq)
            batch_CER+=cer
            batch_WER+=wer

        total_WER+=batch_WER/batch_size
        total_CER+=batch_CER/batch_size
    save_predictions(targets, predicted, model_path)
    print("CER: {:>4f} WER: {:>4f}".format(total_CER/len(loader), total_WER/len(loader)))