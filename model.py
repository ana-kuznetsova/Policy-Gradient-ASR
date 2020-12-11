import os
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
import torch.nn.functional as F

from loss import customNLLLoss
from data import extract_feats, encode_trans, pad
from CTCdecoder import CTCDecoder, collapse_fn, collapse_fn_int
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
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(120, 2 * hidden_size)
        self.blstm = nn.LSTM(input_size= 2 * hidden_size, 
                             hidden_size=hidden_size, 
                             num_layers=3,
                             dropout=0.3, 
                             bidirectional=True)
        self.drop = nn.Dropout()
        self.output_layer = nn.Linear(2*hidden_size, output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)
        
    def forward(self, x, mask):
        outputs=[]
        for i in range(x.shape[2]):
            feature = x[:,:,i]
            out = self.input_layer(feature)
            out = torch.nn.LeakyReLU()(out)
            out = self.drop(out)
            outputs.append(out)
        outputs = torch.stack(outputs)
        lengths = torch.sum(mask, dim=1).detach().cpu()
        outputs = pack_padded_sequence(outputs, lengths, enforce_sorted=False)
        output, (hn, cn) = self.blstm(outputs)
        output, _ = pad_packed_sequence(output, total_length=mask.shape[1])
        output = self.output_layer(output)
        output = self.log_softmax(output)
        return output


def train(train_path, dev_path, aud_path, alphabet_path, model_path, maxlen, maxlent,
          num_epochs=10,  batch_size=32, resume='False', device_id=0):

    print("Num epochs:", num_epochs, "Batch size:", batch_size)

    with open(alphabet_path, 'r') as fo:
        alphabet = ['<pad>'] + fo.readlines()
    alphabet = [char.replace('\n', '') for char in alphabet]

    char2ind = {alphabet[i].replace('\n', ''):i for i in range(len(alphabet))}
    ind2char = {char2ind[key]:key for key in char2ind}

    device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")
    if resume=='True':
        print("Loaded weights from pretrained...")
        model = Encoder(256, len(alphabet))
        model.load_state_dict(torch.load(os.path.join(model_path, "model_best.pth")))
        
    else:
        model = Encoder(256, len(alphabet))
    model = model.to(device)

    criterion = torch.nn.CTCLoss(blank=2, zero_infinity=True)
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
            x = batch['aud'].to(device)
            t = batch['trans'].to(device)
            fmask = batch['fmask'].squeeze(1).to(device)
            tmask = batch['tmask'].squeeze(1).to(device)
    
            model_out = model(x, fmask)
            input_lengths = torch.sum(fmask, dim=1).long()
            target_lengths = torch.sum(tmask, dim=1).long()
    
            optimizer.zero_grad()
    
            loss = criterion(model_out, t, input_lengths, target_lengths)
            print("Step {}/{}. Loss: {:>4f}".format(step, num_steps, loss.detach().cpu().numpy()))

            loss.backward()
            optimizer.step()
            epoch_loss+=loss.detach().cpu().numpy()

        losses.append(epoch_loss/len(loader))
        np.save(os.path.join(model_path, 'train_loss.npy'), np.array(losses))
        print('Epoch:{}/{} Training loss:{:>4f}'.format(epoch, num_epochs, epoch_loss/len(loader)))

        torch.cuda.empty_cache()
        #Validation
        dev_dataset = Data(dev_path, aud_path, char2ind, [extract_feats, encode_trans], maxlen, maxlent)
        val_loss = 0
        loader = data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

        for batch in loader:
            x = batch['aud'].to(device)
            t = batch['trans'].to(device)
            fmask = batch['fmask'].squeeze(1).to(device)
            tmask = batch['tmask'].squeeze(1).to(device)
    
            model_out = model(x, fmask)
            input_lengths = torch.sum(fmask, dim=1).long()
            target_lengths = torch.sum(tmask, dim=1).long()

            val_loss+=loss.detach().cpu().numpy()

        curr_val_loss = val_loss/len(loader)
        val_losses.append(curr_val_loss)
        np.save(os.path.join(model_path, "val_loss.npy"), np.array(val_losses))
        torch.cuda.empty_cache() 

        print('Epoch:{}/{} Validation loss:{:>4f}'.format(epoch, num_epochs, curr_val_loss))

        ## Model Selection
        if curr_val_loss < init_val_loss:
            torch.save(best_model, os.path.join(model_path, "model_best.pth"))
            init_val_loss = curr_val_loss
        torch.save(best_model, os.path.join(model_path, "model_last.pth"))


def predict(test_path, aud_path, alphabet_path, model_path, batch_size, maxlen, maxlent, device_id=0):

    with open(alphabet_path, 'r') as fo:
        alphabet = ['<pad>'] + fo.readlines()
    alphabet = [char.replace('\n', '') for char in alphabet]
    print(alphabet)

    char2ind = {alphabet[i].strip():i for i in range(len(alphabet))}
    ind2char = {char2ind[key]:key for key in char2ind}

    ctc_decoder = CTCDecoder(alphabet)
    
    device = torch.device("cuda:"+str(device_id) if torch.cuda.is_available() else "cpu")
    model = Encoder(256, len(alphabet))
    model.load_state_dict(torch.load(os.path.join(model_path, "test_model_best.pth")))
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
        print("Decoding step: ", step)
        batch_WER = 0
        batch_CER = 0

        x = batch['aud'].to(device)
        t = batch['trans'].numpy()
        tmask = batch['tmask'].squeeze(1).numpy()
        fmask = batch['fmask'].squeeze(1).to(device)
        preds = model(x, fmask)
        preds = torch.transpose(preds, 0, 1).detach().cpu().numpy()
        fmask = fmask.detach().cpu().numpy()
        
        for i, probs in enumerate(preds):
            pad_ind = int(np.sum(fmask[i]))
            probs = np.exp(probs[:pad_ind,])
            seq , _ = ctc_decoder.decode(probs, beam_size=5)
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