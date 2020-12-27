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
from data import extract_feats, encode_trans
from metrics import evaluate, save_predictions

### POLICY GRAD IMPORTS ###
from policy_grad import sample_trans

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
                             num_layers=2,
                             dropout=0.3, 
                             bidirectional=True)
        self.drop = nn.Dropout()
        
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
        return output
    
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, enc_hid_states, dec_hid, device):
        enc_hid_states = torch.transpose(enc_hid_states, 0, 1)
        scores = torch.zeros(dec_hid.shape[0], enc_hid_states.shape[0]).to(device)
        for i, enc_hid in enumerate(enc_hid_states):
            score_i = torch.bmm(dec_hid.unsqueeze(1), enc_hid.unsqueeze(2))[:,0,0]
            scores[:, i] = score_i
        
        align = F.softmax(scores, dim=1)
        c_t = torch.zeros(dec_hid.shape).to(device)
        for i, enc_hid in enumerate(enc_hid_states):
            c_t+= align[:, i].unsqueeze(1)*enc_hid
        return c_t
        
    
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.embed_layer = nn.Embedding(output_size, 128)
        #self.lstm_cell = nn.LSTMCell(128, hidden_size)
        self.lstm = nn.LSTM(input_size=128, 
                            hidden_size=hidden_size, 
                            num_layers=1,
                            dropout=0.3)
        self.output = nn.Linear(2* hidden_size, output_size)
        self.attention = Attention()
        self.drop_lstm = nn.Dropout(p=0.3)

    def forward(self, target_inputs, encoder_outputs, device=None):
        '''
        y is a target sentence
        '''
      
        dec_hid = encoder_outputs[-1].unsqueeze(0)

        encoder_outputs = torch.transpose(encoder_outputs, 0, 1)
        c_i = torch.zeros(dec_hid.shape).to(device)
        dec_outputs = []

        for inp in torch.transpose(target_inputs, 0, 1):
            embedded = self.embed_layer(inp)
            dec_out, (dec_hid, _) = self.lstm(embedded.unsqueeze(0), (dec_hid, c_i))
            context = self.attention(encoder_outputs, dec_out.squeeze(0), device)
            combined_input = torch.cat([dec_hid.squeeze(0), context], 1)
            output_i = self.output(combined_input)
            output_i = F.log_softmax(output_i, dim=1)
            dec_outputs.append(output_i)

        dec_outputs = torch.stack(dec_outputs)
        return dec_outputs


class Seq2Seq(nn.Module):
    def __init__(self, alphabet_size, batch_size, maxlen):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(alphabet_size, 512)

    def forward(self, x, t, fmask, device,):
        enc_out = self.encoder(x, fmask)
        dec_out = self.decoder(t, enc_out, device=device)
        return dec_out


def train(train_path, dev_path, aud_path, alphabet_path, model_path, maxlen, maxlent,
          num_epochs=10,  batch_size=32, device_id=0):

    print("Num epochs:", num_epochs, "Batch size:", batch_size)

    with open(alphabet_path, 'r') as fo:
        alphabet = ['<pad>'] + fo.readlines()

    char2ind = {alphabet[i].replace('\n', ''):i for i in range(len(alphabet))}

    device = torch.device("cuda:"+str(device_id) if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(alphabet_size=len(alphabet), batch_size=batch_size, maxlen=maxlen)
    model.apply(weights)

    model = model.to(device)

    criterion = customNLLLoss(ignore_index=0)
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
            tmask = batch['tmask'].squeeze(1).numpy()
            
            model_out = model(x, t, fmask, device)
            model_out = torch.transpose(model_out, 0, 1).detach().cpu().numpy()
            for i in range(model_out.shape[0]):
                sampled_t = sample_trans(model_out[i], tmask[i], alphabet)
            '''
            optimizer.zero_grad()
    
            loss = criterion(model_out, t)
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
            
            model_out = model(x, t, fmask, device)
    
            loss = criterion(model_out, t)

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
'''
'''
def predict(test_path, aud_path, alphabet_path, model_path, batch_size, maxlen, maxlent, device_id=0):
    with open(alphabet_path, 'r') as fo:
        alphabet = ['<pad>'] + fo.readlines()

    char2ind = {alphabet[i].replace('\n', ''):i for i in range(len(alphabet))}
    ind2char = {char2ind[key]:key for key in char2ind}

    ctc_decoder = CTCDecoder(alphabet)
    
    device = torch.device("cuda:"+str(device_id) if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(alphabet_size=len(alphabet), batch_size=batch_size, maxlen=maxlen)
    model.load_state_dict(torch.load(os.path.join(model_path, "model_best.pth")))
    model = model.to(device)

    test_dataset = Data(test_path, aud_path, char2ind, [extract_feats, encode_trans], maxlen, maxlent)
    loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    total_WER = 0
    total_CER = 0
    step = 0
    num_steps = len(loader)

    targets = []
    predicted = []
    
    print("Total number of examples: ", num_steps*batch_size)
    
    for batch in loader:
        step+=1
        print("Decoding step {}/{}...".format(step, num_steps))
        batch_WER = 0
        batch_CER = 0

        x = batch['aud'].to(device)
        t = batch['trans'].to(device)
        fmask = batch['fmask'].squeeze(1).to(device)
        tmask = batch['tmask'].squeeze(1).to(device)
        preds = model(x, t, fmask, device)
        preds = torch.transpose(preds, 0, 1)

        preds = preds.detach().cpu().numpy()
        t = t.detach().cpu().numpy()
        #fmask = fmask.detach().cpu().numpy()
        tmask = tmask.detach().cpu().numpy()
        for i, probs in enumerate(preds):
            pad_ind = int(np.sum(tmask[i]))
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
'''