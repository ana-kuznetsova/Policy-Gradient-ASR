import sys 
import numpy as np
from tqdm import tqdm
import shutil


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import logging

SOS_TOKEN = '#'
PAD_TOKEN = '@'

def glorot_normal_initializer(m):
    """ Applies Glorot Normal initialization to layer parameters.
    
    "Understanding the difficulty of training deep feedforward neural networks" 
    by Glorot, X. & Bengio, Y. (2010)

    Args:
        m (nn.Module): a particular layer whose params are to be initialized.
    """
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

class CharRNN(nn.Module):
    """ Character-level recurrent neural network.

    This module imitates the basic structure and flow of an 
    RNN with embeddings, hidden states and output softmax.
    Alternatively you can use `LSTM` and `GRU` recurrent modules.

    Args:
        n_tokens (int): number of unique tokens in corpus.
        emb_size (int): dimensionality of each embedding.
        hidden_size (int): number of hidden units in RNN hidden layer.
        pad_id (int): token_id of the padding token.
    """

    def __init__(self, num_layers, n_tokens, emb_size, hidden_size, dropout, pad_id):
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
#        self.rnn_type = rnn_type.upper()

        self.embedding = nn.Embedding(n_tokens, emb_size, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
#        if self.rnn_type in ['RNN', 'LSTM', 'GRU']:
#        self.rnn = getattr(nn, self.rnn_type)(input_size=emb_size, hidden_size=hidden_size, 
#                                                num_layers=num_layers, dropout=dropout, batch_first=True)
        self.rnn = getattr(nn, 'LSTM')(input_size=emb_size, hidden_size=hidden_size, 
                                                num_layers=num_layers, dropout=dropout, batch_first=True)
#        else:
#            raise UserWarning('Unknown RNN type.')
        self.logits = nn.Linear(hidden_size, n_tokens).apply(glorot_normal_initializer)

    def forward(self, input_step, hidden):
        """ Implements the forward pass of the char-level RNN.

        Args:
            inputs (torch.LongTensor): input step token batch to feed the network.
            hidden (torch.Tensor): hidden states of the RNN from previous time-step.
        Returns:
            torch.Tensor: output log softmax probability distribution over tokens.
            torch.Tensor: hidden states of the RNN from current time-step.
        """
        embedded = self.embedding(input_step)
        embedded = self.dropout(embedded)
        outputs, hidden = self.rnn(embedded.view(input_step.size(0), 1, -1), hidden)
        logits = self.logits(outputs.view(-1, self.hidden_size))
        probs = F.log_softmax(logits, dim=1)
        return probs, hidden

    def initHidden(self, batch_size):
        #if self.rnn_type == 'LSTM':
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size))
        #return torch.zeros(self.num_layers, batch_size, self.hidden_size)


def sequences_to_tensors(sequences, token_to_idx):
    """ Casts a list of sequences into rnn-digestable padded tensor """
    seq_idx = []
    for seq in sequences:
        seq_idx.append([token_to_idx[token] for token in seq])
    sequences = [torch.LongTensor(x) for x in seq_idx]
    return nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=token_to_idx[PAD_TOKEN])

def load_dataset(file_name='../data/transcripts.txt'):

    with open(file_name) as file:
        sequences = file.read()[:-1].split('\n')
        sequences = [SOS_TOKEN + seq.lower() for seq in sequences]

    logging.info('number of sequences: {}'.format(len(sequences)))
#     for seq in sequences[::1000]:
#         print(seq[1:])

    MAX_LENGTH = max(map(len, sequences))
    logging.info('max length: {}'.format(MAX_LENGTH))

    idx_to_token = list(set([token for seq in sequences for token in seq]))
    idx_to_token.append(PAD_TOKEN)
    n_tokens = len(idx_to_token)
    print('number of unique tokens: {}'.format(n_tokens))

    token_to_idx = {token: idx_to_token.index(token) for token in idx_to_token}
    assert len(idx_to_token) ==  len(token_to_idx), 'dicts must have same lenghts'

    logging.debug('processing tokens')
    sequences = sequences_to_tensors(sequences, token_to_idx)
    return sequences, token_to_idx, idx_to_token

class LanguageModel:
    
    def __init__(self):
        pass

    def iterate_minibatches(self, inputs, batchsize, shuffle=False):
        if shuffle:
            indices = np.random.permutation(inputs.size(0))
        for start_idx in range(0, inputs.size(0) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt]

    def train(self, filename, num_layers, dropout, emb_size, 
            hidden_size, num_epochs, batch_size, learning_rate, checkpoint_path="./lm.ckp"):
        """ Trains a character-level Recurrent Neural Network in PyTorch.

        Args: optional arguments [python train.py --help]
        """
        print('reading `{}` for character sequences'.format(filename))
        inputs, token_to_idx, idx_to_token = load_dataset(file_name=filename)

        idx_to_token.remove(SOS_TOKEN)
        idx_to_token.remove(PAD_TOKEN)
        idx_to_token = [SOS_TOKEN] + idx_to_token + [PAD_TOKEN]
        token_to_idx = {token: idx_to_token.index(token) for token in idx_to_token}

        logging.info(idx_to_token)
        logging.info(token_to_idx)

        n_tokens = len(idx_to_token)
        max_length = inputs.size(1)

        print('creating char-level RNN model')
        model = CharRNN(num_layers=num_layers,  
                        dropout=dropout, n_tokens=n_tokens,
                        emb_size=emb_size, hidden_size=hidden_size, 
                        pad_id=token_to_idx[PAD_TOKEN])
        if torch.cuda.is_available():
            model = model.cuda()

        logging.debug('defining model training operations')
        # define training procedures and operations for training the model
        criterion = nn.NLLLoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-6, 
                                                        factor=0.1, patience=7, verbose=True)

        # train-val-test split of the dataset
        split_index = int(0.9 * inputs.size(0))
        train_tensors, inputs = inputs[: split_index], inputs[split_index: ]
        split_index = int(0.5 * inputs.size(0))
        val_tensors, test_tensors = inputs[: split_index], inputs[split_index: ]
        del inputs
        print('train tensors: {}'.format(train_tensors.size()))
        print('val tensors: {}'.format(val_tensors.size()))
        print('test tensors: {}'.format(test_tensors.size()))

        print('training char-level RNN model')
        # loop over epochs
        for epoch in range(1, num_epochs + 1):
            epoch_loss, n_iter = 0.0, 0
            # loop over batches
            for tensors in tqdm(self.iterate_minibatches(train_tensors, batchsize=batch_size),
                            desc='Epoch[{}/{}]'.format(epoch, num_epochs), leave=False,
                            total=train_tensors.size(0) // batch_size):
                # optimize model parameters
                epoch_loss += self.optimize(model, tensors, max_length, n_tokens, criterion, optimizer)
                n_iter += 1
            # evaluate model after every epoch
            val_loss = self.evaluate(model, val_tensors, max_length, n_tokens, criterion)
            # lr_scheduler decreases lr when stuck at local minima 
            scheduler.step(val_loss)
            # log epoch status info
            logging.info('Epoch[{}/{}]: train_loss - {:.4f}   val_loss - {:.4f}'.format(epoch, num_epochs, epoch_loss / n_iter, val_loss))

            # sample from the model every few epochs
    #        if epoch % sample_every == 0:
    #            print('Epoch[{}/{}]: train_loss - {:.4f}   val_loss - {:.4f}'.format(epoch, num_epochs, epoch_loss / n_iter, val_loss))
    #            for _ in range(num_samples):
    #                sample = generate_sample(model, token_to_idx, idx_to_token, 
    #                                        max_length, n_tokens, seed_phrase="")
    #                logging.debug(sample)

            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': val_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            # save checkpoint
            best_model_path = checkpoint_path
            self.save_ckp(checkpoint, False, checkpoint_path, best_model_path)
            logging.info('Saved Checkpoint')
        return model, token_to_idx, idx_to_token

    def save_ckp(self, state, is_best, checkpoint_path, best_model_path):
        """
        state: checkpoint we want to save
        is_best: is this the best checkpoint; min validation loss
        checkpoint_path: path to save checkpoint
        best_model_path: path to save best model
        """
        f_path = checkpoint_path
        # save checkpoint data to the path given, checkpoint_path
        torch.save(state, f_path)
        # if it is a best model, min validation loss
        if is_best:
            best_fpath = best_model_path
            # copy that checkpoint file to best path given, best_model_path
            shutil.copyfile(f_path, best_fpath)

    def load_ckp(self, checkpoint_fpath, model, optimiser):
        """
        checkpoint_path: path to save checkpoint
        model: model that we want to load checkpoint parameters into       
        optimiser: optimiser we defined in previous training
        """
        # load check point
        checkpoint = torch.load(checkpoint_fpath)
        # initialize state_dict from checkpoint to model
        model.load_state_dict(checkpoint['state_dict'])
        # initialize optimiser from checkpoint to optimizer
        optimiser.load_state_dict(checkpoint['optimizer'])
        # initialize valid_loss_min from checkpoint to valid_loss_min
        valid_loss_min = checkpoint['valid_loss_min']
        # return model, optimiser, epoch value, min validation loss 
        return model, optimiser, checkpoint['epoch'], valid_loss_min #.item()



    def optimize(self, model, inputs, max_length, n_tokens, criterion, optimizer):
        model.train()
        optimizer.zero_grad()
        # compute outputs after one forward pass
        outputs = self.forward(model, inputs, max_length, n_tokens)
        # ignore the first timestep since we don't have prev input for it
        # (timesteps, batches, 1) -> (timesteps x batches x 1)
        targets = inputs[:, 1: ].contiguous().view(-1)
        # compute loss wrt targets
        loss = criterion(outputs, targets)
        # backpropagate error
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        # update model parameters
        optimizer.step()
        return loss.item()

    def evaluate(self, model, inputs, max_length, n_tokens, criterion):
        model.eval()
        # compute outputs after one forward pass
        outputs = self.forward(model, inputs, max_length, n_tokens)
        # ignore the first timestep since we don't have prev input for it
        # (timesteps, batches, 1) -> (timesteps x batches x 1)
        targets = inputs[:, 1: ].contiguous().view(-1)
        # compute loss wrt targets
        loss = criterion(outputs, targets)
        return loss.item()

    def forward(self, model, inputs, max_length, n_tokens):
        hidden = model.initHidden(inputs.size(0))
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            if type(hidden) == tuple:
                hidden = tuple([x.cuda() for x in hidden])
            else:
                hidden = hidden.cuda()
        # tensor for storing outputs of each time-step
        outputs = torch.Tensor(max_length, inputs.size(0), n_tokens)
        # loop over time-steps
        for t in range(max_length):
            # t-th time-step input
            input_t = inputs[:, t]
            outputs[t], hidden = model(input_t, hidden)
        # (timesteps, batches, n_tokens) -> (batches, timesteps, n_tokens)
        outputs = outputs.permute(1, 0, 2)
        # ignore the last time-step since we don't have a target for it.
        outputs = outputs[:, :-1, :]
        # (batches, timesteps, n_tokens) -> (batches x timesteps, n_tokens)
        outputs = outputs.contiguous().view(-1, n_tokens)
        return outputs

    def score(self, model, token_to_idx, idx_to_token, seed_phrase):
        """ Generates samples using seed phrase.

        Args:
            model (nn.Module): the character-level RNN model to use for sampling.
            token_to_idx (dict of `str`: `int`): character to token_id mapping dictionary (vocab).
            idx_to_token (list of `str`): index (token_id) to character mapping list (vocab).
            max_length (int): max length of a sequence to sample using model.
            seed_phrase (str): the initial seed characters to feed the model. If unspecified, defaults to `SOS_TOKEN`.

        Returns:
            str: generated sample from the model using the seed_phrase.
        """
        model.eval()
        if seed_phrase[0] != SOS_TOKEN:
            seed_phrase = SOS_TOKEN + seed_phrase.lower()
        try:
            # convert to token ids for model
            sequence = [token_to_idx[token] for token in seed_phrase]
        except KeyError as e:
            logging.error('unknown token: {}'.format(e))
            return 0


        print('score:', seed_phrase, file=sys.stderr)
#         print('score:', sequence, file=sys.stderr)
        input_tensor = torch.LongTensor([sequence])

        hidden = model.initHidden(1)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            if type(hidden) == tuple:
                hidden = tuple([x.cuda() for x in hidden])
            else:
                hidden = hidden.cuda()

        # feed the seed phrase to manipulate rnn hidden states
        for t in range(len(sequence) - 1):
            _, hidden = model(input_tensor[:, t], hidden)

        # start generating
        score = 0.0
        for i in range(len(seed_phrase)):
            # sample char from previous time-step
            input_tensor = torch.LongTensor([sequence[-1]])
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            probs, hidden = model(input_tensor, hidden)
            # need to use `exp` as output is `LogSoftmax`
            probs = list(np.exp(np.array(probs.data[0].cpu())))
            # normalize probabilities to ensure sum = 1
            probs /= sum(probs)
            c = seed_phrase[i]
            ci = token_to_idx[c]
            score += probs[ci]
            print(c, probs[ci],  file=sys.stderr)

            # sample char randomly based on probabilities
    #        sequence.append(np.random.choice(len(idx_to_token), p=probs))
    #        sequence.append(
        # format the string to ignore `pad_token` and `start_token` and return
    #    res = str(''.join([idx_to_token[ix] for ix in sequence 
    #                if idx_to_token[ix] != PAD_TOKEN and idx_to_token[ix] != SOS_TOKEN]))
        score = score/len(seed_phrase)
        print('SCORE:', score, file=sys.stderr)
        return score

if __name__ == "__main__":
    logging.root.setLevel(logging.NOTSET)
    lm = LanguageModel()
    lm_model, token_to_idx, idx_to_token = lm.train(filename = "transcripts.txt", num_layers=2, dropout=0.05, emb_size=100,                    hidden_size=200, num_epochs=100, batch_size=10, learning_rate=0.01)