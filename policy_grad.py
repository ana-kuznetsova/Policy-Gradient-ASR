from metrics import edit_dist
from CTCdecoder import collapse_fn
import numpy as np
import torch

def reward(true_y, pred_y, t, ind2char, ctc_decoder):

    pred_y , _ = ctc_decoder.decode(pred_y, beam_size=5)
    pred_y = ''.join([ind2char[ind] for ind in pred_y])
    pred_y = collapse_fn(pred_y)

    if t > 1:
        temp1 = edit_dist(true_y, pred_y[:t+1])
        temp2 = edit_dist(true_y, pred_y[:t])
        r_t = - (temp1 - temp2)
    elif t==1:
        r_t = - (edit_dist(true_y, pred_y[:t+1]) - len(true_y))
    return r_t

def sample_trans(probs, alphabet, m=15):
    '''
    Samples M transcriptions from the probability distribution:
    e.g. softmax output.
    Args:
        m: number of samples
        probs: softmax output for current training example
    '''
    
    sampled_trans = []
    for i in range(m):
        #Sample until get EOS
        y_m = []
        for distr in probs:
            char_ind = np.random.choice(len(alphabet), 1, p=distr)
            if int(char_ind) != 0:
                y_m.append(char_ind)
            else:
                y_m.append(char_ind)
                sampled_trans.append(y_m)
                y_m = []
                break
    return np.array(sampled_trans)