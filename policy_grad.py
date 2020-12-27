from metrics import edit_dist
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

def sample_trans(probs, mask, m=15):
    '''
    Samples M transcriptions from the probability distribution:
    e.g. softmax output.
    Args:
        m: number of samples
        probs: softmax output for current training example
    '''
    
    pad_ind = int(np.sum(mask))
    probs = np.exp(probs[:pad_ind+1])
    sampled_trans = []
    for i in range(m):
        y_m = []
        for distr in probs:
            char_ind = np.random.choice(probs.shape[1], 1, p=distr)
            y_m.append(int(char_ind))
            if char_ind==1:
                break
        sampled_trans.append(y_m)
    print(sample_trans)
    return np.array(sampled_trans)