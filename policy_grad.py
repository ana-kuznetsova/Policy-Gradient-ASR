from metrics import edit_dist
from CTCdecoder import collapse_fn

def reward(true_y, pred_y, t, ind2char, ctc_decoder):

    pred_y , score = ctc_decoder.decode(pred_y, beam_size=5)
    pred_y = ''.join([ind2char[ind] for ind in pred_y])
    pred_y = collapse_fn(pred_y)

    if t > 1:
        temp1 = edit_dist(true_y, pred_y[:t+1])
        temp2 = edit_dist(true_y, pred_y[:t])
        r_t = - (temp1 - temp2)
    elif t==1:
        r_t = - (edit_dist(true_y, pred_y[:t+1]) - len(true_y))
    return r_t

def sample_trans(m, probs):
    '''
    Samples M transcriptions from the probability distribution:
    e.g. softmax output.
    Args:
        m: number of samples
        probs: softmax output for current training example
    '''
    pass