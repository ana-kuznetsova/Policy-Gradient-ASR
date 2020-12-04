import numpy as np

def edit_dist(s1, s2):
    '''
    Calculates edit distance between 2 sequences
    s1: reference sequence
    s2: predicted sequence
    Note: for CER calculation s1, s2 are of type str
          for WER calculation s1, s2 are of type list(str)
    '''
    dp = np.zeros((len(s2)+1, len(s1)+1))
    dp[0, :] = np.arange(len(s1)+1)
    dp[:,0] = np.arange(len(s2)+1)
    for i in range(1, len(s2)+1):
        for j in range(1, len(s1)+1):
            if s2[i-1] == s1[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(dp[i, j-1], dp[i-1, j-1], dp[i-1, j])
    return int(dp[-1, -1]), len(s1)

def evaluate(s1, s2):
    ed_dist, seq_len = edit_dist(s1, s2)
    cer = ed_dist/seq_len

    s1 = s1.split(" ")
    s2 = s2.split(" ")
    ed_dist, seq_len = edit_dist(s1, s2)
    wer = ed_dist/seq_len
    return cer, wer