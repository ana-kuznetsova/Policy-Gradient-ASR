from model import TrainData
from data import make_feats





if __name__ == "__main__":
    cv_dataset = TrainData('/N/slate/anakuzne/Data/asr/policy_gradient/eu/train.tsv',
                            '/N/slate/anakuzne/Data/asr/policy_gradient/eu/clips', make_feats)
    
    for i in range(5):
        sample = cv_dataset[i]
        print("Sample:", sample)