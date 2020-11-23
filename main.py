from model import TrainData
from data import make_feats
import torch.utils.data as data





if __name__ == "__main__":
    cv_dataset = TrainData('/N/slate/anakuzne/Data/asr/policy_gradient/eu/train.tsv',
                            '/N/slate/anakuzne/Data/asr/policy_gradient/eu/clips', make_feats)
    
    loader = data.DataLoader(cv_dataset, batch_size=32, shuffle=True)
    for ins in loader:
        print(ins["aud"].shape)
        