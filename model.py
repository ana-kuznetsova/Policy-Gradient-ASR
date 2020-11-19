import os
import torch
import torch.optim as optim
import torch.utils.data as data

class Loader(data.Dataset):
    def __init__(self, feat_path):
        self.feat_path = feat_path
        self.feat_fnames = os.listdir(feat_path)
        self.feat_paths = [os.path.join(feat_path, p) for p in self.feat_fnames]
    def __getitem__(self, index):
        return torch.from_numpy(self.feat_path[index])
    def __len__(self):
        #Number of files
        return len(self.feat_paths)