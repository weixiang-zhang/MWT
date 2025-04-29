import torch
from torch.utils.data import Dataset, DataLoader
import os

SAVED_PATH = "/home/wxzhang/projects/featuring-inr/data/feat_datasets"


class FeatDateset(Dataset):
    def __init__(self, type="cifar_10", train=True, transform=None):
        self.train_str = "train" if train else "test"
        self.transform = transform
        self.read_path = os.path.join(SAVED_PATH,f"{type}_{self.train_str}.pt")
        self.data = torch.load(self.read_path)
        self.x, self.y, self.feat = self.data
        self.y = self.y.squeeze().to(torch.int64)# 和cifar10原生数据集保持一致

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        feat = self.feat[idx]
        if self.transform:
            x = self.transform(x)

        return x,y,feat


####### how to use feat_datasets
def use_feat_dataset():
    fd_train = FeatDateset(type="cifar_10", train=True)
    fd_train_loader = DataLoader(fd_train,
            batch_size=16,
            num_workers=8,
            shuffle=True,
            pin_memory=True,
        )
    for step, batch in enumerate(fd_train_loader):
        x,y,feat = batch
        print()

        


    