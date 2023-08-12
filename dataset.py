
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split

class PretrainDataset(Dataset):
    def __init__(self,data_path_lst,max_length=256,finetune=False):
        super().__init__()
        data_lst=[]
        for data_path in data_path_lst:
            with open(data_path,'rb') as f:
                data=np.fromfile(f,dtype=np.uint16)
                data_lst.append(data)
        #
        data = np.concatenate(data_lst)
        data = data[:max_length*int(len(data)/max_length)]
        self.data = data.reshape(-1,max_length)
        print("downloading finished.....")
        self.max_length = max_length
        
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index: int):
        #
        sample = self.data[index]
        X=np.array(sample[:-1]).astype(np.int64)
        Y=np.array(sample[1:]).astype(np.int64)
        
        return torch.from_numpy(X),torch.from_numpy(Y)
#
if __name__=="__main__":
    data_path_lst=[
        './data/diagnosis/train.csv'
        ]
    
    train_ds = PretrainDataset(data_path_lst, max_length=256)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )
    for i, (X, Y) in enumerate(train_loader):
        print(X.shape,Y.shape)
        print(X[0])
        print(Y[0])
        break