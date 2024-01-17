
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split

class PretrainDataset(Dataset):
    def __init__(self,data_path_lst,max_length=256,memmap=False):
        super().__init__()
        #
        if memmap:
            with open(data_path_lst[0],'r') as f:
                nbytes = f.seek(0,2)
                flen = f.tell() // np.dtype('uint16').itemsize
            self.data = np.memmap(data_path_lst[0],dtype=np.dtype('uint16'),shape=(flen//max_length,max_length))
        else:
            data_lst=[]
            for data_path in data_path_lst:
                with open(data_path,'rb') as f:
                    data=np.fromfile(f,dtype=np.uint16)
                    data_lst.append(data)
            data = np.concatenate(data_lst)
            data = data[:max_length*int(len(data)/max_length)]
            #np.random.shuffle(data)
            self.data = data.reshape(-1,max_length)
        #
        print("memmap:{} train data.shape:{}".format(memmap,self.data.shape))
        print("downloading finished.....")
        
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
    pass