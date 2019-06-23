import h5py
import numpy as np 
import torch
import os
import time
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torch.nn.parallel.data_parallel import DataParallel
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class MyDataset(Dataset):
    def __init__(self, truncation=2.5, flag = 'train'):
        super().__init__()
        self.truncation = truncation
        #print(self.truncation)
        if flag == 'train':
            self.database = [
        	    h5py.File("data/newtrain/flattened_train_1.h5", "r", libver='latest'),
        	    h5py.File("data/newtrain/flattened_train_2.h5", "r", libver='latest'),
        	    h5py.File("data/newtrain/flattened_train_3.h5", "r", libver='latest'),
        	    h5py.File("data/newtrain/flattened_train_4.h5", "r", libver='latest'),
        	    h5py.File("data/newtrain/flattened_train_5.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_6.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_7.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_8.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_9.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_10.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_11.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_12.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_13.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_14.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_15.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_16.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_17.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_18.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_19.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_20.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_21.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_22.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_23.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_24.h5", "r", libver='latest'),
                h5py.File("data/newtrain/flattened_train_25.h5", "r", libver='latest'),
            ]
            
        elif flag == 'val':
            self.database = [
        	    h5py.File("data/val/flattened_val_1.h5", "r", libver='latest'),
                h5py.File("data/val/flattened_val_2.h5", "r", libver='latest')
            ]
        
        
        elif flag == 'test':
            self.database = [
        	    h5py.File("data/test/flattened_test_1.h5", "r", libver='latest'),
                h5py.File("data/test/flattened_test_2.h5", "r", libver='latest')
            ]

        
    def __getitem__(self, index):        
        #print(type(self.truncation))
        file_index = int(index/4000)
        #print(file_index)

        f = self.database[file_index]
        data_index = index - 4000 * file_index
        
        #time1 = time.time()
        data = f.get('data')
        #print(data.shape)

        data = data[data_index].reshape(2,32,32,32)
        #print(data[0].shape)
        
        #only take the first channel
        data = data[0].reshape(1,32,32,32)
        #print(data.shape)
        #time2 = time.time()
        
        #print(time2-time1)
        target = f['target'][data_index].reshape(1,32,32,32)
        
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).float()


        #data.abs_().clamp_(max=self.truncation)
        data[0] = data[0].abs().clamp_(max=self.truncation)
        #print(data.shape)
        target[0] = target[0].clamp_(max=self.truncation)
        #print(target.shape)

        return data, target

    def __len__(self):
    	len_ = 0
    	for database in self.database:
    		len_ += database["data"].shape[0]
    	return len_
        

        #print(self.data.size())
        #print(self.target.size())
            
            #if data.shape[0] > 2:
             #   data[1:4].div_(255)
              #  target[1:].div_(255)
            
            #print(self.data.shape, self.target.shape)

if __name__ == '__main__':
    dataset = MyDataset(flag = 'val')
    print("dataset length: ", len(dataset))
    
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    #print(len(trainloader))
    print(len(trainloader))

    for i, (xv,yv) in enumerate(trainloader):
        break