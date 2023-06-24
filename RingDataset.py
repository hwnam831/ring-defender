import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class RingDataset(Dataset):
    def __init__(self, pklfile, threshold=40, history=32, std=None):
        datalist = pickle.load(open(pklfile, 'rb'))
        targets = []
        inputs = []
        self.offset = history-1
        for data in datalist:
            for i in range(1,len(data)):
                bit, trace = data[i]
                if data[i-1][0] == 0:
                    continue
                if len(trace) >= threshold+self.offset:
                    targets.append(bit)
                    inputs.append(list(trace[:threshold+self.offset]))
        one_count = np.array(targets).sum()
        one_count = min(one_count, len(targets)-one_count)

        self.target_arr = np.zeros([one_count*2], dtype=np.long)
        self.input_arr = np.zeros([one_count*2, threshold+self.offset], dtype=np.float32)
        bcounts = [0,0]
        cnt = 0
        for i,bit in enumerate(targets):
            if bcounts[bit] >= one_count:
                continue
            self.target_arr[cnt] = bit
            self.input_arr[cnt] = np.array(inputs[i])
            cnt += 1
            bcounts[bit] += 1
                    
        self.med = np.median(self.input_arr)
        std1 = np.std(self.input_arr)
        #round to closest pow of 2
        std2 = 1
        while(std2 < std1):
            std2 = std2*2
        std2 = std2 if (std2-std1) < (std1-std2//2) else std2//2
        self.window=threshold
        self.std = std if std else std2
        self.input_arr = self.input_arr - self.med
        self.input_arr = self.input_arr/self.std
    
    def __len__(self):
        return len(self.input_arr)
    
    def __getitem__(self, idx):
        return self.input_arr[idx], self.target_arr[idx]

class EDDSADataset(Dataset):
    def __init__(self, pklfile, std=16, history=8, window=None):
        x_arr, y_arr = pickle.load(open(pklfile, 'rb'))
        self.offset = history-1
        self.window = window if window else x_arr.shape[1] - self.offset
        self.target_arr = y_arr
        self.input_arr = x_arr[:,:self.offset+self.window]
        bcounts = [0,0]
        cnt = 0
                    
        median = np.median(self.input_arr)
        self.med = median
        self.std = std
        self.input_arr = self.input_arr - median
        self.input_arr = self.input_arr/self.std
    
    def __len__(self):
        return len(self.input_arr)
    
    def __getitem__(self, idx):
        return self.input_arr[idx], self.target_arr[idx]
    
class LOTRDataset(Dataset):
    def __init__(self, pklfile, std=16, med=None):
        x_arr, y_arr = pickle.load(open(pklfile, 'rb'))

        self.target_arr = y_arr
        self.input_arr = x_arr
        self.tracelen = x_arr.shape[1]            
        self.med = med if med else np.median(self.input_arr)
        self.std = std
        self.input_arr = self.input_arr - self.med
        self.input_arr = self.input_arr/self.std
    
    def __len__(self):
        return len(self.input_arr)
    
    def __getitem__(self, idx):
        return self.input_arr[idx], self.target_arr[idx]
