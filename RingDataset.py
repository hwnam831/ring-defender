import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class RingDataset(Dataset):
    def __init__(self, pklfile, threshold=44):
        datalist = pickle.load(open(pklfile, 'rb'))
        targets = []
        inputs = []
        for data in datalist:
            for i in range(1,len(data)):
                bit, trace = data[i]
                if data[i-1][0] == 0:
                    continue
                if len(trace) >= threshold:
                    targets.append(bit)
                    inputs.append(list(trace[:threshold]))
                elif len(trace) >= threshold//2 and i < len(data)-1 and len(data[i+1][1]) + len(trace) >= threshold:
                    pass
                    #targets.append(bit)
                    #inputs.append(list(trace) + list(data[i+1][1][:threshold-len(trace)]))
                    
        self.target_arr = np.array(targets)
        self.input_arr = np.array(inputs,dtype=np.float32)
        median = np.median(self.input_arr)
        std = np.std(self.input_arr)
        #round to closest pow of 2
        std2 = 1
        while(std2 < std):
            std2 = std2*2
        std2 = std2 if (std2-std) < (std-std2//2) else std2//2

        self.input_arr = self.input_arr - median
        self.input_arr = self.input_arr/std2
    
    def __len__(self):
        return len(self.input_arr)
    
    def __getitem__(self, idx):
        return self.input_arr[idx], self.target_arr[idx]

if __name__ == '__main__':
    dataset = RingDataset('core2ToSlice3.pkl')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    '''
    for i in range(10):
        idx = np.random.randint(0,len(dataset))
        x,y = dataset.__getitem__(idx)
        print(x)
        print(y)
    '''
    for x,y in loader:
        print(x)
        print(y)
        break