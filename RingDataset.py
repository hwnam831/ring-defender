import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class RingDataset(Dataset):
    def __init__(self, pklfile, threshold=40):
        datalist = pickle.load(open(pklfile, 'rb'))
        targets = []
        inputs = []
        self.offset = 31
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
    dataset = RingDataset('core4ToSlice3_test.pkl')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    '''
    for i in range(10):
        idx = np.random.randint(0,len(dataset))
        x,y = dataset.__getitem__(idx)
        print(x)
        print(y)
    '''
    print(len(dataset))
    onecount = 0
    for x,y in loader:
        onecount += y.sum().item()
    print(onecount)