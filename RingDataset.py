import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import NewModels
import sys
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
        print("Tracelen: " + str(self.tracelen))
    
    def __len__(self):
        return len(self.input_arr)
    
    def __getitem__(self, idx):
        return self.input_arr[idx], self.target_arr[idx]


if __name__=='__main__':
    file_prefix='rsa'
    if len(sys.argv) > 1:
        file_prefix = sys.argv[1]
    
    trainset = LOTRDataset(file_prefix+'_train.pkl')
    print("Train size: "+ str(len(trainset)))
    valset = LOTRDataset(file_prefix+'_valid.pkl', med=trainset.med)
    print("Valid size: "+ str(len(valset)))
    trainloader = DataLoader(trainset, batch_size=256, num_workers=4, shuffle=True)
    valloader = DataLoader(valset, batch_size=256, num_workers=4, shuffle=True)
    testset =  LOTRDataset(file_prefix+'_test.pkl', med=trainset.med)
    print("Test size: "+ str(len(testset)))
    testloader = DataLoader(testset, batch_size=256, num_workers=4)

    classifier = NewModels.ConvAttClassifier().to('cuda:0')
    #classifier = NewModels.ConvClassifier().to('cuda:0')
    optim_c = torch.optim.Adam(classifier.parameters(), lr=2e-4, weight_decay=2e-5)

    criterion = nn.CrossEntropyLoss()

    for e in range(200):
        classifier.train()

        for x,y in valloader:
            xdata, ydata = x.to('cuda:0'), y.to('cuda:0')
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio)
            #train classifier
            optim_c.zero_grad()

            output = classifier(xdata)


            
            loss_c = criterion(output, ydata)

            loss_c.backward()
            


            optim_c.step()

            pred = output.argmax(axis=-1)

    

        if (e+1)%10 == 0:
            totcorrect = 0
            totcount = 0
            mloss = 0.0
            closs = 0.0
            mperturb = 0.0
            with torch.no_grad():
                classifier.eval()

                for x,y in testloader:
                    xdata, ydata = x.to('cuda:0'), y.to('cuda:0')
                    oneratio = ydata.sum().item()/len(ydata)
                    disc_label = 2*(ydata.float()-oneratio)
                    #train classifier
                    optim_c.zero_grad()


                    output = classifier(xdata)
                    closs += criterion(output, ydata)

                    optim_c.step()

                    pred = output.argmax(axis=-1)
                    totcorrect += (pred==ydata).sum().item()
                    totcount += y.size(0)
            mloss = mloss / len(valloader)
            closs = closs / len(valloader)
            mperturb = mperturb/len(valloader)
            macc = float(totcorrect)/totcount
            print("Warmup epoch {} \t acc {:.4f}\t dloss {:.4f}\t  closs {:.4f}\t mperturb: {:.4f}".format(
                e+1, macc, mloss, closs, mperturb))