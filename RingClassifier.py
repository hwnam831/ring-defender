import RingDataset
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn as nn

def ConvBlockRelu(c_in, c_out, ksize, dilation=1):
    pad = ((ksize-1)//2)*dilation
    return nn.Sequential(
            nn.Conv1d(c_in, c_out, ksize, 1, pad, dilation=dilation),
            #nn.BatchNorm1d(c_out),
            nn.ReLU())

class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid, dilation=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
                ConvBlockRelu(c_in, c_mid, 1),
                ConvBlockRelu(c_mid, c_mid, 5, dilation=dilation),
                ConvBlockRelu(c_mid, c_in, 1))

    def forward(self, x):
        out = self.block(x)
        out += x
        return out

class CNNModel(nn.Module):
    def __init__(self, threshold, drop=0.2):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv1d(1, 32, 11, 1, 5),
            nn.Dropout(drop),
            nn.ReLU(),
            ResBlock(32, 16),
            nn.Conv1d(32, 64, 5, 2, 2),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            ResBlock(64, 32),
            nn.Conv1d(64, 128, 5, 2, 2),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
        )
        self.FC = nn.Sequential(
            nn.Linear(128*((threshold+3)//4), 512),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(512,2)
        )
    def forward(self, x):
        out = self.CNN(x.view(x.size(0),1,x.size(1)))
        out = self.FC(out.view(out.size(0),-1))
        return out

class CNNGenerator(nn.Module):
    def __init__(self, threshold, drop=0.2):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv1d(1, 32, 11, 1, 5),
            nn.Dropout(drop),
            nn.ReLU(),
            ResBlock(32, 16),
            nn.Conv1d(32, 64, 5, 1, 2),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            ResBlock(64, 32),
            nn.Conv1d(64, 128, 5, 1, 2),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
        )
    def forward(self, x):
        out = self.CNN(x.view(x.size(0),1,x.size(1)))
        out = self.FC(out.view(out.size(0),-1))
        return out

if __name__ == '__main__':
    threshold = 42
    epochs = 50
    dim=512
    dataset = RingDataset.RingDataset('core4ToSlice3.pkl', threshold=threshold)
    testlen = dataset.__len__()//4
    trainlen = dataset.__len__() - testlen
    testset, trainset = random_split(dataset, [testlen, trainlen], generator=torch.Generator().manual_seed(17))
    #testset, trainset = random_split(dataset, [testlen, trainlen])
    trainloader = DataLoader(trainset, batch_size=16, num_workers=2)
    testloader = DataLoader(testset, batch_size=32)
    MLP = nn.Sequential(
        nn.Linear(threshold,dim//2),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(dim//2,dim),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(dim,dim),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(dim,2)
    ).cuda()
    cnn = CNNModel(threshold).cuda()
    model = cnn
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()
    for e in range(epochs):
        for x,y in trainloader:
            optimizer.zero_grad()
            model.train()
            output = model(x.cuda())
            loss = criterion(output, y.cuda())
            loss.backward()
            optimizer.step()
        mloss = 0.0
        macc = 0.0
        for x,y in testloader:
            model.eval()
            output = model(x.cuda())
            loss = criterion(output, y.cuda())
            pred = output.argmax(axis=-1)
            mloss += loss.item()/len(testloader)
            macc += ((pred==y.cuda()).sum().float()/pred.nelement()).item()/len(testloader)
        print("epoch {} \t acc {:.6f}\t loss {:.6f}\n".format(e+1, macc, mloss))


