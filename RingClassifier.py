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
    def __init__(self, threshold, drop=0.1):
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
    def __init__(self, threshold, window=32, drop=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(window, 64, 1, bias=True),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1, bias=True),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        
        self.resblock = nn.Sequential(
                nn.Conv1d(128, 64, 1, bias=True),
                nn.ReLU(),
                #nn.ConstantPad1d((2,0),0.0),
                nn.Conv1d(64, 128, 1, bias=True),
                nn.ReLU(),
                nn.Dropout(drop)
            )

        self.decoder = nn.Sequential(
            nn.Conv1d(128, 1, 1, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = out + self.resblock(out)
        out = self.decoder(out)
        return out.view(out.size(0),-1)

class GaussianGenerator(nn.Module):
    def __init__(self, threshold, scale=1, window=32, drop=0.2):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(threshold,dtype=torch.float))
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float)*scale)
    def forward(self,x):
        #assuming N,C,S
        perturb = torch.randn([x.size(0), threshold], device=x.device) #[N, S]
        perturb = perturb*self.scale + self.bias
        return perturb


class GaussianSinusoid(nn.Module):
    def __init__(self, threshold, scale=1, window=32, drop=0.2):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(threshold,dtype=torch.float))
        self.amp = nn.Parameter(torch.ones(1, dtype=torch.float)*scale)
        self.freq = nn.Parameter(torch.ones(1,dtype=torch.float)/(threshold/8))
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float)*scale)
        self.threshold=threshold
    def forward(self,x):
        #assuming N,C,S
        noise = self.scale*torch.randn([x.size(0), self.threshold], device=x.device) #[N, S]
        omega = 2*np.pi*self.freq
        t = torch.arange(self.threshold, device=x.device, dtype=torch.float)
        theta = torch.randn(x.size(0),device=x.device)*(2*np.pi)
        sinu = self.amp*torch.sin((omega*t)[None,:] + theta[:,None])
        perturb = sinu[None,:] + noise + self.bias
        return perturb

class RNNGenerator(nn.Module):
    def __init__(self, threshold, window=32, drop=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window, 128),
            nn.Dropout(drop)
        )
        
        self.resblock = nn.GRU(128,128, num_layers=2, batch_first=True, dropout=drop)

        self.decoder = nn.Sequential(
            nn.Linear(128, 1)
        )
        self.scale = nn.Parameter(torch.ones(1))
        self.noise = GaussianSinusoid(threshold)

    def forward(self, x):
        out = self.encoder(x.permute(0,2,1)) #N,C,S -> N,S,C
        res, _ = self.resblock(out)
        out = out + res #N,S,C
        out = self.decoder(out)
        gaussian = self.scale*torch.randn_like(out)
        out = torch.relu(out + gaussian)
        return out.view(out.size(0),-1)    


def shifter(arr, window=32):
    dup = arr[:,None,:].expand(arr.size(0), arr.size(1)+1, arr.size(1))
    dup2 = dup.reshape(arr.size(0), arr.size(1), arr.size(1)+1)
    shifted = dup2[:,:window,:-window]
    return shifted

if __name__ == '__main__':
    threshold = 42
    epochs = 50
    dim=128
    dataset = RingDataset.RingDataset('core4ToSlice3.pkl', threshold=threshold)
    testset =  RingDataset.RingDataset('core4ToSlice3_test.pkl', threshold=threshold)
    #testlen = dataset.__len__()//4
    #trainlen = dataset.__len__() - testlen
    #testset, trainset = random_split(dataset, [testlen, trainlen], generator=torch.Generator().manual_seed(17))
    #testset, trainset = random_split(dataset, [testlen, trainlen])
    trainset=dataset
    trainloader = DataLoader(trainset, batch_size=64, num_workers=2)
    testloader = DataLoader(testset, batch_size=64)
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
    MLPgen = nn.Sequential(
        nn.Linear(threshold,dim//2),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(dim//2,dim),
        nn.ReLU(),
        nn.Linear(dim,dim),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(dim,threshold),
        nn.ReLU()
    ).cuda()
    cnn = CNNModel(threshold).cuda()
    #gen = CNNGenerator(threshold).cuda()
    gen=RNNGenerator(threshold).cuda()
    GG = nn.Sequential(
        GaussianGenerator(threshold, scale=20),
        nn.ReLU(),
    ).cuda()
    
    GS = nn.Sequential(
        GaussianSinusoid(threshold, scale=20),
        nn.ReLU(),
    ).cuda()
    #gen = GS
    #gen=MLPgen
    '''
    for x,y in trainloader:
        xp = gen(x.cuda())
        print(x.shape)
        print(xp.shape)
        break
    '''
    classifier = cnn
    optim_c = torch.optim.Adam(classifier.parameters(), lr=2e-5)
    optim_g = torch.optim.Adam(gen.parameters(), lr=4e-5)

    criterion = nn.CrossEntropyLoss()
    C = 40.0
    repeat=2
    warmup = 5
    scale = 0.001

    for e in range(warmup):
        classifier.train()
        for x,y in trainloader:
            xdata, ydata = x.cuda(), y.cuda()
            #train classifier
            optim_c.zero_grad()
            #interleaving?
            output = classifier(xdata[:,31:])
            loss_c = criterion(output, ydata)
            loss_c.backward()
            optim_c.step()

    for e in range(epochs):
        gen.train()
        classifier.train()
        for x,y in trainloader:
            xdata, ydata = x.cuda(), y.cuda()
            shifted = shifter(xdata)
            #train classifier
            optim_c.zero_grad()
            perturb = gen(shifted).view(shifted.size(0),-1)
            #perturb = gen(xdata[:,31:])
            #interleaving?
            output = classifier(xdata[:,31:]+perturb.detach())
            loss_c = criterion(output, ydata)
            loss_c.backward()
            optim_c.step()

            #train generator
            optim_g.zero_grad()
            perturb = gen(shifted).view(shifted.size(0),-1)
            #perturb = gen(xdata[:,31:])
            output = classifier(xdata[:,31:] + perturb)
            pnorm = torch.norm(perturb, dim=-1) - C
            loss_p = torch.mean(torch.relu(pnorm))
            #loss_p = torch.mean(torch.norm(perturb,dim=-1))
            fake_target = 1-ydata
            loss_adv1 = criterion(output, fake_target)
            loss_adv0 = criterion(output, ydata)
            loss = loss_adv1 + scale*loss_p
            #loss = loss_adv1
            loss.backward()
            optim_g.step()

        mloss = 0.0
        macc = 0.0
        mnorm = 0.0
        #evaluate classifier
        with torch.no_grad():
            classifier.eval()
            gen.eval()
            for x,y in testloader:
                xdata, ydata = x.cuda(), y.cuda()
                shifted = shifter(xdata)
                perturb = gen(shifted).view(shifted.size(0),-1)
                #perturb = gen(xdata[:,31:])
                norm = torch.mean(perturb)
                output = classifier(xdata[:,31:]+perturb)
                loss_c = criterion(output, ydata)
                pred = output.argmax(axis=-1)
                mnorm += norm.item()/len(testloader)
                mloss += loss_c.item()/len(testloader)
                macc += ((pred==ydata).sum().float()/pred.nelement()).item()/len(testloader)
            print("epoch {} \t acc {:.6f}\t loss {:.6f}\t Avg perturb {:.6f}\n".format(e+1, macc, mloss, mnorm))


