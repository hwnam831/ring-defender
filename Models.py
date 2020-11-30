
""" argparse configuration""" 

import os
import argparse
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn as nn

def shifter(arr, window=32):
    dup = arr[:,None,:].expand(arr.size(0), arr.size(1)+1, arr.size(1))
    dup2 = dup.reshape(arr.size(0), arr.size(1), arr.size(1)+1)
    shifted = dup2[:,:window,:-window]
    return shifted

def ConvBlockRelu(c_in, c_out, ksize, dilation=1):
    pad = ((ksize-1)//2)*dilation
    return nn.Sequential(
            nn.Conv1d(c_in, c_out, ksize, 1, pad, dilation=dilation),
            #nn.BatchNorm1d(c_out),
            nn.ReLU())

class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
                ConvBlockRelu(c_in, c_mid, 1),
                ConvBlockRelu(c_mid, c_mid, 5, dilation=dilation),
                ConvBlockRelu(c_mid, c_in, 1))

    def forward(self, x):
        out = self.block(x)
        out += x
        return out

class CNNModel(nn.Module):
    def __init__(self, threshold, dim=128, drop=0.1):
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

class CNNModel2(nn.Module):
    def __init__(self, threshold, dim=128, drop=0.1):
        super().__init__()

        self.CNN = nn.Sequential(
            nn.Conv1d(1, 32, 11, 1, 5),
            nn.Dropout(drop),
            nn.ReLU(),
            ResBlock(32, 16),
            ResBlock(32, 16),
            nn.Conv1d(32, 64, 5, 2, 2),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            ResBlock(64, 32),
            ResBlock(64, 32),
            nn.Conv1d(64, 128, 5, 2, 2),
            nn.Dropout(drop),
            nn.ReLU(),
            ResBlock(128, 64),
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

class RNNModel(nn.Module):
    def __init__(self, threshold, dim=128, drop=0.1):
        super().__init__()

        self.CNN = nn.Sequential(
            nn.Conv1d(1, 32, 11, 1, 5),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, 2, 2),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
        )
        self.RNN = nn.GRU(128,dim, num_layers=2, batch_first=True, dropout=drop)
        self.FC = nn.Sequential(
            nn.Linear(dim*((threshold+3)//4), 512),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(512,2)
        )   
    def forward(self, x):
        out = self.CNN(x.view(x.size(0),1,x.size(1))) # B, C, S
        out, _ = self.RNN(out.permute(0, 2, 1)) # B, S, C
        out = self.FC(out.reshape(out.size(0),-1))
        return out

class CNNGenerator(nn.Module):
    def __init__(self, threshold, scale=1, window=32, drop=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(window, 64, 1, bias=True),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1, bias=True),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        
        self.resblock = nn.Sequential(
                nn.Conv1d(128, 32, 1, bias=True),
                nn.ReLU(),
                ResBlock(32, 16),
                ResBlock(32, 16),
                nn.Conv1d(32, 64, 5, 1, 2),
                nn.Dropout(drop),
                nn.ReLU(),
                #nn.MaxPool1d(2),
                ResBlock(64, 32),
                ResBlock(64, 32),
                nn.Conv1d(64, 128, 5, 1, 2),
                nn.Dropout(drop),
                nn.ReLU(),
                ResBlock(128, 64),
            )

        self.decoder = nn.Sequential(
            nn.Conv1d(128, 1, 1, bias=True),
            #nn.ReLU(),
        )
        self.scale = scale
    def forward(self, x):
        signal = x[:,-1,:]
        noise = self.scale*torch.randn_like(signal) #gaussian
        out = self.encoder(x)
        out = out + self.resblock(out)
        out = self.decoder(out).view(out.size(0),-1)
        return torch.relu(out+noise)

class GaussianGenerator(nn.Module):
    def __init__(self, threshold, scale=1, window=32, drop=0.2):
        super().__init__()
        self.bias = nn.Parameter(torch.rand(1,dtype=torch.float))
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float)*scale)
        self.threshold = threshold
    def forward(self,x):
        #assuming N,C,S
        signal = x[:,-1,:]
        noise = torch.randn_like(signal)*self.scale
        #perturb = torch.randn([x.size(0), self.threshold], device=x.device) #[N, S]
        noise += torch.rand_like(signal)*self.scale - self.scale/2
        perturb = noise - signal + self.bias
        return perturb


class GaussianSinusoid(nn.Module):
    def __init__(self, threshold, scale=1, window=32, drop=0.2):
        super().__init__()
        self.bias = nn.Parameter(torch.rand(1,dtype=torch.float))
        self.amp = nn.Parameter(torch.ones(1, dtype=torch.float)*scale/2)
        self.freq = nn.Parameter(torch.ones(1,dtype=torch.float)/(threshold/8))
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float)*scale)
        self.threshold=threshold
    def forward(self,x):
        #assuming N,C,S
        signal = x[:,-1,:]
        noise = torch.randn_like(signal)*self.scale
        noise += torch.rand_like(signal)*self.scale - self.scale/2
        omega = 2*np.pi*self.freq
        t = torch.arange(self.threshold, device=x.device, dtype=torch.float)
        theta = torch.rand(x.size(0),device=x.device)*(2*np.pi)
        sinu = self.amp*torch.sin((omega*t)[None,:] + theta[:,None])
        #perturb = torch.randn([x.size(0), self.threshold], device=x.device) #[N, S]
        noise += sinu
        perturb = noise - signal + self.bias
        return perturb

class OffsetGenerator(nn.Module):
    def __init__(self, threshold, scale=1, window=32, drop=0.2):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float)*scale)
        self.threshold=threshold
    def forward(self,x):
        #assuming N,C,S
        signal = x[:,-1,:]
        noise = torch.ones_like(signal)*self.scale
        perturb = torch.relu(noise - signal)
        return perturb

class RNNGenerator(nn.Module):
    def __init__(self, threshold, scale=1, dim=128, window=32, drop=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window, dim),
            nn.Dropout(drop)
        )

        self.resblock = nn.GRU(dim,dim, num_layers=2, batch_first=True, dropout=drop)

        self.decoder = nn.Sequential(
            nn.Linear(dim, 1)
        )
        self.scale = nn.Parameter(torch.ones(1))
        self.noise = GaussianSinusoid(threshold)

    def forward(self, x):
        signal = x[:,-1,:]
        #noise = torch.ones_like(signal)*self.scale #offset
        noise = self.scale*torch.randn_like(signal) #gaussian
        
        #noise = torch.relu(noise - signal) #Maya-like
        #x[:,-1,:] += noise
        #xx = x + shifter(noise)
        #xx = x
        out = self.encoder(x.permute(0,2,1)) #N,C,S -> N,S,C
        
        res, _ = self.resblock(out)
        out = out + res #N,S,C
        out = self.decoder(out).view(out.size(0),-1)
        #out = out + self.scale*torch.randn_like(out)
        out = out + noise
        
        return torch.relu(out)   

def MLP(threshold, dim):
        model = nn.Sequential(
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
        )
        return model