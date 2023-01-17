
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
        out = out + x
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
            ResBlock(128, 64),
            nn.Conv1d(128, 256, 3, 1, 1),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
        )
        self.FC = nn.Sequential(
            nn.Linear(256*((threshold+3)//4), 512),
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

class CNNDiscriminator(nn.Module):
    def __init__(self, threshold, dim=128, drop=0.0):
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
            nn.Conv1d(128, 256, 3, 1, 1),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
        )
        self.FC = nn.Sequential(
            nn.Linear(256*((threshold+3)//4), 512),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(512,1)
        )
    def forward(self, x):
        out = self.CNN(x.view(x.size(0),1,x.size(1)))
        out = self.FC(out.view(out.size(0),-1))
        return out.view(-1)

class FCDiscriminator(nn.Module):
    def __init__(self, threshold, dim=128, drop=0.1):
        super().__init__()
        self.FC = nn.Sequential(
        nn.Linear(threshold,dim//2),
        nn.ReLU(),
        nn.Linear(dim//2,dim),
        nn.ReLU(),
        nn.Linear(dim,dim),
        nn.ReLU(),
        nn.Linear(dim,1)
        )
    def forward(self, x):
        out = self.FC(x.view(x.size(0),-1))
        return out.view(-1)


class CNNModelWide(nn.Module):
    def __init__(self, threshold, dim=128, drop=0.1):
        super().__init__()

        self.CNN = nn.Sequential(
            nn.Conv1d(1, 64, 11, 1, 5),
            nn.Dropout(drop),
            nn.ReLU(),
            ResBlock(64, 32),
            nn.Conv1d(64, 128, 5, 2, 2),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            ResBlock(128, 64),
            nn.Conv1d(128, 256, 5, 2, 2),
            nn.Dropout(drop),
            nn.ReLU(),
            ResBlock(256, 128),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
        )
        self.FC = nn.Sequential(
            nn.Linear(256*((threshold+3)//4), 1024),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(1024,2)
        )
    def forward(self, x):
        out = self.CNN(x.view(x.size(0),1,x.size(1)))
        out = self.FC(out.view(out.size(0),-1))
        return out

class CNNModelDeep(nn.Module):
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
            ResBlock(128, 64),
            nn.Conv1d(128, 256, 3, 1, 1),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
        )
        self.FC = nn.Sequential(
            nn.Linear(256*((threshold+3)//4), 512),
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
    def __init__(self, threshold, dim=256, drop=0.1):
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
    def __init__(self, window, scale=1, dim=128, drop=0.1, history=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(history, dim, 1, bias=True),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.resblock =  ResBlock(dim, dim//2)
        self.decoder = nn.Conv1d(dim, 1, 1, bias=True)
        self.scale = scale
    def forward(self, x, distill=False):
        signal = x[:,-1,:]
        noise = self.scale*torch.randn_like(signal) #gaussian
        encoded = self.encoder(x)
        res = self.resblock(encoded)
        out = self.decoder(res).view(res.size(0),-1)
        if distill:
            return torch.relu(out+noise), (encoded, res, out)
        else: 
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
        self.scale = nn.Parameter(torch.ones(1)*scale)
        self.noise = GaussianSinusoid(threshold)

    def forward(self, x, distill=False):
        signal = x[:,-1,:]
        #noise = torch.ones_like(signal)*self.scale #offset
        noise = self.scale*torch.randn_like(signal) #gaussian
        
        #noise = torch.relu(noise - signal) #Maya-like
        #x[:,-1,:] += noise
        #xx = x + shifter(noise)
        #xx = x
        encoded = self.encoder(x.permute(0,2,1)) #N,C,S -> N,S,C
        
        res, _ = self.resblock(encoded)
        out = encoded + res #N,S,C
        out = self.decoder(out).view(out.size(0),-1)
        #out = out + self.scale*torch.randn_like(out)
        #out = out + noise
        
        if distill:
            return torch.relu(out+noise), (encoded, res, out)
        else: 
            return torch.relu(out+noise)

class RNNGenerator2(nn.Module):
    def __init__(self, threshold, scale=1, dim=128, window=32, drop=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window, dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        self.resblock = nn.GRU(dim,dim, num_layers=1, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(dim, 1)
        )
        self.scale = nn.Parameter(torch.ones(1)*scale)
        self.noise = GaussianSinusoid(threshold)

    def forward(self, x, distill=False):
        signal = x[:,-1,:]
        noise = self.scale*torch.randn_like(signal) #gaussian
        
        encoded = self.encoder(x.permute(0,2,1)) #N,C,S -> N,S,C
        
        res, _ = self.resblock(encoded)
        out = encoded + res #N,S,C
        out = self.decoder(out).view(out.size(0),-1)
        
        if distill:
            return torch.relu(out+noise), (encoded, res, out)
        else: 
            return torch.relu(out+noise)

class MLPGen(nn.Module):
    def __init__(self, threshold, scale=1, dim=128, window=32, drop=0.2, depth=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window, dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        self.resblock = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        if depth == 1:
            self.resblock = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(drop)
            )

        self.decoder = nn.Sequential(
            nn.Linear(dim, 1)
        )
        self.scale = nn.Parameter(torch.ones(1)*scale)
        self.noise = GaussianSinusoid(threshold)

    def forward(self, x, distill=False):
        signal = x[:,-1,:]
        #noise = torch.ones_like(signal)*self.scale #offset
        noise = self.scale*torch.randn_like(signal) #gaussian
        
        #noise = torch.relu(noise - signal) #Maya-like
        #x[:,-1,:] += noise
        #xx = x + shifter(noise)
        #xx = x
        encoded = self.encoder(x.permute(0,2,1)) #N,C,S -> N,S,C
        
        res = self.resblock(encoded)
        out = encoded + res #N,S,C
        out = self.decoder(out).view(out.size(0),-1)
        #out = out + self.scale*torch.randn_like(out)
        #out = out + noise
        
        if distill:
            return torch.relu(out+noise), (encoded, res, out)
        else: 
            return torch.relu(out+noise)

class QGRU(nn.Module):
    def __init__(self, threshold, scale=1, dim=128, window=32, drop=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window, dim),
            nn.Dropout(drop)
        )

        self.gru1 = nn.GRUCell(dim,dim)
        self.gru2 = nn.GRUCell(dim,dim)

        self.decoder = nn.Sequential(
            nn.Linear(dim, 1)
        )
        self.scale = nn.Parameter(torch.ones(1)*scale)


    def forward(self, x, distill=False):
        signal = x[:,-1,:]

        noise = self.scale*torch.randn_like(signal) #gaussian
        
        src = x.permute(2,0,1) # N,C,S -> S,N,C

        encoded = self.encoder(src)
        h1 = torch.zeros_like(encoded[0])
        h2 = torch.zeros_like(encoded[0])
        hiddens = []
        for item in encoded:
            h1 = self.gru1(item, h1)
            h2 = self.gru2(h1, h2)
            hiddens.append(h2)
        res = torch.stack(hiddens)
        out = encoded + res #S,N,C
        out = self.decoder(out).view(out.size(0),-1)
        out = out.permute(1,0)
        #out = out + self.scale*torch.randn_like(out)
        #out = out + noise
        
        if distill:
            return torch.relu(out+noise), (encoded.permute(1,0,2), res.permute(1,0,2), out)
        else: 
            return torch.relu(out+noise)
class QGRU2(nn.Module):
    def __init__(self, threshold, scale=1, dim=128, window=32, drop=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window, dim),
            nn.Dropout(drop)
        )

        self.gru1 = nn.GRUCell(dim,dim)

        self.decoder = nn.Sequential(
            nn.Linear(dim, 1)
        )
        self.scale = nn.Parameter(torch.ones(1)*scale)


    def forward(self, x, distill=False):
        signal = x[:,-1,:]

        noise = self.scale*torch.randn_like(signal) #gaussian
        
        src = x.permute(2,0,1) # N,C,S -> S,N,C

        encoded = self.encoder(src)
        h1 = torch.zeros_like(encoded[0])
        hiddens = []
        for item in encoded:
            h1 = self.gru1(item, h1)
            hiddens.append(h1)
        res = torch.stack(hiddens)
        out = encoded + res #S,N,C
        out = self.decoder(out).view(out.size(0),-1)
        out = out.permute(1,0)
        #out = out + self.scale*torch.randn_like(out)
        #out = out + noise
        
        if distill:
            return torch.relu(out+noise), (encoded.permute(1,0,2), res.permute(1,0,2), out)
        else: 
            return torch.relu(out+noise)


class Distiller(nn.Module):
    def __init__(self, threshold, tdim=256, sdim=32, lamb_d = 0.1, lamb_r = 0.1, window=32):
        super().__init__()
        self.map1 = nn.Linear(sdim, tdim)
        self.map2 = nn.Linear(sdim, tdim)
        self.criterion = nn.MSELoss()
        self.lamb_d = lamb_d
        self.lamb_r = lamb_r

    def forward(self, s_out, t_out):
        enc_s, res_s, out_s = s_out
        enc_s2 = self.map1(enc_s)
        res_s2 = self.map2(res_s)

        enc_t, res_t, out_t = t_out
        
        l_distill = self.criterion(enc_s2, enc_t.detach()) + self.criterion(res_s2, res_t.detach())
        l_recon = self.criterion(out_s, out_t.detach())
        return self.lamb_d*l_distill + self.lamb_r*l_recon

class CNNDistiller(nn.Module):
    def __init__(self, threshold, tdim=256, sdim=32, lamb_d = 0.1, lamb_r = 0.1, window=32):
        super().__init__()
        self.map1 = nn.Linear(sdim, tdim)
        self.map2 = nn.Linear(sdim, tdim)
        self.criterion = nn.MSELoss()
        self.lamb_d = lamb_d
        self.lamb_r = lamb_r

    def forward(self, s_out, t_out):
        enc_s, res_s, out_s = s_out
        enc_s2 = self.map1(enc_s)
        res_s2 = self.map2(res_s)

        enc_t, res_t, out_t = t_out
        
        l_distill = self.criterion(enc_s2, enc_t.detach()) + self.criterion(res_s2, res_t.detach())
        l_recon = self.criterion(out_s, out_t.detach())
        return self.lamb_d*l_distill + self.lamb_r*l_recon

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

class RBFLinear(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048):
        super().__init__()
        self.centers = nn.Parameter(torch.Tensor(hidden_dim, in_dim))
        self.gammas = nn.Parameter(torch.Tensor(hidden_dim))
        nn.init.normal_(self.centers, 0, 1)
        nn.init.constant_(self.gammas, 1.0)
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        dist = (x[:,None,:] - self.centers).norm(dim=-1) * self.gammas
        kerns = torch.exp(-dist**2)
        return self.fc(kerns)

class FakeSVM(nn.Module):
    def __init__(self, in_dim, clf, gamma = 0.01):
        super().__init__()
        #self.gamma = 1/(in_dim)
        self.gamma = gamma
        centers = torch.from_numpy(clf.support_vectors_)
        self.centers = nn.Parameter(centers)
        self.fc = nn.Linear(centers.size(0), 1)
        self.fc.weight.data = torch.from_numpy(clf.dual_coef_)
        self.fc.bias.data = torch.from_numpy(clf.intercept_)
    def forward(self, x):
        dist = (x[:,None,:] - self.centers).norm(dim=-1)
        kerns = torch.exp(-self.gamma * dist**2)
        return torch.sigmoid(self.fc(kerns)).view(-1)

class SVMDiscriminator(nn.Module):
    def __init__(self, in_dim, clf, gamma = 0.01):
        super().__init__()
        #self.gamma = 1/(in_dim)
        self.gamma = gamma
        centers = torch.from_numpy(clf.support_vectors_)
        self.centers = nn.Parameter(centers)
        self.fc = nn.Linear(centers.size(0), 1)
        self.fc.weight.data = torch.from_numpy(clf.dual_coef_)
        self.fc.bias.data = torch.from_numpy(clf.intercept_)
    def forward(self, x):
        dist = (x[:,None,:] - self.centers).norm(dim=-1)
        kerns = torch.exp(-self.gamma * dist**2)
        return self.fc(kerns).view(-1)
    def clip(self, low=-0.01, high=0.01):
        self.fc.weight.data.clamp_(low, high)