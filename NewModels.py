import os
import argparse
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

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

class ConvAttClassifier(nn.Module):
    def __init__(self, dim=128, num_classes=2, drop=0.1):
        super().__init__()
        self.dim = dim
        self.CNN = nn.Sequential(
            nn.Conv1d(1, self.dim//4, 11, 1, 5),
            nn.Dropout(drop),
            nn.ReLU(),
            ResBlock(self.dim//4, self.dim//8),
            nn.Conv1d(self.dim//4, self.dim//2, 5, 2, 2),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            ResBlock(self.dim//2, self.dim//4),
            nn.Conv1d(self.dim//2, self.dim, 5, 2, 2),
            nn.Dropout(drop),
            nn.ReLU(),
        )
        self.tf = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(dim,nhead=2,dim_feedforward=256),
                                        num_layers=2,norm=nn.LayerNorm(dim))
        self.cls_key = nn.Parameter(torch.FloatTensor(1, dim))
        self.attn = nn.MultiheadAttention(dim, 1)
        self.fc = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.cls_key)

    def forward(self, input):
        #N,S input
        
        memory = self.CNN(input[:,None,:]).permute(2,0,1) #N,C,S -> S,N,C
        memory = self.tf(memory) + memory
        
        hidden = self.attn(self.cls_key.expand(1,memory.shape[1],memory.shape[2]),
            torch.tanh(memory), memory)[0]
        return self.fc(hidden.view(-1,self.dim)) #1,N,C->N,C

class ConvClassifier(nn.Module):
    def __init__(self, dim=128, num_classes=2, drop=0.1):
        super().__init__()
        self.dim = dim
        self.CNN = nn.Sequential(
            nn.Conv1d(1, self.dim//4, 11, 1, 5),
            nn.Dropout(drop),
            nn.ReLU(),
            ResBlock(self.dim//4, self.dim//8),
            nn.Conv1d(self.dim//4, self.dim//2, 5, 2, 2),
            nn.Dropout(drop),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            ResBlock(self.dim//2, self.dim//4),
            nn.Conv1d(self.dim//2, self.dim, 5, 2, 2),
            nn.Dropout(drop),
            nn.ReLU(),
        )
        self.tf = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(dim,nhead=2,dim_feedforward=256),
                                        num_layers=2,norm=nn.LayerNorm(dim))
        self.cls_key = nn.Parameter(torch.FloatTensor(1, dim))
        self.attn = nn.MultiheadAttention(dim, 1)
        self.fc = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.cls_key)

    def forward(self, input):
        #N,S input
        
        memory = self.CNN(input[:,None,:]).permute(2,0,1) #N,C,S -> S,N,C
        
        hidden = self.attn(self.cls_key.expand(1,memory.shape[1],memory.shape[2]),
            torch.tanh(memory), memory)[0]
        return self.fc(hidden.view(-1,self.dim)) #1,N,C->N,C
    
class FCDiscriminator(nn.Module):
    def __init__(self, tracelen, dim=128, drop=0.1):
        super().__init__()
        self.FC = nn.Sequential(
        nn.Linear(tracelen,dim//2),
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
    

class AttnShaper(nn.Module):
    def __init__(self, history, window, amp=2.0, dim=32, n_patterns=16):
        super().__init__()
        self.history=history
        self.window=window
        self.n_patterns=n_patterns
        self.dim = dim
        self.n_patterns = n_patterns
        self.amp = amp
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,dim,history, stride=window),
            nn.ReLU(),
        )
        self.keys = nn.Parameter(torch.FloatTensor(dim, n_patterns))
        torch.nn.init.xavier_uniform_(self.keys)
        
        self.shapes = nn.Parameter(torch.rand(n_patterns,window)*amp*2)
        self.noiselevel = nn.Parameter(torch.ones(1))
    def forward(self, x, avg_scores=None, mode='train'):
        
        padded = F.pad(x,(self.history-1, 0))
        noise = torch.randn(padded.shape).to(padded.device)*0.1
        #padded = padded+noise
        if avg_scores is None:
            avg_scores = torch.randn(x.shape[0], self.n_patterns).to(x.device)
            avg_scores = avg_scores - avg_scores.mean(dim=-1,keepdim=True)
        out = self.conv1(padded[:,None,:]).permute(2,0,1) #N,C,S -> S,N,C
        attn_scores = F.relu6(torch.matmul(out, self.keys))
        probs = []
        for score in attn_scores:
            if mode == 'inference':
                prob = F.one_hot(torch.argmax(score-avg_scores, dim=-1), 
                                 num_classes=self.n_patterns)
            else:
                prob = torch.softmax(score-avg_scores, dim=-1)
            avg_scores = avg_scores + prob - 1/self.n_patterns
            probs.append(prob)
        attn_probs = torch.stack(probs,dim=1) #N,S,C
       
        signal = torch.matmul(attn_probs, self.shapes).view(x.shape[0],-1)[:,:x.shape[1]]

        return torch.relu(signal-x)
    

class GaussianShaper(nn.Module):
    def __init__(self, history, window, amp=2.0, dim=32, n_patterns=16):
        super().__init__()
        self.history=history
        self.window=window
        self.n_patterns=n_patterns
        self.dim = dim
        self.n_patterns = n_patterns
        self.amp = amp
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,dim,history, stride=window),
            nn.ReLU(),
        )
        self.keys = nn.Parameter(torch.FloatTensor(dim, n_patterns))
        torch.nn.init.xavier_uniform_(self.keys)
        
        self.shapes = nn.Parameter(torch.rand(n_patterns,2)*amp*2) #offset, amp
        
    def forward(self, x, avg_scores=None, mode='train'):
        
        padded = F.pad(x,(self.history-1, 0))

        #padded = padded+noise
        if avg_scores is None:
            avg_scores = torch.randn(x.shape[0], self.n_patterns).to(x.device)
            avg_scores = avg_scores - avg_scores.mean(dim=-1,keepdim=True)
        out = self.conv1(padded[:,None,:]).permute(2,0,1) #N,C,S -> S,N,C
        attn_scores = F.relu6(torch.matmul(out, self.keys))
        probs = []
        for score in attn_scores:
            if mode == 'inference':
                prob = F.one_hot(torch.argmax(score-avg_scores, dim=-1), 
                                 num_classes=self.n_patterns)
            else:
                prob = torch.softmax(score-avg_scores, dim=-1)
            avg_scores = avg_scores + prob - 1/self.n_patterns
            probs.append(prob)
        attn_probs = torch.stack(probs,dim=1) #N,S,C
       
        shapeparams = torch.matmul(attn_probs, self.shapes) #N,S,2
        offset = shapeparams[:,:,0:1].expand(shapeparams.shape[0],shapeparams.shape[1],self.window)
        noise = torch.randn_like(offset) * shapeparams[:,:,1:2]
        signal = (offset + noise).view(x.shape[0],-1)[:,:x.shape[1]]
        return torch.relu(signal-x)


class QuantizedShaper(nn.Module):
    def __init__(self, shaper):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.history=shaper.history
        self.n_patterns=shaper.n_patterns
        self.dim = shaper.dim
        self.window = shaper.window
        self.amp = shaper.amp
        self.conv1 = deepcopy(shaper.conv1)
        self.keys = nn.Linear(self.dim,self.n_patterns,bias=False)
        self.keys.weight.data = shaper.keys.data.permute(1,0)
        self.shapes = nn.Linear(self.n_patterns, self.window, bias=False)
        self.shapes.weight.data = shaper.shapes.data.permute(1,0)
        self.relu6 = nn.ReLU6()
        self.relu = nn.ReLU()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.QF = nn.quantized.FloatFunctional()
    def forward(self, x, avg_scores=None):
        avg_scores = torch.randn(x.shape[0], self.n_patterns).to(x.device)
        avg_scores = avg_scores - avg_scores.mean(dim=-1,keepdim=True)
        #avg_scores = self.quant(avg_scores)
        padded = F.pad(x,(self.history-1, 0))
        #noise = torch.randn(padded.shape)*0.1
        #padded = self.quant(padded + noise)   
        padded = self.quant(padded)    
        out = self.conv1(padded[:,None,:]).permute(2,0,1) #N,C,S -> S,N,C
        attn_scores = self.dequant(self.relu6(self.keys(out)))
        probs = []
        for score in attn_scores:
            prob = F.one_hot(torch.argmax(score-avg_scores, dim=-1), 
                                num_classes=self.n_patterns).float()
            avg_scores = avg_scores + prob - 1/self.n_patterns
            probs.append(prob)
        attn_probs = self.quant(torch.stack(probs,dim=1)) #N,S,C
       
        signal = self.dequant(self.shapes(attn_probs).view(x.shape[0],-1)[:,:x.shape[1]])

        return torch.relu(signal-x)

class FCDiscriminator(nn.Module):
    def __init__(self, window, dim=128, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(window,dim)
        #self.fc2 = nn.Linear(dim,dim)
        self.fc3 = nn.Linear(dim,1)
        self.FC = nn.Sequential(
        self.fc1,
        nn.ReLU(),
        nn.Dropout(drop),
        #self.fc2,
        #nn.ReLU(),
        #nn.Dropout(drop),
        self.fc3,
        )
    def forward(self, x):
        out = self.FC(x.view(x.size(0),-1))
        return out.view(-1)
    def clip(self, low=-0.01, high=0.01):
        self.fc1.weight.data.clamp_(low, high)
        #self.fc2.weight.data.clamp_(low, high)
        self.fc3.weight.data.clamp_(low, high)


    
class GaussianGenerator(nn.Module):
    def __init__(self, amp):
        super().__init__()
        self.amp = amp
    def forward(self, x):
        
        offset = torch.rand([x.size(0),1],device=x.device).expand_as(x) * self.amp
        signal = torch.randn_like(x)*(self.amp-offset) + offset

        return torch.relu(signal-x)

class GaussianSinusoid(nn.Module):
    def __init__(self, amp):
        super().__init__()
        self.amp = amp
    def forward(self, x):
        offset = torch.rand([x.size(0),1],device=x.device).expand_as(x) * self.amp
        freq = torch.rand([x.size(0),1],device=x.device).expand_as(x)
        
        omega = np.pi*freq
        theta = torch.rand([x.size(0),1],device=x.device)*(2*np.pi)
        t = torch.arange(x.size(1),device=x.device)[None,:] + theta
        sinu = (self.amp-offset)*torch.sin((omega*t))
        #perturb = torch.randn([x.size(0), self.threshold], device=x.device) #[N, S]
        signal = sinu + offset

        return torch.relu(signal-x)
    
class OffsetGenerator(nn.Module):
    def __init__(self, amp):
        super().__init__()
        self.amp = amp
    def forward(self,x):
        #assuming N,C,S
        signal = torch.ones_like(x)*self.amp

        return torch.relu(signal-x)
    
class CNNModel(nn.Module):
    def __init__(self, tracelen, dim=128, drop=0.1):
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
        with torch.no_grad():
            testinput = torch.rand([1,1,tracelen])
            testoutput = self.CNN(testinput)
            fcdim = testoutput.shape[2]*256
        self.FC = nn.Sequential(
            nn.Linear(fcdim, 512),
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