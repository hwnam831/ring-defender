import os
import argparse
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F

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
        self.n_patterns=n_patterns
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,dim,history, stride=window),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.keys = nn.Parameter(torch.FloatTensor(dim, n_patterns))
        torch.nn.init.xavier_uniform_(self.keys)
        
        self.shapes = nn.Parameter(torch.rand(n_patterns,window)*amp*2)
        self.noiselevel = nn.Parameter(torch.ones(1))
    def forward(self, x, avg_scores=None):
        
        padded = F.pad(x,(self.history-1, 0))
        if avg_scores is None:
            avg_scores = torch.rand(x.shape[0], self.n_patterns).to(x.device)
            avg_scores = avg_scores - avg_scores.mean(dim=-1,keepdim=True)
        out = self.conv1(padded[:,None,:]).permute(2,0,1) #N,C,S -> S,N,C
        attn_scores = F.relu6(torch.matmul(out, self.keys))
        probs = []
        for score in attn_scores:
            prob = torch.softmax(score-avg_scores, dim=-1)
            avg_scores = avg_scores + prob - 1/self.n_patterns
            probs.append(prob)
        attn_probs = torch.stack(probs,dim=1) #N,S,C
       
        signal = torch.matmul(attn_probs, self.shapes).view(x.shape[0],-1)[:,:x.shape[1]]

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