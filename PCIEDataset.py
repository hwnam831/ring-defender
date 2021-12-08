import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torch.nn.functional as F
import time
import lz4.frame

class PCIEDataset(Dataset):
    def __init__(self, rawdir, mode='raw', window=1024, median=8192, std=1024):
        assert mode in ['raw', 'preprocess', 'both']
        self.filelist = os.listdir(rawdir)
        self.rootdir = rawdir
        self.med = median
        self.std = std
        self.window = window
        self.threshold = 3.0
        self.mode = mode
        if mode == 'preprocess' or mode == 'both':
            self.preprocessed_x = []
            self.labels = []
            with lz4.frame.open(self.rootdir +'.lz4', 'rb') as f:
                self.preprocessed_x , self.labels = pickle.load(f)

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, idx):
        if self.mode == 'raw' or self.mode == 'both':
            with open(self.rootdir +'/'+ self.filelist[idx], 'rb') as f:
                xarr, label = pickle.load(f)
        if self.mode == 'preprocess' or self.mode == 'both':
            preprocessed_x = self.preprocessed_x[idx]
            label = self.labels[idx]
        if self.mode == 'raw':
            x = xarr
        elif self.mode == 'preprocess':
            x = preprocessed_x
        else:
            x = (xarr, preprocessed_x)
        return x, label


class AvgClassifier(nn.Module):
    def __init__(self, windowsize, modelsize, num_layers, classes=50, drop=0.1):
        super().__init__()
        self.window = windowsize
        self.modelsize = modelsize

        self.pool = nn.AvgPool1d(128,128,ceil_mode=True)
        self.cls_key = nn.Parameter(torch.FloatTensor(1, self.modelsize))
        self.blstm = nn.LSTM(1, modelsize//2, 2, bidirectional=True, dropout=0.1)
        self.attn = nn.MultiheadAttention(modelsize, 1)
        self.fc = nn.Linear(modelsize, classes)
        torch.nn.init.xavier_uniform_(self.cls_key)


    def forward(self, raw_x):
        mywindow = 50000
        padsize = (mywindow - raw_x.shape[-1]%mywindow)
        x = F.pad(raw_x, (0,padsize)).reshape(raw_x.shape[0], -1, mywindow)
        x,_ = x.max(dim=-1, keepdim=True)
        memory = self.blstm(x.permute(1,0,2))[0]
        hidden = self.attn(self.cls_key.expand(1,memory.shape[1],memory.shape[2]),
            torch.tanh(memory), memory)[0]
        return self.fc(hidden.view(-1,self.modelsize))

class RawClassifier(nn.Module):
    def __init__(self, windowsize, modelsize, num_layers, classes=10, drop=0.2):
        super().__init__()
        self.window = windowsize
        self.modelsize = modelsize
        #Automatically disables the first 
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, modelsize, windowsize, stride=windowsize),
            nn.BatchNorm1d(modelsize),
            nn.ReLU())
        self.convblocks = nn.ModuleList([
            #must be pytorch 1.10
            nn.Sequential(
                nn.Conv1d(modelsize, modelsize, 8, 4, padding=2),
                nn.Dropout(drop),
                nn.BatchNorm1d(modelsize),
                nn.ReLU())
            for i in range(1,num_layers+1)
        ])
        self.finalsize = modelsize
        self.resblocks = nn.ModuleList([
            #must be pytorch 1.10
            nn.Sequential(
                nn.Conv1d(self.finalsize, self.finalsize, 2, 1, padding='same', dilation=2**i),
                nn.Dropout(drop),
                nn.BatchNorm1d(self.finalsize),
                nn.ReLU())
            for i in range(1,num_layers+1)
        ])
        self.pool = nn.AvgPool1d(128,128,ceil_mode=True)
        self.cls_key = nn.Parameter(torch.FloatTensor(1, self.finalsize))
        self.blstm = nn.LSTM(self.finalsize, self.finalsize//2, 2, bidirectional=True, dropout=drop)
        self.attn = nn.MultiheadAttention(self.finalsize, 1)
        self.fc = nn.Linear(self.finalsize, classes)
        torch.nn.init.xavier_uniform_(self.cls_key)
        self.layernorm = nn.LayerNorm(self.finalsize)
    #Input: N, L
    def forward(self, raw_x):
        #stride = self.window//2
        stride = self.window
        padsize = (stride - (raw_x.shape[-1]-self.window+1)%stride)
        x = F.pad(raw_x, (0,padsize)).reshape(raw_x.shape[0], 1, -1)
        out = self.conv1(x)
        for layer in self.convblocks:
            out = layer(out)
        #for layer in self.resblocks:
        #    out = layer(out) + out
        memory = out.permute(2,0,1)
        memory = self.blstm(memory)[0] + memory
        memory = self.layernorm(memory)
        hidden = self.attn(self.cls_key.expand(1,memory.shape[1],memory.shape[2]),
            torch.tanh(memory), memory)[0]
        return self.fc(hidden.view(-1,self.finalsize))

class RawCNN(nn.Module):
    def __init__(self, windowsize, modelsize, num_layers, maxval=200.0, drop=0.1):
        super().__init__()
        self.window = windowsize
        self.modelsize = modelsize
        self.num_layers=num_layers
        #Automatically disables the first 
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, modelsize, windowsize, stride=windowsize),
            nn.BatchNorm1d(modelsize),
            nn.ReLU())
        self.resblocks = nn.ModuleList([
            #must be pytorch 1.10
            nn.Sequential(
                nn.Conv1d(modelsize, modelsize, 2, 1, padding='same', dilation=2**i),
                nn.Dropout(drop),
                nn.BatchNorm1d(modelsize),
                nn.ReLU())
            for i in range(1,num_layers)
        ])
        self.fc = nn.Sequential(nn.Conv1d(modelsize, windowsize, 2, 1, 
                                            padding='same', dilation=2**num_layers),
                                nn.Sigmoid())
        self.maxval = nn.Parameter(torch.ones(1, dtype=torch.float)*maxval)

    def forward(self, raw_x, distill=False):
        padsize = (self.window - raw_x.shape[-1]%self.window)
        x = F.pad(raw_x, (0,padsize)).reshape(raw_x.shape[0], 1, -1)
        
        out = self.conv1(x)
        if distill:
            intermediates = []
            intermediates.append(out)
        for layer in self.resblocks:
            out = layer(out) + out
            if distill:
                intermediates.append(out)
        perturb = self.fc(out).permute(0,2,1) * self.maxval
        perturb = perturb.reshape(perturb.shape[0],-1)[:,:raw_x.shape[1]-self.window]
        perturb = F.pad(perturb, (self.window, 0))
        perturb = torch.relu(perturb-raw_x)
        if distill:
            return perturb, intermediates
        else:
            return perturb
class PreprocessClassifier(nn.Module):
    def __init__(self, windowsize, modelsize, num_layers, classes=10, drop=0.2):
        super().__init__()
        self.window = windowsize
        self.modelsize = modelsize
        #Automatically disables the first 
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, modelsize, windowsize, stride=windowsize),
            nn.BatchNorm1d(modelsize),
            nn.ReLU())

        self.resblocks = nn.ModuleList([
            #must be pytorch 1.10
            nn.Sequential(
                nn.Conv1d(self.modelsize, self.modelsize, 4, 1, padding='same'),
                nn.Dropout(drop),
                nn.BatchNorm1d(self.modelsize),
                nn.ReLU())
            for i in range(1,num_layers+1)
        ])
        self.cls_key = nn.Parameter(torch.FloatTensor(1, self.modelsize))
        self.blstm = nn.LSTM(self.modelsize, self.modelsize//2, 2, bidirectional=True, dropout=drop)
        self.attn = nn.MultiheadAttention(self.modelsize, 1)
        self.fc = nn.Linear(self.modelsize, classes)
        torch.nn.init.xavier_uniform_(self.cls_key)
        self.layernorm = nn.LayerNorm(self.modelsize)
    #Input: N, S, C=3
    def forward(self, raw_x):
        x = raw_x.permute(0,2,1)
        padsize = (self.window - x.shape[-1]%self.window)
        x = F.pad(x, (0,padsize))
        out = self.conv1(x)
        for layer in self.resblocks:
            out = layer(out) + out
        memory = out.permute(2,0,1)
        memory = self.blstm(memory)[0] + memory
        memory = self.layernorm(memory)
        hidden = self.attn(self.cls_key.expand(1,memory.shape[1],memory.shape[2]),
            torch.tanh(memory), memory)[0]
        return self.fc(hidden.view(-1,self.modelsize))
class PreprocessCNN(nn.Module):
    def __init__(self, windowsize, modelsize, num_layers, maxval=200.0, drop=0.1):
        super().__init__()
        self.window = windowsize
        self.modelsize = modelsize
        self.num_layers=num_layers
        #Automatically disables the first 
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, modelsize, windowsize, stride=1, padding='same'),
            nn.BatchNorm1d(modelsize),
            nn.ReLU())
        self.resblocks = nn.ModuleList([
            #must be pytorch 1.10
            nn.Sequential(
                nn.Conv1d(modelsize, modelsize, 2, 1, padding='same', dilation=2**i),
                nn.Dropout(drop),
                nn.BatchNorm1d(modelsize),
                nn.ReLU())
            for i in range(1,num_layers)
        ])
        self.fc = nn.Sequential(nn.Conv1d(modelsize, 3, 2, 1),
                                nn.ReLU())

    def forward(self, raw_x):
        x = raw_x.permute(0,2,1)
        
        out = self.conv1(x)
        for layer in self.resblocks:
            out = layer(out) + out
        perturb = self.fc(out)
        perturb = F.pad(perturb, (1, 0)).permute(0,2,1)
        perturb = torch.relu(perturb-raw_x)
        return perturb

def Warmup(classifier, gen, trainloader, valloader, epochs=10):
    optim_c = torch.optim.Adam(classifier.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    for e in range(epochs):
        classifier.train()
        curtime = time.time()
        mloss = 0.0
        for x,y in trainloader:
            optim_c.zero_grad()
            xdata = x.cuda().float()
            ydata = y.cuda()
            perturb = gen(xdata)
            out = classifier(xdata+perturb.detach())
            loss = criterion(out,ydata)
            mloss += loss.item()/len(trainloader)
            nn.utils.clip_grad_norm_(classifier.parameters(), 0.1)
            loss.backward()
            optim_c.step()
        print('Warmup Epoch: {}'.format(e+1))
        print('Training time: {}'.format(time.time()-curtime))
        print('Training loss: {}'.format(mloss))
        classifier.eval()
        mloss = 0.0
        macc = 0.0
        for x,y in valloader:
            with torch.no_grad():
                xdata = x.cuda().float()
                ydata = y.cuda()
                out = classifier(xdata)
                loss = criterion(out,ydata)
                mloss += loss.item()/len(valloader)
                pred = out.argmax(axis=-1)
                acc = (ydata==pred).sum().item()/len(ydata)
                macc += acc/len(valloader)
        
        print('Test loss : {}\nTest acc: {}\n'.format(mloss, macc))
def Cooldown(clf_test, gen, trainloader, valloader, epochs=20):
    optim_c_t = torch.optim.Adam(clf_test.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    gen.eval()
    c_matrix = torch.zeros(10,10)
    totcount = 0
    for e in range(epochs):
        clf_test.train()
        curtime = time.time()
        mloss = 0.0
        for x,y in trainloader:
            optim_c_t.zero_grad()
            xdata = x.cuda().float()
            ydata = y.cuda()
            perturb = gen(xdata)
            out = clf_test(xdata+perturb.detach())
            loss = criterion(out,ydata)
            mloss += loss.item()/len(trainloader)
            nn.utils.clip_grad_norm_(clf_test.parameters(), 0.5)
            loss.backward()
            optim_c_t.step()
        print('cooldown Epoch: {}'.format(e+1))
        print('Training time: {}'.format(time.time()-curtime))
        print('Training loss: {}'.format(mloss))
        clf_test.eval()
        mloss = 0.0
        macc = 0.0
        mperturb = 0.0
        
        for x,y in valloader:
            with torch.no_grad():
                xdata = x.cuda().float()
                ydata = y.cuda()
                perturb = gen(xdata)
                mperturb += perturb.mean().item() / len(valloader)
                out = clf_test(xdata+perturb.detach())
                loss = criterion(out,ydata)
                mloss += loss.item()/len(valloader)
                pred = out.argmax(axis=-1)
                acc = (ydata==pred).sum().item()/len(ydata)
                macc += acc/len(valloader)
                if (epochs - e) < 20:
                    for y,yp in zip(ydata, pred):
                        c_matrix[y][yp] += 1
                    totcount += len(ydata)
        print('Test mean perturb : {:.5f}'.format(mperturb))
        print('Test loss : {}\nTest acc: {}\n'.format(mloss, macc))
    return c_matrix/totcount

if __name__ == '__main__':
    raw_dataset = PCIEDataset('./train')
    classifier = RawClassifier(512, 128, 4).cuda()
    gen = RawCNN(1024, 64, 6).cuda()
    if os.path.isfile('pcie/gen_{}_{}_{}.pth'.format(gen.window,gen.modelsize,gen.num_layers)):
        gen.load_state_dict(torch.load('pcie/gen_{}_{}_{}.pth'.format(gen.window,gen.modelsize,gen.num_layers)))
    trainset = []
    testset = []
    for i in range(len(raw_dataset)):
        if i%7 == 0:
            testset.append(i)
        else:
            trainset.append(i)
    trainloader = DataLoader(raw_dataset, batch_size=8, num_workers=4, sampler=
                            torch.utils.data.SubsetRandomSampler(trainset))
    valloader = DataLoader(raw_dataset, batch_size=8, num_workers=4, sampler=
                            torch.utils.data.SubsetRandomSampler(testset))
    
    criterion = nn.CrossEntropyLoss()
    Warmup(classifier, gen, trainloader, valloader, 10)
    C=1.5 # hyperparameter to choose
    scale = 0.1
    optim_c = torch.optim.Adam(classifier.parameters(), lr=1e-5)
    optim_g = torch.optim.Adam(gen.parameters(), lr=2e-5)
    for e in range(50):
        classifier.train()
        gen.train()
        curtime = time.time()
        mloss = 0.0
        mperturb = 0.0
        for x,y in trainloader:
            optim_c.zero_grad()
            
            xdata = x.cuda().float()
            ydata = y.cuda()
            
            perturb = gen(xdata)
            out = classifier(xdata+perturb.detach())
            loss = criterion(out,ydata)
            mloss += loss.item()/len(trainloader)
            nn.utils.clip_grad_norm_(classifier.parameters(), 0.1)
            loss.backward()
            optim_c.step()

            #Train generator
            optim_g.zero_grad()
            fake_labels = torch.zeros_like(y).cuda()
            perturb = gen(xdata)
            out = classifier(xdata+perturb)
            hinge = perturb.mean() - C
            hinge[hinge<0] = 0.0
            loss_p = torch.mean(hinge)
            loss_g = criterion(out,fake_labels)
            loss = loss_p*scale + loss_g
            loss.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), 0.1)
            optim_g.step()
        print('Epoch: {}'.format(e+1))
        print('Training time: {}'.format(time.time()-curtime))
        print('Training loss: {}'.format(mloss))
        classifier.eval()
        gen.eval()
        mloss = 0.0
        macc = 0.0
        for x,y in valloader:
            with torch.no_grad():
                xdata = x.cuda().float()
                ydata = y.cuda()
                perturb = gen(xdata).detach()
                out = classifier(xdata+perturb)
                mperturb += perturb.mean().item() / len(valloader)
                loss = criterion(out,ydata)
                mloss += loss.item()/len(valloader)
                pred = out.argmax(axis=-1)
                acc = (ydata==pred).sum().item()/len(ydata)
                macc += acc/len(valloader)
        
        print('Test loss : {}\nTest acc: {}\n'.format(mloss, macc))
        print('Test mean perturb : {:.5f}'.format(mperturb))
    torch.save(gen.state_dict(), 'pcie/gen_{}_{}_{}.pth'.format(gen.window,gen.modelsize,gen.num_layers))
    test_dataset = PCIEDataset('./nvmessd')
    trainset = []
    testset = []
    for i in range(len(test_dataset)):
        if i%7 == 0:
            testset.append(i)
        else:
            trainset.append(i)
    trainloader = DataLoader(test_dataset, batch_size=8, num_workers=4, sampler=
                            torch.utils.data.SubsetRandomSampler(trainset))
    valloader = DataLoader(test_dataset, batch_size=8, num_workers=4, sampler=
                            torch.utils.data.SubsetRandomSampler(testset))
    clf_test = RawClassifier(512,128,4).cuda()
    Cooldown(clf_test, gen, trainloader, valloader, epochs=20)