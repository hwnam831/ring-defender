from RingDataset import LOTRDataset
import Models
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import re
import time
import NewModels
import Util

def Warmup(args, trainloader, classifier, shaper):
    optim_c = torch.optim.Adam(classifier.parameters(), lr=args.lr*5, weight_decay=args.lr/2)
    criterion = nn.CrossEntropyLoss()
    shaper.train()
    for e in range(args.warmup):
        classifier.train()
        
        for x,y in trainloader:
            xdata, ydata = x.to(device), y.to(device)
            #train classifier
            optim_c.zero_grad()
            perturb = shaper(xdata)
            output = classifier(xdata + perturb)

            loss_c = criterion(output, ydata)
            loss_c.backward()

            optim_c.step()

            pred = output.argmax(axis=-1)

    

        if (e+1)%10 == 0:
            totcorrect = 0
            totcount = 0
            mloss = 0.0
            mperturb = 0.0
            with torch.no_grad():
                classifier.eval()
                for x,y in valloader:
                    xdata, ydata = x.to(device), y.to(device)
                    #train classifier
                    optim_c.zero_grad()

                    perturb = shaper(xdata)
                    mperturb += perturb.mean().item()
                    output = classifier(xdata + perturb)

                    optim_c.step()

                    pred = output.argmax(axis=-1)
                    totcorrect += (pred==ydata).sum().item()
                    totcount += y.size(0)
            mloss = mloss / len(valloader)
            mperturb = mperturb/len(valloader)
            macc = float(totcorrect)/totcount
            print("Warmup epoch {} \t acc {:.4f}\t dloss {}\t mperturb: {:.4f}".format(e+1, macc, mloss, mperturb))

if __name__ == '__main__':
    #torch.backends.cuda.matmul.allow_tf32 = False
    #torch.backends.cudnn.allow_tf32 = False
    args = Util.get_args()
    file_prefix='eddsa'
    device = 'cuda:0'

    trainset = LOTRDataset(file_prefix+'_train.pkl')
    testset =  LOTRDataset(file_prefix+'_test.pkl', med=trainset.med)
    valset = LOTRDataset(file_prefix+'_valid.pkl', med=trainset.med)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    classifier = NewModels.ConvAttClassifier().to(device)
    shaper = NewModels.AttnShaper(history=16, window=8).to(device)
    Warmup(args,trainloader,classifier, shaper)
    
    
    