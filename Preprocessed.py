import PCIEDataset
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torch.nn.functional as F
import time

if __name__ == '__main__':

    raw_dataset = PCIEDataset.PCIEDataset('train', mode='preprocess')
    print("Dataset loading done")
    classifier = PCIEDataset.PreprocessClassifier(32, 128, 3).cuda()
    gen = PCIEDataset.PreprocessCNN(32, 128, 6).cuda()

    trainset = []
    testset = []
    for i in range(len(raw_dataset)):
        if i%7 == 0:
            testset.append(i)
        else:
            trainset.append(i)
    trainloader = DataLoader(raw_dataset, batch_size=16, num_workers=4, sampler=
                            torch.utils.data.SubsetRandomSampler(trainset))
    valloader = DataLoader(raw_dataset, batch_size=16, num_workers=4, sampler=
                            torch.utils.data.SubsetRandomSampler(testset))

    criterion = nn.CrossEntropyLoss()
    PCIEDataset.Warmup(classifier, gen, trainloader, valloader, 10)
    C=10.0 # hyperparameter to choose
    scale = 0.001
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
            cost = 0.01*perturb[:,:,0] + perturb[:,:,1] + 20*perturb[:,:,2]
            hinge = cost.mean() - C
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
                cost = 0.01*perturb[:,:,0] + perturb[:,:,1] + 20*perturb[:,:,2]
                mperturb += cost.mean().item() / len(valloader)
                loss = criterion(out,ydata)
                mloss += loss.item()/len(valloader)
                pred = out.argmax(axis=-1)
                acc = (ydata==pred).sum().item()/len(ydata)
                macc += acc/len(valloader)
        
        print('Test loss : {}\nTest acc: {}\n'.format(mloss, macc))
        print('Test mean perturb : {:.5f}'.format(mperturb))
    torch.save(gen.state_dict(), 'pcie/pregen_{}_{}_{}.pth'.format(gen.window,gen.modelsize,gen.num_layers))
    test_dataset = PCIEDataset('./nvmessd', mode='preprocess')
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
    clf_test = PCIEDataset.PreprocessClassifier(32, 128, 3).cuda()
    PCIEDataset.Cooldown(clf_test, gen, trainloader, valloader, epochs=30)