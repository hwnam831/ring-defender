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
from copy import deepcopy
import math

def Warmup(args, trainloader, classifier, discriminator, shaper):
    optim_c = torch.optim.Adam(classifier.parameters(), lr=args.lr*5, weight_decay=args.lr/2)
    optim_d = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr*5, weight_decay=args.lr/2)
    criterion = nn.CrossEntropyLoss()
    shaper.train()
    for e in range(args.warmup):
        classifier.train()
        discriminator.train()
        for x,y in trainloader:
            xdata, ydata = x.to(device), y.to(device)
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio)
            #train classifier
            optim_c.zero_grad()
            optim_d.zero_grad()
            perturb = shaper(xdata).detach()
            output = classifier(xdata + perturb)
            fakes = discriminator(xdata + perturb)

            
            loss_c = criterion(output, ydata)
            loss_d = torch.mean(fakes*disc_label)
            loss_c.backward()
            
            loss_d.backward()

            optim_c.step()
            optim_d.step()
            discriminator.clip()

            pred = output.argmax(axis=-1)

    

        if (e+1)%10 == 0:
            totcorrect = 0
            totcount = 0
            mloss = 0.0
            closs = 0.0
            mperturb = 0.0
            with torch.no_grad():
                classifier.eval()
                discriminator.eval()
                for x,y in valloader:
                    xdata, ydata = x.to(device), y.to(device)
                    oneratio = ydata.sum().item()/len(ydata)
                    disc_label = 2*(ydata.float()-oneratio)
                    #train classifier
                    optim_c.zero_grad()

                    perturb = shaper(xdata)
                    mperturb += perturb.mean().item()
                    output = classifier(xdata + perturb)
                    closs += criterion(output, ydata)
                    fakes = discriminator(xdata + perturb)
                    mloss += torch.mean(fakes*disc_label).item()

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
            
def Train_DefenderGAN(args, trainloader,valloader, classifier, discriminator, shaper):
    optim_c = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.lr/10)
    optim_d = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr, weight_decay=args.lr/10)
    optim_g = torch.optim.Adam(shaper.parameters(), lr=10*args.lr, weight_decay=args.lr)
    criterion = nn.CrossEntropyLoss()
    shaper.train()
    bestnorm = args.amp * 2
    bestacc = 0.99
    avgnorm = args.amp
    avgacc = 0.99
    bestparams = deepcopy(shaper.state_dict())
    for e in range(args.epochs):
        classifier.train()
        discriminator.train()
        #train both
        for x,y in trainloader:
            xdata, ydata = x.to(device), y.to(device)
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio)
            #train classifier
            optim_c.zero_grad()
            optim_d.zero_grad()
            perturb = shaper(xdata).detach()
            output = classifier(xdata + perturb)
            fakes = discriminator(xdata + perturb)

            
            loss_c = criterion(output, ydata)
            loss_d = torch.mean(fakes*disc_label)
            loss_c.backward()
            
            loss_d.backward()

            optim_c.step()
            optim_d.step()
            discriminator.clip()

            pred = output.argmax(axis=-1)

            optim_g.zero_grad()
            perturb = shaper(xdata)
            hinge = perturb.mean(dim=-1) - args.amp
            hinge[hinge<0] = 0.0
            output = classifier(xdata + perturb)
            fakes = discriminator(xdata + perturb)
            loss_p = torch.mean(hinge)
            loss_d = -torch.mean(fakes*disc_label)
            fake_target = 1-ydata

            loss_adv1 = criterion(output, fake_target)
            loss = loss_adv1 + 0.02*loss_p + 10*loss_d
            loss.backward()
            optim_g.step()

        #train again
        for x,y in trainloader:
            xdata, ydata = x.to(device), y.to(device)
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio)
            #train classifier
            optim_c.zero_grad()
            optim_d.zero_grad()
            perturb = shaper(xdata).detach()
            output = classifier(xdata + perturb)
            fakes = discriminator(xdata + perturb)

            
            loss_c = criterion(output, ydata)
            loss_d = torch.mean(fakes*disc_label)
            loss_c.backward()
            
            loss_d.backward()

            optim_c.step()
            optim_d.step()
            discriminator.clip()

            pred = output.argmax(axis=-1)
        

    

        #validation
        totcorrect = 0
        totcount = 0
        mloss = 0.0
        closs = 0.0
        mperturb = 0.0
        with torch.no_grad():
            classifier.eval()
            discriminator.eval()
            for x,y in valloader:
                xdata, ydata = x.to(device), y.to(device)
                oneratio = ydata.sum().item()/len(ydata)
                disc_label = 2*(ydata.float()-oneratio)
                #train classifier
                optim_c.zero_grad()

                perturb = shaper(xdata)
                mperturb += perturb.mean().item()
                output = classifier(xdata + perturb)
                closs += criterion(output, ydata)
                fakes = discriminator(xdata + perturb)
                mloss += torch.mean(fakes*disc_label).item()

                optim_c.step()

                pred = output.argmax(axis=-1)
                totcorrect += (pred==ydata).sum().item()
                totcount += y.size(0)
        mloss = mloss / len(valloader)
        closs = closs / len(valloader)
        mperturb = mperturb/len(valloader)
        macc = float(totcorrect)/totcount
        avgnorm = avgnorm*0.9 + mperturb*0.1
        avgacc = avgacc*0.9 + macc*0.1
        print("Epoch {} \t acc {:.4f}\t dloss {:.4f}\t  closs {:.4f}\t mperturb: {:.4f}".format(
            e+1, macc, mloss, closs, mperturb))
        
        if abs(bestacc-0.5) + 0.01*bestnorm > abs(avgacc-0.5) + 0.01*avgnorm and e > args.epochs//2:
            bestacc = avgacc
            bestnorm = avgnorm
            bestparams = deepcopy(shaper.state_dict())
    shaper.load_state_dict(bestparams)
    return bestacc, bestnorm

if __name__ == '__main__':
    #torch.backends.cuda.matmul.allow_tf32 = False
    #torch.backends.cudnn.allow_tf32 = False
    args = Util.get_args()
    file_prefix=args.victim
    device = 'cuda:0'

    trainset = LOTRDataset(file_prefix+'_train.pkl')
    testset =  LOTRDataset(file_prefix+'_test.pkl', med=trainset.med)
    valset = LOTRDataset(file_prefix+'_valid.pkl', med=trainset.med)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    classifier = NewModels.ConvAttClassifier().to(device)
    discriminator = NewModels.FCDiscriminator(window=trainset.tracelen).to(device)
    shaper = NewModels.AttnShaper(amp=args.amp, history=16, window=8).to(device)
    Warmup(args,trainloader,classifier, discriminator,shaper)
    bestacc, bestnorm = Train_DefenderGAN(args, trainloader,valloader, classifier, discriminator, shaper)
    filename = "{}_{}_{:.3f}_{:.3f}.pth".format(args.victim,args.dim,bestnorm, bestacc)
    torch.save(shaper.state_dict(), './gans/'+filename)


    
    
    