import RingDataset
from Models import CNNModel, GaussianGenerator, GaussianSinusoid, RNNGenerator, MLP
import os
import argparse
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn as nn


 

window=32 #this is fixed


def get_args():
    """Get all the args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--net",
            type=str,
            choices=['cnn', 'rnn', 'ff'],
            default='cnn',
            help='Classifier choices')
    parser.add_argument(
            "--gen",
            type=str,
            choices=['gau', 'sin', 'adv'],
            default='adv',
            help='Generator choices')
    parser.add_argument(
            "--threshold",
            type=int,
            default='42',
            help='number of samples threshold')
    parser.add_argument(
            "--epochs",
            type=int,
            default='100',
            help='number of epochs')
    parser.add_argument(
            "--train_file",
            type=str,
            default='core4ToSlice3.pkl',
            help='traininig dataset pkl file to load')
    parser.add_argument(
            "--test_file",
            type=str,
            default='core4ToSlice3_test.pkl',
            help='test dataset pkl file to load')
    parser.add_argument(
            "--batch_size",
            type=int,
            default='128',
            help='batch size')
    parser.add_argument(
            "--dim",
            type=int,
            default='128',
            help='internal channel dimension')
    parser.add_argument(
            "--lr",
            type=float,
            default=2e-5,
            help='Default learning rate')
    parser.add_argument(
            "--amp",
            type=float,
            default='10',
            help='noise amp scale')

    return parser.parse_args()

def shifter(arr, window=window):
    dup = arr[:,None,:].expand(arr.size(0), arr.size(1)+1, arr.size(1))
    dup2 = dup.reshape(arr.size(0), arr.size(1), arr.size(1)+1)
    shifted = dup2[:,:window,:-window]
    return shifted

if __name__ == '__main__':
    args = get_args()
    dataset = RingDataset.RingDataset(args.train_file, threshold=args.threshold)
    testset =  RingDataset.RingDataset(args.test_file, threshold=args.threshold)
    #testlen = dataset.__len__()//4
    #trainlen = dataset.__len__() - testlen
    #testset, trainset = random_split(dataset, [testlen, trainlen], generator=torch.Generator().manual_seed(17))
    #testset, trainset = random_split(dataset, [testlen, trainlen])
    trainset=dataset
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)
    
    if args.gen == 'gau':
        gen = nn.Sequential(
        GaussianGenerator(args.threshold, scale=args.amp),
        nn.ReLU(),
        ).cuda()
    elif args.gen == 'sin':
        gen = nn.Sequential(
        GaussianSinusoid(args.threshold, scale=args.amp),
        nn.ReLU(),
        ).cuda()
    elif args.gen == 'adv':
        gen=RNNGenerator(args.threshold).cuda()
    else:
        print(args.gen + ' not supported\n')
        exit(-1)
    
    '''
    for x,y in trainloader:
        xp = gen(x.cuda())
        print(x.shape)
        print(xp.shape)
        break
    '''
    if args.net == 'ff':
        classifier = MLP(args.threshold, args.dim).cuda()
    else:
        classifier = CNNModel(args.threshold).cuda()
    optim_c = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    optim_g = torch.optim.Adam(gen.parameters(), lr=args.lr*2)

    criterion = nn.CrossEntropyLoss()
    C = args.amp * 4
    warmup = 10
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

    for e in range(args.epochs):
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
        totcorrect = 0
        totcount = 0
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
                #macc += ((pred==ydata).sum().float()/pred.nelement()).item()/len(testloader)
                totcorrect += (pred==ydata).sum().item()
                totcount += y.size(0)
            macc = float(totcorrect)/totcount
            print("epoch {} \t acc {:.6f}\t loss {:.6f}\t Avg perturb {:.6f}\n".format(e+1, macc, mloss, mnorm))


