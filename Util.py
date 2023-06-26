import argparse
import torch
import torch.nn as nn
from RingDataset import RingDataset, EDDSADataset
import Models
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import re
import time
from sklearn import svm
import copy

def get_parser():
    """Get all the args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--net",
            type=str,
            choices=['cnn', 'rnn', 'ff'],
            default='cnn',
            help='Classifier choices')
    parser.add_argument(
            "--testnet",
            type=str,
            choices=['cnn', 'rnn', 'ff'],
            default='cnn',
            help='Test Classifier choices')
    parser.add_argument(
            "--victim",
            type=str,
            choices=['rsa', 'eddsa', 'both', 'rsa_noise'],
            default='rsa',
            help='Victim dataset choices')
    parser.add_argument(
            "--gen",
            type=str,
            choices=['gau', 'sin', 'adv', 'off', 'cnn', 'rnn', 'mlp', 'rnn3'],
            default='adv',
            help='Generator choices')
    parser.add_argument(
            "--window",
            type=int,
            default='42',
            help='number of samples window')
    parser.add_argument(
            "--history",
            type=int,
            default='8',
            help='number of samples window')
    parser.add_argument(
            "--epochs",
            type=int,
            default='200',
            help='number of epochs')
    parser.add_argument(
            "--warmup",
            type=int,
            default='100',
            help='number of warmup epochs')
    parser.add_argument(
            "--cooldown",
            type=int,
            default='100',
            help='number of cooldown epochs')
    parser.add_argument(
            "--batch_size",
            type=int,
            default='200',
            help='batch size')
    parser.add_argument(
            "--dim",
            type=int,
            default='160',
            help='internal channel dimension')
    parser.add_argument(
            "--lr",
            type=float,
            default=2e-5,
            help='Default learning rate')
    parser.add_argument(
            "--student",
            type=int,
            default=16,
            help='Student dim')
    parser.add_argument(
            "--amp",
            type=float,
            default='3.0',
            help='noise amp scale')
    parser.add_argument(
            "--gamma",
            type=float,
            default='0.97',
            help='decay scale for optimizer')
    parser.add_argument(
            "--lambda_h",
            type=float,
            default='0.01',
            help='lambda coef for hinge loss')
    parser.add_argument(
            "--lambda_d",
            type=float,
            default='5.0',
            help='lambda coef for discriminator loss')   
    parser.add_argument(
            "--lambda_r",
            type=float,
            default='0.0005',
            help='lambda coef for reconstruction loss')     
    parser.add_argument(
            "--fresh",
            action='store_true',
            help='Fresh start without loading')

    return parser

def get_args():
    return get_parser().parse_args()

def shifter(arr, history=32):
    dup = arr[:,None,:].expand(arr.size(0), arr.size(1)+1, arr.size(1))
    dup2 = dup.reshape(arr.size(0), arr.size(1), arr.size(1)+1)
    shifted = dup2[:,:history,:-history]
    return shifted

def quantizer(arr, std=16):
    return torch.round(arr*std)/std

class Env(object):
    def __init__(self, args):
        self.both = False
        if args.victim == 'both': # 'both'
                self.both=True
                file_prefix='rsa'
                trainset = EDDSADataset(file_prefix+'_train.pkl')
                testset =  EDDSADataset(file_prefix+'_test.pkl', std=trainset.std, window=trainset.window)
                valset = EDDSADataset(file_prefix+'_valid.pkl', std=trainset.std, window=trainset.window)
                self.window = trainset.window
                file_prefix='eddsa'
                trainset2 = EDDSADataset(file_prefix+'_train.pkl')
                testset2 =  EDDSADataset(file_prefix+'_test.pkl', std=trainset.std)
                valset2 = EDDSADataset(file_prefix+'_valid.pkl', std=trainset.std)
                self.window = trainset.window
                
        elif args.victim == 'rsa_noise':
                file_prefix=args.victim
                trainset = EDDSADataset(file_prefix+'_train.pkl')
                testset =  EDDSADataset(file_prefix+'_test.pkl', std=trainset.std, window=trainset.window)
                valset = EDDSADataset(file_prefix+'_valid.pkl', std=trainset.std, window=trainset.window)
                self.window = trainset.window
        elif args.victim == 'rsa':
                file_prefix='rsa'
                trainset = EDDSADataset(file_prefix+'_train.pkl')
                testset =  EDDSADataset(file_prefix+'_test.pkl', std=trainset.std, window=trainset.window)
                valset = EDDSADataset(file_prefix+'_valid.pkl', std=trainset.std, window=trainset.window)
                self.window = trainset.window
        elif args.victim == 'eddsa':
                file_prefix='eddsa'
                trainset = EDDSADataset(file_prefix+'_train.pkl')
                testset =  EDDSADataset(file_prefix+'_test.pkl', std=trainset.std)
                valset = EDDSADataset(file_prefix+'_valid.pkl', std=trainset.std)
                self.window = trainset.window
        #print("std: " + str(testset.std))
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)
        self.valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=4, shuffle=True)
        if self.both:
                self.trainloader2 = DataLoader(trainset2, batch_size=args.batch_size, num_workers=4, shuffle=True)
                self.testloader2 = DataLoader(testset2, batch_size=args.batch_size, num_workers=4)
                self.valloader2 = DataLoader(valset2, batch_size=args.batch_size, num_workers=4, shuffle=True)
        #TODO: no more threshold
        if args.gen == 'gau':
                self.gen = nn.Sequential(
                Models.GaussianGenerator(self.window, scale=args.amp),
                nn.ReLU(),
                ).cuda()
        elif args.gen == 'sin':
                self.gen = nn.Sequential(
                Models.GaussianSinusoid(self.window, scale=args.amp),
                nn.ReLU(),
                ).cuda()
        elif args.gen == 'cnn':
                self.gen=Models.CNNGenerator(self.window, scale=0.25, dim=args.dim).cuda()
        elif args.gen == 'adv':
                self.gen=Models.RNNGenerator2(self.window, scale=0.25, dim=args.dim, window=args.history).cuda()
        elif args.gen == 'rnn3':
                self.gen=Models.RNNGenerator3(self.window, scale=0.25, dim=args.dim, window=args.history).cuda()
        elif args.gen == 'rnn':
                self.gen=Models.RNNGenerator(self.window, scale=0.25, dim=args.dim).cuda()
        elif args.gen == 'mlp':
                self.gen=Models.MLPGen(self.window, scale=0.25, dim=args.dim).cuda()
        elif args.gen == 'off':
                self.gen=Models.OffsetGenerator(self.window, scale=args.amp).cuda()
        else:
                print(args.gen + ' not supported\n')
                exit(-1)
        self.fresh = True
        if args.gen in ['cnn', 'adv', 'rnn', 'mlp', 'rnn3'] and \
                os.path.isfile('./gans/best_{}_{}_{}.pth'.format(args.victim,args.gen, args.dim)) and not args.fresh:
                print('Previous best found: loading the model...')
                self.gen.load_state_dict(torch.load('./gans/best_{}_{}_{}.pth'.format(args.victim,args.gen, args.dim)))
                self.fresh = False
        
        self.gen2 = copy.deepcopy(self.gen)
        
        if args.net == 'ff':
                self.classifier = Models.MLP(self.window, dim=args.dim).cuda()
        elif args.net == 'rnn':
                self.classifier = Models.RNNModel(self.window, dim=args.dim).cuda()
        else:
                self.classifier = Models.CNNModel(self.window, dim=args.dim).cuda()

        if args.testnet == 'ff':
                self.classifier_test = Models.MLP(self.window, dim=args.dim).cuda()
        elif args.testnet == 'rnn':
                self.classifier_test = Models.RNNModel(self.window, dim=args.dim).cuda()
        else:
                self.classifier_test = Models.CNNModel(self.window, dim=args.dim).cuda()
        
        if self.both:
                self.classifier2 = type(self.classifier)(self.window2, dim=args.dim).cuda()
                self.classifier_test2 = type(self.classifier_test)(self.window2, dim=args.dim).cuda()


        train_x = []
        train_y = []
        with torch.no_grad():
            for x,y in self.trainloader:
                xdata= x.cuda()
                shifted = shifter(xdata, args.history)
                #train classifier
                perturb = self.gen(shifted).view(shifted.size(0),-1)
                perturbed_x = xdata[:,args.history-1:]+perturb
                for p in perturbed_x:
                        train_x.append(p.cpu().numpy())
                for y_i in y:
                        train_y.append(y_i.item())
        clf = svm.SVC(gamma=0.01)
        clf.fit(train_x, train_y)
        
        self.disc = Models.SVMDiscriminator(self.window, clf, 0.01).cuda() #discriminator
        if self.both:
            train_x = []
            train_y = []
            with torch.no_grad():
                for x,y in self.trainloader2:
                        xdata= x.cuda()
                        shifted = shifter(xdata, args.history)
                        #train classifier
                        perturb = self.gen(shifted).view(shifted.size(0),-1)
                        perturbed_x = xdata[:,args.history-1:]+perturb
                        for p in perturbed_x:
                                train_x.append(p.cpu().numpy())
                        for y_i in y:
                                train_y.append(y_i.item())
            clf = svm.SVC(gamma=0.01)
            clf.fit(train_x, train_y)
                
            self.disc2 = Models.SVMDiscriminator(self.window2, clf, 0.01).cuda() #discriminator

#No doubleblind
def cooldown(args, env, gen, prevgen, epochs=None):
    halfgen = gen.half()
    halfgen.train()
    lastacc = 0.0
    lastnorm = 0.0
    lastacc2 = 0.0
    lastnorm2 = 0.0
    optim_c_t = torch.optim.Adam(env.classifier_test.parameters(), lr=args.lr)
    optim_g_t = torch.optim.Adam(prevgen.parameters(), lr=args.lr)
    if env.both:
        optim_c_t2 = torch.optim.Adam(env.classifier_test2.parameters(), lr=args.lr)
    criterion=nn.CrossEntropyLoss()
    
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    with torch.no_grad():
        for x,y in env.valloader:
            xdata= x.cuda()
            shifted = shifter(xdata, args.history)
            #train classifier
            perturb = halfgen(shifted.half()).float().view(shifted.size(0),-1)
            perturb = quantizer(perturb)
            perturbed_x = xdata[:,args.history-1:]+perturb
            for p in perturbed_x:
                train_x.append(p.cpu().numpy())
            for y_i in y:
                train_y.append(y_i.item())
        for x,y in env.testloader:
            xdata= x.cuda()
            shifted = shifter(xdata, args.history)
            #train classifier
            perturb = halfgen(shifted.half()).view(shifted.size(0),-1)
            perturb = quantizer(perturb)
            perturbed_x = xdata[:,args.history-1:]+perturb.float()
            for p in perturbed_x:
                test_x.append(p.cpu().numpy())
            for y_i in y:
                test_y.append(y_i.item())
    clf = svm.SVC(gamma='auto')
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)

    svmacc = (pred_y == test_y).sum()/len(pred_y)
    print("[{}]\tSVM acc: {:.6f}".format(args.victim,svmacc))
    if env.both:
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        with torch.no_grad():
            for x,y in env.valloader2:
                xdata= x.cuda()
                shifted = shifter(xdata, args.history)
                #train classifier
                perturb = halfgen(shifted.half()).float().view(shifted.size(0),-1)
                perturb = quantizer(perturb)
                perturbed_x = xdata[:,args.history-1:]+perturb
                for p in perturbed_x:
                        train_x.append(p.cpu().numpy())
                for y_i in y:
                        train_y.append(y_i.item())
            for x,y in env.testloader2:
                xdata= x.cuda()
                shifted = shifter(xdata, args.history)
                #train classifier
                perturb = halfgen(shifted.half()).view(shifted.size(0),-1)
                perturb = quantizer(perturb)
                perturbed_x = xdata[:,args.history-1:]+perturb.float()
                for p in perturbed_x:
                        test_x.append(p.cpu().numpy())
                for y_i in y:
                        test_y.append(y_i.item())
        clf = svm.SVC(gamma='auto')
        clf.fit(train_x, train_y)
        pred_y = clf.predict(test_x)

        svmacc2 = (pred_y == test_y).sum()/len(pred_y)
        print("[eddsa]\tSVM acc: {:.6f}".format(svmacc2))

    cd = epochs if epochs else args.cooldown
    for e in range(cd):
        env.classifier_test.train()
        for x,y in env.valloader:
            xdata, ydata = x.cuda(), y.cuda()
            shifted = shifter(xdata, args.history)
            #train classifier
            optim_c_t.zero_grad()
            optim_g_t.zero_grad()
            perturb = prevgen(shifted).view(shifted.size(0),-1)
            perturb2 = halfgen(shifted.half()).float().detach()
            #interleaving?
            output = env.classifier_test(xdata[:,args.history-1:]+perturb2)
            loss_c = criterion(output, ydata)
            loss_c.backward()
            optim_c_t.step()
            loss_d = nn.functional.mse_loss(perturb, perturb2)
            loss_d.backward()
            optim_g_t.step()

        mloss = 0.0
        totcorrect = 0
        totcount = 0
        mnorm = 0.0
        totcorrect2 = 0
        totcount2 = 0
        onecorrect = 0
        onecount = 0
        #evaluate classifier
        with torch.no_grad():
            env.classifier_test.eval()
            for x,y in env.testloader:
                xdata, ydata = x.cuda(), y.cuda()
                shifted = shifter(xdata, args.history)
                perturb = halfgen(shifted.half()).view(shifted.size(0),-1)
                perturb = quantizer(perturb)
                norm = torch.mean(perturb)
                output = env.classifier_test(xdata[:,args.history-1:]+perturb.float())
                loss_c = criterion(output, ydata)
                pred = output.argmax(axis=-1)
                mnorm += norm.item()/len(env.testloader)
                mloss += loss_c.item()/len(env.testloader)
                totcorrect += (pred==ydata).sum().item()
                totcount += y.size(0)
                totcorrect2 += ((pred==0)*(ydata==0)).sum().item()
                totcount2 += (ydata==0).sum().item()
                onecorrect += ((pred==1)*(ydata==1)).sum().item()
                onecount += (ydata==1).sum().item()
            macc = float(totcorrect)/totcount
            zacc = float(totcorrect2)/totcount2
            oacc = float(onecorrect)/onecount
            if (e+1)%10 == 0:
                print("[{}]\tepoch {} \t zacc {:.4f}\t oneacc {:.4f}\t acc {:.4f}\t Avg perturb {:.4f}\n".format(
                        args.victim,e+1, zacc, oacc, macc, mnorm))
            if cd - e <= 10:
                lastacc += macc/10
                lastnorm += mnorm/10
        if env.both:
            env.classifier_test2.train()
            for x,y in env.valloader2:
                xdata, ydata = x.cuda(), y.cuda()
                shifted = shifter(xdata, args.history)
                #train classifier
                optim_c_t2.zero_grad()
                optim_g_t.zero_grad()
                perturb = prevgen(shifted).view(shifted.size(0),-1)
                perturb2 = halfgen(shifted.half()).float().detach()
                #interleaving?
                output = env.classifier_test2(xdata[:,args.history-1:]+perturb2)
                loss_c = criterion(output, ydata)
                loss_c.backward()
                optim_c_t2.step()
                loss_d = nn.functional.mse_loss(perturb, perturb2)
                loss_d.backward()
                optim_g_t.step()

            
                #evaluate classifier
            with torch.no_grad():
                mloss = 0.0
                totcorrect = 0
                totcount = 0
                mnorm = 0.0
                totcorrect2 = 0
                totcount2 = 0
                onecorrect = 0
                onecount = 0
                env.classifier_test2.eval()
                for x,y in env.testloader2:
                        xdata, ydata = x.cuda(), y.cuda()
                        shifted = shifter(xdata, args.history)
                        perturb = halfgen(shifted.half()).view(shifted.size(0),-1)
                        perturb = quantizer(perturb)
                        norm = torch.mean(perturb)
                        output = env.classifier_test2(xdata[:,args.history-1:]+perturb.float())
                        loss_c = criterion(output, ydata)
                        pred = output.argmax(axis=-1)
                        mnorm += norm.item()/len(env.testloader2)
                        mloss += loss_c.item()/len(env.testloader2)
                        totcorrect += (pred==ydata).sum().item()
                        totcount += y.size(0)
                        totcorrect2 += ((pred==0)*(ydata==0)).sum().item()
                        totcount2 += (ydata==0).sum().item()
                        onecorrect += ((pred==1)*(ydata==1)).sum().item()
                        onecount += (ydata==1).sum().item()
                macc = float(totcorrect)/totcount
                zacc = float(totcorrect2)/totcount2
                oacc = float(onecorrect)/onecount
                if (e+1)%10 == 0:
                    print("[eddsa]\tepoch {} \t zacc {:.4f}\t oneacc {:.4f}\t acc {:.4f}\t Avg perturb {:.4f}\n".format(
                    e+1, zacc, oacc, macc, mnorm))
                if cd - e <= 10:
                        lastacc2 += macc/10
                        lastnorm2 += mnorm/10  
    print("[{}]\tLast 10 acc: {:.6f}\t perturb: {:.6f}".format(args.victim, lastacc,lastnorm))
    if env.both:
        print("[eddsa]\tLast 10 acc: {:.6f}\t perturb: {:.6f}".format(lastacc2,lastnorm2)) 
    
        return max((lastacc+lastacc2)/2, (svmacc+svmacc2)/2), (lastnorm+lastnorm2)/2
    return max(lastacc, svmacc), lastnorm