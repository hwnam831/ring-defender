import RingDataset
from Models import CNNModel, CNNGenerator, RNNModel, GaussianGenerator, GaussianSinusoid, MLPGen, RNNGenerator2, RNNGenerator, MLP, OffsetGenerator
import os
import argparse
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import re
import time
from sklearn import svm
 

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
            "--testnet",
            type=str,
            choices=['cnn', 'rnn', 'ff'],
            default='cnn',
            help='Test Classifier choices')
    parser.add_argument(
            "--gen",
            type=str,
            choices=['gau', 'sin', 'adv', 'off', 'cnn', 'rnn', 'mlp'],
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
            "--file_prefix",
            type=str,
            default='core4ToSlice3',
            help='traininig dataset pkl file to load')
    parser.add_argument(
            "--batch_size",
            type=int,
            default='256',
            help='batch size')
    parser.add_argument(
            "--dim",
            type=int,
            default='128',
            help='internal channel dimension')
    parser.add_argument(
            "--lr",
            type=float,
            default=5e-5,
            help='Default learning rate')
    parser.add_argument(
            "--amp",
            type=float,
            default='3',
            help='noise amp scale')
    parser.add_argument(
            "--fresh",
            action='store_true',
            help='Fresh start without loading')

    return parser.parse_args()

def shifter(arr, window=window):
    dup = arr[:,None,:].expand(arr.size(0), arr.size(1)+1, arr.size(1))
    dup2 = dup.reshape(arr.size(0), arr.size(1), arr.size(1)+1)
    shifted = dup2[:,:window,:-window]
    return shifted

def quantizer(arr, std=8):
    return torch.round(arr*std)/std

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args = get_args()
    dataset = RingDataset.RingDataset(args.file_prefix+'_train.pkl', threshold=args.threshold)
    testset =  RingDataset.RingDataset(args.file_prefix+'_test.pkl', threshold=args.threshold)
    valset = RingDataset.RingDataset(args.file_prefix+'_valid.pkl', threshold=args.threshold)

    trainset=dataset
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    
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
    elif args.gen == 'cnn':
        gen=CNNGenerator(args.threshold, scale=0.5).cuda()
    elif args.gen == 'adv':
        gen=RNNGenerator2(args.threshold, scale=0.25, dim=args.dim).cuda()
        if os.path.isfile('./models/best_{}_{}.pth'.format(args.gen, args.dim)) and not args.fresh:
            print('Previous best found: loading the model...')
            gen.load_state_dict(torch.load('./models/best_{}_{}.pth'.format(args.gen, args.dim)))
    elif args.gen == 'rnn':
        gen=RNNGenerator(args.threshold, scale=0.25, dim=args.dim).cuda()
        if os.path.isfile('./models/best_{}_{}.pth'.format(args.gen, args.dim)) and not args.fresh:
            print('Previous best found: loading the model...')
            gen.load_state_dict(torch.load('./models/best_{}_{}.pth'.format(args.gen, args.dim)))
    elif args.gen == 'mlp':
        gen=MLPGen(args.threshold, scale=0.25, dim=args.dim).cuda()
        if os.path.isfile('./models/best_{}_{}.pth'.format(args.gen, args.dim)) and not args.fresh:
            print('Previous best found: loading the model...')
            gen.load_state_dict(torch.load('./models/best_{}_{}.pth'.format(args.gen, args.dim)))
    elif args.gen == 'off':
        gen=OffsetGenerator(args.threshold, scale=args.amp/2).cuda()
    else:
        print(args.gen + ' not supported\n')
        exit(-1)
    
    if args.net == 'ff':
        classifier = MLP(args.threshold, dim=args.dim).cuda()
    elif args.net == 'rnn':
        classifier = RNNModel(args.threshold, dim=args.dim).cuda()
    else:
        classifier = CNNModel(args.threshold, dim=args.dim).cuda()

    if args.testnet == 'ff':
        classifier_test = MLP(args.threshold, dim=args.dim).cuda()
    elif args.testnet == 'rnn':
        classifier_test = RNNModel(args.threshold, dim=args.dim).cuda()
    else:
        classifier_test = CNNModel(args.threshold, dim=args.dim).cuda()


    optim_c = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    optim_c2 = torch.optim.Adam(classifier_test.parameters(), lr=args.lr)
    optim_g = torch.optim.Adam(gen.parameters(), lr=args.lr*2)

    gamma = 0.98
    sched_c   = torch.optim.lr_scheduler.StepLR(optim_c, 1, gamma=gamma)
    sched_c2   = torch.optim.lr_scheduler.StepLR(optim_c2, 1, gamma=gamma)
    sched_g   = torch.optim.lr_scheduler.StepLR(optim_g, 1, gamma=gamma)

    criterion = nn.CrossEntropyLoss()
    C = args.amp
    warmup = 70
    cooldown = 100
    #scale = 0.002
    scale = 0.02
    for e in range(warmup):
        classifier.train()
        totcorrect = 0
        totcount = 0
        zerocorrect = 0
        zerocount = 0
        onecorrect = 0
        onecount = 0
        for x,y in trainloader:
            xdata, ydata = x.cuda(), y.cuda()
            shifted = shifter(xdata)
            #train classifier
            optim_c.zero_grad()
            perturb = gen(shifted).view(shifted.size(0),-1)
            output = classifier(xdata[:,31:]+perturb.detach())
            loss_c = criterion(output, ydata)
            loss_c.backward()
            optim_c.step()
            pred = output.argmax(axis=-1)
            totcorrect += (pred==ydata).sum().item()
            totcount += y.size(0)
            zerocorrect += ((pred==0)*(ydata==0)).sum().item()
            zerocount += (ydata==0).sum().item()
            onecorrect += ((pred==1)*(ydata==1)).sum().item()
            onecount += (ydata==1).sum().item()
        macc = float(totcorrect)/totcount
        zacc = float(zerocorrect)/zerocount
        oacc = float(onecorrect)/onecount
        if (e+1)%10 == 0:
            print("Warmup epoch {} \t acc {:.4f}".format(e+1, macc))
            print("zacc {:.6f}\t oneacc {:.6f}\n".format(zacc, oacc))

    for e in range(args.epochs):
        gen.train()
        classifier.train()
        trainstart = time.time()
        for x,y in trainloader:
            xdata, ydata = x.cuda(), y.cuda()
            shifted = shifter(xdata)
            #train classifier
            optim_c.zero_grad()
            perturb = gen(shifted).view(shifted.size(0),-1)
            #perturb = quantizer(perturb)
            #perturb = gen(xdata[:,31:])
            #interleaving?
            p_input = xdata[:,31:]+perturb.detach()
            output = classifier(p_input)
            loss_c = criterion(output, ydata)
            loss_c.backward()
            optim_c.step()

            #train generator
            optim_g.zero_grad()
            #perturb = gen(shifted).view(shifted.size(0),-1)
            #perturb = gen(xdata[:,31:])
            p_input = xdata[:,31:]+perturb
            output = classifier(p_input)
            #perturb = quantizer(perturb)
            #pnorm = torch.norm(perturb, dim=-1) - C
            #loss_p = torch.mean(torch.relu(pnorm))
            hinge = perturb.mean(dim=-1) - C
            hinge[hinge<0] = 0.0
            loss_p = torch.mean(hinge)

            fake_target = 1-ydata

            loss_adv1 = criterion(output, fake_target)
            #loss = loss_adv1 + scale*loss_p + 0.1*loss_d
            loss = loss_adv1 + scale*loss_p
            loss.backward()
            optim_g.step()
        classifier_test.train()
        gen.eval()
        for x,y in valloader:
            xdata, ydata = x.cuda(), y.cuda()
            shifted = shifter(xdata)
            #train classifier
            optim_c2.zero_grad()
            perturb = gen(shifted).view(shifted.size(0),-1)
            perturb = quantizer(perturb)
            #perturb = gen(xdata[:,31:])
            #interleaving?
            output = classifier_test(xdata[:,31:]+perturb.detach())
            loss_c = criterion(output, ydata)
            loss_c.backward()
            optim_c2.step()
        sched_c.step()

        sched_c2.step()
        sched_g.step()

        mloss = 0.0
        mloss2 = 0.0
        totcorrect = 0
        totcount = 0
        mnorm = 0.0
        zerocorrect = 0
        zerocount = 0
        onecorrect = 0
        onecount = 0
        #evaluate classifier
        with torch.no_grad():
            classifier_test.eval()
            for x,y in testloader:
                xdata, ydata = x.cuda(), y.cuda()
                shifted = shifter(xdata)
                perturb = gen(shifted).view(shifted.size(0),-1)
                perturb = quantizer(perturb)
                #perturb = gen(xdata[:,31:])
                norm = torch.mean(perturb)
                #output = classifier_test(xdata[:,31:]+perturb)
                output = classifier(xdata[:,31:]+perturb.detach())
                loss_c = criterion(output, ydata)
                pnorm = torch.norm(perturb, dim=-1) - C
                loss_p = torch.mean(torch.relu(pnorm))
                pred = output.argmax(axis=-1)
                mnorm += norm.item()/len(testloader)
                mloss += loss_c.item()/len(testloader)
                mloss2 += scale*loss_p.item()/len(testloader)
                #macc += ((pred==ydata).sum().float()/pred.nelement()).item()/len(testloader)
                totcorrect += (pred==ydata).sum().item()
                totcount += y.size(0)
                zerocorrect += ((pred==0)*(ydata==0)).sum().item()
                zerocount += (ydata==0).sum().item()
                onecorrect += ((pred==1)*(ydata==1)).sum().item()
                onecount += (ydata==1).sum().item()
            macc = float(totcorrect)/totcount
            zacc = float(zerocorrect)/zerocount
            oacc = float(onecorrect)/onecount
            #print("epoch {} \t acc {:.4f}\t loss {:.4f}\t loss_p {:.4f}\t Avg perturb {:.4f}\t duration {:.4f}".format(e+1, macc, mloss, mloss2, mnorm, time.time()-trainstart))
            #print("zacc {:.6f}\t oneacc {:.6f}\n".format(zacc, oacc))
            print("epoch {} \t acc {:.4f}\t zacc {:.4f}\t oneacc {:.4f}\t Avg perturb {:.4f}".format(e+1, macc, zacc, oacc, mnorm))
            if e > (args.epochs*4)//5 and macc - 0.5 < 0.001:
                break
    gen.eval()
    lastacc = 0.0
    lastnorm = 0.0
    optim_c2 = torch.optim.Adam(classifier_test.parameters(), lr=args.lr)
    sched_c2   = torch.optim.lr_scheduler.StepLR(optim_c2, 1, gamma=gamma)
    for e in range(cooldown):
        classifier_test.train()
        for x,y in valloader:
            xdata, ydata = x.cuda(), y.cuda()
            shifted = shifter(xdata)
            #train classifier
            optim_c2.zero_grad()
            perturb = gen(shifted).view(shifted.size(0),-1)
            #perturb = gen(xdata[:,31:])
            #interleaving?
            output = classifier_test(xdata[:,31:]+perturb.detach())
            loss_c = criterion(output, ydata)
            loss_c.backward()
            optim_c2.step()


        mloss = 0.0
        totcorrect = 0
        totcount = 0
        mnorm = 0.0
        zerocorrect = 0
        zerocount = 0
        onecorrect = 0
        onecount = 0
        #evaluate classifier
        with torch.no_grad():
            classifier_test.eval()
            for x,y in testloader:
                xdata, ydata = x.cuda(), y.cuda()
                shifted = shifter(xdata)
                perturb = gen(shifted).view(shifted.size(0),-1)
                perturb = quantizer(perturb)
                #perturb = gen(xdata[:,31:])
                norm = torch.mean(perturb)
                output = classifier_test(xdata[:,31:]+perturb)
                loss_c = criterion(output, ydata)
                pred = output.argmax(axis=-1)
                mnorm += norm.item()/len(testloader)
                mloss += loss_c.item()/len(testloader)
                #macc += ((pred==ydata).sum().float()/pred.nelement()).item()/len(testloader)
                totcorrect += (pred==ydata).sum().item()
                totcount += y.size(0)
                zerocorrect += ((pred==0)*(ydata==0)).sum().item()
                zerocount += (ydata==0).sum().item()
                onecorrect += ((pred==1)*(ydata==1)).sum().item()
                onecount += (ydata==1).sum().item()
            macc = float(totcorrect)/totcount
            zacc = float(zerocorrect)/zerocount
            oacc = float(onecorrect)/onecount
            print("epoch {} \t zacc {:.4f}\t oneacc {:.4f}\t acc {:.4f}\t Avg perturb {:.4f}\n".format(e+1, zacc, oacc, macc, mnorm))
            if cooldown - e <= 10:
                lastacc += macc/10
                lastnorm += mnorm/10
    print("Last 10 acc: {:.6f}\t perturb: {:.6f}".format(lastacc,lastnorm))

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    with torch.no_grad():
        for x,y in valloader:
            xdata= x.cuda()
            shifted = shifter(xdata)
            #train classifier
            perturb = gen(shifted).view(shifted.size(0),-1)
            perturb = quantizer(perturb)
            perturbed_x = xdata[:,31:]+perturb
            for p in perturbed_x:
                train_x.append(p.cpu().numpy())
            for y_i in y:
                train_y.append(y_i.item())
        for x,y in testloader:
            xdata= x.cuda()
            shifted = shifter(xdata)
            #train classifier
            perturb = gen(shifted).view(shifted.size(0),-1)
            perturb = quantizer(perturb)
            perturbed_x = xdata[:,31:]+perturb
            for p in perturbed_x:
                test_x.append(p.cpu().numpy())
            for y_i in y:
                test_y.append(y_i.item())
    clf = svm.SVC(gamma=0.02)
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)

    svmacc = (pred_y == test_y).sum()/len(pred_y)
    print("SVM acc: {:.6f}".format(svmacc))
    lastacc = max(lastacc, svmacc)

    if args.gen == 'adv' or args.gen == 'rnn':
        filename = "{}_{}_{:.3f}_{:.3f}.pth".format(args.gen,args.dim,lastnorm, lastacc)
        flist = os.listdir('models')
        best = 1.0
        smallest = 10.0
        rp = re.compile(r"{}_{}_(\d\.\d+)_(\d\.\d+)\.pth".format(args.gen,args.dim))
        for fn in flist:
            m = rp.match(fn)
            if m:
                facc = float(m.group(2))
                fperturb = float(m.group(1))
                if facc <= best:
                    best = facc
                if fperturb <= smallest:
                    smallest = fperturb
        if lastacc <= best:
            print('New best found')
            torch.save(gen.state_dict(), './models/'+filename)
            torch.save(gen.state_dict(), './models/'+'best_{}_{}.pth'.format(args.gen, args.dim))
        elif lastnorm <= smallest:
            torch.save(gen.state_dict(), './models/'+filename)