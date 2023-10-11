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
from sklearn import svm


def evaluate(args):
    file_prefix=args.victim

    trainset = LOTRDataset(file_prefix+'_train.pkl')
    testset =  LOTRDataset(file_prefix+'_test.pkl', med=trainset.med)
    valset = LOTRDataset(file_prefix+'_valid.pkl', med=trainset.med)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=8)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=8, shuffle=True)
    

    shaper = NewModels.AttnShaper(amp=args.amp, history=args.window*2, window=args.window, dim=args.dim, n_patterns=args.n_patterns).to(args.device)
    if args.gen == 'adv':
        shaper = NewModels.GaussianShaper(history=args.window*2, window=args.window, amp=args.amp, dim=args.dim, n_patterns=args.n_patterns)
    elif args.gen == 'shaper':
        shaper = NewModels.AttnShaper2(amp=args.amp, history=args.window*2, window=args.window, dim=args.dim, n_patterns=args.n_patterns).to(args.device)
    elif args.gen == 'rnn':
        shaper = NewModels.RNNShaper(amp=args.amp, history=args.window*2, window=args.window, dim=args.dim).to(args.device)
    flist = os.listdir('gans')
    reverse = {'rsa':'eddsa', 'eddsa':'rsa'}
    victim = args.victim
    if args.cross:
        victim = reverse[args.victim]
    fname = "{}_{}_{}x_{}_{}_best.pth".format(victim, args.gen, args.window,args.dim,args.n_patterns)
    print("loading " + fname)
    modeltoacc = {}

    shaper.load_state_dict(torch.load('./gans/'+fname, map_location=args.device))
    half=False
    if args.gen == 'adv':
        model_fp32 = NewModels.GaussianQuantizedShaper(shaper).to('cpu').eval()
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
        input_fp32 = torch.randn(args.batch_size, valset.tracelen)
        model_fp32_prepared(input_fp32)
        qshaper = torch.ao.quantization.convert(model_fp32_prepared)
    elif args.gen == 'shaper':
        qshaper=shaper.half()
        half=True
    elif args.gen == 'rnn':
        qshaper=shaper.half()
        half=True
    else:
        model_fp32 = NewModels.QuantizedShaper(shaper).to('cpu').eval()
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
        input_fp32 = torch.randn(args.batch_size, valset.tracelen)
        model_fp32_prepared(input_fp32)
        qshaper = torch.ao.quantization.convert(model_fp32_prepared)
    clfs = {
        'cnn' : NewModels.CNNModel(trainset.tracelen).to(args.device),
        'rnn' : NewModels.RNNClassifier().to(args.device),
        'att' : NewModels.ConvAttClassifier().to(args.device),
        'deep' : NewModels.CNNModelDeep(trainset.tracelen).to(args.device),
        'wide' : NewModels.CNNModelWide(trainset.tracelen).to(args.device)
    }
    svmacc = svm_cooldown(args, qshaper, valloader, testloader, half)
    modeltoacc['svm'] = svmacc
    for cname in clfs:
        classifier = clfs[cname]
        print('\nEvaluating ' + cname)
        print(classifier)

        bestacc, mperturb = cooldown(args, qshaper, classifier, valloader, testloader, half)
        modeltoacc[cname] = bestacc
    print(modeltoacc)

def svm_cooldown(args, qshaper, valloader, testloader, half=False):
    avgsvmacc = 0.0
    for i in range(10):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        with torch.no_grad():
            for x,y in valloader:
                if half:
                    perturb = qshaper(x.to(args.device).half()).float().cpu()
                else:
                    perturb = qshaper(x)
                perturbed_x = x + perturb
                for p in perturbed_x:
                    train_x.append(p.cpu().numpy())
                for y_i in y:
                    train_y.append(y_i.item())
            for x,y in testloader:
                if half:
                    perturb = qshaper(x.to(args.device).half()).float().cpu()
                else:
                    perturb = qshaper(x)
                perturbed_x = x + perturb
                for p in perturbed_x:
                    test_x.append(p.cpu().numpy())
                for y_i in y:
                    test_y.append(y_i.item())
        clf = svm.SVC(gamma='auto')
        clf.fit(train_x, train_y)
        pred_y = clf.predict(test_x)

        svmacc = (pred_y == test_y).sum()/len(pred_y)
        print("SVM epoch {}\t acc: {:.6f}".format(i,svmacc)) 
        avgsvmacc += svmacc/10
    return avgsvmacc

def cooldown(args, qshaper, classifier, valloader, testloader, half=False):
    
    optim_c = torch.optim.Adam(classifier.parameters(), lr=args.lr*2, weight_decay=args.lr/5)
    criterion = nn.CrossEntropyLoss()

    before=time.time()

    bestacc = 0.5
    avgnorm = args.amp
    avgacc = 0.5
    for e in range(args.cooldown):
        classifier.train()
        for x,y in valloader:
            xdata, ydata = x, y.to(args.device)
            #train classifier
            optim_c.zero_grad()
            if half:
                perturb = qshaper(x.to(args.device).half()).float()
            else:
                perturb = qshaper(xdata).to(args.device)
            output = classifier(xdata.to(args.device) + perturb)
            
            loss_c = criterion(output, ydata)
            loss_c.backward()
            
            optim_c.step()

            pred = output.argmax(axis=-1)

        #validation
        totcorrect = 0
        totcount = 0
        closs = 0.0
        mperturb = 0.0
        with torch.no_grad():
            classifier.eval()
            for x,y in testloader:
                xdata, ydata = x, y.to(args.device)
                #train classifier
                optim_c.zero_grad()

                if half:
                    perturb = qshaper(x.to(args.device).half()).float()
                else:
                    perturb = qshaper(xdata).to(args.device)
                mperturb += perturb.mean().item()
                output = classifier(xdata.to(args.device) + perturb)
                closs += criterion(output, ydata)

                pred = output.argmax(axis=-1)
                totcorrect += (pred==ydata).sum().item()
                totcount += y.size(0)
        closs = closs / len(testloader)
        mperturb = mperturb/len(testloader)
        macc = float(totcorrect)/totcount
        avgnorm = avgnorm*0.9 + mperturb*0.1
        
        avgacc = avgacc*0.9 + macc*0.1

        if e==0:
            avgnorm = mperturb
            avgacc = macc
        if e%10 == 9:
            elapsed = time.time() - before
            before += elapsed
            print("Evaluate epoch {} \t acc {:.4f}\t  closs {:.4f}\t mperturb: {:.4f}\t time: {: .4f}".format(
                e+1, avgacc, closs, avgnorm, elapsed))
        if avgacc > bestacc and e > args.cooldown//2:
            bestacc = avgacc
    
      
    return avgacc, mperturb

def eval_noisegen(args):
    if args.gen == 'gau':
        shaper = NewModels.GaussianGenerator(args.amp)
    elif args.gen == 'sin':
        shaper = NewModels.GaussianSinusoid(args.amp)
    else:
        shaper = NewModels.OffsetGenerator(args.amp)
    file_prefix=args.victim
    trainset = LOTRDataset(file_prefix+'_train.pkl')
    testset =  LOTRDataset(file_prefix+'_test.pkl', med=trainset.med)
    valset = LOTRDataset(file_prefix+'_valid.pkl', med=trainset.med)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=8)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=8, shuffle=True)
    clfs = {
        'cnn' : NewModels.CNNModel(trainset.tracelen).to(args.device),
        'rnn' : NewModels.RNNClassifier().to(args.device),
        'att' : NewModels.ConvAttClassifier().to(args.device),
        'deep' : NewModels.CNNModelDeep(trainset.tracelen).to(args.device),
        'wide' : NewModels.CNNModelWide(trainset.tracelen).to(args.device)
    }
    modeltoacc = {}
    svmacc = svm_cooldown(args, shaper, valloader, testloader)
    modeltoacc['svm'] = svmacc
    for cname in clfs:
        classifier = clfs[cname]
        print('\nEvaluating ' + cname)
        print(classifier)

        bestacc, mperturb = cooldown(args, shaper, classifier, valloader, testloader)
        modeltoacc[cname] = bestacc
    print(modeltoacc)

if __name__ == '__main__':
    #torch.backends.cuda.matmul.allow_tf32 = False
    #torch.backends.cudnn.allow_tf32 = False
    args = Util.get_args()
    if args.gen in ['gau','sin','off']:
        eval_noisegen(args)
    else:
        evaluate(args)


    
    
    