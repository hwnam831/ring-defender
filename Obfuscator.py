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

def Warmup(args, trainloader,valloader, classifier, discriminator, shaper):
    optim_c = torch.optim.Adam(classifier.parameters(), lr=args.lr*5, weight_decay=args.lr/2)
    optim_d = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr*5, weight_decay=args.lr/2)
    criterion = nn.CrossEntropyLoss()
    shaper.train()
    for e in range(args.warmup):
        classifier.train()
        discriminator.train()
        for x,y in trainloader:
            xdata, ydata = x.to(args.device), y.to(args.device)
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
                    xdata, ydata = x.to(args.device), y.to(args.device)
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
    optim_g = torch.optim.Adam(shaper.parameters(), lr=args.lr, weight_decay=args.lr/2)
    if args.gen == 'rnn':
        optim_g = torch.optim.Adam(shaper.parameters(), lr=args.lr, weight_decay=args.lr/2)
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
        shaper.train()
        #train both
        for x,y in trainloader:
            xdata, ydata = x.to(args.device), y.to(args.device)
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
            loss = loss_adv1 + args.lambda_h*loss_p + args.lambda_d*loss_d
            loss.backward()
            optim_g.step()

        #train again
        for x,y in trainloader:
            xdata, ydata = x.to(args.device), y.to(args.device)
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
            shaper.train()
            for x,y in valloader:
                xdata, ydata = x.to(args.device), y.to(args.device)
                oneratio = ydata.sum().item()/len(ydata)
                disc_label = 2*(ydata.float()-oneratio)
                #train classifier

                perturb = shaper(xdata)
                mperturb += perturb.mean().item()
                output = classifier(xdata + perturb)
                closs += criterion(output, ydata)
                fakes = discriminator(xdata + perturb)
                mloss += torch.mean(fakes*disc_label).item()

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

def train(args):
    file_prefix=args.victim


    trainset = LOTRDataset(file_prefix+'_train.pkl')
    valset = LOTRDataset(file_prefix+'_valid.pkl', med=trainset.med)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=8, shuffle=True)
    testset =  LOTRDataset(file_prefix+'_test.pkl', med=trainset.med)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=8)

    classifier = NewModels.CNNModel(trainset.tracelen).to(args.device)
    discriminator = NewModels.FCDiscriminator(window=trainset.tracelen).to(args.device)

    shaper = NewModels.AttnShaper(amp=args.amp, history=args.window*2, window=args.window, dim=args.dim, n_patterns=args.n_patterns).to(args.device)
    if args.gen == 'adv':
        shaper = NewModels.GaussianShaper(history=args.window*2, window=args.window, amp=args.amp, dim=args.dim, n_patterns=args.n_patterns).to(args.device)
    if args.gen == 'qat':
        shaper = NewModels.QATShaper(history=args.window*2, window=args.window, amp=args.amp, dim=args.dim, n_patterns=args.n_patterns)
        shaper.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        torch.quantization.prepare_qat(shaper, inplace=False)
        shaper = shaper.to(args.device)
    elif args.gen == 'shaper':
        shaper = NewModels.AttnShaper2(amp=args.amp, history=args.window*2, window=args.window, dim=args.dim, n_patterns=args.n_patterns).to(args.device)
    elif args.gen == 'rnn':
        shaper = NewModels.RNNShaper(amp=args.amp, history=args.window*2, window=args.window, dim=args.dim).to(args.device)
    Warmup(args,trainloader, valloader, classifier, discriminator,shaper)
    bestacc, bestnorm = Train_DefenderGAN(args, trainloader,valloader, classifier, discriminator, shaper)
    if args.gen == 'adv':
        model_fp32 = NewModels.GaussianQuantizedShaper(shaper).to('cpu').eval()
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
        input_fp32 = torch.randn(args.batch_size, valset.tracelen)
        model_fp32_prepared(input_fp32)
        qshaper = torch.ao.quantization.convert(model_fp32_prepared)
        print('\nEvaluating')
        bestacc, mperturb = cooldown(args, qshaper, classifier, valloader, testloader)
    elif args.gen=='qat':
        print('\nEvaluating')
        shaper.to('cpu')
        qshaper = torch.ao.quantization.convert(shaper)
        qshaper.eval()
        bestacc, mperturb = cooldown(args, qshaper, classifier, valloader, testloader)
    elif args.gen=='shaper':
        bestacc, mperturb = cooldown(args, shaper.cpu(), classifier, valloader, testloader)
    elif args.gen == 'rnn':
        print('\nEvaluating')
        bestacc, mperturb = cooldown(args, shaper.cpu(), classifier, valloader, testloader)
    else:
        model_fp32 = NewModels.QuantizedShaper(shaper).to('cpu').eval()
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
        input_fp32 = torch.randn(args.batch_size, valset.tracelen)
        model_fp32_prepared(input_fp32)
        qshaper = torch.ao.quantization.convert(model_fp32_prepared)
        print('\nEvaluating')
        bestacc, mperturb = cooldown(args, qshaper, classifier, valloader, testloader)
            
    filename = "{}_{}_{}x_{}_{}_{:.3f}_{:.3f}.pth".format(args.victim,args.gen,args.window, args.dim, args.n_patterns,mperturb, bestacc)
    torch.save(shaper.state_dict(), './gans/'+filename)

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
    rp = re.compile(r"{}_{}_{}x_{}_{}_(\d+\.\d+)_(\d+\.\d+)\.pth".format(victim, args.gen, args.window,args.dim,args.n_patterns))
    modeltoacc = {}
    for fname in flist:
        m = rp.match(fname)
        if m:
            shaper.load_state_dict(torch.load('./gans/'+fname, map_location=args.device))
            if args.gen == 'adv':
                model_fp32 = NewModels.GaussianQuantizedShaper(shaper).to('cpu').eval()
                model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
                model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
                input_fp32 = torch.randn(args.batch_size, valset.tracelen)
                model_fp32_prepared(input_fp32)
                qshaper = torch.ao.quantization.convert(model_fp32_prepared)
            elif args.gen == 'shaper':
                qshaper=shaper
            elif args.gen == 'rnn':
                qshaper=shaper
            else:
                model_fp32 = NewModels.QuantizedShaper(shaper).to('cpu').eval()
                model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
                model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
                input_fp32 = torch.randn(args.batch_size, valset.tracelen)
                model_fp32_prepared(input_fp32)
                qshaper = torch.ao.quantization.convert(model_fp32_prepared)
            
            print('\nEvaluating ' + fname)
            #classifier = NewModels.ConvAttClassifier().to(args.device)
            classifier = NewModels.CNNModel(trainset.tracelen).to(args.device)
            bestacc, mperturb = cooldown(args, qshaper, classifier, valloader, testloader)
            modeltoacc[fname] = (bestacc, mperturb)
    print(modeltoacc)


def cooldown(args, qshaper, classifier, valloader, testloader):
    
    optim_c = torch.optim.Adam(classifier.parameters(), lr=args.lr*5, weight_decay=args.lr/2)
    criterion = nn.CrossEntropyLoss()

    avgsvmacc = 0.0
    before=time.time()
    for i in range(10):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        with torch.no_grad():
            for x,y in valloader:
                perturb = qshaper(x)
                perturbed_x = x + perturb
                for p in perturbed_x:
                    train_x.append(p.cpu().numpy())
                for y_i in y:
                    train_y.append(y_i.item())
            for x,y in testloader:
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

    bestacc = 0.5
    avgnorm = args.amp
    avgacc = 0.5
    for e in range(args.cooldown):
        classifier.train()
        for x,y in valloader:
            xdata, ydata = x, y.to(args.device)
            #train classifier
            optim_c.zero_grad()

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
    
      
    return max(avgacc,avgsvmacc), mperturb

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
    classifier = NewModels.CNNModel(trainset.tracelen).to(args.device)
    macc, mperturb = cooldown(args, shaper, classifier,valloader,testloader)
    print("macc: {}\tmperturb: {}".format(macc,mperturb))
if __name__ == '__main__':
    #torch.backends.cuda.matmul.allow_tf32 = False
    #torch.backends.cudnn.allow_tf32 = False
    args = Util.get_args()
    if args.gen in ['gau', 'sin', 'off']:
        eval_noisegen(args)
    elif args.mode == 'train':
        train(args)
    else:
        evaluate(args)


    
    
    