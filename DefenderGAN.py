from RingDataset import RingDataset, EDDSADataset
import Models
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import re
import time
from sklearn import svm
import Util
from Util import shifter, quantizer

if __name__ == '__main__':
    #torch.backends.cuda.matmul.allow_tf32 = False
    #torch.backends.cudnn.allow_tf32 = False
    args = Util.get_args()
    env = Util.Env(args)
    
    optim_c = torch.optim.Adam(env.classifier.parameters(), lr=args.lr)
    optim_d = torch.optim.RMSprop(env.disc.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.BCELoss()
    C = args.amp
    scale = args.lambda_h
    #scale = 0.1
    for e in range(args.warmup):
        env.classifier.train()
        env.disc.train()
        totcorrect = 0
        totcount = 0

        mloss = 0.0
        for x,y in env.trainloader:
            xdata, ydata = x.cuda(), y.cuda()
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio)
            shifted = shifter(xdata, args.history)
            #train classifier
            optim_c.zero_grad()
            optim_d.zero_grad()

            perturb = env.gen(shifted).view(shifted.size(0),-1)
            p_input = xdata[:,args.history-1:]+perturb.detach()
            output = env.classifier(p_input)

            fakes = env.disc(p_input)
            loss_c = criterion(output, ydata)
            loss_c.backward()

            loss_d = torch.mean(fakes*disc_label)
            mloss = mloss + loss_d.item()
            loss_d.backward()
            optim_c.step()

            optim_d.step()
            env.disc.clip()

            pred = output.argmax(axis=-1)
            totcorrect += (pred==ydata).sum().item()
            totcount += y.size(0)

        mloss = mloss / len(env.trainloader)
        macc = float(totcorrect)/totcount


        if (e+1)%10 == 0:
            print("Warmup epoch {} \t acc {:.4f}\t dloss {}".format(e+1, macc, mloss))

    if env.both:
        optim_c2 = torch.optim.Adam(env.classifier2.parameters(), lr=args.lr)
        optim_d2 = torch.optim.RMSprop(env.disc2.parameters(), lr=args.lr)
        for e in range(args.warmup):
            env.classifier2.train()
            env.disc2.train()
            totcorrect = 0
            totcount = 0

            mloss = 0.0
            for x,y in env.trainloader2:
                xdata, ydata = x.cuda(), y.cuda()
                oneratio = ydata.sum().item()/len(ydata)
                disc_label = 2*(ydata.float()-oneratio)
                shifted = shifter(xdata, args.history)
                #train classifier
                optim_c2.zero_grad()
                optim_d2.zero_grad()

                perturb = env.gen(shifted).view(shifted.size(0),-1)
                p_input = xdata[:,args.history-1:]+perturb.detach()
                output = env.classifier2(p_input)

                fakes = env.disc2(p_input)
                loss_c = criterion(output, ydata)
                loss_c.backward()

                loss_d = torch.mean(fakes*disc_label)
                mloss = mloss + loss_d.item()
                loss_d.backward()
                optim_c2.step()

                optim_d2.step()
                env.disc2.clip()

                pred = output.argmax(axis=-1)
                totcorrect += (pred==ydata).sum().item()
                totcount += y.size(0)     

            mloss = mloss / len(env.trainloader2)
            macc = float(totcorrect)/totcount
            if (e+1)%10 == 0:
                print("Warmup epoch {} \t acc {:.4f}\t dloss {}".format(e+1, macc, mloss))
        optim_c2 = torch.optim.Adam(env.classifier2.parameters(), lr=2*args.lr)
        optim_c_t2 = torch.optim.Adam(env.classifier_test2.parameters(), lr=args.lr)
        optim_d2 = torch.optim.RMSprop(env.disc2.parameters(), lr=2*args.lr)

        sched_c2   = torch.optim.lr_scheduler.StepLR(optim_c2, 1, gamma=args.gamma)
        sched_c_t2   = torch.optim.lr_scheduler.StepLR(optim_c_t2, 1, gamma=args.gamma)
        sched_d2   = torch.optim.lr_scheduler.StepLR(optim_d2, 1, gamma=args.gamma)

    optim_c = torch.optim.Adam(env.classifier.parameters(), lr=2*args.lr)
    optim_c_t = torch.optim.Adam(env.classifier_test.parameters(), lr=args.lr)
    optim_g = torch.optim.Adam(env.gen.parameters(), lr=args.lr)
    optim_d = torch.optim.RMSprop(env.disc.parameters(), lr=2*args.lr)

    sched_c   = torch.optim.lr_scheduler.StepLR(optim_c, 1, gamma=args.gamma)
    sched_c_t   = torch.optim.lr_scheduler.StepLR(optim_c_t, 1, gamma=args.gamma)
    sched_g   = torch.optim.lr_scheduler.StepLR(optim_g, 1, gamma=args.gamma)
    sched_d   = torch.optim.lr_scheduler.StepLR(optim_d, 1, gamma=args.gamma)
    for e in range(args.epochs):
        env.gen.train()
        env.classifier.train()
        
        env.disc.train()
        trainstart = time.time()
        # Train classifier and discriminator

        for x,y in env.trainloader:
            xdata, ydata = x.cuda(), y.cuda()
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio) # 1 for ones, -1 for zeros
            shifted = shifter(xdata, args.history)
            #train classifier
            optim_c.zero_grad()
            
            optim_d.zero_grad()
            perturb = env.gen(shifted).view(shifted.size(0),-1)

            #interleaving?
            p_input = xdata[:,args.history-1:]+perturb.detach()
            output = env.classifier(p_input)
            
            fakes = env.disc(p_input)
            loss_c = criterion(output, ydata)
            loss_c.backward()
            
            loss_d = torch.mean(fakes*disc_label)
            loss_d.backward()
            optim_c.step()
            
            optim_d.step()
            env.disc.clip()

            #train generator
            optim_g.zero_grad()

            p_input = xdata[:,args.history-1:]+perturb
            output = env.classifier(p_input)
            
            fakes = env.disc(p_input)
            hinge = perturb.mean(dim=-1) - C
            hinge[hinge<0] = 0.0
            loss_p = torch.mean(hinge)
            loss_d = -torch.mean(fakes*disc_label)
            fake_target = 1-ydata

            loss_adv1 = criterion(output, fake_target)
            loss = loss_adv1 + scale*loss_p + 0.02*loss_d
            loss.backward()
            optim_g.step()

        env.classifier.train()
        for x,y in env.trainloader:
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio) # 1 for ones, -1 for zeros
            shifted = shifter(xdata, args.history)
            #train classifier
            optim_c.zero_grad()
            
            optim_d.zero_grad()
            perturb = env.gen(shifted).view(shifted.size(0),-1)

            #interleaving?
            p_input = xdata[:,args.history-1:]+perturb.detach()
            output = env.classifier(p_input)
            
            fakes = env.disc(p_input)
            loss_c = criterion(output, ydata)
            loss_c.backward()
            
            loss_d = torch.mean(fakes*disc_label)
            loss_d.backward()
            optim_c.step()
            
            optim_d.step()
            env.disc.clip()

        mloss = 0.0
        mloss2 = 0.0
        totcorrect = 0
        totcount = 0
        mnorm = 0.0

        #evaluate classifier
        with torch.no_grad():
            env.classifier.eval()
            
            for x,y in env.testloader:
                xdata, ydata = x.cuda(), y.cuda()
                oneratio = ydata.sum().item()/len(ydata)
                disc_label = 2*(ydata.float()-oneratio) # 1 for ones, -1 for zeros
                shifted = shifter(xdata, args.history)
                perturb = env.gen(shifted).view(shifted.size(0),-1)
                perturb = quantizer(perturb)
                norm = torch.mean(perturb)
                p_input = xdata[:,args.history-1:]+perturb.detach()
                output = env.classifier(p_input)
                
                fakes = env.disc(p_input)
                loss_c = criterion(output, ydata)
                pnorm = torch.norm(perturb, dim=-1) - C
                loss_p = torch.mean(torch.relu(pnorm))
                pred = output.argmax(axis=-1)
                
                mnorm += norm.item()/len(env.testloader)
                mloss += loss_c.item()/len(env.testloader)
                mloss2 += torch.mean(fakes*disc_label).item()/len(env.testloader)
                totcorrect += (pred==ydata).sum().item()
                totcount += y.size(0)                
            macc = float(totcorrect)/totcount

            print("[{}]\tepoch {} \t acc {:.4f}\t Avg perturb {:.4f}\t closs {:.4f}\t dloss {}".format(
                args.victim, e+1, macc, mnorm, mloss, mloss2))
        if env.both:
            env.gen.train()
            env.classifier2.train()
            
            env.disc2.train()
            trainstart = time.time()
            # Train classifier and discriminator

            for x,y in env.trainloader2:
                xdata, ydata = x.cuda(), y.cuda()
                oneratio = ydata.sum().item()/len(ydata)
                disc_label = 2*(ydata.float()-oneratio) # 1 for ones, -1 for zeros
                shifted = shifter(xdata, args.history)
                #train classifier
                optim_c2.zero_grad()
                
                optim_d2.zero_grad()
                perturb = env.gen(shifted).view(shifted.size(0),-1)

                #interleaving?
                p_input = xdata[:,args.history-1:]+perturb.detach()
                output = env.classifier2(p_input)
                
                fakes = env.disc2(p_input)
                loss_c = criterion(output, ydata)
                loss_c.backward()
                
                loss_d = torch.mean(fakes*disc_label)
                loss_d.backward()
                optim_c2.step()
                
                optim_d2.step()
                env.disc2.clip()

                #train generator
                optim_g.zero_grad()

                p_input = xdata[:,args.history-1:]+perturb
                output = env.classifier2(p_input)
                
                fakes = env.disc2(p_input)
                hinge = perturb.mean(dim=-1) - C
                hinge[hinge<0] = 0.0
                loss_p = torch.mean(hinge)
                loss_d = -torch.mean(fakes*disc_label)
                fake_target = 1-ydata

                loss_adv1 = criterion(output, fake_target)
                loss = (loss_adv1 + scale*loss_p + 0.02*loss_d)
                loss.backward()
                optim_g.step()

            env.classifier2.train()
            for x,y in env.trainloader2:
                oneratio = ydata.sum().item()/len(ydata)
                disc_label = 2*(ydata.float()-oneratio) # 1 for ones, -1 for zeros
                shifted = shifter(xdata, args.history)
                #train classifier
                optim_c2.zero_grad()
                
                optim_d2.zero_grad()
                perturb = env.gen(shifted).view(shifted.size(0),-1)

                #interleaving?
                p_input = xdata[:,args.history-1:]+perturb.detach()
                output = env.classifier2(p_input)
                
                fakes = env.disc2(p_input)
                loss_c = criterion(output, ydata)
                loss_c.backward()
                
                loss_d = torch.mean(fakes*disc_label)
                loss_d.backward()
                optim_c2.step()
                
                optim_d2.step()
                env.disc2.clip()

            mloss = 0.0
            mloss2 = 0.0
            totcorrect = 0
            totcount = 0
            mnorm = 0.0

            #evaluate classifier
            with torch.no_grad():
                env.classifier2.eval()
                for x,y in env.testloader2:
                    xdata, ydata = x.cuda(), y.cuda()
                    oneratio = ydata.sum().item()/len(ydata)
                    disc_label = 2*(ydata.float()-oneratio) # 1 for ones, -1 for zeros
                    shifted = shifter(xdata, args.history)
                    perturb = env.gen(shifted).view(shifted.size(0),-1)
                    perturb = quantizer(perturb)
                    norm = torch.mean(perturb)
                    p_input = xdata[:,args.history-1:]+perturb.detach()
                    output = env.classifier2(p_input)
                    
                    fakes = env.disc2(p_input)
                    loss_c = criterion(output, ydata)
                    pnorm = torch.norm(perturb, dim=-1) - C
                    loss_p = torch.mean(torch.relu(pnorm))
                    pred = output.argmax(axis=-1)
                    
                    mnorm += norm.item()/len(env.testloader2)
                    mloss += loss_c.item()/len(env.testloader2)
                    mloss2 += torch.mean(fakes*disc_label).item()/len(env.testloader2)
                    totcorrect += (pred==ydata).sum().item()
                    totcount += y.size(0)                
                macc = float(totcorrect)/totcount

            print("[eddsa]\tepoch {} \t acc {:.4f}\t Avg perturb {:.4f}\t closs {:.4f}\t dloss {}".format(
                e+1, macc, mnorm, mloss, mloss2))
            sched_c2.step()
            sched_d2.step()
            sched_c_t2.step()
        sched_c.step()
        sched_d.step()
        sched_c_t.step()
        sched_g.step()

    lastacc, lastnorm = Util.cooldown(args, env, env.gen, env.gen2)

    if args.gen in ['adv', 'rnn', 'cnn', 'mlp', 'rnn3']:
        filename = "{}_{}_{}_{:.3f}_{:.3f}.pth".format(args.victim,args.gen,args.dim,lastnorm, lastacc)
        flist = os.listdir('gans')
        best = 1.0
        smallest = 5.0
        bperturb = 5.0
        rp = re.compile(r"{}_{}_{}_(\d\.\d+)_(\d\.\d+)\.pth".format(args.victim,args.gen,args.dim))
        for fn in flist:
            m = rp.match(fn)
            if m:
                facc = float(m.group(2))
                fperturb = float(m.group(1))
                if facc <= best:
                    best = facc
                    bperturb = fperturb
                if fperturb <= smallest:
                    smallest = fperturb
        if lastacc + 0.01*lastnorm <= best + 0.01*bperturb:
            print('New best found')
            torch.save(env.gen.state_dict(), './gans/'+filename)
            torch.save(env.gen.state_dict(), './gans/'+'best_{}_{}_{}.pth'.format(args.victim, args.gen, args.dim))
        elif lastnorm <= smallest:
            torch.save(env.gen.state_dict(), './gans/'+filename)