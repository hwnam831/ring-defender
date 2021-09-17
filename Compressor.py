from RingDataset import RingDataset, EDDSADataset
from Models import CNNModel, RNNGenerator2, MLPGen, Distiller, MLP, RNNModel, QGRU, QGRU2
import Models
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import re
import time
from sklearn import svm
from Util import get_args, shifter, quantizer
import Util

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args = get_args()
    env = Util.Env(args)
    '''
    if args.victim == 'rsa':
        file_prefix='core4ToSlice3'
        dataset = RingDataset(file_prefix+'_train.pkl', threshold=args.window)
        testset =  RingDataset(file_prefix+'_test.pkl', threshold=args.window)
        valset = RingDataset(file_prefix+'_valid.pkl', threshold=args.window)
        window = args.window
    else:
        file_prefix='eddsa'
        dataset = EDDSADataset(file_prefix+'_train.pkl')
        testset =  EDDSADataset(file_prefix+'_test.pkl')
        valset = EDDSADataset(file_prefix+'_valid.pkl')
        window = dataset.window

    trainset=dataset
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    gen=RNNGenerator2(window, scale=0.25, dim=args.dim, drop=0.0).cuda()
    if args.gen == 'mlp':
        gen=Models.MLPGen(window, scale=0.25, dim=args.dim).cuda()
    assert os.path.isfile('./gans/best_{}_{}_{}.pth'.format(args.victim, args.gen, args.dim))
    gen.load_state_dict(torch.load('./gans/best_{}_{}_{}.pth'.format(args.victim, args.gen, args.dim)))

    student=QGRU2(window, scale=0.25, dim=args.student,  drop=0.0).cuda()
    if args.gen == 'mlp':
        student = Models.MLPGen(window, scale=0.25, dim=args.student, depth=1).cuda()
    distiller = Distiller(window, args.dim, args.student, lamb_r = 0.1).cuda()

    if args.net == 'ff':
        classifier = MLP(window, dim=args.dim).cuda()
    elif args.net == 'rnn':
        classifier = RNNModel(window, dim=args.dim).cuda()
    else:
        classifier = CNNModel(window, dim=args.dim).cuda()

    if args.testnet == 'ff':
        classifier_test = MLP(window, dim=args.dim).cuda()
    elif args.testnet == 'rnn':
        classifier_test = RNNModel(window, dim=args.dim).cuda()
    else:
        classifier_test = CNNModel(window, dim=args.dim).cuda()
    
    train_x = []
    train_y = []
    with torch.no_grad():
        for x,y in env.trainloader:
            xdata= x.cuda()
            shifted = shifter(xdata, args.history)
            #train classifier
            perturb = env.gen(shifted).view(shifted.size(0),-1)
            perturbed_x = xdata[:,args.history-1:]+perturb
            for p in perturbed_x:
                train_x.append(p.cpu().numpy())
            for y_i in y:
                train_y.append(y_i.item())
    clf = svm.SVC(gamma='auto')
    clf.fit(train_x, train_y)

    disc = Models.SVMDiscriminator(window, clf, 0.02).cuda() #discriminator
    '''
    if args.gen == 'mlp':
        env.gen=Models.MLPGen(env.window, scale=0.25, dim=args.dim).cuda()
    assert os.path.isfile('./gans/best_{}_{}_{}.pth'.format(args.victim, args.gen, args.dim))
    env.gen.load_state_dict(torch.load('./gans/best_{}_{}_{}.pth'.format(args.victim, args.gen, args.dim)))

    student=QGRU2(env.window, scale=0.25, dim=args.student,  drop=0.0).cuda()
    if args.gen == 'mlp':
        student = Models.MLPGen(env.window, scale=0.25, dim=args.student, depth=1).cuda()
    distiller = Distiller(env.window, args.dim, args.student, lamb_r = 0.1).cuda()
    optim_disc = torch.optim.RMSprop(env.disc.parameters(), lr=args.lr)
    optim_c = torch.optim.Adam(env.classifier.parameters(), lr=args.lr)
    optim_c_t = torch.optim.Adam(env.classifier_test.parameters(), lr=args.lr)
    optim_g = torch.optim.Adam(student.parameters())
    optim_d = torch.optim.RMSprop(distiller.parameters())

    criterion = nn.CrossEntropyLoss()
    warmup = args.warmup
    cooldown = args.cooldown
    scale = args.lambda_r
    #gen.eval()
    env.classifier.train()
    student.train()
    env.disc.train()
    #warmup and distillation
    for e in range(warmup):
        mloss = 0.0
        for x,y in env.trainloader:
            xdata, ydata = x.cuda(), y.cuda()
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio)
            shifted = shifter(xdata, args.history)
            #train classifier
            optim_c.zero_grad()
            optim_g.zero_grad()
            optim_d.zero_grad()
            optim_disc.zero_grad()
            perturb, t_out = env.gen(shifted, distill=True)
            perturb = perturb.view(shifted.size(0),-1)
            p_input = xdata[:,args.history-1:]+perturb.detach()
            output = env.classifier(p_input)
            fakes = env.disc(p_input)
            loss_disc = torch.mean(fakes*disc_label)
            loss_c = criterion(output, ydata)
            _, s_out = student(shifted, distill=True)
            loss_d = distiller(s_out, t_out)
            mloss += loss_d.item()/len(env.trainloader)
            loss_d.backward()
            loss_disc.backward()
            optim_g.step()
            optim_d.step()
            loss_c.backward()
            optim_c.step()
            optim_disc.step()
            env.disc.clip()
            #for p in disc.parameters():
            #    p.data.clamp_(-0.01, 0.01)
        print("Warmup {} \t Distill loss {:.4f}".format(e+1, mloss))
    if env.both:
        optim_disc2 = torch.optim.RMSprop(env.disc2.parameters(), lr=args.lr)
        optim_c2 = torch.optim.Adam(env.classifier2.parameters(), lr=args.lr)
        optim_c_t2 = torch.optim.Adam(env.classifier_test2.parameters(), lr=args.lr)

        #gen.eval()
        env.classifier2.train()
        student.train()
        env.disc2.train()
        #warmup and distillation
        for e in range(warmup):
            mloss = 0.0
            for x,y in env.trainloader2:
                xdata, ydata = x.cuda(), y.cuda()
                oneratio = ydata.sum().item()/len(ydata)
                disc_label = 2*(ydata.float()-oneratio)
                shifted = shifter(xdata, args.history)
                #train classifier
                optim_c2.zero_grad()
                optim_g.zero_grad()
                optim_d.zero_grad()
                optim_disc2.zero_grad()
                perturb, t_out = env.gen(shifted, distill=True)
                perturb = perturb.view(shifted.size(0),-1)
                p_input = xdata[:,args.history-1:]+perturb.detach()
                output = env.classifier2(p_input)
                fakes = env.disc2(p_input)
                loss_disc = torch.mean(fakes*disc_label)
                loss_c = criterion(output, ydata)
                _, s_out = student(shifted, distill=True)
                loss_d = distiller(s_out, t_out)
                mloss += loss_d.item()/len(env.trainloader2)
                loss_d.backward()
                loss_disc.backward()
                optim_g.step()
                optim_d.step()
                loss_c.backward()
                optim_c2.step()
                optim_disc2.step()
                env.disc2.clip()
                #for p in disc.parameters():
                #    p.data.clamp_(-0.01, 0.01)
            print("Warmup {} \t Distill loss {:.4f}".format(e+1, mloss))
    optim_c = torch.optim.Adam(env.classifier.parameters(), lr=args.lr*2)
    optim_g = torch.optim.Adam(student.parameters(), lr=args.lr)
    optim_d = torch.optim.Adam(distiller.parameters(), lr=args.lr)
    optim_disc = torch.optim.RMSprop(env.disc.parameters(), lr=args.lr*2)

    gamma = 0.98
    sched_c   = torch.optim.lr_scheduler.StepLR(optim_c, 1, gamma=gamma)
    sched_c_t   = torch.optim.lr_scheduler.StepLR(optim_c_t, 1, gamma=gamma)
    sched_g   = torch.optim.lr_scheduler.StepLR(optim_g, 1, gamma=gamma)
    sched_d   = torch.optim.lr_scheduler.StepLR(optim_d, 1, gamma=gamma)
    sched_disc = torch.optim.lr_scheduler.StepLR(optim_disc, 1, gamma=gamma)
    if env.both:
        optim_c2 = torch.optim.Adam(env.classifier2.parameters(), lr=args.lr*2)
        optim_disc2 = torch.optim.RMSprop(env.disc2.parameters(), lr=args.lr*2)

        sched_c2   = torch.optim.lr_scheduler.StepLR(optim_c2, 1, gamma=gamma)
        sched_c_t2   = torch.optim.lr_scheduler.StepLR(optim_c_t2, 1, gamma=gamma)
        sched_disc2 = torch.optim.lr_scheduler.StepLR(optim_disc2, 1, gamma=gamma)
    for e in range(args.epochs):
        student.train()
        env.classifier.train()
        env.disc.train()
        trainstart = time.time()
        for x,y in env.trainloader:
            xdata, ydata = x.cuda(), y.cuda()
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio)
            shifted = shifter(xdata, args.history)
            #train classifier
            optim_c.zero_grad()
            optim_disc.zero_grad()
            perturb, s_out = student(shifted, distill=True)
            perturb = perturb.view(shifted.size(0),-1)

            p_input = xdata[:,args.history-1:]+perturb.detach()
            output = env.classifier(p_input)
            fakes = env.disc(p_input)
            loss_c = criterion(output, ydata)
            loss_c.backward()
            loss_disc = torch.mean(fakes*disc_label)
            loss_disc.backward()
            optim_c.step()
            optim_disc.step()
            env.disc.clip()
            #for p in disc.parameters():
            #    p.data.clamp_(-0.01, 0.01)
            #train student
            optim_g.zero_grad()
            optim_d.zero_grad()
            _, t_out = env.gen(shifted, distill=True)
            
            p_input = xdata[:,args.history-1:]+perturb.detach()
            output = env.classifier(p_input)
            fakes = env.disc(p_input)
            loss_comp = distiller(s_out, t_out)
            fake_target = 1-ydata
            loss_disc = -torch.mean(fakes*disc_label)
            loss_adv1 = criterion(output, fake_target)

            #loss = 0.5*loss_adv1 + loss_comp + 0.001*loss_disc
            loss = loss_adv1 + scale*loss_comp + 0.02*loss_disc

            loss.backward()
            optim_g.step()
            optim_d.step()
        env.classifier_test.train()
        student.eval()
        for x,y in env.valloader:
            xdata, ydata = x.cuda(), y.cuda()
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio)
            shifted = shifter(xdata, args.history)
            #train classifier
            optim_c_t.zero_grad()
            perturb = student(shifted).view(shifted.size(0),-1)
            perturb = quantizer(perturb)

            #interleaving?
            output = env.classifier_test(xdata[:,args.history-1:]+perturb.detach())
            loss_c = criterion(output, ydata)
            loss_c.backward()
            optim_c_t.step()
        if env.both:
            student.train()
            env.classifier2.train()
            env.disc2.train()
            for x,y in env.trainloader2:
                xdata, ydata = x.cuda(), y.cuda()
                oneratio = ydata.sum().item()/len(ydata)
                disc_label = 2*(ydata.float()-oneratio)
                shifted = shifter(xdata, args.history)
                #train classifier
                optim_c2.zero_grad()
                optim_disc2.zero_grad()
                perturb, s_out = student(shifted, distill=True)
                perturb = perturb.view(shifted.size(0),-1)

                p_input = xdata[:,args.history-1:]+perturb.detach()
                output = env.classifier2(p_input)
                fakes = env.disc2(p_input)
                loss_c = criterion(output, ydata)
                loss_c.backward()
                loss_disc = torch.mean(fakes*disc_label)
                loss_disc.backward()
                optim_c2.step()
                optim_disc2.step()
                env.disc2.clip()
                #for p in disc.parameters():
                #    p.data.clamp_(-0.01, 0.01)
                #train student
                optim_g.zero_grad()
                optim_d.zero_grad()
                _, t_out = env.gen(shifted, distill=True)
                
                p_input = xdata[:,args.history-1:]+perturb.detach()
                output = env.classifier2(p_input)
                fakes = env.disc2(p_input)
                loss_comp = distiller(s_out, t_out)
                fake_target = 1-ydata
                loss_disc = -torch.mean(fakes*disc_label)
                loss_adv1 = criterion(output, fake_target)

                #loss = 0.5*loss_adv1 + loss_comp + 0.001*loss_disc
                loss = loss_adv1 + scale*loss_comp + 0.02*loss_disc

                loss.backward()
                optim_g.step()
                optim_d.step()
            env.classifier_test2.train()
            student.eval()
            for x,y in env.valloader2:
                xdata, ydata = x.cuda(), y.cuda()
                oneratio = ydata.sum().item()/len(ydata)
                disc_label = 2*(ydata.float()-oneratio)
                shifted = shifter(xdata, args.history)
                #train classifier
                optim_c_t2.zero_grad()
                perturb = student(shifted).view(shifted.size(0),-1)
                perturb = quantizer(perturb)

                #interleaving?
                output = env.classifier_test2(xdata[:,args.history-1:]+perturb.detach())
                loss_c = criterion(output, ydata)
                loss_c.backward()
                optim_c_t2.step()
            sched_c2.step()
            sched_c_t2.step()
            sched_disc2.step()
        sched_c.step()
        sched_c_t.step()
        sched_g.step()
        sched_d.step()
        sched_disc.step()


        mloss = 0.0
        totcorrect = 0
        totcount = 0
        mnorm = 0.0
        #evaluate classifier
        with torch.no_grad():
            env.classifier_test.eval()
            for x,y in env.testloader:
                xdata, ydata = x.cuda(), y.cuda()
                shifted = shifter(xdata, args.history)
                perturb = student(shifted).view(shifted.size(0),-1)
                perturb = quantizer(perturb)
                #perturb = gen(xdata[:,args.history-1:])
                norm = torch.mean(perturb)
                output = env.classifier_test(xdata[:,args.history-1:]+perturb)
                loss_c = criterion(output, ydata)
                pred = output.argmax(axis=-1)
                mnorm += norm.item()/len(env.testloader)
                mloss += loss_c.item()/len(env.testloader)
                #macc += ((pred==ydata).sum().float()/pred.nelement()).item()/len(testloader)
                totcorrect += (pred==ydata).sum().item()
                totcount += y.size(0)
            macc = float(totcorrect)/totcount
            print("[{}]\tepoch {} \t acc {:.4f}\t loss {:.4f}\t Avg perturb {:.4f}\t duration {:.4f}\n".format(
                args.victim, e+1, macc, mloss, mnorm, time.time()-trainstart))
        if env.both:
            mloss = 0.0
            totcorrect = 0
            totcount = 0
            mnorm = 0.0
            #evaluate classifier
            with torch.no_grad():
                env.classifier_test2.eval()
                for x,y in env.testloader2:
                    xdata, ydata = x.cuda(), y.cuda()
                    shifted = shifter(xdata, args.history)
                    perturb = student(shifted).view(shifted.size(0),-1)
                    perturb = quantizer(perturb)
                    norm = torch.mean(perturb)
                    output = env.classifier_test2(xdata[:,args.history-1:]+perturb)
                    loss_c = criterion(output, ydata)
                    pred = output.argmax(axis=-1)
                    mnorm += norm.item()/len(env.testloader2)
                    mloss += loss_c.item()/len(env.testloader2)
                    totcorrect += (pred==ydata).sum().item()
                    totcount += y.size(0)
                macc = float(totcorrect)/totcount
                print("[{}]\tepoch {} \t acc {:.4f}\t loss {:.4f}\t Avg perturb {:.4f}\t duration {:.4f}\n".format(
                    'eddsa', e+1, macc, mloss, mnorm, time.time()-trainstart))
    
    lastacc, lastnorm = Util.cooldown(args, env, student, env.gen2)
    studentname = 'qgru' if args.gen == 'adv' else 'q' + args.gen
    filename = "{}_{}_{}_{:.3f}_{:.3f}.pth".format(studentname, args.victim, args.student,lastnorm, lastacc)
    flist = os.listdir('./gans')
    best = 1.0
    rp = re.compile(r"{}_{}_{}_(\d\.\d+)_(\d\.\d+)\.pth".format(studentname, args.victim, args.student))
    for fn in flist:
        m = rp.match(fn)
        if m:
            facc = float(m.group(2))
            if facc <= best:
                best = facc
    if lastacc <= best:
        print('New best found')
        torch.save(student.state_dict(), './gans/'+filename)
        torch.save(student.state_dict(), './gans/'+'best_{}_{}_{}.pth'.format(studentname, args.victim, args.student))
