from RingDataset import RingDataset, EDDSADataset
from Models import CNNModel, RNNGenerator2, Distiller, MLP, RNNModel, QGRU, QGRU2
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

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args = get_args()

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
    assert os.path.isfile('./gans/best_{}_{}_{}.pth'.format(args.victim, args.gen, args.dim))
    gen.load_state_dict(torch.load('./gans/best_{}_{}_{}.pth'.format(args.victim, args.gen, args.dim)))

    student=QGRU2(window, scale=0.25, dim=args.student,  drop=0.0).cuda()
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
        for x,y in trainloader:
            xdata= x.cuda()
            shifted = shifter(xdata)
            #train classifier
            perturb = gen(shifted).view(shifted.size(0),-1)
            perturbed_x = xdata[:,31:]+perturb
            for p in perturbed_x:
                train_x.append(p.cpu().numpy())
            for y_i in y:
                train_y.append(y_i.item())
    clf = svm.SVC(gamma='auto')
    clf.fit(train_x, train_y)

    disc = Models.SVMDiscriminator(window, clf, 0.02).cuda() #discriminator

    optim_disc = torch.optim.RMSprop(disc.parameters(), lr=args.lr)
    optim_c = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    optim_c2 = torch.optim.Adam(classifier_test.parameters(), lr=args.lr)
    optim_g = torch.optim.Adam(student.parameters())
    optim_d = torch.optim.RMSprop(distiller.parameters())

    criterion = nn.CrossEntropyLoss()
    warmup = 20
    cooldown = 100
    scale = 0.0005
    #gen.eval()
    classifier.train()
    student.train()
    gen.train()
    #warmup and distillation
    for e in range(warmup):
        mloss = 0.0
        for x,y in trainloader:
            xdata, ydata = x.cuda(), y.cuda()
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio)
            shifted = shifter(xdata)
            #train classifier
            optim_c.zero_grad()
            optim_g.zero_grad()
            optim_d.zero_grad()
            optim_disc.zero_grad()
            perturb, t_out = gen(shifted, distill=True)
            perturb = perturb.view(shifted.size(0),-1)
            p_input = xdata[:,31:]+perturb.detach()
            output = classifier(p_input)
            fakes = disc(p_input)
            loss_disc = torch.mean(fakes*disc_label)
            loss_c = criterion(output, ydata)
            _, s_out = student(shifted, distill=True)
            loss_d = distiller(s_out, t_out)
            mloss += loss_d.item()/len(trainloader)
            loss_d.backward()
            loss_disc.backward()
            optim_g.step()
            optim_d.step()
            loss_c.backward()
            optim_c.step()
            optim_disc.step()
            disc.clip()
            #for p in disc.parameters():
            #    p.data.clamp_(-0.01, 0.01)
        print("Warmup {} \t Distill loss {:.4f}".format(e+1, mloss))
    optim_c = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    optim_g = torch.optim.Adam(student.parameters(), lr=args.lr)
    optim_d = torch.optim.Adam(distiller.parameters(), lr=args.lr)
    optim_disc = torch.optim.RMSprop(disc.parameters(), lr=args.lr)

    gamma = 0.98
    sched_c   = torch.optim.lr_scheduler.StepLR(optim_c, 1, gamma=gamma)
    sched_c2   = torch.optim.lr_scheduler.StepLR(optim_c2, 1, gamma=gamma)
    sched_g   = torch.optim.lr_scheduler.StepLR(optim_g, 1, gamma=gamma)
    sched_d   = torch.optim.lr_scheduler.StepLR(optim_d, 1, gamma=gamma)
    sched_disc = torch.optim.lr_scheduler.StepLR(optim_disc, 1, gamma=gamma)
    for e in range(args.epochs):
        student.train()
        classifier.train()
        trainstart = time.time()
        for x,y in trainloader:
            xdata, ydata = x.cuda(), y.cuda()
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio)
            shifted = shifter(xdata)
            #train classifier
            optim_c.zero_grad()
            optim_disc.zero_grad()
            perturb, s_out = student(shifted, distill=True)
            perturb = perturb.view(shifted.size(0),-1)

            p_input = xdata[:,31:]+perturb.detach()
            output = classifier(p_input)
            fakes = disc(p_input)
            loss_c = criterion(output, ydata)
            loss_c.backward()
            loss_disc = torch.mean(fakes*disc_label)
            loss_disc.backward()
            optim_c.step()
            optim_disc.step()
            disc.clip()
            #for p in disc.parameters():
            #    p.data.clamp_(-0.01, 0.01)
            #train student
            optim_g.zero_grad()
            optim_d.zero_grad()
            _, t_out = gen(shifted, distill=True)
            
            p_input = xdata[:,31:]+perturb.detach()
            output = classifier(p_input)
            fakes = disc(p_input)
            loss_comp = distiller(s_out, t_out)
            fake_target = 1-ydata
            loss_disc = -torch.mean(fakes*disc_label)
            loss_adv1 = criterion(output, fake_target)

            #loss = 0.5*loss_adv1 + loss_comp + 0.001*loss_disc
            loss = loss_adv1 + scale*loss_comp + 0.02*loss_disc

            loss.backward()
            optim_g.step()
            optim_d.step()
        classifier_test.train()
        student.eval()
        for x,y in valloader:
            xdata, ydata = x.cuda(), y.cuda()
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio)
            shifted = shifter(xdata)
            #train classifier
            optim_c2.zero_grad()
            perturb = student(shifted).view(shifted.size(0),-1)
            perturb = quantizer(perturb)

            #interleaving?
            output = classifier_test(xdata[:,31:]+perturb.detach())
            loss_c = criterion(output, ydata)
            loss_c.backward()
            optim_c2.step()
        sched_c.step()
        sched_c2.step()
        sched_g.step()
        sched_d.step()
        sched_disc.step()

        mloss = 0.0
        totcorrect = 0
        totcount = 0
        mnorm = 0.0
        #evaluate classifier
        with torch.no_grad():
            classifier_test.eval()
            for x,y in testloader:
                xdata, ydata = x.cuda(), y.cuda()
                shifted = shifter(xdata)
                perturb = student(shifted).view(shifted.size(0),-1)
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
            macc = float(totcorrect)/totcount
            print("epoch {} \t acc {:.4f}\t loss {:.4f}\t Avg perturb {:.4f}\t duration {:.4f}\n".format(e+1, macc, mloss, mnorm, time.time()-trainstart))
            if e > (args.epochs*4)//5 and macc - 0.5 < 0.001:
                break

    lastacc = 0.0
    lastnorm = 0.0
    optim_c2 = torch.optim.Adam(classifier_test.parameters(), lr=args.lr)
    sched_c2   = torch.optim.lr_scheduler.StepLR(optim_c2, 1, gamma=gamma)
    halfstudent = student.half()
    #halfstudent.eval()
    for e in range(cooldown):
        classifier_test.train()
        for x,y in valloader:
            xdata, ydata = x.cuda(), y.cuda()
            shifted = shifter(xdata)
            #train classifier
            optim_c2.zero_grad()
            perturb = halfstudent(shifted.half()).view(shifted.size(0),-1)
            #perturb = gen(xdata[:,31:])
            #interleaving?
            output = classifier_test(xdata[:,31:]+perturb.detach().float())
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
                perturb = halfstudent(shifted.half()).view(shifted.size(0),-1)
                perturb = quantizer(perturb)
                #perturb = gen(xdata[:,31:])
                norm = torch.mean(perturb)
                output = classifier_test(xdata[:,31:]+perturb.float())
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
            if (e+1)%10 == 0:
                print("epoch {} \t zacc {:.6f}\t oneacc {:.6f}\t loss {:.6f}\t Avg perturb {:.6f}\n".format(e+1, zacc, oacc, mloss, mnorm))
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
            perturb = halfstudent(shifted.half()).view(shifted.size(0),-1)
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
            perturb = halfstudent(shifted.half()).view(shifted.size(0),-1)
            perturb = quantizer(perturb)
            perturbed_x = xdata[:,31:]+perturb
            for p in perturbed_x:
                test_x.append(p.cpu().numpy())
            for y_i in y:
                test_y.append(y_i.item())
    clf = svm.SVC(gamma='auto')
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)

    svmacc = (pred_y == test_y).sum()/len(pred_y)
    print("SVM acc: {:.6f}".format(svmacc))
    lastacc = max(lastacc, svmacc)

    filename = "qgru_{}_{}_{:.3f}_{:.3f}.pth".format(args.victim, args.student,lastnorm, lastacc)
    flist = os.listdir('./gans')
    best = 1.0
    rp = re.compile(r"qgru_{}_{}_(\d\.\d+)_(\d\.\d+)\.pth".format(args.victim, args.student))
    for fn in flist:
        m = rp.match(fn)
        if m:
            facc = float(m.group(2))
            if facc <= best:
                best = facc
    if lastacc <= best:
        print('New best found')
        torch.save(student.state_dict(), './gans/'+filename)
        torch.save(student.state_dict(), './gans/'+'best_qgru_{}_{}.pth'.format(args.victim, args.student))
