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
    #TODO: no more threshold
    if args.gen == 'gau':
        gen = nn.Sequential(
        Models.GaussianGenerator(window, scale=args.amp),
        nn.ReLU(),
        ).cuda()
    elif args.gen == 'sin':
        gen = nn.Sequential(
        Models.GaussianSinusoid(window, scale=args.amp),
        nn.ReLU(),
        ).cuda()
    elif args.gen == 'cnn':
        gen=Models.CNNGenerator(window, scale=0.25, dim=args.dim).cuda()
    elif args.gen == 'adv':
        gen=Models.RNNGenerator2(window, scale=0.25, dim=args.dim).cuda()
    elif args.gen == 'rnn':
        gen=Models.RNNGenerator(window, scale=0.25, dim=args.dim).cuda()
    elif args.gen == 'mlp':
        gen=Models.MLPGen(window, scale=0.25, dim=args.dim).cuda()
    elif args.gen == 'off':
        gen=Models.OffsetGenerator(window, scale=args.amp/2).cuda()
    else:
        print(args.gen + ' not supported\n')
        exit(-1)
    if args.gen in ['cnn', 'adv', 'rnn', 'mlp'] and \
        os.path.isfile('./gans/best_{}_{}_{}.pth'.format(args.victim,args.gen, args.dim)) and not args.fresh:
            print('Previous best found: loading the model...')
            gen.load_state_dict(torch.load('./gans/best_{}_{}_{}.pth'.format(args.victim,args.gen, args.dim)))
    if args.net == 'ff':
        classifier = Models.MLP(window, dim=args.dim).cuda()
    elif args.net == 'rnn':
        classifier = Models.RNNModel(window, dim=args.dim).cuda()
    else:
        classifier = Models.CNNModel(window, dim=args.dim).cuda()

    if args.testnet == 'ff':
        classifier_test = Models.MLP(window, dim=args.dim).cuda()
    elif args.testnet == 'rnn':
        classifier_test = Models.RNNModel(window, dim=args.dim).cuda()
    else:
        classifier_test = Models.CNNModel(window, dim=args.dim).cuda()

    

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

    xvar = np.array(train_x).var()
    
    disc = Models.SVMDiscriminator(window, clf, 0.02).cuda() #discriminator
    optim_c = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    
    optim_d = torch.optim.RMSprop(disc.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.BCELoss()
    C = args.amp
    warmup = 70
    cooldown = 100
    scale = 0.02
    #scale = 0.1
    for e in range(warmup):
        classifier.train()

        disc.train()
        totcorrect = 0
        totcount = 0

        mloss = 0.0
        for x,y in trainloader:
            xdata, ydata = x.cuda(), y.cuda()
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio)
            shifted = shifter(xdata)
            #train classifier
            optim_c.zero_grad()
            optim_d.zero_grad()

            perturb = gen(shifted).view(shifted.size(0),-1)
            p_input = xdata[:,31:]+perturb.detach()
            output = classifier(p_input)

            fakes = disc(p_input)
            #reals = disc(xdata[:,31:])
            loss_c = criterion(output, ydata)

            loss_c.backward()

            loss_d = torch.mean(fakes*disc_label)
            mloss = mloss + loss_d.item()
            loss_d.backward()
            optim_c.step()

            optim_d.step()
            disc.clip()
            #for p in disc.parameters():
            #    p.data.clamp_(-0.01, 0.01)

            pred = output.argmax(axis=-1)

            totcorrect += (pred==ydata).sum().item()
            totcount += y.size(0)

            

        mloss = mloss / len(trainloader)
        macc = float(totcorrect)/totcount


        if (e+1)%10 == 0:
            print("Warmup epoch {} \t acc {:.4f}\t dloss {}".format(e+1, macc, mloss))
            #print("zacc {:.6f}\t oneacc {:.6f}\n".format(zacc, oacc))

    optim_c = torch.optim.Adam(classifier.parameters(), lr=2*args.lr)

    optim_c_t = torch.optim.Adam(classifier_test.parameters(), lr=args.lr)
    optim_g = torch.optim.Adam(gen.parameters(), lr=args.lr)
    optim_d = torch.optim.RMSprop(disc.parameters(), lr=2*args.lr)
    gamma = 0.97
    sched_c   = torch.optim.lr_scheduler.StepLR(optim_c, 1, gamma=gamma)
    
    sched_c_t   = torch.optim.lr_scheduler.StepLR(optim_c_t, 1, gamma=gamma)
    sched_g   = torch.optim.lr_scheduler.StepLR(optim_g, 1, gamma=gamma)
    sched_d   = torch.optim.lr_scheduler.StepLR(optim_d, 1, gamma=gamma)
    for e in range(args.epochs):
        gen.train()
        classifier.train()
        
        disc.train()
        trainstart = time.time()
        for x,y in trainloader:
            xdata, ydata = x.cuda(), y.cuda()
            oneratio = ydata.sum().item()/len(ydata)
            disc_label = 2*(ydata.float()-oneratio) # 1 for ones, -1 for zeros
            shifted = shifter(xdata)
            #train classifier
            optim_c.zero_grad()
            
            optim_d.zero_grad()
            perturb = gen(shifted).view(shifted.size(0),-1)

            #interleaving?
            p_input = xdata[:,31:]+perturb.detach()
            output = classifier(p_input)
            
            fakes = disc(p_input)
            #reals = disc(xdata[:,31:])
            loss_c = criterion(output, ydata)
            loss_c.backward()
            
            
            loss_d = torch.mean(fakes*disc_label)
            loss_d.backward()
            optim_c.step()
            
            optim_d.step()
            disc.clip()
            #for p in disc.parameters():
            #    p.data.clamp_(-0.01, 0.01)

            #train generator
            optim_g.zero_grad()
            #perturb = gen(shifted).view(shifted.size(0),-1)
            #perturb = gen(xdata[:,31:])
            p_input = xdata[:,31:]+perturb
            output = classifier(p_input)
            
            fakes = disc(p_input)
            #perturb = quantizer(perturb)
            #pnorm = torch.norm(perturb, dim=-1) - C
            #loss_p = torch.mean(torch.relu(pnorm))
            hinge = perturb.mean(dim=-1) - C
            hinge[hinge<0] = 0.0
            loss_p = torch.mean(hinge)
            loss_d = -torch.mean(fakes*disc_label)
            #loss_p = torch.mean(torch.norm(perturb,dim=-1))
            fake_target = 1-ydata

            loss_adv1 = criterion(output, fake_target)
            
            #loss = 0.5*loss_adv1 + 0.1*loss_adv2 + scale*loss_p + 0.5*loss_d
            loss = loss_adv1 + scale*loss_p + 0.02*loss_d
            #loss = loss_adv1 + scale*loss_p
            loss.backward()
            optim_g.step()
        classifier_test.train()
        gen.eval()
        for x,y in valloader:
            xdata, ydata = x.cuda(), y.cuda()
            shifted = shifter(xdata)
            #train classifier
            optim_c_t.zero_grad()
            perturb = gen(shifted).view(shifted.size(0),-1)
            perturb = quantizer(perturb)
            #perturb = gen(xdata[:,31:])
            #interleaving?
            output = classifier_test(xdata[:,31:]+perturb.detach())
            loss_c = criterion(output, ydata)
            loss_c.backward()
            optim_c_t.step()
        sched_c.step()
        sched_d.step()
        sched_c_t.step()
        sched_g.step()

        mloss = 0.0
        mloss2 = 0.0
        totcorrect = 0
        totcount = 0
        mnorm = 0.0
        

        #evaluate classifier
        with torch.no_grad():
            classifier.eval()
            
            for x,y in testloader:
                xdata, ydata = x.cuda(), y.cuda()
                oneratio = ydata.sum().item()/len(ydata)
                disc_label = 2*(ydata.float()-oneratio) # 1 for ones, -1 for zeros
                shifted = shifter(xdata)
                perturb = gen(shifted).view(shifted.size(0),-1)
                perturb = quantizer(perturb)
                #perturb = gen(xdata[:,31:])
                norm = torch.mean(perturb)
                #output = classifier_test(xdata[:,31:]+perturb)
                p_input = xdata[:,31:]+perturb.detach()
                output = classifier(p_input)
                
                fakes = disc(p_input)
                loss_c = criterion(output, ydata)
                pnorm = torch.norm(perturb, dim=-1) - C
                loss_p = torch.mean(torch.relu(pnorm))
                pred = output.argmax(axis=-1)
                
                mnorm += norm.item()/len(testloader)
                mloss += loss_c.item()/len(testloader)
                mloss2 += torch.mean(fakes*disc_label).item()/len(testloader)
                #macc += ((pred==ydata).sum().float()/pred.nelement()).item()/len(testloader)
                totcorrect += (pred==ydata).sum().item()
                totcount += y.size(0)
                
                
            macc = float(totcorrect)/totcount
            

            #print("epoch {} \t acc {:.4f}\t loss {:.4f}\t loss_p {:.4f}\t Avg perturb {:.4f}\t duration {:.4f}".format(e+1, macc, mloss, mloss2, mnorm, time.time()-trainstart))
            #print("zacc {:.6f}\t oneacc {:.6f}\n".format(zacc, oacc))
            print("epoch {} \t acc {:.4f}\t Avg perturb {:.4f}\t closs {:.4f}\t dloss {}".format(e+1, macc, mnorm, mloss, mloss2))
            if e > (args.epochs*4)//5 and macc - 0.5 < 0.001:
                break
    gen.eval()
    lastacc = 0.0
    lastnorm = 0.0
    optim_c_t = torch.optim.Adam(classifier_test.parameters(), lr=args.lr)
    sched_c_t   = torch.optim.lr_scheduler.StepLR(optim_c_t, 1, gamma=gamma)
    for e in range(cooldown):
        classifier_test.train()
        for x,y in valloader:
            xdata, ydata = x.cuda(), y.cuda()
            shifted = shifter(xdata)
            #train classifier
            optim_c_t.zero_grad()
            perturb = gen(shifted).view(shifted.size(0),-1)
            #perturb = gen(xdata[:,31:])
            #interleaving?
            output = classifier_test(xdata[:,31:]+perturb.detach())
            loss_c = criterion(output, ydata)
            loss_c.backward()
            optim_c_t.step()


        mloss = 0.0
        dloss = 0.0
        totcorrect = 0
        totcount = 0
        mnorm = 0.0
        totcorrect2 = 0
        totcount2 = 0
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
                totcorrect2 += ((pred==0)*(ydata==0)).sum().item()
                totcount2 += (ydata==0).sum().item()
                onecorrect += ((pred==1)*(ydata==1)).sum().item()
                onecount += (ydata==1).sum().item()
            macc = float(totcorrect)/totcount
            zacc = float(totcorrect2)/totcount2
            oacc = float(onecorrect)/onecount
            if (e+1)%10 == 0:
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
    clf = svm.SVC(gamma='auto')
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)

    svmacc = (pred_y == test_y).sum()/len(pred_y)
    print("SVM acc: {:.6f}".format(svmacc))
    lastacc = max(lastacc, svmacc)
    if args.gen in ['adv', 'rnn', 'cnn', 'mlp']:
        filename = "{}_{}_{}_{:.3f}_{:.3f}.pth".format(args.victim,args.gen,args.dim,lastnorm, lastacc)
        flist = os.listdir('gans')
        best = 1.0
        smallest = 10.0
        rp = re.compile(r"{}_{}_{}_(\d\.\d+)_(\d\.\d+)\.pth".format(args.victim,args.gen,args.dim))
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
            torch.save(gen.state_dict(), './gans/'+filename)
            torch.save(gen.state_dict(), './gans/'+'best_{}_{}_{}.pth'.format(args.victim, args.gen, args.dim))
        elif lastnorm <= smallest:
            torch.save(gen.state_dict(), './gans/'+filename)