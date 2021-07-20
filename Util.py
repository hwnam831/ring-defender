import argparse
import torch
import torch.nn as nn

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
            "--victim",
            type=str,
            choices=['rsa', 'eddsa'],
            default='eddsa',
            help='Victim dataset choices')
    parser.add_argument(
            "--gen",
            type=str,
            choices=['gau', 'sin', 'adv', 'off', 'cnn', 'rnn', 'mlp'],
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
            default='32',
            help='number of samples window')
    parser.add_argument(
            "--epochs",
            type=int,
            default='100',
            help='number of epochs')
    parser.add_argument(
            "--batch_size",
            type=int,
            default='100',
            help='batch size')
    parser.add_argument(
            "--dim",
            type=int,
            default='192',
            help='internal channel dimension')
    parser.add_argument(
            "--lr",
            type=float,
            default=2e-5,
            help='Default learning rate')
    parser.add_argument(
            "--student",
            type=int,
            default=8,
            help='Student dim')
    parser.add_argument(
            "--amp",
            type=float,
            default='2.9',
            help='noise amp scale')
    parser.add_argument(
            "--fresh",
            action='store_true',
            help='Fresh start without loading')

    return parser.parse_args()

def shifter(arr, history=32):
    dup = arr[:,None,:].expand(arr.size(0), arr.size(1)+1, arr.size(1))
    dup2 = dup.reshape(arr.size(0), arr.size(1), arr.size(1)+1)
    shifted = dup2[:,:history,:-history]
    return shifted

def quantizer(arr, std=8):
    return torch.round(arr*std)/std

def cooldown(epochs, classifier_test, gen, valloader,  testloader, lr, criterion=nn.CrossEntropyLoss()):
    gen.eval()
    lastacc = 0.0
    lastnorm = 0.0
    optim_c_t = torch.optim.Adam(classifier_test.parameters(), lr=lr)
    sched_c_t   = torch.optim.lr_scheduler.StepLR(optim_c_t, 1, gamma=0.98)
    for e in range(epochs):
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