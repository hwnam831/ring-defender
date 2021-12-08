import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torch.nn.functional as F
import time
from PCIEDataset import RawClassifier, RawCNN, PCIEDataset, Cooldown

class Distiller(nn.Module):
    def __init__(self, teacher, student, lamb_d = 0.1):
        super().__init__()
        tdim = teacher.modelsize
        sdim = student.modelsize
        self.map1 = nn.Conv1d(sdim, tdim, 1)
        self.maps = nn.ModuleList(
            [nn.Conv1d(sdim, tdim, 1) for _ in student.resblocks]
        )
        self.criterion = nn.MSELoss()
        self.lamb_d = lamb_d

    def forward(self, t_out, s_out):
        perturb_s, intermediates_s = s_out
        perturb_t, intermediates_t = t_out
        l_distill = self.criterion(self.map1(intermediates_s[0]), intermediates_t[0])
        for i, (out_t, out_s) in enumerate(zip(intermediates_t[1:],intermediates_s[1:])):
            l_distill += self.criterion(self.maps[i](out_s),out_t.detach())

        l_recon = self.criterion(perturb_s, perturb_t.detach())
        return l_recon + self.lamb_d*l_distill

def Warmup2(classifier, teacher, student, distiller, trainloader, valloader, epochs=10):
    optim_c = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    optim_distill = torch.optim.RMSprop(distiller.parameters())
    optim_student = torch.optim.Adam(student.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    for e in range(epochs):
        classifier.train()
        curtime = time.time()
        mloss = 0.0
        mdistill = 0.0
        for x,y in trainloader:
            optim_c.zero_grad()
            optim_distill.zero_grad()
            optim_student.zero_grad()
            xdata = x.cuda().float()
            ydata = y.cuda()
            t_out = teacher(xdata, distill=True)
            s_out = student(xdata, distill=True)
            perturb = t_out[0]
            out = classifier(xdata+perturb.detach())
            loss = criterion(out,ydata)
            mloss += loss.item()/len(trainloader)
            nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            loss.backward()
            optim_c.step()
            loss_distill = distiller(t_out, s_out)
            mdistill += loss_distill.item()/len(trainloader)
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            nn.utils.clip_grad_norm_(distiller.parameters(), 1.0)
            loss_distill.backward()
            optim_student.step()
            optim_distill.step()
        print('Warmup Epoch: {}'.format(e+1))
        print('Training time: {}'.format(time.time()-curtime))
        print('Training loss: {}\nDistill loss: {}'.format(mloss,mdistill))
        classifier.eval()
        mloss = 0.0
        macc = 0.0
        for x,y in valloader:
            with torch.no_grad():
                xdata = x.cuda().float()
                ydata = y.cuda()
                out = classifier(xdata)
                loss = criterion(out,ydata)
                mloss += loss.item()/len(valloader)
                pred = out.argmax(axis=-1)
                acc = (ydata==pred).sum().item()/len(ydata)
                macc += acc/len(valloader)
        
        print('Test loss : {}\nTest acc: {}\n'.format(mloss, macc))

if __name__ == '__main__':
    raw_dataset = PCIEDataset('./train')
    classifier = RawClassifier(512, 64, 4).cuda()
    teacher = RawCNN(512, 64, 7).cuda()
    teacher.load_state_dict(torch.load('pcie/gen_{}_{}.pth'.format(teacher.window,teacher.modelsize)))
    for param in teacher.parameters():
        param.requires_grad = False
    student = RawCNN(512, 32, 7).cuda()
    distiller = Distiller(teacher, student, 0.05).cuda()
    trainset = []
    testset = []
    for i in range(len(raw_dataset)):
        if i%7 == 0:
            testset.append(i)
        else:
            trainset.append(i)
    trainloader = DataLoader(raw_dataset, batch_size=8, num_workers=4, sampler=
                            torch.utils.data.SubsetRandomSampler(trainset))
    valloader = DataLoader(raw_dataset, batch_size=8, num_workers=4, sampler=
                            torch.utils.data.SubsetRandomSampler(testset))
    
    Warmup2(classifier, teacher, student, distiller, trainloader, valloader, 15)

    criterion = nn.CrossEntropyLoss()
    C=6.0 # hyperparameter to choose
    scale = 0.1
    optim_c = torch.optim.Adam(classifier.parameters(), lr=1e-5)
    optim_student = torch.optim.Adam(student.parameters(), lr=2e-5)
    optim_distill = torch.optim.RMSprop(distiller.parameters())
    for e in range(30):
        classifier.train()
        student.train()
        curtime = time.time()
        mloss = 0.0
        mperturb = 0.0
        for x,y in trainloader:
            optim_c.zero_grad()
            
            xdata = x.cuda().float()
            ydata = y.cuda()
            
            perturb = student(xdata)
            out = classifier(xdata+perturb.detach())
            loss = criterion(out,ydata)
            mloss += loss.item()/len(trainloader)
            nn.utils.clip_grad_norm_(classifier.parameters(), 0.1)
            loss.backward()
            optim_c.step()

            #Train generator
            optim_student.zero_grad()
            optim_distill.zero_grad()
            fake_labels = torch.zeros_like(y).cuda()
            s_out = student(xdata, distill=True)
            perturb = s_out[0]
            out = classifier(xdata+perturb)
            loss_g = criterion(out,fake_labels)
            loss_distill = distiller(teacher(xdata, distill=True), s_out)
            loss = loss_distill*scale + loss_g
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 0.1)
            nn.utils.clip_grad_norm_(distiller.parameters(), 0.1)
            optim_student.step()
            optim_distill.step()
        print('Epoch: {}'.format(e+1))
        print('Training time: {}'.format(time.time()-curtime))
        print('Training loss: {}'.format(mloss))
        classifier.eval()
        student.eval()
        mloss = 0.0
        macc = 0.0
        mdistill = 0.0
        for x,y in valloader:
            with torch.no_grad():
                xdata = x.cuda().float()
                ydata = y.cuda()
                s_out = student(xdata, distill=True)
                perturb = s_out[0].detach()
                out = classifier(xdata+perturb)
                mperturb += perturb.mean().item() / len(valloader)
                loss = criterion(out,ydata)
                loss_distill = distiller(teacher(xdata, distill=True), s_out)
                mloss += loss.item()/len(valloader)
                mdistill += loss_distill.item()/len(valloader)
                pred = out.argmax(axis=-1)
                acc = (ydata==pred).sum().item()/len(ydata)
                macc += acc/len(valloader)
        
        print('Test loss : {}\nDistill loss : {}\nTest acc: {}\n'.format(mloss, mdistill, macc))
        print('Test mean perturb : {:.5f}'.format(mperturb))
    torch.save(student.state_dict(), 'pcie/student_{}_{}.pth'.format(student.window,student.modelsize))
    test_dataset = PCIEDataset('./nvmessd')
    trainset = []
    testset = []
    for i in range(len(test_dataset)):
        if i%7 == 0:
            testset.append(i)
        else:
            trainset.append(i)
    trainloader = DataLoader(test_dataset, batch_size=8, num_workers=4, sampler=
                            torch.utils.data.SubsetRandomSampler(trainset))
    valloader = DataLoader(test_dataset, batch_size=8, num_workers=4, sampler=
                            torch.utils.data.SubsetRandomSampler(testset))
    classifier = RawClassifier(512,128,4).cuda()
    Cooldown(classifier, student, trainloader, valloader, epochs=20)
