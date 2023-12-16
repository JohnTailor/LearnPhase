import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as tca

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))

class B2lock2(nn.Module):
    def __init__(self, in_planes, planes,ker=3,stride=1,down=True,affine=True,track=True,leak=0):
        super(B2lock2, self).__init__()
        usemp=False
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=ker, stride=stride if not down or usemp else 2, padding=int(ker>1), bias=False)
        self.bnF = nn.BatchNorm2d(planes,affine=affine,track_running_stats=track)
        self.relu = nn.ReLU() #if leak==0 else nn.LeakyReLU(negative_slope=leak)
        self.mp = nn.MaxPool2d((2, 2), stride=2) if down and usemp else None

    def forward(self, out):
        out=self.conv1(out)
        out = self.bnF(out)
        out = self.relu(out)
        out = out if self.mp is None else self.mp(out)
        return out

def getCla(cfg,net=None,leak=0):
    if net is None: net=cfg["net"]
    if "F" in net:
        return DenseClassifier( net,cfg["num_classes"],cfg)
    if "V" in net or "S" in net:
        return Classifier( net,cfg["num_classes"],cfg,leak=leak)
    elif "R" in net:
        track=True#cfg["trackBnStats"] if "trackBnStats" in cfg else False
        import resnet as resnet
        return resnet.R10(cfg["num_classes"], cfg["imCh"],cfg["netSi"],cfg["imSi"],cfg["affine"],track)
    print("NOT FOUND Net in getCla",net)
    import sys
    sys.exit(-1)

class DenseClassifier(nn.Module):
    def __init__(self, net,ncl,cfg):
        super(DenseClassifier, self).__init__()
        tr = lambda x: int(x*cfg["netSi"])
        self.in_channels=cfg["imSi"] if cfg["imCh"]<0 else int(cfg["imSi"]*cfg["imSi"]*cfg["imCh"]) #if not "Cu" in cfg["ds"][0] else cfg["dataCu"]["Fea"][0]
        self.flat = Flatten()
        # nCen = 256 #if not "Cu" in cfg["ds"][0] else cfg["dataCu"]["hidFac"]*self.in_channels
        # chans = [self.in_channels]+[tr(nCen)]*int(cfg["net"][1:])+ [cfg["num_classes"]]
        # self.convLays=nn.ModuleList()
        # for i in range(len(chans)-2):
        #     #print(i,chans)
        #     #self.convLays.append(nn.Sequential(nn.Linear(chans[i],chans[i+1]),nn.BatchNorm1d(chans[i+1],affine=cfg["affine"]),nn.ReLU()))
        #     self.convLays.append(nn.Sequential(nn.Linear(chans[i], chans[i + 1]), nn.BatchNorm1d(chans[i + 1], affine=cfg["affine"]), nn.ReLU()))
        self.dummy=nn.Identity()
        self.dummy2 = nn.Identity()
        # self.pred=nn.Linear(chans[-2], chans[-1])
        self.pred = nn.Linear(self.in_channels, cfg["num_classes"],bias=False)

    def forward(self, x):
        x = self.flat(x)
        #for il,l in enumerate(self.convLays):  x = l(x)
        x=self.pred(x)
        return x

class Classifier(nn.Module):

    def __init__(self, net,ncl,cfg,leak=0):
        super(Classifier, self).__init__()
        tr = lambda x: int(cfg["netSi"]*x)
        track = cfg["trackBnStats"] if "trackBnStats" in cfg else True
        self.in_channels=cfg["imCh"]
        if net[0]=="V":
            self.mp = nn.Identity()
            if int(net[1:])==8:
                chans = [self.in_channels, 32,  64,  128, 256,  512]
                down = [1,1,1,1,1]
            elif int(net[1:])==11:
                chans = [self.in_channels, 32,  64,  128, 128, 256, 256, 512, 512]
                down = [1,1,0,1,0,1,0,1]
            elif int(net[1:]) == 13:
                chans = [self.in_channels, 32,32, 64, 128, 128, 256, 256, 512,512, 512]
                down = [1, 0,1, 0, 1, 0, 1, 0,0, 1]
            elif int(net[1:])==16:
                chans = [self.in_channels, 32,32,  64,64,  128,128, 128, 256,256, 256, 512, 512,512]
                down = [0,1,0,1,0,0,1,0,0,1,0,0,1]
            if cfg["imSi"] == 16:
                chans = chans[:-1]
                down = chans[:-1]
            if cfg["imSi"] == 64:
                chans = chans+[512]
                down = down+[1]
            finaldim=1


        elif net[0]=="S": #does not work with different netsize than 1!
            self.mp = nn.MaxPool2d((2, 2), stride=2) if cfg["imSi"]==32 else nn.Identity()
            chans = [self.in_channels]+[128]*int(net[1:])
            down=np.zeros(int(net[1:]))
            finaldim = 4

        def getConv(i,ker=3, down=True):
            return B2lock2( (tr(chans[i]) if i>0 else chans[i]),tr(chans[i+1]), ker=ker,down=down,track=track,leak=leak)

        self.convLays=[]#nn.ModuleList()
        for i, d in enumerate(down):
            self.convLays.append(getConv(i,down=d))
        self.convLays = nn.Sequential(*self.convLays)
        self.mp2 = nn.MaxPool2d((8, 8), stride=8) if net[0] == "S" else nn.Identity()

        self.flat = Flatten()
        self.dropout = nn.Dropout(cfg["drop"]) if "drop" in cfg else nn.Identity()
        self.nfea = chans[-1]*finaldim
        self.pred = nn.Linear(tr(chans[-1])*finaldim, cfg["num_classes"])
        #self.allExplainLays = self.convLays+[self.pred]
        #self.convLays = nn.Sequential(*self.convLays)

    def forward(self, x):
        x=self.mp(x)
        x=self.convLays(x)
        #for l in self.convLays:
         #   x = l(x)
        #print(x.shape)
        x=self.mp2(x)
        #print(x.shape)
        x=self.flat(x)
        x = self.dropout(x)
        x=self.pred(x)
        return x



def decay(opt,epoch,optimizerCl):
    if opt[0] == "S" and (epoch + 1) % (opt[1] // 3+opt[1]//10+2 ) == 0:
        for p in optimizerCl.param_groups: p['lr'] *= 0.1
        #print("  D", np.round(optimizerCl.param_groups[0]['lr'],5))

def getAcc(net, dataset,  niter=100000,setTrain=False,onCuda=False,mulCl=False,xind=0,cl=-1):
    correct,total = 0,0
    net.eval()
    with torch.no_grad():
        for cit,data in enumerate(dataset):
            with tca.autocast():
                if onCuda: dsx, dsy = data[xind], data[1]
                else: dsx,dsy = data[xind].cuda(),data[1].cuda()
                if cl!=-1:
                    dsx=dsx[dsy==cl]
                    dsy=dsy[dsy==cl]
                outputs = net(dsx.float())
                if mulCl:
                    result = outputs > 0.5
                    correct += (result == dsy).sum().item()
                    total += dsy.size(0)*dsy.size(1)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    correct += torch.eq(predicted, dsy).sum().item()
                    total += dsy.size(0)
                if cit>=niter: break
    if setTrain: net.train()
    return np.round(correct/(total+1e-6),5)


def getLo(net, dataset,setTrain=False, onCuda=False,xind=0):
    salo=[]
    totlo,n=0,0
    loss=nn.CrossEntropyLoss(reduction="none")
    net.eval()
    with torch.no_grad():
        for cit,data in enumerate(dataset):
            with tca.autocast():
                if onCuda: dsx, dsy = data[xind], data[1]
                else: dsx,dsy = data[xind].cuda(),data[1].cuda()
                outputs = net(dsx.float())
                salo.append(loss(outputs,dsy.long()).detach().cpu().numpy())
                totlo+=np.sum(salo[-1])
                n+=len(dsx)
    #salo=np.concatenate(salo)
    if setTrain: net.train()
    return totlo/n #salo,

def getRawAcc(net, dataset, onCuda=False):
    correct=[]
    net.eval()
    with torch.no_grad():
        for cit,data in enumerate(dataset):
            with tca.autocast():
                if onCuda: dsx, dsy = data[0], data[1]
                else: dsx,dsy = data[0].cuda(),data[1].cuda()
                outputs = net(dsx.float())
                _, predicted = torch.max(outputs.data, 1)
                correct.append(torch.eq(predicted, dsy).cpu().numpy().astype(np.int32))
    return np.concatenate(correct)


def getclassifier(cfg,train_dataset,val_dataset,netCl,opt,onCuda=False):
    optimizerCl = optim.SGD(netCl.parameters(), lr=opt[2], momentum=0.9, weight_decay=opt[3])
    isCeleb="celeb" in cfg["ds"][0]
    closs,teaccs,trep,loss,clr = 0,[],opt[1],nn.CrossEntropyLoss() if not isCeleb else nn.BCEWithLogitsLoss(), opt[2]
    print("Train Classifier to explain")
    scaler = tca.GradScaler()
    teAccs,trAccs=[],[]

    clAcc = lambda dataset: getAcc(netCl, dataset,  niter=1e10,onCuda=onCuda,mulCl=isCeleb)
    for epoch in range(trep):
        netCl.train()
        for i, data in enumerate(train_dataset):
          with tca.autocast():
            optimizerCl.zero_grad()
            if onCuda: dsx,dsy = data[0],data[1]
            else: dsx,dsy = data[0].cuda(),data[1].cuda()
            output = netCl(dsx.float())
            errD_real = loss(output, dsy.long() if not isCeleb else dsy.float())
            scaler.scale(errD_real).backward()
            scaler.step(optimizerCl)
            scaler.update()
            closs = 0.97 * closs + 0.03 * errD_real.item() if i > 20 else 0.8 * closs + 0.2 * errD_real.item()
        decay(opt,epoch,optimizerCl)
        netCl.eval()
        teAccs.append(clAcc(val_dataset))
        if (epoch % 4 == 0 and epoch<=13) or (epoch % 20==0 and epoch>13):
            print(epoch, np.round(np.array([closs, teAccs[-1], clAcc(train_dataset)]), 5),cfg["pr"])
    lcfg = {"testAcc": clAcc(val_dataset), "trainAcc": clAcc(train_dataset)}
    netCl.eval()
    return lcfg

def getEvCl(cfg,actShape,ncl=None):
    nNeu=np.prod(actShape[1:])
    return nn.Sequential(nn.Flatten(),nn.Linear(nNeu, cfg["num_classes"] if ncl is None else ncl))

def getEvClassifier(cfg,train_dataset,val_dataset,netCl,opt,verbose=False,onCuda=False):
    optimizerCl = optim.SGD(netCl.parameters(), lr=opt[2], momentum=0.9, weight_decay=opt[3])
    closs,teaccs,trep,loss,clr = 0,[],opt[1],nn.CrossEntropyLoss() if not cfg["binaryCl"] else nn.BCEWithLogitsLoss(), opt[2]
    if verbose==2: print("Train Ev Classifier")
    scaler = tca.GradScaler()
    clAcc = lambda dataset: getAcc(netCl, dataset,  niter=1e10,xind=2)
    netCl.train()
    for epoch in range(trep):
        for i, data in enumerate(train_dataset):
          with tca.autocast():
            optimizerCl.zero_grad()
            if onCuda:dsx, dsy = data[2], data[1]
            else: dsx, dsy = data[2].cuda(), data[1].cuda()
            output = netCl(dsx.float())
          #  print(netCl,"net")
            errD_real = loss(output, dsy.long())
            scaler.scale(errD_real).backward()
            scaler.step(optimizerCl)
            scaler.update()
            closs = 0.97 * closs + 0.03 * errD_real.item() if i > 20 else 0.8 * closs + 0.2 * errD_real.item()
        #decay(opt,epoch,optimizerCl)
        if opt[0] == "S":
            if (trep<16 and epoch==int(trep*3/4)) or ((epoch + 1) % (opt[1] // 3 + opt[1] // 10 + 2) == 0):
                for p in optimizerCl.param_groups: p['lr'] *= 0.1

        if verbose==2 and ((epoch % 4 == 0 and epoch<=13) or (epoch % 20==0 and epoch>13)):
            print(epoch, np.round(np.array([closs, clAcc(val_dataset), clAcc(train_dataset)]), 5),cfg["pr"])
    netCl.eval()
    lcfg = {"testAcc": clAcc(val_dataset), "trainAcc": clAcc(train_dataset),"testLo": getLo(netCl,val_dataset,xind=2), "trainLo": getLo(netCl,train_dataset,xind=2)}
    if verbose: print("EvCl",lcfg)
    netCl.eval()
    return lcfg


def taclAcc(net, dataset,  niter=100000,setTrain=False,onCuda=False,mulCl=False,xind=0):
    correct,total = 0,0
    net.eval()
    with torch.no_grad():
        for cit,data in enumerate(dataset):
            with tca.autocast():
                dsx = data[2] if onCuda else data[2].cuda()
                dsy=data[1] if onCuda else data[1].cuda()

                outputs = net(dsx.float())
                if mulCl:
                    result = outputs > 0.5
                    correct += (result == dsy).sum().item()
                    total += dsy.size(0)*dsy.size(1)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    correct += torch.eq(predicted, dsy).sum().item()
                    total += dsy.size(0)
                if cit>=niter: break
    if setTrain: net.train()
    return np.round(correct/(total+1e-6),5)

def getTaClassifier(cfg,train_dataset,val_dataset,netCl,opt,verbose=False,onCuda=False):
    optimizerCl = optim.SGD(netCl.parameters(), lr=opt[2], momentum=0.9, weight_decay=opt[3])
    closs,teaccs,trep,loss,clr = 0,[],opt[1],nn.CrossEntropyLoss() if not cfg["binaryCl"] else nn.BCEWithLogitsLoss(), opt[2]
    if verbose==2: print("Train Ta Classifier")
    scaler = tca.GradScaler()
 #   clAcc = lambda dataset: getAcc(netCl, dataset,  niter=1e10,xind=2)
    netCl.train()
    #from dutils import getnorm
    #norm=getnorm(cfg["ds"][0])
    for epoch in range(trep):
        #print("taTcL",epoch, np.round(np.array([closs, taclAcc(netCl, val_dataset), taclAcc(netCl, train_dataset)]), 5), cfg["pr"])
        for i, data in enumerate(train_dataset):
          with tca.autocast():
            optimizerCl.zero_grad()
            # if onCuda:dsx, dsy = data[2], data[1]
            # else: dsx, dsy = data[2].cuda(), data[1].cuda()
            dsx= data[2] if onCuda else data[2].cuda()
            dsy = data[1] if onCuda else data[1].cuda()
            # ox = data[0] if onCuda else data[0].cuda()
            # with torch.no_grad():
            #     dsy=torch.sum(ox,dim=(-1,-2))
            #     dsy=torch.argmax(dsy,dim=1)
            output = netCl(dsx.float())
          #  print(netCl,"net")
            errD_real = loss(output, dsy.long())
            scaler.scale(errD_real).backward()
            scaler.step(optimizerCl)
            scaler.update()
            closs = 0.97 * closs + 0.03 * errD_real.item() if i > 20 else 0.8 * closs + 0.2 * errD_real.item()
        #decay(opt,epoch,optimizerCl)
        if opt[0] == "S":
            if (trep<16 and epoch==int(trep*3/4)) or ((epoch + 1) % (opt[1] // 3 + opt[1] // 10 + 2) == 0):
                for p in optimizerCl.param_groups: p['lr'] *= 0.1

        if verbose==2 and ((epoch % 4 == 0 and epoch<=13) or (epoch % 20==0 and epoch>13)):
            print(epoch, np.round(np.array([closs, taclAcc(netCl,val_dataset), taclAcc(netCl,train_dataset)]), 5),cfg["pr"])
    netCl.eval()
    lcfg = {"testAcc": taclAcc(netCl,val_dataset), "trainAcc": taclAcc(netCl,train_dataset),"testLo": getLo(netCl,val_dataset,xind=2), "trainLo": getLo(netCl,train_dataset,xind=2)}
    if verbose: print("TaCl",lcfg)
    netCl.eval()
    return lcfg