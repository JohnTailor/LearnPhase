import numpy as np
import torch.cuda.amp as tca
import torch
from torch import nn
from models import decay

class LinDecNet(nn.Module):
    def __init__(self, cfg,inShape,nFea,layInd=99):
        super(LinDecNet, self).__init__()
        self.channel_mult = int(64*(cfg['netSi'] if 'netSi' in cfg else 1))
        self.input_dim = np.prod(inShape[1:])
        self.fc = nn.Linear(self.input_dim, cfg["imSi"] if cfg["imCh"]<0 else cfg["imSi"]*cfg["imSi"]*cfg["imCh"] )
        self.dims= (-1,1,cfg["imSi"],1) if cfg["imCh"]<0 else (-1,cfg["imCh"],cfg["imSi"],cfg["imSi"])
        self.soft= cfg["linCla"]==2
        if self.soft and cfg["layInd"][0]!=0:
            print(" LinClaDec: Want softmax but not for last layer",cfg["layInd"])
            return -1
        if self.soft: self.soft=nn.Softmax(dim=1)


    def forward(self, x):
        x=torch.flatten(x,start_dim=1)
        if self.soft:
            x=self.soft(x)
        x=self.fc(x)
        x=torch.reshape(x,self.dims)
        return x

        # self.flat = torch.flaFlatten()
        # x=self.sig(x) #This sometimes gives better visualization, but you need to take care to standardize inputs as well
        #self.fc = nn.Linear(self.input_dim, cfg["imSi"] if cfg["imCh"]<0 else cfg["imSi"]*cfg["imSi"]//4*cfg["imCh"] )
        #self.dims= (-1,1,cfg["imSi"],1) if cfg["imCh"]<0 else (-1,cfg["imCh"],cfg["imSi"]//2,cfg["imSi"]//2)

class ClaDecNet(nn.Module):
    def __init__(self, cfg,inShape,nFea,layInd=99):
        super(ClaDecNet, self).__init__()
        self.channel_mult = int(64*(cfg['netSi'] if 'netSi' in cfg else 1))
        self.expLinLay = len(inShape) == 2 #linear layer...

        dim = cfg["imSi"] if self.expLinLay else cfg["imSi"]//int(inShape[-1]) #dimension of input
        self.input_dim=np.prod(inShape[1:])
        self.inFea=inShape[-2] if not self.expLinLay else 1
        nDeconvLay = int(np.round(np.log2(dim)-1))

        bn, bias = lambda x: nn.Identity(), True
        rel = lambda x: nn.ReLU()

        if self.inFea==1: #There is no spatial dimension (or it is one) -> use a dense layer as the first layer
            self.useDense=True
            self.fc_output_dim = max(self.input_dim, self.channel_mult) if nDeconvLay>-1 else nFea # number of input features
            self.fc = nn.Sequential(nn.Linear(self.input_dim, self.fc_output_dim), nn.ReLU(True) if nDeconvLay>-1 else nn.Identity())#, nn.BatchNorm1d(self.fc_output_dim)
        else: # The spatial extend is larger one, use a conv layer, otherwise have too many parameters
            self.fc_output_dim = inShape[1] if nDeconvLay>-1 else nFea # number of input features
            self.fc = nn.Sequential(nn.Conv2d(inShape[1],self.fc_output_dim, 3, stride=1,padding=1, bias=bias), nn.ReLU(True) if nDeconvLay>-1 else nn.Identity())  # , nn.BatchNorm1d(self.fc_output_dim)
            self.useDense=False

        if nDeconvLay>-1:
            self.deconv = []
            indim=self.fc_output_dim
            for j in range(nDeconvLay,0,-1):
                self.deconv+=[nn.ConvTranspose2d(indim,self.channel_mult * (2**(j-1)), kernel_size=4, stride=2,padding=1, bias = bias), bn(self.channel_mult * (2**(j-1))), rel(None)]
                indim=int(self.channel_mult * (2**(j-1)))
            self.deconv.append(nn.ConvTranspose2d(indim, nFea, kernel_size=4, stride=2, padding=1, bias=True))
            self.deconv = nn.Sequential(*self.deconv)
        else:
            self.deconv=nn.Identity()
        #self.sig=nn.Identity()#nn.Sigmoid() if cfg["useSig"] else nn.Identity()

    def forward(self, x):
        if self.useDense:
            x = x.view(-1, self.input_dim)
            x = self.fc(x)
            x = x.view(-1, self.fc_output_dim,self.inFea,self.inFea)
        else:
            x=self.fc(x)
        x=self.deconv(x)
        #x=self.sig(x) #This sometimes gives better visualization, but you need to take care to standardize inputs as well
        return x

def getClaDecNet(cfg,inShape,nFea,layInd=99):
            if cfg["linCla"]:
                return LinDecNet(cfg,inShape,nFea,layInd=99)
            if "V" in cfg["net"] or "R" in cfg["net"] :
                return ClaDecNet(cfg,inShape,nFea,layInd=99)#'(cfg["net"], cfg["num_classes"], cfg)

def getMSE(net, dataset,cfg):
    err,total = 0,0
    net.eval()
    loss = nn.MSELoss()
    with torch.no_grad():
        for cit,data in enumerate(dataset):
            with tca.autocast():
                dsx = data[0].cuda()
                #if dsx.shape[-1]>cfg["mpool"] and len(dsx.shape)==4:                    dsx=torch.max_pool2d(dsx,2,2)
                dact = data[2].cuda()
                total += data[0].shape[0]
                outputs = net(dact.float())
                recloss = loss(outputs, dsx)
                err += recloss
    return  err.item()#np.float(100*err.cpu().numpy())/total #100 only since otherwise always have 0.00

def getMSE_Ele(net, dataset):
    err,total = None,0
    err,y=[],[]
    net.eval()
    loss = nn.MSELoss(reduction='none')
    for cit,data in enumerate(dataset):
        with tca.autocast():
            with torch.no_grad():
                dsx = data[0].cuda()
                dact = data[2].cuda()
                total += data[0].shape[0]
                outputs = net(dact.float())
                lo=loss(outputs, dsx)
                # recloss = torch.sum(lo,dim=0)/100.0
                # if err is None: err = recloss
                # else: err+=recloss
                err.append(lo.reshape(len(lo),-1).cpu().numpy())
                y.append(data[1])


    return np.concatenate(err,axis=0)#err.cpu().numpy() /total*100.0 #100 only since otherwise always have 0.00 #torch.mean(err,dims=1)


def trainClaPlane(cfg, netDec, train_dataset, opt, verbose=False,onCuda=False):
    closs,cclloss,crloss, teaccs, trep, clloss, clr = 0,0,0, [], opt[1], nn.CrossEntropyLoss(), opt[2]
    scaler = tca.GradScaler()
    netDec.train()
    optimizerCl = torch.optim.Adam(netDec.parameters(), lr=0.0003, weight_decay=1e-5)
    aeloss=nn.MSELoss()
    ulo=lambda closs,totloss,i: 0.97 * closs + 0.03 * totloss.item() if epoch > 20 else 0.8 * closs + 0.2 * totloss.item()
    for epoch in range(trep):
        for i, data in enumerate(train_dataset):
            with tca.autocast():
                optimizerCl.zero_grad()
                if onCuda: dsx,  dsact = data[0], data[2]
                else: dsx,  dsact = data[0].cuda(), data[2].cuda()
                output = netDec(dsact.float())
                recloss = aeloss(output,dsx)
                scaler.scale(recloss).backward()
                scaler.step(optimizerCl)
                scaler.update()
                crloss=ulo(crloss,recloss,epoch)
        decay(opt, epoch, optimizerCl)
        if verbose==2 and ((epoch % 2 == 0 and epoch <= 10) or (epoch % 10 == 0 and epoch > 10)):
            print(epoch, np.round(np.array([crloss]), 5),cfg["pr"])
    lcfg = {"ClaRecLo":crloss}
    if verbose: print(lcfg)
    netDec.eval()
    return lcfg



def getClaDec(cfg,netCl,netDec,train_dataset,opt,moveCuda=False):
    alpha=cfg["alpha"]
    closs,cclloss,crloss, teaccs, trep, clloss, clr = 0,0,0, [], opt[1], nn.CrossEntropyLoss(), opt[2]
    print("Train CLaDec")
    scaler = tca.GradScaler()
    netDec.train()
    optimizerCl = torch.optim.Adam(netDec.parameters(), lr=0.0003, weight_decay=1e-5)
    aeloss=nn.MSELoss()
    ulo=lambda closs,totloss,i: 0.97 * closs + 0.03 * totloss.item() if epoch > 20 else 0.8 * closs + 0.2 * totloss.item()
    for epoch in range(trep):
        for i, data in enumerate(train_dataset):
            with tca.autocast():
                optimizerCl.zero_grad()
                if moveCuda: dsx, dsy,dsact = data[0].cuda(), data[1].cuda(), data[2].cuda()
                else: dsx, dsy, dsact = data[0], data[1], data[2]
                output = netDec(dsact.float())
                recloss = aeloss(output,dsx)
                claloss=clloss(netCl(output),dsy.long())
                totloss=(1-alpha)*recloss+alpha*claloss
                scaler.scale(totloss).backward()
                scaler.step(optimizerCl)
                scaler.update()
                closs,cclloss,crloss=ulo(closs,totloss,epoch),ulo(cclloss,claloss,epoch),ulo(crloss,recloss,epoch)

        decay(opt, epoch, optimizerCl)
        if (epoch % 2 == 0 and epoch <= 10) or (epoch % 10 == 0 and epoch > 10): print(epoch, np.round(np.array([closs,crloss,cclloss]), 5),cfg["pr"])

    lcfg = {"ClaTotLo": closs,"ClaClaLo":cclloss,"ClaRecLo":crloss}
    netDec.eval()
    return lcfg