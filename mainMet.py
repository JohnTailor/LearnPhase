from torch.utils.data import Dataset,TensorDataset
import models as clModel
from models import Classifier,getCla
from dutils import cds
import dutils
import cladec
#import benchHelp
#from benchWei import cfgs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as tca
from models import getAcc,getLo

def getActs(ds, actModel, cfg, shuffle, netCl=None, onCuda=True, factBa=1):
    acts = []
    X, y = [], []
    for i, data in enumerate(ds):
        with tca.autocast():
            dsx = data[0] if onCuda else data[0].cuda()
            X.append(data[0])
            y.append(data[1])
            mact = actModel(dsx).detach()
            #if len(mact.shape) == 4 and mact.shape[-1] > cfg["mpool"]:                mact = torch.mean(mact, dim=(-1, -2))
            #if cfg["dropA"]: mact = mact[:, ::cfg["dropA"]]
            acts.append(mact if onCuda else mact.cpu())
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    conacts = torch.cat(acts, dim=0)
    dsact = TensorDataset(X, y, conacts)
    return torch.utils.data.DataLoader(dsact, batch_size=factBa * cfg["batchSize"], shuffle=shuffle, num_workers=0), conacts.shape

def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters): m.reset_parameters()

def trainOneMet(cfg):
    #Get Data
    (trX,trY), (teX,teY)=dutils.getFullDS(cfg)   #img=train_data[0][0]    #imgy=train_data[1][0]

    trEpIter,teEpIter = len(trY)//cfg["batchSize"],len(trY) // cfg["batchSize"]
    train_data, val_data=cds(trX,trY,True,cfg["oCuda"],cfg["batchSize"]), cds(teX,teY,False,cfg["oCuda"],cfg["batchSize"])
    # Get initial classifier, trained cladec and trained Ev
    netCl = getCla(cfg).cuda()

    opt = cfg["clOpt"]
    closs, trep, clr = 0, opt[1],  opt[2]
    loss= nn.CrossEntropyLoss()
    scaler = tca.GradScaler()
    monitorClMods,actMods,claMods,evMods,taskMods=[],[],[],[],[] #Models used to track learning per layer
    saveRecImgs,saveActs=[],[]
    for ipos,ind in enumerate(cfg["layInd"]):
        print("Preparing RecLayer", ind, "all",cfg["layInd"])

        if not "S" in cfg["net"]:
            if ind < -1: ind = ind - 2
        else: ind = ind - 3
        kids = list(netCl.children())
        if "F" in cfg["net"] or "S" in cfg["net"] or "V" in cfg["net"]:
                kids = [kids[0]]+list(kids[1].children()) + kids[2:]
        if "R" in cfg["net"]: kids=netCl.convLays

        actModel = nn.Sequential(*kids[:len(kids)+ind])
        #print("Kids cut off",kids[len(kids)+ind:])
        #print("actModel",actModel)
        #print("Last 3", actModel[-4:])
        actModel.eval()
        actMods.append(actModel)
        _, actshape = getActs(val_data, actModel, cfg, False,netCl,onCuda=cfg["oCuda"])
        netDec = cladec.getClaDecNet(cfg, actshape, cfg["imCh"], ind).cuda() if cfg["trainCla"] else None
        #print(netDec)
        if cfg["obj"]=="A" and ind==0: aeDec=cladec.getClaDecNet(cfg, actshape, cfg["imCh"],ind).cuda()
        claMods.append(netDec)
        evCl=clModel.getEvCl(cfg, actshape).cuda()
        if ipos==0 or ipos==len(cfg["layInd"])-1:
            mevCl = clModel.getEvCl(cfg, actshape).cuda()
            monitorClMods.append(mevCl)
        evMods.append(evCl)
        taCl=clModel.getEvCl(cfg, actshape,(10 if not cfg["ds"][0][-2:]=="10" else 100) if cfg["trainTask"] else 3).cuda() if cfg["trainRec"][0] else None
        taskMods.append(taCl)
    if cfg["obj"] == "A":
        aeloss = nn.MSELoss()
        optimizerCl = torch.optim.Adam(list(netCl.parameters())+list(aeDec.parameters()), lr=0.0003, weight_decay=1e-5)
    else:
        if opt[0]=="S":optimizerCl = optim.SGD(netCl.parameters(), lr=opt[2], momentum=0.9, weight_decay=opt[3])
        elif opt[0]=="A":
            optimizerCl = optim.Adam(netCl.parameters(), lr=opt[2], weight_decay=opt[3])

    def getLayMet(il, actModel, netDec, evCl,taCl=None):
        #print("Updating", epoch, il)
        # get Activation
        trds, _ = getActs(train_data, actModel, cfg, True,onCuda=cfg["oCuda"])
        teds, actshape = getActs(val_data, actModel, cfg, shuffle=False,onCuda=cfg["oCuda"])
        evCl.apply(weight_reset)
        if not taCl is None: taCl.apply(weight_reset)
        if not netDec is None:
            netDec.apply(weight_reset)
        eccfg = clModel.getEvClassifier(cfg, trds, teds, evCl, cfg["evOpt"],onCuda=cfg["oCuda"],verbose=0)
        if not taCl is None:
            if cfg["trainTask"]==0:  #task (only if trainRec) task (0... color dominace for cifar 10/100 -> R,G,B; 1..data switch Fashion <-> MNIST ; C10 <->Ci100
                def conv(dataset):
                    ox,dx,dy=[],[],[]
                    for i, data in enumerate(dataset):
                        with tca.autocast():
                            #dsx = data[2] if cfg["oCuda"] else data[2].cuda()
                            cox = data[0] if cfg["oCuda"] else data[0].cuda()
                            with torch.no_grad():
                                dsy = torch.sum(cox, dim=(-1, -2))
                                dsy = torch.argmax(dsy, dim=1).cpu()
                            dx.append(data[2].cpu())
                            dy.append(dsy)
                            ox.append(data[0].cpu())
                    dsact = TensorDataset(torch.cat(ox), torch.cat(dy), torch.cat(dx))
                    return torch.utils.data.DataLoader(dsact, batch_size= cfg["batchSize"], shuffle=True, num_workers=0)
                    #return cds(np.concatenate(dx),np.concatenate(dy),True,cfg["oCuda"],cfg["batchSize"])
                tatrds,tateds=conv(trds),conv(teds)
            else:
                if "Ci100" in cfg["ds"][0]: nds=("Ci10",10)
                elif "Ci10" in cfg["ds"][0]: nds=("Ci100",100)
                elif "Fash" in cfg["ds"][0]: nds = ("MNIST", 10)
                elif "MNI" in cfg["ds"][0]:  nds = ("Fash", 10)
               # cds = [("MNIST", 10, 50000 // sc if not dummy else 1000), ("Fash", 10, 60000 // sc if not dummy else 1000), ("Ci100", 100, 50000 // sc if not dummy else 1000)]  # ,("Ci10",10,50000//sc if not dummy else 1000)]#]#("Ci10",10,50000 if not dummy else 5000),("Ci100",100,50000 if not dummy else 2000),("Ci10",10,10000 if not dummy else 5000),("Ci100",100,10000 if not dummy else 2000)]#("Ci10",10,30000 if not dummy else 5000)]#,("Ci100",100,50000 if not dummy else 2000)]#("Fash",10,60000)]
                ncfg = dict.copy(cfg)
                ncfg["ds"] = (nds[0], nds[1], ncfg["ds"][2])
                (tatrX, tatrY), (tateX, tateY) = dutils.getFullDS(ncfg)
                tatrain_data, taval_data = cds(tatrX, tatrY, True, cfg["oCuda"], cfg["batchSize"]), cds(tateX, tateY, False, cfg["oCuda"], cfg["batchSize"])
                tatrds, _ = getActs(tatrain_data, actModel, cfg, True, onCuda=cfg["oCuda"])
                tateds, _ = getActs(taval_data, actModel, cfg, shuffle=False, onCuda=cfg["oCuda"])
            tacfg = clModel.getTaClassifier(cfg, tatrds, tateds, taCl, cfg["evOpt"], onCuda=cfg["oCuda"], verbose=0)
        else:
            tacfg={"testAcc":-1, "trainAcc":-1}

        if not netDec is None:
            claTr=cladec.trainClaPlane(cfg, netDec, trds, cfg["claOpt"],onCuda=cfg["oCuda"],verbose=0)

        weightEntropy=None
        wDiff,ws=0,0
        recx,ract=None,None
        claRecLo=claTr["ClaRecLo"] if not netDec is None else 0        #print(netDec)        print(actModel)
        lossPerEle= cladec.getMSE_Ele(netDec, teds) if cfg["lossPerEle"] and not netDec is None else None
        return eccfg["testAcc"], eccfg["trainAcc"], cladec.getMSE(netDec, teds,cfg) if not netDec is None else None,claRecLo,recx,ract,-1,weightEntropy,wDiff,ws,lossPerEle,eccfg["testLo"], eccfg["trainLo"],tacfg["testAcc"], tacfg["trainAcc"],None

    teLo,trLo,teAccs, trAccs,evteAccs, evtrAccs, clateLo, clatrLo,logEps,enWei,enAct,wSpread,lossPerEle,evtrLo,evteLo,hists,tateAccs, tatrAccs,  = [],[],[],[],[],[],[],[], [], [], [],[],[],[],[],[],[],[]  #All tracked metric
    def addTrackedMetrics(giter):
        netCl.eval()
        teAccs.append(getAcc(netCl, val_data, setTrain=True, onCuda=cfg["oCuda"]))
        trAccs.append(getAcc(netCl, train_data, setTrain=True, onCuda=cfg["oCuda"]))
        teLo.append(getLo(netCl, val_data, setTrain=True, onCuda=cfg["oCuda"]))
        trLo.append(getLo(netCl, train_data, setTrain=True, onCuda=cfg["oCuda"]))
        if cfg["layTrack"]:
            mets = [getLayMet(il, actModel, netDec, evCl,taCl) for il, (actModel, netDec, evCl,taCl) in enumerate(zip(actMods, claMods, evMods,taskMods))]
            for icl,li in enumerate([evteAccs,evtrAccs,clateLo,clatrLo,saveRecImgs,saveActs,enAct,enWei,wSpread,evtrLo,evteLo]):
                li.append([m[icl] for m in mets])
            tateAccs.append([m[13] for m in mets])
            tatrAccs.append([m[14] for m in mets])
            print(" ", { " Reconstruction loss (test data)": clateLo[-1] if not claMods[-1] is None else None,"   Transfer Task Test Acc": np.round(tateAccs[-1], 3), "  Train": np.round(tatrAccs[-1], 3)})#"Test Accuracy for classifier": np.round(evteAccs[-1], 3), "  Train": np.round(evtrAccs[-1], 3),

        netCl.train()
        logEps.append(giter)#np.round(epoch + iter * 1.0 / trEpIter, 4))


    icep,giter = 0,0
    for epoch in range(trep):
        #Train Classifier
        netCl.train()
        for citer, data in enumerate(train_data):
          if cfg["evEps"][icep] == giter:
              print("---------------------------","\nUpdate:  Epoch, Iteration", epoch, citer, "  Global Iter", giter)
              addTrackedMetrics(giter)
              print("  Classifier Accuracy Test/Train", np.round(np.array([teAccs[-1], trAccs[-1]]), 5))
              icep+= 1

          with tca.autocast():
            optimizerCl.zero_grad()
            if cfg["oCuda"]: dsx, dsy = data[0], data[1]
            else: dsx, dsy = data[0].cuda(), data[1].cuda()
            output = netCl(dsx.float())
            if cfg["obj"] == "A":
                recx=aeDec(output)
                errD_real = aeloss(recx, dsx)
            else:
                errD_real = loss(output, dsy.long())
            scaler.scale(errD_real).backward()
            scaler.step(optimizerCl)
            scaler.update()
            closs = 0.97 * closs + 0.03 * errD_real.item() if citer > 20 else 0.8 * closs + 0.2 * errD_real.item()
            giter+=1


if __name__ == '__main__':
    #full
    cfg={'gid': 0, 'ds': ('Ci10', 10, 50000), 'batchSize': 128, 'dummy': False, 'enh': 0, 'aedat': 0, 'imSi': 32, 'dataCu': ('NotSet', 0), 'clMerge': None, 'remDot': (0, 0), 'binaryCl': 0, 'net': 'V8', 'clOpt': ('S', 257, 0.002, 0, 'None'), 'lrInc': 0, 'netSi': 1, 'affine': True, 'avgTrack': 0, 'layTrack': 1, 'actTrack': 0, 'initWeiSc': 1, 'fixInit': 0, 'saScale': 1, 'hingeLo': (0, 0), 'alpha': 0, 'layInd': (-1,), 'claOpt': ('A', 20, 0.003, 0.0), 'evOpt': ('S', 20, 0.01, 0.0), 'evChOpt': ('S', 16, 0.01, 0.0), 'useSig': 0, 'grad': 0, 'evEps': [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296], 'accGap': 100, 'obj': 'N', 'saveRec': 0, 'trainCla': 1, 'dropA': 0, 'weiTrack': 0, 'lossPerEle': 0, 'linCla': 1, 'mpool': 4, 'trainRec': (1,), 'trainTask': 0, 'oCuda': 0, 'loadTrain': 1, 'loadCl': 1, 'loadClaDec': 1, 'bname': 'claWeiTrainRec', 'resFolder': '../../../results/', 'datFolder': '../../../datasets/', 'nruns': 1, 'id': 0, 'bid': 0}
    print("We train a VGG-8 network on Cifar-10 and use as transfer task computing the color dominance (R,G,B) on Cifar-100")
    # dummy for testing
    # cfg={'gid': 0, 'ds': ('Ci10', 10, 1000), 'batchSize': 128, 'dummy': False, 'enh': 0, 'aedat': 0, 'imSi': 32, 'dataCu': ('NotSet', 0), 'clMerge': None, 'remDot': (0, 0), 'binaryCl': 0, 'net': 'V8', 'clOpt': ('S', 4, 0.002, 0, 'None'), 'lrInc': 0, 'netSi': 1, 'affine': True, 'avgTrack': 0, 'layTrack': 1, 'actTrack': 0, 'initWeiSc': 1, 'fixInit': 0, 'saScale': 1, 'hingeLo': (0, 0), 'alpha': 0, 'layInd': (-1,), 'claOpt': ('A', 1, 0.003, 0.0), 'evOpt': ('S', 1, 0.01, 0.0), 'evChOpt': ('S', 1, 0.01, 0.0), 'useSig': 0, 'grad': 0, 'evEps': [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296], 'accGap': 100, 'obj': 'N', 'saveRec': 0, 'trainCla': 1, 'dropA': 0, 'weiTrack': 0, 'lossPerEle': 0, 'linCla': 1, 'mpool': 5, 'trainRec': (1,), 'trainTask': 0, 'oCuda': 0, 'loadTrain': 1, 'loadCl': 1, 'loadClaDec': 1, 'bname': 'claWeiTrainRec_3', 'resFolder': '../../../results/', 'datFolder': '../../../datasets/', 'nruns': 1, 'id': 0, 'bid': 0, 'pr': {'oCuda': 0, 'remDot': (0, 0), 'bname': 'claWeiTrainRec_3', 'ds': ('Ci10', 10, 1000), 'alpha': 0, 'evEps': [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296], 'accGap': 100, 'obj': 'N', 'imSi': 32, 'clOpt': ('S', 4, 0.002, 0, 'None'), 'lrInc': 0, 'claOpt': ('A', 1, 0.003, 0.0), 'evOpt': ('S', 1, 0.01, 0.0), 'evChOpt': ('S', 1, 0.01, 0.0), 'layInd': (-1,), 'weiTrack': 0, 'actTrack': 0, 'avgTrack': 0, 'saveRec': 0, 'netSi': 1, 'affine': True, 'net': 'V8', 'linCla': 1, 'trainRec': (1,), 'trainTask': 0, 'bid': 0}, 'dt': '2023-12-15__22_42_03'}

    cfg["num_classes"] = cfg["ds"][1]
    trainOneMet(cfg)