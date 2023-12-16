import torchvision,torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset,TensorDataset
import numpy as np,sklearn
import torch.nn.functional as F
datFolder="."


gds = lambda dataset,cfg: torch.utils.data.DataLoader(TensorDataset(*[torch.from_numpy(x) for x in dataset]), batch_size=cfg["batchSize"])

def cds(trX,trY,shuffle,cuda=False,batchsize=128):
    cX, cY = torch.from_numpy(trX), torch.from_numpy(trY)
    ds = TensorDataset(cX if not cuda else cX.cuda(),cY if not cuda else cY.cuda())
    return torch.utils.data.DataLoader(ds, batch_size=batchsize, shuffle=shuffle, num_workers=0)  # cfg["batchSize"]

def getnorm(dname):
     if dname == "Ci10": return (torch.from_numpy(np.array((0.4914, 0.4822, 0.4465),np.float32).reshape(1,3,1,1)).cuda(), torch.from_numpy(np.array((0.2023, 0.1994, 0.2010),np.float32).reshape(1,3,1,1)).cuda())
     elif dname == "Ci100": return (torch.from_numpy(np.array((0.5060725 , 0.48667726, 0.4421305),np.float32).reshape(1,3,1,1)).cuda() , torch.from_numpy(np.array((0.2675421,0.25593522,0.27593908),np.float32).reshape(1,3,1,1)).cuda())
     elif dname == "Fash": return (torch.from_numpy(np.array((0.281),np.float32).reshape(1,1,1,1)).cuda() , torch.from_numpy(np.array((0.352),np.float32).reshape(1,1,1,1)).cuda())

     elif dname == "MNIST":
        return (torch.from_numpy(np.array((0.1307), np.float32).reshape(1, 1, 1, 1)).cuda(), torch.from_numpy(np.array((0.3081), np.float32).reshape(1, 1, 1, 1)).cuda())
     elif dname == "Tiny":
        return (torch.from_numpy(np.array((10.48576, 10.48576, 10.48576), np.float32).reshape(1, 3, 1, 1)).cuda(), torch.from_numpy(np.array((51.810757, 51.810757, 51.810757), np.float32).reshape(1, 3, 1, 1)).cuda())
     elif dname == "GDraw":
        return (torch.from_numpy(np.array((42.156), np.float32).reshape(1, 1, 1, 1)).cuda(), torch.from_numpy(np.array((82.773), np.float32).reshape(1, 1, 1, 1)).cuda())
        #return (42.156,82.773)

def getFullDS(cfg,ntrain=60000):
    dname=cfg["ds"][0]
    trans=transforms.Compose([transforms.ToTensor()])
    if  dname== "Ci10":
        cdat = torchvision.datasets.CIFAR10  # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #transform = transforms.Compose([transforms.ToTensor(), norm])
        cfg["imCh"] = 3
    elif dname == "Ci100":
        cdat = torchvision.datasets.CIFAR100  # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        cfg["imCh"] = 3
    elif dname == "Fash":
        cdat = torchvision.datasets.FashionMNIST
        # refu = lambda x: F.interpolate(x.unsqueeze(0), size=32).squeeze(0) #img = img - np.array([0.281])            img = img / np.array([0.352])
        # trans = transforms.Compose([transforms.ToTensor(), refu])
        cfg["imCh"] = 1

    down=True
    def loadStore(isTrain,ndat):
            trainset = cdat(root=datFolder, train=isTrain, download=down,transform=trans)
            train_dataset = torch.utils.data.DataLoader(trainset, batch_size=ndat, num_workers=4)  # cfg["batchSize"]
            ds = next(iter(train_dataset))
            X=ds[0].clone().numpy()
            print("Data stats",cdat,X.shape,np.mean(X,axis=(0,2,3)),np.std(X,axis=(0,2,3)))
            if dname == "MNIST" or dname == "Fash" or dname=="GDraw":
                from scipy import ndimage
                X=[ndimage.zoom(X[i,0],32/28) for i in range(X.shape[0])]
                X=np.stack(X,axis=0)
                X=np.expand_dims(X,axis=1)
            if cfg["imSi"]!=X.shape[2]:
                from scipy import ndimage
                fac=(cfg["imSi"]+0.51) / X.shape[2] #If zoom by 0.5 get a row and column of black pixels, with this (and the [:cfg["imSi"],:cfg["imSi"]] ) we avoid this
                if X.shape[1]>1:
                    X = [ndimage.zoom(X[i], (1,fac,fac))[:,:cfg["imSi"],:cfg["imSi"]] for i in range(X.shape[0])]
                    X = np.stack(X, axis=0)
                else:
                    X = [ndimage.zoom(X[i,0], fac)[:cfg["imSi"],:cfg["imSi"]] for i in range(X.shape[0])]
                    X = np.stack(X, axis=0)
                    X = np.expand_dims(X, axis=1)

            ds = [X, ds[1].clone().numpy()]
            ds = sklearn.utils.shuffle(*ds)  # , random_state=cfg["seed"])
            t=np.float16
            return ds[0].astype(t),ds[1].astype(np.int16)
    trX,trY=loadStore(True,ntrain)
    teX,teY=loadStore(False, ntrain)

    norm=getnorm(dname)
    trX = (trX - norm[0].cpu().numpy()) / norm[1].cpu().numpy()
    teX = (teX - norm[0].cpu().numpy()) / norm[1].cpu().numpy()
    return (trX, trY), (teX, teY)
