#Step3
#在每个CLIENT进行本地训练时初始化，用来计算CLIENT本地训练过程的隐私消耗
from Global_Parameter import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from dlgAttack import DLA
from torchvision import models, datasets, transforms
#k_instance = EPS_instance(data_ind,model,Epoch,BSize,eps)
class EPS_instance:
    def __init__(self,data_ind,model,Epoch,BSize,eps,device):
        self.ori_data = data_ind
        np.random.shuffle(self.ori_data)
        splitNum = np.ceil(len(self.ori_data) / BSize)
        self.data_ind = np.array_split(np.asarray(self.ori_data), splitNum, 0)
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=learning_rate)
        self.Epoch = Epoch
        print("Epoch:",self.Epoch)
        self.BSize = BSize
        print("BSize:",self.BSize)
        self.rho = (eps**2)/(4*np.log(1/delta))
        self.device = device
    def Decay(self,local_iter):
        if DecayClip == "LD":
            C0 = fix_Clip
            kc = 0.5
            return C0*(1 - kc*local_iter)
        elif DecayClip == "ED":
            C0 = fix_Clip
            kc = 0.01
            return C0*np.exp((-1)*kc*local_iter)
        else:
            C0 = fix_Clip
            kc = 0.5
            return C0/(1 + kc*local_iter)
    def DecayADP(self,local_iter,c0):
        C0 = c0
        if DecayClip == "LD":
            
            kc = 0.5
            return C0*(1 - kc*local_iter)
        elif DecayClip == "ED":
            
            kc = 0.01
            return C0*np.exp((-1)*kc*local_iter)
        else:
            
            kc = 0.5
            return C0/(1 + kc*local_iter)
    def Decay2(self,local_iter):
        if DecayClip == "LD":
            C0 = fix_Clip
            kc = 0.5
            return C0*(1 - kc*local_iter)
        elif DecayClip == "ED":
            C0 = 10
            kc = 0.01
            return C0*np.exp((-1)*kc*local_iter)
        else:
            C0 = fix_Clip
            kc = 0.5
            return C0/(1 + kc*local_iter)
    def DecayBudget(self,e,K_segma):
        rho_0 = self.rho/(self.Epoch)
        C0 = 1/np.sqrt(2*rho_0)
        if DecayMODE == "LD":
            kc = K_segma
            return C0*(1 - kc * e)
        elif DecayMODE == "ED":
            kc = K_segma
            return C0*np.exp((-1)*kc * e)
        else:
            kc = K_segma
            return C0/(1 + kc * e)
    def DecayTIFS(self,e,segma0,K_segma):
        C0 = segma0
        
        if DecayMODE == "LD":
            kc = K_segma
            return C0*(1 - kc * e)
        elif DecayMODE == "ED":
            kc = K_segma
            return C0*np.exp((-1)*kc * e)
        else:
            kc = K_segma
            return C0/(1 + kc * e)
    def DecaySP(self,e,segma,K_segma):
        C0 = segma
        if DecayMODE == "LD":
            kc = K_segma
            return C0*(1 - kc * e)
        elif DecayMODE == "ED":
            kc = K_segma
            return C0*np.exp((-1)*kc * e)
        else:
            kc = K_segma
            return C0/(1 + kc * e)    
    def DecayTIFSC(self,e,c0):
        C0 = c0
        k = 0.01
        if DecayMODE == "LD":
            kc = k
            return C0*(1 - kc * e)
        elif DecayMODE == "ED":
            kc = k
            return C0*np.exp((-1)*kc * e)
        else:
            kc = k
            return C0/(1 + kc * e)
    def DecayFix(self):
        rho_0 = self.rho/(self.Epoch)
        C0 = 1/np.sqrt(2*rho_0)
        return C0
    def runFix(self,data_set_asarray,label_set_asarray,loss_fn,DecayClip,Attack,device,xvali,yvali):
        num_weights = 0
        self.model.train()
        for epoch in range(self.Epoch):
            print("mode:%s,epoch:%d" % ("run fix", epoch))
            if epoch == 0:
                rho = self.rho
            else:
                rho = rho - 1/(2*(segma**2))
                if rho <= 0:
                    return self.model,num_weights
            
            segma = self.DecayFix()
            print("rho:%s, segma:%s" % (rho, segma))
            running_loss = 0.0
            #print(len(self.data_ind))
            for local_iter in range(len(self.data_ind)):
                batch_ind=self.data_ind[local_iter]
                batch_instance = [batch_ind[i] for i in range(len(batch_ind))]
                x = data_set_asarray[[int(j) for j in batch_instance]]
                y = label_set_asarray[[int(j) for j in batch_instance]]
                x, y = x.to(device), y.to(device)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss_value = loss_fn(logits,y)
                myGrad = []
                Ct = fix_Clip
                for idx in range(len(logits)):
                    with torch.no_grad():
                        running_loss += loss_value[idx]
                    dy_dx = torch.autograd.grad(loss_value[idx], self.model.parameters(),retain_graph=True)
                    cur_gradients = list((_.detach().clone() for _ in dy_dx))
                    num_weights = len(cur_gradients)
                    l2_norm = 0.0
                    for i in range(num_weights):
                        l2_norm = l2_norm + (torch.norm(cur_gradients[i],2)**2)
                    l2_norm_ = torch.sqrt(l2_norm)
                    print('l2_norm', l2_norm_)
                    factor = l2_norm_/Ct
                    if factor < 1.0:
                        factor = 1.0
                    clip_gradients = [cur_gradients[i]/factor for i in range(num_weights)]

                    if idx==0:
                        per_gradient = [torch.unsqueeze(clip_gradients[i], -1) for i in range(num_weights)]
                    else:
                        per_gradient = [torch.cat((per_gradient[i], torch.unsqueeze(clip_gradients[i], -1)),-1) for i in range(num_weights)]

                myGrad = [torch.mean(per_gradient[i], -1) for i in range(num_weights)]
                GaussianNoises = [
                    1.0 / x.shape[0] * np.random.normal(loc=0.0, scale=float(segma * Ct),
                                                        size=myGrad[i].shape) for i in
                    range(num_weights)]  # layerwise gaussian noise
                noiseGrad = [myGrad[i].cpu().numpy() + GaussianNoises[i] for i in
                             range(num_weights)]  # add gaussian noise

                for p, newGrad in zip(self.model.parameters(), noiseGrad):
                    p.grad = torch.from_numpy(newGrad).type(torch.FloatTensor).to(device)
                self.optimizer.step()
                if local_iter % 100==0 and local_iter != 0:
                    with torch.no_grad():
                        # preds = self.model(xvali)
                        # preds = torch.argmax(preds, dim=1)
                        # accuracy = torchmetrics.Accuracy().to('cuda')
                        #print("local iter:%d loss:%.6f"%(local_iter,running_loss/100))
                        # acc = accuracy(preds, yvali)
                        # print("acc:%.6f" % acc)
                        running_loss = 0.0
                if local_iter == 999:
                    print('\n')
                # print("====更新后======")
            print("loss:%.6f"%(running_loss/len(self.data_ind)))
            np.random.shuffle(self.ori_data)
            splitNum = np.ceil(len(self.ori_data) / self.BSize)
            self.data_ind = np.array_split(np.asarray(self.ori_data), splitNum, 0) 
        return num_weights
    def runOurs(self,data_set_asarray,label_set_asarray,loss_fn,DecayClip,Attack,device,xvali,yvali):
        num_weights = 0
        self.model.train()
        for epoch in range(self.Epoch):
            print("mode:%s,epoch:%d" % ("ours but gradient clip", epoch))
            if epoch == 0:
                rho = self.rho
                print("rho:",rho)
            else:
                rho = rho - 1/(2*(segma**2))
                print("rho:",rho)
                if rho <= 0:
                    return self.model,num_weights
            segma = self.DecayBudget(epoch,K_segma)
            print("segma~:",segma)
            running_loss = 0.0
            for local_iter in range(len(self.data_ind)):
                batch_ind=self.data_ind[local_iter]
                batch_instance = [batch_ind[i] for i in range(len(batch_ind))]
                x = data_set_asarray[[int(j) for j in batch_instance]]
                y = label_set_asarray[[int(j) for j in batch_instance]]
                x, y = x.to(device), y.to(device)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss_value = loss_fn(logits,y)
                myGrad = []
                Ct = self.Decay(local_iter)
                for idx in range(len(logits)):
                    with torch.no_grad():
                        running_loss += loss_value[idx]
                    dy_dx = torch.autograd.grad(loss_value[idx], self.model.parameters(),retain_graph=True)
                    cur_gradients = list((_.detach().clone() for _ in dy_dx))
                    num_weights = len(cur_gradients)
                    l2_norm = 0.0
                    for i in range(num_weights):
                        l2_norm = l2_norm + (torch.norm(cur_gradients[i],2)**2)
                    l2_norm_ = torch.sqrt(l2_norm)
                    factor = l2_norm_/Ct
                    if factor < 1.0:
                        factor = 1.0
                    clip_gradients = [cur_gradients[i]/factor for i in range(num_weights)]

                    if idx==0:
                        per_gradient = [torch.unsqueeze(clip_gradients[i], -1) for i in range(num_weights)]
                    else:
                        per_gradient = [torch.cat((per_gradient[i], torch.unsqueeze(clip_gradients[i], -1)),-1) for i in range(num_weights)]

                myGrad = [torch.mean(per_gradient[i], -1) for i in range(num_weights)]
                GaussianNoises = [
                    1.0 / x.shape[0] * np.random.normal(loc=0.0, scale=float(segma * Ct),
                                                        size=myGrad[i].shape) for i in
                    range(num_weights)]  # layerwise gaussian noise
                noiseGrad = [myGrad[i].cpu().numpy() + GaussianNoises[i] for i in
                             range(num_weights)]  # add gaussian noise

                for p, newGrad in zip(self.model.parameters(), noiseGrad):
                    p.grad = torch.from_numpy(newGrad).type(torch.FloatTensor).to(device)
                self.optimizer.step()
                if local_iter % 100==0 and local_iter != 0:
                    with torch.no_grad():
                        # preds = self.model(xvali)
                        # preds = torch.argmax(preds, dim=1)
                        # accuracy = torchmetrics.Accuracy().to('cuda')
                        #print("local iter:%d loss:%.6f"%(local_iter,running_loss/100))
                        # acc = accuracy(preds, yvali)
                        # print("acc:%.6f" % acc)
                        running_loss = 0.0
                if local_iter == 999:
                    print('\n')
                # print("====更新后======")
            print("loss:%.6f"%(running_loss/len(self.data_ind)))
            np.random.shuffle(self.ori_data)
            splitNum = np.ceil(len(self.ori_data) / self.BSize)
            self.data_ind = np.array_split(np.asarray(self.ori_data), splitNum, 0)
        return num_weights
    def run_NoPrvy_NoClip(self,data_set_asarray,label_set_asarray,loss_fn,DecayClip,Attack,device,xvali,yvali):
        num_weights = 0
        self.model.train()
        for epoch in range(self.Epoch):
            print("mode:%s,epoch:%d" % ("No privacy", epoch))
            running_loss = 0.0
            for local_iter in range(len(self.data_ind)):
                batch_ind=self.data_ind[local_iter]
                batch_instance = [batch_ind[i] for i in range(len(batch_ind))]
                x = data_set_asarray[[int(j) for j in batch_instance]]
                y = label_set_asarray[[int(j) for j in batch_instance]]
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss_value = loss_fn(logits,y)
                myGrad = None
                for idx in range(len(logits)):
                    with torch.no_grad():
                        running_loss += loss_value[idx]
                    dy_dx = torch.autograd.grad(loss_value[idx], self.model.parameters(),retain_graph=True)
                    cur_gradients = list((_.detach().clone() for _ in dy_dx))
                    num_weights = len(cur_gradients)
                    if idx==0:
                        per_gradient = [torch.unsqueeze(cur_gradients[i], -1) for i in range(num_weights)]
                    else:
                        per_gradient = [torch.cat((per_gradient[i], torch.unsqueeze(cur_gradients[i], -1)),-1) for i in range(num_weights)]
                myGrad = [torch.mean(per_gradient[i],-1) for i in range(num_weights)]
                for p,newGrad in zip(self.model.parameters(),myGrad):
                    p.grad = newGrad
                self.optimizer.step()
                if local_iter % 100==0 and local_iter != 0:
                    with torch.no_grad():
                        # preds = self.model(xvali)
                        # preds = torch.argmax(preds, dim=1)
                        # accuracy = torchmetrics.Accuracy().to('cuda')
                        print("local iter:%d loss:%.6f"%(local_iter,running_loss/100))
                        # acc = accuracy(preds, yvali)
                        # print("acc:%.6f" % acc)
                        running_loss = 0.0
                if local_iter == (50000/total_client_num-1):
                    print("\n")
                # print("====更新后======")
            print("loss:%.6f"%(running_loss/len(self.data_ind)))
            np.random.shuffle(self.ori_data)
            splitNum = np.ceil(len(self.ori_data) / self.BSize)
            self.data_ind = np.array_split(np.asarray(self.ori_data), splitNum, 0) 
        return num_weights
    def runOurs_butLayerClip(self,data_set_asarray,label_set_asarray,loss_fn,DecayClip,Attack,device,xvali,yvali):
        num_weights = 0
        self.model.train()
        for epoch in range(self.Epoch):
            print("mode:%s,epoch:%d"%("ours but layer clip",epoch))
            if epoch == 0:
                rho = self.rho
            else:
                rho = rho - 0.5*(segma**2)
                if rho <= 0:
                    return self.model,num_weights
            segma = self.DecayBudget(epoch,K_segma)
            print("segma::",segma)
            running_loss = 0.0
            for local_iter in range(len(self.data_ind)):
                batch_ind=self.data_ind[local_iter]
                torch.cuda.normal()
                batch_instance = [batch_ind[i] for i in range(self.BSize)]
                x = data_set_asarray[[int(j) for j in batch_instance]]
                y = label_set_asarray[[int(j) for j in batch_instance]]
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss_value = loss_fn(logits,y)
                myGrad = None
                Ct = self.Decay(local_iter)
                for idx in range(len(logits)):
                    with torch.no_grad():
                        running_loss += loss_value[idx]
                    dy_dx = torch.autograd.grad(loss_value[idx], self.model.parameters(),retain_graph=True)
                    cur_gradients = list((_.detach().clone() for _ in dy_dx))
                    num_weights = len(cur_gradients)
                    if idx==0:
                        per_gradient = [torch.unsqueeze(cur_gradients[i], -1) for i in range(num_weights)]
                    else:
                        per_gradient = [torch.cat((per_gradient[i], torch.unsqueeze(cur_gradients[i], -1)),-1) for i in range(num_weights)]

                norms = [torch.sqrt(torch.sum(((per_gradient[i]) ** 2),axis=tuple(range(per_gradient[i].ndim)[:-1]),keepdims=True)) for i in range(num_weights)]
                factors = [norms[i] / l2_norm_clip for i in range(num_weights)]
                for i in range(num_weights):
                    factors[i][factors[i]<1.0]=1.0
                clipped_gradients = [per_gradient[i] / factors[i] for i in range(num_weights)]  # do clipping
                myGrad = [torch.mean(clipped_gradients[i], -1) for i in range(num_weights)]
                GaussianNoises = [
                    1.0 / x.shape[0] * np.random.normal(loc=0.0, scale=float(segma * Ct),
                                                        size=myGrad[i].shape) for i in
                    range(num_weights)]  # layerwise gaussian noise
                noiseGrad = [myGrad[i].cpu().numpy() + GaussianNoises[i] for i in range(num_weights)]  # add gaussian noise

                for p,newGrad in zip(self.model.parameters(),noiseGrad):
                    p.grad = torch.from_numpy(newGrad).type(torch.FloatTensor).to(device)
                self.optimizer.step()
                if local_iter % 80==0 and local_iter != 0:
                    with torch.no_grad():
                        # preds = self.model(xvali)
                        # preds = torch.argmax(preds, dim=1)
                        # accuracy = torchmetrics.Accuracy().to('cuda')
                        print("local iter:%d loss:%.6f"%(local_iter,running_loss/80))
                        if local_iter == 240:
                            print("\n")
                        # acc = accuracy(preds, yvali)
                        # print("acc:%.6f" % acc)
                        running_loss = 0.0
                # print("====更新后======")
        return num_weights
    def SP(self,data_set_asarray,label_set_asarray,loss_fn,DecayClip,Attack,device,test_set_loader,c0,segma0,k0):
        num_weights = 0
        self.model.train()
        for epoch in range(self.Epoch):
            total_train = 0
            correct_train = 0
            print("mode:%s,epoch:%d" % ("S&P", epoch))
            segma = self.DecaySP(epoch,segma0,k0)
            print("segma~:",segma)
            running_loss = 0.0
            for local_iter in range(len(self.data_ind)):
                batch_ind=self.data_ind[local_iter]
                batch_instance = [batch_ind[i] for i in range(len(batch_ind))]
                x = data_set_asarray[[int(j) for j in batch_instance]]
                y = label_set_asarray[[int(j) for j in batch_instance]]
                x, y = x.to(device), y.to(device)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss_value = loss_fn(logits,y)
                with torch.no_grad():
                    _, predictions = torch.max(logits, 1)
                    total_train += y.size(0)
                    correct_train += (predictions == y).sum().item()
                myGrad = []
                Ct = c0
                l2 = 0
                for idx in range(len(logits)):
                    with torch.no_grad():
                        running_loss += loss_value[idx]
                    dy_dx = torch.autograd.grad(loss_value[idx], self.model.parameters(),retain_graph=True)
                    cur_gradients = list((_.detach().clone() for _ in dy_dx))
                    num_weights = len(cur_gradients)
                    l2_norm = 0.0
                    for i in range(num_weights):
                        l2_norm = l2_norm + (torch.norm(cur_gradients[i],2)**2)
                    l2_norm_ = torch.sqrt(l2_norm)
                    l2 += l2_norm_
                    factor = l2_norm_/Ct
                    if factor < 1.0:
                        factor = 1.0
                    clip_gradients = [cur_gradients[i]/factor for i in range(num_weights)]

                    if idx==0:
                        per_gradient = [torch.unsqueeze(clip_gradients[i], -1) for i in range(num_weights)]
                    else:
                        per_gradient = [torch.cat((per_gradient[i], torch.unsqueeze(clip_gradients[i], -1)),-1) for i in range(num_weights)]

                print("l2",l2)
                myGrad = [torch.mean(per_gradient[i], -1) for i in range(num_weights)]
                GaussianNoises = [
                    1.0 / x.shape[0] * np.random.normal(loc=0.0, scale=float(segma * Ct),
                                                        size=myGrad[i].shape) for i in
                    range(num_weights)]  # layerwise gaussian noise
                noiseGrad = [myGrad[i].cpu().numpy() + GaussianNoises[i] for i in
                             range(num_weights)]  # add gaussian noise

                for p, newGrad in zip(self.model.parameters(), noiseGrad):
                    p.grad = torch.from_numpy(newGrad).type(torch.FloatTensor).to(device)
                self.optimizer.step()
                if local_iter % 100==0 and local_iter != 0:
                    with torch.no_grad():
                        # preds = self.model(xvali)
                        # preds = torch.argmax(preds, dim=1)
                        # accuracy = torchmetrics.Accuracy().to('cuda')
                        #print("local iter:%d loss:%.6f"%(local_iter,running_loss/100))
                        # acc = accuracy(preds, yvali)
                        # print("acc:%.6f" % acc)
                        running_loss = 0.0
                if local_iter == 999:
                    print('\n')
                # print("====更新后======")
            print("loss:%.6f"%(running_loss/len(self.data_ind)))
            print("train acc:%.2f"%(correct_train/total_train))
            total = 0
            correct = 0
            with torch.no_grad():
                for data in test_set_loader:
                    images, labels = data
                    images= images.to(device)
                    labels = labels.to(device)
                    _, predictions = torch.max(self.model(images), 1)
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()
            print("test acc:%.2f"%(correct/total))
            np.random.shuffle(self.ori_data)
            splitNum = np.ceil(len(self.ori_data) / self.BSize)
            self.data_ind = np.array_split(np.asarray(self.ori_data), splitNum, 0)
        return num_weights
    def ADP(self,data_set_asarray,label_set_asarray,loss_fn,DecayClip,Attack,device,test_set_loader,c0,segma0,k0):
        num_weights = 0
        self.model.train()
        for epoch in range(self.Epoch):
            total_train = 0
            correct_train = 0
            print("mode:%s,epoch:%d" % ("ADP", epoch))
            segma = self.DecaySP(epoch,segma0,k0)
            print("segma~:",segma)
            running_loss = 0.0
            for local_iter in range(len(self.data_ind)):
                batch_ind=self.data_ind[local_iter]
                batch_instance = [batch_ind[i] for i in range(len(batch_ind))]
                x = data_set_asarray[[int(j) for j in batch_instance]]
                y = label_set_asarray[[int(j) for j in batch_instance]]
                x, y = x.to(device), y.to(device)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss_value = loss_fn(logits,y)
                with torch.no_grad():
                    _, predictions = torch.max(logits, 1)
                    total_train += y.size(0)
                    correct_train += (predictions == y).sum().item()
                myGrad = []
                Ct = self.DecayADP(local_iter,c0)
                print("Ct",Ct)
                l2 = 0
                for idx in range(len(logits)):
                    with torch.no_grad():
                        running_loss += loss_value[idx]
                    dy_dx = torch.autograd.grad(loss_value[idx], self.model.parameters(),retain_graph=True)
                    cur_gradients = list((_.detach().clone() for _ in dy_dx))
                    num_weights = len(cur_gradients)
                    l2_norm = 0.0
                    for i in range(num_weights):
                        l2_norm = l2_norm + (torch.norm(cur_gradients[i],2)**2)
                    l2_norm_ = torch.sqrt(l2_norm)
                    l2 += l2_norm_
                    factor = l2_norm_/Ct
                    if factor < 1.0:
                        factor = 1.0
                    clip_gradients = [cur_gradients[i]/factor for i in range(num_weights)]

                    if idx==0:
                        per_gradient = [torch.unsqueeze(clip_gradients[i], -1) for i in range(num_weights)]
                    else:
                        per_gradient = [torch.cat((per_gradient[i], torch.unsqueeze(clip_gradients[i], -1)),-1) for i in range(num_weights)]

                print("l2:",l2)
                myGrad = [torch.mean(per_gradient[i], -1) for i in range(num_weights)]
                GaussianNoises = [
                    1.0 / x.shape[0] * np.random.normal(loc=0.0, scale=float(segma * Ct),
                                                        size=myGrad[i].shape) for i in
                    range(num_weights)]  # layerwise gaussian noise
                noiseGrad = [myGrad[i].cpu().numpy() + GaussianNoises[i] for i in
                             range(num_weights)]  # add gaussian noise

                for p, newGrad in zip(self.model.parameters(), noiseGrad):
                    p.grad = torch.from_numpy(newGrad).type(torch.FloatTensor).to(device)
                self.optimizer.step()
                if local_iter % 100==0 and local_iter != 0:
                    with torch.no_grad():
                        # preds = self.model(xvali)
                        # preds = torch.argmax(preds, dim=1)
                        # accuracy = torchmetrics.Accuracy().to('cuda')
                        #print("local iter:%d loss:%.6f"%(local_iter,running_loss/100))
                        # acc = accuracy(preds, yvali)
                        # print("acc:%.6f" % acc)
                        running_loss = 0.0
                if local_iter == 999:
                    print('\n')
                # print("====更新后======")
            print("loss:%.6f"%(running_loss/len(self.data_ind)))
            print("train acc:%.2f"%(correct_train/total_train))
            total = 0
            correct = 0
            with torch.no_grad():
                for data in test_set_loader:
                    images, labels = data
                    images= images.to(device)
                    labels = labels.to(device)
                    _, predictions = torch.max(self.model(images), 1)
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()
            print("test acc:%.2f"%(correct/total))
            np.random.shuffle(self.ori_data)
            splitNum = np.ceil(len(self.ori_data) / self.BSize)
            self.data_ind = np.array_split(np.asarray(self.ori_data), splitNum, 0)
        return num_weights
    def TIFS(self,data_set_asarray,label_set_asarray,loss_fn,DecayClip,Attack,device,test_set_loader,c0,segma0,k0):
        num_weights = 0
        self.model.train()
        cnt = 0
        for epoch in range(self.Epoch):
            total_train = 0
            correct_train = 0
            print("mode:%s,epoch:%d" % ("TIFS", epoch))
            running_loss = 0.0
            for local_iter in range(len(self.data_ind)):
                segma = self.DecayTIFS(cnt,segma0,k0)
                batch_ind=self.data_ind[local_iter]
                batch_instance = [batch_ind[i] for i in range(len(batch_ind))]
                x = data_set_asarray[[int(j) for j in batch_instance]]
                y = label_set_asarray[[int(j) for j in batch_instance]]
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss_value = loss_fn(logits,y)
                with torch.no_grad():
                    _, predictions = torch.max(logits, 1)
                    total_train += y.size(0)
                    correct_train += (predictions == y).sum().item()
                myGrad = []
                Ct = self.DecayTIFSC(cnt,c0)
                cnt += 1
                if local_iter%100==0:
                    print("segma~:",segma)
                    print("C~:",Ct)
                for idx in range(len(logits)):
                    with torch.no_grad():
                        running_loss += loss_value[idx]
                    dy_dx = torch.autograd.grad(loss_value[idx], self.model.parameters(),retain_graph=True)
                    cur_gradients = list((_.detach().clone() for _ in dy_dx))
                    num_weights = len(cur_gradients)
                    l2_norm = 0.0
                    for i in range(num_weights):
                        l2_norm = l2_norm + (torch.norm(cur_gradients[i],2)**2)
                    l2_norm_ = torch.sqrt(l2_norm)
                    factor = l2_norm_/Ct
                    if factor < 1.0:
                        factor = 1.0
                    clip_gradients = [cur_gradients[i]/factor for i in range(num_weights)]

                    if idx==0:
                        per_gradient = [torch.unsqueeze(clip_gradients[i], -1) for i in range(num_weights)]
                    else:
                        per_gradient = [torch.cat((per_gradient[i], torch.unsqueeze(clip_gradients[i], -1)),-1) for i in range(num_weights)]

                myGrad = [torch.mean(per_gradient[i], -1) for i in range(num_weights)]
                GaussianNoises = [
                    1.0 / x.shape[0] * np.random.normal(loc=0.0, scale=float(segma * Ct),
                                                        size=myGrad[i].shape) for i in
                    range(num_weights)]  # layerwise gaussian noise
                noiseGrad = [myGrad[i].cpu().numpy() + GaussianNoises[i] for i in
                             range(num_weights)]  # add gaussian noise

                for p, newGrad in zip(self.model.parameters(), noiseGrad):
                    p.grad = torch.from_numpy(newGrad).type(torch.FloatTensor).to(device)
                self.optimizer.step()
                if local_iter % 100==0 and local_iter != 0:
                    with torch.no_grad():
                        # preds = self.model(xvali)
                        # preds = torch.argmax(preds, dim=1)
                        # accuracy = torchmetrics.Accuracy().to('cuda')
                        #print("local iter:%d loss:%.6f"%(local_iter,running_loss/100))
                        # acc = accuracy(preds, yvali)
                        # print("acc:%.6f" % acc)
                        running_loss = 0.0
                if local_iter == 999:
                    print('\n')
                # print("====更新后======")
            print("loss:%.6f"%(running_loss/len(self.data_ind)))
            print("train acc:%.2f"%(correct_train/total_train))
            total = 0
            correct = 0
            with torch.no_grad():
                for data in test_set_loader:
                    images, labels = data
                    images= images.to(device)
                    labels = labels.to(device)
                    _, predictions = torch.max(self.model(images), 1)
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()
            print("test acc:%.2f"%(correct/total))
            np.random.shuffle(self.ori_data)
            splitNum = np.ceil(len(self.ori_data) / self.BSize)
            self.data_ind = np.array_split(np.asarray(self.ori_data), splitNum, 0)
        return num_weights
    