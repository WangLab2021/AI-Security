import copy
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils import data
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from Global_Parameter import *

from PIL import Image
def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def dummy_pat(channel, batch, pixel):
    pat_1 = torch.rand([channel, int(pixel/2), int(pixel/2)])
    pat_2 = torch.cat((pat_1,pat_1),dim=1)
    pat_4 = torch.cat((pat_2,pat_2),dim=2)
    dummy_data = torch.unsqueeze(pat_4,dim=0)
    for i in range(int(batch/2)):
        dummy_data = torch.cat((dummy_data, dummy_data),dim=0)
    return dummy_data


def DecayClip(clip_value, local_iter, K_clip):
    C0 = clip_value
    kc = K_clip
    if DecayClip == "LD":
        return C0*(1 - kc*local_iter)
    elif DecayClip == "ED":
        return C0*np.exp((-1)*kc*local_iter)
    else:
        return C0/(1 + kc*local_iter)

def DecaySegma(rho, Epoch, e, K_segma):
    rho_0 = rho/(beta * Epoch)
    segma_0 = 1/np.sqrt(2 * rho_0)
    kc = K_segma
    if DecayMODE == "LD":
        return segma_0*(1 - kc * e)
    elif DecayMODE == "ED":
        return segma_0*np.exp((-1)*kc * e)
    else:
        return segma_0/(1 + kc * e)

def DecayFix(rho, Epoch):
    rho_0 = rho/(Epoch)
    segma = 1/np.sqrt(2*rho_0)
    return segma

# decay the noise scale and clip value (because E=N, the process equals to Adp)
def TIFS_Privacy(model, data, label, Epoch, LR, loss_fn, rho, clip_value, device, K_segma, K_clip):
    net = copy.deepcopy(model)
    num_weights = 0
    left_rho = 0

    segma = DecaySegma(rho, Epoch, 0, K_segma)
    clip_decay = DecayClip(clip_value, 0, K_clip)
    localoptimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0)

    for epoch in range(Epoch):        
        if epoch == 0:
            left_rho = rho
        else:
            left_rho = left_rho - 1/(2*(segma**2))
            segma = DecaySegma(rho, Epoch, epoch, K_segma)
            clip_decay = DecayClip(clip_value, epoch, K_clip)
        if rho <= 0:
            break
        
        print("mode:%s,epoch:%d, left privacy (rho): %f" % ("run DP-Dyn[S,σ]", epoch, left_rho))
        print("segma:%s, clip_value:%s" % (segma, clip_decay))
        
        localoptimizer.zero_grad()
        logits = net(data)
        loss_value = loss_fn(logits,label)

        dy_dx = torch.autograd.grad(loss_value, net.parameters(),retain_graph=True)
        cur_gradients = list((_.detach().clone() for _ in dy_dx))
        num_weights = len(cur_gradients)
        l2_norm = 0
        for i in range(num_weights):
            l2_norm = l2_norm + (torch.norm(cur_gradients[i],2)**2)
        l2_norm_ = torch.sqrt(l2_norm)
        #print('l2_norm', l2_norm_)
        factor = l2_norm_/clip_decay
        if factor < 1.0:
            factor = 1.0
        clip_gradients = [cur_gradients[i]/factor for i in range(num_weights)]
        GaussianNoises = [np.random.normal(loc=0.0, scale=float(segma * clip_decay),
                                                        size=clip_gradients[i].shape) for i in range(num_weights)] 
        noiseGrad = [clip_gradients[i].cpu().numpy() + GaussianNoises[i] for i in range(num_weights)]

        for p, newGrad in zip(net.parameters(), noiseGrad):
            p.grad = torch.from_numpy(newGrad).type(torch.FloatTensor).to(device)
        localoptimizer.step()

    original_dy_dx = []
    for Gi, Gg in zip(net.parameters(), model.parameters()):
        tmp_tensor = Gg - Gi
        original_dy_dx.append(tmp_tensor / LR)       
    return original_dy_dx



#decay the noise scale and fix clip value
def SP_Privacy(model, data, label, Epoch, LR, loss_fn, rho, clip_value, device, K_segma):
    net = copy.deepcopy(model)
    num_weights = 0
    left_rho = 0

    segma = DecaySegma(rho, Epoch, 0, K_segma)

    localoptimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0)

    for epoch in range(Epoch):        
        if epoch == 0:
            left_rho = rho
        else:
            left_rho = left_rho - 1/(2*(segma**2))
            segma = DecaySegma(rho, Epoch, epoch, K_segma)

        if rho <= 0:
            break
        
        print("mode:%s,epoch:%d, left privacy (rho): %f" % ("run DP-DynB", epoch, left_rho))
        print("segma:%s, clip_value:%s" % (segma, clip_value))
        
        localoptimizer.zero_grad()
        logits = net(data)
        loss_value = loss_fn(logits,label)

        dy_dx = torch.autograd.grad(loss_value, net.parameters(),retain_graph=True)
        cur_gradients = list((_.detach().clone() for _ in dy_dx))
        num_weights = len(cur_gradients)
        l2_norm = 0
        for i in range(num_weights):
            l2_norm = l2_norm + (torch.norm(cur_gradients[i],2)**2)
        l2_norm_ = torch.sqrt(l2_norm)
        #print('l2_norm', l2_norm_)
        factor = l2_norm_/clip_value
        if factor < 1.0:
            factor = 1.0
        clip_gradients = [cur_gradients[i]/factor for i in range(num_weights)]
        GaussianNoises = [np.random.normal(loc=0.0, scale=float(segma * clip_value),
                                                        size=clip_gradients[i].shape) for i in range(num_weights)] 
        noiseGrad = [clip_gradients[i].cpu().numpy() + GaussianNoises[i] for i in range(num_weights)]

        for p, newGrad in zip(net.parameters(), noiseGrad):
            p.grad = torch.from_numpy(newGrad).type(torch.FloatTensor).to(device)
        localoptimizer.step()

    original_dy_dx = []
    for Gi, Gg in zip(net.parameters(), model.parameters()):
        tmp_tensor = Gg - Gi
        original_dy_dx.append(tmp_tensor / LR)       
    return original_dy_dx


#ADP(our)
def ADP_Privacy(model, data, label, Epoch, LR, loss_fn, rho, clip_value, device, K_segma, K_clip):
    net = copy.deepcopy(model)
    num_weights = 0
    left_rho = 0

    segma = DecaySegma(rho, Epoch, 0, K_segma)
    clip_decay = DecayClip(clip_value, 0, K_clip)
    localoptimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0)

    for epoch in range(Epoch):        
        if epoch == 0:
            left_rho = rho
        else:
            left_rho = left_rho - 1/(2*(segma**2))
            segma = DecaySegma(rho, Epoch, epoch, K_segma)
            clip_decay = DecayClip(clip_value, epoch, K_clip)
        if rho <= 0:
            break
        
        print("mode:%s,epoch:%d, left privacy (rho): %f" % ("run adp", epoch, left_rho))
        print("segma:%s, clip_value:%s" % (segma, clip_decay))
        
        localoptimizer.zero_grad()
        logits = net(data)
        loss_value = loss_fn(logits,label)

        dy_dx = torch.autograd.grad(loss_value, net.parameters(),retain_graph=True)
        cur_gradients = list((_.detach().clone() for _ in dy_dx))
        num_weights = len(cur_gradients)
        l2_norm = 0
        for i in range(num_weights):
            l2_norm = l2_norm + (torch.norm(cur_gradients[i],2)**2)
        l2_norm_ = torch.sqrt(l2_norm)
        #print('l2_norm', l2_norm_)
        factor = l2_norm_/clip_decay
        if factor < 1.0:
            factor = 1.0
        clip_gradients = [cur_gradients[i]/factor for i in range(num_weights)]
        GaussianNoises = [np.random.normal(loc=0.0, scale=float(segma * clip_decay),
                                                        size=clip_gradients[i].shape) for i in range(num_weights)] 
        noiseGrad = [clip_gradients[i].cpu().numpy() + GaussianNoises[i] for i in range(num_weights)]

        for p, newGrad in zip(net.parameters(), noiseGrad):
            p.grad = torch.from_numpy(newGrad).type(torch.FloatTensor).to(device)
        localoptimizer.step()

    original_dy_dx = []
    for Gi, Gg in zip(net.parameters(), model.parameters()):
        tmp_tensor = Gg - Gi
        original_dy_dx.append(tmp_tensor / LR)       
    return original_dy_dx



#DP-SGD
def Fix_Privacy(model, data, label, Epoch, LR, loss_fn, rho, clip_value, device):
    net = copy.deepcopy(model)
    num_weights = 0
    left_rho = 0

    print('The clip value is ', clip_value)
    segma = DecayFix(rho, Epoch)
    # segma = 0.0001
    print("segma:",segma)
    localoptimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0)
    for epoch in range(Epoch):
        
        if epoch == 0:
            left_rho = rho
        else:
            left_rho = left_rho - 1/(2*(segma**2))
        if rho <= 0:
            break
        
        print("mode:%s,epoch:%d, left privacy (rho): %f" % ("run fix", epoch, left_rho))

        
        localoptimizer.zero_grad()
        logits = net(data)
        loss_value = loss_fn(logits,label)

        dy_dx = torch.autograd.grad(loss_value, net.parameters(),retain_graph=True)
        cur_gradients = list((_.detach().clone() for _ in dy_dx))
        num_weights = len(cur_gradients)
        l2_norm = 0
        for i in range(num_weights):
            l2_norm = l2_norm + (torch.norm(cur_gradients[i],2)**2)
        l2_norm_ = torch.sqrt(l2_norm)
        # print('l2_norm', l2_norm_)
        factor = l2_norm_/clip_value
        if factor < 1.0:
            factor = 1.0
        clip_gradients = [cur_gradients[i]/factor for i in range(num_weights)]
        GaussianNoises = [np.random.normal(loc=0.0, scale=float(segma * clip_value),
                                                        size=clip_gradients[i].shape) for i in range(num_weights)] 
        noiseGrad = [clip_gradients[i].cpu().numpy() + GaussianNoises[i] for i in range(num_weights)]

        for p, newGrad in zip(net.parameters(), noiseGrad):
            p.grad = torch.from_numpy(newGrad).type(torch.FloatTensor).to(device)
        localoptimizer.step()

    original_dy_dx = []
    for Gi, Gg in zip(net.parameters(), model.parameters()):
        tmp_tensor = Gg - Gi
        original_dy_dx.append(tmp_tensor / LR)       
    return original_dy_dx

def No_Privacy(model, data, label, Epoch, LR, loss_fn):
    net = copy.deepcopy(model)
    num_weights = 0
    localoptimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0)

    for epoch in range(Epoch):
        print("mode:%s,epoch:%d" % ("No privacy", epoch))
        #localoptimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
        localoptimizer.zero_grad()
        logits = net(data)
        loss_value = loss_fn(logits,label)


        dy_dx = torch.autograd.grad(loss_value, net.parameters(),retain_graph=True)
        cur_gradients = list((_.detach().clone() for _ in dy_dx))

        num_weights = len(cur_gradients)
        per_gradient = [torch.unsqueeze(cur_gradients[i], -1) for i in range(num_weights)]
        # myGrad = [torch.mean(per_gradient[i],-1) for i in range(num_weights)]
        for p,newGrad in zip(net.parameters(), cur_gradients):
            p.grad = newGrad
        localoptimizer.step()
    
    original_dy_dx = []
    for Gi, Gg in zip(net.parameters(), model.parameters()):
        tmp_tensor = Gg - Gi
        original_dy_dx.append(tmp_tensor / LR)       
    return original_dy_dx


def Vitrual_VulnerableClient(Epoch,device,model,privacy_mode,dataset = DATAMODE,rd=0,eps = 0 ,channel = 3):
    print('The privacy budget:', eps)
    torch.manual_seed(1234)
    # torch.manual_seed(1234)
    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    save_path = os.path.join(root_path, 'myresults/DLG_%s_%s_final' % (dataset, privacy_mode)).replace('\\', '/')
    rho = (eps**2)/(4*np.log(1/delta))
    print('the privacy budget is (epsilon:%s, rho:%s)'%(eps,rho))
    l2_norm_clip = 30
    lr = 0.1
    num_dummy = 1
    Iteration = 100
    num_exp = 1
    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('myresults'):
        os.mkdir('myresults')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ''' load data '''
    if dataset == 'mnist':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 768
        transform = transforms.Compose([transforms.Resize((32,32))])
        dst = datasets.MNIST('./data', download=False,transform=transform)
        idx = 1123

    elif dataset == 'cifar10':
        shape_img = (32, 32)
        num_classes = 10
        channel = 3
        hidden = 768
        dst = datasets.CIFAR10('./data', download=True)
        idx = 48850

    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 106
        channel = 3
        hidden = 768
        lfw_people = fetch_lfw_people(min_faces_per_person=14, color=True, slice_=(slice(61, 189), slice(61, 189)),
                                      resize=0.25)
        x = lfw_people.images
        y = lfw_people.target

        target_names = lfw_people.target_names
        n_classes = target_names.shape[0]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=False)
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
        # X_train = torch.transpose
        # X_train = X_train.astype('float32')
        X_train /= 255.0
        X_test /= 255.0

        x_train = torch.FloatTensor(X_train).to(device)
        x_train = x_train.transpose(2, 3).transpose(1, 2)
        y_train = torch.LongTensor(y_train).to(device)

        x_test = torch.FloatTensor(X_test).to(device)
        x_test = x_test.transpose(2, 3).transpose(1, 2)
        y_test = torch.LongTensor(y_test).to(device)

        training = data.TensorDataset(x_train, y_train)

        testing = data.TensorDataset(x_test, y_test)

        dst = training
        idx = 291


    else:
        exit('unknown dataset')

    ''' train DLG and iDLG '''
    for idx_net in range(num_exp):
        net = copy.deepcopy(model)
        net = net.to(device)
        print('running %d|%d experiment' % (idx_net, num_exp))
        idx_shuffle = np.random.permutation(len(dst))

        for method in ['DLG']:
            print('%s, Try to generate %d images' % (method, num_dummy))
            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []
            for imidx in range(num_dummy):
                imidx_list.append(idx)
                if dataset == 'lfw':
                    tmp_datum = dst[idx][0].float().to(device)
                    tmp_label = dst[idx][1].long().to(device)
                else:
                    tmp_datum = tt(dst[idx][0]).float().to(device)
                    tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                    gt_onehot_label = label_to_onehot(gt_label, num_classes=num_classes)
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)

            print('original lable:', gt_label)
            if privacy_mode == 'adp':
                update = ADP_Privacy(net, gt_data, gt_label, Epoch, lr, criterion, rho, l2_norm_clip, device, K_segma, K_clip)
                original_dy_dx = list((_.detach().clone() for _ in update))
            elif privacy_mode == 'fix':
                update = Fix_Privacy(net, gt_data, gt_label, Epoch, lr, criterion, rho, l2_norm_clip, device)
                original_dy_dx = list((_.detach().clone() for _ in update))
            elif privacy_mode == 'sp':
                update = SP_Privacy(net, gt_data, gt_label, Epoch, lr, criterion, rho, l2_norm_clip, device, K_segma)
                original_dy_dx = list((_.detach().clone() for _ in update))
            elif privacy_mode == 'tifs':
                update = TIFS_Privacy(net, gt_data, gt_label, Epoch, lr, criterion, rho, l2_norm_clip, device, K_segma, K_clip)
                original_dy_dx = list((_.detach().clone() for _ in update))
            else:
                update = No_Privacy(net, gt_data, gt_label, Epoch, lr, criterion)      
                original_dy_dx = list((_.detach().clone() for _ in update))
            
                
            dummy_data = dummy_pat(channel, num_dummy, 32).to(device).requires_grad_(True)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
            # dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)

            if method == 'DLG':
                optimizer = torch.optim.LBFGS([dummy_data, ])
                # predict the ground-truth label
                label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
                    (1,)).requires_grad_(False)

            history = []
            history_iters = []
            final = None
            cpy_final = None
            min_MSE = float('inf')
            losses = []
            mses = []
            train_iters = []

            print('lr =', lr)
            for iters in range(Iteration):

                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    dummy_loss = criterion(pred, label_pred)
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                    dummy_dy_dx = [dummy_dy_dx[i] for i in range(len(dummy_dy_dx))]
                    original_dy_dx2 = [original_dy_dx[i] for i in range(len(original_dy_dx))]
                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx2):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()
                    return grad_diff

                optimizer.step(closure)
                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

                if (iters+1) % 10 == 0 or iters ==0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                    history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                    history_iters.append(iters)
                    if mses[-1] < min_MSE:
                        min_MSE = mses[-1]
                        #print("now final mse and final:",min_MSE)
                        final = dummy_data[0].cpu().detach().numpy().copy()
                        # cpy_final = final
                        # print(final)

                    for imidx in range(num_dummy):
                        plt.figure(figsize=(12, 8))
                        plt.subplot(3, 4, 1)
                        plt.imshow(tp(gt_data[imidx].cpu()))
                        for i in range(min(len(history), 14)):
                            plt.subplot(3, 4, i + 2)
                            plt.imshow(history[i][imidx])
                            plt.title('iter=%d' % (history_iters[i]))
                            plt.axis('off')
                        if method == 'DLG':
                            plt.savefig('%s/DLG_on_%s_%s_%s.png' % (save_path, rd, label_pred.item(), Epoch))
                            plt.close()
                        elif method == 'iDLG':
                            plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()

            if method == 'DLG':
                loss_DLG = losses
                mse_DLG = min_MSE
            elif method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses

        print('imidx_list:', imidx_list)
        print('gt_label:', gt_label.detach().cpu().data.numpy(),"final MSE:",mse_DLG)
        #print("final:",final)
        print('----------------------\n\n')
        print("FINALLY")
        
        np.save('./myresults/%s_%s_%d_final'%(dataset,privacy_mode,rd),final)
        return mse_DLG

