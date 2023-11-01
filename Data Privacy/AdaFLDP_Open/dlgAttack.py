import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.functional as F
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image

import copy
class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs  # img paths
        self.labs = labs  # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab


def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def DLA(model,gt_data,gt_label,update,device,num_classes,channel):
    model.eval()
    gt_label.view(1,*gt_label.size())
    dataset = '1'
    torch.manual_seed(57)
    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    save_path = os.path.join(root_path, 'results/DLG_%s' % dataset).replace('\\', '/')

    lr = 1.0
    num_dummy = 1
    Iteration = 3
    num_exp = 1




    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    ''' train DLG and iDLG '''
    for idx_net in range(num_exp):
        net = model



        #print('running %d|%d experiment' % (idx_net, num_exp))

        for method in ['DLG']:
            #print('%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []


            # compute original gradient
            gt_data = torch.unsqueeze(gt_data,dim = 0)
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            # original_dy_dx = update

            # generate dummy data and label
            # dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            pat_1 = torch.rand([channel, 16, 16])
            pat_2 = torch.cat((pat_1, pat_1), dim=1)
            pat_4 = torch.cat((pat_2, pat_2), dim=2)
            dummy_data = torch.unsqueeze(pat_4, dim=0).to(device).requires_grad_(True)
            #dummy_data = torch.ones(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
            

            
            optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr,max_iter=300)
            # predict the ground-truth label
            label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
                (1,)).requires_grad_(False)

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []

            #print('lr =', lr)
            for iters in range(Iteration):

                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
        
                    dummy_loss = criterion(pred, gt_label)

                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                    original_dy_dx2 = [update[i]  for i in range(len(original_dy_dx))]
                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx2):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()
                    return grad_diff


                history.append(tp(dummy_data[0].cpu()))
                optimizer.step(closure)

                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

                if iters % 10 == 0 and iters != 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                    # history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                    history_iters.append(iters)


            if method == 'DLG':
                loss_DLG = losses
                label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                mse_DLG = mses
            elif method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses

        # print('----------------------')
    if np.isnan(mses[-1]) or mses[-1] >= mses[0] or mses[-1] > 1.0:
        return 1.0
    else:
        return mses[-1]


def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

#
# def DLA_atk(model, gt_data, gt_label,info,batch = 1):
#     criterion = nn.CrossEntropyLoss()
#     root_path = '.'
#     save_path = os.path.join(root_path, 'results/rec_%s' % info).replace('\\', '/')
#     use_cuda = torch.cuda.is_available()
#     device = 'cuda' if use_cuda else 'cpu'
#
#     tp = transforms.Compose([
#         transforms.Resize(32),
#         transforms.CenterCrop(32),
#         transforms.ToTensor()
#     ])
#     tt = transforms.ToPILImage()
#     if not os.path.exists('results'):
#         os.mkdir('results')
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#
#     ''' train DLG and iDLG '''
#     gt_onehot_label = label_to_onehot(gt_label, num_classes=10)
#     dy_dx = []
#     original_dy_dx = []
#     original_pred = []
#     # for item in range(batch):
#     #     gt_data_single = torch.unsqueeze(gt_data[item],0)
#     #     out = model(gt_data_single)
#     #     #y = criterion(out, gt_onehot_label[item])
#     #     y = criterion(out, gt_label[item])
#     #     dy_dx = torch.autograd.grad(y, model.parameters(),retain_graph=True)
#     #     original_dy_dx_tmp = list((_.detach().clone() for _ in dy_dx))
#     #     original_dy_dx.append(original_dy_dx_tmp)
#     #     out_tmp = out.detach().clone()
#     #     original_pred.append(out_tmp)
#     # out = model(gt_data)
#     # for item in range(batch):
#     #     # y = criterion(out, gt_onehot_label[item])
#     #     # gt_label_batch = torch.squeeze(gt_label, dim=1)
#     #     y = criterion(out[item], gt_label[item])
#     #     dy_dx = torch.autograd.grad(y, model.parameters(), retain_graph=True)
#     #     original_dy_dx_tmp = list((_.detach().clone() for _ in dy_dx))
#     #     original_dy_dx.append(original_dy_dx_tmp)
#     #     out_tmp = out.detach().clone()
#     #     original_pred.append(out_tmp)
#     out = model(gt_data)
#     # y = criterion(out, gt_onehot_label[item])
#     # gt_label_batch = torch.squeeze(gt_label, dim=1)
#     gt_label_batch = gt_label
#     y = criterion(out, gt_label_batch)
#     dy_dx = torch.autograd.grad(y, model.parameters(), retain_graph=True)
#     original_dy_dx_tmp = list((_.detach().clone() for _ in dy_dx))
#     original_dy_dx.append(original_dy_dx_tmp)
#     out_tmp = out.detach().clone()
#     original_pred.append(out_tmp)
#
#     for item in range(1):
#         for rd in range(1):
#
#             torch.manual_seed(1234)
#             dummy_data = torch.unsqueeze(torch.randn(gt_data[item].size()),0).to(device).requires_grad_(True)
#             # dummy_data = torch.unsqueeze(torch.zeros(gt_data[item].size()),0).to(device).requires_grad_(True)
#             # dummy_data = torch.unsqueeze(torch.ones(gt_data[item].size()),0).to(device).requires_grad_(True)
#             # background = torch.unsqueeze(torch.zeros(gt_data[item].size()),0)
#             # background[0,2,::] = 1
#             # dummy_data = background.to(device).requires_grad_(True)
#             # dummy_data = (torch.unsqueeze(torch.randn(gt_data[item].size()),0)+background).to(device).requires_grad_(True)
#             # surrogate = torch.unsqueeze(gt_data[item+1],0)
#             # aaa = torch.rand([3,16,16])
#             # surrogate[0,:,8:24,8:24] =aaa
#             # dummy_data = surrogate.to(device).requires_grad_(True)
#             # dummy_data = torch.unsqueeze(gt_data[item+1],0).to(device).requires_grad_(True)
#             # k = np.random.randint(0,95)
#             # dummy_data = torch.unsqueeze(gt_data[k],0).to(device).requires_grad_(True)
#             # pat_1 = torch.rand([3, 16, 16])
#             # pat_2 = torch.cat((pat_1, pat_1), dim=1)
#             # pat_4 = torch.cat((pat_2, pat_2), dim=2)
#             # dummy_data = torch.unsqueeze(pat_4, dim=0).to(device).requires_grad_(True)
#             # aaa = torch.rand([3,8,8])
#             # bbb = torch.cat((aaa,aaa),dim=1)
#             # ccc = torch.cat((bbb,bbb),dim=1)
#             # ddd = torch.cat((ccc,ccc),dim=2)
#             # eee = torch.cat((ddd,ddd),dim=2)
#             # dummy_data = torch.unsqueeze(eee,dim=0).to(device).requires_grad_(True)
#
#             # aaa = torch.rand([3,4,4])
#             # bbb = torch.cat((aaa,aaa),dim=1)
#             # ccc = torch.cat((bbb,bbb),dim=1)
#             # ddd = torch.cat((ccc,ccc),dim=1)
#             # eee = torch.cat((ddd,ddd),dim=2)
#             # fff = torch.cat((eee,eee),dim=2)
#             # ggg = torch.cat((fff,fff),dim=2)
#             # dummy_data = torch.unsqueeze(ggg,dim=0).to(device).requires_grad_(True)
#
#             # dummy_data = plt.imread("./attack_image/replacement_69.png")
#             # print (dummy_data.shape)
#             # dummy_data = torch.FloatTensor(dummy_data).to(device)
#             # dummy_data = dummy_data.transpose(2,3).transpose(1,2)
#
#             dummy_unsqueeze = torch.unsqueeze(gt_onehot_label[item], dim=0)
#
#             dummy_label = torch.randn(dummy_unsqueeze.size()).to(device).requires_grad_(True)
#             label_pred = torch.argmin(torch.sum(original_dy_dx[item][-2], dim=-1),
#                                       dim=-1).detach().reshape((1,)).requires_grad_(False)
#             # print (original_dy_dx[item][-1].shape)
#             # print (original_dy_dx[item][-1].argmin())
#
#             # print (torch.sum(original_dy_dx[item][-2], dim=-1).argmin())
#
#             plt.imshow(tt(gt_data[0].cpu()))
#             plt.title("origin data")
#             # plt.savefig("./random_seed/index_%s_rand_seed_%s_label_%s"%(item,rd,torch.argmax(dummy_label, dim=-1).item()))
#
#             plt.show()
#             plt.clf()
#             print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())
#             print("stolen label is %d." % label_pred.item())
#
#             # optimizer = torch.optim.LBFGS([dummy_data,dummy_label])
#             optimizer = torch.optim.LBFGS([dummy_data, ])
#             # optimizer = torch.optim.AdamW([dummy_data,],lr=0.01)
#
#             history = []
#
#             percept_dis = np.zeros(300)
#             recover_dis = np.zeros(300)
#             for iters in range(256):
#
#                 # percept_dis[iters] = ssim(dummy_data, torch.unsqueeze(gt_data[item], dim=0), data_range=0).item()
#                 recover_dis[iters] = torch.dist(dummy_data, torch.unsqueeze(gt_data[item], dim=0), 2).item()
#
#                 history.append(tt(dummy_data[0].cpu()))
#
#
#                 def closure():
#                     optimizer.zero_grad()
#
#                     pred = model(dummy_data)
#                     # dummy_onehot_label = F.softmax(dummy_label, dim=-1).long()
#
#                     # dummy_loss = criterion(pred, dummy_onehot_label) # TODO: fix the gt_label to dummy_label in both code and slides.
#                     # print (pred)
#                     # print (label_pred)
#
#                     dummy_loss = criterion(pred, label_pred)
#                     dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
#                     # dummy_dy_dp = torch.autograd.grad(dummy_loss, dummy_data, create_graph=True)
#                     # print (dummy_dy_dp[0].shape)
#
#                     grad_diff = 0
#                     grad_count = 0
#                     # count =0
#                     for gx, gy in zip(dummy_dy_dx, original_dy_dx[item]):  # TODO: fix the variablas here
#
#                         # if iters==500 or iters== 1200:
#                         # print (gx[0])
#                         #    print ('hahaha')
#                         # print (gy[0])
#                         lasso = torch.norm(dummy_data, p=1)
#                         ridge = torch.norm(dummy_data, p=2)
#                         grad_diff += ((gx - gy) ** 2).sum()  # + 0.0*lasso +0.01*ridge
#                         # if count == 9:
#                         #    break
#                         # count=count+1
#                     # grad_diff = grad_diff / grad_count * 1000
#
#                     # grad_diff += ((original_pred[item]-pred)**2).sum()
#
#                     grad_diff.backward()
#                     # print (dummy_dy_dx)
#                     # print (original_dy_dx)
#                     return grad_diff
#
#
#                 optimizer.step(closure)
#                 if iters % 5 == 0:
#                     current_loss = closure()
#                     # if iters == 0:
#                     print("iter:%d loss:%.8f" %(iters,current_loss.item()))
#                     # print(iters, "%.8f" % current_loss.item())
#                 history.append(tt(dummy_data[0].cpu()))
#
#             # plt.figure(figsize=(18, 12))
#             # for i in range(60):
#             #  plt.subplot(6, 10, i + 1)
#             #  plt.imshow(history[i * 5])
#             #  plt.title("iter=%d" % (i * 5))
#             #  plt.axis('off')
#
#
#             iter_idx = [0, 10, 20, 30, 40, 50, 100, 150, 200]
#             for i in range(9):
#                 plt.subplot(1, 9, i + 1)
#                 plt.imshow(history[iter_idx[i]])
#                 plt.title("iter=%d" % (iter_idx[i]))
#                 plt.axis('off')
#             #plt.gray()
#             plt.show()
#             print("atk end")
#             # np.savetxt('ssim_random_batch8',percept_dis,fmt="%4f")
#             # np.savetxt('mse_random_batch8',recover_dis,fmt="%4f")
#
#             # print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())
#             # plt.savefig("./attack_image/index_%s_rand_%s_label_%s" % (item, rd, label_pred.item()))
