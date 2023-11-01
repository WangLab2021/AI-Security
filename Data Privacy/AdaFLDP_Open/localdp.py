from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tkinter import FALSE

import matplotlib.pyplot as plt
import numpy
import torchmetrics
import argparse
from Global_Parameter import *
from EPS_round import *
from EPS_instance import *
#import csv
import time
import struct
import pickle
import copy
from dataset import ResNet18
import os
# from dataset import LeNet
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
import collections
from dlgAttack import DLA
from DLG import Vitrual_VulnerableClient
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser(description="FLDP")
parser.add_argument("--ds", type=str, help="",default='mnist')
parser.add_argument("--p", type=str, help="",default='ori')
parser.add_argument("--e", type=int, help="",default=8)
parser.add_argument("--b", type=int, help="",default=500)
parser.add_argument("--clip", type=int, help="",default=15)
parser.add_argument("--c", type=str, help="",default='0')
parser.add_argument("--segma", type=int, help="",default=10)
parser.add_argument("--decay", type=float, help="",default=0.1)
args = parser.parse_args()
c0 = args.clip
segma0 = args.segma
k0 = args.decay
bs = args.b
E = args.e
DATAMODE = args.ds
PRIVACY_MODE = args.p
device = "cuda:"+args.c

print("Running on %s" % device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from DLGlfw import LFWatk

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)
class LeNet(nn.Module):
    def __init__(self,channel = 3,num_classes = 10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act()
        )
        self.fc = nn.Sequential(
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.contiguous().view(out.shape[0], -1)
        # print(out.size())
        out = self.fc(out)
        return out



if DATAMODE == 'mnist':
    channel = 1
else:
    channel = 3
if DATAMODE == 'lfw':
    num_classes = 106
else:
    num_classes = 10
if DATAMODE == 'lfw':
    # model = torchvision.models.resnet18()
    # model.fc = nn.Linear(512, num_classes)
    # model.to(device)
    model = LeNet(channel=channel, num_classes=num_classes).to(device)
    model.apply(weights_init)
else:
    model = LeNet(channel=channel,num_classes=num_classes).to(device)
    #model.apply(weights_init)
    #torch.save(model.state_dict(), "./data/model_parameter4.pkl")
    model.load_state_dict(torch.load("./data/model_parameter2.pkl"))

#----------------------------------------------------------------------#
#get_data() return说明：
#非降序x集（按label的序）,非降序y集，验证x集，验证y集，测试x集，测试y集
#----------------------------------------------------------------------#
def get_data(datamode = DATAMODE):
    # load the data
    transform = transforms.Compose([transforms.Resize((32,32)),transforms.CenterCrop(32),transforms.ToTensor()])
    if datamode == 'mnist':
        train_dataset = datasets.MNIST('./data', train = True, transform=transform,download = True)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    elif datamode == 'cifar10':
        train_dataset = datasets.CIFAR10('./data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    elif datamode == 'lfw':
        # train_dataset = torchvision.datasets.LFWPeople('./data', split='train', transform=transform, download=True)
        # test_dataset = torchvision.datasets.LFWPeople('./data', split='train', transform=transform, download=True)
        lfw_people = fetch_lfw_people(min_faces_per_person=14, color=True, slice_=(slice(61, 189), slice(61, 189)),
                                      resize=0.25)
        x = lfw_people.images
        y = lfw_people.target

        target_names = lfw_people.target_names
        n_classes = target_names.shape[0]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=False)
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
        # X_train = torch.transpose
        # X_train = X_train.astype('float32')
        X_train /= 255.0
        X_test /= 255.0
        sorted_x_train = torch.FloatTensor(X_train)
        sorted_x_train = sorted_x_train.transpose(2, 3).transpose(1, 2)
        sorted_y_train = y_train

        x_test = torch.FloatTensor(X_test)
        x_test = x_test.transpose(2, 3).transpose(1, 2)
        x_vali = x_test
        y_vali = y_test
        test_dataset = torch.utils.data.TensorDataset(x_test, torch.LongTensor(y_test))
        print(
            "dadtaSet:%s ,get train points:%d ,get test points:%d" % (DATAMODE, len(sorted_x_train), len(x_test)))
    if DATAMODE == 'mnist' or DATAMODE == 'cifar10':
        mylist = []
        print("dadtaSet:%s ,get train points:%d ,get test points:%d"%(DATAMODE,len(train_dataset),len(test_dataset)))
        for i in range(len(train_dataset)):
            img,label = train_dataset[i]
            mylist.append(img)
        x_train = torch.stack(mylist,dim=0)
        y_train = np.array(train_dataset.targets)
        # create validation set,size:10000
        x_vali = x_train[50000:]
        y_vali = y_train[50000:]
        # create train_set,size:50000
        x_train = x_train[:50000]
        y_train = y_train[:50000]

        # sort train set (to make federated learning non i.i.d.)
        #indices_train是label非降序的索引
        indices_train = np.argsort(y_train)
        sorted_x_train = x_train[indices_train] #获得一个排序好的x_train
        sorted_y_train = np.array(y_train)[indices_train] #获得一个排序好的y_train

        # create a test set
        mylist = []
        for i in range(10000):
            img, label = test_dataset[i]
            mylist.append(img)
        x_test = torch.stack(mylist, dim=0)
        y_test = np.array(test_dataset.targets)
    return sorted_x_train.to(device), torch.from_numpy(sorted_y_train).type(torch.LongTensor).to(device), x_vali.to(device), torch.from_numpy(y_vali).type(torch.LongTensor).to(device), x_test.to(device), torch.from_numpy(y_test).type(torch.LongTensor).to(device),test_dataset

def getDataExample(x_,y_):
    if DATAMODE == 'lfw':
        id = 291
    elif DATAMODE == 'mnist':
        id = 1123
    else:
        id = 48850
    y = y_[id].view(1,)
    return x_[id],y
if __name__ == '__main__':
    if DATAMODE == 'lfw':
        client_set = pickle.load(open('./DATA/clients_lfw/' + str(total_client_num) + '_clients.pkl', 'rb'))
    elif DATAMODE == 'mnist':
        client_set = np.load('./DATA/clients/'+str(total_client_num)+'clients.npy')
    else:
        client_set = pickle.load(open('./DATA/clients_cifar10/' + str(total_client_num) + '_clients.pkl', 'rb'))
    total_data = np.load('./DATA/clients/1clients.npy')
    print("test shape:")
    print(total_data.shape)
    sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test,test_dataset = get_data()
    old_global_model = copy.deepcopy(model).to(device)
    atk_model = copy.deepcopy(model).to(device)  # 保存step2要用的模型
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])
    test_set_loader = torch.utils.data.DataLoader(test_dataset,batch_size=500,shuffle=True)
    for round in range(1):
        for k_t in range(1):
            local_model = copy.deepcopy(old_global_model)
            Epoch = E
            eps = 1
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            k_instance = EPS_instance(total_data[0], local_model, E, bs, eps, device)
            # ---------------------------------------------------------------------------------#
            #step3
            print("Client:%d is running in %d round,mode = %s"%(k_t+1,round,PRIVACY_MODE))
            if PRIVACY_MODE == 'tifs':
                num_weights = k_instance.TIFS(sorted_x_train, sorted_y_train, loss_fn, DecayClip, FALSE,device,test_set_loader,c0,segma0,k0)
            elif PRIVACY_MODE == 'sp':
                num_weights = k_instance.SP(sorted_x_train, sorted_y_train, loss_fn, DecayClip, FALSE, device,
                                                 test_set_loader,c0,segma0,k0)
            else:
                num_weights = k_instance.ADP(sorted_x_train, sorted_y_train, loss_fn, DecayClip, FALSE, device,
                                                 test_set_loader,c0,segma0,k0)
            # ---------------------------------------------------------------------------------#





















