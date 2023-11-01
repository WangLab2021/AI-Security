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
parser.add_argument("--e", type=int, help="",default=4)
parser.add_argument("--b", type=int, help="",default=128)
parser.add_argument("--eps", type=int, help="",default=150)
parser.add_argument("--c", type=str, help="",default='0')
parser.add_argument("--w", type=int, help="",default=100)
parser.add_argument("--r", type=float, help="",default=0.02)
args = parser.parse_args()
total_client_num = args.w
selected_rate = args.r
parti_client_num= int(total_client_num * selected_rate)
bs = args.b
E = args.e
eps_global_init = args.eps
DATAMODE = args.ds
PRIVACY_MODE = args.p
device = "cpu"
if torch.cuda.is_available() and PRIVACY_MODE == 'fix':
    device = "cuda:"+args.c
if torch.cuda.is_available() and PRIVACY_MODE == 'ori':
    device = "cuda:"+args.c
if torch.cuda.is_available() and PRIVACY_MODE == 'adp':
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
    model.apply(weights_init)
    #torch.save(model.state_dict(), "./data/model_parameter4.pkl")
    # model.load_state_dict(torch.load("./data/model_parameter2.pkl"))

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
        total_x = sorted_x_train[0:]
        total_y = torch.LongTensor(y_train)
        x_test = torch.FloatTensor(X_test)
        x_test = x_test.transpose(2, 3).transpose(1, 2)
        x_vali = x_test
        y_vali = y_test
        test_dataset = torch.utils.data.TensorDataset(x_test, torch.LongTensor(y_test))
        print(
            "dadtaSet:%s ,get train points:%d ,get test points:%d" % (DATAMODE, len(sorted_x_train), len(x_test)))
    if DATAMODE == 'mnist' or DATAMODE == 'cifar10':
        if DATAMODE == 'mnist':
            index_start = 50000
        else:
            index_start = 40000
        mylist = []
        print("dadtaSet:%s ,get train points:%d ,get test points:%d"%(DATAMODE,len(train_dataset),len(test_dataset)))
        for i in range(len(train_dataset)):
            img,label = train_dataset[i]
            mylist.append(img)
        x_train = torch.stack(mylist,dim=0)
        y_train = np.array(train_dataset.targets)
        total_x = x_train[0:]
        total_y = y_train[0:]
    
        # create validation set,size:10000
        x_vali = x_train[index_start:]
        y_vali = y_train[index_start:]
        # create train_set,size:50000
        x_train = x_train[:index_start]
        y_train = y_train[:index_start]

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
        total_x = torch.FloatTensor(total_x)
        total_y = torch.LongTensor(total_y)
    return total_x.to(device),total_y.to(device),sorted_x_train.to(device), torch.from_numpy(sorted_y_train).type(torch.LongTensor).to(device), x_vali.to(device), torch.from_numpy(y_vali).type(torch.LongTensor).to(device), x_test.to(device), torch.from_numpy(y_test).type(torch.LongTensor).to(device),test_dataset

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
    torch.set_num_threads(10)
    print("total num:",total_client_num)
    print("par num:",parti_client_num)
    if DATAMODE == 'lfw':
        client_set = pickle.load(open('./DATA/clients_lfw/' + str(total_client_num) + '_clients.pkl', 'rb'))
    elif DATAMODE == 'mnist':
        client_set = np.load('./DATA/clients/'+str(total_client_num)+'clients.npy')
    else:
        client_set = np.load('./DATA/clients_cifar10/'+str(total_client_num)+'clients.npy')
    total_x,total_y,sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test,test_dataset = get_data()
    vali_set =torch.utils.data.TensorDataset(x_vali,y_vali)
    valiloader = torch.utils.data.DataLoader(vali_set, batch_size=100, shuffle=FALSE)
    # dataExample = sorted_x_train[100]
    # dataExample = torch.unsqueeze(dataExample,0)
    # labelExample = sorted_y_train[100]
    # labelExample = labelExample.view(1,)
    new_global_model = copy.deepcopy(model).to(device)
    old_global_model = copy.deepcopy(model).to(device)
    atk_model = copy.deepcopy(model).to(device)  # 保存step2要用的模型
    starting_time = time.time()
    eps_global = eps_global_init
    epsRoundAccount = EPS_round(valiloader)  # 传入固定vali集,实例化*注意此处的实例化全局唯一
    client_models = []
    server_save_update = {}
    server_save_deltaU = {}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])
    test_set_loader = torch.utils.data.DataLoader(test_dataset,batch_size=4,shuffle=True)
    last_S = 10
    t = 0
    for round in range(rounds):
        tested = False
        #  Server端 start
        # ---------------------------------------------------------------------------------#
        #  数据划分方式：shuffle，每个用户固定拥有500个数据，每轮选取partiClientNum个用户进行计算#
        np.random.seed(round)
        perm = np.random.permutation(total_client_num)  # 对所有的client进行一个shuffle
        s = perm[0:parti_client_num].tolist() #S为本轮选出来的clients
        participating_clients_data = [client_set[k] for k in s]
        E_list = [2,4,1,2,4]
        client_random = [E_list[s[k] % 5] for k in range(parti_client_num)]
        print("#-----------随机选取用户完毕--------------------#")
        # ---------------------------------------------------------------------------------#
        # Step 1
        # 划分该轮的epsilon，用到vali_set（10000个样本）与old_global_model，并利用得到的准确率进行一个计算
        if round != 0:
            atk_model.load_state_dict(old_global_model.state_dict())
        old_global_model.load_state_dict(new_global_model.state_dict())
        old_global_model=old_global_model.to(device)
        print("round %d test Acc == :" % (round))
        with torch.no_grad():
            old_global_model.eval()
            total = 0
            correct = 0
            for data in test_set_loader:
                images, labels = data
                images= images.to(device)
                labels = labels.to(device)
                _, predictions = torch.max(old_global_model(images), 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
            deltaS = correct / total * 100 - last_S
            last_S = correct / total * 100
            with open('./Acc/epsilon/%s_%s_epsilon%d.csv'%(PRIVACY_MODE,DATAMODE,eps_global_init), mode='a') as train_file:
                writer_train2 = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer_train2.writerow([round, correct / total])
        print('Accuracy: %d/%d = %.2f%%' % (correct, total, correct / total * 100))
        
        if t == rounds:
            print("END")
            exit(0)
        if PRIVACY_MODE == 'adp':
            eps_round = epsRoundAccount.RoundlyAccount(old_global_model, eps_global,t,device,E)  # 计算得该轮的隐私预算
        else:
            print('privacy mode:%s,total eps:%d,rounds:%d'%(PRIVACY_MODE,eps_global_init,rounds))
            eps_round = eps_global_init / rounds
        eps_global -= eps_round
        # 保留每轮的eps
        # if PRIVACY_MODE == 'adp':
        #     with open('eps_round0603.csv', mode='a') as train_file:
        #         writer_train = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #         writer_train.writerow([t, eps_round])
        print("#step1 完毕")
        # ---------------------------------------------------------------------------------#
        # Step 2
        # 利用服务器的DLA对该轮选定的client进行攻击，获得一个预算分配的方案
        # 输入：eps_round
        # 输出：对每个client的隐私预算list: eps_clients[parti_client_num]
        eps_clients = [eps_round for n in range(parti_client_num)]
        maxU = float('-inf')
        # print("MaxU:",maxU)
        if PRIVACY_MODE == 'adp':
            dataExample,labelExample = getDataExample(total_x,total_y)
            # print(dataExample,labelExample)
            for key,original_dy_dx in server_save_update.items(): #若参与了上一轮计算且得到了更新的梯度，更新histortList
                dy_dx_ = [original_dy_dx[j].detach().clone() for j in range(len(original_dy_dx))]
                newU = DLA(atk_model,dataExample,labelExample,dy_dx_,device,num_classes,channel)
                if server_save_deltaU.get(key):
                    server_save_deltaU[key].append(newU)
                else:
                    server_save_deltaU[key] = []
                    server_save_deltaU[key].append(newU)
            finalList = [0 for n in range(parti_client_num)]
            for i in range(parti_client_num):
                if server_save_deltaU.get(s[i]):
                    if len(server_save_deltaU[s[i]]) > 1:
                        finalList[i] = Lambda * np.mean(server_save_deltaU[s[i]][0:-1]) + (1 - Lambda) * server_save_deltaU[s[i]][-1]
                        if maxU < finalList[i]:
                            maxU = finalList[i]
                    else:
                        finalList[i] = - 1.0
                else:
                    finalList[i] = - 1.0
            for i in range(parti_client_num):
                if finalList[i] <= 0 :
                    eps_clients[i] = eps_round
                else:
                    eps_clients[i] = (finalList[i] / maxU)*eps_round
            print("final list:",finalList)
            print("final list:",maxU)
        else:
            eps_clients = [eps_round for n in range(parti_client_num)]
        server_save_update.clear()
        print("#step2 完毕")
        print("Server端 waiting...")
        # Server端 waiting...
        # ---------------------------------------------------------------------------------#
        # Client端
        print("#Client端 start")
        client_models = []
        norm_client_gradient = []
        for k_t in range(parti_client_num):
            #BSize = 4
            local_model = copy.deepcopy(old_global_model)
            Epoch = E
            eps = eps_clients[k_t]
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            k_instance = EPS_instance(participating_clients_data[k_t], local_model, E, bs, eps,device)
            # ----------------------------test of Vulearable client----------------------------#
            tested = True
            if not tested:
                mse = Vitrual_VulnerableClient(E,device,local_model, PRIVACY_MODE,dataset=DATAMODE,rd=round, eps=eps,channel = channel)
                #LFWatk(local_model,device,round,eps)
                with open('./Acc/epoch/mse_%s_%s_epoch_%d.csv'%(PRIVACY_MODE,DATAMODE,E), mode='a') as train_file:
                    writer_train2 = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer_train2.writerow([round, mse])
                tested = True
            # ---------------------------------------------------------------------------------#
            #step3
            print("Client:%d is running in %d round,mode = %s"%(k_t+1,round,PRIVACY_MODE))
            if PRIVACY_MODE == 'adp':
                num_weights = k_instance.runOurs(sorted_x_train, sorted_y_train, loss_fn, DecayClip, FALSE,device,x_vali,y_vali)
            elif PRIVACY_MODE == 'ori':
                num_weights = k_instance.run_NoPrvy_NoClip(sorted_x_train, sorted_y_train, loss_fn, DecayClip, FALSE, device,
                                                 x_vali, y_vali)
            else:
                num_weights = k_instance.runFix(sorted_x_train, sorted_y_train, loss_fn, DecayClip, FALSE, device,
                                                 x_vali, y_vali)
            # ---------------------------------------------------------------------------------#
            client_models.append(local_model)
            final_gradient = []
            for Gi, Gg in zip(local_model.parameters(), old_global_model.parameters()):
                tmp_tensor = Gg - Gi
                final_gradient.append(tmp_tensor / learning_rate)
            server_save_update[s[k_t]] = final_gradient
        # Client端 END
        #  Server端  聚合----------------------------------------------------------------------#
        worker_state_dict = [copy.deepcopy(x.state_dict()) for x in client_models]
        sum_parameters = None
        for x in worker_state_dict:
            if sum_parameters == None:
                sum_parameters = {}
                for key,var in x.items():
                    sum_parameters[key] = var.detach().clone()
            else:
                for key,var in x.items():
                    sum_parameters[key] = sum_parameters[key] + var.detach().clone()
        fed_state_dict = {}
        for var in sum_parameters:
            fed_state_dict[var] = (sum_parameters[var] / parti_client_num)
        new_global_model.load_state_dict(fed_state_dict)
        new_global_model = new_global_model.to(device)
        print("global model after Avg:")
        t = round+1





















