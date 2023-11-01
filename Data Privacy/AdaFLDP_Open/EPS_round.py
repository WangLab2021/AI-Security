#Step1
import torch
from Global_Parameter import *
import numpy as np
import csv
import time
import struct
import pickle
import copy
import os
import math
import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms

class EPS_round:
    def __init__(self,valiloader):
        self.valiloader = valiloader
        self.delta_S = 1
        self.mum_S = 0
        self.best_count = 0.0
        self.factor = 0.5
        self.lr_start = False
    def RoundlyAccount(self,old_global_model,eps_global,t,device,Epoch): #计算得该轮的隐私预算
        with torch.no_grad():
            old_global_model.eval()
            total = 0
            correct = 0
            for data in self.valiloader:
                images, labels = data
                images= images.to(device)
                labels = labels.to(device)
                _, predictions = torch.max(old_global_model(images), 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        daughter_S = correct/total
        
        #对delta_S的处理#
        
        if t > 2:
            if daughter_S - self.mum_S > 0.07:
                self.lr_start = True
            if not self.lr_start:
                if daughter_S <= self.best_count: 
                    self.factor = self.factor * 0.4
                else:
                    self.factor = self.factor * 0.6 
            if daughter_S > self.best_count:
                self.best_count = daughter_S
            if not self.lr_start:
                delta_S = self.factor
            else:
                delta_S = min(daughter_S - self.mum_S,self.factor)
            if delta_S>0.01 and delta_S < self.delta_S:
                self.delta_S = delta_S
            elif delta_S <0.01 and delta_S >=0:
                self.delta_S = 0
        #对该轮S的保存，本质上其实就是上轮oldmodel的验证准确率
        #接着对eps_round的计算
        eps_round = np.exp((-1.0)* (self.delta_S)) * eps_global / (rounds - t + 1)
        # with open('./epsRO_%s_%s_epoch%d.csv'%(PRIVACY_MODE,DATAMODE,E), mode='a') as train_file:
        #         writer_train2 = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #         writer_train2.writerow([t, "eps:",eps_round,"delta_S:",self.delta_S,"daugther&mum_S:",daughter_S,self.mum_S])
        self.mum_S = daughter_S
        #
        #
        #
        return  eps_round
    
    