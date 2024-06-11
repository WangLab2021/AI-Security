import copy

import numpy as np
import syft as sy
import os
from PIL import Image
import torchvision as tv            #里面含有许多数据集
import torch
import torchvision.transforms as transforms    #实现图片变换处理的包
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import sample
import logging
import cifar10_dataloader
from torch.autograd import Variable
from clustering import reclustering
from datetime import datetime
from set_clients import set_client_ability
hook = sy.TorchHook(torch)
logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')

class Argument():
    def __init__(self):
        self.user_num = 100       # number of total clients P
        self.K = 20     # number of participant clients K
        self.lr = 0.0005      # learning rate of global model
        self.batch_size = 8      # batch size of each client for local training
        self.itr_test = 100        # number of iterations for the two tests on test datasets
        self.test_batch_size = 128    # batch size for test datasets
        self.total_iterations = 10000     # total number of iterations
        self.seed = 1     # parameter for the server to initialize the model
        self.classNum = 1   # number of data classes on each client, which can determine the level of non-IID data
        self.cuda_use = False
        self.frac = 0.1
        self.clustering = 'K-Means' # DBSCAN, K-Means, Graph
        self.standard = 'Cosine' # Cosine, entropy
        self.l_epoch = 1 # local epochs
        self.momentum = False
        self.alpha = 0.1
        self.function = 0 # 0 1/x 1 (e/2)^(-t)
        self.inter = 1 # 0:average 1:weight
        self.intergroup =0 # 0：所有客户端按组数据总量分 1:所有客户端按客户端数据量分

args = Argument()
use_cuda = args.cuda_use and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

##################################获取模型层数和各层的形状#############
def GetModelLayers(model):
    Layers_shape = []
    Layers_nodes = []
    for idx, params in enumerate(model.parameters()):
        Layers_num = idx
        Layers_shape.append(params.shape)
        Layers_nodes.append(params.numel())
    return Layers_num + 1, Layers_shape, Layers_nodes

##################################设置各层的梯度为0#####################
def ZerosGradients(Layers_shape):
    ZeroGradient = []
    for i in range(len(Layers_shape)):
        ZeroGradient.append(torch.zeros(Layers_shape[i]))
    return ZeroGradient

#################################计算范数################################
def L_norm(Tensor):
    norm_Tensor = torch.tensor([0.])
    for i in range(len(Tensor)):
        norm_Tensor += Tensor[i].float().norm()**2
    return norm_Tensor.sqrt()

################################ 定义剪裁 #################################
def TensorClip(Tensor, ClipBound):
    norm_Tensor = L_norm(Tensor)
    if ClipBound<norm_Tensor:
        for i in range(Layers_num):
            Tensor[i] = Tensor[i]*ClipBound/norm_Tensor
    return Tensor

############################定义测试函数################################
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for imagedata, labels in test_loader:
            outputs = model(Variable(imagedata))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_acc = 100. * correct / total
    print('10000张测试集: testacc is {:.4f}%, testloss is {}.'.format(test_acc, test_loss))
    return test_loss, test_acc

##########################定义训练过程，返回梯度########################
def train(learning_rate, model, train_data, train_targets, device, optimizer):
    model.train()
    model.zero_grad()
    train_targets = Variable(train_targets.long())
    optimizer.zero_grad()
    outputs = model(train_data)
    # 计算准确率
    #_, predicted = torch.max(outputs.data, 1)
    #total = labels.size(0)
    #correct = (predicted == labels).sum()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, train_targets)
    loss.backward()

    Gradients_Tensor = []
    for params in model.parameters():
        Gradients_Tensor.append(params.grad.data)    # 把各层的梯度添加到张量Gradients_Tensor
    return Gradients_Tensor, loss


##################################计算Non-IID程度##################################
def JaDis(datasNum, userNum):
    sim = []
    for i in range(userNum):
        data1 = datasNum[i]
        for j in range(i+1, userNum):
            data2 = datasNum[j]
            sameNum = [min(x, y) for x, y in zip(data1, data2)]
            sim.append(sum(sameNum) / (sum(data1) + sum(data2) - sum(sameNum)))
    distance = 2*sum(sim)/(userNum*(userNum-1))
    return distance


###################################################################################
##################################模型和用户生成###################################
model = Net()
workers = []
models = {}
optims = {}
taus = {}
d_workers = {}
#print(args.user_num)
full = []
for i in range(args.user_num):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i,i))
    exec('models["user{}"] = model.copy()'.format(i))
    exec('optims["user{}"] = optim.SGD(params=models["user{}"].parameters(), lr=args.lr)'.format(i,i))
    exec('workers.append(user{})'.format(i))    # 列表形式存储用户
    exec('taus["user{}"] = {}'.format(i,1))      # 列表形式存储用户
    d_workers['user'+str(i)] = i
    full.append('user'+str(i))
    # exec('workers["user{}"] = user{}'.format(i,i))    #字典形式存储用户

user_t = set_client_ability()
user_now = copy.deepcopy(user_t)
optim_sever = optim.SGD(params=model.parameters(), lr=args.lr)    # 定义服务器优化器
###################################################################################
###############################数据载入############################################
# 使用torchvision加载并预处理CIFAR10数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
#把数据变为tensor并且归一化range [0, 255] -> [0.0,1.0]
trainset = tv.datasets.CIFAR10(root='data2/', train=True, download=True, transform=transform)
federated_data, dataNum, user_tag = cifar10_dataloader.dataset_federate_noniid(trainset, workers, transform, args.classNum)
Jaccard = JaDis(dataNum, args.user_num)
print('Jaccard distance is {}'.format(Jaccard))

testset = tv.datasets.CIFAR10('data2/', train=False, download=False)
testset = cifar10_dataloader.testLoader(testset)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, drop_last=True, num_workers=0)

# show = ToPILImage()         # 可以把Tensor转成Image,方便进行可视化
# (data,label) = trainset[100]
# show((data+1)/2).resize((100, 100))
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
# show(tv.utils.make_grid((images+1)/2)).resize((400, 100))    # make_grid的作用是将若干幅图像拼成一幅图像

#定义记录字典
logs = {'train_loss': [], 'test_loss': [], 'test_acc': []}
test_loss, test_acc = test(model, test_loader, device)   # 初始模型的预测精度
logs['test_acc'].append(test_acc.item())
logs['test_loss'].append(test_loss)
f = open("./jiannoweight+noni+data1+K20.txt", "a")

###################################################################################
#################################联邦学习过程######################################
#获取模型层数和各层形状
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
e = torch.exp(torch.tensor(1.))

v = ZerosGradients(Layers_shape)

#定义训练/测试过程
for itr in range(1, args.total_iterations + 1):
    #按设定的每回合用户数量和每个用户的批数量载入数据，单个批的大小为batch_size
    #为了对每个样本上的梯度进行裁剪，令batch_size=1，batch_num=args.batch_size*args.batchs_round，将样本逐个计算梯度                                               worker_num=args.K, batch_num=1)
    # workers_list = federated_train_loader.workers    # 当前回合抽取的用户列表
    #if itr % 100 == 1 :
    if itr == 1:
        if itr == 1:
            gradient_c = []
            for i in range(args.user_num):
                gradient_c.append(ZerosGradients(Layers_shape))
            federated_train_loader = sy.FederatedDataLoader(federated_data, workers = full, batch_size=args.batch_size, shuffle=True,
                                                            worker_num=args.user_num, batch_num=1)
            #print(federated_data.workers)
            for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
                model_round = models[train_data.location.id]
                optimizer = optims[train_data.location.id]
                client_idx = d_workers[train_data.location.id]
                #f.write(str(client_idx))
                train_data, train_targets = train_data.to(device), train_targets.to(device)
                train_data, train_targets = train_data.get(), train_targets.get()
                # optimizer = optims[data.location.id]
                # 返回梯度张量，列表形式；同时返回loss；gradient=False，则返回-lr*grad
                Gradients_Sample, loss = train(args.lr, model_round, train_data, train_targets, device, optimizer)
                for j in range(Layers_num):
                    gradient_c[client_idx][j] += Gradients_Sample[j]
        #f.write("kkkk:"+str(len(gradient_c)))
        #f.write(str(gradient_c))
        tag_normal, cluster, num_cluster = reclustering(args, gradient_c)
        if itr != 1:
            staleness_cluster = []
            for idx_cluster in range(num_cluster):
                staleness_now = []
                for client in cluster[idx_cluster]:
                    staleness_now.append(taus['user'+str(client)])
                staleness_cluster.append(staleness_now)
            print("staleness: ", staleness_cluster)
            f.write("staleness: " + str(staleness_cluster) + ' \n')
        tag_cluster = []
        for idx_cluster in range(num_cluster):
            tag_now = []
            for client in cluster[idx_cluster]:
                tag_now.append(user_tag[client])
            tag_cluster.append(tag_now)
        print("tag:", tag_cluster)
        f.write("tag: " + str(tag_cluster) + ' \n')

    # 组内异步聚合
    All_Gradients = ZerosGradients(Layers_shape)
    G_tau = []
    all = 0
    for idx_cluster in range(num_cluster):
        all += max(int(args.frac * len(cluster[idx_cluster])), 1)
    worker_m = []
    for idx_cluster in range(num_cluster):
        cluster_model = copy.deepcopy(model)
        m = max(int(args.frac * len(cluster[idx_cluster])), 1)
        cluster_user_now = []
        for client in cluster[idx_cluster]:
            cluster_user_now.append(user_now[client])
        now = np.sort(cluster_user_now)
        # print("now: ",idx_cluster,cluster[idx_cluster],now)
        workers_list_idx = []
        for client in cluster[idx_cluster]:
            if user_now[client] <= now[m - 1]:
                workers_list_idx.append(client)
        for client in cluster[idx_cluster]:
            user_now[client] -= now[m - 1]
        for idx in workers_list_idx:
            user_now[idx] = user_t[idx]
        # workers_list_idx = sample(cluster[idx_cluster], m)
        #print(m, workers_list_idx)
        workers_list = []
        for idx in workers_list_idx:
            workers_list.append('user' + str(idx))
            worker_m.append('user' + str(idx))
            gradient_c[idx] = ZerosGradients(Layers_shape)
        federated_train_loader = sy.FederatedDataLoader(federated_data, workers = workers_list,batch_size=args.batch_size, shuffle=True,
                                                        worker_num=m, batch_num=1)
        #federated_train_loader.workers = workers_list
        f.write(str(idx_cluster)+": "+str(workers_list)+str(cluster[idx_cluster])+'\n')
        # 生成与模型梯度结构相同的元素=0的列表
        Loss_train = torch.tensor(0.)
        K_tau = []
        Collect_Gradients = ZerosGradients(Layers_shape)
        for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
            #f.write(str(train_data.location.id))
            model_round = models[train_data.location.id]
            optimizer = optims[train_data.location.id]
            client_idx = d_workers[train_data.location.id]
            user_tau = taus[train_data.location.id]
            K_tau.append(user_tau)
            train_data, train_targets = train_data.to(device), train_targets.to(device)
            train_data, train_targets = train_data.get(), train_targets.get()
            # optimizer = optims[data.location.id]
            # 返回梯度张量，列表形式；同时返回loss；gradient=False，则返回-lr*grad
            Gradients_Sample, loss = train(args.lr, model_round, train_data, train_targets, device, optimizer)
            Loss_train += loss
            if args.function == 0:
                weight = 1 / user_tau
            else:
                if args.function == 1:
                    weight = (e / 2) ** (-user_tau)
            for j in range(Layers_num):
                Collect_Gradients[j] += Gradients_Sample[j] * args.lr * weight / m
            for j in range(Layers_num):
                gradient_c[client_idx][j] += Gradients_Sample[j]
                # 因为学习率和陈旧度成反比
        G_tau.append(min(K_tau))
        if args.function == 0:
            weight = 1 / min(K_tau)
        else:
            if args.function == 1:
                weight = (e / 2) ** (-min(K_tau))
        if args.inter == 0:
            weight = 1
        if args.intergroup == 1:
            weight_data = len(workers_list_idx) / all
        else :
            weight_data = len(cluster[idx_cluster]) / args.user_num
        if args.momentum == True:
            for j in range(Layers_num):
                All_Gradients[j] += Collect_Gradients[j]  * weight * weight_data
            if itr == 1:
                v = copy.deepcopy(All_Gradients)
            else :
                for j in range(Layers_num):
                    v[j] = v[j] * args.alpha + All_Gradients[j] * (1 - args.alpha)
            All_Gradients = copy.deepcopy(v)
        else :
            for j in range(Layers_num):
                All_Gradients[j] += Collect_Gradients[j] \
                                    * len(cluster[idx_cluster]) * (weight / args.user_num)
        # 升级延时信息
        for i in cluster[idx_cluster]:
            taus['user'+str(i)] += 1
        for worker in workers_list:
            taus[worker] = 1

        for grad_idx, params_sever in enumerate(cluster_model.parameters()):
            params_sever.data.add_(-1., Collect_Gradients[grad_idx])

        #同步更新不需要下面代码；异步更新需要下段代码
        for worker_idx in range(len(workers_list)):
            worker_model = models[workers_list[worker_idx]]
            for idx, (params_server, params_client) in enumerate(zip(cluster_model.parameters(),worker_model.parameters())):
                params_client.data = params_server.data
            models[workers_list[worker_idx]] = worker_model###添加把更新后的模型返回给用户

    # 组间同步聚合
    for grad_idx, params_sever in enumerate(model.parameters()):
        params_sever.data.add_(-1., All_Gradients[grad_idx])

    for worker_idx in range(len(worker_m)):
        worker_model = models[worker_m[worker_idx]]
        for idx, (params_server, params_client) in enumerate(zip(model.parameters(), worker_model.parameters())):
            params_client.data = params_server.data
        models[worker_m[worker_idx]] = worker_model  ###添加把更新后的模型返回给用户

    if itr == 1 or itr % args.itr_test == 0:
        print('itr: {}'.format(itr))
        test_loss, test_acc = test(model, test_loader, device)  # 初始模型的预测精度
        logs['test_acc'].append(test_acc)
        logs['test_loss'].append(test_loss)
        f.write(str(test_acc)+'\n\n\n')

    if itr == 1 or itr % args.itr_test == 0:
        # 平均训练损失
        Loss_train /= (idx_outer + 1)
        logs['train_loss'].append(Loss_train)

f.close()


with open('./results/cifar10_Ours_testacc.txt', 'a+') as fl:
    fl.write('\n' + date +  ' Results (UN is {}, K is {}, classnum is {}, BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
             format( args.user_num, args.K, args.classNum, args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write(str(args))
    fl.write('Ours: ' + str(logs['test_acc']))

with open('./results/cifar10_Ours_trainloss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (UN is {}, K is {}, BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
             format(args.user_num, args.K, args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write('train_loss: ' + str(logs['train_loss']))

with open('./results/cifar10_Ours_testloss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (UN is {}, K is {}, BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
             format(args.user_num, args.K, args.batch_size, args.lr, args.itr_test, args.total_iterations))
    fl.write('test_loss: ' + str(logs['test_loss']))

