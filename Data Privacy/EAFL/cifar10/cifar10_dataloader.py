import numpy
import syft as sy
import torch
import torchvision
import logging
from torch.autograd import Variable
import torchvision as tv            #里面含有许多数据集
import torch
import torchvision.transforms as transforms    #实现图片变换处理的包
from torchvision.transforms import ToPILImage

logger = logging.getLogger(__name__)

class testDataLoader():
    def __init__(self, data, targets):
        self.data = data
        self.labels = targets
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label
def testLoader(testset):
    datas = testset.data
    data = [torch.unsqueeze(1.*torch.tensor(datas[i].transpose()),0) for i in range(10000)]
    data = torch.cat(data, 0)
    return testDataLoader(data, testset.targets)

def dataset_federate_noniid(trainset, workers, transform, classNum):
    """
    Add a method to easily transform a torch.Dataset or a sy.BaseDataset
    into a sy.FederatedDataset. The dataset given is split in len(workers)
    part and sent to each workers
    """
    logger.info(f"Scanning and sending data to {', '.join([w.id for w in workers])}...")
    datas = trainset.data
    labels = trainset.targets
    labels = torch.tensor(labels)
    dataset = {}

    data_new = []
    for i in range(50000):
        data_new.append(torch.unsqueeze(1.*torch.tensor(datas[i].transpose()),0))
    datas = torch.cat(data_new,0)

    for i in range(10):
        index = (labels==i)
        dataset[str(i)] = datas[index]

    datasets = []
    datasTotalNum = []
    user_num = len(workers)
    user_tag = []

    for i in range(user_num):
        user_data = []
        user_label = []
        labelClass = torch.randperm(10)[0:classNum]
        dataRate = torch.rand([classNum])
        dataRate = dataRate / torch.sum(dataRate)
        dataNum = torch.randperm(40)[0] + 500
        dataNum = torch.round(dataNum * dataRate)
        if classNum>1:
            datasnum = torch.zeros([10])
            datasnum[labelClass.tolist()] = dataNum
            datasTotalNum.append(datasnum)
            now_tag = []
            for j in range(classNum):
                datanum = int(dataNum[j].item())
                index = torch.randperm(5000)[0:datanum]
                user_data.append(dataset[str(labelClass[j].item())][index, :, :, :])
                user_label.append(labelClass[j] * torch.ones(datanum))
                #now_tag += labelClass[j].tolist()
            user_data = torch.cat(user_data, 0)
            user_label = torch.cat(user_label, 0)
            #user_label = torch.cat(user_label, 0)
            #user_tag = append(now_tag)

        else:
            j = 0
            datasnum = torch.zeros([10])
            datasnum[labelClass] = dataNum
            datasTotalNum.append(datasnum)

            datanum = int(dataNum[j].item())
            index = torch.randperm(5000)[0:datanum]
            user_data = dataset[str(labelClass[j].item())][index, :, :, :]
            user_label = labelClass[j] * torch.ones(datanum)
            user_data = torch.tensor(user_data)
            user_tag.append(labelClass[j].tolist())


        worker = workers[i]
        logger.debug("Sending data to worker %s", worker.id)
        user_data = user_data.send(worker)
        user_label = user_label.send(worker)
        datasets.append(sy.BaseDataset(user_data, user_label))  # .send(worker)
    logger.debug("Done!")
    return sy.FederatedDataset(datasets), datasTotalNum, user_tag


'''
import torchvision as tv            #里面含有许多数据集
import torch
import torchvision.transforms as transforms    #实现图片变换处理的包
from torchvision.transforms import ToPILImage

#使用torchvision加载并预处理CIFAR10数据集
show = ToPILImage()         #可以把Tensor转成Image,方便进行可视化
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))])#把数据变为tensor并且归一化range [0, 255] -> [0.0,1.0]
trainset = tv.datasets.CIFAR100(root='data1/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testset = tv.datasets.CIFAR100('data1/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=0)

(data,label) = trainset[100]
show((data+1)/2).resize((100, 100))
dataiter = iter(trainloader)
images, labels = dataiter.next()
#print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images+1)/2)).resize((400, 100))#make_grid的作用是将若干幅图像拼成一幅图像

#定义网络
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,100)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return  x

net = Net()
print(net)

# 定义损失函数和优化器
from torch import optim
criterion = nn.CrossEntropyLoss()    # 定义交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 训练网络
from torch.autograd import Variable
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):    # enumerate将其组成一个索引序列，利用它可以同时获得索引和值,enumerate还可以接收第二个参数，用于指定索引起始值
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 ==0:
            print('[{}, {}] loss: {}, acc is {}'.format(epoch+1,i+1,running_loss/2000, 1.*correct/total))
            running_loss = 0.0
print("----------finished training---------")
dataiter = iter(testloader)
images, labels = dataiter.next()
# print('实际的label: ',' '.join('%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images/2 - 0.5)).resize((400,100))
outputs = net(Variable(images))
_, predicted = torch.max(outputs.data, 1)    # 返回最大值和其索引
# print('预测结果:',' '.join('%5s'%classes[predicted[j]] for j in range(4)))
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('10000张测试集中的准确率为: %d %%'%(100*correct/total))
if torch.cuda.is_available():
    net.cuda()
    images = images.cuda()
    labels = labels.cuda()
    output = net(Variable(images))
    loss = criterion(output, Variable(labels))
'''