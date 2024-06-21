import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库模块
import matplotlib.pyplot as plt
# torch.manual_seed(1)  # reproducible
EPOCH = 1  # 训练整批数据次数，训练次数越多，精度越高，为了演示，我们训练5次
BATCH_SIZE = 50  # 每次训练的数据集个数
LR = 0.001  # 学习效率
DOWNLOAD_MNIST = True  # 如果你已经下载好了EMNIST数据就设置 False

# EMNIST 手写字母 训练集
train_data = torchvision.datasets.EMNIST(
    root='../data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,
    split = 'letters'
)
# EMNIST 手写字母 测试集
test_data = torchvision.datasets.EMNIST(
    root='../data',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False,
    split = 'letters'
)