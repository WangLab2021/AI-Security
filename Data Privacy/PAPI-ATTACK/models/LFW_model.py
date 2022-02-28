import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet


class LFWNet(SimpleNet):
    def __init__(self, name=None, created_time=None):
        super(LFWNet, self).__init__(f'{name}_Simple', created_time)
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, 2)
        # self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x=  self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        # print('shape:', x.size())
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x