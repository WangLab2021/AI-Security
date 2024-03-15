# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDiscriminator(nn.Module):
    def __init__(self, n_feature):
        super(FeatureDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_feature, out_channels=128, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # (batch_size,128, x/2, x/2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (batch_size,128, x/2, x/2)
        x = self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
    