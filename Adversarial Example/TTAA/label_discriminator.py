# coding:utf8
import torch.nn as nn
from torchvision.models import *

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()
        self.net = vgg16(pretrained=True).eval()

    def forward(self, x):
        pred = self.net(x.clone())
        features = []
        for ii, model in enumerate(self.features):
            x = model(x)
            features.append(x.clone())
        return pred, features


class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        features = list(vgg19(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()
        self.net = vgg19(pretrained=True).eval()

    def forward(self, x):
        pred = self.net(x.clone())
        features = []
        for ii, model in enumerate(self.features):
            x = model(x)
            features.append(x.clone())
        return pred, features


class Vgg19_bn(nn.Module):
    def __init__(self):
        super(Vgg19_bn, self).__init__()
        features = list(vgg19_bn(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()
        self.net = vgg19_bn(pretrained=True).eval()

    def forward(self, x):
        pred = self.net(x.clone())
        features = []
        for ii, model in enumerate(self.features):
            x = model(x)
            features.append(x.clone())
        return pred, features


class Densenet121(nn.Module):
    def __init__(self):
        super(Densenet121, self).__init__()
        features = list(densenet121(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()
        self.net = densenet121(pretrained=True).eval()

    def forward(self, x):
        pred = self.net(x.clone())
        features = []
        for ii, model in enumerate(self.features):
            x = model(x)
            features.append(x.clone())
        return pred, features


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.features = resnet50(pretrained=True).eval()._modules.items()
        self.net = resnet50(pretrained=True).eval()

    def forward(self, x):
        pred = self.net(x.clone())
        features = []
        for ii, model in self.features:
            if ii == "fc":
                x = x.view(x.size(0), -1)
            x = model(x)
            features.append(x.clone())
        return pred, features
