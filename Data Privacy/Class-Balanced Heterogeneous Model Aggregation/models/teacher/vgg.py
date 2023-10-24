import torch
import torch.nn as nn
import torch.nn.init as init

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class VGG(nn.Module):

    def __init__(self, features, num_classes=10):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        #self.apply(_weights_init)

    def forward(self, x):
        #print(self.features)
       # print('hi')
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 1
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def VGG11(num_classes):
    return VGG(make_layers(cfg['A'], batch_norm=False),num_classes=num_classes)
def VGG13(num_classes):
    return VGG(make_layers(cfg['B'], batch_norm=False),num_classes=num_classes)

def VGG16(num_classes):
    return VGG(make_layers(cfg['D'], batch_norm=False),num_classes=num_classes)
def VGG19(num_classes):
    return VGG(make_layers(cfg['E'], batch_norm=False),num_classes=num_classes)

# x=torch.rand([1,1,224,224])
# model=VGG11(num_classes=10)
# y=model(x)
# print(y.shape)

