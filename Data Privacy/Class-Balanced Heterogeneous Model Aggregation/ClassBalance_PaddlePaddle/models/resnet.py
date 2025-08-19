'''ResNet in PaddlePaddle.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# PADDLE-SPECIFIC: Base class is nn.Layer
class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # PADDLE-SPECIFIC: Use nn.Conv2D and bias_attr=False
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        # PADDLE-SPECIFIC: Use nn.BatchNorm2D
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# PADDLE-SPECIFIC: Base class is nn.Layer
class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # PADDLE-SPECIFIC: Use nn.Conv2D and bias_attr=False
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=1, bias_attr=False)
        # PADDLE-SPECIFIC: Use nn.BatchNorm2D
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, self.expansion*planes, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# PADDLE-SPECIFIC: Base class is nn.Layer
class ResNet(nn.Layer):
    def __init__(self, block, num_blocks, num_classes=10, name=None):
        super(ResNet, self).__init__()
        self.model_name = name
        self.in_planes = 64

        # PADDLE-SPECIFIC: Use nn.Conv2D and bias_attr=False
        # Note: The input channel is 1, suitable for grayscale images like MNIST.
        self.conv1 = nn.Conv2D(1, 64, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) #64,28,28
        b1 = self.layer1(out)
        b2 = self.layer2(b1) #128,14,14
        b3 = self.layer3(b2) #256,7,7
        b4 = self.layer4(b3) #512,4,4
        pool = F.avg_pool2d(b4, kernel_size=4) #512,1,1
        
        # PADDLE-SPECIFIC: Use paddle.flatten for reshaping
        out = paddle.flatten(pool, start_axis=1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, name='ResNet18')

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, name='ResNet34')

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, name='ResNet50')

def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, name='ResNet101')

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, name='ResNet152')


def test():
    net = ResNet18()
    # PADDLE-SPECIFIC: Use paddle.randn and provide shape as a list/tuple
    # Corrected input channels from 3 to 1 to match the model's conv1 layer
    y = net(paddle.randn([1, 1, 32, 32]))
    # PADDLE-SPECIFIC: Use .shape attribute instead of .size() method
    logger.info(y.shape)

# To run the test, uncomment the following line:
# test()