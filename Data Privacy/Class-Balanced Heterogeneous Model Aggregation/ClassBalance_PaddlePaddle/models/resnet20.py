'''
本代码是论文 [1] 中描述的用于 CIFAR10 的 ResNet 的 PaddlePaddle 实现。
此文件的实现和结构在很大程度上受到了 [2] 的影响，该实现是为 ImageNet 设计的，并且没有用于恒等映射的选项 A。
此外，网络上的大多数实现都是从 torchvision 的 resnet 复制粘贴而来，参数数量是错误的。

用于 CIFAR10 的标准 ResNet（为了公平比较等）具有以下层数和参数量：
名称      | 层数   | 参数量
ResNet20  |   20   | 0.27M
ResNet32  |   32   | 0.46M
ResNet44  |   44   | 0.66M
ResNet56  |   56   | 0.85M
ResNet110 |  110   |  1.7M
ResNet1202|  1202  | 19.4M
此实现确实具有以上参数量。

参考文献:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

如果您在工作中使用此实现，请不要忘记提及原作者 Yerlan Idelbayev。
由 Gemini 将其转换为 PaddlePaddle 实现。
'''

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingNormal

__all__ = ['ResNet', 'ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202']


def _weights_init(m):
    """
    使用 Kaiming 正态分布初始化卷积层和线性层的权重。
    """
    if isinstance(m, (nn.Conv2D, nn.Linear)):
        # 在 PaddlePaddle 中，权重初始化器是一个类实例
        init = KaimingNormal()
        init(m.weight)


class LambdaLayer(nn.Layer):
    """
    一个简单的自定义层，用于包装一个 lambda 函数。
    """

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if option == 'A':
                """
                对于 CIFAR10，ResNet 论文使用选项 A。
                通过下采样和零填充来匹配维度。
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], [0, 0, 0, 0, planes // 4, planes // 4], "constant", 0)
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias_attr=False),
                    nn.BatchNorm2D(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Layer):
    def __init__(self, block, num_blocks, num_classes=10, inp_channels=3, name=None):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.model_name = name

        # CIFAR10 输入是 3 通道，如果用于 MNIST 等灰度图，可改为 1
        self.conv1 = nn.Conv2D(inp_channels, 16, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

        # 应用权重初始化
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # 使用 adaptive_avg_pool2d 更具通用性，或保持与原文一致
        out = F.avg_pool2d(out, out.shape[3])
        # flatten a tensor
        out = paddle.flatten(out, 1)
        out = self.linear(out)
        return out


def ResNet20(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, inp_channels, name='ResNet20')


def ResNet32(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, inp_channels, name='ResNet32')


def ResNet44(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [7, 7, 7], num_classes, inp_channels, name='ResNet44')


def ResNet56(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [9, 9, 9], num_classes, inp_channels, name='ResNet56')


def ResNet110(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [18, 18, 18], num_classes, inp_channels, name='ResNet110')


def ResNet1202(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [200, 200, 200], num_classes, inp_channels, name='ResNet1202')


if __name__ == "__main__":
    # 示例：创建 ResNet20 模型并打印其摘要信息
    # 假设输入图像大小为 3x32x32，这是 CIFAR10 的标准尺寸
    models = {
        'ResNet20': ResNet20,
        'ResNet32': ResNet32,
        'ResNet44': ResNet44,
        'ResNet56': ResNet56,
        'ResNet110': ResNet110,
        'ResNet1202': ResNet1202,
    }
    for name, model_fn in models.items():
        model = model_fn(num_classes=10, inp_channels=3)
        logger.info(f"--- {name} Summary ---")
        paddle.summary(model, (1, 3, 32, 32))
        logger.info("\n" + "=" * 30 + "\n")
