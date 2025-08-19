import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal
import numpy as np

__all__ = ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202']


def weight_variable():
    return HeNormal()


class LambdaLayer(nn.Cell):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def construct(self, x):
        return self.lambd(x)


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            pad_mode='pad',
            padding=1,
            has_bias=False,
            weight_init=weight_variable(),
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            pad_mode='pad',
            padding=1,
            has_bias=False,
            weight_init=weight_variable(),
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                # 修正pad格式
                self.shortcut = LambdaLayer(
                    lambda x: ops.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2),
                        mode="constant",
                        value=0,
                    )
                )
            elif option == 'B':
                self.shortcut = nn.SequentialCell(
                    [
                        nn.Conv2d(
                            in_planes,
                            planes * self.expansion,
                            kernel_size=1,
                            stride=stride,
                            has_bias=False,
                            weight_init=weight_variable(),
                        ),
                        nn.BatchNorm2d(planes * self.expansion),
                    ]
                )
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Cell):
    def __init__(self, block, num_blocks, num_classes=10, inp_channels=3, name=None):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.model_name = name

        self.conv1 = nn.Conv2d(
            inp_channels,
            16,
            kernel_size=3,
            stride=1,
            pad_mode='pad',
            padding=1,
            has_bias=False,
            weight_init=weight_variable(),
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Dense(64, num_classes)
        self.relu = nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(layers)

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.mean(axis=(2, 3))
        out = self.linear(out)
        return out


def ResNet20(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, inp_channels=inp_channels, name='ResNet20')


def ResNet32(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, inp_channels=inp_channels, name='ResNet32')


def ResNet44(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, inp_channels=inp_channels, name='ResNet44')


def ResNet56(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, inp_channels=inp_channels, name='ResNet56')


def ResNet110(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, inp_channels=inp_channels, name='ResNet110')


def ResNet1202(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes, inp_channels=inp_channels, name='ResNet1202')


def get_params_info(param):
    shape = param.shape
    n_layers = 1 if len(shape) > 1 else 0
    n_params = np.prod(shape)
    return shape, n_layers, n_params


def test(net):
    total_params = 0
    total_layers = 0
    for param in net.trainable_params():
        shape, n_layers, n_params = get_params_info(param)
        total_params += n_params
        total_layers += n_layers
    input_tensor = Tensor(np.zeros((1, 3, 32, 32)), mindspore.float32)
    output = net(input_tensor)
    logger.info("Total number of params:", int(total_params))
    logger.info("Total trainable layers:", int(total_layers))
    logger.info(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")


if __name__ == '__main__':
    for model_name in __all__:
        if model_name.startswith('ResNet') and model_name != 'ResNet':
            logger.info(f"\nTesting {model_name}...")
            model = eval(model_name)()
            test(model)
