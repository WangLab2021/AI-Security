"""ResNet in MindSpore.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal
import numpy as np

__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            pad_mode='pad',
            has_bias=False,
            weight_init=HeNormal(),
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False, weight_init=HeNormal()
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    has_bias=False,
                    weight_init=HeNormal(),
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, has_bias=False, weight_init=HeNormal())
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            pad_mode='pad',
            has_bias=False,
            weight_init=HeNormal(),
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, has_bias=False, weight_init=HeNormal())
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    has_bias=False,
                    weight_init=HeNormal(),
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Cell):
    def __init__(self, block, num_blocks, num_classes=10, inp_channels=3, name=None):
        super(ResNet, self).__init__()
        self.model_name = name
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            inp_channels, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False, weight_init=HeNormal()
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Dense(512 * block.expansion, num_classes, weight_init=HeNormal())

        self.relu = nn.ReLU()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(*layers)

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))  # 64,32,32
        b1 = self.layer1(out)
        b2 = self.layer2(b1)  # 128,16,16
        b3 = self.layer3(b2)  # 256,8,8
        b4 = self.layer4(b3)  # 512,4,4
        pool = self.mean(b4, (2, 3))  # 512,1,1
        out = self.flatten(pool)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, inp_channels=inp_channels, name='ResNet18')


def ResNet34(num_classes=10, inp_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, inp_channels=inp_channels, name='ResNet34')


def ResNet50(num_classes=10, inp_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, inp_channels=inp_channels, name='ResNet50')


def ResNet101(num_classes=10, inp_channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, inp_channels=inp_channels, name='ResNet101')


def ResNet152(num_classes=10, inp_channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, inp_channels=inp_channels, name='ResNet152')


def get_params_info(param):
    # param is a Tensor
    shape = param.shape
    n_layers = 1 if len(shape) > 1 else 0
    n_params = np.prod(shape)
    return shape, n_layers, n_params


def test(net):
    total_params = 0
    total_layers = 0
    logger.info(f"Model parameters: {len(net.trainable_params())}")
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


# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# """ResNet."""


# import numpy as np
# import mindspore
# import mindspore.nn as nn
# import mindspore.common.dtype as mstype
# from mindspore.ops import operations as P
# from mindspore.ops import functional as F
# from mindspore.common.tensor import Tensor
# from scipy.stats import truncnorm


# __all__ = [
#     'ResNet18',
#     # 'ResNet34',
#     'ResNet50',
#     'ResNet101',
#     # 'ResNet152',
# ]


# def _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
#     fan_in = in_channel * kernel_size * kernel_size
#     scale = 1.0
#     scale /= max(1.0, fan_in)
#     stddev = (scale**0.5) / 0.87962566103423978
#     mu, sigma = 0, stddev
#     weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
#     weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
#     return Tensor(weight, dtype=mstype.float32)


# def _weight_variable(shape, factor=0.01):
#     init_value = np.random.randn(*shape).astype(np.float32) * factor
#     return Tensor(init_value)


# def _conv3x3(in_channel, out_channel, stride=1, use_se=False):
#     if use_se:
#         weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=3)
#     else:
#         weight_shape = (out_channel, in_channel, 3, 3)
#         weight = _weight_variable(weight_shape)
#     return nn.Conv2dBnAct(
#         in_channel, out_channel, kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight
#     )


# def _conv1x1(in_channel, out_channel, stride=1, use_se=False):
#     if use_se:
#         weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=1)
#     else:
#         weight_shape = (out_channel, in_channel, 1, 1)
#         weight = _weight_variable(weight_shape)
#     return nn.Conv2dBnAct(
#         in_channel, out_channel, kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight
#     )


# def _conv7x7(in_channel, out_channel, stride=1, use_se=False):
#     if use_se:
#         weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=7)
#     else:
#         weight_shape = (out_channel, in_channel, 7, 7)
#         weight = _weight_variable(weight_shape)
#     return nn.Conv2dBnAct(
#         in_channel, out_channel, kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight
#     )


# def _bn(channel):
#     return nn.BatchNorm2d(
#         channel, eps=1e-4, momentum=0.9, gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1
#     )


# def _bn_last(channel):
#     return nn.BatchNorm2d(
#         channel, eps=1e-4, momentum=0.9, gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1
#     )


# def _fc(in_channel, out_channel, use_se=False):
#     if use_se:
#         weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
#         weight = Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=mstype.float32)
#     else:
#         weight_shape = (out_channel, in_channel)
#         weight = _weight_variable(weight_shape)
#     return nn.DenseBnAct(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


# class ResidualBlock(nn.Cell):
#     """
#     ResNet V1 residual block definition.

#     Args:
#         in_channel (int): Input channel.
#         out_channel (int): Output channel.
#         stride (int): Stride size for the first convolutional layer. Default: 1.
#         use_se (bool): enable SE-ResNet50 net. Default: False.
#         se_block(bool): use se block in SE-ResNet50 net. Default: False.

#     Returns:
#         Tensor, output tensor.

#     Examples:
#         >>> ResidualBlock(3, 256, stride=2)
#     """

#     expansion = 4

#     def __init__(self, in_channel, out_channel, stride=1, use_se=False, se_block=False):
#         super(ResidualBlock, self).__init__()
#         self.stride = stride
#         self.use_se = use_se
#         self.se_block = se_block
#         channel = out_channel // self.expansion
#         self.conv1 = _conv1x1(in_channel, channel, stride=1, use_se=self.use_se)
#         self.bn1 = _bn(channel)
#         if self.use_se and self.stride != 1:
#             self.e2 = nn.SequentialCell(
#                 [
#                     _conv3x3(channel, channel, stride=1, use_se=True),
#                     _bn(channel),
#                     nn.ReLU(),
#                     nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same'),
#                 ]
#             )
#         else:
#             self.conv2 = _conv3x3(channel, channel, stride=stride, use_se=self.use_se)
#             self.bn2 = _bn(channel)

#         self.conv3 = _conv1x1(channel, out_channel, stride=1, use_se=self.use_se)
#         self.bn3 = _bn_last(out_channel)
#         if self.se_block:
#             self.se_global_pool = P.ReduceMean(keep_dims=False)
#             self.se_dense_0 = _fc(out_channel, int(out_channel / 4), use_se=self.use_se)
#             self.se_dense_1 = _fc(int(out_channel / 4), out_channel, use_se=self.use_se)
#             self.se_sigmoid = nn.Sigmoid()
#             self.se_mul = P.Mul()
#         self.relu = nn.ReLU()

#         self.down_sample = False

#         if stride != 1 or in_channel != out_channel:
#             self.down_sample = True
#         self.down_sample_layer = None

#         if self.down_sample:
#             if self.use_se:
#                 if stride == 1:
#                     self.down_sample_layer = nn.SequentialCell(
#                         [_conv1x1(in_channel, out_channel, stride, use_se=self.use_se), _bn(out_channel)]
#                     )
#                 else:
#                     self.down_sample_layer = nn.SequentialCell(
#                         [
#                             nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same'),
#                             _conv1x1(in_channel, out_channel, 1, use_se=self.use_se),
#                             _bn(out_channel),
#                         ]
#                     )
#             else:
#                 self.down_sample_layer = nn.SequentialCell(
#                     [_conv1x1(in_channel, out_channel, stride, use_se=self.use_se), _bn(out_channel)]
#                 )
#         self.add = P.TensorAdd()

#     def construct(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         if self.use_se and self.stride != 1:
#             out = self.e2(out)
#         else:
#             out = self.conv2(out)
#             out = self.bn2(out)
#             out = self.relu(out)
#         out = self.conv3(out)
#         out = self.bn3(out)
#         if self.se_block:
#             out_se = out
#             out = self.se_global_pool(out, (2, 3))
#             out = self.se_dense_0(out)
#             out = self.relu(out)
#             out = self.se_dense_1(out)
#             out = self.se_sigmoid(out)
#             out = F.reshape(out, F.shape(out) + (1, 1))
#             out = self.se_mul(out, out_se)

#         if self.down_sample:
#             identity = self.down_sample_layer(identity)

#         out = self.add(out, identity)
#         out = self.relu(out)

#         return out


# class ResNet(nn.Cell):
#     """
#     ResNet architecture.

#     Args:
#         block (Cell): Block for network.
#         layer_nums (list): Numbers of block in different layers.
#         in_channels (list): Input channel in each layer.
#         out_channels (list): Output channel in each layer.
#         strides (list):  Stride size in each layer.
#         num_classes (int): The number of classes that the training images are belonging to.
#         use_se (bool): enable SE-ResNet50 net. Default: False.
#         se_block(bool): use se block in SE-ResNet50 net in layer 3 and layer 4. Default: False.
#     Returns:
#         Tensor, output tensor.

#     Examples:
#         >>> ResNet(ResidualBlock,
#         >>>        [3, 4, 6, 3],
#         >>>        [64, 256, 512, 1024],
#         >>>        [256, 512, 1024, 2048],
#         >>>        [1, 2, 2, 2],
#         >>>        10)
#     """

#     def __init__(self, block, layer_nums, in_channels, out_channels, strides, num_classes, inp_channels=3, use_se=False):
#         super(ResNet, self).__init__()

#         if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
#             raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")
#         self.use_se = use_se
#         self.se_block = False
#         if self.use_se:
#             self.se_block = True

#         if self.use_se:
#             self.conv1_0 = _conv3x3(inp_channels, 32, stride=2, use_se=self.use_se)
#             self.bn1_0 = _bn(32)
#             self.conv1_1 = _conv3x3(32, 32, stride=1, use_se=self.use_se)
#             self.bn1_1 = _bn(32)
#             self.conv1_2 = _conv3x3(32, 64, stride=1, use_se=self.use_se)
#         else:
#             self.conv1 = _conv7x7(inp_channels, 64, stride=2)
#         self.bn1 = _bn(64)
#         self.relu = P.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
#         self.layer1 = self._make_layer(
#             block,
#             layer_nums[0],
#             in_channel=in_channels[0],
#             out_channel=out_channels[0],
#             stride=strides[0],
#             use_se=self.use_se,
#         )
#         self.layer2 = self._make_layer(
#             block,
#             layer_nums[1],
#             in_channel=in_channels[1],
#             out_channel=out_channels[1],
#             stride=strides[1],
#             use_se=self.use_se,
#         )
#         self.layer3 = self._make_layer(
#             block,
#             layer_nums[2],
#             in_channel=in_channels[2],
#             out_channel=out_channels[2],
#             stride=strides[2],
#             use_se=self.use_se,
#             se_block=self.se_block,
#         )
#         self.layer4 = self._make_layer(
#             block,
#             layer_nums[3],
#             in_channel=in_channels[3],
#             out_channel=out_channels[3],
#             stride=strides[3],
#             use_se=self.use_se,
#             se_block=self.se_block,
#         )

#         self.mean = P.ReduceMean(keep_dims=True)
#         self.flatten = nn.Flatten()
#         self.end_point = _fc(out_channels[3], num_classes, use_se=self.use_se)

#     def _make_layer(self, block, layer_num, in_channel, out_channel, stride, use_se=False, se_block=False):
#         """
#         Make stage network of ResNet.

#         Args:
#             block (Cell): Resnet block.
#             layer_num (int): Layer number.
#             in_channel (int): Input channel.
#             out_channel (int): Output channel.
#             stride (int): Stride size for the first convolutional layer.
#             se_block(bool): use se block in SE-ResNet50 net. Default: False.
#         Returns:
#             SequentialCell, the output layer.

#         Examples:
#             >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
#         """
#         layers = []

#         resnet_block = block(in_channel, out_channel, stride=stride, use_se=use_se)
#         layers.append(resnet_block)
#         if se_block:
#             for _ in range(1, layer_num - 1):
#                 resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
#                 layers.append(resnet_block)
#             resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se, se_block=se_block)
#             layers.append(resnet_block)
#         else:
#             for _ in range(1, layer_num):
#                 resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
#                 layers.append(resnet_block)
#         return nn.SequentialCell(layers)

#     def construct(self, x):
#         if self.use_se:
#             x = self.conv1_0(x)
#             x = self.bn1_0(x)
#             x = self.relu(x)
#             x = self.conv1_1(x)
#             x = self.bn1_1(x)
#             x = self.relu(x)
#             x = self.conv1_2(x)
#         else:
#             x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         c1 = self.maxpool(x)

#         c2 = self.layer1(c1)
#         c3 = self.layer2(c2)
#         c4 = self.layer3(c3)
#         c5 = self.layer4(c4)

#         out = self.mean(c5, (2, 3))
#         out = self.flatten(out)
#         out = self.end_point(out)

#         return out


# def ResNet50(num_classes=10, inp_channels=3):
#     """
#     Get ResNet50 neural network.

#     Args:
#         num_classes (int): Class number.

#     Returns:
#         Cell, cell instance of ResNet50 neural network.

#     Examples:
#         >>> net = resnet50(10)
#     """
#     return ResNet(
#         ResidualBlock,
#         [3, 4, 6, 3],
#         [64, 256, 512, 1024],
#         [256, 512, 1024, 2048],
#         [1, 2, 2, 2],
#         num_classes,
#         inp_channels=inp_channels,
#     )


# def ResNet18(num_classes=10, inp_channels=3):
#     """
#     Get ResNet50 neural network.

#     Args:
#         num_classes (int): Class number.

#     Returns:
#         Cell, cell instance of ResNet50 neural network.

#     Examples:
#         >>> net = resnet50(10)
#     """
#     return ResNet(
#         ResidualBlock,
#         [2, 2, 2, 2],
#         [64, 128, 256, 512],
#         [128, 256, 512, 1024],
#         [1, 2, 2, 2],
#         num_classes,
#         inp_channels=inp_channels,
#     )


# def se_resnet50(num_classes=1001, inp_channels=3):
#     """
#     Get SE-ResNet50 neural network.

#     Args:
#         num_classes (int): Class number.

#     Returns:
#         Cell, cell instance of SE-ResNet50 neural network.

#     Examples:
#         >>> net = se-resnet50(1001)
#     """
#     return ResNet(
#         ResidualBlock,
#         [3, 4, 6, 3],
#         [64, 256, 512, 1024],
#         [256, 512, 1024, 2048],
#         [1, 2, 2, 2],
#         num_classes,
#         use_se=True,
#         inp_channels=inp_channels,
#     )


# def ResNet101(num_classes=1001, inp_channels=3):
#     """
#     Get ResNet101 neural network.

#     Args:
#         num_classes (int): Class number.

#     Returns:
#         Cell, cell instance of ResNet101 neural network.

#     Examples:
#         >>> net = resnet101(1001)
#     """
#     return ResNet(
#         ResidualBlock,
#         [3, 4, 23, 3],
#         [64, 256, 512, 1024],
#         [256, 512, 1024, 2048],
#         [1, 2, 2, 2],
#         num_classes,
#         inp_channels=inp_channels,
#     )


# def get_params_info(param):
#     # param is a Tensor
#     shape = param.shape
#     n_layers = 1 if len(shape) > 1 else 0
#     n_params = np.prod(shape)
#     return shape, n_layers, n_params


# def test(net):
#     total_params = 0
#     total_layers = 0
#     logger.info(f"Model parameters: {len(net.trainable_params())}")
#     for param in net.trainable_params():
#         shape, n_layers, n_params = get_params_info(param)
#         total_params += n_params
#         total_layers += n_layers
#     input_tensor = Tensor(np.zeros((1, 3, 32, 32)), mindspore.float32)
#     output = net(input_tensor)
#     logger.info("Total number of params:", int(total_params))
#     logger.info("Total trainable layers:", int(total_layers))
#     logger.info(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")


# if __name__ == '__main__':
#     for model_name in __all__:
#         if model_name.startswith('ResNet') and model_name != 'ResNet':
#             logger.info(f"\nTesting {model_name}...")
#             model = eval(model_name)()
#             test(model)
