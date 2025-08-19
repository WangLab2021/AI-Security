import tensorflow as tf
from tensorflow.keras import layers, models, initializers

__all__ = [
    'ResNet20',
    'ResNet32',
    'ResNet44',
    'ResNet56',
    'ResNet110',
    'ResNet1202',
]


# Option A shortcut for CIFAR10 as described in the original paper
class LambdaLayer(layers.Layer):
    def __init__(self, out_channels, stride):
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride

    def call(self, x):
        # Downsample spatial size by stride
        x = x[:, :: self.stride, :: self.stride, :]
        # Pad channel dimension
        ch = x.shape[-1]
        ch_pad = self.out_channels - ch
        pad1 = ch_pad // 2
        pad2 = ch_pad - pad1
        paddings = tf.constant([[0, 0], [0, 0], [0, 0], [pad1, pad2]])
        return tf.pad(x, paddings, "CONSTANT")


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super().__init__()
        self.conv1 = layers.Conv2D(
            planes,
            kernel_size=3,
            strides=stride,
            padding='same',
            use_bias=False,
            kernel_initializer=initializers.HeNormal(),
        )
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            planes, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=initializers.HeNormal()
        )
        self.bn2 = layers.BatchNormalization()
        self.option = option
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(planes, stride)
            elif option == 'B':
                self.shortcut = tf.keras.Sequential(
                    [
                        layers.Conv2D(
                            planes,
                            kernel_size=1,
                            strides=stride,
                            use_bias=False,
                            kernel_initializer=initializers.HeNormal(),
                        ),
                        layers.BatchNormalization(),
                    ]
                )
        else:
            self.shortcut = lambda x: x

    def call(self, x, training=False):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        shortcut = self.shortcut(x)
        out = layers.add([out, shortcut])
        out = tf.nn.relu(out)
        return out


class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10, inp_channels=1, option='A', name=None):
        super().__init__(name=name)
        self.in_planes = 16

        self.conv1 = layers.Conv2D(
            16,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            input_shape=(None, None, inp_channels),
            kernel_initializer=initializers.HeNormal(),
        )
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, option=option)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, option=option)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, option=option)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, kernel_initializer=initializers.HeNormal())

    def _make_layer(self, block, planes, num_blocks, stride, option='A'):
        strides = [stride] + [1] * (num_blocks - 1)
        layers_ = []
        for s in strides:
            layers_.append(block(self.in_planes, planes, stride=s, option=option))
            self.in_planes = planes * block.expansion
        return tf.keras.Sequential(layers_)

    def call(self, x, training=False):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)
        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.avgpool(out)
        out = self.fc(out)
        return out

    # def build(self, input_shape):
    #     # 调用一次 __call__ 即可自动 build 各层
    #     if self.built:
    #         return
    #     super().build(input_shape)
    #     dummy = tf.keras.Input(shape=input_shape[1:])
    #     self(dummy, training=False)
    #     self.built = True  # 标记模型已构建


def ResNet20(num_classes=10, inp_channels=1):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, inp_channels=inp_channels, name='ResNet20')


def ResNet32(num_classes=10, inp_channels=1):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, inp_channels=inp_channels, name='ResNet32')


def ResNet44(num_classes=10, inp_channels=1):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, inp_channels=inp_channels, name='ResNet44')


def ResNet56(num_classes=10, inp_channels=1):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, inp_channels=inp_channels, name='ResNet56')


def ResNet110(num_classes=10, inp_channels=1):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, inp_channels=inp_channels, name='ResNet110')


def ResNet1202(num_classes=10, inp_channels=1):
    return ResNet(
        BasicBlock, [200, 200, 200], num_classes=num_classes, inp_channels=inp_channels, name='ResNet1202'
    )


def test(net):
    # net.build(input_shape=(None, 32, 32, 1))
    dummy_input = tf.zeros((1, 32, 32, 1), dtype=tf.float32)
    net(dummy_input)
    net.summary()
    total_params = net.count_params()
    logger.info("Total number of params", total_params)
    # Count only conv and dense layers as "layers"
    total_layers = sum([1 for v in net.trainable_variables if len(v.shape) > 1])
    logger.info("Total layers", total_layers)


if __name__ == "__main__":
    for net_fn in __all__:
        fn = globals()[net_fn]
        logger.info(fn.__name__)
        test(fn())
        logger.info()
