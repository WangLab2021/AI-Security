import tensorflow as tf
from tensorflow.keras import layers, models

__all__ = [
    'ResNet18',
    'ResNet34',
    'ResNet50',
    'ResNet101',
    'ResNet152',
]


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(planes, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = tf.keras.Sequential(
                [
                    layers.Conv2D(self.expansion * planes, kernel_size=1, strides=stride, use_bias=False),
                    layers.BatchNormalization(),
                ]
            )

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        out += self.shortcut(x, training=training) if hasattr(self, 'shortcut') else x
        out = tf.nn.relu(out)
        return out


class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = layers.Conv2D(planes, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion * planes, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = tf.keras.Sequential(
                [
                    layers.Conv2D(self.expansion * planes, kernel_size=1, strides=stride, use_bias=False),
                    layers.BatchNormalization(),
                ]
            )

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training=training))
        out = self.bn3(self.conv3(out), training=training)
        out += self.shortcut(x, training=training) if hasattr(self, 'shortcut') else x
        out = tf.nn.relu(out)
        return out

    # def build(self, input_shape):
    #     # 调用一次 __call__ 即可自动 build 各层
    #     super().build(input_shape)
    #     dummy = tf.keras.Input(shape=input_shape[1:] if len(input_shape) > 4 else input_shape)
    #     self(dummy, training=False)


class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10, name=None, inp_channels=1):
        super(ResNet, self).__init__(name=name)
        self.in_planes = 64
        self.conv1 = layers.Conv2D(
            64, kernel_size=3, strides=1, padding='same', use_bias=False, input_shape=(None, None, inp_channels)
        )
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.flatten = layers.Flatten()
        self.linear = layers.Dense(num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers_ = []
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers_.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return tf.keras.Sequential(layers_)

    # def build(self, input_shape):
    #     # 调用一次 __call__ 即可自动 build 各层
    #     if self.built:
    #         return
    #     super().build(input_shape)
    #     dummy = tf.keras.Input(shape=input_shape[1:])
    #     self(dummy, training=False)
    #     self.built = True

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        b1 = self.layer1(out, training=training)
        b2 = self.layer2(b1, training=training)
        b3 = self.layer3(b2, training=training)
        b4 = self.layer4(b3, training=training)
        pool = tf.keras.layers.GlobalAveragePooling2D()(b4)
        out = self.linear(pool)
        return out


def ResNet18(num_classes=10, inp_channels=1):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, name='ResNet18', inp_channels=inp_channels)


def ResNet34(num_classes=10, inp_channels=1):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, name='ResNet34', inp_channels=inp_channels)


def ResNet50(num_classes=10, inp_channels=1):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, name='ResNet50', inp_channels=inp_channels)


def ResNet101(num_classes=10, inp_channels=1):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, name='ResNet101', inp_channels=inp_channels)


def ResNet152(num_classes=10, inp_channels=1):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, name='ResNet152', inp_channels=inp_channels)


def test(net):
    net.build(input_shape=(None, 32, 32, 1))
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
