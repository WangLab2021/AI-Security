import io
import sys

import numpy as np
import tensorflow as tf
import PIL.Image

sys.path.insert(0, 'slim')
from nets import nets_factory
from preprocessing import preprocessing_factory
# from tensorflow.contrib.slim.nets import nets_factory
# from tensorflow.contrib.slim.preprocessing import preprocessing_factory

import jpeg

def vgg_normalization(image):
  return image - [123.68, 116.78, 103.94]


def inception_normalization(image):
  return ((image / 255.) - 0.5) * 2


normalization_fn_map = {
    'inception': inception_normalization,
    'inception_v1': inception_normalization,
    'inception_v2': inception_normalization,
    'inception_v3': inception_normalization,
    'inception_v4': inception_normalization,
    'inception_resnet_v2': inception_normalization,
    'mobilenet_v1': inception_normalization,
    'nasnet_mobile': inception_normalization,
    'nasnet_large': inception_normalization,
    'resnet_v1_50': vgg_normalization,
    'resnet_v1_101': vgg_normalization,
    'resnet_v1_152': vgg_normalization,
    'resnet_v1_200': vgg_normalization,
    'resnet_v2_50': inception_normalization,
    'resnet_v2_101': inception_normalization,
    'resnet_v2_152': inception_normalization,
    'resnet_v2_200': inception_normalization,
    'vgg': vgg_normalization,
    'vgg_a': vgg_normalization,
    'vgg_16': vgg_normalization,
    'vgg_19': vgg_normalization,
}


def batch(iterable, size):
  iterator = iter(iterable)
  batch = []
  while True:
    try:
      batch.append(next(iterator))
    except StopIteration:
      if batch:
        yield batch
      return

    if len(batch) == size:
      yield batch
      batch = []


def load_image(fn, image_size):
  # Resize the image appropriately first
  image = PIL.Image.open(fn)
  image = image.convert('RGB')
  image = image.resize((image_size, image_size), PIL.Image.BILINEAR)
  image = np.array(image, dtype=np.float32)
  return image



def differentiable_jpeg(image, quality):
  return jpeg.jpeg_compress_decompress(
      image, rounding=jpeg.diff_round, factor=jpeg.quality_to_factor(quality))


def create_model(name):
  offset = {
    'inception': 1,
    'inception_v1': 1,
    'inception_v2': 1,
    'inception_v3': 1,
    'inception_v4': 1,
    'inception_resnet_v2': 1,
    'mobilenet_v1': 1,
    'nasnet_mobile': 1,
    'nasnet_large': 1,
    'resnet_v1_50': 0,
    'resnet_v1_101': 0,
    'resnet_v1_152': 0,
    'resnet_v1_200': 0,
    'resnet_v2_50': 1,
    'resnet_v2_101': 1,
    'resnet_v2_152': 1,
    'resnet_v2_200': 1,
    'vgg': 0,
    'vgg_a': 0,
    'vgg_16': 0,
    'vgg_19': 0,
  }[name]
  num_classes = 1000 + offset

  normalization_fn = normalization_fn_map[name]
  network_fn = nets_factory.get_network_fn(
      name, num_classes=num_classes, is_training=False)
  image_size = network_fn.default_image_size

  return normalization_fn, network_fn, image_size, offset
