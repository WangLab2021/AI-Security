import tensorflow as tf
import PIL
import io
import numpy as np

def jpeg_defense_tf(images, quality):
  with tf.device('/cpu:0'):
    result = tf.map_fn(
        lambda image: tf.image.decode_jpeg(
          tf.image.encode_jpeg(image, quality=quality)),
        tf.cast(tf.round(images), tf.uint8),
        parallel_iterations=64,
        back_prop=False)

    result = tf.cast(result, tf.float32)
    result.set_shape(images.shape.as_list())
    return result

def jpeg_numpy(images, quality):
    images = list(images.round().astype(np.uint8))
    new_images = []
    for image in images:
        image = PIL.Image.fromarray(image)
        buf = io.BytesIO()
        image.save(buf, 'jpeg', quality=int(quality))
        buf.seek(0)
        imarr = np.array(PIL.Image.open(buf), dtype=np.float32)
        new_images.append(imarr)
    return np.array(new_images)

def webp_numpy(images,quality):
    images = list(images.round().astype(np.uint8))
    new_images = []
    for image in images:
        image = PIL.Image.fromarray(image)
        buf = io.BytesIO()
        image.save(buf, 'webp', quality=int(quality))
        buf.seek(0)
        imarr = np.array(PIL.Image.open(buf), dtype=np.float32)
        new_images.append(imarr)
    return np.array(new_images)

def jp2_numpy(images,quality):
    images = list(images.round().astype(np.uint8))
    new_images = []
    for image in images:
        image = PIL.Image.fromarray(image)
        buf = io.BytesIO()
        image.save(buf, 'jpeg2000', irreversible=False,quality_mode='db',quality_layers=[quality])
        buf.seek(0)
        imarr = np.array(PIL.Image.open(buf), dtype=np.float32)
        new_images.append(imarr)
    return np.array(new_images)

def webp_defense(images,quality):
  return tf.py_func(webp_numpy,[images,quality],[tf.float32],stateful=False)[0]

def jpeg_defense(images, quality):
  return tf.py_func(
      jpeg_numpy, [images, quality], [tf.float32], stateful=False)[0]

def jpeg2000_defense(images, quality):
  return tf.py_func(
      jp2_numpy, [images, quality], [tf.float32], stateful=False)[0]


