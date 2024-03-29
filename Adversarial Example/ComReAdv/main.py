import tensorflow as tf
# from keras.applications.resnet50 import ResNet50,preprocess_input
# from keras.applications.inception_v3 import InceptionV3,preprocess_input
import numpy as np
from scipy import misc
from keras.layers import Reshape
from keras.models import Input
import time
from PIL import Image

import utils
import defense
import ComModel

import os
import io

os.environ['CUDA_VISIBLE_DEVICES']="0"

def FGSM(x, target, logits, ord=np.inf, alpha=1):

    loss = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target)
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        normalized_grad = tf.sign(grad)
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        abs_sum = tf.reduce_sum(tf.abs(grad), [0, 1, 2])
        normalized_grad = grad / abs_sum
    elif ord == 2:
        square = tf.reduce_sum(tf.square(grad), [0, 1, 2])
        normalized_grad = grad / tf.sqrt(square)

    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # scaled_grad = alpha * normalized_grad
    # adv_x = x + scaled_grad
    # adv_x = tf.clip_by_value(adv_x, 0, 255)

    return normalized_grad


def make_network_fn(network_fn,input_shape,output_shape):
    def fn(images):
        images.set_shape(input_shape)
        logits,_=network_fn(images)
        return logits
    return fn




def main():

    tf.get_variable_scope()._reuse=tf.AUTO_REUSE
    
    eps=3
    iters=10
    alpha=eps/iters
    
    model_name='resnet_v1_50'

    normalization_fn, network_fn, image_size, offset = utils.create_model(model_name)
    num_classes = 1000 + offset
    # image_size = network_fn.default_image_size
    logits_fn = make_network_fn(lambda image: network_fn(normalization_fn(image)),
                              [None, image_size, image_size,3], [None, num_classes])
    pred_fn = lambda image: tf.argmax(logits_fn(image), axis=1)


    image_ph=tf.placeholder(tf.float32,[None,image_size,image_size,3])
    label_ph=tf.placeholder(tf.int32,[None])

    logit=logits_fn(image_ph)
    pred=pred_fn(image_ph)

    def_pred=pred_fn(defense.jpeg_defense(image_ph,quality=75))

    # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        model_vars = tf.contrib.framework.get_variables_to_restore()
        saver = tf.train.Saver(model_vars)
        saver.restore(sess,'./checkpoint/'+model_name+'.ckpt')


        input_img=Input(shape=(224,224,3))
        model_c=ComModel.build_model(input_img)
        # model.summary()
        att_img=image_ph/255.
        att_img=model_c(att_img)
        att_img=Reshape(target_shape=(224,224,3))(att_img)
        att_img=att_img*255.
        att_img=tf.clip_by_value(att_img,0,255)
        att_logit=logits_fn(att_img)
        att_pred=pred_fn(att_img)
        model_c.load_weights('./model_weights/jpeg_75/model_weights.h5')

        att_logit_2=logits_fn(utils.differentiable_jpeg(image_ph,quality=75))
        att_pred_2=pred_fn(utils.differentiable_jpeg(image_ph,quality=75))

        fgsm=FGSM(image_ph,label_ph,att_logit,alpha=alpha)

        count=0
        for i in range(1,1001,1):
            try:
                print(i,end=' ')
                image_path='./data/test/{0}.png'.format(i)
                # image=misc.imread(image_path)
                # image=misc.imresize(image,(image_size,image_size))
                # image=image.reshape((1,image_size,image_size,3))
                image=Image.open(image_path)
                image=image.resize((image_size,image_size))
                image=np.array(image)
                image=image.reshape((1,image_size,image_size,3))
            except:
                print('read image error')
                continue

            ori=sess.run(pred,feed_dict={image_ph:image})
            print('ori:',ori,end=' ')

            ori_image=image.copy()
            targets=(ori+500)%1000
            print('target:',targets,end=' ')

            #FGSM
            # grad = sess.run(fgsm, feed_dict={image_ph: image, label_ph: ori})
            # adv = image + grad * eps
            # image = np.clip(adv, 0, 255)
            # #
            #BIM
            # for j in range(iters):
            #     grad=sess.run(fgsm,feed_dict={image_ph:image,label_ph:targets})
            #     adv=image+grad*alpha
            #     eta=adv-ori_image
            #     eta=np.clip(eta,-eps,eps)
            #     image=np.clip(ori_image+eta,0,255)

            #MIM
            m=np.zeros_like(image)
            decay=1
            for j in range(iters):
                grad=sess.run(fgsm,feed_dict={image_ph:image,label_ph:targets})
                m=m*decay+grad
                adv=m*alpha+image
                eta=adv-ori_image
                eta=np.clip(eta,-eps,eps)
                image=np.clip(ori_image+eta,0,255)

            image = np.clip(np.round(image), 0, 255)
            image = image.astype(np.uint8)

            d_pred=sess.run(def_pred,feed_dict={image_ph:image})
            print("defense result:",d_pred,end=' ')

            if d_pred==targets:
                print('√', end=' ')
                count+=1
            print()

        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print(count)


if __name__ == '__main__':
    main()
