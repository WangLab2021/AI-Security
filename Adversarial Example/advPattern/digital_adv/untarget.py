#首先测试我们resize的数据集的效果:把5个人随机挑选一部分作为probe，其余人并入gallery做galery
# 有必要的话做transfer learning
#接着搜集mask、trans的信息
#写adv代码
#10.23 上述全部完成

import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import numpy as np
import scipy.misc
import json
from matplotlib import pyplot as plt

import time
import os
import sys
sys.path.append('../')
from baseline_dis.evaluate_v2 import extract_feature, sort_similarity, map_rank_quick_eval
from utils.file_helper import write
from utils.file_cpy import filecopy, rm_jpg

#dir_path为路径，img_names为待load的照片名list（用的时候是dir下全体照片的名字）
#index有值的时候，仅load index；否则load全体img_names
#load函数使用的是tf.image.resize_images
# Noted that get tf.image.resize_images and preprocess operations
def load(dir_path, img_names, index=None):
    image, infos = load_raw(dir_path, img_names, index)

    img1_tf = tf.placeholder(shape=(None, 550, 220, 3), dtype='float32')
    img2_tf = tf.image.resize_images(img1_tf, [224, 224])
    img3_tf = preprocess_input(img2_tf)

    with tf.Session() as sess:
        image = sess.run(img3_tf, feed_dict={img1_tf:image})

    return image, infos

#相比与load()，没有resize和preprocessing
def load_raw(dir_path, img_names, index=None):
    if index is None:
        index = np.arange(len(img_names))

    image=[]
    infos = []

    for i in index:
        image_path = os.path.join(dir_path, img_names[i])
        x = np.array(scipy.misc.imread(image_path),dtype=np.float32)
        image.append(x)

        arr = img_names[i].split('_')
        person = int(arr[0])
        camera = int(arr[1][1])
        infos.append((person, camera))

    image = np.array(image)
    return image, infos

def evaluate(net, probe_path, gallery_path, adv_id, adv_index):

    result = np.zeros(4, dtype=np.float32)
    num_query = len(adv_index)

    for i in range(num_query):
        tmp_adv_index = np.copy(adv_index)

        probe_index = tmp_adv_index[i]
        tmp_adv_index[i] = tmp_adv_index[0]
        tmp_adv_index[0] = probe_index

        tmp_result = single_evaluate(net, probe_path, gallery_path, adv_id, tmp_adv_index)
        result = result + tmp_result

    result = result /num_query

    print('result')
    print('Rank 1:\t%f, Rank 5:\t%f, Rank 10:\t%f' % (result[0], result[1], result[2]))
    print('mAP:\t%f' % result[3])

# evaluate a certain person's CMC and mAP of our made dataset "BoBo"
# different from evaluate(), tBobo is loaded using tf.image.resize_image()
# the Market-1501 is loaded using scipy.misc.imresize()
def single_evaluate(net, probe_path, gallery_path, adv_id, adv_index):
    img_names = sorted(os.listdir(probe_path))
    imgs, infos = load(probe_path, img_names)

    #probe
    adv_probe_imgs = imgs[adv_index[0:1]]
    adv_probe_infos = [infos[i] for i in adv_index[0:1]]

    #gallery
    num_per_pc = 6
    gal_add_pid = sorted(list(set(range(1502,1513))-{adv_id}))
    otherpid_index = np.random.randint(0, 24, size=3 * (11-1) * num_per_pc) + \
                     np.array([np.arange(3)+(i-1502)*3 for i in gal_add_pid]).flatten().repeat(num_per_pc)*24

    gal_add_index = np.concatenate(
        (adv_index[1:], otherpid_index)
    )

    gallery_imgs_add = imgs[gal_add_index]
    gallery_infos_add = [infos[i] for i in gal_add_index]

    result = eval_without_noise(net, adv_probe_imgs, adv_probe_infos, gallery_imgs_add, gallery_infos_add)
    return result

def eval_without_noise(net, probe, probe_infos, gallery_imgs_add, gallery_infos_add):
    net.load_weights('../baseline_dis/market-pair-pretrain-withoutwarp.h5')

    #calculate the features, infos of gallery,probe
    gallery_add_f = net.predict(gallery_imgs_add, batch_size=128)
    gallery_f = np.load('gallery_f.npy') #本地持久化，节约时间
    gallery_info = [tuple(x) for x in json.load(open('gallery_info.json', 'r'))]

    gallery_f = np.concatenate((gallery_f, gallery_add_f), axis=0)
    gallery_info.extend([info for info in gallery_infos_add]) #由于info是元组的list

    query_f = net.predict(probe, batch_size=128)
    query_info = probe_infos

    # do the eval for CMC and mAP
    result, result_argsort = sort_similarity(query_f, gallery_f)
    log_path = 'target_result_eval.log'
    rank1_acc, rank5_acc, rank10_acc, mAP = map_rank_quick_eval(query_info, gallery_info, result_argsort)
    #write(log_path, '%f\t%f\t%f\t%f\n' % (rank1_acc, rank5_acc, rank10_acc, mAP))

    return [rank1_acc, rank5_acc, rank10_acc, mAP]


def eucl_dist(inputs):
    x, y = inputs
    # return K.mean(K.square((x - y)), axis=1)
    return tf.square((x - y))

LEARNING_RATE = 1e-2
MAX_ITERATIONS = 800
wights_VT = 6e-5

#mask表示对a0图的加噪区域
#img1,img2表示用来计算adv用的集合.读入的是未resize，未preprocess的图像
#img1,img2是pair对，表示同一个人不同C下的pair
#trans1表示img1每张图相对a0的transform集合；同理trans2
def adv(net, mask, generator, batch_size):

    #初始化噪声出纯色（目前是灰色）
    #modifier = np.ones((550,220,3),dtype=np.float32)*(-1)

    r=np.ones((550, 220, 1), dtype=np.float32) * (1)
    g=np.ones((550, 220, 1), dtype=np.float32) * (1)
    b=np.ones((550, 220, 1), dtype=np.float32) * (1)
    modifier = np.concatenate((r,g,b), axis=2)

    modifier = tf.Variable(modifier, name='modifier')
    noise = (tf.tanh(modifier)+1)*255.0/2

    x1 = tf.placeholder(shape=(None,550, 220, 3), dtype='float32')
    transform1 = tf.placeholder(shape=(None, 8), dtype='float32')
    x2 = tf.placeholder(shape=(None,550, 220, 3), dtype='float32')
    transform2 = tf.placeholder(shape=(None, 8), dtype='float32')

    noise_patch = tf.stack([noise*mask] * batch_size)

    noise1 = tf.contrib.image.transform(noise_patch, transform1, 'BILINEAR') #这里transform可能要转置
    noise2 = tf.contrib.image.transform(noise_patch, transform2, 'BILINEAR')

    x1_adv_o = x1*(tf.cast(noise1<1e-4, dtype=tf.float32)) + noise1 #the function output
    x2_adv_o = x2*(tf.cast(noise2<1e-4, dtype=tf.float32)) + noise2

    x1_adv = tf.image.resize_images(x1_adv_o, [224,224]) #这里的双向线性插值是默认值可以丢掉
    x2_adv = tf.image.resize_images(x2_adv_o, [224,224])

    x1_adv = preprocess_input(x1_adv)
    x2_adv = preprocess_input(x2_adv)

    feature1 = tf.squeeze(net(x1_adv),axis=[1,2])
    feature2 = tf.squeeze(net(x2_adv),axis=[1,2])

    feature1_norm = tf.nn.l2_normalize(feature1, axis=1)
    feature2_norm = tf.nn.l2_normalize(feature2, axis=1)

    #计算两个batch的feature的距离，不是直接的矩阵相乘
    #dist_batch = tf.matmul(feature1_norm, feature2_norm, transpose_a=False, transpose_b=True)
    #dist = tf.reduce_sum(dist_batch, axis=[0])
    dist = tf.reduce_sum(tf.multiply(feature1_norm, feature2_norm))

    def total_variation(x):
        a = tf.square((x[:, :550 - 1, :220 - 1, :] - x[:, 1:, :220 - 1, :]) / 255.0)
        b = tf.square((x[:, :550 - 1, :220 - 1, :] - x[:, :550 - 1, 1:, :]) / 255.0)
        return tf.reduce_sum(tf.sqrt(a + b + 1e-8))

    tv = wights_VT * total_variation(noise_patch)
    total_loss = dist + tv

    # Setup the adam optimizer and keep track of variables we're creating
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train = optimizer.minimize(total_loss, var_list=[modifier]) #是dist而不是-dist
    init = tf.global_variables_initializer()

    #Run the attack
    # method2: BIM
    # w_grads = tf.gradients(dist, modifier)

    #method1: optimization
    with tf.Session() as sess:
        sess.run(init)
        net.load_weights('../baseline_dis/market-pair-pretrain-withoutwarp.h5')
        for iteration in range(MAX_ITERATIONS):
            img1, img2, trans1, trans2 = next(generator)
            _, dist_np, tv_np, loss_np = sess.run([train, dist, tv, total_loss],
                                                  feed_dict={x1:img1, x2:img2,
                                                             transform1:trans1,
                                                             transform2:trans2})

            if iteration % (MAX_ITERATIONS // 10) == 0:
                print(iteration, [dist_np, tv_np, loss_np])
        noise_np = sess.run(noise)

    return noise_np

def pair_generator(imgs, infos, trans, batch_size):
    camera_id = [x[1] for x in infos] #infos[(pid, cid)]

    while True:
        left_index=[]
        right_index=[]
        tmp_len=0

        while True:
            tmp = np.random.randint(len(imgs), size=2)
            if camera_id[tmp[0]] != camera_id[tmp[1]]:
                left_index.append(tmp[0])
                right_index.append(tmp[1])
                tmp_len+=1
            else:
                continue

            if tmp_len == batch_size:
                break

        yield imgs[left_index], imgs[right_index], trans[left_index], trans[right_index]


def attack(net, probe_path, mask_path, adv_path, noise_path, adv_id, adv_index, batch_size=4):
    #load mask
    mask = np.load(os.path.join(mask_path, str(adv_id)+'.npy'))

    #load imgs, infos, trans
    img_names = sorted(os.listdir(probe_path))
    imgs, infos = load_raw(probe_path, img_names)
    trans = np.load(os.path.join(mask_path,'transforms.npy'))  #trans.npy是包含了792张照片的trans，(792，8)

    adv_imgs = imgs[adv_index]
    adv_infos = [infos[i] for i in adv_index]
    adv_trans = trans[adv_index]

    #generate the pair
    generator = pair_generator(adv_imgs, adv_infos, adv_trans, batch_size)

    noise = adv(net, mask, generator, batch_size)
    np.save(os.path.join(noise_path, str(adv_id)+'_noise.npy'),noise)

    '''
    save the adv
    '''
    save_adv(adv_index, imgs, trans, mask, noise, img_names, adv_path)
    rm_jpg(adv_path)


'''
save the perturbed imgs to dir_path
imgs: raw size, without any process
'''
def save_adv(index, imgs_raw, trans_all, mask, noise, img_names, dir_path):
    img_tf = tf.placeholder(shape=(550, 220, 3), dtype='float32')
    transform_tf = tf.placeholder(shape=8, dtype='float32')
    noise_tmp = tf.contrib.image.transform(noise * mask, transform_tf, 'BILINEAR')
    adv_tf = img_tf * (tf.cast(noise_tmp < 1e-1, dtype=tf.float32)) + noise_tmp

    with tf.Session() as sess:
        for i in index:
            #calculate the i_th perturbed image
            adv_tmp = sess.run(adv_tf, feed_dict={img_tf:imgs_raw[i], transform_tf:trans_all[i]})

            adv_tmp = adv_tmp.astype(np.uint8)
            img_path_tmp = os.path.join(dir_path, img_names[i].replace('JPG','png'))
            scipy.misc.imsave(img_path_tmp, adv_tmp)


if __name__ == '__main__':
    with tf.variable_scope('base_model'):
        net = load_model('../baseline_dis/market-pair-pretrain-withoutwarp.h5')

    gallery_path = '../../dataset' + '/Market-1501/bounding_box_test'
    probe_path = '../../dataset' + '/bobo/resize'
    mask_path = '../../dataset' + '/bobo/mask'
    adv_path = '../../dataset' + '/bobo/untar_adv'

    noise_path = '../../dataset' +'/bobo/untar_noise'

    #generate adv, saved in adv_path
    adv_id = 1502
    adv_index = np.array([54,68, 30, 50, 20, 18, 0, 4, 8, 12, 24, 28, 32, 40, 44, 46,  58, 62, 70])

    #attack(net, probe_path, mask_path, adv_path, noise_path, adv_id, adv_index, batch_size=8)

    evaluate(net, adv_path, gallery_path, adv_id, adv_index)
    #evaluate(net, probe_path, gallery_path, adv_id, adv_index)