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
from untarget import load, load_raw, eucl_dist


#loading the index, infos of the target person
def load_market(dir_path, target_id):
    img_names = sorted(os.listdir(dir_path))
    target_index = []
    target_infos = []

    for i in range(len(img_names)):
        img_name = img_names[i]

        if '.txt' in img_name or '.db' in img_name:
            continue
        if 'f' in img_name or 's' in img_name:
            arr = img_name.split('_')
            person = int(arr[0])
            camera = int(arr[1][1])

            if person == target_id:
                target_index.append(i)
                target_infos.append((person, camera))

    return target_index, target_infos

LEARNING_RATE = 1e-2
MAX_ITERATIONS = 1200
weights_TV = 0e-4
weights_untarget = 0.52
weights_PEAK = 0.4
EARLY_STOP_PATIENCE = 300

#mask表示对a0图的加噪区域
#img1,img2表示用来计算adv用的集合.读入的是未resize，未preprocess的图像
#img1,img2是pair对，表示同一个人不同C下的pair
#trans1表示img1每张图相对a0的transform集合；同理trans2
def adv(net, mask, generator, batch_size):

    #初始化噪声出纯色（目前是灰色）
    #modifier = np.ones((550,220,3),dtype=np.float32)*(-1)

    r=np.ones((550, 220, 1), dtype=np.float32) * (0)
    g=np.ones((550, 220, 1), dtype=np.float32) * (0)
    b=np.ones((550, 220, 1), dtype=np.float32) * (0)
    modifier = np.concatenate((r,g,b), axis=2)

    #read from the file
    #noise = np.load(os.path.join(noise_path, '{}to{}_noise.npy'.format(adv_id, target_id)))
    #modifier = np.tan(noise*2/255.0-1)

    modifier = tf.Variable(modifier, name='modifier')
    noise = (tf.tanh(modifier)+1)*255.0/2

    x1 = tf.placeholder(shape=(None,550, 220, 3), dtype='float32')
    transform1 = tf.placeholder(shape=(None, 8), dtype='float32')
    x2 = tf.placeholder(shape=(None,550, 220, 3), dtype='float32')
    transform2 = tf.placeholder(shape=(None, 8), dtype='float32')

    target_f_tf = tf.placeholder(shape=(None, 1, 1, 2048), dtype='float32')

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
    feature0 = tf.squeeze(target_f_tf,axis=[1,2])

    feature1_norm = tf.nn.l2_normalize(feature1, axis=1)
    feature2_norm = tf.nn.l2_normalize(feature2, axis=1)
    feature0_norm = tf.nn.l2_normalize(feature0, axis=1)

    #计算两个batch的feature的距离，不是直接的矩阵相乘
    #dist_batch = tf.matmul(feature1_norm, feature2_norm, transpose_a=False, transpose_b=True)
    #dist = tf.reduce_sum(dist_batch, axis=[0])
    dist1 = tf.reduce_sum(tf.multiply(feature0_norm, feature1_norm))
    dist2 = tf.reduce_sum(tf.multiply(feature0_norm, feature2_norm))
    dist3 = tf.reduce_sum(tf.multiply(feature1_norm, feature2_norm))

    def total_variation(x):
        a = tf.square((x[:, :550 - 1, :220 - 1, :] - x[:, 1:, :220 - 1, :]) / 255.0)
        b = tf.square((x[:, :550 - 1, :220 - 1, :] - x[:, :550 - 1, 1:, :]) / 255.0)
        return tf.reduce_sum(tf.sqrt(a + b + 1e-8))

    tv = weights_TV * total_variation(noise_patch)
    total_loss = -(dist1 + dist2) + weights_untarget*dist3 + tv

    oft1 = tf.reduce_sum(tf.multiply(feature0_norm, feature1_norm), axis=[1])
    oft2 = tf.reduce_sum(tf.multiply(feature0_norm, feature2_norm), axis=[1])
    peak_tf = tf.reduce_max(oft1) + tf.reduce_max(oft2)
    total_loss += -weights_PEAK * peak_tf

    # Setup the adam optimizer and keep track of variables we're creating
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train = optimizer.minimize(total_loss, var_list=[modifier]) #是dist而不是-dist
    init = tf.global_variables_initializer()

    #optimization
    best_dist = 0
    best_epoch = 0
    best_noise = None

    with tf.Session() as sess:
        sess.run(init)
        net.load_weights('../baseline_dis/market-pair-pretrain-withoutwarp.h5')
        for iteration in range(MAX_ITERATIONS):
            target_f, adv1_img, adv2_img, trans1, trans2 = next(generator)

            dist1_np, dist2_np, dist3_np, tv_np, peak_np, total_loss_np, _ = \
                sess.run([dist1, dist2, dist3, tv, peak_tf, total_loss, train], feed_dict={x1:adv1_img, x2:adv2_img,
                                                                              transform1:trans1,
                                                                              transform2:trans2,
                                                                              target_f_tf: target_f})

            if iteration % 100 == 0:
                print(iteration, [dist1_np+dist2_np, dist3_np, tv_np, peak_np, total_loss_np])

            if dist1_np+dist2_np-dist3_np > best_dist:
                best_dist = dist1_np+dist2_np-dist3_np
                best_epoch = iteration
                best_noise = sess.run(noise)

                log = (iteration, [dist1_np + dist2_np, dist3_np, tv_np, peak_np, total_loss_np])
            elif iteration - best_epoch > EARLY_STOP_PATIENCE:
                break
        print(log)

    return best_noise

#adv imgs only from 7, 8 cameras
def pair_generator(adv_imgs, adv_infos, trans, target_f, batch_size):
    adv_c7 = [i for i in range(len(adv_infos)) if adv_infos[i][1] == 7]
    adv_c8 = [i for i in range(len(adv_infos)) if adv_infos[i][1] == 8]
    adv_c9 = [i for i in range(len(adv_infos)) if adv_infos[i][1] == 9]
    adv_c_list = [adv_c7, adv_c8, adv_c9]

    while True:
        target_index = np.random.randint(0, len(target_f), batch_size)
        adv1_index = np.random.choice(adv_c_list[0], batch_size)
        adv2_index = np.random.choice(adv_c_list[1], batch_size)

        yield target_f[target_index], \
              adv_imgs[adv1_index], adv_imgs[adv2_index], \
              trans[adv1_index], trans[adv2_index]


def attack(net, probe_path, gallery_path, mask_path, adv_path, noise_path,
           adv_index, adv_id, target_id, num_target=6, batch_size=8):

    #load imgs, infos, trans of adv
    img_names = sorted(os.listdir(probe_path))
    imgs, infos = load_raw(probe_path, img_names)

    adv_imgs = imgs[adv_index]
    adv_infos = [infos[i] for i in adv_index]
    trans = np.load(os.path.join(mask_path, 'transforms.npy'))[adv_index]  # trans.npy是包含了792张照片的trans，(792，8)

    #load the target images' info and features
    target_index, _ = load_market(gallery_path, target_id)
    gallery_f = np.load('gallery_f.npy') #本地持久化，节约时间
    target_f = gallery_f[target_index[0:num_target]]  #the first num_target imgs are assumed as the target imgs

    #generate the pair
    generator = pair_generator(adv_imgs, adv_infos, trans, target_f, batch_size)

    #load mask
    mask = np.load(os.path.join(mask_path, str(adv_id)+'.npy'))

    noise = adv(net, mask, generator, batch_size)
    np.save(os.path.join(noise_path, '{}to{}_noise.npy'.format(adv_id, target_id)), noise)

    '''
    show the adv
    '''
    #plt.subplot(121), plt.imshow(noise), plt.title('Input')
    #plt.show()

    '''
    save the adv
    '''
    trans_all = np.load(os.path.join(mask_path, 'transforms.npy'))
    save_adv(adv_index, imgs, trans_all, mask, noise, img_names, adv_path)
    rm_jpg(adv_path)


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

def evaluate(net, probe_path, gallery_path, adv_index, adv_id, target_id, num_target):
    untarget = np.zeros(4, dtype=np.float32)
    target = np.zeros(4, dtype=np.float32)

    num_query = len(adv_index)
    for i in range(num_query):
        tmp_adv_index = np.copy(adv_index)

        probe_index = tmp_adv_index[i]
        tmp_adv_index[i] = tmp_adv_index[0]
        tmp_adv_index[0] = probe_index

        [result_untarget, result_target] = \
            single_evaluate(net, probe_path, gallery_path, tmp_adv_index, adv_id, target_id, num_target)
        untarget = untarget + result_untarget
        target = target + result_target

    untarget = untarget / num_query
    target = target / num_query

    print('untarget')
    print('Rank 1:\t%f, Rank 5:\t%f, Rank 10:\t%f' % (untarget[0], untarget[1], untarget[2]))
    print('mAP:\t%f' % untarget[3])

    print('target')
    print('Rank 1:\t%f, Rank 5:\t%f, Rank 10:\t%f' % (target[0], target[1], target[2]))
    print('mAP:\t%f' % target[3])

def single_evaluate(net, probe_path, gallery_path, adv_index, adv_id, target_id, num_target):

    #load the bobo dataset. 1)adv images; 2) target images; 3)other pid images
    #and their infos (pid, cid)
    img_names = sorted(os.listdir(probe_path))
    imgs, infos = load(probe_path, img_names)

    # divide the adv images into 2 part. 1) probe  2)gallery_add
    adv_probe_imgs = imgs[adv_index[0:1]]
    adv_probe_infos = [infos[i] for i in adv_index[0:1]]

    # construct the gallery_add imgs and the infos
    num_per_pc = 6
    gal_add_pid = sorted(list(set(range(1502,1513))-{adv_id}))
    otherpid_index = np.random.randint(0, 24, size=3 * (11-1) * num_per_pc) + \
                     np.array([np.arange(3)+(i-1502)*3 for i in gal_add_pid]).flatten().repeat(num_per_pc)*24

    gal_add_index = np.concatenate(
        (adv_index[1:], otherpid_index)
    )

    gallery_imgs_add = imgs[gal_add_index]
    gallery_infos_add = [infos[i] for i in gal_add_index]

    #a. untarget test
    result_untarget = eval_without_noise(net, adv_probe_imgs, adv_probe_infos, gallery_path, gallery_imgs_add, gallery_infos_add,
                                         target_id, num_target)

    #b. target test
    for i in adv_index:
        if infos[i][1] == infos[adv_index[0]][1]:  #means we set the first img of adv_index as the probe
            infos[i] = (target_id, infos[i][1])

    adv_probe_infos = [infos[i] for i in adv_index[0:1]]
    gal_add_index = np.concatenate(
        (adv_index[1:], otherpid_index)
    )
    gallery_infos_add = [infos[i] for i in gal_add_index]
    result_target = eval_without_noise(net, adv_probe_imgs, adv_probe_infos, gallery_path, gallery_imgs_add, gallery_infos_add,
                                       target_id, num_target)

    return [result_untarget, result_target]

def eval_without_noise(net, probe, probe_infos, gallery_path, gallery_imgs_add, gallery_infos_add, target_id, num_target):
    net.load_weights('../baseline_dis/market-pair-pretrain-withoutwarp.h5')

    #calculate the features, infos of gallery,probe
    gallery_add_f = net.predict(gallery_imgs_add, batch_size=128)

    gallery_f = np.load('gallery_f.npy')
    gallery_info = [tuple(x) for x in json.load(open('gallery_info.json', 'r'))]

    target_index, _ = load_market(gallery_path, target_id)
    gallery_index = list(set(list(range(len(gallery_f)))) - set(target_index[num_target:]))
    gallery_f = gallery_f[gallery_index]
    gallery_info = [gallery_info[i] for i in gallery_index]

    gallery_f = np.concatenate((gallery_f, gallery_add_f), axis=0)
    gallery_info.extend([info for info in gallery_infos_add]) #由于info是元组的list

    query_f = net.predict(probe, batch_size=128)
    query_info = probe_infos

    # do the eval for CMC and mAP
    result, result_argsort = sort_similarity(query_f, gallery_f)
    rank1_acc, rank5_acc, rank10_acc, mAP = map_rank_quick_eval(query_info, gallery_info, result_argsort)

    return [rank1_acc, rank5_acc, rank10_acc, mAP]

if __name__ == '__main__':
    with tf.variable_scope('base_model'):
        net = load_model('../baseline_dis/market-pair-pretrain-withoutwarp.h5')

    gallery_path = '../../dataset' + '/Market-1501/bounding_box_test'
    probe_path = '../../dataset' + '/bobo/resize'
    mask_path = '../../dataset' + '/bobo/mask'
    adv_path = '../../dataset' + '/bobo/tar_adv'

    noise_path = '../../dataset' +'/bobo/tar_noise'

    adv_id = 1502
    adv_index = np.array([44,0, 4, 8, 12,  20,18,  28, 30,  32, 40,  46])
    target_id = 591

    #generate adv, saved in adv_path
    #attack(net, probe_path, gallery_path, mask_path, adv_path, noise_path,
    #       adv_index, adv_id, target_id, 6, batch_size=16)

    # test the adv acc
    evaluate(net, adv_path, gallery_path, adv_index, adv_id, target_id, 6)
