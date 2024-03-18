from __future__ import division, print_function, absolute_import

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import numpy as np
import time
import glob
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Input
from keras.preprocessing import image
from keras.models import load_model

from utils.file_helper import write

def extract_info(dir_path):
    infos = []
    for image_name in sorted(os.listdir(dir_path)):
        if '.txt' in image_name or '.db' in image_name:
            continue
        if 's' in image_name or 'f' in image_name:
            # market && duke
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1][1])
        elif 's' not in image_name:
            # grid
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1])
        else:
            continue
        infos.append((person, camera))

    return infos


def extract_feature(dir_path, net):
    image_data = []
    infos = []

    for image_name in sorted(os.listdir(dir_path)):
        if '.txt' in image_name or '.db' in image_name:
            continue
        if 'f' in image_name or 's' in image_name:
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1][1])
        elif 's' not in image_name:
            # grid
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1])
        else:
            continue
        infos.append((person, camera))
        image_path = os.path.join(dir_path, image_name)
        x = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(x)
        image_data.append(x)
    image_data = np.array(image_data)
    image_data = preprocess_input(image_data)
    features = net.predict(image_data, batch_size=128)
    return features, infos


def similarity_matrix(query_f, test_f):
    # Tensorflow graph
    # use GPU to calculate the similarity matrix
    query_t = tf.placeholder(tf.float32, (None, None))
    test_t = tf.placeholder(tf.float32, (None, None))
    query_t_norm = tf.nn.l2_normalize(query_t, dim=1)
    test_t_norm = tf.nn.l2_normalize(test_t, dim=1)
    tensor = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    if query_f.ndim == 4:
        query_f = np.squeeze(query_f, axis=(1,2))
    if test_f.ndim == 4:
        test_f = np.squeeze(test_f, axis=(1,2))

    result = sess.run(tensor, {query_t: query_f, test_t: test_f})
    print(result.shape)
    # descend
    return result


def sort_similarity(query_f, test_f):
    result = similarity_matrix(query_f, test_f)
    result_argsort = np.argsort(-result, axis=1)
    return result, result_argsort


def map_rank_quick_eval(query_info, test_info, result_argsort):
    # much more faster than hehefan's evaluation
    match = []
    junk = []
    QUERY_NUM = len(query_info)

    for q_index, (qp, qc) in enumerate(query_info):
        tmp_match = []
        tmp_junk = []
        for t_index in range(len(result_argsort[q_index])):
            p_t_idx = result_argsort[q_index][t_index]
            p_info = test_info[int(p_t_idx)]

            tp = p_info[0]
            tc = p_info[1]
            if tp == qp and qc != tc:
                tmp_match.append(t_index)
            elif tp == qp or tp == -1:
                tmp_junk.append(t_index)
        match.append(tmp_match)
        junk.append(tmp_junk)

    rank_1 = 0.0
    rank_5 = 0.0
    rank_10 = 0.0
    mAP = 0.0
    for idx in range(len(query_info)):
        if idx % 100 == 0:
            print('evaluate img %d' % idx)
        recall = 0.0
        precision = 1.0
        ap = 0.0
        YES = match[idx]
        IGNORE = junk[idx]
        ig_cnt = 0

        #rank-k acc 计算
        ig_cnt = len([ x for x in IGNORE if x<YES[0] ])
        rank_1  += (1 if ig_cnt >= (YES[0]-0) else 0)
        rank_5  += (1 if ig_cnt >= (YES[0]-4) else 0)
        rank_10 += (1 if ig_cnt >= (YES[0]-9) else 0)

        #mAP 计算
        for i, k in enumerate(YES):
            ig_cnt = len([x for x in IGNORE if x < k])

            cnt = k + 1 - ig_cnt
            hit = i + 1
            tmp_recall = hit / len(YES)
            tmp_precision = hit / cnt
            ap = ap + (tmp_recall - recall) * ((precision + tmp_precision) / 2)
            recall = tmp_recall
            precision = tmp_precision

        mAP += ap
    rank1_acc = rank_1 / QUERY_NUM
    rank5_acc = rank_5 / QUERY_NUM
    rank10_acc = rank_10 / QUERY_NUM
    mAP = mAP / QUERY_NUM
    print('Rank 1:\t%f, Rank 5:\t%f, Rank 10:\t%f' % (rank1_acc, rank5_acc, rank10_acc))
    print('mAP:\t%f' % mAP)
    return rank1_acc, rank5_acc, rank10_acc, mAP


def train_predict(net, train_path, pid_path, score_path):
    net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output]) #get_layer('flatten')
    train_f, test_info = extract_feature(train_path, net)
    result, result_argsort = sort_similarity(train_f, train_f)
    for i in range(len(result)):
        result[i] = result[i][result_argsort[i]]
    result = np.array(result)
    # ignore top1 because it's the origin image
    np.savetxt(score_path, result[:, 1:], fmt='%.4f')
    np.savetxt(pid_path, result_argsort[:, 1:], fmt='%d')
    return result

def test_predict(net, probe_path, gallery_path):
    net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output]) #get_layer('flatten')
    test_f, test_info = extract_feature(gallery_path, net)
    query_f, query_info = extract_feature(probe_path, net)
    result, result_argsort = sort_similarity(query_f, test_f)

    #for compute the mean similarity score
    match = []
    for q_index, (qp, qc) in enumerate(query_info):
        tmp_match = []
        for t_index in range(len(result[q_index])):
            p_info = test_info[int(t_index)]

            tp = p_info[0]
            tc = p_info[1]
            if tp == qp and qc != tc:
                tmp_match.append(t_index)
        match.append(tmp_match)

    sum = 0.0
    num = 0
    for i in range(len(query_info)):
        length = len(match[i])
        score = result[i][match[i]]

        num += length
        sum += np.sum(score)
    print(sum, num, sum/num)

    for i in range(len(result)):
        result[i] = result[i][result_argsort[i]]
    return result_argsort, test_info, query_info

def extract_feature_MQ(dir_path, gt_path, net):
    image_data = []
    infos = []
    length=[]

    for image_name in sorted(os.listdir(dir_path)):
        if '.txt' in image_name or '.db' in image_name:
            continue
        if 'f' in image_name or 's' in image_name:
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1][1])
        elif 's' not in image_name:
            # grid
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1])
        else:
            continue
        infos.append((person, camera))

        tmp_queries_file = glob.glob(gt_path + '/' + image_name[:7] + '*')
        tmp_len = len(tmp_queries_file)
        length.append(tmp_len)

        for path in tmp_queries_file:
            x = image.load_img(path, target_size=(224, 224))
            x = image.img_to_array(x)
            image_data.append(x)

    image_data = np.array(image_data)
    image_data = preprocess_input(image_data)
    tmp_features = net.predict(image_data, batch_size=128)

    #将每个用户在每个摄像头下的多张照片，融合成一个query feature
    index = 0
    features=[]
    for tmp_len in length:
        x = np.mean(tmp_features[index:index+tmp_len],axis=0)
        #x = np.max(tmp_features[index:index+tmp_len],axis=0)
        features.append(x)
        index += tmp_len
    features = np.array(features)
    return features, infos

#multiple queries
def test_predict_MQ(net, probe_path, gt_path, gallery_path):
    net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output])
    query_f, query_info = extract_feature_MQ(probe_path, gt_path, net)
    test_f, test_info = extract_feature(gallery_path, net)
    result, result_argsort = sort_similarity(query_f, test_f)
    for i in range(len(result)):
        result[i] = result[i][result_argsort[i]]
    return result_argsort, test_info, query_info

def market_result_eval(result_argsort, test_info, query_info, log_path='market_result_eval.log'):
    res = result_argsort
    print('start evaluate map and rank acc')
    rank1_acc, rank5_acc, rank10_acc, mAP = map_rank_quick_eval(query_info, test_info, res)
    write(log_path, '%f\t%f\t%f\t%f\n' % (rank1_acc, rank5_acc, rank10_acc, mAP))


#func evaluate can be used outside
def evaluate(net, flag, probe_path, gallery_path, gt_path=None): # flag: 1 for (single query); 0 for (multiple queries)
    if flag == 1:
        result_argsort, test_info, query_info = test_predict(net, probe_path,
                                                             gallery_path)  # recording the predict index info
        market_result_eval(result_argsort, test_info, query_info, log_path='testset_eval.log')

    else:
        result_argsort, test_info, query_info = test_predict_MQ(net, probe_path, gt_path, gallery_path)
        market_result_eval(result_argsort, test_info, query_info, log_path='testset_eval.log')


if __name__ == '__main__':
    #net = load_model('../baseline_dis/market-pair-pretrain-withoutwarp.h5')

    base_model = load_model('../baseline_dis/market_softmax_pretrain.h5')
    net = Model(inputs=base_model.input, outputs=[base_model.get_layer('avg_pool').output],
                name='resnet50')

    gallery_path = '../../dataset' + '/Market-1501/bounding_box_test'
    probe_path = '../../dataset' + '/Market-1501/query'
    evaluate(net, 1, probe_path, gallery_path)

    #gt_path = '../../dataset' + '/Market-1501/gt_bbox'
    #evaluate(net, 0, probe_path, gallery_path, gt_path)