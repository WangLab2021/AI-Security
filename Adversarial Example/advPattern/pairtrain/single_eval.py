import os

import tensorflow as tf

from keras import backend as K
from keras.engine import Model
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

import numpy as np
from utils.file_helper import write

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

    result = sess.run(tensor, {query_t: query_f, test_t: test_f})
    print(result.shape)
    # descend
    return result

def sort_similarity(query_f, test_f):
    result = similarity_matrix(query_f, test_f)
    result_argsort = np.argsort(-result, axis=1)
    return result, result_argsort

def extract_feature(dir_path, net):
    features = []
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
        image_path = os.path.join(dir_path, image_name)
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = net.predict(x)
        features.append(np.squeeze(feature))
        infos.append((person, camera))

    return features, infos

def test_predict(net, probe_path, gallery_path, pid_path, score_path):
    net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output])
    test_f, test_info = extract_feature(gallery_path, net)
    query_f, query_info = extract_feature(probe_path, net)
    result, result_argsort = sort_similarity(query_f, test_f)
    for i in range(len(result)):
        result[i] = result[i][result_argsort[i]]
    result = np.array(result)
    #np.savetxt(pid_path, result_argsort, fmt='%d')
    #np.savetxt(score_path, result, fmt='%.4f')
    return result_argsort, test_info, query_info

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
    mAP = 0.0
    rank1_list = list()
    for idx in range(len(query_info)):
        if idx % 100 == 0:
            print('evaluate img %d' % idx)
        recall = 0.0
        precision = 1.0
        ap = 0.0
        YES = match[idx]
        IGNORE = junk[idx]
        ig_cnt = 0
        for ig in IGNORE:
            if ig < YES[0]:
                ig_cnt += 1
            else:
                break
        if ig_cnt >= YES[0]:
            rank_1 += 1
            rank1_list.append(1)
        else:
            rank1_list.append(0)

        for i, k in enumerate(YES):
            ig_cnt = 0
            for ig in IGNORE:
                if ig < k:
                    ig_cnt += 1
                else:
                    break
            cnt = k + 1 - ig_cnt
            hit = i + 1
            tmp_recall = hit / len(YES)
            tmp_precision = hit / cnt
            ap = ap + (tmp_recall - recall) * ((precision + tmp_precision) / 2)
            recall = tmp_recall
            precision = tmp_precision

        mAP += ap
    rank1_acc = rank_1 / QUERY_NUM
    mAP = mAP / QUERY_NUM
    print('Rank 1:\t%f' % rank1_acc)
    print('mAP:\t%f' % mAP)
    np.savetxt('rank_1.log', np.array(rank1_list), fmt='%d')
    return rank1_acc, mAP

def market_result_eval(result_argsort, test_info, query_info, log_path='market_result_eval.log'):

    #res = np.genfromtxt(predict_path, delimiter=' ')
    res = result_argsort
    print('start evaluate map and rank acc')
    rank1, mAP = map_rank_quick_eval(query_info, test_info, res)
    write(log_path, '%f\t%f\n' % (rank1, mAP))

def test_pair_predict(pair_model_path, target_probe_path, target_gallery_path, pid_path, score_path):
    # todo
    model = load_model(pair_model_path)
    # model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    # model = Model(inputs=[model.input], outputs=[model.get_layer('avg_pool').output])
    result_argsort, test_info, query_info = test_predict(model, target_probe_path, target_gallery_path, pid_path, score_path)
    return result_argsort, test_info, query_info

def single_directory_eval(source, probe_path, gallery_path):
    return test_pair_predict(source + '_pair_pretrain.h5',
                                       probe_path, gallery_path,
                                       'single_directory_eval_market_market_pid.log', 'single_directory_eval_market_market_score.log')

if __name__ == '__main__':
    # among the given /probe directory and /gallery
    probe_path = '../../dataset/whu_prob'
    gallery_path = '../../dataset/whu+gallery'
    #gallery_path = '../../dataset/Market-1501/bounding_box_test'


    result_argsort, test_info, query_info = single_directory_eval('market', probe_path, gallery_path)
    market_result_eval(result_argsort,test_info, query_info, log_path='single_market_result_eval.log')