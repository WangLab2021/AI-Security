
import torch
import torch.nn as nn
import os
import logging
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.externals import joblib
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("logger")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

root = 'saved_attacker/result/poi'
folder_path = f'{root}/accuracy_{current_time}'
try:
    os.mkdir(folder_path)
except FileExistsError:
    logger.info('Folder already exists')
logger.addHandler(logging.FileHandler(filename=f'{folder_path}/log.txt'))
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
logger.info(f'current path: {folder_path}')


def read_testdata(target_EPOCHES, prop_epoch, exp_i):
    model_update = []
    model_update_label = []
    for epoch in range(1, target_EPOCHES + 1):
        epoch_update = torch.load(
            'E:/lfw_data/race/exp{}/test/Gupdate_{}_epoch.pkl'.format(exp_i, epoch))
        if epoch in prop_epoch:
            epoch_label = 1
        else:
            epoch_label = 0

        if epoch == 1:
            model_update = epoch_update
        else:
            model_update = torch.cat((model_update, epoch_update))
        model_update_label.append(epoch_label)
    # logger.info('loaded test data from exp{}'.format(exp_i))
    model_update_label = torch.cuda.FloatTensor(model_update_label)
    return model_update, model_update_label


def read_traindata(EPOCHES, update_type, exp_i, model_id):
    update_label = []
    for epoch in EPOCHES:

        epoch_update = torch.load(
            'E:/lfw_data/race/exp{}/train/{}_{}_epoch.pkl'.format(exp_i, update_type, epoch))
        if update_type == f'S{model_id}_nonp':
            epoch_label = 0
        elif update_type == f'S{model_id}_p':
            epoch_label = 1
        else:
            print('wrong update_type!')
        update_label.append(epoch_label)

        if epoch == EPOCHES[0]:
            update_tensorset = epoch_update
        else:
            update_tensorset = torch.cat((update_tensorset, epoch_update))
    # logger.info('loaded train data from exp{}'.format(exp_i))
    # logger.info('dataset length:{}'.format(len(update_tensorset)))
    label_tensorset = torch.cuda.FloatTensor(update_label)
    return update_tensorset, label_tensorset


if __name__ == '__main__':

    adv_EPOCH = 1000
    adv_LR = 0.01
    adv_BATCHSIZE = 20
    target_epoches = 100
    no_shadow_models = 10

    exp_times = 10
    start = 21
    shadow_p_epoch = []
    shadow_np_epoch = []
    prop_epoch = [3, 5, 11, 16, 30, 50, 60, 70, 80, 90]
    # prop_epoch=[ 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    for epoch in range(1, target_epoches + 1):
        if epoch % 2 == 0:
            shadow_p_epoch.append(epoch)
        else:
            shadow_np_epoch.append(epoch)

    ACC_lr = []
    ACC_rt = []
    ACC_gb = []
    auc_lr = []
    auc_rt = []
    auc_gb = []
    prop_score_lr = []
    prop_score_rt = []
    prop_score_gb = []
    pred_whole1 = []
    pred_whole2 = []
    pred_whole3 = []
    best_threshold1 = []
    best_threshold2 = []
    best_threshold3 = []
    for exp_i in range(start, exp_times + start):
        for model_id in range(no_shadow_models):
            p_update, p_update_label = read_traindata(EPOCHES=shadow_p_epoch,
                                                      update_type=f'S{model_id}_p', exp_i=exp_i, model_id=model_id)
            nonp_update, nonp_update_label = read_traindata(EPOCHES=shadow_np_epoch,
                                                            update_type=f'S{model_id}_nonp', exp_i=exp_i,
                                                            model_id=model_id)

            if model_id == 0:
                train_data = torch.cat((p_update, nonp_update))
                train_target = torch.cat((p_update_label, nonp_update_label))
            else:
                train_data = torch.cat((train_data, p_update, nonp_update))
                train_target = torch.cat((train_target, p_update_label, nonp_update_label))

        model_update, model_update_label = read_testdata(target_EPOCHES=target_epoches,
                                                         prop_epoch=prop_epoch, exp_i=exp_i)
        test_data = model_update
        test_target = model_update_label

        train_data = train_data.cpu()
        train_target = train_target.cpu()
        index = [i for i in range(len(train_data))]
        random.shuffle(index)
        train_data = train_data[index]
        train_target = train_target[index]
        test_data = test_data.cpu()
        test_target = test_target.cpu()

        scaler = preprocessing.StandardScaler().fit(train_data)
        scaler.transform(train_data)
        scaler.transform(test_data)
        logger.info(f'exp{exp_i}: begin training and infering.......')

        attack_model_1 = LogisticRegression()
        # attack_model_1 = joblib.load("saved_attacker/attacker_lr.m")
        # train_sizes, train_score, test_score = learning_curve(attack_model_1, train_data.numpy(), train_target.numpy(),
        #                                                       train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1], cv=10,
        #                                                       scoring='accuracy')
        # train_error = 1 - np.mean(train_score, axis=1)
        # test_error = 1 - np.mean(test_score, axis=1)
        # plt.plot(train_sizes, train_error, 'o-', color='r', label='training')
        # plt.plot(train_sizes, test_error, 'o-', color='g', label='testing')
        # plt.legend(loc='best')
        # plt.xlabel('traing examples')
        # plt.ylabel('error')
        # plt.show()

        attack_model_1.fit(train_data, train_target)
        joblib.dump(attack_model_1, '{}/attacker_lr.m'.format(folder_path))
        pred_prop1 = attack_model_1.predict_proba(test_data)[:, 1]
        pred_whole1.append(pred_prop1)
        fpr1, tpr1, thresholds1 = roc_curve(test_target, pred_prop1, pos_label=1)
        y1 = tpr1 + (1 - fpr1) - 1
        ind1 = np.argmax(y1)
        best_threshold1.append(thresholds1[ind1])
        # logger.info('lr  fpr: {}'.format(fpr1))
        # logger.info('lr  tpr: {}'.format(tpr1))
        # logger.info('lr thresholds: {}'.format(thresholds1))
        auc1 = roc_auc_score(test_target, pred_prop1)
        auc_lr.append(auc1)
        logger.info('lr auc: {}'.format(auc1))

        predict = attack_model_1.predict(test_data)
        acc_lr = attack_model_1.score(test_data, test_target)
        ACC_lr.append(acc_lr)
        logger.info('LR acc:{}'.format(acc_lr))
        logger.info(classification_report(predict, test_target))

        attack_model_2 = RandomForestClassifier(n_estimators=50)
        # attack_model_2 = joblib.load("saved_attacker/attacker_rt.m")
        attack_model_2.fit(train_data, train_target)
        joblib.dump(attack_model_2, '{}/attacker_rt.m'.format(folder_path))
        pred_prop2 = attack_model_2.predict_proba(test_data)[:, 1]
        pred_whole2.append(pred_prop2)
        fpr2, tpr2, thresholds2 = roc_curve(test_target, pred_prop2, pos_label=1)
        roc_file = f'{folder_path}\\roc_p{exp_i+start}.csv'

        with open(roc_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(fpr2)
            writer.writerow(tpr2)

        y2 = tpr2 + (1 - fpr2) - 1
        ind2 = np.argmax(y2)
        best_threshold2.append(thresholds2[ind2])
        # logger.info('random forest  fpr: {}'.format(fpr2))
        # logger.info('random forest  tpr: {}'.format(tpr2))
        # logger.info('random forest thresholds: {}'.format(thresholds2))
        auc2 = roc_auc_score(test_target, pred_prop2)
        auc_rt.append(auc2)
        logger.info('random forest auc: {}'.format(auc2))
        predict_2 = attack_model_2.predict(test_data)
        acc_rt = attack_model_2.score(test_data, test_target)
        ACC_rt.append(acc_rt)
        logger.info('Random Forest acc:{}'.format(acc_rt))
        logger.info(classification_report(predict_2, test_target))

        attack_model_3 = GradientBoostingClassifier()
        # attack_model_3 = joblib.load("saved_attacker/attacker_gb.m")
        attack_model_3.fit(train_data, train_target)
        joblib.dump(attack_model_3, '{}/attacker_gb.m'.format(folder_path))
        pred_prop3 = attack_model_3.predict_proba(test_data)[:, 1]
        pred_whole3.append(pred_prop3)
        fpr3, tpr3, thresholds3 = roc_curve(test_target, pred_prop3, pos_label=1)
        y3 = tpr3 + (1 - fpr3) - 1
        ind3 = np.argmax(y3)
        best_threshold3.append(thresholds3[ind3])
        # logger.info('gradient boosting  fpr: {}'.format(fpr3))
        # logger.info('gradient boosting  tpr: {}'.format(tpr3))
        # logger.info('gradient boosting  thresholds: {}'.format(thresholds3))
        auc3 = roc_auc_score(test_target, pred_prop3)
        auc_gb.append(auc3)
        logger.info('gb auc: {}'.format(auc3))
        predict_3 = attack_model_3.predict(test_data)
        acc_gb = attack_model_3.score(test_data, test_target)
        ACC_gb.append(acc_gb)
        logger.info('Gradient Boosting acc:{}'.format(acc_gb))
        logger.info(classification_report(predict_3, test_target))

        pred_file = f'{folder_path}\\pred_{exp_i-start}.csv'

        with open(pred_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(np.around(pred_whole1[int(exp_i - start)], 2))
            writer.writerow(np.around(pred_whole2[int(exp_i - start)], 2))
            writer.writerow(np.around(pred_whole3[int(exp_i - start)], 2))

        prop_score_lr.append(np.max(pred_prop1))
        prop_score_rt.append(np.max(pred_prop2))
        prop_score_gb.append(np.max(pred_prop3))

    ave_acc_lr = np.mean(ACC_lr)
    ave_acc_rt = np.mean(ACC_rt)
    ave_acc_gb = np.mean(ACC_gb)

    auc_file = f'{folder_path}\\auc_{start}-{start+exp_times-1}.csv'
    eval_file = f'{folder_path}\\eval_score_{start}-{start+exp_times-1}.csv'
    with open(auc_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(np.around(auc_lr, 2))
        writer.writerow(np.around(auc_rt, 2))
        writer.writerow(np.around(auc_gb, 2))

    logger.info('===========Average Accuracy=========')
    logger.info('lr model acc: {}, auc:{}'.format(np.around(ACC_lr, 2), np.around(auc_lr, 2)))
    logger.info('random forest: {}, auc:{}'.format(np.around(ACC_rt, 2), np.around(auc_rt, 2)))
    logger.info('gradient boosting: {}, auc:{}'.format(np.around(ACC_gb, 2), np.around(auc_gb, 2)))

    logger.info('============property score=============')
    logger.info('lr model: {}'.format(np.around(prop_score_lr, 2)))
    logger.info('random forest: {}'.format(np.around(prop_score_rt, 2)))
    logger.info('gradient boosting: {}'.format(np.around(prop_score_gb, 2)))
    logger.info('=================eval score==================')
    logger.info('lr model: {}'.format(np.around(np.multiply(prop_score_lr, auc_lr), 2)))
    logger.info('rt model: {}'.format(np.around(np.multiply(prop_score_rt, auc_rt), 2)))
    logger.info('gb model: {}'.format(np.around(np.multiply(prop_score_gb, auc_gb), 2)))

    with open(eval_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(np.around(np.multiply(prop_score_lr, auc_lr), 2))
        writer.writerow(np.around(np.multiply(prop_score_rt, auc_rt), 2))
        writer.writerow(np.around(np.multiply(prop_score_gb, auc_gb), 2))

    threshold_file = f'{folder_path}\\threshold_{start}-{start+exp_times-1}.csv'
    with open(threshold_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(np.around(best_threshold1, 2))
        writer.writerow(np.around(best_threshold2, 2))
        writer.writerow(np.around(best_threshold3, 2))