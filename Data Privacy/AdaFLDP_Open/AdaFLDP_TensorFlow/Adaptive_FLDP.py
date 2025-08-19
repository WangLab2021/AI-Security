from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from Global_Parameter import *
from EPS_round import *
from EPS_instance import *
import csv
import time
import pickle
import copy
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist, cifar10
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from dlgAttack import DLA
from tools import *

import os
import sys
import warnings
import logging

# 1. 关闭 warnings
# warnings.filterwarnings("ignore")

# # 2. 禁用 logging 输出
# logging.disable(logging.CRITICAL)

# # 3. 重定向标准错误输出
# sys.stderr = open(os.devnull, 'w')

# # 4. 特殊库设置（如 TensorFlow）
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 3 = only fatal
# Set up argument parser
parser = argparse.ArgumentParser(description="FLDP")
parser.add_argument("--ds", type=str, help="", default='mnist')
parser.add_argument("--p", type=str, help="", default='ori')
parser.add_argument("--e", type=int, help="", default=4)
parser.add_argument("--b", type=int, help="", default=128)
parser.add_argument("--eps", type=int, help="", default=150)
parser.add_argument("--c", type=str, help="", default='0')
parser.add_argument("--w", type=int, help="", default=100)
parser.add_argument("--r", type=float, help="", default=0.02)
parser.add_argument("--use_gpu", action="store_true", help="use GPU device if cuda is available", default=False)
args = parser.parse_args()
logger.info("args:", args)

# Global parameters
total_client_num = args.w
selected_rate = args.r
parti_client_num = int(total_client_num * selected_rate)
bs = args.b
E = args.e
eps_global_init = args.eps
DATAMODE = args.ds
PRIVACY_MODE = args.p

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus and args.use_gpu:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        device = f'/GPU:{args.c}'
    except RuntimeError as e:
        logger.info(e)
        device = '/CPU:0'
else:
    device = '/CPU:0'

'''# Device configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        device = '/GPU:0'
    except RuntimeError as e:
        logger.info(e)
        device = '/CPU:0'
else:
    device = '/CPU:0'''
# device = '/CPU:0'
logger.info("Running on %s" % device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def create_lenet_model(input_shape, num_classes):
    """Create LeNet-like model using TensorFlow/Keras"""
    model = models.Sequential(
        [
            layers.Conv2D(12, (5, 5), strides=2, padding='same', activation='sigmoid', input_shape=input_shape),
            layers.Conv2D(12, (5, 5), strides=2, padding='same', activation='sigmoid'),
            layers.Conv2D(12, (5, 5), strides=1, padding='same', activation='sigmoid'),
            layers.Flatten(),
            layers.Dense(num_classes),
        ]
    )

    # Initialize weights similar to PyTorch version
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            layer.kernel.assign(tf.random.uniform(layer.kernel.shape, -0.5, 0.5))
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.assign(tf.random.uniform(layer.bias.shape, -0.5, 0.5))

    return model


def get_model_config():
    """Get model configuration based on dataset"""
    if DATAMODE == 'mnist':
        input_shape = (32, 32, 1)
        num_classes = 10
    elif DATAMODE == 'cifar10':
        input_shape = (32, 32, 3)
        num_classes = 10
    elif DATAMODE == 'lfw':
        input_shape = (32, 32, 3)
        num_classes = 106
    else:
        input_shape = (32, 32, 3)
        num_classes = 10

    return input_shape, num_classes


def preprocess_image(image, target_size=(32, 32)):
    """Preprocess image to target size"""
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def get_data(datamode=DATAMODE):
    """Load and preprocess data"""

    if datamode == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Add channel dimension and resize
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        x_train = tf.image.resize(x_train, [32, 32]).numpy()
        x_test = tf.image.resize(x_test, [32, 32]).numpy()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        index_start = 50000

    elif datamode == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = tf.image.resize(x_train, [32, 32]).numpy()
        x_test = tf.image.resize(x_test, [32, 32]).numpy()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        index_start = 40000

    elif datamode == 'lfw':
        lfw_people = fetch_lfw_people(
            min_faces_per_person=14, color=True, slice_=(slice(61, 189), slice(61, 189)), resize=0.25
        )
        x = lfw_people.images
        y = lfw_people.target

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=False)
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        total_x = X_train
        total_y = y_train
        sorted_x_train = X_train
        sorted_y_train = y_train
        x_vali = X_test
        y_vali = y_test
        x_test = X_test

        logger.info("Dataset:%s, train points:%d, test points:%d" % (DATAMODE, len(X_train), len(X_test)))

        return (
            tf.constant(total_x),
            tf.constant(total_y),
            tf.constant(sorted_x_train),
            tf.constant(sorted_y_train),
            tf.constant(x_vali),
            tf.constant(y_vali),
            tf.constant(x_test),
            tf.constant(y_test),
        )

    if datamode in ['mnist', 'cifar10']:
        logger.info("Dataset:%s, train points:%d, test points:%d" % (datamode, len(x_train), len(x_test)))

        total_x = x_train
        total_y = y_train

        # Create validation set
        x_vali = x_train[index_start:]
        y_vali = y_train[index_start:]

        # Create training set
        x_train = x_train[:index_start]
        y_train = y_train[:index_start]

        # Sort training set for non-IID federated learning
        indices_train = np.argsort(y_train)
        sorted_x_train = x_train[indices_train]
        sorted_y_train = y_train[indices_train]

        # Limit test set to 10000 samples
        x_test = x_test[:10000]
        y_test = y_test[:10000]

    return (
        tf.constant(total_x),
        tf.constant(total_y),
        tf.constant(sorted_x_train),
        tf.constant(sorted_y_train),
        tf.constant(x_vali),
        tf.constant(y_vali),
        tf.constant(x_test),
        tf.constant(y_test),
    )


def getDataExample(x_, y_):
    """Get a data example for attacks"""
    if DATAMODE == 'lfw':
        id = 291
    elif DATAMODE == 'mnist':
        id = 1123
    else:
        id = 48850

    return x_[id : id + 1], y_[id : id + 1]


def compute_gradients(model, x_batch, y_batch, loss_fn):
    """Compute gradients for a batch"""
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients, loss


def apply_gradients(model, gradients, learning_rate):
    """Apply gradients to model"""
    optimizer = optimizers.SGD(learning_rate=learning_rate)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def federated_averaging(client_models):
    """Perform federated averaging"""
    # Get the global model structure
    global_weights = client_models[0].get_weights()

    # Initialize sum with zeros
    sum_weights = [np.zeros_like(w) for w in global_weights]

    # Sum all client weights
    for model in client_models:
        client_weights = model.get_weights()
        for i, w in enumerate(client_weights):
            sum_weights[i] += w

    # Average the weights
    avg_weights = [w / len(client_models) for w in sum_weights]

    return avg_weights


def evaluate_model(model, x_test, y_test, batch_size=100):
    """Evaluate model accuracy"""
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    total = 0
    correct = 0

    for x_batch, y_batch in test_dataset:
        predictions = model(x_batch, training=False)
        predicted_labels = tf.argmax(predictions, axis=1)
        total += len(y_batch)
        # logger.info(predicted_labels,y_batch)
        correct += tf.reduce_sum(tf.cast(predicted_labels == tf.cast(y_batch, tf.int64), tf.int32)).numpy()

    accuracy = correct / total * 100
    return accuracy, correct, total


if __name__ == '__main__':
    from datetime import datetime

    current_time = datetime.now()
    _file_name = Path(__file__).stem
    log_file = Path("./logs", _file_name, current_time.strftime("%Y-%m-%d_%H-%M-%S_%f") + ".log")
    setup_logger(log_file=log_file, level='DEBUG')
    logger.info("total num: {} par num: {}".format(total_client_num, parti_client_num))

    if DATAMODE == 'lfw':
        client_set_file = Path('./DATA/clients_lfw', str(total_client_num) + '_clients.pkl')
        if not client_set_file.exists():
            copy_files(
                srcs=[Path("../app/", client_set_file.absolute().relative_to(Path(__file__).parent.absolute()))],
                dsts=[client_set_file],
            )
        client_set = pickle.load(open(client_set_file, 'rb'))
    elif DATAMODE == 'mnist':
        client_set_file = Path('./DATA/clients', str(total_client_num) + 'clients.npy')
        if not client_set_file.exists():
            copy_files(
                srcs=[Path("../app/", client_set_file.absolute().relative_to(Path(__file__).parent.absolute()))],
                dsts=[client_set_file],
            )
        client_set = np.load(client_set_file)
    else:
        client_set_file = Path('./DATA/clients_cifar10', str(total_client_num) + 'clients.npy')
        if not client_set_file.exists():
            copy_files(
                srcs=[Path("../app/", client_set_file.absolute().relative_to(Path(__file__).parent.absolute()))],
                dsts=[client_set_file],
            )
        client_set = np.load(client_set_file)
    logger.info("load client set ({}) from {}".format(len(client_set), client_set_file))

    # Get data
    total_x, total_y, sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test = get_data()
    logger.info("get data done, total_x shape: {}, total_y shape: {}".format(total_x.shape, total_y.shape))
    logger.info("sorted_x_train shape: {}, sorted_y_train shape: {}".format(sorted_x_train.shape, sorted_y_train.shape))
    logger.info("x_vali shape: {}, y_vali shape: {}".format(x_vali.shape, y_vali.shape))
    logger.info("x_test shape: {}, y_test shape: {}".format(x_test.shape, y_test.shape))
    vali_dataset = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
    vali_dataset = vali_dataset.batch(100)

    # Get model configuration and create models
    input_shape, num_classes = get_model_config()

    with tf.device(device):
        new_global_model = create_lenet_model(input_shape, num_classes)
        old_global_model = create_lenet_model(input_shape, num_classes)
        atk_model = create_lenet_model(input_shape, num_classes)

    # Initialize models with same weights
    new_global_model.set_weights(new_global_model.get_weights())
    old_global_model.set_weights(new_global_model.get_weights())
    atk_model.set_weights(new_global_model.get_weights())

    starting_time = time.time()
    eps_global = eps_global_init
    epsRoundAccount = EPS_round(vali_dataset)  # 传入固定vali集,实例化*注意此处的实例化全局唯一
    client_models = []
    server_save_update = {}
    server_save_deltaU = {}

    # Create test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(4)

    last_S = 10
    t = 0

    # Loss function
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    for round in range(rounds + 1):
        tested = False

        # Server side operations
        logger.info(f"#----------- Round {round+1}/{rounds} - Random client selection completed ----------#")

        # Select participating clients
        np.random.seed(round)
        perm = np.random.permutation(total_client_num)
        s = perm[0:parti_client_num].tolist()
        participating_clients_data = [client_set[k] for k in s]

        E_list = [2, 4, 1, 2, 4]
        client_random = [E_list[s[k] % 5] for k in range(parti_client_num)]
        logger.info("#-----------随机选取用户完毕--------------------#")

        # Step 1: Privacy budget allocation
        if round != 0:
            atk_model.set_weights(old_global_model.get_weights())

        old_global_model.set_weights(new_global_model.get_weights())

        logger.info("round %d test Acc == :" % (round))
        accuracy, correct, total = evaluate_model(old_global_model, x_test, y_test)

        deltaS = accuracy - last_S
        last_S = accuracy
        res_file = Path("./Acc/epsilon", '%s_%s_epsilon%d.csv' % (PRIVACY_MODE, DATAMODE, eps_global_init))
        res_file.parent.mkdir(parents=True, exist_ok=True)
        with open(res_file, mode='a', newline="") as train_file:
            writer_train2 = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer_train2.writerow([round, correct / total])
        logger.info('Accuracy: %d/%d = %.2f%%' % (correct, total, correct / total * 100))

        if t == rounds:
            logger.info("END")
            exit(0)

        # Calculate round privacy budget
        if PRIVACY_MODE == 'adp':
            eps_round = epsRoundAccount.RoundlyAccount(old_global_model, eps_global, t, device, E)
        else:
            logger.info('privacy mode:%s,total eps:%d,rounds:%d' % (PRIVACY_MODE, eps_global_init, rounds))
            eps_round = eps_global_init / rounds

        eps_global -= eps_round
        logger.info("#step1 完毕")

        # Step 2: Attack-based budget allocation
        eps_clients = [eps_round for _ in range(parti_client_num)]
        maxU = float('-inf')

        if PRIVACY_MODE == 'adp':
            dataExample, labelExample = getDataExample(total_x, total_y)

            for key, original_dy_dx in server_save_update.items():
                dy_dx_ = [tf.identity(grad) for grad in original_dy_dx]
                newU = DLA(atk_model, dataExample, labelExample, dy_dx_, device, num_classes, input_shape[-1])
                server_save_deltaU.setdefault(key, []).append(newU)

            finalList = [0.0 for _ in range(parti_client_num)]
            for i in range(parti_client_num):
                if s[i] in server_save_deltaU:
                    if len(server_save_deltaU[s[i]]) > 1:
                        finalList[i] = (
                            Lambda * np.mean(server_save_deltaU[s[i]][0:-1])
                            + (1 - Lambda) * server_save_deltaU[s[i]][-1]
                        )
                        if maxU < finalList[i]:
                            maxU = finalList[i]
                    else:
                        finalList[i] = -1.0
                else:
                    finalList[i] = -1.0

            for i in range(parti_client_num):
                if finalList[i] <= 0:
                    eps_clients[i] = eps_round
                else:
                    eps_clients[i] = (finalList[i] / maxU) * eps_round
            logger.info("final list: {} max U: {}".format(finalList, maxU))
        else:
            eps_clients = [eps_round for n in range(parti_client_num)]

        server_save_update.clear()
        logger.info("#step2 完毕")
        logger.info("Server端 waiting...")

        # Client side operations
        # Client端
        logger.info("#Client端 start")
        client_models = []
        norm_client_gradient = []

        for k_t in range(parti_client_num):
            # Create local model copy
            local_model = create_lenet_model(input_shape, num_classes)
            local_model.set_weights(old_global_model.get_weights())

            eps = eps_clients[k_t]

            # Initialize EPS instance (assuming adapted for TensorFlow)
            k_instance = EPS_instance(participating_clients_data[k_t], local_model, E, bs, eps, device)
            # ----------------------------test of Vulearable client----------------------------#
            tested = True
            # ---------------------------------------------------------------------------------#
            # step3
            logger.info("Client:%d is running in %d round,mode = %s" % (k_t + 1, round, PRIVACY_MODE))

            if PRIVACY_MODE == 'adp':
                num_weights = k_instance.runOurs(
                    sorted_x_train, sorted_y_train, loss_fn, DecayClip, False, device, x_vali, y_vali
                )
            else:
                # Handle other privacy modes
                pass

            client_models.append(local_model)

            # Compute final gradients
            local_weights = local_model.get_weights()
            global_weights = old_global_model.get_weights()
            final_gradient = [
                (global_w - local_w) / learning_rate for local_w, global_w in zip(local_weights, global_weights)
            ]

            server_save_update[s[k_t]] = final_gradient
        # Client端 END
        #  Server端  聚合----------------------------------------------------------------------#
        worker_state_dict = [m.get_weights() for m in client_models]
        sum_parameters = {}
        for model in client_models:
            model_weights = model.get_weights()
            for i, weight in enumerate(model_weights):
                if i not in sum_parameters:
                    sum_parameters[i] = np.zeros_like(weight)
                sum_parameters[i] += weight

        averaged_weights = [sum_parameters[i] / len(client_models) for i in sorted(sum_parameters.keys())]
        new_global_model.set_weights(averaged_weights)

        logger.info("global model after Avg:")
        t = round + 1
