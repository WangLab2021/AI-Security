from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import time
import pickle
import copy
import os
import shutil
from pathlib import Path

import numpy as np
import paddle
import matplotlib.pyplot as plt
import paddle.nn as nn
from paddle.vision import datasets, transforms
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# Import the previously translated PaddlePaddle modules
from Global_Parameter import *
from EPS_round import EPS_round  # Assuming this file is translated to Paddle
from EPS_instance import EPS_instance  # Assuming this file is translated to Paddle
from dlgAttack import DLA  # Assuming this file is translated to Paddle
from tools import *

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="FLDP-Paddle")
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

total_client_num = args.w
selected_rate = args.r
parti_client_num = int(total_client_num * selected_rate)
bs = args.b
E = args.e
eps_global_init = args.eps
DATAMODE = args.ds
PRIVACY_MODE = args.p

# --- Device Setup (PaddlePaddle Idiomatic Way) ---
device = f"gpu:{args.c}" if args.use_gpu and paddle.is_compiled_with_cuda() else "cpu"
paddle.set_device(device)
logger.info(f"Running on {device}")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# --- Weight Initialization (PaddlePaddle Way) ---
def weights_init(m):
    if hasattr(m, "weight"):
        initializer = paddle.nn.initializer.Uniform(low=-0.5, high=0.5)
        initializer(m.weight)
    if hasattr(m, "bias") and m.bias is not None:
        initializer = paddle.nn.initializer.Uniform(low=-0.5, high=0.5)
        initializer(m.bias)


# --- Model Definition (PaddlePaddle Layer) ---
class LeNet(nn.Layer):
    def __init__(self, channel=3, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2D(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2D(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2D(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(nn.Linear(768, num_classes))

    def forward(self, x):
        out = self.body(x)
        # PaddlePaddle's reshape is equivalent to PyTorch's view
        out = paddle.reshape(out, shape=[out.shape[0], -1])
        out = self.fc(out)
        return out


# --- Model Initialization ---
channel = 1 if DATAMODE == 'mnist' else 3
num_classes = 106 if DATAMODE == 'lfw' else 10

# The model is automatically placed on the device set by paddle.set_device()
model = LeNet(channel=channel, num_classes=num_classes)
model.apply(weights_init)



# --- Data Loading (PaddlePaddle APIs) ---
def get_data(datamode=DATAMODE):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.CenterCrop(32), transforms.ToTensor()])

    dataset_path = Path('./data')
    dataset_path.mkdir(parents=True, exist_ok=True)
    default_dataset_path = Path('~/.cache/paddle/dataset').expanduser()

    if datamode == 'mnist':
        dataset_filepaths = [
            dataset_path / f'{datamode.upper()}_{d}_{l}.gz' for d in ['train', 'test'] for l in ['images', 'labels']
        ]
        default_dataset_filepaths = [
            default_dataset_path / datamode / datasets.MNIST.TRAIN_IMAGE_URL.split('/')[-1],
            default_dataset_path / datamode / datasets.MNIST.TRAIN_LABEL_URL.split('/')[-1],
            default_dataset_path / datamode / datasets.MNIST.TEST_IMAGE_URL.split('/')[-1],
            default_dataset_path / datamode / datasets.MNIST.TEST_LABEL_URL.split('/')[-1],
        ]
        if not all(p.exists() for p in dataset_filepaths):
            datasets.MNIST(mode='train', download=True)
            datasets.MNIST(mode='test', download=True)
            # 将默认数据集路径复制到指定路径
            move_files(default_dataset_filepaths, dataset_filepaths)
        dataset_filepaths_str = [p.as_posix() for p in dataset_filepaths]
        train_dataset = datasets.MNIST(*dataset_filepaths_str[0:2], mode='train', transform=transform)
        test_dataset = datasets.MNIST(*dataset_filepaths_str[2:4], mode='test', transform=transform)
    elif datamode == 'cifar10':
        data_file = dataset_path / f'{datamode.upper()}.tar.gz'
        default_dataset_path = default_dataset_path / 'cifar' / datasets.cifar.CIFAR10_URL.split('/')[-1]
        if not data_file.exists():
            datasets.Cifar10(mode='train', download=True)
            datasets.Cifar10(mode='test', download=True)
            # 将默认数据集路径复制到指定路径
            move_files([default_dataset_path], [data_file])
        train_dataset = datasets.Cifar10(data_file.as_posix(), mode='train', transform=transform, download=True)
        test_dataset = datasets.Cifar10(data_file.as_posix(), mode='test', transform=transform, download=True)
    elif datamode == 'lfw':
        lfw_people = fetch_lfw_people(
            data_home="./data",
            min_faces_per_person=14,
            color=True,
            slice_=(slice(61, 189), slice(61, 189)),
            resize=0.25,
        )
        x, y = lfw_people.images, lfw_people.target
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=False)

        # Preprocess and convert to Paddle Tensors
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3) / 255.0
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3) / 255.0

        sorted_x_train = paddle.to_tensor(X_train, dtype='float32')
        # Efficiently transpose from (N, H, W, C) to (N, C, H, W)
        sorted_x_train = sorted_x_train.transpose(perm=[0, 3, 1, 2])
        sorted_y_train = y_train

        total_x = sorted_x_train[0:]
        total_y = paddle.to_tensor(y_train, dtype='int64')

        x_test = paddle.to_tensor(X_test, dtype='float32').transpose(perm=[0, 3, 1, 2])
        y_test_tensor = paddle.to_tensor(y_test, dtype='int64')

        x_vali = x_test
        y_vali = y_test_tensor
        test_dataset = paddle.io.TensorDataset([x_test, y_test_tensor])
        logger.info(f"DataSet:{DATAMODE}, get train points:{len(sorted_x_train)}, get test points:{len(x_test)}")

    if datamode in ['mnist', 'cifar10']:
        index_start = 50000 if datamode == 'mnist' else 40000
        logger.info(f"DataSet:{datamode}, get train points:{len(train_dataset)}, get test points:{len(test_dataset)}")

        x_train, y_train = zip(*train_dataset)
        x_train = paddle.stack(x_train, axis=0)
        if datamode == 'cifar10':
            y_train = paddle.to_tensor(y_train, dtype='int64')
        else:
            y_train = np.array([lbl[0] for lbl in y_train])

        total_x = x_train
        total_y = paddle.to_tensor(y_train, dtype='int64')

        x_vali = x_train[index_start:]
        y_vali = paddle.to_tensor(y_train[index_start:], dtype='int64')

        x_train = x_train[:index_start]
        y_train = y_train[:index_start]

        indices_train = np.argsort(y_train)
        sorted_x_train = x_train[indices_train]
        sorted_y_train = paddle.to_tensor(y_train[indices_train], dtype='int64')

        x_test_list, y_test_list = zip(*test_dataset)
        x_test = paddle.stack(x_test_list, axis=0)

        if datamode == 'cifar10':
            y_test = paddle.to_tensor(y_test_list, dtype='int64')
        else:
            y_test = paddle.to_tensor(np.array([lbl[0] for lbl in y_test_list]), dtype='int64')

    # Tensors are already on the correct device due to paddle.set_device()
    return total_x, total_y, sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test, test_dataset


def getDataExample(x_, y_):
    id = 291 if DATAMODE == 'lfw' else (1123 if DATAMODE == 'mnist' else 48850)
    # Use paddle.reshape
    y = paddle.reshape(y_[id], shape=[1])
    return x_[id], y


# --- Main Execution Block ---
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
                dsts=[client_set_file]
            )
        client_set = pickle.load(open(client_set_file, 'rb'))
    elif DATAMODE == 'mnist':
        client_set_file = Path('./DATA/clients', str(total_client_num) + 'clients.npy')
        if not client_set_file.exists():
            copy_files(
                srcs=[Path("../app/", client_set_file.absolute().relative_to(Path(__file__).parent.absolute()))],
                dsts=[client_set_file]
            )
        client_set = np.load(client_set_file)
    else:
        client_set_file = Path('./DATA/clients_cifar10', str(total_client_num) + 'clients.npy')
        if not client_set_file.exists():
            copy_files(
                srcs=[Path("../app/", client_set_file.absolute().relative_to(Path(__file__).parent.absolute()))],
                dsts=[client_set_file]
            )
        client_set = np.load(client_set_file)
    logger.info("load client set ({}) from {}".format(len(client_set), client_set_file))

    total_x, total_y, sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test, test_dataset = get_data()
    logger.info("get data done, total_x shape: {}, total_y shape: {}".format(total_x.shape, total_y.shape))
    logger.info("sorted_x_train shape: {}, sorted_y_train shape: {}".format(sorted_x_train.shape, sorted_y_train.shape))
    logger.info("x_vali shape: {}, y_vali shape: {}".format(x_vali.shape, y_vali.shape))
    logger.info("x_test shape: {}, y_test shape: {}".format(x_test.shape, y_test.shape))
    
    vali_set = paddle.io.TensorDataset([x_vali, y_vali])
    valiloader = paddle.io.DataLoader(vali_set, batch_size=100, shuffle=False)
    test_set_loader = paddle.io.DataLoader(test_dataset, batch_size=4, shuffle=True)

    new_global_model = copy.deepcopy(model)
    old_global_model = copy.deepcopy(model)
    atk_model = copy.deepcopy(model)

    starting_time = time.time()
    eps_global = float(eps_global_init)
    epsRoundAccount = EPS_round(valiloader)

    server_save_update = {}
    server_save_deltaU = {}
    last_S = 10.0
    t = 0

    for round in range(rounds + 1):
        tested = False
        #  Server端 start
        # ---------------------------------------------------------------------------------#
        #  数据划分方式：shuffle，每个用户固定拥有500个数据，每轮选取partiClientNum个用户进行计算#
        np.random.seed(round)
        perm = np.random.permutation(total_client_num)  # 对所有的client进行一个shuffle
        s = perm[0:parti_client_num].tolist()  # S为本轮选出来的clients
        participating_clients_data = [client_set[k] for k in s]
        E_list = [2, 4, 1, 2, 4]
        client_random = [E_list[s[k] % 5] for k in range(parti_client_num)]
        logger.info("#-----------随机选取用户完毕--------------------#")

        # --- Server-side Operations ---
        if round != 0:
            atk_model.set_state_dict(old_global_model.state_dict())
        old_global_model.set_state_dict(new_global_model.state_dict())
        old_global_model = old_global_model.to(device)
        logger.info("round %d test Acc == :" % (round))
        with paddle.no_grad():
            old_global_model.eval()
            total, correct = 0, 0
            for data in test_set_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                predictions = paddle.argmax(old_global_model(images), axis=1)
                total += np.prod(labels.shape)
                correct += (predictions.flatten() == labels.flatten()).sum().item()
            deltaS = correct / total * 100 - last_S
            last_S = correct / total * 100
            
            res_file = Path("./Acc/epsilon", '%s_%s_epsilon%d.csv' % (PRIVACY_MODE, DATAMODE, eps_global_init))
            res_file.parent.mkdir(parents=True, exist_ok=True)
            with open(res_file, mode='a', newline="") as train_file:
                writer_train2 = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer_train2.writerow([round, correct / total])
        logger.info('Accuracy: %d/%d = %.2f%%' % (correct, total, correct / total * 100))

        if t == rounds:
            logger.info("END")
            exit(0)

        # Step 1: Calculate privacy budget for the round
        if PRIVACY_MODE == 'adp':
            eps_round = epsRoundAccount.RoundlyAccount(old_global_model, eps_global, t, device, E)
        else:
            logger.info('privacy mode:%s,total eps:%d,rounds:%d' % (PRIVACY_MODE, eps_global_init, rounds))
            eps_round = eps_global_init / rounds

        eps_global -= eps_round
        # 保留每轮的eps
        logger.info("#step1 完毕")

        # ---------------------------------------------------------------------------------#
        # Step 2
        # 利用服务器的DLA对该轮选定的client进行攻击，获得一个预算分配的方案
        # 输入：eps_round
        # 输出：对每个client的隐私预算list: eps_clients[parti_client_num]
        eps_clients = [eps_round for _ in range(parti_client_num)]
        maxU = float('-inf')
        if PRIVACY_MODE == 'adp':
            # getDataExample 函数需要确保返回的是 PaddlePaddle 张量
            dataExample, labelExample = getDataExample(total_x, total_y)

            # 遍历上一轮参与计算的客户端梯度
            for key, original_dy_dx in server_save_update.items():
                # 将梯度张量分离计算图并克隆
                # 在 PaddlePaddle 中，语法与 PyTorch 相同
                dy_dx_ = [original_dy_dx[j].detach().clone() for j in range(len(original_dy_dx))]
                
                # 调用 DLA 函数计算 newU
                # 假设 DLA 函数内部已适配 PaddlePaddle
                newU = DLA(atk_model, dataExample, labelExample, dy_dx_, device, num_classes, channel)
                
                # 更新 server_save_deltaU 字典
                if server_save_deltaU.get(key):
                    server_save_deltaU[key].append(newU)
                else:
                    server_save_deltaU[key] = [newU]

            # 初始化最终列表
            finalList = [0 for n in range(parti_client_num)]
            for i in range(parti_client_num):
                if server_save_deltaU.get(s[i]):
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
            # 如果不是 'adp' 模式，所有客户端使用相同的 epsilon
            eps_clients = [eps_round for n in range(parti_client_num)]
        server_save_update.clear()
        logger.info("#step2 完毕")
        logger.info("Server端 waiting...")
        # Server端 waiting...
        # ---------------------------------------------------------------------------------#
        # Client端
        logger.info("#Client端 start")
        client_models = []
        for k_t in range(parti_client_num):
            local_model = copy.deepcopy(old_global_model)
            eps = eps_clients[k_t]
            loss_fn = paddle.nn.CrossEntropyLoss(reduction='none')
            k_instance = EPS_instance(participating_clients_data[k_t], local_model, E, bs, eps, device)

            # step3
            logger.info("Client:%d is running in %d round,mode = %s" % (k_t + 1, round, PRIVACY_MODE))
            if PRIVACY_MODE == 'adp':
                k_instance.runOurs(sorted_x_train, sorted_y_train, loss_fn, DecayClip, False, device, x_vali, y_vali)
            else:
                pass  # Placeholder for other privacy modes

            client_models.append(local_model)
            final_gradient = []
            for Gg, Gi in zip(old_global_model.parameters(), local_model.parameters()):
                final_gradient.append((Gg - Gi) / learning_rate)
            server_save_update[s[k_t]] = final_gradient

        # --- Server-side Aggregation ---
        worker_state_dict = [copy.deepcopy(x.state_dict()) for x in client_models]
        sum_parameters = None
        for x in worker_state_dict:
            if sum_parameters is None:
                sum_parameters = x
            else:
                for key in sum_parameters:
                    sum_parameters[key] = sum_parameters[key] + x[key]

        fed_state_dict = {}
        for var in sum_parameters:
            fed_state_dict[var] = sum_parameters[var] / parti_client_num

        new_global_model.set_state_dict(fed_state_dict)
        logger.info("global model after Avg:")
        t = round + 1
