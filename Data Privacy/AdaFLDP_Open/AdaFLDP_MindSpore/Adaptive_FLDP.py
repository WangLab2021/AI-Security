import time
import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter, context
from mindspore.common.initializer import initializer, Uniform
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset, Cifar10Dataset
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import argparse
import pickle
import copy
import csv
import matplotlib.pyplot as plt
from download import download
from pathlib import Path

# Global parameters would need to be defined or imported
from Global_Parameter import *
from EPS_round import *
from EPS_instance import *
from tools import *

parser = argparse.ArgumentParser(description="FLDP")
parser.add_argument("--ds", type=str, help="", default='mnist')
parser.add_argument("--p", type=str, help="", default='ori')
parser.add_argument("--e", type=int, help="", default=4)
parser.add_argument("--b", type=int, help="", default=128)
parser.add_argument("--eps", type=int, help="", default=150)
parser.add_argument("--c", type=str, help="", default='0')
parser.add_argument("--w", type=int, help="", default=100)
parser.add_argument("--r", type=float, help="", default=0.02)
args = parser.parse_args()

total_client_num = args.w
selected_rate = args.r
parti_client_num = int(total_client_num * selected_rate)
bs = args.b
E = args.e
eps_global_init = args.eps
DATAMODE = args.ds
PRIVACY_MODE = args.p

# Set device context
device = 'CPU'
ms.set_device(device)
context.set_context(mode=context.PYNATIVE_MODE)
logger.info("Running on %s" % device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Dense)):
        m.weight.set_data(initializer(Uniform(0.5), m.weight.shape))
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.set_data(initializer(Uniform(0.5), m.bias.shape))


class LeNet(nn.Cell):
    def __init__(self, channel=3, num_classes=10, H=32, W=32):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.SequentialCell(
            nn.Conv2d(channel, 12, kernel_size=5, pad_mode='pad', padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, pad_mode='pad', padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, pad_mode='pad', padding=5 // 2, stride=1),
            act(),
        )
        final_in = (H // 4) * (W // 4) * 12
        self.fc = nn.SequentialCell(nn.Dense(final_in, num_classes))

    def construct(self, x):
        out = self.body(x)
        out = ops.flatten(out, start_dim=1)
        out = self.fc(out)
        return out


def get_data(datamode=DATAMODE):
    # ! MindSpore 没有dataloader，在此函数外部直接进行dataloader的重新加载
    transform = transforms.Compose([vision.Resize((32, 32)), vision.CenterCrop(32), vision.ToTensor()])

    if datamode == 'mnist':
        dataset_path = Path('./data/MNIST_Data')
        dataset_files = [
            'test/t10k-images-idx3-ubyte',
            'test/t10k-labels-idx1-ubyte',
            'train/train-images-idx3-ubyte',
            'train/train-labels-idx1-ubyte',
        ]
        if not all([(dataset_path / f).absolute().exists() for f in dataset_files]):
            dataset_path.mkdir(parents=True, exist_ok=True)
            mnist_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
            download(mnist_url, dataset_path.parent.as_posix(), kind="zip", replace=True)

        train_dataset = MnistDataset((dataset_path / 'train').as_posix(), shuffle=False)
        test_dataset = MnistDataset((dataset_path / 'test').as_posix(), shuffle=False)
    elif datamode == 'cifar10':
        dataset_path = Path('./data/cifar-10-batches-bin')
        dataset_files = [*[f'data_batch_{i}.bin' for i in range(1, 6)], 'test_batch.bin', 'batches.meta.txt']
        if not all([(dataset_path / f).absolute().exists() for f in dataset_files]):
            dataset_path.mkdir(parents=True, exist_ok=True)
            cifar10_url = (
                "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"
            )
            download(cifar10_url, dataset_path.parent.as_posix(), kind="tar.gz", replace=True)

        train_dataset = Cifar10Dataset(dataset_path.as_posix(), usage='train', shuffle=False)
        test_dataset = Cifar10Dataset(dataset_path.as_posix(), usage='test', shuffle=False)
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

        sorted_x_train = Tensor(X_train.transpose(0, 3, 1, 2))
        sorted_y_train = y_train
        total_x = sorted_x_train[0:]
        total_y = Tensor(y_train, dtype=ms.int32)
        x_test = Tensor(X_test.transpose(0, 3, 1, 2))
        x_vali = x_test
        y_vali = Tensor(y_test, dtype=ms.int32)

        logger.info(f"dataSet:{DATAMODE}, get train points:{len(sorted_x_train)}, get test points:{len(x_test)}")
        return (
            total_x,
            total_y,
            sorted_x_train,
            Tensor(sorted_y_train, dtype=ms.int32),
            x_vali,
            y_vali,
            x_test,
            Tensor(y_test, dtype=ms.int32),
            None,
        )

    if DATAMODE == 'mnist':
        index_start = 50000
    else:
        index_start = 40000

    x_train, y_train = zip(*train_dataset.map(operations=transform, input_columns=["image"]))
    x_train: ms.Tensor = ops.stack(x_train).astype(np.float32)
    y_train: ms.Tensor = ops.stack(y_train).astype(np.int64)
    
    logger.info(x_train.flatten().max())

    total_x = x_train[0:]
    total_y = y_train[0:]

    x_vali = x_train[index_start:]
    y_vali = y_train[index_start:]
    x_train = x_train[:index_start]
    y_train = y_train[:index_start]

    indices_train = np.argsort(y_train.asnumpy())
    sorted_x_train = ms.from_numpy(x_train.numpy()[indices_train])
    sorted_y_train = ms.from_numpy(y_train.numpy()[indices_train])

    x_test, y_test = zip(*test_dataset.map(operations=transform, input_columns=["image"]))
    x_test = ops.stack(x_test).astype(np.float32)
    y_test = ops.stack(y_test).astype(np.int64)

    logger.info(f"dataSet:{DATAMODE}, get train points:{len(sorted_x_train)}, get test points:{len(x_test)}")
    return total_x, total_y, sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test, None


def getDataExample(x_, y_):
    if DATAMODE == 'lfw':
        id = 291
    elif DATAMODE == 'mnist':
        id = 1123
    else:
        id = 48850
    y = ops.reshape(y_[id], (1,))
    return x_[id], y


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
                srcs=[Path("../app/", client_set_file.absolute().relative_to(Path(__file__).parent.absolute())).resolve()],
                dsts=[client_set_file]
            )
        client_set = pickle.load(open(client_set_file, 'rb'))
    elif DATAMODE == 'mnist':
        client_set_file = Path('./DATA/clients', str(total_client_num) + 'clients.npy')
        if not client_set_file.exists():
            copy_files(
                srcs=[Path("../app/", client_set_file.absolute().relative_to(Path(__file__).parent.absolute())).resolve()],
                dsts=[client_set_file]
            )
        client_set = np.load(client_set_file)
    else:
        client_set_file = Path('./DATA/clients_cifar10', str(total_client_num) + 'clients.npy')
        if not client_set_file.exists():
            copy_files(
                srcs=[Path("../app/", client_set_file.absolute().relative_to(Path(__file__).parent.absolute())).resolve()],
                dsts=[client_set_file]
            )
        client_set = np.load(client_set_file)
    logger.info("load client set ({}) from {}".format(len(client_set), client_set_file))

    total_x, total_y, sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test, test_dataset = get_data()
    logger.info("get data done, total_x shape: {}, total_y shape: {}".format(total_x.shape, total_y.shape))
    logger.info("sorted_x_train shape: {}, sorted_y_train shape: {}".format(sorted_x_train.shape, sorted_y_train.shape))
    logger.info("x_vali shape: {}, y_vali shape: {}".format(x_vali.shape, y_vali.shape))
    logger.info("x_test shape: {}, y_test shape: {}".format(x_test.shape, y_test.shape))

    vali_set = list(zip(x_vali.asnumpy(), y_vali.asnumpy()))
    valiloader = ms.dataset.GeneratorDataset(vali_set, column_names=["image", "label"], shuffle=False).batch(100)
    test_set = list(zip(x_test.asnumpy(), y_test.asnumpy()))
    testloader = ms.dataset.GeneratorDataset(test_set, column_names=["image", "label"], shuffle=False).batch(100)

    channel = 3 if DATAMODE != 'mnist' else 1
    num_classes = 10 if DATAMODE != 'lfw' else 106
    H = W = 32

    model = LeNet(channel=channel, num_classes=num_classes, H=H, W=W)
    weights_init(model)

    new_global_model = copy.deepcopy(model)
    old_global_model = copy.deepcopy(model)
    atk_model = copy.deepcopy(model)

    starting_time = time.time()
    eps_global = eps_global_init
    epsRoundAccount = EPS_round(valiloader)
    client_models = []
    server_save_update = {}
    server_save_deltaU = {}

    last_S = 10
    t = 0

    for round in range(rounds + 1):
        tested = False
        np.random.seed(round)
        perm = np.random.permutation(total_client_num)
        s = perm[0:parti_client_num].tolist()
        participating_clients_data = [client_set[k] for k in s]
        E_list = [2, 4, 1, 2, 4]
        client_random = [E_list[s[k] % 5] for k in range(parti_client_num)]
        logger.info("#-----------随机选取用户完毕--------------------#")
        # ---------------------------------------------------------------------------------#
        # Step 1
        # 划分该轮的epsilon，用到vali_set（10000个样本）与old_global_model，并利用得到的准确率进行一个计算
        if round != 0:
            atk_model.load_state_dict(old_global_model.parameters_dict())
        old_global_model.load_state_dict(new_global_model.parameters_dict())

        logger.info(f"round {round} test Acc == :")
        old_global_model.set_train(False)
        total = 0
        correct = 0

        for data in testloader.create_dict_iterator():
            images = data['image']
            labels = data['label']
            outputs = old_global_model(images)
            predictions = ops.argmax(outputs, 1)
            total += labels.shape[0]
            correct += (predictions == labels).sum().asnumpy()

        accuracy = correct / total * 100
        deltaS = accuracy - last_S
        last_S = accuracy

        res_filepath = Path(f'./Acc/epsilon/{PRIVACY_MODE}_{DATAMODE}_epsilon{eps_global_init}.csv')
        if not res_filepath.exists():
            res_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save accuracy
        with open(res_filepath, mode='a', newline="") as train_file:
            writer = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([round, accuracy / 100])

        logger.info('Accuracy: %d/%d = %.2f%%' % (correct, total, correct / total * 100))

        if t == rounds:
            logger.info("END")
            exit(0)

        if PRIVACY_MODE == 'adp':
            eps_round = epsRoundAccount.RoundlyAccount(old_global_model, eps_global, t, device, E)
        else:
            logger.info(f'Privacy mode: {PRIVACY_MODE}, total eps: {eps_global_init}, rounds: {rounds}')
            eps_round = eps_global_init / rounds

        eps_global -= eps_round
        logger.info("#step1 完毕")
        # ---------------------------------------------------------------------------------#
        # Step 2
        # 利用服务器的DLA对该轮选定的client进行攻击，获得一个预算分配的方案
        # 输入：eps_round
        # 输出：对每个client的隐私预算list: eps_clients[parti_client_num]
        eps_clients = [eps_round for n in range(parti_client_num)]
        maxU = float('-inf')

        if PRIVACY_MODE == 'adp':
            dataExample, labelExample = getDataExample(total_x, total_y)

            for key, original_dy_dx in server_save_update.items():
                dy_dx_ = [original_dy_dx[j].copy() for j in range(len(original_dy_dx))]
                newU = DLA(atk_model, dataExample, labelExample, dy_dx_, device, num_classes, channel)
                server_save_deltaU.setdefault(key, []).append(newU)

            finalList = [0.0 for n in range(parti_client_num)]
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

            logger.info("final list:", finalList)
            logger.info("final list:", maxU)
        else:
            eps_clients = [eps_round for n in range(parti_client_num)]

        server_save_update.clear()
        logger.info("#step2 completed")
        logger.info("Server waiting...")
        # Server端 waiting...
        # ---------------------------------------------------------------------------------#
        # Client端
        logger.info("#Client start")
        client_models = []
        norm_client_gradient = []

        for k_t in range(parti_client_num):
            local_model = copy.deepcopy(old_global_model)
            Epoch = E
            eps = eps_clients[k_t]
            loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
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
                pass

            client_models.append(local_model)
            final_gradient = [
                (Gg - Gi) / learning_rate
                for Gi, Gg in zip(local_model.trainable_params(), old_global_model.trainable_params())
            ]
            server_save_update[s[k_t]] = final_gradient
        # Client端 END
        #  Server端  聚合----------------------------------------------------------------------#
        sum_parameters = {}
        for model in client_models:
            model_weights = model.parameters_dict()
            for i, weight in model_weights.items():
                if i not in sum_parameters:
                    sum_parameters[i] = ops.zeros_like(weight)
                sum_parameters[i] += weight

        for i, weight in sum_parameters.items():
            sum_parameters[i] = weight / parti_client_num

        new_global_model.load_state_dict(sum_parameters)
        logger.info("global model after Avg:")
        t = round + 1
