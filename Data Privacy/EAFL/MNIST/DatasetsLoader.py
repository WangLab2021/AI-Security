import numpy as np
import struct
import syft as sy
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import numpy as np
import struct
import syft as sy
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
hook = sy.TorchHook(torch)

logger = logging.getLogger(__name__)

class testDataLoader():
    def __init__(self, datasets):
        self.data = datasets['test_images']
        self.labels = datasets['test_labels']
    def __len__(self):
        return len(self.labels.tolist())
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label

def decode_idx3_ubyte(img_file, datasize):
    """
    解析idx3文件，加载图片
    :param img_file: 图片路径
    :return: 图片，数据格式为list，[images_num, 28, 28]
    """
    # 读取二进制数据
    buf_img = open(img_file, 'rb').read()
    # 解析文件头信息
    offset = 0
    fmt_header = '>II'  # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, img_num = struct.unpack_from(fmt_header, buf_img, offset)
    print("magic number is {}, images number is {}.".format(magic_number, img_num))

    offset += struct.calcsize('>IIII')
    img_list = []
    image_size = 784
    fmt_image = '>' + str(
        image_size) + 'B'  # 图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    for i in range(img_num):
        if i > datasize:
            break;
        img = struct.unpack_from(fmt_image, buf_img, offset)
        img_list.append(np.reshape(img, (28, 28)))
        offset += struct.calcsize(fmt_image)
    return torch.tensor(img_list)

def decode_idx1_ubyte(label_file, datasize):
    """
    解析idx1文件
    :param label_file: 标签路径
    :return: 标签，格式为 list
    """
    # 读取二进制数据
    bin_data = open(label_file, 'rb').read()
    # 解析文件头信息
    offset = 0
    fmt_header = '>II'
    magic_number, label_num = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic_number is {}, label number is {}'.format(magic_number, label_num))
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    label_list = []
    for i in range(label_num):
        if i > datasize:
            break;
        label_item = int(struct.unpack_from('>B', bin_data, offset)[0])
        label_list.append(label_item)
        offset += struct.calcsize('B')
    print(len(label_list))
    return torch.tensor(label_list)

def loadWriters(writers_file, datasize):
    '''
    加载作者信息
    :param writers_file: 作者文件
    :return: 返回作者列表，数据格式为list
    '''
    f = open(writers_file)
    datas = f.readlines()
    writers = []
    for i in range(len(datas)):
        if i > datasize:
            break
        data = int(float(datas[i].split()[0]))
        writers.append(data)
    return writers

def loadDatesets(trainDataSize, testDataSize, dataType='mnist'):
    train_images_file = r'../data/EMNIST-ByClass/' + dataType + '/emnist-' + dataType + '-train-images-idx3-ubyte'
    train_labels_file = r'../data/EMNIST-ByClass/' + dataType + '/emnist-' + dataType + '-train-labels-idx1-ubyte'
    train_writers_file = '../data/EMNIST-ByClass/' + dataType + '/emnist-' + dataType + '-train-writers.txt'
    test_images_file = r'../data/EMNIST-ByClass/' + dataType + '/emnist-' + dataType + '-test-images-idx3-ubyte'
    test_labels_file = r'../data/EMNIST-ByClass/' + dataType + '/emnist-' + dataType + '-test-labels-idx1-ubyte'
    test_writers_file = '../data/EMNIST-ByClass/' + dataType + '/emnist-' + dataType + '-test-writers.txt'

    train_images = decode_idx3_ubyte(train_images_file, trainDataSize)
    train_labels = decode_idx1_ubyte(train_labels_file, trainDataSize)
    train_writers = loadWriters(train_writers_file, trainDataSize)
    test_images = decode_idx3_ubyte(test_images_file, testDataSize)
    test_labels = decode_idx1_ubyte(test_labels_file, testDataSize)
    test_writers = loadWriters(test_writers_file, testDataSize)

    all_writers = set(train_writers)
    fun = dict((name, value) for name, value in zip(all_writers, range(len(all_writers))))
    train_writers = torch.tensor(train_writers)
    test_writers = torch.tensor(test_writers)
    for writer in all_writers:
        train_writers[train_writers == writer] = fun[writer]
        test_writers[test_writers == writer] = fun[writer]

    datasets = {'train_images': train_images, 'train_labels': train_labels, 'train_writers': train_writers,
                'test_images': test_images, 'test_labels': test_labels, 'test_writers': test_writers}
    # 查看前十个数据及其标签以读取是否正确
    '''
    for i in range(10):
        plt.imshow(train_images[i], cmap='gray')
        plt.pause(1)
        plt.show()
    '''
    print('done')
    return datasets

def testImages(datasets, P=5000):
    images = datasets['test_images']
    labels = datasets['test_labels']
    writers = datasets['test_writers']
    '''
    test_images = []
    test_labels = []
    for writer in selectWriter:
        index = (writers==writer)
        test_images.append(images[index])
        test_labels.append(labels[index])
    '''
    test_data = {'test_images': images, 'test_labels': labels}
    return testDataLoader(test_data)

def dataset_federate_noniid(dataset, workers, args, net='NOT CNN' ):
    """
    Add a method to easily transform a torch.Dataset or a sy.BaseDataset
    into a sy.FederatedDataset. The dataset given is split in len(workers)
    part and sent to each workers
    """
    logger.info(f"Scanning and sending data to {', '.join([w.id for w in workers])}...")
    datas = dataset['train_images']
    labels = dataset['train_labels']
    writers = dataset['train_writers']
    datasNum = torch.zeros([args.user_num])
    datasets = []
    distance = 10
    workerNum = 0
    lunNum = 0
    lundata = []
    lunLabel = []
    for i in range(torch.max(writers)):
        lunNum = lunNum + 1
        index = (writers == i)
        lundata.append(datas[index,:,:])
        lunLabel.append(labels[index])
        if lunNum >= distance:
            lunNum = 0
            lundata = torch.cat(lundata, 0)
            lunLabel = torch.cat(lunLabel, 0)
            for j in range(10):
                index = (lunLabel == j)
                label = lunLabel[index]
                data = lundata[index]
                worker = workers[workerNum]
                datasNum[workerNum] = torch.sum(index)
                workerNum = workerNum + 1
                logger.debug("Sending data to worker %s", worker.id)
                data = data.send(worker)
                label = label.send(worker)
                datasets.append(sy.BaseDataset(data, label))  # .send(worker)
                if workerNum >= args.user_num:
                    break
            lundata = []
            lunLabel = []
            lunNum = 0
        if workerNum >= args.user_num:
            break

    logger.debug("Done!")
    return sy.FederatedDataset(datasets), datasNum

'''
class Argument():
    def __init__(self):
        self.user_num = 1000
        self.K = 10
        self.lr = 0.001
        self.batch_size = 8
        self.Clip_Bound = 1
        self.test_batch_size = 128
        self.total_iterations = 10000
        self.seed = 1
        self.cuda_use = False


args = Argument()
workers = []

for i in range(1, args.user_num+1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i,i))
    exec('workers.append(user{})'.format(i))    # 列表形式存储用户
dataType = 'mnist'   # 可选bymerge, byclass, digits, mnist, letters, balanced
datasets = loadDatesets(trainDataSize = 70000, testDataSize = 20000, dataType=dataType)
#训练集，测试集
federated_data, datasNum = dataset_federate_noniid(datasets, workers, args)
'''