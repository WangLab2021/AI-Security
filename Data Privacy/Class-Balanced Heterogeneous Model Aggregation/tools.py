import argparse
from collections import defaultdict
from pathlib import Path
from loguru import logger
import sys

import numpy as np


def get_train_teacher_argparser(model_names=["Alexnet", 'densenet', 'googlenet', 'resnet']):

    parser = argparse.ArgumentParser(description='Propert ResNets')
    parser.add_argument(
        '--arch',
        '-a',
        metavar='ARCH',
        default='ResNet32',
        choices=model_names,
        help='model architecture: ' + ' | '.join(model_names) + ' (default: ResNet32)',
    )
    parser.add_argument(
        '-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)'
    )
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument(
        '--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)'
    )
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument(
        '--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate'
    )
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument(
        '--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 5e-4)'
    )
    parser.add_argument('--printfreq', '-p', default=25, type=int)
    parser.add_argument('--trainset', default=1000, type=int)
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)'
    )
    parser.add_argument(
        '-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set'
    )
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true', help='use half-precision(16-bit) ')
    parser.add_argument(
        '--save-dir',
        dest='save_dir',
        help='The directory used to save the trained models',
        default='save_model',
        type=str,
    )
    parser.add_argument(
        '--save-every',
        dest='save_every',
        help='Saves checkpoints at every specified number of epochs',
        type=int,
        default=20,
    )
    parser.add_argument(
        '-c',
        '--classes',
        type=str,
        nargs='+',
        default=['134789', '023689', '245689', '045679', '023579'],
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU when it's available"
    )
    return parser


def get_train_student_argparser():
    parser = argparse.ArgumentParser(description='Knowledge Distillation')
    parser.add_argument(
        '-j', '--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        default=256,
        type=int,
        metavar='N',
        help='number of training samples per iteration (default: 128)',
    )
    parser.add_argument(
        '-lr', '--learning_rate', default=0.001, type=float, metavar='LR', help='learning rate of studemt model'
    )
    parser.add_argument('-a', '--alpha', default=0.5, type=float, help='trade-off between ce loss and kd loss')
    parser.add_argument(
        '-t', '--temperature', default=5, type=int, metavar='LR', help='temperature for student model output'
    )
    parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N', help='training epochs')
    parser.add_argument('--classes', type=str, nargs='+', default=['134789', '023689', '245689', '045679', '023579'])
    parser.add_argument(
        '--save-dir',
        dest='save_dir',
        help='The directory used to save the trained models',
        default='save_model',
        type=str,
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU when it's available"
    )
    return parser


def build_indexes(datasets):
    index_dict = defaultdict(list)
    for index, data in enumerate(datasets):
        # 从 numpy.ndarray 中提取标量整数值
        if isinstance(data[1], int):
            label = data[1]
        elif isinstance(data[1], np.ndarray):
            label = data[1].item()
        else:
            raise ValueError("Unsupported label type: {}".format(type(data[1])))
        index_dict[label].append(index)
    return index_dict


def label_transfer(catogaries, real_label):
    assert str(real_label) in catogaries, "Can't transfer real label"
    return catogaries.find(str(real_label))


def label_transfer_inverse(catogaries, train_label):
    assert len(catogaries) > train_label, "Can't transfer train label"
    return int(catogaries[train_label])


def split_data_by_classes(index_dict, catogaries, dataset_size, mode='train'):
    specific_sets = []
    if mode == 'train':
        record = defaultdict(int)
        for catogary in catogaries:
            specific_set = []
            for single_class in catogary:
                indexes = index_dict[int(single_class)][
                    record[int(single_class)] : record[int(single_class)] + dataset_size
                ]
                record[int(single_class)] += dataset_size
                specific_set.extend(indexes)
            specific_sets.append(specific_set)
    else:  # valid
        for catogary in catogaries:
            specific_set = []
            for single_class in catogary:
                indexes = index_dict[int(single_class)][:200]
                specific_set.extend(indexes)
            specific_sets.append(specific_set)
    return specific_sets


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logger(log_file=None, level='DEBUG'):
    logger.remove()
    if log_file is not None:
        f = Path(log_file)
        f.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            f,
            rotation='1 day',
            retention='10 days',
            level=level,
            # format='{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}',
        )
    logger.add(sys.stdout, level=level)
