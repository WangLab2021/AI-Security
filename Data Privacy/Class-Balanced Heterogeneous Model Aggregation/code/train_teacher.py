
import argparse
import os
import time
from sklearn.metrics import classification_report

import torch

# import ctypes
# 引入caffe2_nvrtc库，问题解决
# ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import resnet20,vgg,alexnet,densenet,googlenet



from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

model_names = ["Alexnet",'densenet','googlenet','resnet']

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: ResNet32)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--printfreq', '-p', default=25, type=int)
parser.add_argument('--trainset', default=1000, type=int)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='savemodel_dif', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=20)
parser.add_argument('-c','--classes', type=str,nargs='+', help='...')
parser.add_argument('--modeldir', type=str, help='...')
args = parser.parse_args()
best_prec1 = 0

use_cuda = torch.cuda.is_available()


# use_cuda = False
def build_indexes(datasets):
    # 建立每个类别中图片在datasets中的索引
    # 返回值（defaultdict）key：类别号，value：datasets中此类别的样本序号
    # index_list = [[]]*10
    index_dict = defaultdict(list)
    for index, data in enumerate(datasets):
        index_dict[data[1]].append(index)
    return index_dict


def label_transfer(catogaries, real_label):
    '''from real label to train label'''
    assert str(real_label) in catogaries, "Can't transfer real label"
    return catogaries.find(str(real_label))


def label_transfer_inverse(catogaries, train_label):
    '''from train label to real label'''
    assert len(catogaries) > train_label, "Can't transfer train label"
    return int(catogaries[train_label])


class SubDataset(Dataset):
    # 定义一个子集数据类
    def __init__(self, indexes, whole_set, catogaries, transform=None):
        super().__init__()
        self.indexes = indexes
        self.whole_set = whole_set
        self.transform = transform
        self.catogaries = catogaries

    def __getitem__(self, index: int):
        global_index = self.indexes[index]
        image, label = self.whole_set[global_index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label_transfer(self.catogaries, label)

    def __len__(self):  # 返回整个数据集的大小
        return len(self.indexes)


'''
train:每个教师(共K个)模型能训练的每个类别取前1000个样本序号 list(K)={list(L1*1000),list(L2*1000),...
valid:200个
'''


def split_data_by_classes(index_dict, catogaries, mode='train'):
    specific_sets = []
    num=args.trainset
   # catogaries = catogaries.split()
    if mode == 'train':
        record = defaultdict(int)
        for catogary in catogaries:  # e.g. '1348'
            specific_set = []
            for single_class in catogary:  # e.g. '1'
                indexes = index_dict[int(single_class)][record[int(single_class)]:record[int(single_class)] + num]
                record[int(single_class)] += num
                specific_set.extend(indexes)
            specific_sets.append(specific_set)

    else:
        # valid
        for catogary in catogaries:  # e.g. '1348'
            specific_set = []
            for single_class in catogary:  # e.g. '1'
                indexes = index_dict[int(single_class)][:200]
                specific_set.extend(indexes)
            specific_sets.append(specific_set)
    return specific_sets


def main():
    global  best_prec1

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #catogaries = '134789 023689 245689 045679 023579'
    catogaries=args.classes
    #print(catogaries)
    global g_catogaries
    g_catogaries = args.classes # list
    download = False
    raw_train_set = datasets.FashionMNIST(root='./data', train=True, download=download)
    raw_valid_set = datasets.FashionMNIST(root='./data', train=False, download=download)
    train_index_dict = build_indexes(raw_train_set)
    # print(train_index_dict)
    # exit()
    train_sets = split_data_by_classes(train_index_dict, catogaries)
    val_index_dict = build_indexes(raw_valid_set)
    val_sets = split_data_by_classes(val_index_dict, catogaries, mode='val')
    teacher_models =["Googlenet","Alexnet",'Densenet','Resnet20','Resnet32']
    for teacher_index in range(len(catogaries)):
        global g_teacher_index
        g_teacher_index = teacher_index
        #  Train Teacher number teacher_index
        print(f"Training teacher {teacher_index}"+f" Model: {teacher_models[teacher_index]}")
        num_classes=len(catogaries[teacher_index])
        if teacher_index==0: model=googlenet.GoogleNet(num_classes=num_classes)
        elif teacher_index == 1:   model = alexnet.AlexNet(num_classes=num_classes)
        elif teacher_index==2: model=densenet.DenseNet(num_classes=num_classes)
        elif teacher_index==3: model=resnet20.ResNet20(num_classes=num_classes)
        else: model=resnet20.ResNet32(num_classes=num_classes)


        # model.to('cpu')
        cudnn.benchmark = True

        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        current_train_set = train_sets[teacher_index]
        current_val_set = val_sets[teacher_index]

        # exit()
        train_loader = torch.utils.data.DataLoader(
            SubDataset(current_train_set, raw_train_set, catogaries=catogaries[teacher_index],
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            SubDataset(current_val_set, raw_valid_set, catogaries=catogaries[teacher_index],
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        # define loss function (criterion) and pptimizer
        criterion = nn.CrossEntropyLoss()
        if use_cuda:
            model.cuda()
            criterion = nn.CrossEntropyLoss().cuda()

        if args.half:
            model.half()
            criterion.half()

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150],
                                                            last_epoch=args.start_epoch - 1)

        if args.arch in ['ResNet1202', 'ResNet110']:
            # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
            # then switch back. In this implementation it will correspond for first epoch.
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1

        if args.evaluate:
            validate(val_loader, model, criterion, True)
            return

        for epoch in range(args.start_epoch, args.epochs):

            # train for one epoch
            #  print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            train(train_loader, model, criterion, optimizer, epoch)
            lr_scheduler.step()

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            # if epoch > 0 and epoch % args.save_every == 0:
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         'state_dict': model.state_dict(),
            #         'best_prec1': best_prec1,
            #     }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))
        dirs=os.path.join(args.save_dir,args.modeldir)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(dirs,f'Teacher_{teacher_index}_even.th'))
        # print("done!!!!!!!!!!!!!!!!!!!!!")


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    report_data = [[], []]  # y_true, y_pred
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        input_var = torch.autograd.Variable(input)
        if use_cuda:
            target = target.cuda()
            input_var = input_var.cuda()
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # compute output
        # output = model(input_var)[-1]#shape [128,6]
        output = model(input_var)  # output tuple[5]
        #output = output[-1]

        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        # 这里prec1具体是什么意思：预测正确的百分比
        (prec1, *_), r = accuracy(output.data, target, tc_idx=g_teacher_index)
        report_data[0].extend(r[0])
        report_data[1].extend(r[1])

        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))  # top1:准确度

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0  :
        # if i == len(train_loader) - 1 :
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'current lr {lr:.5e}\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #         epoch + 1, i + 1, len(train_loader), lr=optimizer.param_groups[0]['lr'], batch_time=batch_time,
        #         data_time=data_time, loss=losses, top1=top1))
    if epoch % args.printfreq ==args.printfreq-1:
        print('Training Epoch:',epoch+1)
        print("=" * 70 + "\n", classification_report(report_data[0], report_data[1]), "=" * 70 + "\n")


def validate(val_loader, model, criterion, epoch):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    report_data = [[], []]  # y_true, y_pred
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var = torch.autograd.Variable(input)
            if use_cuda:
                target = target.cuda()
                input_var = input_var.cuda()
            target_var = torch.autograd.Variable(target)

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            (prec1, *_), r = accuracy(output.data, target, tc_idx=g_teacher_index)
            report_data[0].extend(r[0])
            report_data[1].extend(r[1])

            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #         i + 1, len(val_loader), batch_time=batch_time, loss=losses,
            #         top1=top1))
    if epoch % args.printfreq ==args.printfreq-1:
        print("Validate Epoch:",epoch+1)
        print("=" * 70 + "\n", classification_report(report_data[0], report_data[1]), "=" * 70)
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        print("\n\n")

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint/resnet20.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


# val avg sum count
class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def accuracy(output, target, topk=(1,), tc_idx=0):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # .t什么意思
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    y_true = target.view(-1).cpu().numpy()
    y_pred = pred.view(-1).cpu().numpy()
    report = (
        [label_transfer_inverse(g_catogaries[tc_idx], y) for y in y_true],
        [label_transfer_inverse(g_catogaries[tc_idx], y) for y in y_pred]
    )
    # y_true和y_pred得到的是相对类别，report是绝对类别

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, report


if __name__ == '__main__':
    print("Training Teacher [ensemble]")
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('----------------------')
    main()

