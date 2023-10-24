import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import model.resnet as resnet
import model.resnet20 as resnet20
import torch.optim as optim
from torch.autograd import Variable
import utils
import numpy as np
import random
from torch.optim.lr_scheduler import StepLR
from loss import loss_fn_kd

# Hyper-parameter setting
parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='number of training samples per iteration (default: 128)')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, metavar='LR',
                    help='learning rate of studemt model')
parser.add_argument('-a', '--alpha', default=0.5, type=float,
                    help='trade-off between ce loss and kd loss')
parser.add_argument('-t', '--temperature', default=5, type=int, metavar='LR',
                    help='temperature for student model output')
parser.add_argument('-e', '--epoches', default=200, type=int, metavar='N',
                    help='training epoches')
parser.add_argument('--classes', type=str,nargs='+', help='...')

args = parser.parse_args()

use_gpu = torch.cuda.is_available()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_set = datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    normalize, ]))
test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    normalize, ]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

# loading teacher models
teacher_model = []
#model_1:134789 023689 245689 045679 023579
#catogaries = ['134789', '023689', '245689', '045679', '023579']
catogaries=args.classes
print(catogaries)
teacher_models_name = ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110']
classes = [0,1,2,3,4,5,6,7,8,9]

for index in range(len(teacher_models_name)):
    model = resnet20.__dict__[teacher_models_name[index]](num_classes=len(catogaries[index]))
    #multi-teacher生成的都是th文件，不是pth
    #ckpt_path = 'checkpoints/Teacher_' + str(index) + '_even.pth'
    ckpt_path = 'checkpoints/Teacher_' + str(index) + '_even.th'
    utils.load_checkpoint(ckpt_path, model)
    # utils.save_checkpoint(ckpt_path, model)
    if use_gpu:
        model = model.cuda()
    teacher_model.append(model)

# student model definition
model = resnet.ResNet18()
if use_gpu:
    model = model.cuda()
    print("Student model has been run on GPU")
# optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = StepLR(optimizer, step_size=150, gamma=0.1)


def train(model, teacher_model, optimizer, loss_fn_kd, dataloader, args, categ):
    count_n = 0
    teacher_model_eval = []
    # model.train()  设置成训练模式
    model.train()
    num_sample = 0

    for t_model in teacher_model:
        teacher_model_eval.append(t_model.eval())

    for i, (train_batch, labels_batch) in enumerate(dataloader):
        # move to gpu
        if use_gpu:
            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()

        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        output_batch = model(train_batch)

        output_teacher_batch = []
        with torch.no_grad():
            for t_model in teacher_model_eval:
                output_t_batch = t_model(train_batch)

                if use_gpu:
                    output_t_batch = output_t_batch.cuda()

                output_teacher_batch.append(output_t_batch)
        #output_batch：学生模型的预测输出；labels_batch：真实标签；output_teacher_batch:教师模型对样本batch的预测
        loss, kd_loss, ce_loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, args, categ)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # accuracy calculation
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        outputs = np.argmax(output_batch, axis=1)
        num_right = np.sum(outputs == labels_batch)
        count_n +=  num_right # 一个batch中预测输出与gt label一致的次数
        b = args.batch_size
        #print("total loss: ", round(loss.item() / b, 3), " kd loss: ", round(kd_loss.item() / b, 3)," ce loss: ", round(ce_loss.item() / b, 3), "acc_batch: ", round(num_right / b, 3))

        num_sample += len(labels_batch)




    # acc = count_n / (len(dataloader))
    acc = count_n / num_sample  # 所有batch的预测正确次数之和

    print("training acc: ", round(acc, 5))


    return acc


def evaluate(model, dataloader, args):
    count_n = 0
    model.eval()
    num_sample = 0
    num_right = [0] * 10
    num_sample_classes = [0] * 10

    for i, (data_batch, labels_batch) in enumerate(dataloader):
        if use_gpu:
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()

        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        output_batch = model(data_batch)

        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        outputs = np.argmax(output_batch, axis=1)
        count_n += np.sum(outputs == labels_batch)
        num_sample += len(labels_batch)

        num_right_batch = []
        num_sample_batch = []
        for j in range(len(classes)):
            num_right_class = np.count_nonzero((outputs == labels_batch) & (outputs == classes[j]))  # 一个batch中每个类别分类正确的次数
            num_sample_class = np.sum(labels_batch == classes[j])  # 一个batch中每个类别的样本的个数
            num_right_batch.insert(j, num_right_class)#一个batch中所有类别分类正确的次数
            num_sample_batch.insert(j, num_sample_class)

        num_right = [a + b for a, b in zip(num_right, num_right_batch)]
        num_sample_classes = [a + b for a, b in zip(num_sample_classes, num_sample_batch)]


    acc = count_n / num_sample
    acc_classes = [a / b for a, b in zip(num_right, num_sample_classes)]

    print("evaluation acc: ", round(acc, 5), "\nevaluation acc of classes:", acc_classes)


    return acc


if __name__ == '__main__':
    random.seed(230)
    torch.manual_seed(230)
    if use_gpu: torch.cuda.manual_seed(230)

    model_dir = 'save_checkpoints'
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epoches):
        print('Epoch: ', epoch + 1)
        scheduler.step()

        train_acc = train(model, teacher_model, optimizer, loss_fn_kd, train_loader, args, catogaries)

        test_acc = evaluate(model, test_loader, args)

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict()},
                              checkpoint=model_dir,
                              name=str(epoch + 1))

        if test_acc >= best_acc:
            best_acc = test_acc
            best_epoch = epoch
        print("best acc: ", best_acc, " epoch: ", best_epoch)
        print('\n')
