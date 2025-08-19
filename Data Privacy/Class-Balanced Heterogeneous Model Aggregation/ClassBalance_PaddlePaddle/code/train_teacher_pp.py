import os
import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import classification_report

from tqdm import tqdm

import paddle
import paddle.nn as nn
from paddle.io import DataLoader, Dataset
from paddle.vision import transforms as cvtf
from paddle.vision import datasets

import models_pp.resnet20 as resnet20
from tools import *


class SubDataset(Dataset):
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
        # FIX: 在传递给 label_transfer 之前，使用 .item() 提取整数
        return image, label_transfer(self.catogaries, label.item())

    def __len__(self):
        return len(self.indexes)


def save_checkpoint(state, is_best, filename='checkpoint.pdparams'):
    # 使用 paddle.save 保存模型参数
    paddle.save(state, filename)
    logger.info(f"Checkpoint saved to {filename}")


def accuracy(output, target, g_catogaries, tc_idx=0, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    # pred 的 dtype 是 int64
    _, pred = paddle.topk(output, k=maxk, axis=1, largest=True, sorted=True)
    pred = paddle.transpose(pred, perm=[1, 0])

    # FIX: 在比较前，将 target 的 dtype 从 int32 转换为 int64
    target_cast = target.astype('int64')
    correct = paddle.equal(pred, target_cast.reshape([1, -1]).expand_as(pred))

    # .numpy() 直接从 tensor 获取 numpy 数组
    y_true = target.reshape([-1]).numpy()
    y_pred = pred.reshape([-1]).numpy()

    report = (
        [label_transfer_inverse(g_catogaries[tc_idx], y) for y in y_true],
        [label_transfer_inverse(g_catogaries[tc_idx], y) for y in y_pred],
    )

    res = []
    for k in topk:
        # PaddlePaddle 中 bool 不能直接 sum，需要先 cast
        correct_k = paddle.sum(paddle.cast(correct[:k], 'float32'))
        res.append(correct_k * (100.0 / batch_size))
    return res, report


def train(args, train_loader, model, criterion, optimizer, scaler, epoch, g_catogaries, g_teacher_index):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()
    report_data = [[], []]

    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Training Epoch {epoch+1}", unit="batch")

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # AMP (auto_cast)
        with paddle.amp.auto_cast(enable=args.half, level='O1'):
            output = model(input)
            loss = criterion(output, target)

        losses.update(loss.item(), input.shape[0])

        # 使用 GradScaler 进行反向传播
        scaled = scaler.scale(loss)
        scaled.backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.clear_grad()

        # 计算精度
        (prec1, *_), r = accuracy(output, target, g_catogaries, g_teacher_index)
        report_data[0].extend(r[0])
        report_data[1].extend(r[1])
        top1.update(prec1.item(), input.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix(
            loss="{}({})".format(round(losses.val, 5), round(losses.avg, 5)),
            top1="{}({})".format(round(top1.val, 2), round(top1.avg, 2)),
            lr=optimizer.get_lr(),
        )
        pbar.update(1)

        if i % args.printfreq == 0:
            # if i == len(train_loader) - 1 :
            # 获取当前学习率
            current_lr = optimizer.get_lr()

            # logger.info(
            #     'Epoch: [{0}][{1}/{2}]\t'
            #     'current lr {lr:.5e}\t'
            #     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #         epoch + 1,
            #         i + 1,
            #         len(train_loader),
            #         lr=current_lr,
            #         batch_time=batch_time,
            #         data_time=data_time,
            #         loss=losses,
            #         top1=top1,
            #     )
            # )
    pbar.close()
    logger.success(
        "\n".join(
            [
                "=" * 70,
                f"Training Epoch {epoch + 1} finished",
                str(classification_report(report_data[0], report_data[1])),
                "=" * 70,
            ]
        )
    )
    # if epoch % args.printfreq == args.printfreq - 1:
    #     logger.info(f'Training Epoch: {epoch+1}')
    #     logger.info("=" * 70 + "\n", classification_report(report_data[0], report_data[1], zero_division=0), "=" * 70 + "\n")


def validate(args, val_loader, model, criterion, epoch, g_catogaries, g_teacher_index):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()
    report_data = [[], []]

    pbar = tqdm(val_loader, total=len(val_loader), desc=f"Validating Epoch {epoch+1}", unit="batch")

    with paddle.no_grad():
        for i, (input, target) in enumerate(val_loader):
            with paddle.amp.auto_cast(enable=args.half, level='O1'):
                output = model(input)
                loss = criterion(output, target)

            (prec1, *_), r = accuracy(output, target, g_catogaries, g_teacher_index)
            report_data[0].extend(r[0])
            report_data[1].extend(r[1])

            losses.update(loss.item(), input.shape[0])
            top1.update(prec1.item(), input.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix(
                loss="{}({})".format(round(losses.val, 5), round(losses.avg, 5)),
                top1="{}({})".format(round(top1.val, 5), round(top1.avg, 5)),
            )
            pbar.update(1)
    pbar.close()
    logger.success(
        "\n".join(
            [
                "=" * 70,
                f"Validation Epoch {epoch + 1} finished",
                str(classification_report(report_data[0], report_data[1])),
                "=" * 70,
            ]
        )
    )

    # if epoch % args.printfreq == args.printfreq - 1:
    #     logger.info(f"Validate Epoch: {epoch+1}")
    #     logger.info("=" * 70 + "\n", classification_report(report_data[0], report_data[1], zero_division=0), "=" * 70)
    #     logger.info(f' * Prec@1 {top1.avg:.3f}\n\n')

    return top1.avg


def main(args):
    best_prec1 = 0

    # 在开始时设置设备
    use_gpu = args.use_gpu and paddle.is_compiled_with_cuda()
    args.use_gpu = use_gpu
    device = 'gpu' if use_gpu else 'cpu'
    paddle.set_device(device)
    logger.info(f"Using device: {device}")

    # 设置 CUDNN benchmark
    # NOTE: 可能需要使用终端命令设置
    # if use_gpu:
    #     paddle.set_flags({'FLAGS_cudnn_benchmark': True})

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    catogaries = args.classes
    # 使用 paddle.vision.datasets，指定文件路径有问题，默认在 ~/.cache/paddle/dataset/FashionMNIST
    raw_train_set = datasets.FashionMNIST(mode='train', download=True)
    raw_valid_set = datasets.FashionMNIST(mode='test', download=True)

    tmp_image = np.array(next(iter(raw_train_set))[0])
    image_shape = tmp_image.shape
    logger.info(f"Sample image shape: {image_shape} - {tmp_image.dtype} - {tmp_image.min()} ~ {tmp_image.max()}")

    train_index_dict = build_indexes(raw_train_set)
    train_sets = split_data_by_classes(train_index_dict, catogaries, dataset_size=args.trainset)
    val_index_dict = build_indexes(raw_valid_set)
    val_sets = split_data_by_classes(val_index_dict, catogaries, mode='val', dataset_size=args.trainset)

    # teacher_models =["Googlenet","Alexnet",'Densenet','Resnet20','Resnet32']
    teacher_models = ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet156']
    model_map = {
        0: resnet20.ResNet20,
        1: resnet20.ResNet32,
        2: resnet20.ResNet44,
        3: resnet20.ResNet56,
        4: resnet20.ResNet56,
    }

    for teacher_index in range(len(catogaries)):
        logger.info(f"\n{'='*20}\nTraining teacher {teacher_index} | Model: {teacher_models[teacher_index]}\n{'='*20}")

        num_classes = len(catogaries[teacher_index])

        # 使用 paddle 模型
        model = model_map.get(teacher_index, resnet20.ResNet56)(num_classes=num_classes, inp_channels=1)

        # cudnn.benchmark = True
        # TODO: paddle 没找到类似的暂时，之后再看，影响不大

        current_train_set = train_sets[teacher_index]
        current_val_set = val_sets[teacher_index]

        # 使用 paddle.io.DataLoader
        train_loader = DataLoader(
            SubDataset(
                current_train_set,
                raw_train_set,
                catogaries=catogaries[teacher_index],
                transform=cvtf.Compose([cvtf.ToTensor(), cvtf.Normalize(mean=[0.1307], std=[0.3081])]),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            use_shared_memory=True if use_gpu else False,
        )

        val_loader = DataLoader(
            SubDataset(
                current_val_set,
                raw_valid_set,
                catogaries=catogaries[teacher_index],
                transform=cvtf.Compose([cvtf.ToTensor(), cvtf.Normalize(mean=[0.1307], std=[0.3081])]),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            use_shared_memory=True if use_gpu else False,
        )
        
        tmp_image = next(iter(train_loader))[0]
        logger.info(
            f"Sample train image shape: {tmp_image.shape} - {tmp_image.dtype} - {tmp_image.min()} ~ {tmp_image.max()}"
        )

        criterion = nn.CrossEntropyLoss()
        if use_gpu:
            # paddle.set_device('gpu') 是全局设置
            model = model.to(device)
            criterion = criterion.to(device)

        # NOTE:使用 amp.auto_cast 进行自动混合精度训练 来代替下列步骤
        # if args.half:
        #     model.half()
        #     criterion.half()

        # 使用 paddle.optimizer.Momentum 和 lr.MultiStepDecay
        # NOTE: 将 lr 根据模型类型调整换到前面
        # NOTE: PaddlePaddle optimizer 直接使用 lr_scheduler，因此完整的顺序为：
        # 1. 定义学习率
        # 2. 定义 lr_scheduler
        # 3. 定义优化器
        lr = args.lr if args.arch not in ['ResNet1202', 'ResNet110'] else args.lr * 0.1
        lr_scheduler = paddle.optimizer.lr.MultiStepDecay(lr, milestones=[100, 150], last_epoch=args.start_epoch - 1)
        optimizer = paddle.optimizer.SGD(
            parameters=model.parameters(),
            learning_rate=lr_scheduler,
            # momentum=args.momentum, # NOTE: PaddlePaddle 的 SGD 没有 momentum 参数
            weight_decay=args.weight_decay,
        )

        # AMP (Automatic Mixed Precision) Scaler
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024, enable=args.half)

        if args.evaluate:
            validate(args, val_loader, model, criterion, 0, catogaries, teacher_index)
            return

        # 创建保存目录
        save_path = Path(args.save_dir)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(args.start_epoch, args.epochs):
            train(args, train_loader, model, criterion, optimizer, scaler, epoch, catogaries, teacher_index)
            lr_scheduler.step()

            prec1 = validate(args, val_loader, model, criterion, epoch, catogaries, teacher_index)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            # if epoch > 0 and epoch % args.save_every == 0:
            #     save_checkpoint(
            #         {
            #             'state_dict': model.state_dict(),
            #             'best_prec1': best_prec1,
            #             'epoch': epoch + 1,
            #             'optimizer': optimizer.state_dict(),
            #             'scaler': scaler.state_dict() if args.half else None,
            #             'catogaries': catogaries[teacher_index],
            #             'teacher_index': teacher_index,
            #             'model_name': teacher_models[teacher_index],
            #         },
            #         is_best,
            #         filename=(save_path / f'Teacher_{teacher_index}.pdparams').as_posix(),
            #     )
        save_checkpoint(
            {
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            },
            is_best,
            filename=(save_path / f'Teacher_{teacher_index}_even.pdparams').as_posix(),
        )


if __name__ == '__main__':
    log_file = Path("./logs", Path(__file__).stem, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
    setup_logger(log_file=log_file)
    
    logger.info("Training Teacher [ensemble] with PaddlePaddle")
    logger.info('--------args----------')
    args = get_train_teacher_argparser(
        model_names=["ResNet20", "ResNet32", "ResNet44", "ResNet56", "ResNet110"]
    ).parse_args()
    for k, v in vars(args).items():
        logger.info(f'{k}: {v}')
    logger.info('----------------------')
    main(args)
